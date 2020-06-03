from training.base import RepresentationTrainer, init_optimizer
from utils.schedulers import ExponentialScheduler
import torch.nn as nn
import torch
from torch.nn.functional import softplus, softmax
from torch.distributions import Normal, Independent, Bernoulli, Categorical
from utils.modules import Encoder, MIEstimator
from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms import AffineAutoregressive
from torch.distributions import TransformedDistribution




class LabelEncoder(Encoder):
    def __init__(self, n_classes, z_dim):
        super(LabelEncoder, self).__init__(z_dim=z_dim)
        self.net = nn.Linear(n_classes, z_dim * 2)
        self.n_classes = n_classes
        
    def forward(self, input):
        one_hot = torch.eye(self.n_classes).to(input.device)[input]
        return super(LabelEncoder, self).forward(one_hot)


##################
# IDA_v4 Trainer #
##################
class IDAV4Trainer(RepresentationTrainer):
    def __init__(self, z_dim, optimizer_name='Adam', encoder_lr=1e-4,
                 n_classes=2, n_env=2, f_dim=64, beta_start_value=1e-3, beta_end_value=1,
                 beta_n_iterations=100000, beta_start_iteration=50000, **params):
        # The neural networks architectures and initialization procedure is analogous to Multi-View InfoMax
        super(IDAV4Trainer, self).__init__(z_dim=z_dim, optimizer_name=optimizer_name, encoder_lr=encoder_lr, **params)

        self.n_classes = n_classes
        self.n_env = n_env

        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, n_classes)
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(14**2*2, 128),
            nn.ReLU(True),
            nn.Linear(128, f_dim)
        )

        self.mi_estimator = MIEstimator(f_dim, z_dim+n_classes+n_env)

        self.label_encoder = LabelEncoder(n_classes=n_classes, z_dim=z_dim)

        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                   n_iterations=beta_n_iterations, start_iteration=beta_start_iteration)

        self.opt.add_param_group(
            {'params': self.classifier.parameters(), 'lr': encoder_lr}
        )

        self.opt.add_param_group(
            {'params': self.mi_estimator.parameters(), 'lr': encoder_lr}
        )

        self.opt.add_param_group(
            {'params': self.feature_extractor.parameters(), 'lr': encoder_lr}
        )

        self.opt.add_param_group(
            {'params': self.label_encoder.parameters(), 'lr': encoder_lr}
        )

    def _get_items_to_store(self):
        items_to_store = super(IDAV4Trainer, self)._get_items_to_store()

        # store the encoder, classifier, reconstruction and optimizer parameters
        items_to_store['encoder'] = self.encoder.state_dict()
        items_to_store['classifier'] = self.classifier.state_dict()
        items_to_store['mi_estimator'] = self.mi_estimator.state_dict()
        items_to_store['label_encoder'] = self.label_encoder.state_dict()
        items_to_store['feature_extractor'] = self.feature_extractor.state_dict()
        items_to_store['opt'] = self.opt.state_dict()

        return items_to_store

    def _train_step(self, data):
        loss = self._compute_loss(data)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def _compute_loss(self, data):
        # Read the two views v1 and v2 and ignore the label y
        x = data['x']
        y = data['y'].long()
        e = data['e'].long()

        # Encode a batch of data
        p_z_given_x = self.encoder(x)
        z = p_z_given_x.rsample()
        
        q_z_given_y = self.label_encoder(y)

        # Label Reconstruction
        p_y_given_z = Categorical(logits=self.classifier(z))

        # MI Estimation
        eyz = torch.cat([
            torch.eye(self.n_env)[e],
            torch.eye(self.n_classes)[y],
            z,
        ], 1)
        f = self.feature_extractor(x.view(x.shape[0],-1))

        mi_jsd, mi = self.mi_estimator(f, eyz)

        # Loss
        y_rec_loss = - p_y_given_z.log_prob(y).mean()
        kl_loss = (p_z_given_x.log_prob(z) - q_z_given_y.log_prob(z)).mean()

        # Update the value of beta according to the policy
        beta = self.beta_scheduler(self.iterations)

        loss = 1e-5 * y_rec_loss + beta * kl_loss - mi_jsd

        self._add_loss_item('loss/CE_y_z', y_rec_loss.item())
        self._add_loss_item('loss/I_x_eyz', mi.item())
        self._add_loss_item('loss/KL_x_z_y', kl_loss.item())
        self._add_loss_item('loss/beta', beta)

        return loss
