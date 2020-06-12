from training.base import RepresentationTrainer, init_optimizer
from utils.schedulers import ExponentialScheduler, LinearScheduler
import torch.nn as nn
import torch
from utils.modules import MIEstimator, StochasticLinear


class LabelEncoder(nn.Module):
    def __init__(self, n_classes, z_dim):
        super(LabelEncoder, self).__init__()
        self.net = StochasticLinear(n_classes, z_dim,  'Normal')
        self.n_classes = n_classes

    def forward(self, input):
        one_hot = torch.eye(self.n_classes).to(input.device)[input]
        return self.net(one_hot)


####################################################################
# IDA ConditionalVariational Mutual Information Estimation Trainer #
####################################################################
class IDACVMITrainer(RepresentationTrainer):
    def __init__(self, z_dim, x_dim, optimizer_name='Adam', encoder_lr=1e-4,
                 n_classes=2, n_env=2, f_dim=64, beta_start_value=1e-3, beta_end_value=1,
                 beta_n_iterations=100000, beta_start_iteration=50000, **params):
        # The neural networks architectures and initialization procedure is analogous to Multi-View InfoMax
        super(IDACVMITrainer, self).__init__(x_dim=x_dim, z_dim=z_dim, optimizer_name=optimizer_name,
                                             encoder_lr=encoder_lr, **params)

        self.n_classes = n_classes
        self.n_env = n_env

        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            StochasticLinear(1024, n_classes, 'Categorical')
        )

        self.feature_extractor = nn.Sequential(
            nn.Linear(x_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, f_dim)
        )

        self.mi_estimator = MIEstimator(f_dim, z_dim+n_classes+n_env)

        self.label_encoder = LabelEncoder(n_classes=n_classes, z_dim=z_dim)

        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta_scheduler = LinearScheduler(start_value=beta_start_value, end_value=beta_end_value,
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
        items_to_store = super(IDACVMITrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'classifier',
            'mi_estimator',
            'label_encoder',
            'feature_extractor'
        })

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
        p_y_given_z = self.classifier(z)

        # MI Estimation
        eyz = torch.cat([
            torch.eye(self.n_env)[e],
            torch.eye(self.n_classes)[y],
            z,
        ], 1)
        f = self.feature_extractor(x.view(x.shape[0], -1))

        mi_jsd, mi = self.mi_estimator(f, eyz)

        # Loss
        y_rec_loss = - p_y_given_z.log_prob(y).mean()
        kl_loss = (p_z_given_x.log_prob(z) - q_z_given_y.log_prob(z)).mean()

        # Update the value of beta according to the policy
        beta = self.beta_scheduler(self.iterations)

        loss = (1-beta) * y_rec_loss + beta * kl_loss - beta * mi

        self._add_loss_item('loss/CE_y_z', y_rec_loss.item())
        self._add_loss_item('loss/I_x_eyz', mi.item())
        self._add_loss_item('loss/KL_x_z_y', kl_loss.item())
        self._add_loss_item('loss/beta', beta)

        return loss
