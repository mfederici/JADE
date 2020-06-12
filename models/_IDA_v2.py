from training.base import RepresentationTrainer, init_optimizer
from utils.schedulers import ExponentialScheduler
import torch.nn as nn
import torch
from pytorch_revgrad import RevGrad
from torch.nn.functional import softplus, softmax
from torch.distributions import Normal, Independent, Bernoulli, Categorical
from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms import AffineAutoregressive
from torch.distributions import TransformedDistribution


class TransformedFactorized(nn.Module):
    def __init__(self, in_size, out_size):
        super(TransformedFactorized, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.ReLU(True),
            nn.Linear(256, out_size*2)
        )

        self.arn = AutoRegressiveNN(out_size, [out_size])
        self.transform = AffineAutoregressive(self.arn)

    def forward(self, input):
        params = self.net(input)
        mu, sigma = params[:, :params.shape[1]//2], params[:, params.shape[1]//2:]
        sigma = softplus(sigma)
        factor_dist = Independent(Normal(mu, sigma), 1)
        flow_dist = TransformedDistribution(factor_dist, [self.transform])
        return flow_dist


##################
# IDA_v2 Trainer #
##################
class IDAV2Trainer(RepresentationTrainer):
    def __init__(self, z_dim, optimizer_name='Adam', encoder_lr=1e-4,
                 n_classes=2, n_env=2, **params):
        # The neural networks architectures and initialization procedure is analogous to Multi-View InfoMax
        super(IDAV2Trainer, self).__init__(z_dim=z_dim, optimizer_name=optimizer_name, encoder_lr=encoder_lr, **params)

        self.n_classes = n_classes
        self.n_env = n_env

        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, n_classes)
        )

        self.reconstruction = TransformedFactorized(z_dim + n_classes + n_env, 14**2*2)

        # Defining the prior distribution as a factorized normal distribution
        self.mu = nn.Parameter(torch.zeros(self.z_dim), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(self.z_dim), requires_grad=False)
        self.prior = Normal(loc=self.mu, scale=self.sigma)
        self.prior = Independent(self.prior, 1)

        self.opt.add_param_group(
            {'params': self.classifier.parameters(), 'lr': encoder_lr}
        )

        self.opt.add_param_group(
            {'params': self.reconstruction.parameters(), 'lr': encoder_lr}
        )

    def _get_items_to_store(self):
        items_to_store = super(IDAV2Trainer, self)._get_items_to_store()

        # store the encoder, classifier, reconstruction and optimizer parameters
        items_to_store['encoder'] = self.encoder.state_dict()
        items_to_store['classifier'] = self.classifier.state_dict()
        items_to_store['reconstruction'] = self.reconstruction.state_dict()
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

        # Label Reconstruction
        p_y_given_z = Categorical(logits=self.classifier(z))

        # Input Reconstruction
        eyz = torch.cat([
            torch.eye(self.n_env)[e],
            torch.eye(self.n_classes)[y],
            z,
        ], 1)

        p_x_given_eyz = self.reconstruction(eyz)

        # Loss
        y_rec_loss = - p_y_given_z.log_prob(y).mean()
        kl_loss = (p_z_given_x.log_prob(z) - self.prior.log_prob(z)).mean()
        x_rec_loss = - p_x_given_eyz.log_prob(x.view(x.shape[0], -1)).mean()

        loss = (1) * y_rec_loss + kl_loss + x_rec_loss

        self._add_loss_item('loss/CE_y_z', y_rec_loss.item())
        self._add_loss_item('loss/CE_X_eyz', x_rec_loss.item())
        self._add_loss_item('loss/KL_x_z', kl_loss.item())

        return loss
