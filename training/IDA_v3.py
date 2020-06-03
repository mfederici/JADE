from training.base import RepresentationTrainer, init_optimizer
from utils.schedulers import ExponentialScheduler
import torch.nn as nn
import torch
from torch.nn.functional import softplus, softmax
from torch.distributions import Normal, Independent, Bernoulli, Categorical
from pyro.distributions import MixtureOfDiagNormals
from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms import AffineAutoregressive
from torch.distributions import TransformedDistribution

# Auxiliary network for mutual information estimation
class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1

class MixturePrior(nn.Module):
    def __init__(self, z_dim, k=10):
        super(MixturePrior, self).__init__()
        self.mu = nn.Parameter(torch.normal(torch.zeros(1,k,z_dim)))
        self.log_sigma = nn.Parameter(torch.normal(torch.zeros(1,k,z_dim)))
        self.pi_logits = nn.Parameter(torch.normal(torch.zeros(1,k)))

    def forward(self):
        return MixtureOfDiagNormals(self.mu, softplus(self.log_sigma), self.pi_logits)


class TransformedPrior(nn.Module):
    def __init__(self, z_dim):
        super(TransformedPrior, self).__init__()
        self.arn_1 = AutoRegressiveNN(z_dim, [z_dim])
        self.arn_2 = AutoRegressiveNN(z_dim, [z_dim])
        self.transform = [AffineAutoregressive(self.arn_1), AffineAutoregressive(self.arn_2)]

        self.mu = nn.Parameter(torch.zeros(z_dim))
        self.log_sigma = nn.Parameter(torch.zeros(z_dim))

    def forward(self):
        sigma = softplus(self.log_sigma)
        factor_dist = Independent(Normal(self.mu, sigma), 1)
        return TransformedDistribution(factor_dist, self.transform)

##################
# IDA_32 Trainer #
##################
class IDAV3Trainer(RepresentationTrainer):
    def __init__(self, z_dim, optimizer_name='Adam', encoder_lr=1e-4,
                 n_classes=2, n_env=2, f_dim=64, beta_start_value=1e-3, beta_end_value=1,
                 beta_n_iterations=100000, beta_start_iteration=50000, k=10, **params):
        # The neural networks architectures and initialization procedure is analogous to Multi-View InfoMax
        super(IDAV3Trainer, self).__init__(z_dim=z_dim, optimizer_name=optimizer_name, encoder_lr=encoder_lr, **params)

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

        # Defining the prior distribution as a factorized normal distribution
        self.prior = TransformedPrior(z_dim=z_dim)


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
            {'params': self.prior.parameters(), 'lr': encoder_lr}
        )

    def _get_items_to_store(self):
        items_to_store = super(IDAV3Trainer, self)._get_items_to_store()

        # store the encoder, classifier, reconstruction and optimizer parameters
        items_to_store['encoder'] = self.encoder.state_dict()
        items_to_store['classifier'] = self.classifier.state_dict()
        items_to_store['mi_estimator'] = self.mi_estimator.state_dict()
        items_to_store['feature_extractor'] = self.feature_extractor.state_dict()
        items_to_store['prior'] = self.prior.state_dict()
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
        kl_loss = (p_z_given_x.log_prob(z) - self.prior().log_prob(z)).mean()

        # Update the value of beta according to the policy
        beta = self.beta_scheduler(self.iterations)

        loss = (1+1e-4) * y_rec_loss + beta * kl_loss - mi

        self._add_loss_item('loss/CE_y_z', y_rec_loss.item())
        self._add_loss_item('loss/I_x_eyz', mi.item())
        self._add_loss_item('loss/KL_x_z', kl_loss.item())
        self._add_loss_item('loss/beta', beta)

        return loss
