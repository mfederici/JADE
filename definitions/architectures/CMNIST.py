import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn
from torch.nn.functional import softplus
from utils.modules import Flatten, StochasticLinear, StochasticLinear2D, Reshape, OneHot
from data.CMNIST import CMNIST_SIZE, CMNIST_N_CLASSES, CMNIST_SHAPE, CMNIST_N_ENVS
from torch.distributions import Normal, Independent

# Setting number of environments (for vREX)
N_ENVS = CMNIST_N_ENVS


# Model for p(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, dropout,  n_hidden, dist='Delta'):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            Flatten(),
            nn.Linear(CMNIST_SIZE, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(True),
            StochasticLinear(n_hidden, z_dim, dist)
        )

    def forward(self, x):
        return self.net(x)


# Model for q(z)
class Prior(nn.Module):
    def __init__(self, z_dim, dist):
        super(Prior, self).__init__()
        self.dist = dist
        if dist == 'Normal':
            self.mu = nn.Parameter(torch.zeros([1, z_dim]), requires_grad=False)
            self.sigma = nn.Parameter(torch.zeros([1, z_dim])+1, requires_grad=False)
        else:
            raise NotImplemented('%s prior distribution is not implemented' % dist)

    def forward(self):
        if self.dist == 'Normal':
            return Independent(Normal(self.mu, self.sigma), 1)
        else:
            raise NotImplemented('%s prior distribution is not implemented' % self.dist)


# Model for q(y|z)
class LabelClassifier(nn.Module):
    def __init__(self, z_dim, dropout, n_hidden, dist='Categorical'):
        super(LabelClassifier, self).__init__()

        if dist == 'Categorical':
            out_dim = CMNIST_N_CLASSES
        elif dist == 'Bernoulli':
            out_dim = 1
        else:
            raise NotImplemented()
        self.net = nn.Sequential(
            nn.Linear(z_dim, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(True),
            StochasticLinear(n_hidden, out_dim, dist)
        )

    def forward(self, z):
        return self.net(z)


# Model for q(ye|z)
class JointClassifier(nn.Module):
    def __init__(self, z_dim):
        super(JointClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, CMNIST_N_CLASSES * CMNIST_N_ENVS)
        )

    def compute_logits(self, z):
        return self.net(z).view(z.shape[0], CMNIST_N_CLASSES, CMNIST_N_ENVS)


# Model for q(e|z)
class EnvClassifier(nn.Module):
    def __init__(self, z_dim, n_hidden):
        super(EnvClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(True),
            StochasticLinear(n_hidden, CMNIST_N_ENVS, 'Categorical')
        )

    def forward(self, z):
        return self.net(z)


# Model for q(e|zy)
class ConditionalEnvClassifier(nn.Module):
    def __init__(self, z_dim, spectral_norm=False, n_hidden=1024):
        super(ConditionalEnvClassifier, self).__init__()

        if not spectral_norm:
            self.net = nn.Sequential(
                nn.Linear(z_dim+CMNIST_N_CLASSES, n_hidden),
                nn.ReLU(True),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(True),
                StochasticLinear(n_hidden, CMNIST_N_ENVS, 'Categorical')
            )
        else:
            self.net = nn.Sequential(
                sn(nn.Linear(z_dim + CMNIST_N_CLASSES, 1024)),
                nn.ReLU(True),
                sn(nn.Linear(1024, 1024)),
                nn.ReLU(True),
                sn(nn.Linear(1024, CMNIST_N_ENVS))
            )

        self.long2onehot = OneHot(CMNIST_N_CLASSES)

    def forward(self, z, y):
        zy = torch.cat([z, self.long2onehot(y)], 1)
        return self.net(zy)


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