import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn
from torch.nn.functional import softplus
from utils.modules import Flatten, StochasticLinear, StochasticLinear2D, Reshape, OneHot
from data.CMNIST import CMNIST_SIZE, CMNIST_N_CLASSES, CMNIST_SHAPE, CMNIST_N_ENVS
from torch.distributions import Normal, Independent

# Model for p(z|x)
class SimpleEncoder(nn.Module):
    def __init__(self, z_dim, dist='Delta', dropout=0, n_hidden=1024, x_dim=None):
        super(SimpleEncoder, self).__init__()

        if x_dim is None:
            x_dim = CMNIST_SIZE

        self.net = nn.Sequential(
            Flatten(),
            nn.Linear(x_dim, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(True),
            StochasticLinear(n_hidden, z_dim, dist)
        )

    def forward(self, x):
        return self.net(x)


# Model for f(x)
class SimpleFeatureExtractor(nn.Module):
    def __init__(self, f_dim):
        super(SimpleFeatureExtractor, self).__init__()

        self.net = nn.Sequential(
            Flatten(),
            nn.Linear(CMNIST_SIZE, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, f_dim)
        )

    def forward(self, x):
        return self.net(x)


# Model for q(z)
class NormalPrior(nn.Module):
    def __init__(self, z_dim):
        super(NormalPrior, self).__init__()
        self.mu = nn.Parameter(torch.zeros([1, z_dim]), requires_grad=False)
        self.sigma = nn.Parameter(torch.zeros([1, z_dim])+1, requires_grad=False)

    def forward(self):
        return Independent(Normal(self.mu, self.sigma), 1)


# Model for q(y|z)
class SimpleClassifier(nn.Module):
    def __init__(self, z_dim, dropout=0, n_hidden=1024):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(True),
            StochasticLinear(n_hidden, CMNIST_N_CLASSES, 'Categorical')
        )

    def forward(self, z):
        return self.net(z)

# Model for q(ye|z)
class SimpleJointClassifier(nn.Module):
    def __init__(self, z_dim):
        super(SimpleJointClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, CMNIST_N_CLASSES * CMNIST_N_ENVS)
        )

    def compute_logits(self, z):
        return self.net(z).view(z.shape[0], CMNIST_N_CLASSES, CMNIST_N_ENVS)

# Constant Model for q(y|z)
class ConstantClassifier(nn.Module):
    def __init__(self, z_dim):
        super(ConstantClassifier, self).__init__()
        self.z_dim = z_dim

    def forward(self, z):
        params = z[:, :CMNIST_N_CLASSES]
        return torch.distributions.Categorical(logits=params)

# Model for q(e|z)
class SimpleEnvClassifier(nn.Module):
    def __init__(self, z_dim, n_hidden=1024):
        super(SimpleEnvClassifier, self).__init__()
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
class SimpleConditionalEnvClassifier(nn.Module):
    def __init__(self, z_dim, spectral_norm=False, n_hidden=1024):
        super(SimpleConditionalEnvClassifier, self).__init__()

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

# Model for q(x|z)
class SimpleDecoder(nn.Module):
    def __init__(self, z_dim, dist):
        super(SimpleDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, CMNIST_SIZE),
            Reshape(CMNIST_SHAPE),
            StochasticLinear2D(CMNIST_SHAPE[2], CMNIST_SHAPE[2], dist)
        )

    def forward(self, input):
        return self.net(input)


# Auxiliary network for mutual information estimation
class SimpleMIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(SimpleMIEstimator, self).__init__()

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
