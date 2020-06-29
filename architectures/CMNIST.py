import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn
from torch.nn.functional import softplus
from utils.modules import Flatten, StochasticLinear, StochasticLinear2D, Reshape, OneHot

CMNIST_SIZE = 14**2*2
CMNIST_SHAPE = [2,14,14]
CMNIST_N_CLASSES = 2
CMNIST_N_ENVS = 2


# Model for p(z|x)
class SimpleEncoder(nn.Module):
    def __init__(self, z_dim, dist, dropout=0):
        super(SimpleEncoder, self).__init__()

        self.net = nn.Sequential(
            Flatten(),
            nn.Linear(CMNIST_SIZE, 1024),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.Dropout(dropout),
            nn.ReLU(True),
            StochasticLinear(1024, z_dim, dist)
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

# Model for q(y|z)
class SimpleClassifier(nn.Module):
    def __init__(self, z_dim):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            StochasticLinear(1024, CMNIST_N_CLASSES, 'Categorical')
        )

    def forward(self, z):
        return self.net(z)


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
    def __init__(self, z_dim):
        super(SimpleEnvClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            StochasticLinear(1024, CMNIST_N_ENVS, 'Categorical')
        )

    def forward(self, z):
        return self.net(z)


# Model for q(e|zy)
class SimpleConditionalEnvClassifier(nn.Module):
    def __init__(self, z_dim, spectral_norm=False):
        super(SimpleConditionalEnvClassifier, self).__init__()

        if not spectral_norm:
            self.net = nn.Sequential(
                nn.Linear(z_dim+CMNIST_N_CLASSES, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                StochasticLinear(1024, CMNIST_N_ENVS, 'Categorical')
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
