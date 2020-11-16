import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn
from torch.nn.functional import softplus
from utils.modules import Flatten, StochasticLinear, OneHot
from data.SpeechCommands import SPEECH_COMMANDS_N_CLASSES, SPEECH_COMMANDS_N_ENVS
from torch.distributions import Normal, Independent

# Model for p(z|x)
class SimpleEncoder(nn.Module):
    def __init__(self, z_dim, dist='Delta', dropout=0, n_hidden=1024):
        super(SimpleEncoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, n_hidden, 10, 5),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Conv1d(n_hidden, n_hidden, 8, 8),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Conv1d(n_hidden, n_hidden, 4, 2),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Conv1d(n_hidden, n_hidden, 4, 2),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Conv1d(n_hidden, n_hidden, 4, 2),
            nn.Dropout(dropout),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(n_hidden * 48, n_hidden),
            StochasticLinear(n_hidden, z_dim, dist)
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
    def __init__(self, z_dim, dropout=0, n_hidden=1024, dist='Categorical'):
        super(SimpleClassifier, self).__init__()

        if dist == 'Categorical':
            out_dim = SPEECH_COMMANDS_N_CLASSES
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
class SimpleJointClassifier(nn.Module):
    def __init__(self, z_dim):
        super(SimpleJointClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, SPEECH_COMMANDS_N_CLASSES * SPEECH_COMMANDS_N_ENVS)
        )

    def compute_logits(self, z):
        return self.net(z).view(z.shape[0], SPEECH_COMMANDS_N_CLASSES, SPEECH_COMMANDS_N_ENVS)


# Constant Model for q(y|z)
class ConstantClassifier(nn.Module):
    def __init__(self, z_dim):
        super(ConstantClassifier, self).__init__()
        self.z_dim = z_dim

    def forward(self, z):
        params = z[:, :SPEECH_COMMANDS_N_CLASSES]
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
            StochasticLinear(n_hidden, SPEECH_COMMANDS_N_ENVS, 'Categorical')
        )

    def forward(self, z):
        return self.net(z)


# Model for q(e|zy)
class SimpleConditionalEnvClassifier(nn.Module):
    def __init__(self, z_dim, spectral_norm=False, n_hidden=1024):
        super(SimpleConditionalEnvClassifier, self).__init__()

        if not spectral_norm:
            self.net = nn.Sequential(
                nn.Linear(z_dim+SPEECH_COMMANDS_N_CLASSES, n_hidden),
                nn.ReLU(True),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(True),
                StochasticLinear(n_hidden, SPEECH_COMMANDS_N_ENVS, 'Categorical')
            )
        else:
            self.net = nn.Sequential(
                sn(nn.Linear(z_dim + SPEECH_COMMANDS_N_CLASSES, 1024)),
                nn.ReLU(True),
                sn(nn.Linear(1024, 1024)),
                nn.ReLU(True),
                sn(nn.Linear(1024, SPEECH_COMMANDS_N_ENVS))
            )

        self.long2onehot = OneHot(SPEECH_COMMANDS_N_CLASSES)

    def forward(self, z, y):
        zy = torch.cat([z, self.long2onehot(y)], 1)
        return self.net(zy)


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
