import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn
from torch.nn.functional import softplus
from utils.modules import Flatten, StochasticLinear, OneHot
from data.SpeechCommands import SPEECH_COMMANDS_N_CLASSES, SPEECH_COMMANDS_N_ENVS
from torch.distributions import Normal, Independent

N_ENVS = SPEECH_COMMANDS_N_ENVS

# Model for p(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, dist='Delta', dropout=0, n_hidden=1024):
        super(Encoder, self).__init__()

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
class Prior(nn.Module):
    def __init__(self, z_dim):
        super(Prior, self).__init__()
        self.mu = nn.Parameter(torch.zeros([1, z_dim]), requires_grad=False)
        self.sigma = nn.Parameter(torch.zeros([1, z_dim])+1, requires_grad=False)

    def forward(self):
        return Independent(Normal(self.mu, self.sigma), 1)


# Model for q(y|z)
class LabelClassifier(nn.Module):
    def __init__(self, z_dim, dropout, n_hidden, dist='Categorical'):
        super(LabelClassifier, self).__init__()

        if not (dist == 'Categorical'):
            raise NotImplemented()

        self.net = nn.Sequential(
            nn.Linear(z_dim, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(True),
            StochasticLinear(n_hidden, SPEECH_COMMANDS_N_CLASSES, dist)
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
            nn.Linear(1024, SPEECH_COMMANDS_N_CLASSES * SPEECH_COMMANDS_N_ENVS)
        )

    def compute_logits(self, z):
        return self.net(z).view(z.shape[0], SPEECH_COMMANDS_N_CLASSES, SPEECH_COMMANDS_N_ENVS)


# Model for q(e|z)
class EnvClassifier(nn.Module):
    def __init__(self, z_dim, n_hidden=1024, dropout=0):
        super(EnvClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(True),
            StochasticLinear(n_hidden, SPEECH_COMMANDS_N_ENVS, 'Categorical')
        )

    def forward(self, z):
        return self.net(z)


# Model for q(e|zy)
class ConditionalEnvClassifier(nn.Module):
    def __init__(self, z_dim, spectral_norm=False, n_hidden=1024, dropout=0):
        super(ConditionalEnvClassifier, self).__init__()

        if not spectral_norm:
            self.net = nn.Sequential(
                nn.Linear(z_dim+SPEECH_COMMANDS_N_CLASSES, n_hidden),
                nn.Dropout(dropout),
                nn.ReLU(True),
                nn.Linear(n_hidden, n_hidden),
                nn.Dropout(dropout),
                nn.ReLU(True),
                StochasticLinear(n_hidden, SPEECH_COMMANDS_N_ENVS, 'Categorical')
            )
        else:
            self.net = nn.Sequential(
                sn(nn.Linear(z_dim + SPEECH_COMMANDS_N_CLASSES, n_hidden)),
                nn.Dropout(dropout),
                nn.ReLU(True),
                sn(nn.Linear(n_hidden, n_hidden)),
                nn.Dropout(dropout),
                nn.ReLU(True),
                sn(nn.Linear(n_hidden, SPEECH_COMMANDS_N_ENVS)),
                StochasticLinear(n_hidden, SPEECH_COMMANDS_N_ENVS, 'Categorical')
            )

        self.long2onehot = OneHot(SPEECH_COMMANDS_N_CLASSES)

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
