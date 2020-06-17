import torch
import torch.nn as nn
from utils.modules import Flatten, StochasticLinear, StochasticLinear2D, Reshape, OneHot

CMNIST_SIZE = 14**2*2
CMNIST_SHAPE = [2,14,14]
CMNIST_N_CLASSES = 2
CMNIST_N_ENVS = 2

# Model for p(z|x)
class SimpleEncoder(nn.Module):
    def __init__(self, z_dim, dist):
        super(SimpleEncoder, self).__init__()

        self.net = nn.Sequential(
            Flatten(),
            nn.Linear(CMNIST_SIZE, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            StochasticLinear(1024, z_dim, dist)
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
        return torch.distributions.Categorical(logits=z[:, :CMNIST_N_CLASSES])

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
    def __init__(self, z_dim):
        super(SimpleConditionalEnvClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim+CMNIST_N_CLASSES, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            StochasticLinear(1024, CMNIST_N_ENVS, 'Categorical')
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
