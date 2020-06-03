import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, Beta
from pyro.distributions import Delta
from torch.nn.functional import softplus


# Encoder architecture
class Encoder(nn.Module):
    def __init__(self, z_dim, dist):
        super(Encoder, self).__init__()

        self.z_dim = z_dim
        self.dist = dist

        if dist == 'normal' or dist == 'beta':
            self.n_params = 2
        elif dist == 'delta':
            self.n_params = 1
        else:
            raise NotImplementedError('"%s"' % dist)

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(14 * 14 * 2, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, z_dim * self.n_params),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input

        params = torch.split(self.net(x), [self.z_dim]*self.n_params, 1)

        if self.dist == 'normal':
            mu, sigma = params[0], softplus(params[1]) + 1e-7
            dist = Normal(loc=mu, scale=sigma)
            dist = Independent(dist, 1) # Factorized Normal distribution
        elif self.dist == 'beta':
            c1, c0 = softplus(params[0]) + 1e-7, softplus(params[1]) + 1e-7
            dist = Beta(c1, c0)
            dist = Independent(dist, 1)  # Factorized Beta distribution
        elif self.dist == 'delta':
            m = params[0]
            dist = Delta(m)
            dist = Independent(dist, 1)  # Factorized Delta distribution
        else:
            dist = None

        return dist


class Decoder(nn.Module):
    def __init__(self, z_dim, scale=0.39894):
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.scale = scale

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28)
        )

    def forward(self, z):
        x = self.net(z)
        return Independent(Normal(loc=x, scale=self.scale), 1)


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