import torch
import torch.nn as nn
from torch.distributions import Normal, Independent, Beta, Categorical, Bernoulli
from pyro.distributions import Delta
from torch.nn.functional import softplus


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(-1, *self.shape)


class Permute(nn.Module):
    def __init__(self, *permutation):
        super(Permute, self).__init__()
        self.permutation = permutation

    def forward(self, input):
        return input.permute(self.permutation)


class OneHot(nn.Module):
    def __init__(self, n_values):
        super(OneHot, self).__init__()
        self.eye = nn.Parameter(torch.eye(n_values), requires_grad=False)

    def forward(self, input):
        assert isinstance(input, torch.LongTensor)
        return self.eye[input]


class StochasticLinear(nn.Module):
    def __init__(self, in_size, out_size, dist):
        super(StochasticLinear, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dist = dist

        if dist == 'Normal' or dist == 'Beta':
            self.n_params = 2
        elif dist == 'Delta' or dist == 'Categorical' or dist == 'Bernoulli':
            self.n_params = 1
        else:
            raise NotImplementedError('"%s"' % dist)

        self.layer = nn.Linear(in_size, out_size * self.n_params)

    def forward(self, input):
        params = torch.split(self.layer(input), [self.out_size] * self.n_params, 1)

        if self.dist == 'Normal':
            mu, sigma = params[0], softplus(params[1]) + 1e-7
            dist = Normal(loc=mu, scale=sigma)
            dist = Independent(dist, 1)  # Factorized Normal distribution
        elif self.dist == 'Beta':
            c1, c0 = softplus(params[0]) + 1e-7, softplus(params[1]) + 1e-7
            dist = Beta(c1, c0)
            dist = Independent(dist, 1)  # Factorized Beta distribution
        elif self.dist == 'Delta':
            m = params[0]
            dist = Delta(m)
            dist = Independent(dist, 1)  # Factorized Delta distribution
        elif self.dist == 'Categorical':
            logits = params[0]
            dist = Categorical(logits=logits)
        elif self.dist == 'Bernoulli':
            logits = params[0]
            dist = Bernoulli(logits=logits)
        else:
            dist = None

        return dist


class StochasticLinear2D(StochasticLinear):
    def __init__(self, in_channels, out_channels, dist):
        super(StochasticLinear2D, self).__init__(in_channels, out_channels, dist)
        # Changes the order of the dimension so that the linear layer is applied channel-wise
        self.layer = nn.Sequential(
            Permute(0, 2, 3, 1),
            self.layer,
            Permute(0, 3, 1, 2)
        )

    def forward(self, input):
        dist = super(StochasticLinear2D, self).forward(input)
        dist = Independent(dist, 2) # make the spatial and channel dimensions independent

        return dist


# Auxiliary network for mutual information estimation
# TODO update to receive the network as an input
# class MIEstimator(nn.Module):
#     def __init__(self, size1, size2):
#         super(MIEstimator, self).__init__()
#
#         # Vanilla MLP
#         self.net = nn.Sequential(
#             nn.Linear(size1 + size2, 1024),
#             nn.ReLU(True),
#             nn.Linear(1024, 1024),
#             nn.ReLU(True),
#             nn.Linear(1024, 1),
#         )
#
#     # Gradient for JSD mutual information estimation and EB-based estimation
#     def forward(self, x1, x2):
#         pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
#         neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
#         return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1