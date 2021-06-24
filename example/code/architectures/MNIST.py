import torch
import torch.nn as nn
from ..utils.nn import Flatten, StochasticLinear, Reshape
from torch.distributions import Normal, Independent


INPUT_SHAPE = [1, 28, 28]
N_INPUTS = 28*28
N_LABELS = 10

# Create simple layer stacks with relu activations
def make_stack(layers):
    nn_layers = []
    for i in range(len(layers)-1):
        nn_layers.append(nn.Linear(layers[i], layers[i+1]))
        if i<len(layers)-2:
            nn_layers.append(nn.ReLU(True))

    return nn_layers


# Model for q(Z|X)
class Encoder(nn.Module):
    def __init__(self, z_dim, layers):
        super(Encoder, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([N_INPUTS] + layers)

        self.net = nn.Sequential(
            Flatten(),  # Layer to flatten the input
            *nn_layers,  # The previously created stack
            nn.ReLU(True),  # A ReLU activation
            StochasticLinear(layers[-1], z_dim, 'Normal')  # A layer that returns a factorized Normal distribution
        )

    def forward(self, x):
        # Note that the encoder returns a factorized normal distribution and not a vector
        return self.net(x)


# Model for p(Z)
class Prior(nn.Module):
    def __init__(self, z_dim):
        super(Prior, self).__init__()

        self.mu = nn.Parameter(torch.zeros([1, z_dim]), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones([1, z_dim]), requires_grad=False)

    def forward(self):
        # Return a factorized Normal prior
        return Independent(Normal(self.mu, self.sigma), 1)


# Model for p(X|Z)
class Decoder(nn.Module):
    def __init__(self, z_dim, layers, sigma):
        super(Decoder, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([z_dim] + layers + [N_INPUTS])

        self.net = nn.Sequential(
            *nn_layers,                 # The previously created stack
            Reshape(INPUT_SHAPE)        # A layer to reshape to the correct image shape
        )
        self.sigma = sigma

    def forward(self, x):
        # Note that the decoder returns a factorized normal distribution and not a vector
        # the last 3 dimensions (n_channels x x_dim x y_dim) are considered to be independent
        return Independent(Normal(self.net(x), self.sigma), 3)


class LabelClassifier(nn.Module):
    def __init__(self, z_dim):
        super(LabelClassifier, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([z_dim] + [64, 32])

        self.net = nn.Sequential(
            *nn_layers,  # The previously created stack
            nn.ReLU(True),  # A ReLU activation
            StochasticLinear(32, N_LABELS, 'Categorical')  # A layer that returns a Categorical distribution
        )

    def forward(self, x):
        # Note that the encoder returns a Categorical distribution and not a vector
        return self.net(x)

# Definition of an alternative implementation for the label classifier.
# Note that this has been created only for exemplifying how different architectures implementation can be speficied
# but in principle it is possible to achieve the same result (with cleaner code) by passing the depth of the network
# as an hyper-parameter in the constructor arguments.
class DeepLabelClassifier(nn.Module):
    def __init__(self, z_dim):
        super(DeepLabelClassifier, self).__init__()

        # Create a stack of layers with ReLU activations as specified
        nn_layers = make_stack([z_dim] + [1024, 512, 256, 128])

        self.net = nn.Sequential(
            *nn_layers,  # The previously created stack
            nn.ReLU(True),  # A ReLU activation
            StochasticLinear(128, N_LABELS, 'Categorical')  # A layer that returns a Categorical distribution
        )

    def forward(self, x):
        # Note that the encoder returns a Categorical distribution and not a vector
        return self.net(x)