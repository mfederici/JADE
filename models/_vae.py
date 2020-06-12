import torch
import torch.nn as nn
from torch.distributions import Normal, Independent

from utils.modules import StochasticLinear
from training.base import RepresentationTrainer
from utils.schedulers import ExponentialScheduler


###############
# VAE Trainer #
###############
class VAETrainer(RepresentationTrainer):
    def __init__(self, z_dim, decoder_lr=1e-4, beta_start_value=1e-3, beta_end_value=1,
                 beta_n_iterations=100000, beta_start_iteration=50000, **params):
        super(VAETrainer, self).__init__(z_dim=z_dim, **params)

        # Intialization of the decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            StochasticLinear(1024, 14**2*2, 'Normal')
        )

        # Adding the parameters of the estimator to the optimizer
        self.opt.add_param_group(
            {'params': self.decoder.parameters(), 'lr': decoder_lr}
        )

        # Defining the prior distribution as a factorized normal distribution
        self.mu = nn.Parameter(torch.zeros(self.z_dim), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(self.z_dim), requires_grad=False)
        self.prior = Normal(loc=self.mu, scale=self.sigma)
        self.prior = Independent(self.prior, 1)

        self.beta_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                   n_iterations=beta_n_iterations, start_iteration=beta_start_iteration)

    def _get_items_to_store(self):
        items_to_store = super(VAETrainer, self)._get_items_to_store()

        # Add the decoder to items_to_store
        items_to_store = items_to_store.union({'decoder'})

        return items_to_store

    def _compute_loss(self, data):
        # Ignore the second view and the label
        x = data['x']
        x = x.view(x.shape[0], -1)

        # Encode a batch of data
        p_z_given_x = self.encoder(x)

        # Sample from the posteriors with reparametrization
        z = p_z_given_x.rsample()

        # Rate
        rate = p_z_given_x.log_prob(z) - self.prior.log_prob(z)

        # Distortion
        prob_x_given_z = self.decoder(z)
        distortion = -prob_x_given_z.log_prob(x)

        # Average across the batch
        rate = rate.mean()
        distortion = distortion.mean()

        # Update the value of beta according to the policy
        beta = self.beta_scheduler(self.iterations)

        # Logging the components
        self._add_loss_item('loss/distortion', distortion.item())
        self._add_loss_item('loss/rate', rate.item())
        self._add_loss_item('loss/beta', beta)

        # Computing the loss function
        loss = distortion + beta * rate

        return loss


