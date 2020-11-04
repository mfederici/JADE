from models.base import RepresentationTrainer
from utils.functions import ScaleGrad
import torch
import torch.nn as nn
from torch.distributions import Normal

import utils.schedulers as scheduler_module

##############################################
# Variational Information Bottleneck Trainer #
##############################################


class VIBTrainer(RepresentationTrainer):
    def __init__(self, z_dim, classifier, optim, beta_scheduler, **params):

        super(VIBTrainer, self).__init__(z_dim=z_dim, optim=optim, **params)

        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta_scheduler = getattr(scheduler_module, beta_scheduler['class'])(**beta_scheduler['params'])

        self.classifier = self.instantiate_architecture(classifier, z_dim=z_dim)
        self.mu = nn.Parameter(torch.zeros([1, z_dim]), requires_grad=False)
        self.prior = Normal(self.mu, self.mu+1)

        self.opt.add_param_group(
            {'params': self.classifier.parameters()}
        )

    def _get_items_to_store(self):
        items_to_store = super(VIBTrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'classifier'
        })

        return items_to_store

    def _compute_loss(self, data):
        x = data['x']
        y = data['y']

        beta = self.beta_scheduler(self.iterations)

        # Encode a batch of data
        p_z_given_x = self.encoder(x=x)
        z = p_z_given_x.rsample()

        # Label Reconstruction
        p_y_given_z = self.classifier(z=z)
        y_rec_loss = - p_y_given_z.log_prob(y).mean()

        kl = (p_z_given_x.log_prob(z)-self.prior.log_prob(z)).sum() / x.shape[0]

        loss = (1-beta) * y_rec_loss + beta * kl

        self._add_loss_item('loss/CE_y_z', y_rec_loss.item())
        self._add_loss_item('loss/KL_z_x', y_rec_loss.item())

        return loss