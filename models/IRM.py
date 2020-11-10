from models.base import RepresentationTrainer
from utils.functions import ScaleGrad
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli
import torch.autograd as autograd

import utils.schedulers as scheduler_module

#######################################
# Invariant Risk Minimization Trainer #
#######################################


class IRMTrainer(RepresentationTrainer):
    def __init__(self, z_dim, classifier, optim, beta_scheduler, **params):

        super(IRMTrainer, self).__init__(z_dim=z_dim, optim=optim, **params)

        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta_scheduler = getattr(scheduler_module, beta_scheduler['class'])(**beta_scheduler['params'])

        self.classifier = self.instantiate_architecture(classifier, z_dim=z_dim)
        # Dummy vector used for gradient penalization
        self.scale = torch.nn.Parameter(torch.ones(1).float())
        self.opt.add_param_group(
            {'params': self.classifier.parameters()}
        )

    def _get_items_to_store(self):
        items_to_store = super(IRMTrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'classifier'
        })

        return items_to_store

    # See https://github.com/facebookresearch/DomainBed/blob/master/domainbed/algorithms.py
    def _compute_regularization(self, y_rec_loss):
        grad_1 = autograd.grad(y_rec_loss[0::2].mean(), [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(y_rec_loss[1::2].mean(), [self.scale], create_graph=True)[0]
        result = torch.mean(grad_1 * grad_2)
        return result

    def _compute_loss(self, data):
        x = data['x']
        y = data['y'].float()

        beta = self.beta_scheduler(self.iterations)

        # Encode a batch of data
        p_z_given_x = self.encoder(x=x)
        z = p_z_given_x.rsample()

        # Label Reconstruction
        p_y_given_z = self.classifier(z=z)
        p_y_given_z = Bernoulli(logits=self.scale * p_y_given_z.logits[:, 0])

        y_rec_loss = - p_y_given_z.log_prob(y)

        # Gradient penalty
        penalty = self._compute_regularization(y_rec_loss)

        loss = (1-beta) * y_rec_loss.mean() + beta * penalty

        self._add_loss_item('loss/CE_y_z', y_rec_loss.item())
        self._add_loss_item('loss/Gradient_penalty', penalty.item())

        return loss
