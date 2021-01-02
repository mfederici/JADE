from models.base import RegularizedClassifierTrainer
from utils.functions import ScaleGrad
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.distributions import Normal, Bernoulli
import torch.autograd as autograd
import torch.nn.functional as F

import utils.schedulers as scheduler_module

#######################################
# Invariant Risk Minimization Trainer #
#######################################

class IRMTrainer(RegularizedClassifierTrainer):
    # See https://github.com/facebookresearch/DomainBed/blob/master/domainbed/algorithms.py

    def _compute_irm_loss(self, logits, y):
        device = logits.device
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2].long())
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2].long())
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]

        return torch.sum(grad_1 * grad_2)

    def _compute_reg_loss(self, data, z):
        y = data['y'].squeeze()

        p_y_given_z = self.classifier(z=z)
        penalty = self._compute_irm_loss(p_y_given_z.logits, y)
        self._add_loss_item('loss/Gradient_penalty', penalty.item())

        return penalty
