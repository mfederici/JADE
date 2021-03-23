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

    def _compute_irm_loss(self, logits, y, e):
        device = logits.device
        scale = torch.tensor(1.).to(device).requires_grad_()
        penalty = 0
        for i in range(2):
            logits_e = logits[e == i]
            y_e = y[e == i]
            loss_1 = F.cross_entropy(logits_e[::2] * scale, y_e[::2].long())
            loss_2 = F.cross_entropy(logits_e[1::2] * scale, y_e[1::2].long())
            grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
            grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
            penalty += torch.sum(grad_1 * grad_2)
        return penalty

    def _compute_reg_loss(self, data, z, **params):
        y = data['y'].squeeze()
        e = data['e'].squeeze()

        p_y_given_z = self.classifier(z=z)
        penalty = self._compute_irm_loss(p_y_given_z.logits, y, e)
        self._add_loss_item('loss/Gradient_penalty', penalty.item())

        return penalty
