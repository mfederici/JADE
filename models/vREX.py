from models.base import RepresentationTrainer
from utils.functions import ScaleGrad
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.distributions import Normal, Bernoulli
import torch.autograd as autograd

import utils.schedulers as scheduler_module

###############################################
# variance-reduction-based Risk EXtrapolation #
###############################################

# http://arxiv.org/abs/2003.00688
from utils.modules import OneHot


class vREXTrainer(RepresentationTrainer):
    def __init__(self, z_dim, optim, beta_scheduler, label_classifier=None, **params):

        super(vREXTrainer, self).__init__(z_dim=z_dim, optim=optim, **params)

        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta_scheduler = getattr(scheduler_module, beta_scheduler['class'])(**beta_scheduler['params'])

        self.classifier = self.instantiate_architecture('LabelClassifier', z_dim=z_dim, **label_classifier)
        if not hasattr(self.arch_module, 'N_ENVS'):
            raise Exception('Please specify the variable N_ENVS (number of environments) in %s' %
                            self.arch_module.__file__)
        n_envs = getattr(self.arch_module, 'N_ENVS')

        self.one_hot = OneHot(n_envs)

        # Dummy vector used for gradient penalization
        self.scale = torch.nn.Parameter(torch.ones(1).float())

        self.opt.add_param_group(
            {'params': self.classifier.parameters()}
        )

    def _get_items_to_store(self):
        items_to_store = super(vREXTrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'classifier'
        })

        return items_to_store

    def _compute_loss(self, data):
        x = data['x']
        y = data['y'].float()
        e = data['e']

        beta = self.beta_scheduler(self.iterations)

        # Encode a batch of data
        p_z_given_x = self.encoder(x=x)
        z = p_z_given_x.rsample()

        # Label Reconstruction
        p_y_given_z = self.classifier(z=z)

        y_rec_loss = -p_y_given_z.log_prob(y).squeeze()

        # Long to one hot encoding
        one_hot_e = self.one_hot(e.long())

        # Environment variance penalty
        e_sum = one_hot_e.sum(0)
        env_loss = (y_rec_loss.unsqueeze(1) * one_hot_e).sum(0)
        env_loss[e_sum > 0] = env_loss[e_sum > 0] / e_sum[e_sum > 0]
        loss_variance = ((env_loss - env_loss[e_sum > 0].mean()) ** 2)[e_sum > 0].mean()

        loss = (1-beta) * y_rec_loss.mean() + beta * loss_variance

        self._add_loss_item('loss/CE_y_z', y_rec_loss.mean().item())
        self._add_loss_item('loss/V_CE_y_z', loss_variance.item())
        self._add_loss_item('loss/beta', beta)

        return loss
