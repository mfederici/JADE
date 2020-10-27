from models.base import RepresentationTrainer
from utils.functions import ScaleGrad
import torch
from torch import logsumexp
import torch.nn as nn

import utils.schedulers as scheduler_module


#########################################
# IDA Adversarial Cross Entropy Trainer #
#########################################

ADV_ALT_TRAIN = 'alternating'
ADV_SIM_TRAIN = 'simultaneous'

ADV_TRAIN_TYPES = {ADV_SIM_TRAIN, ADV_ALT_TRAIN}


class MarginalClassifier(nn.Module):
    def __init__(self, joint_classifier):
        super(MarginalClassifier, self).__init__()
        self.joint_classifier = joint_classifier

    def forward(self, z):
        logits = self.joint_classifier.compute_logits(z=z)
        return torch.distributions.Categorical(logits=logsumexp(logits, 2))

class CICTrainer(RepresentationTrainer):
    def __init__(self, z_dim, joint_classifier, optim, beta_scheduler, n_adv_steps=5, wdist=False, **params):

        super(CICTrainer, self).__init__(z_dim=z_dim, optim=optim, **params)

        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta_scheduler = getattr(scheduler_module, beta_scheduler['class'])(**beta_scheduler['params'])

        self.joint_classifier = self.instantiate_architecture(joint_classifier, z_dim=z_dim)
        self.classifier = MarginalClassifier(self.joint_classifier)

        self.n_adv_steps = n_adv_steps
        self.step = 0

        self.wdist = wdist

        self.opt.add_param_group(
            {'params': self.joint_classifier.parameters()}
        )

    def _get_items_to_store(self):
        items_to_store = super(CICTrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'joint_classifier',
        })

        return items_to_store

    def _train_step(self, data):

        x = data['x']
        y = data['y']
        e = data['e']
        beta = self.beta_scheduler(self.iterations)

        # Encode a batch of data
        z = self.encoder(x=x).rsample()

        # Label Reconstruction
        logits_ye_given_z = self.joint_classifier.compute_logits(z=z)
        logits_ye_given_z_nograd = self.joint_classifier.compute_logits(z=z.detach())

        norm = logsumexp(logits_ye_given_z.view(x.shape[0],-1), 1)
        norm_nograd = logsumexp(logits_ye_given_z_nograd.view(x.shape[0],-1), 1)

        logits_y_z = logsumexp(logits_ye_given_z, 2).gather(1, y.unsqueeze(1)).squeeze()
        logits_e_z = logsumexp(logits_ye_given_z, 1).gather(1, e.unsqueeze(1)).squeeze()
        logits_ye_z = logits_ye_given_z.view(x.shape[0], -1).gather(1, (e + y * logits_ye_given_z.shape[2]).unsqueeze(1)).squeeze()
        logits_ye_z_nograd = logits_ye_given_z_nograd.view(x.shape[0], -1).gather(1, (e + y * logits_ye_given_z.shape[2]).unsqueeze(1)).squeeze()

        ce_y_z = (-logits_y_z + norm).mean()
        ce_e_z = (-logits_e_z + norm).mean()

        mi_y_e_z = logits_ye_z - logits_y_z - logits_e_z + norm

        mi_y_e_z = mi_y_e_z.mean()

        loss_z = (1-beta) * ce_y_z + beta * mi_y_e_z

        ce_ye_z = - logits_ye_z_nograd + norm_nograd
        ce_ye_z = ce_ye_z.mean()

        self.opt.zero_grad()
        loss_z.backward()

        self.joint_classifier.zero_grad()
        ce_ye_z.backward()

        self.opt.step()

        self._add_loss_item('loss/CE_y_z', ce_y_z.item())
        self._add_loss_item('loss/CE_e_z', ce_e_z.item())
        self._add_loss_item('loss/CE_ye_z', ce_ye_z.item())
        self._add_loss_item('loss/I_Y_E_Z', mi_y_e_z.item())
        self._add_loss_item('loss/Beta', beta)