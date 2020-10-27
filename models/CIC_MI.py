from models.base import RepresentationTrainer
from utils.functions import ScaleGrad
import torch
from torch import logsumexp
import torch.nn as nn

import utils.schedulers as scheduler_module

# TODO remove
from architectures.CMNIST import CMNIST_SIZE

#########################################
# IDA Adversarial Cross Entropy Trainer #
#########################################


class MarginalClassifier(nn.Module):
    def __init__(self, joint_classifier):
        super(MarginalClassifier, self).__init__()
        self.joint_classifier = joint_classifier

    def forward(self, z):
        logits = self.joint_classifier.compute_logits(z=z)
        return torch.distributions.Categorical(logits=logsumexp(logits, 2))

class CICMITrainer(RepresentationTrainer):
    def __init__(self, z_dim, joint_classifier, mi_estimator, optim, beta_scheduler, n_adv_steps, **params):

        super(CICMITrainer, self).__init__(z_dim=z_dim, optim=optim, **params)

        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta_scheduler = getattr(scheduler_module, beta_scheduler['class'])(**beta_scheduler['params'])

        self.joint_classifier = self.instantiate_architecture(joint_classifier, z_dim=z_dim)
        self.mi_estimator = self.instantiate_architecture(mi_estimator, size1=z_dim, size2=CMNIST_SIZE)
        self.classifier = MarginalClassifier(self.joint_classifier)

        self.step = 0

        self.n_adv_steps = n_adv_steps

        self.opt.add_param_group(
            {'params': self.joint_classifier.parameters()}
        )

        self.opt.add_param_group(
            {'params': self.mi_estimator.parameters()}
        )

    def _get_items_to_store(self):
        items_to_store = super(CICMITrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'joint_classifier',
            'mi_estimator'
        })

        return items_to_store

    def _train_step(self, data):



        x = data['x']
        y = data['y']
        e = data['e']
        beta = self.beta_scheduler(self.iterations)

        # Encode a batch of data
        z = self.encoder(x=x).rsample()

        if self.step < self.n_adv_steps:
            logits_ye_given_z_nograd = self.joint_classifier.compute_logits(z=z.detach())

            norm_nograd = logsumexp(logits_ye_given_z_nograd.view(x.shape[0],-1), 1)

            logits_ye_z_nograd = logits_ye_given_z_nograd.view(x.shape[0], -1).gather(1, (e + y * logits_ye_given_z_nograd.shape[2]).unsqueeze(1)).squeeze()

            ce_ye_z = (-logits_ye_z_nograd + norm_nograd).mean()

            #mi_x_z_jsd, mi_x_z = self.mi_estimator(z, x.view(x.shape[0], -1))

            loss = ce_ye_z #- mi_x_z_jsd

            self.opt.zero_grad()

            loss.backward()
            self.encoder.zero_grad()

            self.opt.step()

            self._add_loss_item('loss/CE_ye_z', ce_ye_z.item())
            #self._add_loss_item('loss/I_X_Z', mi_x_z.item())
            self._add_loss_item('loss/Beta', beta)

            self.step += 1
        else:
            self.step = 0

            logits_ye_given_z = self.joint_classifier.compute_logits(z=z)
            norm = logsumexp(logits_ye_given_z.view(x.shape[0],-1), 1)

            logits_y_z = logsumexp(logits_ye_given_z, 2).gather(1, y.unsqueeze(1)).squeeze()
            logits_e_z = logsumexp(logits_ye_given_z, 1).gather(1, e.unsqueeze(1)).squeeze()
            logits_ye_z = logits_ye_given_z.view(x.shape[0], -1).gather(1,
                                                                        (e + y * logits_ye_given_z.shape[2]).unsqueeze(
                                                                            1)).squeeze()

            ce_y_z = (-logits_y_z + norm).mean()
            ce_e_z = (-logits_e_z + norm).mean()

            mi_y_e_z = logits_ye_z - logits_y_z - logits_e_z + norm

            mi_y_e_z = mi_y_e_z.mean()

            mi_x_z_jsd, mi_x_z = self.mi_estimator(z, x.view(x.shape[0], -1))

            loss_z = -(1 - beta) * mi_x_z_jsd + beta * mi_y_e_z

            self.opt.zero_grad()
            loss_z.backward()
            self.joint_classifier.zero_grad()
            self.opt.step()

            self._add_loss_item('loss/CE_y_z', ce_y_z.item())
            self._add_loss_item('loss/CE_e_z', ce_e_z.item())
            self._add_loss_item('loss/I_X_Z', mi_x_z.item())
            self._add_loss_item('loss/I_Y_E_Z', mi_y_e_z.item())
            self._add_loss_item('loss/Beta', beta)

