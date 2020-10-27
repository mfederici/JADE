from models.base import RepresentationTrainer
from utils.functions import ScaleGrad
import torch
from torch import logsumexp
import torch.nn as nn
from utils.modules import OneHot, StopGrad
import utils.schedulers as scheduler_module

from architectures.CMNIST import CMNIST_SIZE, CMNIST_N_ENVS, CMNIST_N_CLASSES

import numpy as np


ESTIMATION_METHODS = {'ce', 'jsd', 'vd'}

#########################################
# IDA Adversarial Cross Entropy Trainer #
#########################################

class JICMITrainer(RepresentationTrainer):
    def __init__(self, z_dim, classifier, mi_estimator, optim, adv_optim, beta_scheduler, n_adv_steps, method,
                 env_mi_estimator=None, env_classifier=None, x_dim=None, **params):

        super(JICMITrainer, self).__init__(z_dim=z_dim, optim=optim, **params)

        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta_scheduler = getattr(scheduler_module, beta_scheduler['class'])(**beta_scheduler['params'])

        assert method in ESTIMATION_METHODS
        assert env_mi_estimator or env_classifier

        self.method = method

        self._encoder = self.encoder

        # TODO fix
        if x_dim is None:
            x_dim = CMNIST_SIZE

        self.mi_estimator = self.instantiate_architecture(mi_estimator, size1=z_dim, size2=x_dim)
        self.classifier = self.instantiate_architecture(classifier, z_dim=z_dim)

        self.step = 0

        self.n_adv_steps = n_adv_steps

        self.adv_opt = self.instantiate_optimizer(adv_optim, params=self.mi_estimator.parameters())

        if method == 'ce':
            self.env_classifier = self.instantiate_architecture(env_classifier, z_dim=z_dim)
            self.adv_opt.add_param_group(
                {'params': self.env_classifier.parameters()}
            )

        else:
            self.env_mi_estimator = self.instantiate_architecture(env_mi_estimator, size1=CMNIST_N_ENVS,
                                                                  size2=CMNIST_N_CLASSES + z_dim)
            self.long2onehot = OneHot(CMNIST_N_CLASSES)
            self.adv_opt.add_param_group(
                {'params': self.env_mi_estimator.parameters()}
            )

        self.adv_opt.add_param_group(
            {'params': self.classifier.parameters()}
        )

    def _get_items_to_store(self):
        items_to_store = super(JICMITrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'classifier',
            'mi_estimator',
            'adv_opt',
            'step'
        })

        if self.method == 'ce':
            items_to_store = items_to_store.union({'env_classifier'})
        else:
            items_to_store = items_to_store.union({'env_mi_estimator'})

        return items_to_store

    def _train_step(self, data):
        x = data['x']
        y = data['y']
        e = data['e']
        beta = self.beta_scheduler(self.iterations)

        if self.step < self.n_adv_steps:
            with torch.no_grad():
                # Encode a batch of data
                z = self._encoder(x).sample()

            ce_y_z = -self.classifier(z.detach()).log_prob(y)

            if (e == 0).sum() > 0:
                ce_y_ze0 = ce_y_z[e == 0].mean()
                self._add_loss_item('loss/CE_y_ze0', ce_y_ze0.item())
            if (e == 1).sum() > 0:
                ce_y_ze1 = ce_y_z[e == 1].mean()
                self._add_loss_item('loss/CE_y_ze1', ce_y_ze1.item())

            ce_y_z = ce_y_z.mean()

            mi_x_z_jsd, mi_x_z = self.mi_estimator(z.detach(), x.view(x.shape[0], -1))

            if self.method == 'ce':
                ce_e_yz = - self.env_classifier(z.detach(), y).log_prob(e).mean()
                mi_e_yz = np.log(2) - ce_e_yz
                self._add_loss_item('loss/CE_e_yz', ce_e_yz.item())
            else:
                zy = torch.cat([z, self.long2onehot(y)], 1)
                mi_e_yz_jsd, mi_e_yz_vd = self.env_mi_estimator(self.long2onehot(e), zy)

                if self.method == 'jsd':
                    mi_e_yz = mi_e_yz_jsd
                else:
                    mi_e_yz = mi_e_yz_vd

            loss = -mi_e_yz + ce_y_z - mi_x_z_jsd

            self.adv_opt.zero_grad()
            loss.backward()
            self.adv_opt.step()

            self._add_loss_item('loss/CE_y_z', ce_y_z.item())


            self._add_loss_item('loss/I_x_z', mi_x_z.item())
            self._add_loss_item('loss/Beta', beta)

            self.step += 1
        else:
            self.step = 0

            z = self._encoder(x=x).rsample()

            #mi_x_z_jsd, mi_x_z = self.mi_estimator(z, x.view(x.shape[0], -1))
            #ce_e_yz = -self.env_classifier(z, y).log_prob(e).mean()
            #ce_y_z = -self.classifier(z.detach()).log_prob(y).mean()

            if self.method == 'ce':
                ce_e_yz = - self.env_classifier(z, y).log_prob(e).mean()
                mi_e_yz = np.log(2) - ce_e_yz
                self._add_loss_item('loss/CE_e_yz', ce_e_yz.item())
            else:
                zy = torch.cat([z, self.long2onehot(y)], 1)
                mi_e_yz_jsd, mi_e_yz_vd = self.env_mi_estimator(self.long2onehot(e), zy)

                if self.method == 'jsd':
                    mi_e_yz = mi_e_yz_jsd
                else:
                    mi_e_yz = mi_e_yz_vd

                self._add_loss_item('loss/I_e_yz', mi_e_yz_vd.item())



            loss = mi_e_yz #+ ce_y_z #-(1 - beta) * mi_x_z_jsd

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            #self._add_loss_item('loss/CE_y_z', ce_y_z.item())
            #self._add_loss_item('loss/I_x_z', mi_x_z.item())
            self._add_loss_item('loss/Beta', beta)


class PJICMITrainer(JICMITrainer):
    def __init__(self, f_optim, feature_extractor, feature_mi_estimator, f_dim, **params):
        super(PJICMITrainer, self).__init__(x_dim=f_dim, **params)

        self.feature_mi_estimator = self.instantiate_architecture(feature_mi_estimator,
                                                                  size1=f_dim, size2=CMNIST_SIZE)
        # TODO update for different datasets

        self.feature_extractor = self.instantiate_architecture(feature_extractor, f_dim=f_dim)

        self.f_opt = self.instantiate_optimizer(f_optim, params=self.feature_extractor.parameters())

        self.f_opt.add_param_group(
            {'params': self.feature_mi_estimator.parameters()}
        )

        self.encoder = nn.Sequential(
            self.feature_extractor,
            StopGrad(),
            self._encoder
        )

    def _get_items_to_store(self):
        items_to_store = super(PJICMITrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'feature_mi_estimator'
        })

        return items_to_store

    def _train_step(self, data):
        x = data['x']

        f = self.feature_extractor(x)
        mi_f_x_jsd, mi_f_x_vd = self.feature_mi_estimator(f, x.view(x.shape[0], -1))

        loss = -mi_f_x_jsd

        self._add_loss_item('loss/I_f_x', mi_f_x_vd.item())

        self.f_opt.zero_grad()
        loss.backward()
        self.f_opt.step()

        data['x'] = f.detach()
        super(PJICMITrainer, self)._train_step(data)


