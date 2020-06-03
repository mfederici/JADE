from training.base import RepresentationTrainer, init_optimizer
from utils.schedulers import LinearScheduler
import torch.nn as nn
import torch
from utils.modules import MIEstimator
from pytorch_revgrad import RevGrad
from torch.nn.functional import softplus, softmax
from torch.optim import Adam


###############
# IDA Trainer #
###############
class IDATrainer(RepresentationTrainer):
    def __init__(self, z_dim, beta_start_value=1e-3, beta_end_value=1,
                 beta_n_iterations=100000, beta_start_iteration=50000, optimizer_name='Adam', encoder_lr=1e-4,
                 n_classes=10, n_env=2, n_adv_steps=4, **params):
        # The neural networks architectures and initialization procedure is analogous to Multi-View InfoMax
        super(IDATrainer, self).__init__(z_dim=z_dim, optimizer_name=optimizer_name, encoder_lr=encoder_lr, **params)

        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta_scheduler = LinearScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                   n_iterations=beta_n_iterations, start_iteration=beta_start_iteration)

        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, n_classes)
        )
        self.n_classes = n_classes
        self.n_env = n_env

        self.mi_estimator = MIEstimator(n_env, n_classes+z_dim)

        self.n_adv_steps = n_adv_steps
        self.step = 0
        self.adv_opt = Adam([
            {'params': self.mi_estimator.parameters(), 'lr': encoder_lr}
        ])

        self.opt.add_param_group(
            {'params': self.classifier.parameters(), 'lr': encoder_lr}
        )

    def _get_items_to_store(self):
        items_to_store = super(IDATrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'classifier',
            'mi_estimator',
            'adv_opt'
        })

        return items_to_store

    def _train_step(self, data):
        if self.step < self.n_adv_steps:
            loss = self._compute_adv_loss(data)

            self.adv_opt.zero_grad()
            loss.backward()
            self.adv_opt.step()
            self.step += 1
        else:
            loss = self._compute_loss(data)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.step = 0

    def _compute_loss(self, data):
        # Read the two views v1 and v2 and ignore the label y
        x = data['x']
        y = data['y'].long()
        e = data['e'].long()

        # Encode a batch of data
        z = self.encoder(x).mean

        # Label Reconstruction
        p_y_given_z = torch.distributions.Categorical(logits=self.classifier(z))

        y_rec_loss = - p_y_given_z.log_prob(y).mean()

        zy = torch.cat([z,  torch.eye(self.n_classes)[y]], 1)
        e = torch.eye(self.n_env)[e]

        mi_yz_e, mi_yz_e_v = self.mi_estimator(zy, e)

        beta = self.beta_scheduler(self.iterations)
        loss = beta * y_rec_loss + mi_yz_e

        self._add_loss_item('loss/CE_y_z', y_rec_loss.item())
        self._add_loss_item('loss/I_e_yz', mi_yz_e_v.item())
        self._add_loss_item('loss/beta', beta)

        return loss

    def _compute_adv_loss(self, data):
        # Read the two views v1 and v2 and ignore the label y
        x = data['x']
        y = data['y'].long()
        e = data['e'].long()

        # Encode a batch of data
        with torch.no_grad():
            z = self.encoder(x).mean

        zy = torch.cat([z.detach(), torch.eye(self.n_classes)[y]], 1)
        e = torch.eye(self.n_env)[e]

        mi_yz_e, mi_yz_e_v = self.mi_estimator(zy, e)

        self._add_loss_item('loss/I_e_yz', mi_yz_e_v.item())

        return -mi_yz_e_v


