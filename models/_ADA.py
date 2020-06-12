from training.base import RepresentationTrainer, init_optimizer
from utils.schedulers import LinearScheduler
from utils.modules import StochasticLinear
from utils.functions import ScaleGrad
import torch.nn as nn
import torch
from torch.distributions import Categorical
from utils.modules import MIEstimator
from pytorch_revgrad import RevGrad
from torch.nn.functional import softplus, softmax
from torch.optim import Adam
import os
import torch.optim as optim



class ScaleGradLayer(nn.Linear):
    def __init__(self, coeff):
        super(ScaleGradLayer, self).__init__()
        self.coeff = coeff

    def forward(self, input):
        return ScaleGrad(input, self.coeff)


###############
# ADA Trainer #
###############
class ADATrainer(RepresentationTrainer):
    def __init__(self, z_dim, beta_start_value=1e-3, beta_end_value=1,
                 beta_n_iterations=100000, beta_start_iteration=50000, optimizer_name='Adam', encoder_lr=1e-4, ce_lr=None,
                 n_classes=10, n_env=2, n_ce_steps=4, load_encoder=None, **params):
        # The neural networks architectures and initialization procedure is analogous to Multi-View InfoMax
        super(ADATrainer, self).__init__(z_dim=z_dim, optimizer_name=optimizer_name, encoder_lr=encoder_lr, **params)

        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta_scheduler = LinearScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                   n_iterations=beta_n_iterations, start_iteration=beta_start_iteration)

        if load_encoder:
            state_dict = torch.load(os.path.join(load_encoder), map_location=torch.device('cpu'))
            self.encoder.load_state_dict(state_dict['encoder'])

        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 1024),
            #nn.Dropout(0.5),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            ##nn.Dropout(0.5),
            nn.ReLU(True),
            StochasticLinear(1024, n_classes, 'Categorical')
        )

        self.n_classes = n_classes
        self.n_env = n_env

        self.e_classifier = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            StochasticLinear(1024, n_env, 'Categorical')
        )

        self.n_ce_steps = n_ce_steps
        self.step = 0

        if ce_lr is None:
            ce_lr = encoder_lr

        # self.opt.add_param_group(
        #     {'params': self.e_classifier.parameters(), 'lr': ce_lr}
        # )
        #
        self.opt.add_param_group(
            {'params': self.classifier.parameters(), 'lr': ce_lr}
        )

        OptimizerClass = getattr(optim, optimizer_name)
        self.ce_opt = OptimizerClass([
            {'params': self.e_classifier.parameters(), 'lr': ce_lr},
            #{'params': self.classifier.parameters(), 'lr': ce_lr}
        ])

    def _get_items_to_store(self):
        items_to_store = super(ADATrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'classifier',
            'e_classifier',
            'ce_opt'
        })

        return items_to_store

    def _train_step(self, data):
       if self.step < self.n_ce_steps:
            # Train the two cross entropies q(y|z) and q(e|yz)
            loss = self._compute_ce_loss(data)

            self.ce_opt.zero_grad()
            loss.backward()
            self.ce_opt.step()
            self.step += 1
       else:
            # Train the representation p(z|x)
            loss = self._compute_loss(data)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.step = 0

    def _compute_loss(self, data):
        x = data['x']
        y = data['y'].long()
        e = data['e'].long()
        beta = self.beta_scheduler(self.iterations)

        # Encode a batch of data
        z = self.encoder(x).rsample()

        # Label Reconstruction
        p_y_given_z = self.classifier(ScaleGrad.apply(z, 1-beta))

        y_rec_loss = - p_y_given_z.log_prob(y).mean()

        p_e_given_z = self.e_classifier(ScaleGrad.apply(z, -beta))
        e_rec_loss = -p_e_given_z.log_prob(e).mean()

        loss = y_rec_loss + e_rec_loss

        self._add_loss_item('loss/CE_y_z', y_rec_loss.item())
        self._add_loss_item('loss/CE_e_z', e_rec_loss.item())
        self._add_loss_item('loss/beta', beta)

        return loss

    def _compute_ce_loss(self, data):
        # Read the two views v1 and v2 and ignore the label y
        x = data['x']
        y = data['y'].long()
        e = data['e'].long()

        # Encode a batch of data
        with torch.no_grad():
            z = self.encoder(x).sample()

        p_e_given_z = self.e_classifier(z.detach())
        e_rec_loss = -p_e_given_z.log_prob(e).mean()

        self._add_loss_item('loss/CE_e_z', e_rec_loss.item())

        loss = e_rec_loss
        return loss
