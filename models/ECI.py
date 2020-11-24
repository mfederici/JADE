from models.base import RepresentationTrainer
from utils.functions import ScaleGrad
import torch

import utils.schedulers as scheduler_module


###############################################
# Environment Conditional Independent Trainer #
###############################################

ADV_ALT_TRAIN = 'alternating'
ADV_SIM_TRAIN = 'simultaneous'

ADV_CE_LOSS = 'CE'

ADV_TRAIN_TYPES = {ADV_SIM_TRAIN, ADV_ALT_TRAIN}
ADV_OBJECTIVES = {ADV_CE_LOSS}


class ECITrainer(RepresentationTrainer):
    def __init__(self, z_dim, optim, beta_scheduler, n_adv_steps=5, adv_optim=None,
                 adv_objective=ADV_CE_LOSS, adv_train_type=ADV_ALT_TRAIN, label_classifier=None, conditional_env_classifier=None,
                 **params):

        super(ECITrainer, self).__init__(z_dim=z_dim, optim=optim, **params)

        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta_scheduler = getattr(scheduler_module, beta_scheduler['class'])(**beta_scheduler['params'])

        self.classifier = self.instantiate_architecture('LabelClassifier', z_dim=z_dim, **label_classifier)
        self.env_classifier = self.instantiate_architecture('ConditionalEnvClassifier', z_dim=z_dim, **conditional_env_classifier)

        self.n_adv_steps = n_adv_steps
        self.step = 0

        self.opt.add_param_group(
            {'params': self.classifier.parameters()}
        )

        assert adv_objective in ADV_OBJECTIVES
        self.adv_objective = adv_objective

        assert adv_train_type in ADV_TRAIN_TYPES
        assert adv_train_type != ADV_ALT_TRAIN or not (adv_optim is None)
        self.adv_train_type = adv_train_type

        if adv_train_type == ADV_ALT_TRAIN:
            assert not (adv_optim is None)
            self.adv_opt = self.instantiate_optimizer(adv_optim, params=self.env_classifier.parameters())
        else:
            self.opt.add_param_group(
                {'params': self.env_classifier.parameters()}
            )
            self.adv_opt = None

    def _get_items_to_store(self):
        items_to_store = super(ECITrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'classifier',
            'env_classifier'
        })
        if self.adv_opt:
            items_to_store = items_to_store.union({'adv_opt'})

        return items_to_store

    def _train_step(self, data):
        # Alternating adversarial procedure
        if self.adv_train_type == ADV_ALT_TRAIN:
           if self.step < self.n_adv_steps:
                # Train the two cross entropy and q(e|yz)
                loss = self._compute_adv_loss(data)

                self.adv_opt.zero_grad()
                loss.backward()
                self.adv_opt.step()
                self.step += 1
           else:
                # Train the representation p(z|x) and the classifier q(y|z)
                loss = self._compute_loss(data)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.step = 0
        # Simultaneous adversarial training
        elif self.adv_train_type == ADV_SIM_TRAIN:
            loss = self._compute_loss(data)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    def _compute_loss(self, data):
        x = data['x']
        y = data['y'].squeeze()
        e = data['e'].squeeze()
        beta = self.beta_scheduler(self.iterations)

        self.env_classifier.eval()

        # Encode a batch of data
        z = self.encoder(x=x).rsample()

        # Label Reconstruction
        p_y_given_z = self.classifier(z=ScaleGrad.apply(z, 1-beta))

        y_rec_loss = - p_y_given_z.log_prob(y).mean()

        p_e_given_zy = self.env_classifier(z=ScaleGrad.apply(z, -beta), y=y)
        if self.adv_objective == ADV_CE_LOSS:
            e_rec_loss = -p_e_given_zy.log_prob(e).mean()
            self._add_loss_item('loss/CE_e_yz', e_rec_loss.item())
        else:
            raise NotImplemented()

        loss = y_rec_loss + e_rec_loss

        self._add_loss_item('loss/CE_y_z', y_rec_loss.item())
        self._add_loss_item('loss/beta', beta)

        self.env_classifier.train()

        return loss

    def _compute_adv_loss(self, data):
        x = data['x']
        y = data['y'].squeeze()
        e = data['e'].squeeze()

        # Encode a batch of data
        with torch.no_grad():
            z = self.encoder(x=x).sample()

        p_e_given_zy = self.env_classifier(z=z.detach(), y=y)
        e_rec_loss = -p_e_given_zy.log_prob(e).mean()
        self._add_loss_item('loss/CE_e_yz', e_rec_loss.item())

        loss = e_rec_loss #+ y_rec_loss
        return loss
