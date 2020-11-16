from models.base import RepresentationTrainer
from utils.functions import ScaleGrad
import torch

import utils.schedulers as scheduler_module


#######################################
# Empirical Risk Minimization Trainer #
#######################################


class ERMTrainer(RepresentationTrainer):
    def __init__(self, z_dim, optim, label_classifier=None, **params):

        super(ERMTrainer, self).__init__(z_dim=z_dim, optim=optim, **params)

        self.classifier = self.instantiate_architecture('LabelClassifier', z_dim=z_dim, **label_classifier)

        self.opt.add_param_group(
            {'params': self.classifier.parameters()}
        )

    def _get_items_to_store(self):
        items_to_store = super(ERMTrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'classifier'
        })

        return items_to_store

    def _compute_loss(self, data):
        x = data['x']
        y = data['y']

        # Encode a batch of data
        z = self.encoder(x=x).rsample()

        # Label Reconstruction
        p_y_given_z = self.classifier(z=z)
        y_rec_loss = - p_y_given_z.log_prob(y).mean()

        loss = y_rec_loss

        self._add_loss_item('loss/CE_y_z', y_rec_loss.item())

        return loss
