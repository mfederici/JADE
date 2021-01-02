from models.base import AdversarialRepresentationTrainer
from utils.functions import ScaleGrad
import torch

import utils.schedulers as scheduler_module


###################################
# Environment Independent Trainer #
###################################

class EITrainer(AdversarialRepresentationTrainer):
    def __init__(self, z_dim, env_classifier=None, **params):

        env_classifier.update({'class_name': 'EnvClassifier', 'z_dim': z_dim})
        super(EITrainer, self).__init__(adversary=env_classifier, z_dim=z_dim, **params)

    def _compute_adv_loss(self, data, z=None):
        x = data['x']
        e = data['e'].squeeze()

        # Compute the loss without gradient for the encoder
        if z is None:
            # Encode a batch of data
            with torch.no_grad():
                z = self.encoder(x=x).sample()

            p_e_given_z = self.adversary(z=z.detach())

        # Compute the loss with the gradient with respect to the encoder
        else:
            p_e_given_z = self.adversary(z=z)

        e_rec_loss = -p_e_given_z.log_prob(e).mean()
        self._add_loss_item('loss/CE_e_z', e_rec_loss.item())
        loss = e_rec_loss

        return loss
