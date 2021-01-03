from models.base import AdversarialRepresentationTrainer
from utils.functions import ScaleGrad
import torch

import utils.schedulers as scheduler_module


###############################################
# Environment Conditional Independent Trainer #
###############################################

class ECITrainer(AdversarialRepresentationTrainer):
    def __init__(self, z_dim, conditional_env_classifier=None, **params):

        conditional_env_classifier.update({'class_name': 'ConditionalEnvClassifier', 'z_dim': z_dim})
        super(ECITrainer, self).__init__(adversary=conditional_env_classifier, z_dim=z_dim, **params)

    def _compute_adv_loss(self, data, z=None):
        x = data['x']
        y = data['y'].squeeze()
        e = data['e'].squeeze()

        # Compute the loss without gradient for the encoder
        if z is None:
            self.adversary.train()
            # Encode a batch of data
            with torch.no_grad():
                z = self.encoder(x=x).sample()

            p_e_given_zy = self.adversary(z=z.detach(), y=y)
            e_rec_loss = -p_e_given_zy.log_prob(e).mean()

        # Compute the loss with the gradient with respect to the encoder
        else:
            self.adversary.eval()
            p_e_given_zy = self.adversary(z=z, y=y)
            e_rec_loss = -p_e_given_zy.log_prob(e).mean()
            self._add_loss_item('loss/CE_e_yz', e_rec_loss.item())

        return e_rec_loss
