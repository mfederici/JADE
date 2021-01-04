from models.base import RegularizedClassifierTrainer
from utils.modules import OneHot

import utils.schedulers as scheduler_module

###############################################
# Variance-reduction-based Risk Extrapolation #
###############################################


# http://arxiv.org/abs/2003.00688
class VRExTrainer(RegularizedClassifierTrainer):
    def __init__(self, use_std=False, **params):
        super(VRExTrainer, self).__init__(**params)
        n_envs = getattr(self.arch_module, 'N_ENVS')
        self.use_std = use_std
        self.one_hot = OneHot(n_envs)

    def _compute_reg_loss(self, data, p_y_given_z, **params):
        y = data['y'].float().squeeze()
        e = data['e'].squeeze()

        y_rec_loss = -p_y_given_z.log_prob(y).squeeze()

        # Long to one hot encoding
        one_hot_e = self.one_hot(e.long())

        # Environment variance penalty
        e_sum = one_hot_e.sum(0)
        env_loss = (y_rec_loss.unsqueeze(1) * one_hot_e).sum(0)
        env_loss[e_sum > 0] = env_loss[e_sum > 0] / e_sum[e_sum > 0]
        loss_variance = ((env_loss - env_loss[e_sum > 0].mean()) ** 2)[e_sum > 0].mean()

        self._add_loss_item('loss/V_CE_y_z', loss_variance.item())

        if self.use_std:
            loss = loss_variance ** 0.5
        else:
            loss = loss_variance

        return loss
