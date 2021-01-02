from models.base import RegularizedClassifierTrainer

##############################################
# Variational Information Bottleneck Trainer #
##############################################


class VIBTrainer(RegularizedClassifierTrainer):
    def __init__(self, z_dim, prior=None, **params):

        super(VIBTrainer, self).__init__(z_dim=z_dim, **params)

        self.prior = self.instantiate_architecture('Prior', z_dim=z_dim, **prior)

        # TODO add prior parameters if any

    def _get_items_to_store(self):
        items_to_store = super(VIBTrainer, self)._get_items_to_store()

        # TODO add prior parameters if any

        return items_to_store

    def _compute_reg_loss(self, data, z):
        x = data['x']

        # Encode a batch of data
        p_z_given_x = self.encoder(x=x)

        p_z = self.prior()
        kl = (p_z_given_x.log_prob(z)-p_z.log_prob(z)).mean()

        self._add_loss_item('loss/KL_z_x', kl.item())

        return kl
