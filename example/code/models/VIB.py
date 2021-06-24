from jade.model import Model
from torch.optim import Adam
from torch.utils.data import DataLoader

######################################
# Variational Information Bottleneck #
######################################


class VariationalInformationBottleneck(Model):
    def initialize(self,  z_dim, encoder_layers, beta, classifier_class='LabelClassifier'):
        self.beta = beta

        # Instantiating the architectures
        self.encoder = self.instantiate_architecture('Encoder', z_dim=z_dim, layers=encoder_layers)
        # If multiple implementations are available for one architectures, the class reference can be passed
        # as a parameter, which can be specifier as a hyper-parameter
        self.label_classifier = self.instantiate_architecture(classifier_class, z_dim=z_dim)
        self.prior = self.instantiate_architecture('Prior', z_dim=z_dim)

    def compute_loss(self, data):
        loss_components = self.compute_loss_components(data)

        loss = loss_components['rec_loss'] + self.beta * loss_components['reg_loss']

        self.add_loss_item('TrainLog/ReconstructionLoss', loss_components['rec_loss'].item())
        self.add_loss_item('TrainLog/RegularizationLoss', loss_components['reg_loss'].item())
        self.add_loss_item('TrainLog/Loss', loss.item())

        return loss

    def compute_loss_components(self, data):
        x = data['x']
        y = data['y']

        q_z_x = self.encoder(x)
        z = q_z_x.rsample()

        p_y_z = self.label_classifier(z)

        rec_loss = -p_y_z.log_prob(y).mean()
        reg_loss = (q_z_x.log_prob(z) - self.prior().log_prob(z)).mean()

        return {'rec_loss': rec_loss, 'reg_loss': reg_loss}

    def predict(self, x, sample=True):
        q_z_x = self.encoder(x)
        if sample:
            z = q_z_x.rsample()
        else:
            z = q_z_x.mean

        return self.label_classifier(z)
