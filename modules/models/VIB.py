from jade.trainer import Trainer
from torch.optim import Adam
from torch.utils.data import DataLoader

##############################################
# Variational Information Bottleneck Trainer #
##############################################


class VIBTrainer(Trainer):
    def initialize(self,  z_dim, encoder_layers, classifier_layers, beta, lr, batch_size, num_workers=0):
        self.beta = beta

        # Instantiating the architectures
        self.encoder = self.instantiate_architecture('Encoder', z_dim=z_dim, layers=encoder_layers)
        self.label_classifier = self.instantiate_architecture('LabelClassifier', z_dim=z_dim, layers=classifier_layers)
        self.prior = self.instantiate_architecture('Prior', z_dim=z_dim)

        # Definition of the optimizer
        self.optim = Adam([
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.label_classifier.parameters(), 'lr': lr},
        ])

        # Specify the attributes to store and load
        self.add_attribute_to_store('encoder')
        self.add_attribute_to_store('label_classifier')
        self.add_attribute_to_store('optim')

        # Instantiate the data Loader
        self.train_loader = DataLoader(dataset=self.datasets['train'],
                                       batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def train_step(self, data):
        loss_components = self.compute_loss_components(data)

        loss = loss_components['rec_loss'] + self.beta * loss_components['reg_loss']

        self.add_loss_item('Rec Loss', loss_components['rec_loss'].item())
        self.add_loss_item('Reg Loss', loss_components['reg_loss'].item())

        self.opt.zero_grad()
        loss.backwards()
        self.opt.step()

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


