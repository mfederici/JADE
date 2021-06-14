from core.trainer import Trainer
from torch.optim import Adam
from torch.utils.data import DataLoader

##############################################
# Variational Information Bottleneck Trainer #
##############################################


class VIBTrainer(Trainer):
    def initalize(self,  z_dim, encoder_layers, classifier_layers, beta, lr, batch_size, num_workers=0):
        self.beta = beta

        # Instantiating the architectures
        self.encoder = self.instantiate_architecture('Encoder', z_dim=z_dim, layers=encoder_layers)
        self.label_classifier = self.instantiate_architecture('LabelClassifier', z_dim=z_dim, layers=classifier_layers)
        self.prior_params = self.instantiate_architecture('Encoder', **prior_params)


        # Definition of the optimizer
        self.optim = Adam([
            {'params': encoder.parameters(), 'lr': lr},
            {'params': label_classifier_params.parameters(), 'lr': lr},
        ])

        # Specify the attributes to store and load
        self.add_attribute_to_store('encoder')
        self.add_attribute_to_store('label_classifier')
        self.add_attribute_to_store('optim')

        # Instantiate the data Loader
        self.train_loader = DataLoader(dataset=self.datasets['train'],
                                       batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def train_step(self, data):
        x = data['x']
        y = data['y']

        p_z_x = self.encoder(x)
        z = p_z.rsample()

        p_y_z = self.label_classifier(z)

        rec_loss = -p_y_z.log_prob(y).mean()
        reg_loss = (p_z_x.log_prob(z)-self.prior.log_prob(z)).mean()

        self.add_loss_item('Rec Loss', rec_loss.item())
        self.add_loss_item('Reg Loss', reg_loss.item())

        loss = rec_loss + self.beta * reg_loss

        self.opt.zero_grad()
        loss.backwards()
        self.opt.step()
