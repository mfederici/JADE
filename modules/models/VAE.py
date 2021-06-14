from core.trainer import Trainer
from torch.optim import Adam
from torch.utils.data import DataLoader


###################################
# Variational Autoencoder Trainer #
###################################


class VariationalAutoencoderTrainer(Trainer):
    def initalize(self, z_dim, encoder_layers, decoder_layers, beta, lr, batch_size, num_workers=0):
        # The value of the z_dim, n_encoder_layers, n_decoder_layers, beta lr, batch_size and n_workers are defined
        # in the configuration file or specified as additional arguments when running the train python file

        # Store the value of the regularization strength beta for the loss computation
        self.beta = beta

        # initialize the encoder "Encoder(z_dim, layers)" defined in the architecture python file
        self.encoder = self.instantiate_architecture('Encoder',
                                                     z_dim=z_dim, layers=encoder_layers)
        # initialize the encoder "Decoder(z_dim, layers, sigma)" defined in the architecture python file
        self.decoder = self.instantiate_architecture('Decoder',
                                                     z_dim=z_dim, layers=decoder_layers, sigma=0.1)
        # Initialize the Gaussian prior passing the number of dimensions
        self.prior = self.instantiate_architecture('Prior',
                                                   z_dim=z_dim)

        # Initialize the optimizer passing the parameters of encoder, decoder and the specified learning rate
        self.opt = Adam([
            {'params': encoder.parameters(), 'lr': lr},
            {'params': decoder.parameters(), 'lr': lr},
        ])

        # Store the parameters of encoder, decoder and optimizer
        self.add_attribute_to_store('encoder')
        self.add_attribute_to_store('decoder')
        self.add_attribute_to_store('opt')

        # Instantiate the data Loader
        self.train_loader = DataLoader(dataset=datasets['train'],
                                       batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def train_step(self, data):
        x = data['x']
        y = data['y']

        p_z_x = self.encoder(x)
        z = p_z.rsample()

        p_y_z = self.label_classifier(z)

        rec_loss = -p_y_z.log_prob(y).mean()
        reg_loss = (p_z_x.log_prob(z) - self.prior.log_prob(z)).mean()

        self.add_loss_item('Rec Loss', rec_loss.item())
        self.add_loss_item('Reg Loss', reg_loss.item())

        loss = rec_loss + self.beta * reg_loss

        self.opt.zero_grad()
        loss.backwards()
        self.opt.step()
