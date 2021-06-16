from jade.trainer import Trainer
from torch.optim import Adam
from torch.utils.data import DataLoader


###################################
# Variational Autoencoder Trainer #
###################################


class VariationalAutoencoderTrainer(Trainer):
    def initialize(self, z_dim, encoder_layers, decoder_layers, beta, lr, batch_size, num_workers=0, sigma=1):
        # The value of the z_dim, n_encoder_layers, n_decoder_layers, beta lr, batch_size and n_workers are defined
        # in the configuration file or specified as additional arguments when running the train python file

        # Store the value of the regularization strength beta for the loss computation
        self.beta = beta

        # initialize the encoder "Encoder(z_dim, layers)" defined in the architecture python file
        self.encoder = self.instantiate_architecture('Encoder',
                                                     z_dim=z_dim, layers=encoder_layers)
        # initialize the encoder "Decoder(z_dim, layers, sigma)" defined in the architecture python file
        self.decoder = self.instantiate_architecture('Decoder',
                                                     z_dim=z_dim, layers=decoder_layers, sigma=sigma)
        # Initialize the Gaussian prior passing the number of dimensions
        self.prior = self.instantiate_architecture('Prior',
                                                   z_dim=z_dim)

        # Initialize the optimizer passing the parameters of encoder, decoder and the specified learning rate
        self.opt = Adam([
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.decoder.parameters(), 'lr': lr},
        ])

        # Store the parameters of encoder, decoder and optimizer
        self.add_attribute_to_store('encoder')
        self.add_attribute_to_store('decoder')
        self.add_attribute_to_store('opt')

        # Instantiate the data Loader
        self.train_loader = DataLoader(dataset=self.datasets['train'],
                                       batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def train_step(self, data):
        loss_components = self.compute_loss_components(data)

        # Add the two loss components to the log
        # Note that only scalars are currently supported although custom logging can be implemented
        # by extending the implemented methods
        self.add_loss_item('Rec Loss', loss_components['rec_loss'].item())
        self.add_loss_item('Reg Loss', loss_components['reg_loss'].item())

        loss = loss_components['rec_loss'] + self.beta * loss_components['reg_loss']

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def compute_loss_components(self, data):
        x = data['x']

        # Encode a batch of data
        q_z_given_x = self.encoder(x)

        # Sample the representation using the re-parametrization trick
        z = q_z_given_x.rsample()

        # Compute the reconstruction distribution
        p_x_given_z = self.decoder(z)

        # The reconstruction loss is the expected negative log-likelihood of the input
        #  - E[log p(X=x|Z=z)]
        rec_loss = - p_x_given_z.log_prob(x).mean()

        # The regularization loss is the KL-divergence between posterior and prior
        # KL(q(Z|X=x)||p(Z)) = E[log q(Z=z|X=x) - log p(Z=z)]
        reg_loss = (q_z_given_x.log_prob(z) - self.prior().log_prob(z)).mean()

        return {'rec_loss': rec_loss, 'reg_loss': reg_loss}

    def reconstruct(self, x, sample_latents=False):
        # If specified sample the latent distribution
        if sample_latents:
            z = self.encoder(x).sample()
        # Otherwise use the mean of the posterior
        else:
            z = self.encoder(x).mean

        # Return the mean of p(X|Z=z)
        x_rec = self.decoder(z).mean
        return x_rec
