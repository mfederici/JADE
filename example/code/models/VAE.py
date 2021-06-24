from jade.model import Model
from torch.optim import Adam
from torch.utils.data import DataLoader


###########################
# Variational Autoencoder #
###########################


class VariationalAutoencoder(Model):
    def initialize(self, z_dim, encoder_layers, decoder_layers, beta, sigma=1):
        # The value of the z_dim, n_encoder_layers, n_decoder_layers, beta lr, batch_size and n_workers are defined
        # in the configuration file or specified as additional arguments when running the train python file

        # Store the value of the regularization strength beta for the loss computation
        self.beta = beta

        # initialize the encoder "Encoder(z_dim, layers)" defined in the architecture python file
        self.encoder = self.instantiate_architecture(
            class_name='Encoder',                       # Name of the class to instantiate (from the architectures file)
            z_dim=z_dim, layers=encoder_layers          # Extra parameters passed to the constructor
        )

        # initialize the encoder "Decoder(z_dim, layers, sigma)" defined in the architecture python file
        self.decoder = self.instantiate_architecture(
            class_name='Decoder',
            z_dim=z_dim, layers=decoder_layers, sigma=sigma
        )

        # Initialize the Gaussian prior passing the number of dimensions
        self.prior = self.instantiate_architecture(
            class_name='Prior',
            z_dim=z_dim
        )

        # Optimize encoder and decoder using the optimizer named 'opt', which must be defined
        # in the training configuration file
        self.optimize(attribute_name='encoder', optimizer_name='default')
        self.optimize(attribute_name='decoder', optimizer_name='default')

        # Attributes that need to be saved (and reloaded), but not optimized
        # can be added using self.add_attribute_to_store(attribute_name)

    def compute_loss(self, data):
        loss_components = self.compute_loss_components(data)

        loss = loss_components['rec_loss'] + self.beta * loss_components['reg_loss']

        # Add the two loss components to the log
        # Note that only scalars are currently supported although custom logging can be implemented
        # by extending the implemented methods
        self.add_loss_item('TrainLog/ReconstructionLoss', loss_components['rec_loss'].item())
        self.add_loss_item('TrainLog/RegularizationLoss', loss_components['reg_loss'].item())
        self.add_loss_item('TrainLog/Loss', loss.item())

        return loss

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
