from jade.eval import Evaluation


class ReconstructionError(Evaluation):
    def initialize(self, evaluate_on, n_samples=2048, resample_x=False, batch_size=256):
        # Make a data_loader for the specified dataset (names are defined in the dataset configuration file).
        self.data_loader = DataLoader(self.datasets[evaluate_on], shuffle=resample_x, batch_size=batch_size)

        self.n_samples = n_samples

        # Check that the model has a definition of encoder and decoder
        if not hasattr(self.trainer, 'encoder') or not hasattr(self.trainer, 'decoder'):
            raise Exception('The trainer must have an encoder and decoder models')

    def evaluate(self):
        evaluations = 0.
        rec_loss = 0.
        device = trainer.get_device()

        self.trainer.eval()
        with torch.no_grad():
            for data in self.data_loader:
                x = data['x'].to(device)

                z = self.trainer.encoder(x).sample()
                rec_loss += self.trainer.decoder(z).log_prob(x).sum().item()

                evaluations += self.batch_size
                if evaluations >= self.n_samples:
                    break

        return {
            'type': 'scalar',  # Type of the logged object, to be interpreted by the logger
            'value': rec_loss/evaluations,  # Grid of images to log
            'iteration': trainer.iterations  # Iteration count at the point of logging
        }