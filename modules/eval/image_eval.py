from jade.eval import Evaluation
from torchvision.utils import make_grid
import torch


class ReconstructionEvaluation(Evaluation):
    def initialize(self, evaluate_on, n_pictures=10, sample_images=False, sample_latents=False):
        # Consider the dataset labeled with the specified name (names are defined in the dataset configuration file).
        self.dataset = self.datasets[evaluate_on]

        self.n_pictures = n_pictures
        self.sample_images = sample_images
        self.sample_latents = sample_latents

        # Check that the model has a definition of a method to reconstrut the inputs
        if not hasattr(self.trainer, 'reconstruct'):
            raise Exception('The trainer must implement a reconstruct(x) method with `x` as a picture')

    def sample_new_images(self):
        # sample the required number of pictures randomly
        ids = np.random.choice(len(self.dataset), self.n_pictures)
        images_batch = torch.cat([dataset[id]['x'].unsqueeze(0) for id in ids])

        return images_batch

    def evaluate(self):
        # If the images are not sampled dynamically, pick the first n_pictures from the dataset
        if not self.sample_images:
            x = torch.cat([self.dataset[id]['x'].unsqueeze(0) for id in range(self.n_pictures)])
        # Otherwise pick random ones
        else:
            ids = np.random.choice(len(self.dataset), self.n_pictures)
            x = torch.cat([self.dataset[id]['x'].unsqueeze(0) for id in ids])

        # Move the images to the correct device
        x = x.to(self.trainer.get_device())

        # Compute the reconstructions
        x_rec = self.trainer.reconstruct(x).to('cpu')

        # Concatenate originals and reconstructions
        x_all = torch.cat([x, x_rec], 2)

        # Return a dictionary used for logging
        return {
            'type': 'figure',  # Type of the logged object, to be interpreted by the logger
            'value': make_grid(x_all, nrow=self.n_pictures),  # Value to log
            'iteration': self.trainer.iterations  # Iteration count at the point of logging
        }