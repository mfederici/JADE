from jade.eval import Evaluation
from torch.utils.data import DataLoader
import torch


class DatasetEvaluation(Evaluation):
    def initialize(self, evaluate_on, n_samples=2048, resample=False, batch_size=256):
        self.dataset = self.datasets[evaluate_on]

        self.n_samples = n_samples
        self.batch_size = batch_size
        self.resample = resample

    def evaluate_batch(self, data):
        raise NotImplemented()

    def evaluate(self):
        # Make a data_loader for the specified dataset (names are defined in the dataset configuration file).
        data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.resample
        )

        values = {}
        evaluations = 0.
        device = self.trainer.get_device()

        self.trainer.eval()
        with torch.no_grad():
            for data in data_loader:
                for key in data:
                    data[key] = data[key].to(device)

                new_values = self.trainer.evaluate_batch(data)
                for k, v in new_values.items():
                    if k in values:
                        values[k] += v
                    else:
                        values[k] = v

                evaluations += x.shape[0]
                if evaluations >= self.n_samples:
                    break

        values = {k: v/evaluations for k, v in values.items()}
        if len(values) == 1:
            for k in values:
                value = values[k]
            return {
                'type': 'scalar',  # Type of the logged object, to be interpreted by the logger
                'value': value,
                'iteration': self.trainer.iterations  # Iteration count at the point of logging
            }
        else:
            return {
                'type': 'scalars',  # Type of the logged object, to be interpreted by the logger
                'value': values,
                'iteration': self.trainer.iterations  # Iteration count at the point of logging
            }