from torch.utils.data import DataLoader
import torch

from jade.utils import TimeInterval


# TODO: add evaluate every in epochs
class Evaluation:
    def __init__(self, datasets, evaluate_every='1 epochs', verbose=False, **params):
        self.datasets = datasets
        self.verbose = verbose

        self.evaluation_timer = TimeInterval(evaluate_every)
        self.initialize(**params)

    def evaluate_if_time(self, model):
        if self.evaluation_timer.is_time(model):
            log = self.evaluate(model)
            self.evaluation_timer.update(model)
        else:
            log = None
        return log

    def initialize(self, **params):
        raise NotImplemented()

    def evaluate(self, model):
        raise NotImplemented()


class DatasetEvaluation(Evaluation):
    def initialize(self, evaluate_on, n_samples=2048, resample=False, batch_size=256):
        self.dataset = self.datasets[evaluate_on]

        self.n_samples = n_samples
        self.batch_size = batch_size
        self.resample = resample

    def evaluate_batch(self, data, model):
        raise NotImplemented()

    def evaluate(self, model):
        # Make a data_loader for the specified dataset (names are defined in the dataset configuration file).
        data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.resample
        )

        values = {}
        evaluations = 0.
        device = model.get_device()

        model.eval()
        with torch.no_grad():
            for data in data_loader:
                if isinstance(data, dict):
                    for key in data:
                        data[key] = data[key].to(device)
                        if hasattr(data[key], 'shape'):
                            batch_len = data[key].shape[0]
                elif isinstance(data, list) or isinstance(data, tuple):
                    data_ = []
                    for d in data:
                        data_.append(d.to(device))
                        if hasattr(d, 'shape'):
                            batch_len = d.shape[0]
                    if isinstance(data, tuple):
                        data = tuple(data_)
                    else:
                        data = data_

                new_values = self.evaluate_batch(data, model)
                for k, v in new_values.items():
                    if k in values:
                        values[k] += v * batch_len
                    else:
                        values[k] = v * batch_len

                evaluations += batch_len
                if evaluations >= self.n_samples:
                    break

        values = {k: v/evaluations for k, v in values.items()}
        if len(values) == 1:
            for k in values:
                value = values[k]
            return {
                'type': 'scalar',  # Type of the logged object, to be interpreted by the logger
                'value': value
            }
        else:
            return {
                'type': 'scalars',  # Type of the logged object, to be interpreted by the logger
                'value': values
            }