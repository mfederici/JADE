from eval.base import Evaluation
from eval.utils import evaluate
import torch
import numpy as np
from torch.distributions import Distribution, Bernoulli, Categorical
from torch.nn.functional import binary_cross_entropy_with_logits


class LinearAccuracyEvaluation(Evaluation):
    def __init__(self, train_on, test_on, model='encoder', **params):
        super(LinearAccuracyEvaluation, self).__init__(**params)
        self.train_set = self.datasets[train_on]
        self.test_set = self.datasets[test_on]
        self.model = getattr(self.trainer, model)

    def evaluate(self):
        return {
            'type': 'scalar',
            'value': evaluate(self.model, self.train_set, self.test_set, self.trainer.get_device())
        }


class AccuracyEvaluation(Evaluation):
    def __init__(self, evaluate_on, encoder='encoder', classifier='classifier', batch_size=256, n_samples=None, sample=True, **params):
        super(AccuracyEvaluation, self).__init__(**params)
        self.dataset = self.datasets[evaluate_on]
        self.encoder = getattr(self.trainer, encoder)
        self.classifier = getattr(self.trainer, classifier)
        self.batch_size = batch_size
        self.sample = sample
        self.n_samples = n_samples

    def evaluate(self):
        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False)
        device = list(self.encoder.parameters())[0].device
        correct = 0
        n_batches = 0
        total = 0

        max_batches = -1
        if not (self.n_samples is None):
            max_batches = self.n_samples//self.batch_size

        self.encoder.eval()
        self.classifier.eval()

        with torch.no_grad():
            for batch in data_loader:
                if max_batches > 0:
                    if n_batches > max_batches:
                        break
                z = self.encoder(batch['x'].to(device))

                if isinstance(z, Distribution):
                    if self.sample:
                        z = z.sample()
                    else:
                        z = z.mean

                y = self.classifier(z)
                if isinstance(y, Categorical):
                    y = y.logits
                    y = torch.argmax(y, 1)
                elif isinstance(y, Bernoulli):
                    y = (y.probs >= 0.5).long()
                else:
                    raise NotImplemented()

                correct += (y == batch['y'].to(device).long()).sum().item()
                total += batch['y'].shape[0]
                n_batches += 1

        return {
            'type': 'scalar',
            'value': float(correct)/total
        }


class CrossEntropyEvaluation(Evaluation):
    def __init__(self, evaluate_on, encoder='encoder', classifier='classifier', batch_size=256, n_samples=None, sample=True, **params):
        super(CrossEntropyEvaluation, self).__init__(**params)
        self.dataset = self.datasets[evaluate_on]
        self.encoder = getattr(self.trainer, encoder)
        self.classifier = getattr(self.trainer, classifier)
        self.batch_size = batch_size
        self.sample = sample
        self.n_samples = n_samples

    def evaluate(self):
        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True)
        device = list(self.encoder.parameters())[0].device
        ce = []

        max_batches = -1
        if not (self.n_samples is None):
            max_batches = self.n_samples//self.batch_size

        self.encoder.eval()
        self.classifier.eval()
        n_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                if max_batches > 0:
                    if n_batches > max_batches:
                        break
                y = batch['y'].to(device)
                z = self.encoder(batch['x'].to(device))

                if isinstance(z, Distribution):
                    if self.sample:
                        z = z.sample()
                    else:
                        z = z.mean

                y_pred = self.classifier(z)

                if isinstance(y_pred, Bernoulli):
                    c_ = binary_cross_entropy_with_logits(y_pred.logits.squeeze(), y.float(), reduction='none')
                else:
                    c_ = -y_pred.log_prob(y)

                ce.append(c_.mean().item())
                n_batches += 1

        return {
            'type': 'scalar',
            'value': np.mean(ce)
        }
