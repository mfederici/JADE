from jade.eval import Evaluation
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.distributions import Distribution, Bernoulli, Categorical
from torch.nn.functional import binary_cross_entropy_with_logits


class AccuracyEvaluation(Evaluation):
    def initialize(self, evaluate_on, encoder='encoder', classifier='classifier', batch_size=256, n_samples=None, sample=True, num_workers=0, **params):
        super(AccuracyEvaluation, self).__init__(**params)
        self.dataset = self.datasets[evaluate_on]
        self.encoder = getattr(self.trainer, encoder)
        self.classifier = getattr(self.trainer, classifier)
        self.batch_size = batch_size
        self.sample = sample
        self.n_samples = n_samples
        self.num_workers = num_workers

    def evaluate(self):
        data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers)
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

                correct += (y.squeeze() == batch['y'].to(device).long().squeeze()).sum().item()
                total += batch['y'].shape[0]
                n_batches += 1

        return {
            'type': 'scalar',
            'value': float(correct)/total,
            'iteration': self.trainer.iterations
        }


class CrossEntropyEvaluation(Evaluation):
    def __init__(self, evaluate_on, encoder='encoder', classifier='label_classifier', batch_size=256, num_workers=8, n_samples=None, sample=True, **params):
        super(CrossEntropyEvaluation, self).__init__(**params)
        self.dataset = self.datasets[evaluate_on]
        self.encoder = getattr(self.trainer, encoder)
        self.classifier = getattr(self.trainer, classifier)
        self.batch_size = batch_size
        self.sample = sample
        self.n_samples = n_samples
        self.num_workers = num_workers

    def evaluate(self):
        data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)

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
                y = batch['y'].to(device).squeeze()
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
            'value': np.mean(ce),
            'iteration': self.trainer.iterations
        }
