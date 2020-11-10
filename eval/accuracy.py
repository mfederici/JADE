from eval.base import Evaluation
from eval.utils import evaluate
import torch
import numpy as np
from torch.distributions import Distribution, Bernoulli, Categorical


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
    def __init__(self, evaluate_on, encoder='encoder', classifier='classifier', batch_size=256, sample=True, **params):
        super(AccuracyEvaluation, self).__init__(**params)
        self.dataset = self.datasets[evaluate_on]
        self.encoder = getattr(self.trainer, encoder)
        self.classifier = getattr(self.trainer, classifier)
        self.batch_size = batch_size
        self.sample = sample

    def evaluate(self):
        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False)
        device = list(self.encoder.parameters())[0].device
        correct = 0
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            for batch in data_loader:
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

        return {
            'type': 'scalar',
            'value': float(correct)/len(self.dataset)
        }


class CrossEntropyEvaluation(Evaluation):
    def __init__(self, evaluate_on, encoder='encoder', classifier='classifier', batch_size=256, sample=True, **params):
        super(CrossEntropyEvaluation, self).__init__(**params)
        self.dataset = self.datasets[evaluate_on]
        self.encoder = getattr(self.trainer, encoder)
        self.classifier = getattr(self.trainer, classifier)
        self.batch_size = batch_size
        self.sample = sample

    def evaluate(self):
        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False)
        device = list(self.encoder.parameters())[0].device
        ce = []
        self.encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            for batch in data_loader:
                y = batch['y'].to(device)
                z = self.encoder(batch['x'].to(device))

                if isinstance(z, Distribution):
                    if self.sample:
                        z = z.sample()
                    else:
                        z = z.mean

                y_pred = self.classifier(z)

                ce.append(-y_pred.log_prob(y).mean().item())

        return {
            'type': 'scalar',
            'value': np.mean(ce)
        }
