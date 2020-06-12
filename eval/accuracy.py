from eval.base import Evaluation
from eval.utils import evaluate
import torch
from torch.distributions import Distribution


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
                if isinstance(y, Distribution):
                    y = y.logits

                correct += (torch.argmax(y, 1) == batch['y'].long()).sum().item()


        return {
            'type': 'scalar',
            'value': float(correct)/len(self.dataset)
        }


