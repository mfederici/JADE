from eval.base import Evaluation
from eval.utils import evaluate


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
