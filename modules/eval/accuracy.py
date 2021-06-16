from modules.eval.base import DatasetEvaluation
import torch


class AccuracyEvaluation(DatasetEvaluation):
    def initialize(self, predict_params=None, **params):
        super(AccuracyEvaluation, self).__init__(**params)

        self.predict_params = predict_params

        if not hasattr(self.trainer, 'predict'):
            raise Exception(
                'The trainer must implement a predict(x, **predict_params) method to use the Accuracy evaluation metric.'\
                'The predict function must return a discrete distribution object.')

    def evaluate_batch(self, data):
        x = data['x']
        y = data['y'].squeeze().long()
        y_given_x = trainer.predict(x, **self.predict_params)

        y_pred = torch.argmax(y_given_x.p, 1).squeeze().long()

        return {'Accuracy': (y == y_pred).sum().item()}
