from .base import DatasetEvaluation
import torch


class AccuracyEvaluation(DatasetEvaluation):
    def initialize(self, predict_params=None, **params):
        super(AccuracyEvaluation, self).initialize(**params)

        if predict_params is None:
            predict_params = dict()
        self.predict_params = predict_params

        if not hasattr(self.model, 'predict'):
            raise Exception(
                'The trainer must implement a predict(x, **predict_params) method to use the Accuracy evaluation metric.'\
                'The predict function must return a discrete distribution object.')

    def evaluate_batch(self, data):
        x = data['x']
        y = data['y'].squeeze().long()
        y_given_x = self.model.predict(x, **self.predict_params)

        y_pred = torch.argmax(y_given_x.probs, 1).squeeze().long()

        return {'Accuracy': (y == y_pred).float().mean().item()}
