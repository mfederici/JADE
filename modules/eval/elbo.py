from modules.eval.base import DatasetEvaluation
from torch.utils.data import DataLoader
import torch


class ErrorComponentsEvaluation(DatasetEvaluation):
    def initialize(self, **params):
        super(ErrorComponentsEvaluation, self).initialize(**params)

        # Check that the model has a definition of encoder and decoder
        if not hasattr(self.trainer, 'compute_loss_components'):
            raise Exception('The trainer must implement a compute_loss_components(data) method to use this evaluator.')

    def evaluate_batch(self, data):
        return self.trainer.compute_loss_components(data)


class ELBOEvaluation(ErrorComponentsEvaluation):
    def evaluate_batch(self, data):
        loss_components = self.trainer.compute_loss_components(data)
        return {'ELBO': -(loss_components['rec_loss'] + loss_components['reg_loss'])}
