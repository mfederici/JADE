import os
import yaml

from utils.instance_manager import DatasetManager

import torchvision.datasets as torchvision_dataset_module
import models as model_module
import data as dataset_module
import data.transforms.dataset as dataset_transform_module
import eval as eval_module


def load_desc_file(desc_filename):
    with open(desc_filename, 'r') as file:
        d = yaml.safe_load(file)
    return d


class RunManager:
    def __init__(self, run_id, run_name, experiments_dir, desc=None, verbose=False):

        self.verbose = verbose
        self.run_name = run_name

        if not self.run_exists(run_id):
            self.model_desc = load_desc_file(desc['model_file'])
            self.data_desc = load_desc_file(desc['data_file'])
            self.eval_desc = load_desc_file(desc['eval_file'])

            self.config = {
                'model': self.model_desc,
                'data': self.data_desc,
                'eval': self.eval_desc
            }
            self.resume = False

        else:
            self.resume = True
            self.config = self.resume_run(run_id)

        self.experiment_dir = os.path.join(experiments_dir, run_id)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.run_id = run_id

    def resume_run(self, run_id):
        raise NotImplementedError()

    def run_exists(self, run_id):
        raise NotImplementedError()

    def _make_instance(self, class_name, modules, **params):
        class_found = False

        for module in modules:
            if hasattr(module, class_name):
                Class = getattr(module, class_name)
                class_found = True
                break

        if not class_found:
            raise Exception('No description for %s has been found in %s' % (
                class_name, str([module.__name__ for module in self.modules])))

        if self.verbose:
            print('Instantiating class %s from %s' %
                  (class_name, module.__name__))

        instance = Class(**params)

        return instance

    def load_last_model(self, trainer):
        raise NotImplemented()

    def make_instances(self):
        dataset_manager = DatasetManager(descriptions=self.data_desc,
                                         modules=[torchvision_dataset_module,
                                                  dataset_module,
                                                  dataset_transform_module])
        train_set = dataset_manager['train']
        trainer = self._make_instance(class_name=self.model_desc['class'],
                                      modules=[model_module],
                                      writer=self,
                                      dataset=train_set,
                                      **self.model_desc['params'])

        # Load the evaluators
        evaluators = {name:
            self._make_instance(
                class_name=desc['class'],
                modules=[eval_module],
                datasets=dataset_manager,
                trainer=trainer,
                **desc['params'])
            for name, desc in self.eval_desc.items()}

        # Resume the training if specified
        if self.resume:
            trainer = self.load_last_model()

        return train_set, trainer, evaluators

    def log(self, name, value, entry_type, iteration):
        raise NotImplementedError()

    def make_checkpoint(self, trainer):
        if self.verbose:
            print('Storing model checkpoint')
        checkpoint_filename = os.path.join(self.run_dir, 'checkpoint_%d.pt' % trainer.iterations)
        trainer.save(checkpoint_filename)

    def make_backup(self, trainer):
        if self.verbose:
            print('Updating the model backup')
        model_filename = os.path.join(self.run_dir, 'model.pt')
        trainer.save(model_filename)



