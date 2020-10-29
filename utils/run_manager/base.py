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
    def __init__(self, run_id, run_name, experiments_root, config, run_dir, resume, num_workers=0, data_root='.',
                 verbose=False):

        self.verbose = verbose
        self.run_name = run_name
        self.run_id = run_id
        self.config = config
        self.run_dir = run_dir
        self.resume = resume
        self.num_workers = num_workers
        self.experiments_root = experiments_root
        self.data_root = data_root
        # os.makedirs(self.run_dir, exist_ok=True)

    @ staticmethod
    def load_config(desc):
        return {
            'model': load_desc_file(desc['model_file']),
            'data': load_desc_file(desc['data_file']),
            'eval': load_desc_file(desc['eval_file'])
        }

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
                class_name, str([module.__name__ for module in modules])))

        if self.verbose:
            print('Instantiating class %s from %s' %
                  (class_name, module.__name__))

        instance = Class(**params)

        return instance

    def load_last_model(self, trainer):
        raise NotImplemented()

    def make_instances(self):
        dataset_manager = DatasetManager(descriptions=self.config['data'],
                                         modules=[torchvision_dataset_module,
                                                  dataset_module,
                                                  dataset_transform_module],
                                         data_root=self.data_root)
        train_set = dataset_manager['train']
        trainer = self._make_instance(class_name=self.config['model']['class'],
                                      modules=[model_module],
                                      writer=self,
                                      dataset=train_set,
                                      num_workers=self.num_workers,
                                      **self.config['model']['params'])

        # Load the evaluators
        evaluators = {name:
            self._make_instance(
                class_name=desc['class'],
                modules=[eval_module],
                datasets=dataset_manager,
                trainer=trainer,
                **desc['params'])
            for name, desc in self.config['eval'].items()}

        # Resume the training if specified
        if self.resume:
            trainer = self.load_last_model()

        return trainer, evaluators

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



