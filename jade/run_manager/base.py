import os
from envyaml import EnvYAML
import importlib

from jade.instance_manager import InstanceManager, make_instance

import torchvision.datasets as torchvision_dataset_module
from examples.modules import models as model_module, eval as eval_module
import data as dataset_module
import data.transforms.dataset as dataset_transform_module


class RunManager:
    def __init__(self, run_id, run_name, arch_filepath, experiments_root, config, run_dir, resume, verbose=False):

        self.verbose = verbose
        self.run_name = run_name
        self.run_id = run_id
        self.config = config
        self.run_dir = run_dir
        self.resume = resume
        self.experiments_root = experiments_root
        self.data_root = data_root
        self.arch_filepath = arch_filepath
        # os.makedirs(self.run_dir, exist_ok=True)

    @ staticmethod
    def load_config(desc):
        return {
            'model': EnvYAML(desc['model_file']),
            'data': EnvYAML(desc['data_file']),
            'eval': EnvYAML(desc['eval_file'])
        }

    def resume_run(self, run_id):
        raise NotImplementedError()

    def run_exists(self, run_id):
        raise NotImplementedError()

    def load_last_model(self, trainer):
        raise NotImplemented()

    def make_instances(self):
        dataset_manager = InstanceManager(descriptions=self.config['data'],
                                         modules=[torchvision_dataset_module,
                                                  dataset_module],
                                         data_root=self.data_root,
                                         verbose=self.verbose)

        arch_spec = importlib.util.spec_from_file_location(self.arch_filepath.split('.')[-2].split('/')[-1], self.arch_filepath)
        arch_module = importlib.util.module_from_spec(arch_spec)
        arch_spec.loader.exec_module(arch_module)

        trainer_params = self.config['model']['params']
        trainer_params['writer'] = self
        trainer_params['datasets'] = dataset_manager
        trainer_params['arch_module'] = arch_module
        trainer_params['verbose'] = self.verbose

        trainer = InstanceManager(class_name=self.config['model']['class'],
                                modules=[model_module],
                                verbose=self.verbose,
                                params=trainer_params)

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
            trainer = self.load_last_model(trainer)

        return trainer, evaluators

    def log(self, name, value, type, iteration):
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



