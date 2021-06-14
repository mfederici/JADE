import os
from envyaml import EnvYAML
import yaml
import importlib

from jade.instance_manager import InstanceManager, make_instance

import torchvision.datasets as torchvision_dataset_module

def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_modules(root_dir):
    model_modules = []
    for module_path in root_dir:
        for root, subFolder, files in os.walk(module_path):
            for item in files:
                if item.endswith(".py"):
                    print(root, item)
                    file_path = os.path.join(root, item)
                    module = module_from_file(file_path.split('.')[0].replace('/', '.'), file_path)
                    model_modules.append(module)
        return model_modules


def read_and_resolve(yaml_file):
    with open(yaml_file, 'r') as f:
        keys = yaml.safe_load(f).keys()

    d = EnvYAML(yaml_file)
    return {k: d[k] for k in keys}


class RunManager:
    def __init__(self, run_id, run_name, arch_filepath, experiments_root, config, run_dir, resume, verbose=False):

        self.verbose = verbose
        self.run_name = run_name
        self.run_id = run_id
        self.config = config
        self.run_dir = run_dir
        self.resume = resume
        self.experiments_root = experiments_root
        self.arch_filepath = arch_filepath

        # TODO: accept as extra arguments
        model_paths = ['modules/models']
        dataset_paths = ['modules/data']
        eval_paths = ['modules/eval']

        self.model_modules = load_modules(model_paths)
        self.dataset_modules = load_modules(dataset_paths)
        self.eval_modules = load_modules(eval_paths)
        # os.makedirs(self.run_dir, exist_ok=True)

    @ staticmethod
    def load_config(desc):
        return {
            'model': read_and_resolve(desc['model_file']),
            'data': read_and_resolve(desc['data_file']),
            'eval': read_and_resolve(desc['eval_file'])
        }

    def resume_run(self, run_id):
        raise NotImplementedError()

    def run_exists(self, run_id):
        raise NotImplementedError()

    def load_last_model(self, trainer):
        raise NotImplemented()

    def make_instances(self):
        dataset_manager = InstanceManager(descriptions=self.config['data'],
                                         modules=[torchvision_dataset_module] + self.dataset_modules,
                                         verbose=self.verbose)

        arch_module = module_from_file(self.arch_filepath.split('.')[-2].split('/')[-1], self.arch_filepath)

        model_params = self.config['model']['params']
        model_params['writer'] = self
        model_params['datasets'] = dataset_manager
        model_params['arch_module'] = arch_module
        trainer = make_instance(class_name=self.config['model']['class'],
                                     modules=self.model_modules,
                                     verbose=self.verbose,
                                     params=model_params)


        # Load the evaluators
        evaluators = {}

        for name, desc in self.config['eval'].items():
            eval_params = desc['params']
            eval_params['datasets'] = dataset_manager
            eval_params['trainer'] = trainer
            evaluators[name] = make_instance(
                class_name=desc['class'],
                modules=self.eval_modules,
                params=eval_params
            )


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



