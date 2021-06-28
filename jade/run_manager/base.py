import os
import importlib
import sys

import torch
import numpy as np

from jade.instance_manager import InstanceManager, make_instance
import jade.trainer as base_trainer_module

import torchvision.datasets as torchvision_dataset_module

import yaml
from envyaml import EnvYAML

import json

BACKUP_NAME = 'last_checkpoint.pt'
DATASET_DIR = DATASET_KEY ='datasets'
MODELS_DIR = 'models'
MODEL_KEY = 'model'
ARCH_DIR = ARCH_KEY = 'architectures'
EVAL_DIR = EVAL_KEY = 'evaluation'
TRAINER_DIR = 'trainers'
TRAINER_KEY = 'trainer'

def module_from_file(path):
    name = path.replace('/', '.')
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
                    module = module_from_file(file_path)
                    model_modules.append(module)
        return model_modules


def resolve_variables(config,  experiments_root):
    config_file = os.path.join(experiments_root, "config.yml")
    with open(config_file, "w") as file:
        yaml.dump(config, file)

    d = EnvYAML(yaml_file=config_file)
    return {k: d[k] for k in config}


class RunManager:
    def __init__(self, run_id, run_name, experiments_root, config, run_dir, resume, verbose=False, code_dir='example',
                 arch_filename=None):

        # Resolve config
        config = resolve_variables(config, run_dir)
        print(json.dumps(config, sort_keys=True, indent=4))


        self.verbose = verbose
        self.run_name = run_name
        self.run_id = run_id
        self.config = config
        self.run_dir = run_dir
        if verbose:
            print('Run Directory: %s' % run_dir)

        self.resume = resume
        self.experiments_root = experiments_root

        sys.path.append(code_dir)

        # TODO: accept as extra arguments
        model_paths = [os.path.join(code_dir, MODELS_DIR)]
        dataset_paths = [os.path.join(code_dir, DATASET_DIR)]
        eval_paths = [os.path.join(code_dir, EVAL_DIR)]
        if arch_filename is None:
            arch_paths = [os.path.join(code_dir, ARCH_DIR)]
            self.arch_modules = load_modules(arch_paths)

        else:
            self.arch_modules = [module_from_file(os.path.join(code_dir, ARCH_DIR, arch_filename))]
        trainer_paths = [os.path.join(code_dir, TRAINER_DIR)]

        self.model_modules = load_modules(model_paths)
        self.dataset_modules = load_modules(dataset_paths)
        self.eval_modules = load_modules(eval_paths)
        self.trainers_modules = load_modules(trainer_paths)+[base_trainer_module]
        # os.makedirs(self.run_dir, exist_ok=True)

        # Set random seed
        if 'seed' in config:
            seed = config['seed']
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(seed)
            np.random.seed(seed)

    def load_checkpoint(self, trainer, checkpoint_file):
        raise NotImplementedError()

    def load_model(self, model, checkpoint_file):
        raise NotImplementedError()

    def resume_run(self, run_id):
        raise NotImplementedError()

    def run_exists(self, run_id):
        raise NotImplementedError()
    
    def load_last_trainer(self, trainer, device='cpu'):
        raise NotImplementedError()

    def load_last_model(self, model, device='cpu'):
        raise NotImplementedError()

    def instantiate_datasets(self):
        dataset_manager = InstanceManager(descriptions=self.config[DATASET_KEY],
                                          modules=[torchvision_dataset_module] + self.dataset_modules,
                                          verbose=self.verbose)
        return dataset_manager

    def instantiate_model(self, resume=False, device='cpu'):
        model_params = self.config[MODEL_KEY]['params']
        model_params['writer'] = self
        model_params['arch_modules'] = self.arch_modules
        model = make_instance(class_name=self.config[MODEL_KEY]['class'],
                              modules=self.model_modules,
                              verbose=self.verbose,
                              params=model_params)

        # Resume the training if specified
        if resume:
            model = self.load_last_model(model, device=device)
        else:
            model.to(device)
        return model

    def instantiate_trainer(self, model, datasets, resume=False, device='cpu'):
        trainer_params = self.config[TRAINER_KEY]['params']
        trainer_params['writer'] = self
        trainer_params['model'] = model
        trainer_params['datasets'] = datasets
        trainer = make_instance(class_name=self.config[TRAINER_KEY]['class'],
                                modules=self.trainers_modules,
                                verbose=self.verbose,
                                params=trainer_params)

        # Resume the training if specified
        if resume:
            trainer = self.load_last_trainer(trainer, device=device)
        else:
            trainer.to(device)

        return trainer

    def instantiate_evaluators(self, model, dataset_manager):
        # Load the evaluators
        evaluators = {}

        for name, desc in self.config[EVAL_KEY].items():
            eval_params = desc['params']
            eval_params['datasets'] = dataset_manager
            eval_params['model'] = model
            evaluators[name] = make_instance(
                class_name=desc['class'],
                modules=self.eval_modules,
                params=eval_params
            )
        return evaluators

    def make_instances(self, device='cpu'):
        dataset_manager = self.instantiate_datasets()
        model = self.instantiate_model(device=device)
        trainer = self.instantiate_trainer(model=model, datasets=dataset_manager, resume=self.resume, device=device)
        evaluators = self.instantiate_evaluators(model=model, dataset_manager=dataset_manager)

        return trainer, evaluators

    def log(self, name, value, type, iteration):
        raise NotImplementedError()

    def make_checkpoint(self, trainer):
        if self.verbose:
            print('Storing model checkpoint')
        checkpoint_filename = os.path.join(self.run_dir, 'checkpoint_%d.pt' % trainer.model.iterations)
        trainer.save(checkpoint_filename)

    def make_backup(self, trainer):
        if self.verbose:
            print('Updating the model backup')
        model_filename = os.path.join(self.run_dir, BACKUP_NAME)
        trainer.save(model_filename)



