import wandb
import os
from jade.run_manager.base import RunManager, BACKUP_NAME
import matplotlib.pyplot as plt

from shutil import copytree, rmtree
import torch

SPLIT_TOKEN = '.'
DEFAULT_CODE_ROOT_NAME = 'modules'
CODE_DIR = 'code'

# utilities to flatten and re-inflate the configuration for wandb
def _flatten_config(config, prefix, flat_config):
    for key, value in config.items():
        flat_key = SPLIT_TOKEN.join([prefix, key] if prefix else [key])
        if isinstance(value, dict):
            _flatten_config(value, flat_key, flat_config)
        else:
            flat_config[flat_key] = value


def flatten_config(config):
    flat_config = {}
    _flatten_config(config, None, flat_config)
    return flat_config


def inflate_config(flat_config):
    config = {}
    for key, value in flat_config.items():
        sub_config = config
        keys = key.split(SPLIT_TOKEN)
        for sub_key in keys[:-1]:
            if not (sub_key in sub_config):
                sub_config[sub_key] = dict()
            sub_config = sub_config[sub_key]
        sub_config[keys[-1]] = value
    return config


class WANDBRunManager(RunManager):
    def __init__(self, config=None, experiments_root='.', run_name=None, run_id=None, run_dir=None, verbose=False,
                 upload_checkpoints=True, init=True,  code_dir=DEFAULT_CODE_ROOT_NAME, username=None, project=None, **params):
        self.verbose = verbose

        if not (run_id is None) and self.verbose and not (config is None) :
                print('Warning: the specified configuration will be overrided by the one of the specified run_id')


        if 'WANDB_PROJECT' in os.environ and project is None:
            self.PROJECT = os.environ['WANDB_PROJECT']
        elif not(project is None):
            self.PROJECT = project
        else:
            raise Exception(
                'In order to use the wandb framework the environment variable WANDB_PROJECT needs to be set')

        if 'WANDB_USER' in os.environ and username is None:
            self.USER = os.environ['WANDB_USER']
        elif not (username is None):
            self.USER = username
        else:
            raise Exception('In order to use the wandb framework the environment variable WANDB_USER needs to be set')

        self.api = wandb.Api()
        self.upload_checkpoints = upload_checkpoints

        run_exists = self.run_exists(run_id)

        if run_exists:
            config = self.read_config(run_id)
            self.download_code(os.path.join(experiments_root), run_id)
            code_dir = os.path.join(experiments_root, run_id, CODE_DIR)

        flat_config = flatten_config(config) # wandb can't process nested dictionaries
        resume = run_exists

        if init:
            wandb.init(name=run_name, project=self.PROJECT, config=flat_config, dir=experiments_root,
                       resume=resume, id=(run_id if run_exists else None), save_code=False)

            flat_config = dict(wandb.config)
            config = inflate_config(flat_config)

            run_id = wandb.run.id
            run_dir = wandb.run.dir

            if not os.path.isdir(code_dir):
                raise Exception('The specified code directory "%s" does not exist' % code_dir)

            # Upload the code
            if not resume:
                for path, subdirs, files in os.walk(code_dir):
                    for name in files:
                        if name.endswith('.py'):
                            if self.verbose:
                                print('Storing %s' % os.path.join(path, name))
                                wandb.save(os.path.join(path, name), base_path=code_dir)


        super(WANDBRunManager, self).__init__(run_name=run_name, run_id=run_id, run_dir=run_dir,
                                              config=config, resume=resume, verbose=verbose,
                                              experiments_root=experiments_root,
                                              code_dir=code_dir, **params)

    def run_exists(self, run_id):
        if run_id is None:
            success = False
        else:
            try:
                run = self.api.run('%s/%s/%s' % (self.USER, self.PROJECT, run_id))
                success = True
                if self.verbose:
                    print('Run %s/%s/%s has been found' % (self.USER, self.PROJECT, run_id))
            except Exception as e:
                print(e)
                success = False
        return success

    def download_code(self, path, run_id):
        run = self.api.run('%s/%s/%s' % (self.USER, self.PROJECT, run_id))

        for file in run.files():
            if file.name.endswith('.py'):
                file.download(os.path.join(path, CODE_DIR), replace=True)
                if self.verbose:
                    print('Downloading the code for %s' % file.name)

    def read_config(self, run_id):
        run = self.api.run('%s/%s/%s' % (self.USER, self.PROJECT, run_id))
        return inflate_config(run.config)

    def download_checkpoint(self, checkpoint_file):
        # Download the last model
        if self.verbose:
            print("Dowloading the last checkpoint: %s" % checkpoint_file)
        file_path = os.path.join(self.experiments_root, self.run_id)
        os.makedirs(file_path, exist_ok=True)

        run = self.api.run('%s/%s/%s' % (self.USER, self.PROJECT, self.run_id))
        run.file(checkpoint_file).download(file_path, replace=True)

        return os.path.join(file_path, checkpoint_file)

    def load_checkpoint(self, trainer, checkpoint_file, device='cpu'):
        file_path = self.download_checkpoint(checkpoint_file)

        if self.verbose:
            print("Resuming Training")

        trainer.load(file_path, device=device)
        if self.verbose:
            print("Resuming Training from iteration %d" % trainer.model.iterations)

        return trainer

    def load_model(self, model, checkpoint_file, device='cpu'):
        file_path = self.download_checkpoint(checkpoint_file)

        if self.verbose:
            print("Resuming Training")

        model.load(file_path, device=device)
        if self.verbose:
            print("Resuming Training from iteration %d" % model.iterations)

        return model

    def load_last_trainer(self, trainer, device='cpu'):
        return self.load_checkpoint(trainer, BACKUP_NAME, device=device)

    def load_last_model(self, model, device='cpu'):
        return self.load_model(model, BACKUP_NAME, device=device)

    def log(self, name, value, type, iteration):
        if type == 'scalar':
            wandb.log({name: value}, step=iteration)
        elif type == 'scalars':
            for sub_name, v in value.items():
                wandb.log({'%s/%s' % (name, sub_name): v}, step=iteration)
        elif type == 'figure':
            wandb.log({name: wandb.Image(value)}, step=iteration)
            plt.close(value)
        else:
            raise Exception('Type %s is not recognized by WandBLogWriter' % type)

    def make_checkpoint(self, trainer):
        super(WANDBRunManager, self).make_checkpoint(trainer)
        if self.upload_checkpoints:
            wandb.save('checkpoint_%d.pt' % trainer.model.iterations)

    def make_backup(self, trainer):
        super(WANDBRunManager, self).make_backup(trainer)
        if self.upload_checkpoints:
            wandb.save(BACKUP_NAME)