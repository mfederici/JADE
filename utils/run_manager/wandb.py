import wandb
import os
from utils.run_manager.base import RunManager

SPLIT_TOKEN = '.'


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
    def __init__(self, desc, experiments_root, run_name=None, run_id=None, verbose=False, upload_checkpoints=True, **params):
        if 'WANDB_PROJECT' in os.environ:
            self.PROJECT = os.environ['WANDB_PROJECT']
        else:
            raise Exception(
                'In order to use the wandb framework the environment variable WANDB_PROJECT needs to be set')

        if 'WANDB_USER' in os.environ:
            self.USER = os.environ['WANDB_USER']
        else:
            raise Exception('In order to use the wandb framework the environment variable WANDB_USER needs to be set')

        self.api = wandb.Api()
        self.upload_checkpoints = upload_checkpoints

        resume = self.run_exists(run_id)
        if resume:
            config = self.resume_run(run_id)
        else:
            config = self.load_config(desc)

        flat_config = flatten_config(config) # wandb can't process nested dictionaries

        wandb.init(name=run_name, project=self.PROJECT, config=flat_config, dir=experiments_root,
                   resume=resume, id=(run_id if resume else None))

        flat_config = dict(wandb.config)
        config = inflate_config(flat_config)

        run_id = wandb.run.id
        run_dir = wandb.run.dir

        super(WANDBRunManager, self).__init__(run_name=run_name, run_id=run_id, run_dir=run_dir,
                                              config=config, resume=resume, verbose=verbose, **params)

    def run_exists(self, run_id):
        return run_id in [run.id for run in self.api.runs('%s/%s' % (self.USER, self.PROJECT))]

    def resume_run(self, run_id):
        run = self.api.run('%s/%s/%s' % (self.USER, self.PROJECT, run_id))
        if self.verbose:
            print('Warning: the specified configuration will be ingnored since the run is being resumed')
        return run.config

    def load_last_model(self, trainer):
        # Download the last model
        if self.verbose:
            print("Dowloading the last checkpoint")
        restored_model = wandb.restore("model.pt", root=wandb.run.dir, replace=True)

        if self.verbose:
            print("Resuming Training")

        trainer.load(restored_model.name)
        if self.verbose:
            print("Resuming Training from iteration %d" % trainer.iterations)

        return trainer

    def make_instances(self):
        trainer, evaluators = super(WANDBRunManager, self).make_instances()
        # wandb.watch(trainer)
        return trainer, evaluators

    def log(self, name, value, entry_type, iteration):
        if entry_type == 'scalar':
            wandb.log({name: value}, step=iteration)
        else:
            raise Exception('Type %s is not recognized by WandBLogWriter' % type)

    def make_checkpoint(self, trainer):
        super(WANDBRunManager, self).make_checkpoint(trainer)
        if self.upload_checkpoints:
            wandb.save('checkpoint_%d.pt' % trainer.iterations)

    def make_backup(self, trainer):
        super(WANDBRunManager, self).make_backup(trainer)
        if self.upload_checkpoints:
            wandb.save('checkpoint_%d.pt' % trainer.iterations)