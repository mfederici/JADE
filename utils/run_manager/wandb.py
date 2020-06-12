import wandb
import os
from utils.run_manager.base import RunManager


class WANDBRunManager(RunManager):
    def __init__(self, upload_checkpoints=False, **params):
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
        run_id = wandb.util.generate_id()

        super(WANDBRunManager, self).__init__(run_id=run_id, **params)

        wandb.init(project=self.PROJECT, name=self.run_name, config=self.config, dir=self.experiment_dir,
                   id=self.run_id, resume=not (self.resume is None))

        self.config = wandb.config
        self.run_dir = wandb.run.dir

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
        train_set, trainer, evaluators = super(WANDBRunManager, self).make_instances()
        # wandb.watch(trainer)
        return train_set, trainer, evaluators

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