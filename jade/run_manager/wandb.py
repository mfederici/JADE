import wandb
import os
from jade.run_manager.base import RunManager, BACKUP_NAME, CODE_DIR, flatten_config, inflate_config
import matplotlib.pyplot as plt


class WANDBRunManager(RunManager):
    def __init__(self, config=None, run_name=None, run_id=None, verbose=False,
                 code_dir=CODE_DIR, username=None, project=None,
                 wandb_dir=None, wandb_verbose=False):
        self.verbose = verbose

        if not wandb_verbose:
            os.environ["WANDB_SILENT"] = "true"

        if config is None and run_id is None:
            raise Exception('Please specify an existing run_id or a configuration for a new run')

        if not (run_id is None) and self.verbose and not (config is None):
            print('Warning: the specified configuration will be overwritten by the one of the specified run_id')

        if 'WANDB_PROJECT' in os.environ and project is None:
            self.PROJECT = os.environ['WANDB_PROJECT']
        elif not (project is None):
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

        if 'WANDB_DIR' in os.environ and wandb_dir is None:
            wandb_dir = os.environ['WANDB_DIR']

        if verbose:
            print('Weights and Biases root directory: %s' % wandb_dir)

        # Instantiate api
        self.api = wandb.Api()
        run_exists = self.run_exists(run_id)

        # If the run exists read the configuration
        if run_exists:
            config = self.read_config(run_id)

        # Initialize wandb with the flattened configuration
        flat_config = flatten_config(config)  # wandb can't process nested dictionaries
        resume = run_exists
        wandb.init(name=run_name, project=self.PROJECT, config=flat_config, dir=wandb_dir,
                   resume=resume, id=(run_id if run_exists else None), save_code=False)

        # Use the wandb config as the new one
        flat_config = dict(wandb.config)
        config = inflate_config(flat_config)

        # And save the run_object
        self.wandb_run = wandb.run

        # If resuming a run, download the run code and set it as the code directory
        if resume:
            self.download_code(self.wandb_run.dir)
            code_dir = os.path.join(self.wandb_run.dir, CODE_DIR)

        super(WANDBRunManager, self).__init__(run_name=run_name, run_id=run_id, run_dir=self.wandb_run.dir,
                                              config=config, resume=resume, verbose=verbose,
                                              code_dir=code_dir)

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

    def download_code(self, download_dir):
        run = self.api.run('%s/%s/%s' % (self.USER, self.PROJECT, self.wandb_run.id))
        for file in run.files():
            if file.name.endswith('.py'):
                file.download(download_dir, replace=True)
                if self.verbose:
                    print('Downloading the code for %s in %s' % (file.name, download_dir))

    def read_config(self, run_id):
        run = self.api.run('%s/%s/%s' % (self.USER, self.PROJECT, run_id))
        return inflate_config(run.config)

    def download_checkpoint(self, checkpoint_file, download_path=None):
        if self.verbose:
            print("Dowloading the checkpoint: %s" % checkpoint_file)

        run = self.api.run('%s/%s/%s' % (self.USER, self.PROJECT, self.run_id))

        if download_path is None:
            download_path = os.path.join(self.wandb_run.dir)

        run.file(checkpoint_file).download(download_path, replace=True)

        return os.path.join(download_path, checkpoint_file)

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

    def make_checkpoint(self, trainer, force_upload=False):
        super(WANDBRunManager, self).make_checkpoint(trainer)
        if force_upload:
            checkpoint_filename = os.path.join(self.run_dir, 'checkpoint_%d.pt' % trainer.model.iterations)
            wandb.save(checkpoint_filename, base_path=self.run_dir)

    def make_backup(self, trainer, force_upload=False):
        super(WANDBRunManager, self).make_backup(trainer)
        if force_upload:
            model_filename = os.path.join(self.run_dir, BACKUP_NAME)
            wandb.save(model_filename, base_path=self.run_dir)

    def checkpoint_list(self):
        checkpoints = []
        run = self.api.run('%s/%s/%s' % (self.USER, self.PROJECT, self.wandb_run.id))
        for file in run.files():
            if file.name.endswith('.pt'):
                checkpoints.append(file.name)

        return checkpoints
