import os
import yaml
import wandb
from log.wandb import WandBLogWriter
from utils.instance_manager import InstanceManager, ExtendTransformInstanceManager

import torchvision.datasets as torchvision_dataset_module
import training as training_module
import data as dataset_module
import data.transforms.dataset as dataset_transform_module
import eval as eval_module


def load_descriptions_from_dir(def_dir, verbose=False):
    descriptions = dict()
    for filename in os.listdir(def_dir):
        if filename.endswith('.yml') or filename.endswith('.yaml'):
            # Load a descripton file
            with open(os.path.join(def_dir, filename), 'r') as file:
                d = yaml.safe_load(file)
                for desc in d:
                    if desc['name'] in descriptions:
                        raise Exception('Multiple definitions found for %s' % desc['name'])
                    descriptions[desc['name']] = desc
        else:
            if verbose:
                print('The file %s has been ignored' % filename)

    return descriptions


class RunManager:
    PROJECT = "causal-mib"
    USER = "mfederici"

    def __init__(self, run_name, experiments_dir, repeat, config=None, verbose=False, upload_checkpoints=False, resume=False):
        self.api = wandb.Api()
        self.runs = self.api.runs('%s/%s' % (self.USER, self.PROJECT))
        self.repeat = repeat
        self.verbose = verbose
        self.upload_checkpoints = upload_checkpoints
        self.resume = resume

        current_id = -1
        # Determine the run id
        if run_name in [run.name for run in self.runs]:
            run_counter = [int(run.id.split('_')[-1]) for run in self.runs if run.name == run_name]
            current_id = max(run_counter)

        run_id = '%s_%d' % (run_name.replace('/', '_'), current_id)
        if self.repeat or current_id < 0:
            if self.verbose:
                print('Starting a new run')
            current_id += 1
            run_id = '%s_%d' % (run_name.replace('/', '_'), current_id)
            run = None
        else:
            if self.verbose:
                print('Resuming run %s' % run_id)
            run = self._get_run(run_id)
            if run is None:
                if self.verbose:
                    print('Unable to resume the experiment %s' % run_id)
                current_id += 1
                run_id = '%s_%d' % (run_name.replace('/', '_'), current_id)
                if self.verbose:
                    print('Starting a new run')

        self.run = run

        self.experiment_dir = os.path.join(experiments_dir, run_id)
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.run_id = run_id

        if self.run is None or not resume:
            assert not (config is None)
        else:
            # Load the remote configuration
            config = self.run.config

        self.config = config

        wandb.init(project=self.PROJECT,
                   name=run_name, dir=self.experiment_dir, id=self.run_id, resume=not (self.run is None))
        self.writer = WandBLogWriter()

    def make_instances(self):
        dataset_manager = ExtendTransformInstanceManager(descriptions=self.config['data'],
                                                         modules=[torchvision_dataset_module,
                                                                  dataset_module,
                                                                  dataset_transform_module],
                                                         verbose=self.verbose)

        train_set = dataset_manager[self.config['var']['train_set']]

        # Load the trainer definition files
        trainer_manager = InstanceManager(descriptions=self.config['train'],
                                          modules=[training_module],
                                          verbose=self.verbose,
                                          dataset=train_set,
                                          writer=self.writer)

        trainer = trainer_manager[self.config['var']['model']]

        # Load the evaluation definition files
        eval_manager = InstanceManager(descriptions=self.config['eval'],
                                       modules=[eval_module],
                                       verbose=self.verbose,
                                       datasets=dataset_manager,
                                       trainer=trainer)

        evaluators = {name: eval_manager[name] for name in self.config['var']['eval']}

        # Resume the training if specified
        if not (self.run is None) and self.resume:
            # Download the last model
            if self.verbose:
                print("Dowloading the last checkpoint")
            restored_model = wandb.restore("model.pt", root=wandb.run.dir, replace=True)

            if self.verbose:
                print("Resuming Training")

            trainer.load(restored_model.name)
            if self.verbose:
                print("Resuming Training from iteration %d" % trainer.iterations)

        self.config = {'train': trainer_manager.get_config(),
                       'data': dataset_manager.get_config(),
                       'eval': eval_manager.get_config(),
                       'var': self.config['var']}

        wandb.config.update(self.config)

        return train_set, trainer, evaluators

    def log(self, **params):
        self.writer.log(**params)

    def make_checkpoint(self, trainer):
        if self.verbose:
            print('Storing model checkpoint')
        checkpoint_filename = os.path.join(wandb.run.dir, 'checkpoint_%d.pt' % trainer.iterations)
        trainer.save(checkpoint_filename)
        if self.upload_checkpoints:
            wandb.save('checkpoint_%d.pt' % trainer.iterations)

    def make_backup(self, trainer):
        if self.verbose:
            print('Updating the model backup')
        model_filename = os.path.join(wandb.run.dir, 'model.pt')
        trainer.save(model_filename)
        wandb.save('model.pt')

    def _get_run(self, run_id):
        try:
            return self.api.run('%s/%s/%s' % (self.USER, self.PROJECT, run_id))
        except Exception as err:
            if self.verbose:
                print(err)
            return None

    def _run_exists(self, run_name):
        return run_name in [run.id for run in self.runs]
