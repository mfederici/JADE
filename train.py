import os
import argparse
from tqdm import tqdm
from envyaml import EnvYAML
import yaml
from dotenv import dotenv_values

from jade.run_manager.wandb import WANDBRunManager

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="", nargs='+',
                    help='List of configuration .yml files containing the description of "model", "data", "trainer" '\
                         'and "evaluators".')
parser.add_argument("--run_name", type=str, default=None,
                    help="Unique name associated to the experiment")
parser.add_argument("--code-dir", type=str, default='modules',
                    help='path to the folder containing the code (it must contain the "architectures", "data", "eval"'
                         ', "models" (and "trainers") sub-directories.')
parser.add_argument("--experiments-root", type=str, default="experiments",
                    help="Root of the experiment directory. Checkpoints will be stored in sub-directories corresponding"
                         " to their respective run id.")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device on which the experiment is executed (as for tensor.device). Specify 'cpu' to "
                         "force execution on CPU.")
parser.add_argument("--run_id", type=str, default=None, help='Wandb run id for resuming experiments.')
# Change checkpoint frequency
parser.add_argument("--checkpoint-every", type=int, default=500, help="Frequency of model checkpointing (in epochs).")
parser.add_argument("--backup-every", type=int, default=100, help="Frequency of model backups (in epochs).")
parser.add_argument("--epochs", type=int, default=100, help="Total number of training epochs")
parser.add_argument("--overwrite", type=str, default="", nargs='+',
                    help='Parameters to of configuration files to overwrite.'
                    'use the dot "." operator to accessed nested arguments (e.g. --override model.z_dim=0.001 )')
parser.add_argument('--env-file', type=str, default='.env',
                    help='File containing the definition of the environment variables')

args = parser.parse_args()
code_dir = args.code_dir

run_name = args.run_name
run_id = args.run_id

checkpoint_every = args.checkpoint_every
backup_every = args.backup_every
epochs = args.epochs

experiments_root = args.experiments_root

overwrite = args.overwrite
env_file = args.env_file

env = dotenv_values(env_file)
for k, v in env.items():
    os.environ[k] = v

device = args.device
if 'DEVICE' in os.environ:
    device = os.environ['DEVICE']

if 'EXPERIMENTS_ROOT' in os.environ:
    experiments_root = os.environ['EXPERIMENTS_ROOT']

upload_checkpoints = True
verbose = True

if len(args.config) > 0:
    config = {}
    for filename in args.config:
        with open(filename, 'r') as f:
            d = yaml.safe_load(f)
        for k in d:
            if '.' in d:
                raise Exception('The special character \'.\' can not be used in the definition of %s' % k)
            if k in config:
                print('Warning: Duplicate entry for %s, the value %s is overwritten by %s' % (k, str(config[k]), (d[k])))
            else:
                config[k] = d[k]

    # parse and update the configuration from the override argument
    print('overwrite: %s' % str(overwrite))
    for entry in overwrite:
        key, value = entry.split('=')[0], entry.split('=')[1]
        value = yaml.safe_load(value)

        d = config
        last_d = d
        for sub_key in key.split('.'):
            if not (sub_key in d):
                raise Exception(
                    'The parameter %s in %s specified in the --overwrite flag is not defined in the configuration.\n'
                    'The accessible keys are %s' %
                    (sub_key, key, str(d.keys()))
                )
            last_d = d
            d = d[sub_key]

        last_d[key.split('.')[-1]] = value
        if verbose:
            print('%s\n\tOriginal value: %s\n\t Overwritten value: %s' % (key, str(d), str(value)))

else:
    config = None
    if len(overwrite) > 0:
        raise Exception(
            'The argument --overwrite can be used only when a configuration file is specified (with --config)'
        )

run_manager = WANDBRunManager(run_name=run_name, config=config,
                              experiments_root=experiments_root,
                              code_dir=code_dir,
                              verbose=verbose, upload_checkpoints=upload_checkpoints,
                              run_id=run_id)

experiment_dir = run_manager.run_dir

trainer, evaluators = run_manager.make_instances()

# Moving the models to the specified device
trainer.to(device)

# Training loop
for epoch in tqdm(range(epochs)):
    # Evaluation
    for name, evaluator in evaluators.items():
        if epoch % evaluator.evaluate_every == 0:
            entry = evaluator.evaluate()
            run_manager.log(name=name, **entry)

    # Checkpoint
    if epoch % checkpoint_every == 0:
        run_manager.make_checkpoint(trainer)

    # Backup
    if epoch % backup_every == 0:
        run_manager.make_backup(trainer)

    # Epoch train
    trainer.train_epoch()
    if trainer.model.training_done:
        print('Training has finished')
        break

# Save the model at the end of the run
run_manager.make_backup(trainer)