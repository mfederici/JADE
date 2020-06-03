import os
import argparse
from tqdm import tqdm

from utils.run_manager import RunManager, load_descriptions_from_dir

parser = argparse.ArgumentParser()
parser.add_argument("run_name", type=str,
                    help="Unique name associated to the experiment")
parser.add_argument("--var", type=str,
                    help="String to specify which components are used during the training procedure. "
                         "Note that the variables \'train_on\' and \'model\' have to be specified.")
parser.add_argument("--experiments-root", type=str, default="experiments",
                    help="Root of the experiment directory. Checkpoints will be stored in sub-directories corresponding"
                         " to their respective run id.")
parser.add_argument("--def-dir", type=str, default='definitions',
                    help="Path to the directory containing the .yaml datasets, models, architectures and evaluations "
                         "definition files.")
parser.add_argument("--repeat", action="store_true",
                    help="Force re-running the experiments with the same name. "
                         "By default experiments with the same name are resumed instead")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device on which the experiment is executed (as for tensor.device). Specify 'cpu' to "
                         "force execution on CPU.")
parser.add_argument("--num-workers", type=int, default=1,
                    help="Number of CPU threads used during the data loading procedure.")
# Change checkpoint frequency
parser.add_argument("--checkpoint-every", type=int, default=50, help="Frequency of model checkpointing (in epochs).")
parser.add_argument("--backup-every", type=int, default=5, help="Frequency of model backups (in epochs).")
parser.add_argument("--evaluate-every", type=int, default=5, help="Frequency of model evaluation.")
parser.add_argument("--epochs", type=int, default=1000, help="Total number of training epochs")

args = parser.parse_args()

#TODO
var = {
    'train_set': 'coloured_MNIST_train',
    'model': 'IDA',
    'eval': ['test_accuracy/0.1', 'test_accuracy/0.2', 'test_accuracy/0.9']
}

# --data-config=configs/data/AugmentedMultiviewColouredMNIST.yml --trainer-config=configs/training/MIB_1d.yml --eval-config=configs/eval/ColouredMNIST.yml
logging = True
run_name = args.run_name
repeat = args.repeat
experiments_root = args.experiments_root
def_dir = args.def_dir
device = args.device
num_workers = args.num_workers
# load_model_file = args.load_model_file
checkpoint_every = args.checkpoint_every
backup_every = args.backup_every
evaluate_every = args.evaluate_every
epochs = args.epochs
upload_checkpoints = True
verbose = True


# Check if the experiment directory already contains a model
config = {name: load_descriptions_from_dir(os.path.join(def_dir, name), verbose=verbose)
          for name in ['data', 'train', 'eval']}
config['var'] = var

run_manager = RunManager(run_name=run_name, config=config, experiments_dir=experiments_root,
                         repeat=repeat, verbose=verbose, upload_checkpoints=upload_checkpoints)
experiment_dir = run_manager.experiment_dir

train_set, trainer, evaluators = run_manager.make_instances()
# Moving the models to the specified device
trainer.to(device)

# Training loop
for epoch in tqdm(range(epochs)):
    # Evaluation
    for name, evaluator in evaluators.items():
        if epoch % evaluator.evaluate_every == 0:
            entry = evaluator.evaluate()
            run_manager.log(name=name, value=entry['value'], entry_type=entry['type'], iteration=trainer.iterations)

    # Checkpoint
    if epoch % checkpoint_every == 0:
        run_manager.make_checkpoint(trainer)

    # Backup
    if epoch % backup_every == 0:
        run_manager.make_backup(trainer)

    # Epoch train
    trainer.train_epoch()
