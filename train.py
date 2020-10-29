import os
import argparse
from tqdm import tqdm

import torch
import numpy as np

from utils.run_manager.wandb import WANDBRunManager

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default=None,
                    help="Path to the .yml file containing the dataset description.")
parser.add_argument("--model_file", type=str, default=None,
                    help="Path to the .yml file containing the model description.")
parser.add_argument("--eval_file", type=str, default=None,
                    help="Path to the .yml file containing the evaluation description.")
parser.add_argument("--run_name", type=str, default=None,
                    help="Unique name associated to the experiment")
parser.add_argument("--experiments-root", type=str, default="experiments",
                    help="Root of the experiment directory. Checkpoints will be stored in sub-directories corresponding"
                         " to their respective run id.")
parser.add_argument("--data-root", type=str, default=".",
                    help="Root directory for the datasets")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device on which the experiment is executed (as for tensor.device). Specify 'cpu' to "
                         "force execution on CPU.")
parser.add_argument("--num-workers", type=int, default=1,
                    help="Number of CPU threads used during the data loading procedure.")
# Change checkpoint frequency
parser.add_argument("--checkpoint-every", type=int, default=500, help="Frequency of model checkpointing (in epochs).")
parser.add_argument("--backup-every", type=int, default=100, help="Frequency of model backups (in epochs).")
parser.add_argument("--epochs", type=int, default=1000, help="Total number of training epochs")
parser.add_argument("--seed", type=int, default=42, help="Random seed for the experiment")

args = parser.parse_args()

logging = True

data_file = args.data_file
model_file = args.model_file
eval_file = args.eval_file

run_name = args.run_name

checkpoint_every = args.checkpoint_every
backup_every = args.backup_every
epochs = args.epochs

experiments_root = args.experiments_root
data_root = args.data_root

device = args.device
if 'DEVICE' in os.environ:
    device = os.environ['DEVICE']

num_workers = args.num_workers
if 'N_WORKERS' in os.environ:
    num_workers = int(os.environ['N_WORKERS'])

seed = args.seed

# Set random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)

upload_checkpoints = True
verbose = True

run_manager = WANDBRunManager(run_name=run_name, desc={'data_file': data_file,
                                                       'model_file': model_file,
                                                       'eval_file': eval_file},
                              num_workers=num_workers,
                              experiments_root=experiments_root, data_root=data_root,
                              verbose=verbose, upload_checkpoints=upload_checkpoints)

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
            run_manager.log(name=name, value=entry['value'], entry_type=entry['type'], iteration=trainer.iterations)

    # Checkpoint
    if epoch % checkpoint_every == 0:
        run_manager.make_checkpoint(trainer)

    # Backup
    if epoch % backup_every == 0:
        run_manager.make_backup(trainer)

    # Epoch train
    trainer.train_epoch()

# Save the model at the end of the run
run_manager.make_backup(trainer)