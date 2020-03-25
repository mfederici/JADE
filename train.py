import os
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, RandomAffine, ToTensor
from torch.utils.data import random_split

from data import *
from utils.data import PixelCorruption, AugmentedDataset
from utils.evaluation import evaluate, split
import training as training_module
import data as data_module
import eval as eval_module


parser = argparse.ArgumentParser()
parser.add_argument("experiment_dir", type=str,
                    help="Full path to the experiment directory. Logs and checkpoints will be stored in this location")
parser.add_argument("--data-config", type=str, default=None, help="Path to the .yml data description file.")
parser.add_argument("--trainer-config", type=str, default=None, help="Path to the .yml training configuration file.")
parser.add_argument("--eval-config", type=str, default=None, help="Path to the .yml evaluation configuration file.")
parser.add_argument("--no-logging", action="store_true", help="Disable tensorboard logging")
parser.add_argument("--overwrite", action="store_true",
                    help="Force the over-writing of the previous experiment in the specified directory.")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device on which the experiment is executed (as for tensor.device). Specify 'cpu' to "
                         "force execution on CPU.")
parser.add_argument("--num-workers", type=int, default=8,
                    help="Number of CPU threads used during the data loading procedure.")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size used for the experiments.")
parser.add_argument("--load-model-file", type=str, default=None,
                    help="Checkpoint to load for the experiments. Note that the specified configuration file needs "
                         "to be compatible with the checkpoint.")
parser.add_argument("--checkpoint-every", type=int, default=50, help="Frequency of model checkpointing (in epochs).")
parser.add_argument("--backup-every", type=int, default=5, help="Frequency of model backups (in epochs).")
parser.add_argument("--evaluate-every", type=int, default=5, help="Frequency of model evaluation.")
parser.add_argument("--epochs", type=int, default=1000, help="Total number of training epochs")

args = parser.parse_args()

logging = not args.no_logging
experiment_dir = args.experiment_dir
trainer_config_file = args.trainer_config
data_config_file = args.data_config
eval_config_file = args.eval_config
overwrite = args.overwrite
device = args.device
num_workers = args.num_workers
batch_size = args.batch_size
load_model_file = args.load_model_file
checkpoint_every = args.checkpoint_every
backup_every = args.backup_every
evaluate_every = args.evaluate_every
epochs = args.epochs

# Check if the experiment directory already contains a model
pretrained = os.path.isfile(os.path.join(experiment_dir, 'model.pt')) \
             and os.path.isfile(os.path.join(experiment_dir, 'config.yml'))


if pretrained and not (trainer_config_file is None) and not overwrite:
    raise Exception("The experiment directory %s already contains a trained model, please specify a different "
                    "experiment directory or remove the --config-file option to resume training or use the --overwrite"
                    "flag to force overwriting")

resume_training = pretrained and not overwrite

if resume_training:
    load_model_file = os.path.join(experiment_dir, 'model.pt')
    trainer_config_file = os.path.join(experiment_dir, 'trainer_config.yml')
    data_config_file = os.path.join(experiment_dir, 'data_config.yml')
    eval_config_file = os.path.join(experiment_dir, 'eval_config.yml')

if logging:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=experiment_dir)
else:
    os.makedirs(experiment_dir, exist_ok=True)
    writer = None

# Load the configuration file
with open(trainer_config_file, 'r') as file:
    config = yaml.safe_load(file)

# Copy it to the experiment folder
with open(os.path.join(experiment_dir, 'trainer_config.yml'), 'w') as file:
    yaml.dump(config, file)


###########
# Dataset #
###########

# Load the data_description file
with open(data_config_file, 'r') as file:
    data_config = yaml.safe_load(file)

# Copy it to the experiment folder
with open(os.path.join(experiment_dir, 'data_config.yml'), 'w') as file:
    yaml.dump(data_config, file)

# Instantiate the different datasets used for training and evaluation
datasets = {}
for dataset_description in data_config:
    DatasetClass = getattr(data_module, dataset_description['class'])
    datasets[dataset_description['name']] = DatasetClass(**dataset_description['params'])

if not ('train_set' in datasets):
    raise Exception('The data description file %s must contain a train_set' % data_config_file)

# Instantiating the trainer according to the specified configuration
TrainerClass = getattr(training_module, config['trainer'])
trainer = TrainerClass(dataset=datasets['train_set'], writer=writer, **config['params'])

# Resume the training if specified
if load_model_file:
    trainer.load(load_model_file)

# Moving the models to the specified device
trainer.to(device)

##############
# Evaluation #
##############

# Load the evaluation file
with open(eval_config_file, 'r') as file:
    eval_config = yaml.safe_load(file)

# Copy it to the experiment folder
with open(os.path.join(experiment_dir, 'eval_config.yml'), 'w') as file:
    yaml.dump(eval_config, file)

evaluators = {}
for entry in eval_config:
    EvalClass = getattr(eval_module, entry['class'])
    evaluators[entry['name']] = EvalClass(datasets=datasets, model=getattr(trainer, entry['model']),
                                          device=trainer.get_device(), **entry['params'])

checkpoint_count = 1

for epoch in tqdm(range(epochs)):
    for name, evaluator in evaluators.items():
        if epoch % evaluator.evaluate_every == 0:
            entry = evaluator.evaluate()

            logging_function = getattr(writer, 'add_%s' % entry['type'])
            logging_function(tag=name, global_step=trainer.iterations, **entry['params'])

    if epoch % checkpoint_every == 0:
        tqdm.write('Storing model checkpoint')
        while os.path.isfile(os.path.join(experiment_dir, 'checkpoint_%d.pt' % checkpoint_count)):
            checkpoint_count += 1

        trainer.save(os.path.join(experiment_dir, 'checkpoint_%d.pt' % checkpoint_count))
        checkpoint_count += 1

    if epoch % backup_every == 0:
        tqdm.write('Updating the model backup')
        trainer.save(os.path.join(experiment_dir, 'model.pt'))

    trainer.train_epoch()
