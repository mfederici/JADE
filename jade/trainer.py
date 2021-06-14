from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.optim as optim_module
from jade.instance_manager import make_instance


##########################
# Generic training class #
##########################

class Trainer(nn.Module):
    def __init__(self, datasets, arch_module, log_loss_every=100, writer=None, seed=None, verbose=False, **params):
        super(Trainer, self).__init__()
        self._arch_module = arch_module
        self._items_to_store = {'iterations', 'epochs'}

        self.datasets = datasets
        self.verbose = verbose

        self.iterations = 0
        self.epochs = 0
        self.training_done = False

        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.train_loader = None

        self.writer = writer
        self.log_loss_every = log_loss_every

        self.loss_items = {}
        self.arch_module = arch_module

        self.last_logged = -1
        self.first_iteration = True

        self.initialize(**params)

        if self.train_loader is None:
            raise Exception(
            'The attribute self.train_loader needs to be defined within the `initialize` method with a valid DataLoader'
            )

    def instantiate_architecture(self, class_name, **params):
        instance = make_instance(class_name=class_name, modules=[self._arch_module], params=params)
        return instance

    def get_dataset(self, dataset_name):
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]
        else:
            raise Exception('The dataset %s has not been specified in the dataset configuration file' % dataset_name)

    def get_device(self):
        return list(self.parameters())[0].device

    def on_start(self):
        pass

    def add_attribute_to_store(self, attribute):
        if hasattr(self, attribute):
            self._items_to_store.add(attribute)
        else:
            raise Exception('No attribute named `%s` specified in the trainer' % attribute)

    def train_epoch(self):
        if self.first_iteration:
            self.first_iteration = False
            self.on_start()

        if self.training_done:
            return

        for data in tqdm(self.train_loader):

            # Log the values in loss_items every log_loss_every iterations
            if not (self.writer is None) and not self.iterations == self.last_logged:
                if (self.iterations + 1) % self.log_loss_every == 0:
                    self._log_loss()
                    self.last_logged = self.iterations

            # Move the data to the appropriate device
            device = self.get_device()
            if hasattr(data, 'items'):
                for name, value in data.items():
                    data[name] = value.to(device)
            elif isinstance(data, tuple):
                data = (element.to(device) for element in data)
            else:
                data = data.to(device)

            # Set all the models in training mode
            self.train(True)

            self.train_step(data)
            self.on_iteration_end()
            if self.training_done:
                return

        self.on_epoch_end()

    def on_iteration_end(self):
        self.iterations += 1

    def end_training(self):
        self.training_done = True

    def on_epoch_end(self):
        self.epochs += 1

    def add_loss_item(self, name, value):
        assert isinstance(name, str)
        assert isinstance(value, float) or isinstance(value, int)

        if not (name in self.loss_items):
            self.loss_items[name] = []

        self.loss_items[name].append(value)

    def _log_loss(self):
        # Log the expected value of the items in loss_items
        for key, values in self.loss_items.items():
            self.writer.log(name=key, value=np.mean(values), type='scalar', iteration=self.iterations)
            self.loss_items[key] = []

    def save(self, model_path):
        save_dict = {}
        for name in self._items_to_store:
            value = getattr(self, name)
            if isinstance(value, nn.Module) or isinstance(value, Optimizer):
                value = value.state_dict()
            save_dict[name] = value

        # Save the model and increment the checkpoint count
        torch.save(save_dict, model_path)

    def load(self, model_path):
        items_to_load = torch.load(model_path, map_location=torch.device('cpu'))
        for key, value in items_to_load.items():
            assert hasattr(self, key)
            attribute = getattr(self, key)

            # Load the state dictionary for the stored modules and optimizers
            if isinstance(attribute, nn.Module) or isinstance(attribute, Optimizer):
                attribute.load_state_dict(value)

            # Otherwise just copy the value
            else:
                setattr(self, key, value.to(self.get_device()))

    def train_step(self, data):
        raise NotImplemented()

    def initialize(self, **params):
        raise NotImplemented()
