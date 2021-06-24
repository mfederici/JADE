from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.optim as optimizer_module

from jade.model import Model


#######################
# Generic Model class #
#######################

class Trainer:
    def __init__(self, optimizers, writer, model, verbose=False, log_loss_every=100, **params):
        super(Trainer, self).__init__()

        assert isinstance(model, Model)
        self._items_to_store = set()

        self.writer = writer
        self.model = model
        self.verbose = verbose

        self.optimizer_descriptions = optimizers
        self.initialize_optimizers()

        self.training_done = False

        self.log_loss_every = log_loss_every
        self.last_logged = -1
        self.loss_items = {}

        self.initialize(**params)

    def initialize(self, **params):
        raise NotImplementedError()

    def to(self, device):
        self.model = self.model.to(device)

    def get_device(self):
        return self.model.get_device()

    def initialize_optimizers(self):
        opt_ref = {}
        for attribute_name, optimizer_name in self.model._attributes_to_optimize.items():
            parameters = getattr(self.model, attribute_name).parameters()
            if optimizer_name in opt_ref:
                opt_ref[optimizer_name].append({'params': parameters})
                if self.verbose:
                    print('The attribute "%s" of "%s" is optimized with "%s"' % (attribute_name,
                                                                                 self.model.__class__.__name__,
                                                                                 optimizer_name))
            else:
                opt_ref[optimizer_name] = [{'params': parameters}]

        optimizers = {}
        for optimizer_name in opt_ref:
            if not optimizer_name in self.optimizer_descriptions:
                raise Exception(
                    'The optimizer named "%s" referred in the model class "%s" has not been specified '\
                    'in the trainer configuration file.\n The available optimizers are "%s"' %
                    (optimizer_name, self.model.__class__.__name__, ','.join(list(self.optimizer_descriptions.keys())))
                )
            optimizer_description = self.optimizer_descriptions[optimizer_name]
            OptimizerClass = getattr(optimizer_module, optimizer_description['class'])
            optimizers[optimizer_name] = OptimizerClass(opt_ref[optimizer_name], **optimizer_description['params'])

        self.optimizers = optimizers

    def add_attribute_to_store(self, attribute):
        if hasattr(self, attribute):
            self._items_to_store.add(attribute)
        else:
            raise Exception('No attribute named `%s` specified in the trainer' % attribute)

    def train_epoch(self):
        raise NotImplementedError()

    def save(self, path):
        save_dict = {}
        for name in self._items_to_store:
            value = getattr(self, name)
            if isinstance(value, nn.Module) or isinstance(value, Optimizer):
                value = value.state_dict()
            save_dict[name] = value

        torch.save({'model': self.model.get_state_dict(),
                    'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
                    'attributes': save_dict}, path)

    def load(self, path, device='cpu'):
        items_to_load = torch.load(path, map_location=device)

        self.model.load_state_dict(items_to_load['model'])

        # self.initialize_optimizers()

        for key, state_dict in items_to_load['optimizers'].items():
            assert key in self.optimizers

            self.optimizers[key].load_state_dict(state_dict)

            if self.verbose:
                print('Loading state dict for the optimizer %s' % key)

        for key, value in items_to_load['attributes'].items():
            assert hasattr(self, key)
            attribute = getattr(self, key)

            # Load the state dictionary for the stored example and optimizers
            if isinstance(attribute, nn.Module) or isinstance(attribute, Optimizer):
                attribute.load_state_dict(value)
                print('Loadind state dict for %s' % key)
            # Otherwise just copy the value
            else:
                if hasattr(value, 'to'):
                    value = value.to(self.get_device())
                setattr(self, key, value)


class DatasetTrainer(Trainer):
    def __init__(self, datasets, **params):
        self.datasets = datasets
        super(DatasetTrainer, self).__init__(**params)


class BatchTrainer(DatasetTrainer):
    def initialize(self, train_on, batch_size, num_workers=0, shuffle=True,):

        self.train_loader = DataLoader(
            self.datasets[train_on],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        self.first_iteration = True
        self.last_logged = -1

        if not(hasattr(self.model, 'compute_loss')):
            raise Exception('The model "%s" must implement a compute_loss(data) method.' % self.model.__class__.__name__)

    def train_epoch(self):
        if self.first_iteration:
            self.first_iteration = False
            self.model.on_start()

        if self.model.training_done:
            return

        for data in tqdm(self.train_loader):
            # Log the values in loss_items every log_loss_every iterations
            if not self.model.iterations == self.last_logged:
                if (self.model.iterations + 1) % self.log_loss_every == 0:
                    self.model._log_loss()
                    self.last_logged = self.model.iterations

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
            self.model.train(True)

            loss = self.model.compute_loss(data)

            for opt in self.optimizers.values():
                opt.zero_grad()
            loss.backward()
            for opt in self.optimizers.values():
                opt.step()

            self.model.on_iteration_end()

            if self.model.training_done:
                return

        self.model.on_epoch_end()


