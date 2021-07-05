
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.optim as optimizer_module

from jade.model import Model
from jade.utils import TimeInterval, tqdm_wrap


def iteration(f):
    pbar = None

    def wrap(trainer, *args, **kwargs):
        nonlocal pbar
        assert isinstance(trainer, Trainer)

        trainer.model.iterations += 1
        if pbar is None:
            pbar = tqdm_wrap(total=100)
        pbar.update(trainer._train_timer.percentage(trainer.model) - pbar.n)

        if trainer._train_timer.is_time(trainer.model):
            trainer.model.end_training()
            return

        # Log the loss values every log_loss_every iterations
        if trainer.loss_log_timer.is_time(trainer.model):
            trainer.loss_log_timer.update(trainer.model)
            for key, value in trainer.model.get_items_to_log().items():
                trainer.writer.log(name=key, value=value, type='scalar', iteration=trainer.model.iterations)

        # Evaluation
        for name, evaluator in trainer.evaluators.items():

            entry = evaluator.evaluate_if_time(trainer.model)
            if not (entry is None):
                trainer.writer.log(name=name, iteration=trainer.model.iterations, **entry)

        # Backup
        if trainer.backup_timer.is_time(trainer.model):
            trainer.backup_timer.update(trainer.model)
            trainer.writer.make_backup(trainer)

        # Checkpoint
        if trainer.checkpoint_timer.is_time(trainer.model):
            trainer.checkpoint_timer.update(trainer.model)
            trainer.writer.make_checkpoint(trainer)

        return f(trainer, *args, **kwargs)

    return wrap


def epoch(f):
    def wrap(trainer, *args, **kwaargs):
        assert isinstance(trainer, Trainer)
        trainer.model.epochs += 1
        return f(trainer, *args, **kwaargs)
    return wrap


#######################
# Generic Model class #
#######################

class Trainer:
    def __init__(self, optimizers, evaluators, writer, model, verbose=False,
                 log_loss_every='100 iterations',
                 checkpoint_every='100 epochs',
                 backup_every='1 epochs',
                 **params):
        super(Trainer, self).__init__()

        assert isinstance(model, Model)
        self._items_to_store = set()

        self.writer = writer
        self.model = model
        self.evaluators = evaluators
        self.verbose = verbose

        self.optimizer_descriptions = optimizers
        self.initialize_optimizers()

        self.training_done = False
        self._train_timer = None

        self.loss_log_timer = TimeInterval(log_loss_every)
        self.checkpoint_timer = TimeInterval(checkpoint_every)
        self.backup_timer = TimeInterval(backup_every)

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

    def train(self, train_timer):
        self._train_timer = train_timer
        self.model.start_training()
        while (not self.model.training_done):
            self.train_epoch()

    @epoch
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

        if self.verbose:
            print("Saving the model at %d iterations in %s" % (self.model.iterations, path))

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
                print('Loading state dict for %s' % key)
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
    def initialize(self, train_on, batch_size, num_workers=0, shuffle=True):

        self.train_loader = DataLoader(
            self.datasets[train_on],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        self.first_iteration = True

        if not(hasattr(self.model, 'compute_loss')):
            raise Exception('The model "%s" must implement a compute_loss(data) method.' % self.model.__class__.__name__)

    @epoch
    def train_epoch(self):
        if self.first_iteration:
            self.first_iteration = False

        if self.model.training_done:
            return

        for data in tqdm_wrap(self.train_loader):

            device = self.get_device()
            # Move the data to the appropriate device
            if hasattr(data, 'items'):
                for name, value in data.items():
                    data[name] = value.to(device)
            elif isinstance(data, tuple):
                data = (element.to(device) for element in data)
            else:
                data = data.to(device)

            self.train_step(data)

            if self.model.training_done:
                return

    @ iteration
    def train_step(self, data):
        # Set all the models in training mode
        self.model.train()

        loss = self.model.compute_loss(data)

        for opt in self.optimizers.values():
            opt.zero_grad()
        loss.backward()
        for opt in self.optimizers.values():
            opt.step()



