from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.optim as optim_module


##########################
# Generic training class #
##########################

class Trainer(nn.Module):
    def __init__(self, dataset, batch_size, arch_module, log_loss_every=10, num_workers=0, writer=None):
        super(Trainer, self).__init__()
        self.iterations = 0

        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.writer = writer
        self.log_loss_every = log_loss_every

        self.loss_items = {}
        self.arch_module = arch_module

    def instantiate_architecture(self, class_name, **arch_params):
        if not hasattr(self.arch_module, class_name):
            raise Exception('A class implementation of %s(%s) has to be included in %s' % (
                class_name, ','.join(arch_params.keys()), self.arch_module.__file__
            ))

        print('Instantiating %s with the parameters %s'%(class_name, str(arch_params)))
        return getattr(self.arch_module, class_name)(**arch_params)

    @staticmethod
    def instantiate_optimizer(opt_desc, **opt_params):
        if 'params' in opt_desc:
            opt_params.update(opt_desc['params'])
        return getattr(optim_module, opt_desc['class'])(**opt_params)

    def get_device(self):
        return list(self.parameters())[0].device

    def train_epoch(self):
        for data in tqdm(self.train_loader):
            self.train_step(data)

    def train_step(self, data):
        # Set all the models in training mode
        self.train(True)

        # Log the values in loss_items every log_loss_every iterations
        if not (self.writer is None):
            if (self.iterations + 1) % self.log_loss_every == 0:
                self._log_loss()

        # Move the data to the appropriate device
        device = self.get_device()
        for name, value in data.items():
            data[name] = value.to(device)

        # Perform the training step and update the iteration count
        self._train_step(data)
        self.iterations += 1

    def _add_loss_item(self, name, value):
        assert isinstance(name, str)
        assert isinstance(value, float) or isinstance(value, int)

        if not (name in self.loss_items):
            self.loss_items[name] = []

        self.loss_items[name].append(value)

    def _log_loss(self):
        # Log the expected value of the items in loss_items
        for key, values in self.loss_items.items():
            self.writer.log(name=key, value=np.mean(values), entry_type='scalar', iteration=self.iterations)
            self.loss_items[key] = []

    def save(self, model_path):
        save_dict = {}
        for name in self._get_items_to_store():
            value = getattr(self, name)
            if isinstance(value, nn.Module) or isinstance(value, Optimizer):
                value = value.state_dict()
            save_dict[name] = value

        # Save the model and increment the checkpoint count
        torch.save(save_dict, model_path)

    def load(self, model_path):
        items_to_load = torch.load(model_path)
        for key, value in items_to_load.items():
            assert hasattr(self, key)
            attribute = getattr(self, key)

            # Load the state dictionary for the stored modules and optimizers
            if isinstance(attribute, nn.Module) or isinstance(attribute, Optimizer):
                attribute.load_state_dict(value)

                # Move the optimizer parameters to the correct device.
                # see https://github.com/pytorch/pytorch/issues/2830 for further details
                # if isinstance(attribute, Optimizer):
                #     device = list(value['state'].values())[0]['exp_avg'].device # Hack to identify the device
                #     for state in attribute.state.values():
                #         for k, v in state.items():
                #             if isinstance(v, torch.Tensor):
                #                 state[k] = v.to(device)

            # Otherwise just copy the value
            else:
                setattr(self, key, value)

    def _get_items_to_store(self):
        return {'iterations'}

    def _train_step(self, data):
        raise NotImplemented()


##########################
# Representation Trainer #
##########################

# Generic class to train an model with a (stochastic) neural network encoder
class RepresentationTrainer(Trainer):
    def __init__(self, z_dim, encoder, optim, **params):
        super(RepresentationTrainer, self).__init__(**params)

        self.z_dim = z_dim

        # Instantiating the encoder
        self.encoder = self.instantiate_architecture('Encoder', z_dim=z_dim, **encoder)

        # Instantiating the optimizer
        self.opt = self.instantiate_optimizer(optim, params=self.encoder.parameters())

    def _get_items_to_store(self):
        items_to_store = super(RepresentationTrainer, self)._get_items_to_store()

        # store the encoder and optimizer parameters
        items_to_store = items_to_store.union({
            'encoder',
            'opt'}
        )

        return items_to_store

    def _train_step(self, data):
        loss = self._compute_loss(data)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def _compute_loss(self, data):
        raise NotImplemented()
