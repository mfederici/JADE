from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
from jade.instance_manager import make_instance


#######################
# Generic Model class #
#######################

class Model(nn.Module):
    def __init__(self, arch_modules, writer=None, verbose=False, **params):
        super(Model, self).__init__()
        self._arch_modules = arch_modules

        self.iterations = 0
        self.epochs = 0

        self._attributes_to_store = {'iterations', 'epochs'}
        self._attributes_to_optimize = {}

        self.verbose = verbose
        self.training_done = False
        self.writer = writer
        self.first_iteration = True

        self.loss_items = {}

        self.initialize(**params)

        for component_name in dir(self):
            component = getattr(self, component_name)
            if isinstance(component, nn.Module):
                store = False
                for name, parameter_group in component.named_parameters():
                    store = store or parameter_group.requires_grad
                if store:
                    self.add_attribute_to_store(component_name)

    def initialize(self, **params):
        raise NotImplemented()

    def end_training(self):
        self.training_done = True

    def instantiate_architecture(self, class_name, **params):
        instance = make_instance(class_name=class_name, modules=self._arch_modules, params=params)
        return instance

    def optimize(self, attribute_name, optimizer_name):
        if not hasattr(self, attribute_name):
            raise Exception(
                'The attribute "%s" is not defined in the "initialize(**params)" function of the class "%s"' %
                (attribute_name, self.__class__.__name__)
            )
        attribute = getattr(self, attribute_name)
        if not hasattr(attribute, 'parameters'):
            raise Exception(
                'The attribute "%s" in "%s" can not be optimized' % (attribute_name, self.__class__.__name__)
            )
        self._attributes_to_optimize[attribute_name] = optimizer_name
        self.add_attribute_to_store(attribute_name)

    def get_device(self):
        return list(self.parameters())[0].device

    def add_attribute_to_store(self, attribute_name):
        if hasattr(self, attribute_name):
            self._attributes_to_store.add(attribute_name)
        else:
            raise Exception(
                'No attribute named "%s" specified in the class "%s"' % (attribute_name, self.__class__.__name__)
            )

    def get_state_dict(self):
        state_dict = {}
        for name in self._attributes_to_store:
            value = getattr(self, name)
            if isinstance(value, nn.Module):
                value = value.state_dict()
            state_dict[name] = value

        # Save the model and increment the checkpoint count
        return state_dict

    def load(self, path, device='cpu'):
        items_to_load = torch.load(path, map_location=device)['model']
        self.load_state_dict(items_to_load)

    def load_state_dict(self, state_dict, strict=True):
        for key, value in state_dict.items():
            assert hasattr(self, key)
            attribute = getattr(self, key)
            # Otherwise just copy the value
            if hasattr(value, 'to'):
                value = value.to(self.get_device())

            if isinstance(attribute, nn.Module):
                attribute.load_state_dict(value)
            else:
                setattr(self, key, value)

    def on_start(self):
        pass

    def on_iteration_end(self):
        self.iterations += 1

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

    def compute_loss(self, data):
        raise NotImplemented()
