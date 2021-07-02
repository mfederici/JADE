import inspect
import torch
import torch.nn as nn
import numpy as np
from jade.instance_manager import make_instance
import time


#######################
# Generic Model class #
#######################

class Model(nn.Module):
    def __init__(self, arch_modules, verbose=False, **params):
        super(Model, self).__init__()
        self._arch_modules = arch_modules

        self.iterations = 0
        self.epochs = 0
        self._seconds = 0
        self.last_time = 0

        self._attributes_to_store = {'iterations', 'epochs', '_seconds'}
        self._attributes_to_optimize = {}

        self.verbose = verbose
        self.training_done = False
        self.first_iteration = True

        self.loss_items = {}

        try:
            self.initialize(**params)
        except TypeError as e:
            extra_param = str(e).split('got an unexpected keyword argument ')[1]
            error_message = '%s does not have a parameter %s. The available parameters are %s' % \
                            (self.__class__.__name__, extra_param, inspect.signature(self.initialize))
            raise Exception(error_message)

    def __getattr__(self, item):
        if item == 'seconds':
            current_time = time.time()
            self._seconds += current_time - self.last_time
            self.last_time = current_time
            return self._seconds
        elif item == 'minutes':
            return self.seconds / 60.
        elif item == 'hours':
            return self.minutes / 60.
        elif item == 'days':
            return self.hours / 24.
        else:
            return super(Model, self).__getattr__(item)

    def initialize(self, **params):
        raise NotImplemented()

    def start_training(self):
        self.last_time = time.time()

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
            if not hasattr(self, key):
                raise Exception('Missing attribute: %s' % key)
            attribute = getattr(self, key)
            # Otherwise just copy the value
            if hasattr(value, 'to'):
                value = value.to(self.get_device())

            if isinstance(attribute, nn.Module):
                attribute.load_state_dict(value)
            else:
                setattr(self, key, value)

    def add_loss_item(self, name, value):
        assert isinstance(name, str)
        assert isinstance(value, float) or isinstance(value, int)

        if not (name in self.loss_items):
            self.loss_items[name] = []

        self.loss_items[name].append(value)

    def get_items_to_log(self):
        # Log the expected value of the items in loss_items
        loss_items = {}
        for key, values in self.loss_items.items():
            loss_items[key] = np.mean(values)
            self.loss_items[key] = []
        return loss_items

    def compute_loss(self, data):
        raise NotImplemented()
