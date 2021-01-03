from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.optim as optim_module
import utils.schedulers as scheduler_module

##########################
# Generic training class #
##########################

class Trainer(nn.Module):
    def __init__(self, dataset, batch_size, arch_module, log_loss_every=100, num_workers=0, writer=None, verbose=True):
        super(Trainer, self).__init__()
        self.iterations = 0
        self.epochs = 0

        self.verbose = verbose

        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.writer = writer
        self.log_loss_every = log_loss_every

        self.loss_items = {}
        self.arch_module = arch_module

        self.last_logged = -1
        self.first_iteration = True

    def instantiate_architecture(self, class_name, **arch_params):
        if not hasattr(self.arch_module, class_name):
            raise Exception('A class implementation of %s(%s) has to be included in %s' % (
                class_name, ','.join(arch_params.keys()), self.arch_module.__file__
            ))

        if self.verbose:
            print('Instantiating %s with the parameters %s'%(class_name, str(arch_params)))
        return getattr(self.arch_module, class_name)(**arch_params)

    @staticmethod
    def instantiate_optimizer(opt_desc, **opt_params):
        if 'params' in opt_desc:
            opt_params.update(opt_desc['params'])
        return getattr(optim_module, opt_desc['class'])(**opt_params)

    def get_device(self):
        return list(self.parameters())[0].device

    def on_start(self):
        pass

    def train_epoch(self):
        if self.first_iteration:
            self.first_iteration = False
            self.on_start()

        for data in tqdm(self.train_loader):

            # Log the values in loss_items every log_loss_every iterations
            if not (self.writer is None) and not self.iterations == self.last_logged:
                if (self.iterations + 1) % self.log_loss_every == 0:
                    self._log_loss()
                    self.last_logged = self.iterations

            # Move the data to the appropriate device
            device = self.get_device()
            for name, value in data.items():
                data[name] = value.to(device)

            # Set all the models in training mode
            self.train(True)

            self.train_step(data)
            self.on_iteration_end()

        self.on_epoch_end()

    def on_iteration_end(self):
        self.iterations += 1

    def on_epoch_end(self):
        self.epochs += 1

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
        items_to_load = torch.load(model_path, map_location=torch.device('cpu'))
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

    def train_step(self, data):
        raise NotImplemented()


##########################
# Representation Trainer #
##########################

# Generic class to train an model with a (stochastic) neural network encoder
class RepresentationTrainer(Trainer):
    def __init__(self, z_dim, encoder, optim, lr_schedule=None, **params):
        super(RepresentationTrainer, self).__init__(**params)

        self.z_dim = z_dim

        # Instantiating the encoder
        self.encoder = self.instantiate_architecture('Encoder', z_dim=z_dim, **encoder)

        # Instantiating the optimizer
        self.opt = self.instantiate_optimizer(optim, params=self.encoder.parameters())

        # Definition of learning rate schedulers
        self.lr_scheduler = []
        self.lr_schedule = lr_schedule

    def on_start(self):
        if not (self.lr_schedule is None):
            LRScheduleClass = getattr(torch.optim.lr_scheduler, self.lr_schedule['class'])
            for opt_name in self.lr_schedule['apply_to']:
                opt = getattr(self, opt_name)
                self.lr_scheduler.append(LRScheduleClass(opt, **self.lr_schedule['params']))

    def on_iteration_end(self):
        super(RepresentationTrainer, self).on_iteration_end()
        for lr_schedule in self.lr_scheduler:
            lr_schedule.step()

    def _get_items_to_store(self):
        items_to_store = super(RepresentationTrainer, self)._get_items_to_store()

        # store the encoder and optimizer parameters
        items_to_store = items_to_store.union({
            'encoder',
            'opt'}
        )

        return items_to_store

    def train_step(self, data):
        loss = self._compute_loss(data)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def _compute_loss(self, data):
        raise NotImplemented()


######################
# Classifier Trainer #
######################

class ClassifierTrainer(RepresentationTrainer):
    def __init__(self, z_dim, optim, label_classifier=None, **params):

        super(ClassifierTrainer, self).__init__(z_dim=z_dim, optim=optim, **params)

        self.classifier = self.instantiate_architecture('LabelClassifier', z_dim=z_dim, **label_classifier)

        self.opt.add_param_group(
            {'params': self.classifier.parameters()}
        )

    def _get_items_to_store(self):
        items_to_store = super(ClassifierTrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'classifier'
        })

        return items_to_store

    def _compute_loss(self, data):
        x = data['x']
        y = data['y'].squeeze()

        # Encode a batch of data
        z = self.encoder(x=x).rsample()

        # Label Reconstruction
        p_y_given_z = self.classifier(z=z)
        y_rec_loss = - p_y_given_z.log_prob(y).mean()

        loss = y_rec_loss

        self._add_loss_item('loss/CE_y_z', y_rec_loss.item())

        return loss


##################################
# Regularized Classifier Trainer #
##################################

class RegularizedClassifierTrainer(ClassifierTrainer):
    def __init__(self, beta_scheduler, normalize_reg_coeff=True, **params):

        super(RegularizedClassifierTrainer, self).__init__(**params)

        # Definition of the scheduler to update the value of the regularization coefficient beta over time
        self.beta_scheduler = getattr(scheduler_module, beta_scheduler['class'])(**beta_scheduler['params'])
        self.normalize_reg_coeff = normalize_reg_coeff

    def _compute_reg_loss(self, data, z):
        raise NotImplemented()

    def _compute_loss(self, data):

        x = data['x']
        y = data['y'].squeeze()

        # Encode a batch of data
        z = self.encoder(x=x).rsample()

        # Label Reconstruction
        p_y_given_z = self.classifier(z=z)
        y_rec_loss = - p_y_given_z.log_prob(y).mean()

        reg_loss = self._compute_reg_loss(data, z)

        beta = self.beta_scheduler(self.iterations)

        if self.normalize_reg_coeff:
            loss = 1. / (beta + 1.) * y_rec_loss + beta / (beta + 1.) * reg_loss
        else:
            loss = y_rec_loss + beta * reg_loss

        self._add_loss_item('loss/beta', beta)
        self._add_loss_item('loss/CE_y_z', y_rec_loss.item())

        return loss


######################################
# Adversarial Representation trainer #
######################################

ADV_ALT_TRAIN = 'alternating'
#ADV_SIM_TRAIN = 'simultaneous'


ADV_TRAIN_TYPES = {ADV_ALT_TRAIN} #, ADV_SIM_TRAIN}


class AdversarialRepresentationTrainer(RegularizedClassifierTrainer):
    def __init__(self, adversary, n_adv_steps=5, adv_optim=None,
                 adv_train_type=ADV_ALT_TRAIN,
                 **params):

        super(AdversarialRepresentationTrainer, self).__init__(**params)

        self.n_adv_steps = n_adv_steps
        self.step = 0
        self.discriminator_step = False

        # Instantiate the adversary

        self.adversary = self.instantiate_architecture(**adversary)

        if adv_train_type == ADV_ALT_TRAIN:
            assert not (adv_optim is None)
            self.adv_opt = self.instantiate_optimizer(adv_optim, params=self.adversary.parameters())
        else:
            raise NotImplemented()

        assert adv_train_type in ADV_TRAIN_TYPES
        assert adv_train_type != ADV_ALT_TRAIN or not (adv_optim is None)
        self.adv_train_type = adv_train_type

    def _get_items_to_store(self):
        items_to_store = super(AdversarialRepresentationTrainer, self)._get_items_to_store()

        items_to_store = items_to_store.union({
            'adversary'
        })

        if self.adv_opt:
            items_to_store = items_to_store.union({'adv_opt'})

        return items_to_store

    def train_step(self, data):
        # Alternating adversarial procedure
        if self.adv_train_type == ADV_ALT_TRAIN:
           if self.step < self.n_adv_steps:
                # Train the adversary
                self.adversary.train()
                loss = self._compute_adv_loss(data)

                self.adv_opt.zero_grad()
                loss.backward()
                self.adv_opt.step()
                self.step += 1

                self.discriminator_step = True
           else:
                self.adversary.eval()

                # Train the model
                loss = self._compute_loss(data)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.step = 0
                self.discriminator_step = False

    def on_iteration_end(self):
        # Update the iteration count only when updating the model
        if not self.discriminator_step:
            self.iterations += 1

    def _compute_reg_loss(self, data, z):
        return - self._compute_adv_loss(data, z)

    def _compute_adv_loss(self, data, z=None):
        raise NotImplemented()


