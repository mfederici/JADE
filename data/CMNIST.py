import os
import torch
import numpy as np
from torch.distributions import Bernoulli
from torch.utils.data import ConcatDataset, random_split, Dataset, Subset
from torchvision.transforms import ToTensor
import torchvision
from torch.distributions import Categorical

from data.transforms.dataset import DatasetTransform, MoveToCache

MNIST_TRAIN_EXAMPLES = 50000
CMNIST_SIZE = 28 ** 2 * 2
CMNIST_SHAPE = [2, 28, 28]
CMNIST_N_CLASSES = 2
CMNIST_N_ENVS = 2

# Wrapper for the torchvision MNIST dataset with validation split
class MNIST(Dataset):
    def __init__(self, path, split, data_root='.'):
        super(MNIST, self).__init__()

        dataset = torchvision.datasets.MNIST(os.path.join(data_root, path), download=True,
                                             train=split in ['train', 'valid', 'train+valid'],
                                             transform=ToTensor())

        if split == 'train':
            dataset = Subset(dataset, range(MNIST_TRAIN_EXAMPLES))
        elif split == 'valid':
            dataset = Subset(dataset, range(MNIST_TRAIN_EXAMPLES, len(dataset)))
        elif not (split == 'test') and not (split == 'train+valid'):
            raise Exception('The possible splits are "train", "valid", "train+valid" and "test"')

        self.dataset = dataset

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return {'x':x, 'y': torch.LongTensor([y])}

    def __len__(self):
        return len(self.dataset)


class CMNIST(Dataset):
    def __init__(self, path, environments, split, data_root='.'):
        super(CMNIST, self).__init__()

        dataset = MNIST(path=path, split=split, data_root=data_root)

        dataset_splits = random_split(dataset, [environment['size'] for environment in environments])
        color_flip_masks = []
        label_flip_masks = []
        envs = []

        for i, environment in enumerate(environments):
            label_flip_masks.append(Bernoulli(probs=environment['flip_label']).sample([len(dataset_splits[i])]))
            color_flip_masks.append(Bernoulli(probs=environment['flip_color']).sample([len(dataset_splits[i])]))
            envs.append(torch.zeros(len(dataset_splits[i])) + i)

        self.environments = environments
        self.dataset = ConcatDataset(dataset_splits)
        self.envs = torch.cat(envs, 0).long()
        self.label_flip_mask = torch.cat(label_flip_masks, 0)
        self.color_flip_mask = torch.cat(color_flip_masks, 0)

    def __getitem__(self, index):
        data = self.dataset[index]
        im = data['x']
        label = data['y']
        im = torch.cat([im, torch.zeros(1, CMNIST_SHAPE[1], CMNIST_SHAPE[2])], 0)

        digit = int(label > 4)
        y = (digit * (1 - self.label_flip_mask[index]) +
             (1 - digit) * (self.label_flip_mask[index]))
        color = (y * (1 - self.color_flip_mask[index]) +
                 (1 - y) * (self.color_flip_mask[index]))

        if color == 1:
            im = torch.roll(im, 1, 0)
        return {'x': im, 'y': y.long(), 'e': self.envs[index], 'c': color, 'd': torch.LongTensor([digit]).squeeze()}

    def __len__(self):
        return len(self.dataset)


# Dinamically sample label, environment and color for a given digit
class BuildDynamicCMNIST(DatasetTransform):
    def __init__(self, p, **params):
        super(BuildDynamicCMNIST, self).__init__(**params)
        assert isinstance(p, torch.FloatTensor) or isinstance(p, torch.cuda.FloatTensor)

        # tensor expressing digit x label x environment x color
        assert len(p.shape) == 4
        assert (p < 0).long().sum() == 0
        for i in range(p.shape[0]):
            assert np.abs(p[i].sum().item() - 1.) <= 1e-6

        self.p = p

    def sample_yec(self, d):

        flat_p = self.p[d].view(-1)

        sample = Categorical(probs=flat_p).sample()
        c = sample % self.p.shape[-1]
        sample = sample // self.p.shape[-1]
        e = sample % self.p.shape[-2]
        y = sample // self.p.shape[-2]
        return y, e, c

    def __getitem__(self, index):
        data = self.dataset[index]
        im = data['x']
        d = data['y']

        # Binarize the digits
        d = torch.LongTensor([int(d > 4)]).squeeze()

        if self.p.device != im.device:
            self.p = self.p.to(im.device)

        # Sample label, environment and color conditioned on the digit
        y, e, c = self.sample_yec(d)

        x = torch.cat([im, im * 0], 0)
        if c == 1:
            x = torch.roll(x, 1, 0)

        return {'x': x, 'y': y, 'd': d, 'c': c, 'e': e}


class DynamicCMNIST(Dataset):
    def __init__(self, path, split, cond_dist_file, data_root='.', device='cuda'):
        super(DynamicCMNIST, self).__init__()

        dataset = MNIST(path=path, split=split, data_root=data_root)
        dataset = MoveToCache(instance=dataset, device=device)
        # Load a tensor representing p(yec|d)
        p = torch.load(cond_dist_file)
        self.dataset = BuildDynamicCMNIST(instance=dataset, p=p)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
