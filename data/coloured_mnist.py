import os
import torch
import numpy as np
from torch.distributions import Bernoulli
from torch.utils.data import ConcatDataset, random_split, Dataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

MNIST_TRAIN_EXAMPLES = 50000
CMNIST_SIZE = 28**2*2
CMNIST_SHAPE = [2,28,28]
CMNIST_N_CLASSES = 2
CMNIST_N_ENVS = 2

# class ColouredMNISTDataset(Dataset):
#     def __init__(self, base_mnist_dataset, color_flip, label_flip=0.25):
#         self.base_mnist_dataset = base_mnist_dataset
#         self.label_flip_mask = torch.distributions.Bernoulli(probs=label_flip).sample([len(base_mnist_dataset)])
#         self.color_flip_mask = torch.distributions.Bernoulli(probs=color_flip).sample([len(base_mnist_dataset)])
#
#     def __getitem__(self, index):
#         im, label = self.base_mnist_dataset[index]
#         im = torch.cat([im[:, ::2, ::2], torch.zeros(1, 14, 14)], 0)
#
#         label = int(label > 4)
#         label = (label * (1 - self.label_flip_mask[index]) +
#                  (1 - label) * (self.label_flip_mask[index]))
#         color = (label * (1 - self.color_flip_mask[index]) +
#                  (1 - label) * (self.color_flip_mask[index]))
#
#         if color == 1:
#             im = torch.roll(im, 1, 0)
#
#         return {'x': im, 'y': label}
#
#     def __len__(self):
#         return len(self.base_mnist_dataset)


class ColouredMNIST(Dataset):
    def __init__(self, path, environments, split, data_root='.'):
        super(ColouredMNIST, self).__init__()

        dataset = MNIST(os.path.join(data_root, path), download=True, train=split in ['train', 'valid'], transform=ToTensor())

        if split == 'train':
            dataset = Subset(dataset, range(MNIST_TRAIN_EXAMPLES))
        elif split == 'valid':
            dataset = Subset(dataset, range(MNIST_TRAIN_EXAMPLES, len(dataset)))
        elif not (split == 'test'):
            raise Exception('The possible splits are "train", "valid" and "test"')

        dataset_splits = random_split(dataset, [environment['size'] for environment in environments])
        color_flip_masks = []
        label_flip_masks = []
        envs = []

        for i, environment in enumerate(environments):
            label_flip_masks.append(Bernoulli(probs=environment['flip_label']).sample([len(dataset_splits[i])]))
            color_flip_masks.append(Bernoulli(probs=environment['flip_color']).sample([len(dataset_splits[i])]))
            envs.append(torch.zeros(len(dataset_splits[i]))+i)

        self.environments = environments
        self.dataset = ConcatDataset(dataset_splits)
        self.envs = torch.cat(envs, 0).long()
        self.label_flip_mask = torch.cat(label_flip_masks, 0)
        self.color_flip_mask = torch.cat(color_flip_masks, 0)

    def __getitem__(self, index):
        im, label = self.dataset[index]
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

