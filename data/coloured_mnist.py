import os
import torch
import numpy as np
from torch.utils.data import ConcatDataset, random_split, Dataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from data.utils import CacheMemoryDataset

MNIST_TRAIN_EXAMPLES = 50000


class ColouredMNISTDataset(Dataset):
    def __init__(self, base_mnist_dataset, color_flip, label_flip=0.25):
        self.base_mnist_dataset = base_mnist_dataset
        self.label_flip_mask = torch.distributions.Bernoulli(probs=label_flip).sample([len(base_mnist_dataset)])
        self.color_flip_mask = torch.distributions.Bernoulli(probs=color_flip).sample([len(base_mnist_dataset)])

    def __getitem__(self, index):
        im, label = self.base_mnist_dataset[index]
        im = torch.cat([im[:, ::2, ::2], torch.zeros(1, 14, 14)], 0)

        label = int(label > 4)
        label = (label * (1 - self.label_flip_mask[index]) +
                 (1 - label) * (self.label_flip_mask[index]))
        color = (label * (1 - self.color_flip_mask[index]) +
                 (1 - label) * (self.color_flip_mask[index]))

        if color == 1:
            im = torch.roll(im, 1, 0)

        return {'x': im, 'y': label}

    def __len__(self):
        return len(self.base_mnist_dataset)


class MultiEnvColouredMNIST(Dataset):
    def __init__(self, root, environments, split, cached=True, device='cpu', multiview=False):
        dataset = MNIST(root, download=True, train=split in ['train', 'valid'], transform=ToTensor())
        if split == 'train':
            dataset = Subset(dataset, range(MNIST_TRAIN_EXAMPLES))
        elif split == 'valid':
            dataset = Subset(dataset, range(MNIST_TRAIN_EXAMPLES, len(dataset)))
        elif not (split == 'test'):
            raise Exception('The possible splits are "train", "valid" and "test"')

        dataset_splits = random_split(dataset, [environment['size'] for environment in environments])
        datasets = []

        for i, environment in enumerate(environments):
            datasets.append(ColouredMNISTDataset(dataset_splits[i],
                                                 environment['flip_color'],
                                                 environment['flip_label'])
                            )
        self.environments = environments
        self.dataset = ConcatDataset(datasets)
        if cached:
            self.dataset = CacheMemoryDataset(self.dataset, device=device)

        if multiview:
            self.dataset = MultiViewColouredMNISTDataset(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class MultiViewColouredMNISTDataset(Dataset):
    def __init__(self, coloured_mnist_dataset):
        self.labels = np.array([d['y'].item() for d in coloured_mnist_dataset])
        self.dataset = coloured_mnist_dataset
        self.ids = np.arange(len(self.dataset))

    def __getitem__(self, index):
        entry = self.dataset[index]
        v2_index = np.random.choice(self.ids[self.labels == entry['y'].item()])
        v2 = self.dataset[v2_index]['x']

        return {'v1': entry['x'], 'v2': v2, 'y': entry['y']}

    def __len__(self):
        return len(self.dataset)