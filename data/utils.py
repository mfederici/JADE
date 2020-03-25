import torch
from torch.utils.data import Dataset


class CacheMemoryDataset(Dataset):
    def __init__(self, dataset, device):
        super(CacheMemoryDataset, self).__init__()
        self.dataset = dataset
        self.device = device
        entry = dataset[0]

        self.cache = {}
        self.is_cached = torch.zeros(len(dataset)).bool()
        for name, element in entry.items():
            shape = element.shape
            self.cache[name] = (element.unsqueeze(0).repeat(len(dataset), *([1] * len(shape))))

        for name in self.cache:
            self.cache[name] = self.cache[name].to(device)

    def __getitem__(self, index):
        if self.is_cached[index]:
            return {name: element[index] for name, element in self.cache.items()}
        else:
            values = self.dataset[index]
            for name, value in values.items():
                self.cache[name][index] = value.to(self.device)
            self.is_cached[index] = True
            return values

    def __len__(self):
        return len(self.dataset)
