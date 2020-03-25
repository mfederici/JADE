import torch
from torch.utils.data import Dataset


class CacheMemoryDataset(Dataset):
    def __init__(self, dataset, device):
        super(CacheMemoryDataset, self).__init__()
        self.dataset = dataset
        self.device = device
        entry = dataset[0]
        if not isinstance(entry, tuple):
            entry = (entry,)

        self.cache = []
        self.is_cached = torch.zeros(len(dataset)).bool()
        for element in entry:
            shape = element.shape
            self.cache.append(element.unsqueeze(0).repeat(len(dataset), *([1] * len(shape))))

        for i in range(len(self.cache)):
            self.cache[i] = self.cache[i].to(device)

    def __getitem__(self, index):
        if self.is_cached[index]:
            return [element[index] for element in self.cache]
        else:
            values = self.dataset[index]
            values = [value.to(self.device) for value in values]
            for i, value in enumerate(values):
                self.cache[i][index] = value
            self.is_cached[index] = True
            return values

    def __len__(self):
        return len(self.dataset)
