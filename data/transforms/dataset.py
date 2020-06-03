import torch
import numpy as np

import torchvision.transforms as transform_module
import data.transforms.augmentations as augmentation_module
from torch.utils.data import Dataset


class DatasetTransform(Dataset):
    def __init__(self, instance):
        self.dataset = instance

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.dataset)


class Apply(DatasetTransform):
    def __init__(self, f, f_in, f_out, **params):
        super(Apply, self).__init__(**params)
        self.f = f

        if not isinstance(f_in, list):
            f_in = [f_in]
        if not isinstance(f_out, list):
            f_out = [f_out]
        self.f_in = f_in
        self.f_out = f_out

    def __getitem__(self, index):
        data = {k: v for k, v in self.dataset[index].items() if k not in self.f_in}
        data_in = {k: v for k, v in self.dataset[index].items() if k in self.f_in}
        data_out = self.f(data_in)
        data_out.update(data)
        return data_out


class Augment(Apply):
    def __init__(self, apply_to, aug_class, aug_params, **params):
        f_in = apply_to
        f_out = apply_to
        if hasattr(transform_module, aug_class):
            AugmentationClass = getattr(transform_module, aug_class)
        elif hasattr(augmentation_module, aug_class):
            AugmentationClass = getattr(augmentation_module, aug_class)
        else:
            raise Exception('Unknown augmentation %s' % aug_class)
        augment = AugmentationClass(**aug_params)

        def f(data):
            return {k: augment(data[k]) for k in apply_to}

        super(Augment, self).__init__(f=f, f_in=f_in, f_out=f_out, **params)


class MoveToCache(DatasetTransform):
    def __init__(self, device=None, **params):
        super(MoveToCache, self).__init__(**params)

        if device is None:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'

        self.device = device
        entry = self.dataset[0]

        self.cache = {}
        self.is_cached = torch.zeros(len(self.dataset)).bool()
        for name, element in entry.items():
            shape = element.shape
            self.cache[name] = (element.unsqueeze(0).repeat(len(self.dataset), *([1] * len(shape))))

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


class PairByValue(Apply):
    def __init__(self, instance, pair, by, out_names, **params):
        f_in = [pair, by]
        f_out = out_names + [by]

        values = np.array([d[by].item() for d in instance])
        ids = np.arange(len(instance))

        def f(data):
            v1 = data[pair]
            value = data[by].item()
            v2_index = np.random.choice(ids[values == value])
            v2 = self.dataset[v2_index][pair]

            return {out_names[0]: v1,
                    out_names[1]: v2,
                    by: value}

        super(PairByValue, self).__init__(instance=instance, f=f, f_in=f_in, f_out=f_out, **params)


class PairByLabel(PairByValue):
    def __init__(self, **params):
        super(PairByLabel, self).__init__(pair='x', by='y', out_names=['v1', 'v2'], **params)


class Copy(Apply):
    def __init__(self, copy, to, **params):
        f_in = [copy]
        f_out = to
        def f(data):
            return {k: data[copy] for k in to}

        super(Copy, self).__init__(f=f, f_in=f_in, f_out=f_out, **params)