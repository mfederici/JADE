import torch
from torch.distributions import Bernoulli


class ColorAugmentation:
    def __init__(self, flip_probability=0.5):
        self.flip_dist = Bernoulli(probs=flip_probability)

    def __call__(self, data):
        flip = self.flip_dist.sample()
        if flip > 0.5:
            data = torch.roll(data, 1, 0)
        return data
