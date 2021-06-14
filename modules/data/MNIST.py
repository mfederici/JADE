from torchvision.datasets import MNIST as MNIST_base

class MNIST(MNIST_base):
    def __init__(self, **params):
        super(MNIST, self).__init__(**params)
        self.to_tensor = ToTensor()

    def __getitem__(self, item):
        x, y = super(MNIST, self).__getitem__(item)
        if not isinstance(x, torch.tensor):
            x = self.to_tensor(x)
        return {'x': x, 'y': y}
