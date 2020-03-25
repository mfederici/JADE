class Evaluation:
    def __init__(self, model, datasets, device, evaluate_every=1):
        self.model = model
        self.datasets = datasets
        self.evaluate_every = evaluate_every
        self.device = device

    def evaluate(self):
        raise NotImplemented()
