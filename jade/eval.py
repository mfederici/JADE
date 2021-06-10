class Evaluation:
    def __init__(self, trainer, datasets, evaluate_every=1, **params):
        self.trainer = trainer
        self.datasets = datasets
        self.evaluate_every = evaluate_every

        self.initialize(**params)

    def initialize(self, **params):
        raise NotImplemented()

    def evaluate(self):
        raise NotImplemented()
