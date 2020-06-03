class Evaluation:
    def __init__(self, trainer, datasets, evaluate_every=1):
        self.trainer = trainer
        self.datasets = datasets
        self.evaluate_every = evaluate_every

    def evaluate(self):
        raise NotImplemented()
