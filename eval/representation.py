import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from eval.base import Evaluation
from eval.utils import EmbeddedDataset, build_matrix


class RepresetationVisualiaztion(Evaluation):
    def __init__(self, dataset, model='encoder', downsample=None, **params):
        super(RepresetationVisualiaztion, self).__init__(**params)
        self.dataset = self.datasets[dataset]
        self.model = getattr(self.trainer, model)
        self.project = PCA(n_components=2)

        self.select_ids = np.arange(len(dataset))
        if not (downsample is None):
            self.select_ids = np.random.choice(self.select_ids, downsample)

    def evaluate(self):
        embedded_dataset = EmbeddedDataset(self.dataset, self.model, device=self.trainer.get_device())
        z, y = build_matrix(embedded_dataset)
        if z.shape[1] > 2:
            z = self.project.fit_transform(z)

        fig, ax = plt.subplots()
        unique_y = set(y)
        for label in unique_y:
            z_selected = z[y == label]
            ax.plot(z_selected[:, 0], z_selected[:, 1], 'o', label=label, alpha=0.05)

        return {
            'type': 'figure',
            'value': fig
        }


