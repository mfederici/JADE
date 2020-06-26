import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from eval.base import Evaluation
from eval.utils import EmbeddedDataset, build_matrix

import seaborn as sns


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


class RepresentationDensityVisualization(Evaluation):
    def __init__(self, dataset, model='encoder', downsample=None, **params):
        super(RepresentationDensityVisualization, self).__init__(**params)
        self.dataset = self.datasets[dataset]
        self.model = getattr(self.trainer, model)

        self.select_ids = np.arange(len(dataset))
        if not (downsample is None):
            self.select_ids = np.random.choice(self.select_ids, downsample)

    def evaluate(self):
        sns.set_style('whitegrid')
        embedded_dataset = EmbeddedDataset(self.dataset, self.model, device=self.trainer.get_device())
        matrices = build_matrix(embedded_dataset, ['z', 'y', 'e'])
        splits = {}

        splits['00'] = matrices['z'][np.logical_and(matrices['y'] == 0, matrices['e'] == 0).bool()]
        splits['01'] = matrices['z'][np.logical_and(matrices['y'] == 0, matrices['e'] == 1).bool()]
        splits['10'] = matrices['z'][np.logical_and(matrices['y'] == 1, matrices['e'] == 0).bool()]
        splits['11'] = matrices['z'][np.logical_and(matrices['y'] == 1, matrices['e'] == 1).bool()]

        for name in splits:
            splits[name] = splits[name].to('cpu').numpy()

        fig, ax = plt.subplots(2, 2, figsize=(6, 6))

        sns.kdeplot(splits['00'][:, 0], splits['00'][:, 1], ax=ax[0, 0], cmap="Reds", shade=True, shade_lowest=False)
        sns.kdeplot(splits['01'][:, 0], splits['01'][:, 1], ax=ax[0, 1], cmap="Reds", shade=True, shade_lowest=False)
        sns.kdeplot(splits['10'][:, 0], splits['10'][:, 1], ax=ax[1, 0], cmap="Blues", shade=True, shade_lowest=False)
        sns.kdeplot(splits['11'][:, 0], splits['11'][:, 1], ax=ax[1, 1], cmap="Blues", shade=True, shade_lowest=False)

        for j in range(2):
            x_min = min(ax[j, 0].get_xlim()[0], ax[j, 1].get_xlim()[0])
            x_max = max(ax[j, 0].get_xlim()[1], ax[j, 1].get_xlim()[1])
            y_min = min(ax[j, 0].get_ylim()[0], ax[j, 1].get_ylim()[0])
            y_max = max(ax[j, 0].get_ylim()[1], ax[j, 1].get_ylim()[1])

            ax[j, 0].set_xlim(x_min, x_max)
            ax[j, 1].set_xlim(x_min, x_max)
            ax[j, 0].set_ylim(y_min, y_max)
            ax[j, 1].set_ylim(y_min, y_max)

        ax[0, 0].set_title('p(z|e=0,y=0)')
        ax[0, 1].set_title('p(z|e=1,y=0)')
        ax[1, 0].set_title('p(z|e=0,y=1)')
        ax[1, 1].set_title('p(z|e=1,y=1)')

        return {
            'type': 'figure',
            'value': fig
        }




