from eval.base import Evaluation
import torch.nn as nn
from utils.modules import StochasticLinear

class LinearAccuracyEvaluation(Evaluation):
    def __init__(self, train_on, model='encoder', in_size, out_size, n_iter=1000, batch_size=64, dist='Categorical',
                 sample=True, **params):
        super(LinearAccuracyEvaluation, self).__init__(**params)
        self.train_set = self.datasets[train_on]
        self.model = getattr(self.trainer, model)
        self.n_iter = n_iter
        self.batch_size = batch_size

        self.cd_estimator_net = nn.Sequential(
            nn.Linear(in_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            StochasticLinear(1024, out_size=out_size, dist=dist)
        )

        self.sample = sample


    def evaluate(self):
        # Color and digit estimation for logging purposes
        c = data['c'].view(-1, 1).float()
        d = data['d'].view(-1, 1).float()
        cd = torch.cat([c, d], 1)

        p_cd_given_z = self.cd_estimator_net(z.detach())
        cd_rec_loss = -p_cd_given_z.log_prob(cd).mean(0)
        c_rec_loss = cd_rec_loss[0]
        d_rec_loss = cd_rec_loss[1]
        return {
            'type': 'scalar',
            'value':
        }

