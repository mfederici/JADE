import numpy as np
import torch
from torch.distributions import Distribution
from torch.utils.data import Subset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from data.transforms.dataset import EmbeddedDataset

from data.transforms.dataset import Apply, DatasetTransform


# class EmbeddedDataset(Apply):
#
#     def __init__(self, encoder, device='cpu', f_in='x', f_out='z', **params):
#         encoder = encoder.to(device)
#         encoder.eval()
#         if not isinstance(f_in, list):
#             f_in = [f_in]
#         assert isinstance(f_out, str)
#
#         def f(data):
#             in_data = {}
#             for name, value in data.items():
#                 if name in f_in:
#                     in_data[name] = data[name].to(device).unsqueeze(0)
#             with torch.no_grad():
#                 z = encoder(in_data).detach().squeeze(0)
#             out_data = in_data
#             out_data.update({f_out: z})
#             return out_data
#
#         super(EmbeddedDataset, self).__init__(f=f, f_in=f_in, f_out=[f_out]+f_in, **params)



def split(dataset, size, split_type):
    if split_type == 'Random':
        data_split, _ = torch.utils.data.random_split(dataset, [size, len(dataset) - size])
    elif split_type == 'Balanced':
        class_ids = {}
        for idx, (_, y) in enumerate(dataset):
            if isinstance(y, torch.Tensor):
                y = y.item()
            if y not in class_ids:
                class_ids[y] = []
            class_ids[y].append(idx)

        ids_per_class = size // len(class_ids)

        selected_ids = []

        for ids in class_ids.values():
            selected_ids += list(np.random.choice(ids, min(ids_per_class, len(ids)), replace=False))
        data_split = Subset(dataset, selected_ids)

    return data_split


def build_matrix(dataset, keys):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    matrices = {k: [] for k in keys}

    for data in data_loader:
        for k in keys:
            matrices[k].append(data[k])
    for k in keys:
        matrices[k] = torch.cat(matrices[k], 0).to('cpu')

    return matrices


def evaluate(encoder, train_on, test_on, device):
    embedded_train = EmbeddedDataset(train_on, encoder, device=device)
    embedded_test = EmbeddedDataset(test_on, encoder, device=device)
    return train_and_evaluate_linear_model(embedded_train, embedded_test)


def train_and_evaluate_linear_model_from_matrices(x_train, y_train, solver='saga', multi_class='multinomial', tol=.1, C=10):
    model = LogisticRegression(solver=solver, multi_class=multi_class, tol=tol, C=C)
    model.fit(x_train, y_train)
    return model


def train_and_evaluate_linear_model(train_set, test_set, solver='saga', multi_class='multinomial', tol=.1, C=10):
    train_data = build_matrix(train_set, ['z', 'y'])
    test_data = build_matrix(test_set, ['z', 'y'])

    scaler = MinMaxScaler()

    train_data['z'] = scaler.fit_transform(train_data['z'])
    test_data['z'] = scaler.transform(test_data['z'])

    model = LogisticRegression(solver=solver, multi_class=multi_class, tol=tol, C=C)
    try:
        model.fit(train_data['z'], train_data['y'])

        test_accuracy = model.score(test_data['z'], test_data['y'])
    except:
        test_accuracy = 0
    return test_accuracy
