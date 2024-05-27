# Taken and adapted from: https://github.com/gasmichel/PathNNs_expressive/blob/main/synthetic/main.py

import argparse
import os

import numpy as np
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from src.synthetic.datasets.EXPdataset import EXPDataset
from src.synthetic.datasets.SR25dataset import SR25Dataset
from src.synthetic.model.compute_paths import ComputePaths
from src.synthetic.model.path_gnn import PathGNN

SR25_NAMES = [
    'sr16622',
    'sr251256',
    'sr261034',
    'sr281264',
    'sr291467',
    'sr361446',
    'sr401224'
]

def main_iso(dataset, model_config, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = PathGNN(**model_config).to(device)
    res = []
    for i in [69, 42, 666, 13, 420]:
        torch.manual_seed(i)
        model.reset_parameters()
        embeddings, lst = [], []
        model.eval()
        for data in tqdm(loader):
            pre = model(data.to(device))
            embeddings.append(pre.detach().cpu())
        failure_rate = _isomorphism(torch.cat(embeddings, 0).detach().cpu().numpy())
        print(f'Failure Rate: {failure_rate}')
        res.append(failure_rate)

    return np.mean(np.asarray(res))


def _isomorphism(preds, eps=1e-5, p=2):
    # NB: here we return the failure percentage... the smaller the better!
    assert preds is not None
    # assert preds.dtype == np.float64
    preds = torch.tensor(preds, dtype=torch.float64)
    mm = torch.pdist(preds, p=p)
    wrong = (mm < eps).sum().item()
    metric = wrong / mm.shape[0]
    return metric


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset')
    parser.add_argument('-l', '--length')
    parser.add_argument('-m', '--mark_neighbors', action='store_true')

    args = parser.parse_args()

    dataset = args.dataset
    length = int(args.length)

    mark_neighbors = False
    if args.mark_neighbors:
        mark_neighbors = True

    model_config = {'lstm_in': 1, 'lstm_out': 16, 'lstm_layers': 2, 'path_layers': 1, 'mlp_layers': 2,
                    'num_out': 0, 'node_encoder': False, 'readout_agg': 'sum', 'path_agg': 'sum', 'predict': False,
                    'mark_neighbors': mark_neighbors}

    device = 'cpu'
    batch_size = 1

    if not os.path.exists('results'):
        os.makedirs('results')

    if dataset == 'SR25':

        with open(f'results/{dataset}-path-{length}.txt', 'w') as f:
            f.write('dataset,mark_neighbors,path_length,mean_failure_rate\n')

        # go through all SR graphs:
        for dataset_name in SR25_NAMES:
            print(f'Using dataset {dataset_name}, path length {length} and marking neighbors is set to {mark_neighbors}')

            data = SR25Dataset(name=dataset_name, length=length, pre_transform=ComputePaths(length=length))

            res = main_iso(data, model_config, device=device, batch_size=batch_size)

            print(f'results/{dataset}-path-{length}.txt')
            with open(f'results/{dataset}-path-{length}.txt', 'a') as f:
                f.write(f'{dataset_name},{mark_neighbors},{length},{res}\n')

    elif dataset == 'EXP':

        with open(f'results/{dataset}.txt', 'w') as f:
            f.write('dataset,mark_neighbors,path_length,mean_failure_rate\n')

        print(f'Using dataset {dataset}, path length {length} and marking neighbors is set to {mark_neighbors}')
        data = EXPDataset(name='EXP', length=length, pre_transform=ComputePaths(length=length))
        res = main_iso(data, model_config, device=device, batch_size=batch_size)
        with open(f'results/{dataset}.txt', 'a') as f:
            f.write(f'{dataset},{mark_neighbors},{length},{res}\n')

    else:
        raise NotImplementedError("Only dataset choices are SR25 and EXP")


if __name__ == "__main__":
    main()
