from torch import nn
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU


class PathSequential(nn.Sequential):

    def forward(self, *inputs):
        path_index, mask_index, path_lengths, path_agg, neighbor_index = None, None, None, None, None

        for name, module in self._modules.items():
            if 'path' in name:
                if type(inputs) == tuple:
                    (path_index, mask_index, path_lengths, path_agg, neighbor_index), x = inputs
                    inputs = module(path_index, mask_index, path_lengths, path_agg, neighbor_index, x)
                else:
                    inputs = module(path_index, mask_index, path_lengths, path_agg, neighbor_index, inputs)
            else:
                inputs = module(inputs)
        return inputs


def get_node_encoder(in_dim, hidden_dim) -> Sequential:
    node_encoder = Sequential(Linear(in_dim, hidden_dim),
                              BatchNorm1d(hidden_dim),
                              ReLU(),
                              Linear(hidden_dim, hidden_dim),
                              BatchNorm1d(hidden_dim),
                              ReLU())
    return node_encoder