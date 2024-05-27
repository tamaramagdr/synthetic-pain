from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import scatter


class PathConv(MessagePassing):
    def __init__(self, rnn: Callable, mark_neighbors=False, **kwargs):
        super().__init__(**kwargs)
        self.rnn = rnn
        self.mark_neighbors = mark_neighbors
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.rnn)

    def forward(self, path_index: Tensor, mask_index: Tensor, path_lengths: Tensor, path_agg: str, neighbor_index: Optional[Tensor], x: Tensor) -> Tensor:

        path_features = x[path_index[:, :]]

        if self.mark_neighbors:
            path_neighbors = neighbor_index.T[mask_index]
            neighbor_mask = torch.zeros(path_index.shape)

            for i in range(path_index.T.shape[0]):
                neighbor_mask.T[i, :] = torch.torch.isin(path_index.T[i, :], path_neighbors[i])

            path_features[neighbor_mask.bool(), :] = torch.tensor(2.)  # add value of 1 to each neighbor

        packed_tensor = pack_padded_sequence(path_features, path_lengths.cpu(), batch_first=False, enforce_sorted=False).float()
        _, (hidden_states, _) = self.rnn(packed_tensor)
        out = scatter(hidden_states[-1], mask_index, dim=0, reduce=path_agg)
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.rnn})'

