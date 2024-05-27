import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import aggr

from src.synthetic.model.path_conv import PathConv
from src.synthetic.model.utils import PathSequential, get_node_encoder


class PathGNN(torch.nn.Module):
    def __init__(self, lstm_in, lstm_out, lstm_layers, path_layers, mlp_layers, num_out, node_encoder=False,
                 readout_agg='sum', path_agg='sum', predict=True, mark_neighbors=False):
        """
        :param lstm_in: size of initial features
        :param lstm_out: hidden dimension of path embeddings
        :param lstm_layers: number of layers in LSTM
        :param path_layers: number of path convolution layers
        :param mlp_layers: number of layers in MLP used for classification/regression
        :param num_out: number of nodes in final MLP depending on type of task
        :param node_encoder: whether to use node encoding prior to path convolution
        :param readout_agg: type of aggregation function for readout (currently supports 'mean' and 'sum')
        :param path_agg: type of aggregation function for path embeddings
        """
        super().__init__()

        self.path_agg = path_agg
        self.predict = predict
        self.encoder = None

        if node_encoder:
            self.encoder = get_node_encoder(lstm_in, lstm_out)

        self.path_conv, self.mlp = PathSequential(), nn.Sequential()

        for conv_layer in range(path_layers):  # dynamically add path layers
            if conv_layer == 0 and lstm_in != lstm_out and not node_encoder:
                self.path_conv.add_module(f"path_conv{conv_layer}",
                                          PathConv(nn.LSTM(lstm_in, lstm_out, lstm_layers, batch_first=True),
                                                   mark_neighbors=mark_neighbors))
            else:
                self.path_conv.add_module(f"path_conv{conv_layer}",
                                          PathConv(nn.LSTM(lstm_out, lstm_out, lstm_layers, batch_first=True),
                                                   mark_neighbors=mark_neighbors))

        for mlp_layer in range(mlp_layers):
            if mlp_layer == mlp_layers - 1:
                self.mlp.add_module(f"linear{mlp_layer}", nn.Linear(lstm_out, num_out))
                self.mlp.add_module(f"relu{mlp_layer}", nn.ReLU())
            else:
                self.mlp.add_module(f"linear{mlp_layer}", nn.Linear(lstm_out, lstm_out))
                self.mlp.add_module(f"relu{mlp_layer}", nn.ReLU())

        if readout_agg == 'sum':
            self.readout = aggr.SumAggregation()
        else:
            self.readout = aggr.MeanAggregation()

    def reset_parameters(self):
        if self.encoder:
            for c in self.encoder.children():
                if hasattr(c, 'reset_parameters'):
                    c.reset_parameters()
        for path_conv in self.path_conv:
            path_conv.reset_parameters()
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data):
        path_index, mask_index, path_lengths, x, neighbor_index, batch = (data.path_index, data.mask_index, data.path_lengths, data.x, data.neighbor_index, data.batch)

        if self.encoder:
            x = self.encoder(x)

        x = self.path_conv([path_index, mask_index, path_lengths, self.path_agg, neighbor_index], x)
        x = F.normalize(x, p=2, dim=1)

        x = self.readout(x, batch)

        if self.predict:
            x = self.mlp(x)

        return x

