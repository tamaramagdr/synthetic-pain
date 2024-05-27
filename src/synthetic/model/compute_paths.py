import igraph
import networkx as nx
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx


@functional_transform('compute_paths')
class ComputePaths(BaseTransform):

    def __init__(self, length):
        self.length = length

    # just here to look at things with my tiny brain
    def create_dummy_graph(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(list(range(4)))
        g.add_edges_from([[0, 1], [0, 2], [1, 2], [2, 3]])
        return g

    def __call__(self, data: Data) -> Data:
        g = to_networkx(data).to_undirected()

        compute_paths = self.compute_paths_faster

        paths = []
        path_mask = []
        lengths = []
        neighbors = []

        for node in g.nodes():
            paths_node, path_mask_node, lengths_node, neighbors_node = compute_paths(node, g)
            paths.append(paths_node), path_mask.append(path_mask_node), lengths.append(lengths_node), neighbors.append(neighbors_node)

        path_mask = [x for y in path_mask for x in y]
        paths = [x for y in paths for x in y]
        lengths = [x for y in lengths for x in y]

        sorted_lists = sorted(zip(paths, lengths, path_mask), key=lambda x: len(x[0]), reverse=True)
        paths, lengths, path_mask = map(list, zip(*sorted_lists))

        # hack to pad first sequence to max length to make sure all graphs are padded to max length
        paths[0] = nn.ConstantPad1d((0, self.length + 1 - paths[0].shape[0]), -10)(paths[0])

        padded_tensor = pad_sequence(paths, batch_first=False, padding_value=-10)
        mask_tensor = torch.tensor(path_mask)

        data.path_index = padded_tensor
        data.path_lengths = torch.tensor(lengths)
        data.mask_index = mask_tensor

        neighbor_len = len(next(iter(neighbors)))  # check length of list element

        # only compute neighbors for SR25 = k-regular graph
        if all(len(x) == neighbor_len for x in neighbors):
            data.neighbor_index = torch.stack(neighbors).T
        else:
            data.neighbor_index = torch.tensor(0)

        return data

    def compute_paths_faster(self, node, graph):
        # thanks to tip from https://github.com/gasmichel/PathNNs_expressive/ to use igraph because it's much faster
        g = igraph.Graph.from_networkx(graph)

        lengths = []
        paths = []
        path_mask = []

        paths_s = [torch.tensor([node])]
        lengths.append(1)  # add path of length 0

        ig_paths = list(g.get_all_simple_paths(node, cutoff=self.length))
        paths_s += [torch.tensor(x) for x in ig_paths]
        lengths.extend([len(x) for x in ig_paths])
        paths += paths_s
        path_mask = path_mask + [node] * len(paths_s)

        return paths, path_mask, lengths, torch.tensor(g.neighbors(node))
