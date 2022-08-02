import os.path
from abc import ABC
from collections.abc import Callable
from typing import Any
import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data

from vulexp.data_models.helpers import convert_single_graph, from_networkx


class CustomSet(Dataset, ABC):
    def __init__(self, data: pd.DataFrame, absolute_path=None, to_undirected=False):
        super().__init__()
        self.data = data.copy(deep=True)
        self.data.index = range(0, len(data))
        self.absolute_path = absolute_path
        self.to_undirected = to_undirected

    def len(self) -> int:
        return len(self.data)

    def get(self, idx: int) -> Data:
        new_g, new_g_label = self.get_raw(idx, convert_single_graph)
        tmp_data = from_networkx(new_g, group_node_attrs=['feat'])
        tmp_data.y = torch.LongTensor([new_g_label])
        tmp_data.x = tmp_data.x.float()
        return tmp_data

    def get_path(self, idx: int):
        candidate = self.data.iloc[idx]['path']
        return os.path.join(self.absolute_path, candidate)

    def get_raw(self, idx: int,
                is_transform: Callable[nx.MultiDiGraph, nx.DiGraph] = None,
                check: bool = False) -> tuple[nx.DiGraph, Any]:
        # todo: check style this function
        candidate = self.data.iloc[idx]['path']
        if self.absolute_path is not None:
            raw_g = nx.read_gpickle(os.path.join(self.absolute_path, candidate))
        else:
            raw_g = nx.read_gpickle(candidate)
        if is_transform is not None:
            raw_g = convert_single_graph(raw_g, self.to_undirected)
        if check:
            g = self.get(idx)
            assert g.num_nodes == raw_g.number_of_nodes(), 'raw data and data are not identical'
        return raw_g, raw_g.graph['label']

    @staticmethod
    def check(g: Data, raw_g: nx.MultiDiGraph) -> None:
        assert g.num_nodes == raw_g.number_of_nodes(), 'raw data and data are not identical'
        print(g.num_nodes, raw_g.number_of_nodes())
