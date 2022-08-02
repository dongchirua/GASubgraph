from collections import defaultdict
from typing import Optional, Union, List
import numpy as np
import networkx as nx
import torch
import torch_geometric.data
from packaging import version


def convert_single_graph(multi_graph: nx.MultiDiGraph, to_undirected=False) -> Union[nx.DiGraph, nx.Graph]:
    """
    Convert from networkx.MultiDiGraph to networkx.DiGraph.
    Keep edges, and copy node feature
    :param to_undirected:
    :param multi_graph:
    :return:
    """
    single_graph = nx.DiGraph()
    single_graph.graph.update(multi_graph.graph)
    single_graph.add_nodes_from(multi_graph.nodes(data=True))
    for u, v, _ in multi_graph.edges(data=True):
        if single_graph.has_edge(u, v):
            pass
        else:
            single_graph.add_edge(u, v)
    if to_undirected:
        return single_graph.to_undirected()
    return single_graph


def from_networkx(G, group_node_attrs: Optional[Union[List[str], all]] = None,
                  group_edge_attrs: Optional[Union[List[str], all]] = None):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.
    """

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for key, value in data.items():
        try:
            if key in group_node_attrs:
                # https://github.com/pytorch/pytorch/issues/13918
                data[key] = torch.tensor(np.array(value))
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    if data.x is None:
        data.num_nodes = G.number_of_nodes()

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = [data[key] for key in group_node_attrs]
        xs = [x.view(-1, 1) if x.dim() <= 1 else x for x in xs]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        edge_attrs = [data[key] for key in group_edge_attrs]
        edge_attrs = [x.view(-1, 1) if x.dim() <= 1 else x for x in edge_attrs]
        data.edge_attr = torch.cat(edge_attrs, dim=-1)

    return data


def relabel_nodes(G):
    """
    Rearrange node label in G to [0, num_nodes]
    :param G:
    :return:
    """
    mapping = {}
    it = 0
    if version.parse(nx.__version__) < version.parse('2.0'):
        for n in G.nodes():
            mapping[n] = it
            it += 1
    else:
        for n in G.nodes:
            mapping[n] = it
            it += 1

    G.graph['relabel_map'] = mapping
    G = nx.relabel_nodes(G, mapping)
    return G
