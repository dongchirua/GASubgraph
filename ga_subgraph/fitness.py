from typing import Callable, Tuple, List
from functools import partial
import torch
from torch_geometric.data import Data, Batch


def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
    """ subgraph building through masking the unselected nodes with zero features """
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index


def graph_build_split(X, edge_index, node_mask: torch.Tensor):
    """ subgraph building through spliting the selected nodes from the original graph """
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return X, ret_edge_index


def get_graph_build_func(build_method):
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    else:
        raise NotImplementedError


def gnn_score(selected_nodes: List[int], origin_graph: Data, classifier: Callable,
              subgraph_building_method='zero_filling') -> float:
    """ the value of subgraph with selected nodes """
    num_nodes = origin_graph.num_nodes
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    mask = torch.zeros(num_nodes).type(torch.float32).to(origin_graph.x.device)
    mask[selected_nodes] = 1.0
    ret_x, ret_edge_index = subgraph_build_func(origin_graph.x, origin_graph.edge_index, mask)
    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    score = classifier(mask_data)
    # get the score of predicted class for graph or specific node idx
    return score


def get_fitness_func(score_method: str, classifier=None, subgraph_building_method='zero_filling', **kwargs):
    """ Function factory to generate a method measure how quality of a individual
    Args:
        score_method: method to use
        gnn_model:  a blackbox algorithm
        subgraph_building_method: way to construct a suggraph
    """
    if score_method.lower() == 'gnn_score':
        return partial(gnn_score,
                       classifier=classifier,
                       subgraph_building_method=subgraph_building_method,
                       **kwargs)
    else:
        raise NotImplementedError


def wrap_classifier(classifier, model, device):
    """ This function setup environment for classifier
    """
    return partial(classifier, model=model, device=device)


def classifier(data, model, device) -> float:
    """ Wraper for any classification method
    """
    out = model(x=data.x.to(device), edge_index=data.edge_index.to(device), batch=None)
    prod = torch.sigmoid(out)
    return prod.item()


def get_selected_nodes(edge_index: torch.Tensor,
                       edge_mask: torch.Tensor,
                       top_k: int):
    """
    Get the nodes of the top k-edge subgraph.

    Args:
        edge_index (torch.Tensor, [2 x m]): edge index of the graph
        edge_mask (torch.Tensor, [m]): edge mask of the graph
        top_k (int): number of edges to include in the subgraph

    Returns:
        selected_nodes: list of the indices of the selected nodes
    """
    sorted_edge_weights = edge_mask.reshape(-1).sort(descending=True)
    threshold = float(sorted_edge_weights.values[min(top_k, edge_mask.shape[0] - 1)])
    hard_mask = edge_mask > threshold
    edge_idx_list = torch.where(hard_mask == 1)[0]
    selected_nodes = []
    for edge_idx in edge_idx_list:
        selected_nodes += [edge_index[0][edge_idx].item(), edge_index[1][edge_idx].item()]
    selected_nodes = list(set(selected_nodes))
    return selected_nodes


class GraphEvaluation(object):
    """
    """

    def __init__(self, K: int, blackbox_model: torch.nn.Module, classifier: Callable, device: torch.device,
                 origin_graph: Data, score_method: str = 'gnn_score', subgraph_building_method='zero_filling'):
        self.K = K
        self.num_nodes = origin_graph.num_nodes
        self.wraped_classifer = wrap_classifier(classifier, blackbox_model, device)
        self.fitness_func = get_fitness_func(score_method, self.wraped_classifer, origin_graph=origin_graph,
                                             subgraph_building_method=subgraph_building_method)
        self.origin_fitness_value = self.fitness_func(
            selected_nodes=[i for i in range(self.num_nodes)])  # select all nodes

    def __call__(self, chromosome) -> Tuple:
        """ A value of a subgraph is scored by how close gnn output from it and original graph.
        The final value takk size of subgraph to consideration.
        We are going to minize this function
        """
        coalition = [i for i, v in enumerate(chromosome) if v == 1]
        prob = self.fitness_func(selected_nodes=coalition)
        return abs(prob - self.origin_fitness_value) + (abs(len(coalition) - self.K)),
