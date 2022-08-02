import math
from functools import partial
from typing import List, Optional, Tuple

import networkx as nx
import torch
from torch import FloatTensor, LongTensor, Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, to_networkx
from tqdm.auto import trange


class MCTSNode(object):
    """A node in a Monte Carlo Tree Search representing a subgraph."""

    def __init__(self,
                 coalition: Tuple[int, ...],
                 data: Data,
                 ori_graph: nx.Graph,
                 c_puct: float,
                 W: float = 0,
                 N: int = 0,
                 P: float = 0) -> None:
        """Initializes the MCTSNode object.

        :param coalition: A tuple of the nodes in the subgraph represented by this MCTSNode.
        :param data: The full graph.
        :param ori_graph: The original graph in NetworkX format.
        :param W: The sum of the node value.
        :param N: The number of times of arrival at this node.
        :param P: The property score (reward) of this node.
        :param c_puct: The hyperparameter that encourages exploration.
        """
        self.coalition = coalition
        self.data = data
        self.ori_graph = ori_graph
        self.c_puct = c_puct
        self.W = W
        self.N = N
        self.P = P
        self.children: List[MCTSNode] = []

    def Q(self) -> float:
        """Value that encourages exploitation of nodes with high reward."""
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, n: int) -> float:
        """Value that encourages exploration of nodes with few visits."""
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)

    @property
    def size(self) -> int:
        """Returns the number of nodes in the subgraph represented by the MCTSNode."""
        return len(self.coalition)

def gnn_score(coalition: Tuple[int, ...], data: Data, model: torch.nn.Module) -> float:
    """Computes the GNN score of the subgraph with the selected coalition of nodes.

    :param coalition: A list of indices of the nodes to retain in the induced subgraph.
    :param data: A data object containing the full graph.
    :param model: The GNN model to use to compute the score.
    :return: The score of the GNN model applied to the subgraph induced by the provided coalition of nodes.
    """
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    node_mask[list(coalition)] = 1

    row, col = data.edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)

    mask_edge_index = data.edge_index[:, edge_mask]

    mask_data = Data(x=data.x, edge_index=mask_edge_index)
    mask_data = Batch.from_data_list([mask_data])

    logits = model(x=mask_data.x, edge_index=mask_data.edge_index, batch=mask_data.batch)
    score = torch.sigmoid(logits).item()

    return score


def get_best_mcts_node(results: List[MCTSNode], max_nodes: int) -> MCTSNode:
    """Get the MCTSNode with the highest reward (and smallest graph if tied) that has at most max_nodes nodes.

    :param results: A list of MCTSNodes.
    :param max_nodes: The maximum number of nodes allowed in a subgraph represented by an MCTSNode.
    :return: The MCTSNode with the highest reward (and smallest graph if tied) that has at most max_nodes nodes.
    """
    # Filter subgraphs to only include those with at most max_nodes nodes
    results = [result for result in results if result.size <= max_nodes]

    # Check that there exists a subgraph with at most max_nodes nodes
    if len(results) == 0:
        raise ValueError(f'All subgraphs have more than {max_nodes} nodes.')

    # Sort subgraphs by size in case of a tie (max picks the first one it sees, so the smaller one)
    results = sorted(results, key=lambda result: result.size)

    # Find the subgraph with the highest reward and break ties by preferring a smaller graph
    best_result = max(results, key=lambda result: (result.P, -result.size))

    return best_result


class MCTS(object):
    """An object which runs Monte Carlo Tree Search to find optimal subgraphs of a graph."""

    def __init__(self,
                 x: FloatTensor,
                 edge_index: LongTensor,
                 model: torch.nn.Module,
                 num_hops: int,
                 n_rollout: int,
                 min_nodes: int,
                 c_puct: float,
                 num_expand_nodes: int,
                 high2low: bool,
                 verbose: bool) -> None:
        """Creates the Monte Carlo Tree Search (MCTS) object.

        :param x: Input node features.
        :param edge_index: The edge indices.
        :param model: The GNN model to explain.
        :param num_hops: The number of hops to extract the neighborhood of target node.
        :param n_rollout: The number of times to build the Monte Carlo tree.
        :param min_nodes: Number of graph nodes of the leaf node in the search tree.
        :param c_puct: The hyperparameter that encourages exploration.
        :param num_expand_nodes: The number of nodes to expand when extending the child nodes in the search tree.
        :param high2low: Whether to expand children nodes from high degree to low degree
                         when extending the child nodes in the search tree.
        """
        self.x = x
        self.edge_index = edge_index
        self.model = model
        self.num_hops = num_hops
        self.data = Data(x=self.x, edge_index=self.edge_index)
        self.graph = to_networkx(
            Data(x=self.x, edge_index=remove_self_loops(self.edge_index)[0]),
            to_undirected=True
        )
        self.data = Batch.from_data_list([self.data])
        self.num_nodes = self.graph.number_of_nodes()
        self.n_rollout = n_rollout
        self.min_nodes = min_nodes
        self.c_puct = c_puct
        self.num_expand_nodes = num_expand_nodes
        self.high2low = high2low

        self.root_coalition = tuple(range(self.num_nodes))
        self.MCTSNodeClass = partial(MCTSNode, data=self.data, ori_graph=self.graph, c_puct=self.c_puct)
        self.root = self.MCTSNodeClass(coalition=self.root_coalition)
        self.state_map = {self.root.coalition: self.root}
        self.verbose = verbose

    def mcts_rollout(self, tree_node: MCTSNode) -> float:
        """Performs a Monte Carlo Tree Search rollout.

        :param tree_node: An MCTSNode representing the root of the MCTS search.
        :return: The value (reward) of the rollout.
        """
        if len(tree_node.coalition) <= self.min_nodes:
            return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            # Maintain a set of all the coalitions added as children of the tree
            tree_children_coalitions = set()

            # Get subgraph induced by the tree
            tree_subgraph = self.graph.subgraph(tree_node.coalition)

            # Get nodes to try expanding
            all_nodes = sorted(
                tree_subgraph.nodes,
                key=lambda node: tree_subgraph.degree[node],
                reverse=self.high2low
            )
            all_nodes_set = set(all_nodes)

            expand_nodes = all_nodes[:self.num_expand_nodes]

            # For each node, prune it and get the remaining subgraph (only keep the largest connected component)
            for expand_node in expand_nodes:
                subgraph_coalition = all_nodes_set - {expand_node}

                subgraphs = (
                    self.graph.subgraph(connected_component)
                    for connected_component in nx.connected_components(self.graph.subgraph(subgraph_coalition))
                )

                subgraph = max(subgraphs, key=lambda subgraph: subgraph.number_of_nodes())

                new_coalition = tuple(sorted(subgraph.nodes()))

                # Check the state map and merge with an existing subgraph if available
                new_node = self.state_map.setdefault(new_coalition, self.MCTSNodeClass(coalition=new_coalition))

                # Add the subgraph to the children of the tree
                if new_coalition not in tree_children_coalitions:
                    tree_node.children.append(new_node)
                    tree_children_coalitions.add(new_coalition)

            # For each child in the tree, compute its reward using the GNN
            for child in tree_node.children:
                if child.P == 0:
                    child.P = gnn_score(coalition=child.coalition, data=child.data, model=self.model)

        # Select the best child node and unroll it
        sum_count = sum(child.N for child in tree_node.children)
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(n=sum_count))
        v = self.mcts_rollout(tree_node=selected_node)
        selected_node.W += v
        selected_node.N += 1

        return v

    def run_mcts(self) -> List[MCTSNode]:
        """Runs the Monte Carlo Tree search.

        :return: A list of MCTSNode objects representing subgraph explanations sorted from highest to
                 smallest reward (for ties, the smaller graph is first).
        """
        if self.verbose:
            for _ in trange(self.n_rollout):
                self.mcts_rollout(tree_node=self.root)
        else:
            for _ in range(self.n_rollout):
                self.mcts_rollout(tree_node=self.root)

        explanations = [node for _, node in self.state_map.items()]

        # Sort by highest reward and break ties by preferring a smaller graph
        explanations = sorted(explanations, key=lambda x: (x.P, -x.size), reverse=True)

        return explanations

class SubgraphX(object):
    """An object which contains methods to explain a GNN's prediction on a graph in terms of subgraphs."""

    def __init__(self,
                 model: torch.nn.Module,
                 num_hops: Optional[int] = None,
                 n_rollout: int = 20,
                 min_nodes: int = 10,
                 c_puct: float = 10.0,
                 num_expand_nodes: int = 14,
                 high2low: bool = False) -> None:
        """Initializes the SubgraphX object.

        :param model: The GNN model to explain.
        :param num_hops: The number of hops to extract the neighborhood of target node.
                         If None, uses the number of MessagePassing layers in the model.
        :param n_rollout: The number of times to build the Monte Carlo tree.
        :param min_nodes: Number of graph nodes of the leaf node in the search tree.
        :param c_puct: The hyperparameter that encourages exploration.
        :param num_expand_nodes: The number of nodes to expand when extending the child nodes in the search tree.
        :param high2low: Whether to expand children nodes from high degree to low degree
                         when extending the child nodes in the search tree.
        """
        self.model = model
        self.model.eval()
        self.num_hops = num_hops

        if self.num_hops is None:
            self.num_hops = sum(isinstance(module, MessagePassing) for module in self.model.modules())

        # MCTS hyperparameters
        self.n_rollout = n_rollout
        self.min_nodes = min_nodes
        self.c_puct = c_puct
        self.num_expand_nodes = num_expand_nodes
        self.high2low = high2low

    def explain(self, x: Tensor, edge_index: Tensor, max_nodes: int, verbose=True) -> MCTSNode:
        """Explain the GNN behavior for the graph using the SubgraphX method.

        :param x: Node feature matrix with shape [num_nodes, dim_node_feature].
        :param edge_index: Graph connectivity in COO format with shape [2, num_edges].
        :param max_nodes: The maximum number of nodes in the final explanation results.
        :param verbose: silent flag
        :return: The MCTSNode corresponding to the subgraph that best explains the model's prediction on the graph
                 (the smallest graph that has the highest reward such that the subgraph has at most max_nodes nodes).
        """
        # Create an MCTS object with the provided graph
        mcts = MCTS(
            x=x,
            edge_index=edge_index,
            model=self.model,
            num_hops=self.num_hops,
            n_rollout=self.n_rollout,
            min_nodes=self.min_nodes,
            c_puct=self.c_puct,
            num_expand_nodes=self.num_expand_nodes,
            high2low=self.high2low,
            verbose=verbose
        )

        # Run the MCTS search
        mcts_nodes = mcts.run_mcts()

        # Select the MCTSNode that contains the smallest subgraph that has the highest reward
        # such that the subgraph has at most max_nodes nodes
        best_mcts_node = get_best_mcts_node(mcts_nodes, max_nodes=max_nodes)

        return best_mcts_node