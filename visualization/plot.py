from typing import Optional, Set
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data


def visualize_subgraph(graph: nx.Graph,
                       node_set: Optional[Set[int]] = None,
                       edge_set: Optional[Set[int]] = None,
                       title: Optional[str] = None) -> None:
    """Visualizes a subgraph explanation for a graph from the MUTAG dataset.

    Note: Only provide subgraph_node_set or subgraph_edge_set, not both.

    Adapted from https://github.com/divelab/DIG/blob/dig/dig/xgraph/method/subgraphx.py

    :param graph: A NetworkX graph object representing the full graph.
    :param node_set: A set of nodes that induces a subgraph.
    :param edge_set: A set of edges that induces a subgraph.
    :param title: Optional title for the plot.
    """
    if node_set is None:
        node_set = set(graph.nodes())

    if edge_set is None:
        edge_set = {(n_from, n_to) for (n_from, n_to) in graph.edges() if n_from in node_set and n_to in node_set}

    node_idxs = {node: (0 if node in node_set else 1) for node, node_x in graph.nodes(data='x')}
    node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
    colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

    pos = nx.kamada_kawai_layout(graph)

    nx.draw_networkx_nodes(G=graph, pos=pos, nodelist=list(graph.nodes()), node_color=colors, node_size=300)
    nx.draw_networkx_edges(G=graph, pos=pos, width=1, edge_color='gray', arrows=False)
    nx.draw_networkx_edges(G=graph, pos=pos, edgelist=list(edge_set), width=2, edge_color='black', arrows=False)
    nx.draw_networkx_labels(G=graph, pos=pos)

    if title is not None:
        plt.title(title)

    plt.axis('off')
    plt.show()
    plt.close()


def visualize_explanation(sample: Data,
                          selected_nodes: list = None, selected_edges: list = None) -> None:
    """Visualizes the explanations."""
    graph = to_networkx(sample, to_undirected=sample.is_directed())

    visualize_subgraph(
        graph=graph,
        node_set=selected_nodes,
        edge_set=selected_edges,
        title=f'Explanation'
    )
