import random
import networkx as nx
from .individual import Individual
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def mutFlipBit(individual, indpb, origin_graph: Data):
    n_nodes = origin_graph.num_nodes
    for i in range(n_nodes):
        if random.random() < indpb:
            individual[i] = type(individual[i])(not individual[i])

    return individual,


def mutAddNeighbor(individual: Individual, origin_graph: Data, indpb):
    nodes = individual.get_nodes()
    g = to_networkx(origin_graph, to_undirected=not origin_graph.is_directed())
    neighbor = set()
    for i in nodes:
        neighbor.update(g.neighbors(i))
    neighbor = neighbor - set(nodes)
    for i in list(neighbor):
        individual[i] = type(individual[i])(not individual[i])

    return individual,


def mutRemoveBit(individual, origin_graph: Data, **kwargs):
    nodes = individual.get_nodes()
    if len(nodes) < 2:
        return individual,
    sel = random.choice(nodes)
    sel_ind = nodes.index(sel)
    n_nodes = nodes[:sel_ind] + nodes[sel_ind+1:]  # remove sel
    G = to_networkx(origin_graph, to_undirected=not origin_graph.is_directed())
    sub_graph = G.subgraph(n_nodes)
    if origin_graph.is_directed():
        largest_cc = max(nx.weakly_connected_components(sub_graph), key=len)
    else:
        largest_cc = max(nx.connected_components(sub_graph), key=len)

    # remove node not in largest connected component
    for i in nodes:
        if i not in largest_cc:
            individual[i] = type(individual[i])(not individual[i])

    return individual,


def mutate(individual: Individual, origin_graph: Data, indpb):
    op_choice = random.choice([mutFlipBit, mutRemoveBit, mutAddNeighbor])
    return op_choice(**{'individual': individual, 'indpb': indpb, 'origin_graph': origin_graph})
