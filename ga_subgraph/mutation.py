import random
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


def mutRemoveBit(individual, **kwargs):
    nodes = individual.get_nodes()
    sel = random.choice(nodes)
    sel_ind = nodes.index(sel)
    n1 = nodes[:sel_ind]
    n2 = nodes[sel_ind:]
    if len(n1) < len(n2):  # swap
        n1, n2 = n2, n1
    for i in n2:
        individual[i] = type(individual[i])(not individual[i])

    return individual,


def mutate(individual: Individual, origin_graph: Data, indpb):
    op_choice = random.choice([mutFlipBit, mutRemoveBit, mutAddNeighbor])
    return op_choice(**{'individual': individual, 'indpb': indpb, 'origin_graph': origin_graph})
