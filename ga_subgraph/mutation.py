import random
from .individual import Individual
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def mutFlipBit(individual, indpb, **kwargs):
    """Flip the value of the attributes of the input individual and return the
    mutant. The *individual* is expected to be a :term:`sequence` and the values of the
    attributes shall stay valid after the ``not`` operator is called on them.
    The *indpb* argument is the probability of each attribute to be
    flipped. This mutation is usually applied on boolean individuals.

    :param individual: Individual to be mutated.
    :param indpb: Independent probability for each attribute to be flipped.
    :returns: A tuple of one individual.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i in range(len(individual)):
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
        if random.random() < indpb:
            individual[i] = type(individual[i])(not individual[i])

    return individual,


def mutRemoveBit(individual, indpb, **kwargs):
    """Remove a bit of the input individual and return the mutant. The
    *individual* is expected to be a :term:`sequence` and the values of the
    attributes shall stay valid after the ``not`` operator is called on them.
    The *indpb* argument is the probability of each attribute to be
    removed. This mutation is usually applied on boolean individuals.

    :param individual: Individual to be mutated.
    :param indpb: Independent probability for each attribute to be removed.
    :returns: A tuple of one individual.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    nodes = individual.get_nodes()
    for i in nodes:
        if random.random() < indpb:
            individual[i] = type(individual[i])(not individual[i])

    return individual,


def mutate(individual: Individual, origin_graph: Data, indpb):
    op_choice = random.choice([mutFlipBit, mutAddNeighbor, mutRemoveBit])
    return op_choice(**{'individual': individual, 'indpb': indpb, 'origin_graph': origin_graph})
