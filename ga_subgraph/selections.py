import networkx as nx
from .individual import Individual
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from functools import wraps
from itertools import repeat
from deap.tools.constraint import DeltaPenalty

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


def feasible(individual: Individual, origin_graph) -> bool:
    """Feasibility function for the individual. Returns True if feasible False
    otherwise."""
    G = to_networkx(origin_graph, to_undirected=origin_graph.is_directed())
    sub_graph = G.subgraph(individual.get_nodes())
    if origin_graph.is_directed():
        components = [i for i in nx.weakly_connected_components(sub_graph)]
    else:
        sub_graph = sub_graph.to_undirected()
        components = [i for i in nx.connected_components(sub_graph)]
    if len(components) == 1:
        return True
    return False


class Penalty(DeltaPenalty):
    """This decorator returns penalized fitness for invalid individuals and the
    original fitness value for valid individuals.
    """

    def __init__(self, feasibility, delta):
        self.fbty_fct = feasibility
        if not isinstance(delta, Sequence):
            self.delta = repeat(delta)
        else:
            self.delta = delta

    def __call__(self, func):
        @wraps(func)
        def wrapper(individual, *args, **kwargs):
            if self.fbty_fct(individual):
                return func(individual, *args, **kwargs)

            weights = tuple(1 if w >= 0 else -1 for w in individual.fitness.weights)

            dists = tuple(0 for w in individual.fitness.weights)

            return tuple(d - w * dist for d, w, dist in zip(self.delta, weights, dists))

        return wrapper
