class Individual(object):
    """ 
        This class represents for a chromosome
    """

    def __init__(self, nodes):
        if type(nodes) is list:
            self.nodes = nodes
        else:
            self.nodes = [i for i in nodes]

    def __repr__(self):
        coalition = [str(i) for i, v in enumerate(self.nodes) if v == 1]
        return ' '.join(coalition)

    def __get__(self, instance, owner):
        return self.nodes

    def __set__(self, instance, value):
        self.nodes = value

    def __getitem__(self, item):
        return self.nodes[item]

    def __setitem__(self, key, value):
        self.nodes[key] = value

    def __len__(self):
        return len(self.nodes)

    def get_nodes(self):
        coalition = [i for i, v in enumerate(self.nodes) if v == 1]
        return coalition


def init(container, func, num_nodes):
    return container(func(i) for i in range(num_nodes))


def generate_individual(node_idx: int, subgraph_func, num_nodes, Individual_Cls):
    nodes = subgraph_func(node_idx)
    inv = [0] * num_nodes
    for i in nodes.tolist():
        inv[i] = 1
    return Individual_Cls(inv)
