class Individual(object):
    """ 
        This class represents for a chromosome
    """

    def __init__(self, gene):
        """
            Initialize the individual with a gene
        :param gene: is a list of 0 and 1
        """
        if type(gene) is list:
            self.__gene = gene
        else:
            raise NotImplementedError('gene must be a list')

    def __repr__(self):
        return ' '.join([str(i) for i in self.get_nodes()])

    def __get__(self, instance, owner):
        return self.__gene

    def __set__(self, instance, value):
        self.__gene = value

    def __getitem__(self, item):
        return self.__gene[item]

    def __setitem__(self, key, value):
        self.__gene[key] = value

    def __len__(self):
        return len(self.__gene)

    def __hash__(self):
        return hash('|'.join(sorted(f'{x}' for x in self.get_nodes())))

    def __eq__(self, other):
        common = set(self.get_nodes()).intersection(set(other.get_nodes()))
        return (len(common) == len(self.get_nodes())) and (len(common) == len(other.get_nodes()))

    def get_nodes(self):
        coalition = sorted([i for i, v in enumerate(self.__gene) if v == 1])
        return coalition


def init_population(container, func, num_nodes):
    return container(func(i) for i in range(num_nodes))


def generate_individual(node_idx: int, subgraph_func, num_nodes, Individual_Cls):
    nodes = subgraph_func(node_idx)
    inv = [0] * num_nodes
    for i in nodes.tolist():
        inv[i] = 1
    return Individual_Cls(inv)
