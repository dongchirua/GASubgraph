from torch_geometric.utils import k_hop_subgraph, get_num_hops


def subgraph(node_idx: int, x, edge_index, num_hops: int):
    """
    generate subgraph of node_idx bases on num_hops
    :param node_idx:
    :param x: Data.x
    :param edge_index: Data.edge_index
    :param num_hops: number of layers in GNN
    :return:
    """
    subset, edge_index, _, edge_mask = k_hop_subgraph(node_idx, num_hops, edge_index)
    # x = x[subset]
    return subset
