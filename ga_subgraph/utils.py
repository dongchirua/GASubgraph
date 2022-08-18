import torch
from ga_subgraph.fitness import graph_build_zero_filling


def helper(selected_nodes, sample, model, origin_pred, device):
    complementary_nodes = list(set(range(sample.num_nodes)) - set(selected_nodes))
    mask = torch.zeros(sample.num_nodes).type(torch.float32).to(sample.x.device)
    mask[complementary_nodes] = 1
    r_subgraph, r_subgraph_egde = graph_build_zero_filling(sample.x, sample.edge_index, mask)
    o = model(r_subgraph.to(device), r_subgraph_egde.to(device), None)
    inv_prob = torch.sigmoid(o).item()

    mask = torch.zeros(sample.num_nodes).type(torch.float32).to(sample.x.device)
    mask[selected_nodes] = 1
    r_subgraph, r_subgraph_egde = graph_build_zero_filling(sample.x, sample.edge_index, mask)
    o = model(r_subgraph.to(device), r_subgraph_egde.to(device), None)
    prob = torch.sigmoid(o).item()

    print('prob', prob, 'inv_prob', inv_prob, 'inv_fidelity', origin_pred - inv_prob, 'fidelity', abs(origin_pred - prob))
    return prob, inv_prob, abs(origin_pred - prob)
