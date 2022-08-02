from vulexp.data_models.helpers import from_networkx, convert_single_graph


def compute_fidelity(model, graph: Data,
                     raw_graph: nx.Graph,
                     node_set: Optional[Set[int]] = None, device='cpu'):
    batch = torch.zeros(graph.x.shape[0], dtype=int, device=device)

    output = model(graph.x.to(device), graph.edge_index.to(device), batch)
    pred = torch.sigmoid(output).item()
    label = 1 if pred > 0.5 else 0

    # print(pred)

    # extract subgraph by excluding other nodes
    cp_graph = raw_graph.copy()
    cp_graph.remove_nodes_from(cp_graph.nodes() - node_set)
    new_graph = from_networkx(convert_single_graph(cp_graph), group_node_attrs=['feat'])
    new_graph.y = torch.LongTensor([graph.y])
    new_graph.x = new_graph.x.float()
    batch = torch.zeros(new_graph.x.shape[0], dtype=int, device=device)
    output = model(new_graph.x.to(device), new_graph.edge_index.to(device), batch)
    pred_new = torch.sigmoid(output).item()
    # print(pred_new)

    # visualize_subgraph(raw_bar, node_set=subgraph.coalition, title=f'SubgraphX on graph, orginal pred = {pred}, new pred = {pred_new}')
    return pred, pred_new, subgraph.coalition, label, graph.x.shape[0]
