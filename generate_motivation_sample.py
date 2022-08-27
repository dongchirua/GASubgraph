from vulexp.data_models.adhoc_data import Adhoc
from torch_geometric.utils import to_networkx
from visualization.plot import visualize_subgraph
import matplotlib.pyplot as plt
import torch
from do_statistic import explainers
from dataclasses import dataclass
from ga_subgraph.fitness import classifier
from ga_subgraph.utils import helper
from vulexp.visualization.graphs import nx_to_graphviz, save_graph_dot
from graphviz import Source
from visualization.plot import aggregate_figures


data_path = '/Users/bryan/workplace/PhD/writing/vulexp/code/motivation/'

if __name__ == "__main__":
    adhoc_data = Adhoc(data_path, to_undirected=True, over_write=True)
    adhoc_set = adhoc_data.generate()
    sample = adhoc_set.get(0)
    sample_raw, _ = adhoc_set.get_raw(0)

    # graph = to_networkx(sample, to_undirected=not sample.is_directed())
    fig, ax = plt.subplots()
    visualize_subgraph(
        graph=sample_raw,
        node_set=None,
        edge_set=None,
        title='Motivation Sample',
        ax=ax
    )
    plt.show()
    plt.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from vulexp.ml_models.pl_train_module_logit import TrainingModule
    from vulexp.ml_models.gin import GIN

    saved_model = TrainingModule.load_from_checkpoint(model=GIN, map_location=device,
                                                      checkpoint_path="weights/Reveal-GIN-auc_pos=0.78-optimal_t=0.560-f1=0.34-epoch=04.ckpt")
    saved_model.to(device)
    saved_model.eval()
    output = saved_model(sample.x.to(device), sample.edge_index.to(device), None)
    pred = torch.sigmoid(output).item()
    print(pred)


    @dataclass
    class Args:
        """A class containing arguments used for setting up the dataset and model."""
        batch_size: int = 32  # Batch size for the training loop.
        num_workers: int = 2  # Number of workers to use for the data loader.
        learning_rate: float = 0.001  # Learning rate.
        weight_decay: float = 5e-4  # Weight decay.
        num_epochs: int = 300  # Number of training epochs.
        num_layers: int = 3  # Number of message passing layers in the GNN model.
        hidden_features: int = 32  # Dimensionality of the hidden layers in the GNN.
        dropout: float = 0.2  # Dropout probability.
        seed: int = 27  # Random seed.
        pre_train: bool = True  # Change to False if want to retrain
        CXPB = 0.5
        MUTPB = 0.15
        tournsize = 5
        to_undirected = True
        # subgraph_building_method = 'split'
        subgraph_building_method = "zero_filling"
        n_generation = 150


    args = Args()
    k_node = 4

    ga_explain, ga_dtime, \
        sub_explain, sub_dtime, \
        gnn_explainer, gnn_dtime = explainers(saved_model, classifier, sample,
                                            k_node, device, args.n_generation,
                                            args.CXPB, args.MUTPB,
                                            args.tournsize,
                                            args.subgraph_building_method)
    print(ga_explain, sub_explain, gnn_explainer)
    print(ga_dtime, sub_dtime, gnn_dtime)

    ga_explain_prob, inv_ga_explain_prob, ga_fidelity = helper(ga_explain, sample, saved_model, pred, device)
    sub_explain_prob, inv_sub_explain_prob, sub_fidelity = helper(sub_explain, sample, saved_model, pred, device)
    gnn_explain_prob, inv_gnn_explain_prob, gnn_fidelity = helper(gnn_explainer, sample, saved_model, pred, device)
    print(ga_explain_prob, sub_explain_prob, gnn_explain_prob)

    sample_viz = save_graph_dot(sample_raw, f'{sample_raw.name}.gv')

    # fig, ax = plt.subplots()
    # visualize_subgraph(
    #     graph=sample_raw,
    #     node_set=ga_explain,
    #     edge_set=None,
    #     title='Motivation Sample',
    #     ax=ax
    # )
    # plt.show()
    # plt.close()

    aggregate_figures(sample, ga_explain, sub_explain, gnn_explainer, sample_id=None,
                      origin_pred=pred, saved_model=saved_model, device=device)
