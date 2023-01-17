#!/usr/bin/env python
# coding: utf-8

import os
import random
from dataclasses import dataclass
import numpy as np
import torch
from torch_geometric.nn import GNNExplainer
from ga_subgraph.utils import extract_node_from_mask
from visualization.plot import aggregate_figures
from ga_subgraph.explainer import GASubX
from ga_subgraph.fitness import classifier
from ga_subgraph.individual import Individual
from vulexp.data_models.reveal_data import Reveal
from vulexp.explanation.subgraphx import SubgraphX


@dataclass
class Args:
    seed: int = 27  # Random seed.
    to_undirected = False
    gtype = 'cpg'  # cpg or smg
    over_write = False
    CXPB = 0.55
    MUTPB = 0.25
    tournsize = 11
    subgraph_building_method = "zero_filling"
    n_generation = 150


args = Args()

rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

data_dir = 'data/reveal/'
absolute_path = os.getcwd()
reveal_dataset = Reveal(data_dir, absolute_path=absolute_path,
                        over_write=args.over_write, to_undirected=args.to_undirected,
                        seed=args.seed, gtype=args.gtype)

reveal_train, reveal_val, reveal_test = reveal_dataset.generate_train_test()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from vulexp.ml_models.pl_train_module_logit import TrainingModule
from vulexp.ml_models.gin import GIN

saved_model = TrainingModule.load_from_checkpoint(model=GIN, map_location=device,
                                                  checkpoint_path="solo_train/202301171316_88cc311_add_click/checkpoint/GIN_CPG-epoch=00-loss=0.57-f1=0.2576419213973799.ckpt")
saved_model.to(device)
saved_model.eval()

sel = 3
k_node = 6
print(f'select sample #{sel}')
print(f'constraint node #{k_node}')
foo_sample = reveal_test.get(sel)


output = saved_model(foo_sample.x.to(device), foo_sample.edge_index.to(device), None)
pred = torch.sigmoid(output).item()
print(pred, foo_sample.y, foo_sample.num_nodes)

ga_explainer = GASubX(saved_model, classifier, device, Individual, args.n_generation, args.CXPB, args.MUTPB,
                      args.tournsize, args.subgraph_building_method)
ga_subgraph, _ = ga_explainer.explain(foo_sample, k_node, verbose=False)
print('GASubX', ga_subgraph)

subgraphx = SubgraphX(model=saved_model, min_nodes=5, n_rollout=args.n_generation)
subgraph = subgraphx.explain(x=foo_sample.x.to(device), edge_index=foo_sample.edge_index.to(device), max_nodes=k_node)
print('SubgraphX', subgraph.coalition)

gnn_explainer = GNNExplainer(saved_model, epochs=args.n_generation, return_type='raw', log=False)
_, gnn_edge_mask = gnn_explainer.explain_graph(foo_sample.x.to(device), foo_sample.edge_index.to(device), batch=None)
gnn_explainer_nodes = extract_node_from_mask(gnn_edge_mask, k_node, foo_sample)
print('gnnexplainer', gnn_explainer_nodes)

aggregate_figures(foo_sample, ga_subgraph, list(subgraph.coalition), gnn_explainer_nodes,
                  sel, pred, saved_model, device)
