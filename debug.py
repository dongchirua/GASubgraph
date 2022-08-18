#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import math
import os
from argparse import ArgumentParser, Namespace
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from typing_extensions import Literal

import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from ipywidgets import interact
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import FloatTensor, LongTensor, Tensor
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, GNNExplainer, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import GIN
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, to_networkx
from tqdm.auto import trange, tqdm
from visualization.plot import visualize_explanation, aggregate_figures


# run GA with dummy classification but on realdata

# In[2]:


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
    CXPB = 0.55
    MUTPB = 0.25
    tournsize = 11
    to_undirected = True
    # subgraph_building_method = 'split'
    subgraph_building_method = "zero_filling"
    n_generation = 150


args = Args()

# In[3]:


rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# In[4]:


from vulexp.data_models.pl_data_module import DataModule
from vulexp.data_models.reveal_data import Reveal

data_dir = 'data/reveal/'
reveal_dataset = Reveal(data_dir, to_undirected=args.to_undirected, seed=args.seed)

reveal_train, reveal_val, reveal_test = reveal_dataset.generate_train_test()

reveal_train_loader = DataLoader(
    dataset=reveal_train,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True
)
reveal_valid_loader = DataLoader(
    dataset=reveal_val,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True
)
reveal_test_loader = DataLoader(
    dataset=reveal_test,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False
)

# In[5]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# In[6]:


from vulexp.ml_models.pl_train_module_logit import TrainingModule
from vulexp.ml_models.gin import GIN

saved_model = TrainingModule.load_from_checkpoint(model=GIN, map_location=device,
                                                  checkpoint_path="weights/Reveal-GIN-auc_pos=0.78-optimal_t=0.560-f1=0.34-epoch=04.ckpt")
saved_model.to(device)
saved_model.eval()

# In[7]:

# positive = []
# for i in tqdm(range(len(reveal_test))):
#     l = reveal_test.get(i)
#     if l.y.item() == 1:
#         positive.append(i)
# 0 -> 162

sel = 80
k_node = 6
print(f'select sample #{sel}')
print(f'constraint node #{k_node}')
foo_sample = reveal_test.get(sel)

# In[8]:


output = saved_model(foo_sample.x.to(device), foo_sample.edge_index.to(device), None)
pred = torch.sigmoid(output).item()
print(pred, foo_sample.y, foo_sample.num_nodes)

# In[9]:


from ga_subgraph.explainer import GASubX
from ga_subgraph.fitness import classifier
from ga_subgraph.individual import Individual

ga_explainer = GASubX(saved_model, classifier, device, Individual, args.n_generation, args.CXPB, args.MUTPB,
                      args.tournsize, args.subgraph_building_method)
ga_subgraph, _ = ga_explainer.explain(foo_sample, k_node, verbose=False)
print('GASubX', ga_subgraph)

# In[11]:


from vulexp.explanation.subgraphx import SubgraphX

subgraphx = SubgraphX(model=saved_model, min_nodes=5, n_rollout=args.n_generation)
subgraph = subgraphx.explain(x=foo_sample.x.to(device), edge_index=foo_sample.edge_index.to(device), max_nodes=k_node)
print('SubgraphX', subgraph.coalition)


def extract_node_from_mask(mask, select_nodes: int, sample: Data):
    edge_set = {(edge[0].item(), edge[1].item(), m.item()) for edge, m in zip(sample.edge_index.T, mask)}
    edge_set = sorted(edge_set, key=lambda k: k[2], reverse=True)
    nodes = set()
    count = 0
    while count < len(edge_set):
        a, b, _ = edge_set[count]
        if a != b:
            # remove self-loop cases
            nodes.update([a])
            nodes.update([b])
            if len(nodes) >= select_nodes:
                break
        count += 1
    return list(nodes)


gnn_explainer = GNNExplainer(saved_model, epochs=args.n_generation, return_type='raw', log=False)
_, gnn_edge_mask = gnn_explainer.explain_graph(foo_sample.x.to(device), foo_sample.edge_index.to(device))
gnn_explainer_nodes = extract_node_from_mask(gnn_edge_mask, 5, foo_sample)
print('gnnexplainer', gnn_explainer_nodes)

# In[14]:


from ga_subgraph.fitness import graph_build_zero_filling, graph_build_split

# In[15]:


# ga_result = graph_build_zero_filling(foo_sample.x, foo_sample.edge_index, ga_mask)
# sub_result = graph_build_zero_filling(foo_sample.x, foo_sample.edge_index, sub_mask)


# In[16]:

from ga_subgraph.utils import helper

helper(ga_subgraph, foo_sample, saved_model, pred, device)

helper(list(subgraph.coalition), foo_sample, saved_model, pred, device)

helper(gnn_explainer_nodes, foo_sample, saved_model, pred, device)

# In[ ]:
# visualize_explanation(foo_sample, selected_nodes=ga_subgraph, selected_edges=None)

aggregate_figures(foo_sample, ga_subgraph, list(subgraph.coalition), gnn_explainer_nodes,
                  sel, pred, saved_model, device)
