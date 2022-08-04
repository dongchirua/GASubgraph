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
from tqdm.notebook import trange, tqdm


# ## configs

# In[2]:


@dataclass
class Args:
    """A class containing arguments used for setting up the dataset and model."""
    batch_size: int = 32  # Batch size for the training loop.
    num_workers: int = 0  # Number of workers to use for the data loader.
    learning_rate: float = 0.001  # Learning rate.
    weight_decay: float = 5e-4  # Weight decay.
    num_epochs: int = 300  # Number of training epochs.
    num_layers: int = 3  # Number of message passing layers in the GNN model.
    hidden_features: int = 32  # Dimensionality of the hidden layers in the GNN.
    dropout: float = 0.2  # Dropout probability.
    seed: int = 27  # Random seed.
    pre_train: bool = True  # Change to False if want to retrain

args = Args()


# In[3]:


rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


# # Dataset Loading

# In[4]:


# Load MUTAG dataset (graph classification)
mutag_dataset = edict()
mutag_dataset.ds = TUDataset(
    'data/mutag/',
    name='MUTAG',
    use_node_attr=True,
    use_edge_attr=True
)
mutag_dataset.ds.shuffle()
mutag_size = len(mutag_dataset.ds)
mutag_dataset.train_ds = mutag_dataset.ds[:int(0.8 * mutag_size)]
mutag_dataset.valid_ds = mutag_dataset.ds[int(0.8 * mutag_size) : int(0.9 * mutag_size)]
mutag_dataset.test_ds = mutag_dataset.ds[int(0.9 * mutag_size):]

# Since MUTAG has multiple graphs, we use DataLoaders to load the graphs
mutag_train_loader = DataLoader(
    dataset=mutag_dataset.train_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True
)
mutag_valid_loader = DataLoader(
    dataset=mutag_dataset.valid_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False
    )
mutag_test_loader = DataLoader(
    dataset=mutag_dataset.test_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False
)


# # Simple GNN Model

# In[5]:


class GIN_SubgraphX(torch.nn.Module):
    """The GIN model from the SubgraphX paper.

    Adapted from https://github.com/divelab/DIG/blob/dig/dig/xgraph/models/models.py
    """
    def __init__(self, 
                in_channels: int, 
                hidden_channels: int, 
                num_layers: int, 
                out_channels: int, 
                dropout: float = 0.5):
        super(GIN_SubgraphX, self).__init__()

        self.dropout = dropout

        convs_list = [
            GINConv(nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU())
            )
        ]
        convs_list += [
            GINConv(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU())
            )
            for _ in range(num_layers - 1)
        ]

        self.convs = nn.ModuleList(convs_list)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), 
            nn.ReLU(), 
            nn.Dropout(self.dropout), 
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, 
                x: LongTensor, 
                edge_index: LongTensor, 
                batch: LongTensor) -> FloatTensor:
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)

        x = global_max_pool(x, batch)

        x = self.classifier(x)

        return x


# In[6]:


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[7]:


# Load MUTAG GIN model and optimizer
mutag_model = GIN_SubgraphX(
    in_channels=mutag_dataset.ds.num_features,
    hidden_channels=args.hidden_features,
    num_layers=args.num_layers,
    out_channels=1,
    dropout=args.dropout
).to(device)

print('MUTAG model')
print(mutag_model)

mutag_optimizer = torch.optim.Adam(
    mutag_model.parameters(),
    lr=args.learning_rate, 
    weight_decay=args.weight_decay
)


# ## training

# In[8]:


class AverageMeter(object):
    """The AverageMeter keeps track of an average value over multiple updates."""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Resets the meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Updates the meter with a value and a count.

        :param val: The value to add to the running sum.
        :param n: The count, i.e., the number of items included in val.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[9]:


def run_epoch_graph_cls(loader: DataLoader,
                        model: nn.Module,
                        optimizer: torch.optim.Adam,
                        device: torch.device,
                        mode: Literal['train', 'valid', 'test'] = False) -> None:
    """Runs an epoch (train or eval) of a graph classification model/dataset."""
    epoch_loss = AverageMeter()
    epoch_acc = AverageMeter()

    if mode == 'train':
        model.train()
    else:
        model.eval()

    preds, targets = [], []

    for batch in loader:
        batch = batch.to(device)

        if mode == 'train':
            optimizer.zero_grad()

        out = model(batch.x.float(), batch.edge_index, batch.batch)

        pred = out.squeeze()
        target = batch.y.float().squeeze()

        preds.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())

        loss = F.binary_cross_entropy_with_logits(
            input=pred,
            target=target
        )

        if mode == 'train':
            loss.backward()
            optimizer.step()

        epoch_loss.update(loss.detach().item(), out.shape[0])
        
        batch_acc = ((torch.sigmoid(out) > 0.5).int().flatten() == batch.y).float().mean()
        epoch_acc.update(batch_acc, out.shape[0])

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    epoch_auc = roc_auc_score(y_true=targets, y_score=preds)

    print(f'{mode} loss in the epoch: {epoch_loss.avg:.3f}, '
          f'{mode} accuracy in the epoch: {epoch_acc.avg:.3f}, '
          f'{mode} auc in the epoch: {epoch_auc:.3f}')


# ## classificatioon model here!

# In[13]:


if args.pre_train != True or not os.path.exists('pre_train/mutag.pth'):
    # Train MUTAG model
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch}')

        # Train loop
        run_epoch_graph_cls(
            loader=mutag_train_loader,
            model=mutag_model,
            optimizer=mutag_optimizer,
            device=device,
            mode='train'
        )

        # Validation loop
        run_epoch_graph_cls(
            loader=mutag_valid_loader,
            model=mutag_model,
            optimizer=mutag_optimizer,
            device=device,
            mode='valid'
        )

    # save model
    torch.save(mutag_model, 'pre_train/mutag.pth')
else:
    # load pre trained model
    mutag_model = torch.load('pre_train/mutag.pth', map_location=device)
    mutag_model.eval()    

# Test loop
run_epoch_graph_cls(
    loader=mutag_test_loader,
    model=mutag_model,
    optimizer=mutag_optimizer,
    device=device,
    mode='test'
)


# # Explaination

# ## GNNExplaniner

# In[14]:


# Create GNNExplainer model for MUTAG
mutag_explainer = GNNExplainer(mutag_model, epochs=2000, return_type='raw', log=False)


# ## visualization functions

# In[15]:


def visualize_subgraph_mutag(graph: nx.Graph,
                             node_set: Optional[Set[int]] = None,
                             edge_set: Optional[Set[int]] = None,
                             title: Optional[str] = None) -> None:
    """Visualizes a subgraph explanation for a graph from the MUTAG dataset.

    Note: Only provide subgraph_node_set or subgraph_edge_set, not both.

    Adapted from https://github.com/divelab/DIG/blob/dig/dig/xgraph/method/subgraphx.py

    :param graph: A NetworkX graph object representing the full graph.
    :param node_set: A set of nodes that induces a subgraph.
    :param edge_set: A set of edges that induces a subgraph.
    :param title: Optional title for the plot.
    """
    if node_set is None:
        node_set = set(graph.nodes())

    if edge_set is None:
        edge_set = {(n_from, n_to) for (n_from, n_to) in graph.edges() if n_from in node_set and n_to in node_set}

    # node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
    node_idxs = {node: node_x.index(1.0) for node, node_x in graph.nodes(data='x')}
    # node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
    node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
    colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

    pos = nx.kamada_kawai_layout(graph)

    nx.draw_networkx_nodes(G=graph, pos=pos, nodelist=list(graph.nodes()), node_color=colors, node_size=300)
    nx.draw_networkx_edges(G=graph, pos=pos, width=3, edge_color='gray', arrows=False)
    nx.draw_networkx_edges(G=graph, pos=pos, edgelist=list(edge_set), width=6, edge_color='black', arrows=False)
    # nx.draw_networkx_labels(G=graph, pos=pos, labels=node_labels)
    nx.draw_networkx_labels(G=graph, pos=pos)

    if title is not None:
        plt.title(title)

    plt.axis('off')
    plt.show()
    plt.close()


# In[16]:


def visualize_explanation_for_graph(threshold: float, graph_idx: int) -> None:
    """Visualizes the explanations of GNNExplainer for a graph given a mask threshold."""
    mutag_data = mutag_dataset.test_ds[graph_idx]

    _, edge_mask = mutag_explainer.explain_graph(mutag_data.x.to(device), mutag_data.edge_index.to(device))

    batch = torch.zeros(mutag_data.x.shape[0], dtype=int, device=device)
    output = mutag_model(mutag_data.x.to(device), mutag_data.edge_index.to(device), batch)
    pred = torch.sigmoid(output).item()

    edge_set = {(edge[0].item(), edge[1].item()) for edge, mask in zip(mutag_data.edge_index.T, edge_mask) if mask > threshold}
    graph = to_networkx(mutag_data, node_attrs=['x'], edge_attrs=['edge_attr'], to_undirected=True)

    visualize_subgraph_mutag(
        graph=graph,
        edge_set=edge_set,
        title=f'GNNExplainer on graph {graph_idx}: label = {mutag_data.y.item()}, pred = {pred:.2f}'
    )


# In[18]:

if __name__ == '__main__':
    foo_sample = mutag_dataset.test_ds[0]

    _, edge_mask = mutag_explainer.explain_graph(foo_sample.x.to(device), foo_sample.edge_index.to(device))

    print(edge_mask)