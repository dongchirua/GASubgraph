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


args = Args()

rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

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
mutag_dataset.valid_ds = mutag_dataset.ds[int(0.8 * mutag_size): int(0.9 * mutag_size)]
mutag_dataset.test_ds = mutag_dataset.ds[int(0.9 * mutag_size):]

# Since MUTAG has multiple graphs, we use DataLoaders to load the graphs
mutag_dataset = edict()
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mutag_optimizer = torch.optim.Adam(
    # mutag_model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)


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


mutag_model = torch.load('pre_train/mutag.pth', map_location=device)
mutag_model.eval()

run_epoch_graph_cls(
    loader=mutag_test_loader,
    model=mutag_model,
    optimizer=mutag_optimizer,
    device=device,
    mode='test'
)
mutag_explainer = GNNExplainer(mutag_model, epochs=2000, return_type='raw', log=False)

foo_sample = mutag_dataset.test_ds[0], edge_mask = mutag_explainer.explain_graph(foo_sample.x.to(device),
                                                                                 foo_sample.edge_index.to(device))

_, edge_mask = mutag_explainer.explain_graph(foo_sample.x.to(device), foo_sample.edge_index.to(device))

print(edge_mask)
