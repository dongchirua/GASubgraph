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
from vulexp.ml_models.pl_train_module_logit import TrainingModule
from vulexp.ml_models.gin import GIN
from ray import tune
from vulexp.ml_models.helper import tune_ashas_scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = 1 if device == 'cuda' else 0

@dataclass
class Args:
    seed: int = 27  # Random seed.
    to_undirected = False
    gtype = 'cpg'  # cpg or smg
    over_write = False
    n_epoch = 101
    name = 'GIN_CPG'
    feat_dim = 133  # number of node feature


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

# reveal_train, reveal_val, reveal_test = reveal_dataset.generate_train_test()

config = {
    "num_layers": tune.randint(0, 5),
    "dropout": tune.uniform(0.05, 0.4),
    "hidden_channels": tune.choice([16, 32, 64, 128]),
    "out_channels": tune.choice([8, 16, 32]),
    "normalize": tune.choice([True, False]),
    "batch_size": tune.choice([64, 128, 512]),
    "threshold": tune.choice([0.2, 0.5, 0.75]),
}

tune_ashas_scheduler(config, name=args.name, custom_nn_model=GIN, custom_dataset=reveal_dataset,
                     max_epochs=args.n_epoch, n_class=reveal_dataset.n_class, gpus_per_trial=n_gpu,
                     input_channel=args.feat_dim)
