#!/usr/bin/env python
# coding: utf-8


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class Args:
    seed: int = 27  # Random seed.
    to_undirected = False
    gtype = 'cpg'  # cpg or smg
    over_write = True


args = Args()

rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

data_dir = 'data/reveal/'
reveal_dataset = Reveal(data_dir,
                        over_write=args.over_write, to_undirected=args.to_undirected,
                        seed=args.seed, gtype=args.gtype)

reveal_train, reveal_val, reveal_test = reveal_dataset.generate_train_test()

foo = reveal_train.get(0)
