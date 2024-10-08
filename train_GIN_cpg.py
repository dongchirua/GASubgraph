#!/usr/bin/env python
# coding: utf-8

import os
import random
from dataclasses import dataclass
import numpy as np
import torch
import click
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
from vulexp.ml_models.train_helper import tune_ashas_scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

cwd = os.getcwd()
data_dir = f'{cwd}/data/reveal/'

absolute_path = os.getcwd()
reveal_dataset = Reveal(data_dir, absolute_path=absolute_path,
                        over_write=args.over_write, to_undirected=args.to_undirected,
                        seed=args.seed, gtype=args.gtype)


# reveal_train, reveal_val, reveal_test = reveal_dataset.generate_train_test()


@click.command()
@click.option('--mode', default='solo', help='Solo train or Tune params.')
def run_mode(mode):
    if mode == 'solo':
        config = {
            "num_layers": 3,
            "dropout": 0.2,
            "hidden_channels": 128,
            "out_channels": 1,
            "batch_size": 512,
            "threshold": 0.5,
        }
        from vulexp.ml_models.train_helper import train_model, get_run_id
        run_id = get_run_id()
        store_path = os.path.join(cwd, 'solo_train', run_id)

        train_model(config, args.name, GIN, store_path, reveal_dataset, num_workers=8, num_epochs=args.n_epoch,
                    input_channel=args.feat_dim, n_class=reveal_dataset.n_class)

    if mode == 'tune':
        config = {
            "num_layers": tune.choice([1, 2, 3, 5]),
            "dropout": tune.choice([0.1, 0.2, 0.25, 0.34]),
            "hidden_channels": tune.choice([16, 32, 64, 128]),
            "batch_size": tune.choice([64, 128, 512, 1024]),
            "threshold": tune.choice([0.2, 0.5, 0.75]),
            "learning_rate": tune.choice([1e-2, 1e-3, 4e-5]),
            "weight_decay": tune.choice([1e-2, 1e-3, 4e-5]),
        }
        n_gpu = 1 if device.type == 'cuda' else 0

        tune_ashas_scheduler(config, name=args.name, custom_nn_model=GIN, custom_dataset=reveal_dataset,
                             max_epochs=args.n_epoch, n_class=reveal_dataset.n_class,
                             gpus_per_trial=n_gpu,
                             input_channel=args.feat_dim)


if __name__ == '__main__':
    run_mode()
