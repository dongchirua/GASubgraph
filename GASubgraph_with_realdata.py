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


# run GA with dummy classification but on realdata

# In[2]:


import logging
def setup_logger():
    """Configure logger.
    """

    logging.basicConfig(
        format='%(asctime)s | %(name)s | %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    return logging.getLogger('GASub')


# In[3]:


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
    MUTPB =  0.2

args = Args()


# In[4]:


rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


# In[5]:


from vulexp.data_models.pl_data_module import DataModule
from vulexp.data_models.reveal_data import Reveal

data_dir = 'data/reveal/'
reveal_dataset = Reveal(data_dir, to_undirected=True, seed=args.seed)


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
    shuffle=False
    )
reveal_test_loader = DataLoader(
    dataset=reveal_test,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False
)


# In[6]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[7]:


from vulexp.ml_models.pl_train_module_logit import TrainingModule
from vulexp.ml_models.gin import GIN

saved_model = TrainingModule.load_from_checkpoint(model=GIN, map_location=device,
                                                  checkpoint_path="weights/Reveal-GIN-auc_pos=0.78-optimal_t=0.560-f1=0.34-epoch=04.ckpt")
saved_model.to(device)
saved_model.eval()


# In[8]:


sel = 4
foo_sample = reveal_test.get(sel)


# In[9]:


output = saved_model(foo_sample.x.to(device), foo_sample.edge_index.to(device), None)
pred = torch.sigmoid(output).item()
print(pred)


# In[10]:


from deap import base, algorithms, creator, tools
from torch_geometric.utils import k_hop_subgraph, get_num_hops
from torch_geometric.utils.num_nodes import maybe_num_nodes
toolbox = base.Toolbox()


# In[11]:


def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
    """ subgraph building through masking the unselected nodes with zero features """
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index

def graph_build_split(X, edge_index, node_mask: torch.Tensor):
    """ subgraph building through spliting the selected nodes from the original graph """
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return X, ret_edge_index

def get_graph_build_func(build_method):
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    else:
        raise NotImplementedError

def gnn_score(coalition: list, data: Data, gnn_model: Callable,
              subgraph_building_method='zero_filling') -> torch.Tensor:
    """ the value of subgraph with selected nodes """
    num_nodes = data.num_nodes
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    mask = torch.zeros(num_nodes).type(torch.float32).to(data.x.device)
    mask[coalition] = 1.0
    ret_x, ret_edge_index = subgraph_build_func(data.x, data.edge_index, mask)
    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    score = gnn_model(mask_data)
    # get the score of predicted class for graph or specific node idx
    return score.item()

def get_fitness_func(score_method, gnn_model, subgraph_building_method='zero_filling'):
    """ Function factory to generate a method measure how quality of a individual
    Args:
        score_method: method to use
        gnn_model:  a blackbox algorithm
        subgraph_building_method: way to construct a suggraph
    """
    if score_method.lower() == 'gnn_score':
        return partial(gnn_score,
                       gnn_model=gnn_model,
                       subgraph_building_method=subgraph_building_method)
    else:
        raise NotImplementedError

def wrap_classifier(data):
    """ Wraper for any classification method
    """
    out = saved_model(x=data.x.to(device), edge_index=data.edge_index.to(device), batch=None)
    prod = torch.sigmoid(out)
    return prod

def evalSubGraph(individual: list, origin_graph: Data, K=5) -> float:
    """ A value of a subgraph is scored by how close gnn output from it and original graph.
        The final value takk size of subgraph to consideration.
        We are going to minize this function
    """
    coalition = [i for i,v in enumerate(individual) if v==1]
    fitness_func = get_fitness_func('gnn_score', wrap_classifier)
    fitness_value = fitness_func(coalition=coalition, data=origin_graph)
    origin_fitness_value = wrap_classifier(origin_graph)
    return abs(fitness_value - origin_fitness_value.item()) + (abs(len(coalition)-K)/origin_graph.num_nodes),



# In[12]:


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))


# In[13]:


class Individual(object):
    def __init__(self, nodes):
        if type(nodes) is list:
            self.nodes = nodes
        else:
            self.nodes = [i for i in nodes]

    def __repr__(self):
        coalition = [str(i) for i,v in enumerate(self.nodes) if v==1]
        return ' '.join(coalition)

    def __get__(self, instance, owner):
        return self.nodes

    def __set__(self, instance, value):
        self.nodes = value

    def __getitem__(self, item):
        return self.nodes[item]

    def __setitem__(self, key, value):
        self.nodes[key] = value

    def __len__(self):
        return len(self.nodes)
    
    def get_nodes(self):
        coalition = [i for i,v in enumerate(self.nodes) if v==1]
        return coalition

creator.create("Individual", Individual, fitness=creator.FitnessMin)


# In[14]:


def feasible(individual):
    """Feasibility function for the individual. Returns True if feasible False
    otherwise."""
    origin_graph=foo_sample  # todo
    G = to_networkx(origin_graph, to_undirected=origin_graph.is_directed())
    sub_graph = G.subgraph(individual.get_nodes())
    if origin_graph.is_directed():
        components = [i for i in nx.weakly_connected_components(sub_graph)]
    else:
        sub_graph = sub_graph.to_undirected()
        components = [i for i in nx.connected_components(sub_graph)]
    if len(components) == 1:
        return True
    return False


# In[15]:


toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_bool, foo_sample.num_nodes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalSubGraph, origin_graph=foo_sample)
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 10.0))
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=10)


# In[16]:


CXPB = args.CXPB
MUTPB = args.MUTPB

logger = setup_logger()

# keep track of the best individuals
hof = tools.HallOfFame(5)
history = tools.History()

# setting the statistics (displayed for each generation)
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('avg', np.mean, axis=0)
stats.register('min', np.min, axis=0)
stats.register('max', np.max, axis=0)

pop = toolbox.population(200)
history.update(pop)

try:
    final_population, logbook = algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=10, lambda_=50, cxpb=CXPB, mutpb=MUTPB,
        ngen=100, stats=stats, halloffame=hof, verbose=True)
except (Exception, KeyboardInterrupt) as e:
    logging.info(e)
    for individual in hof:
        logger.info(
            f'hof: {individual.fitness.values[0]:.3f} << {individual}')

for individual in hof:
    logger.info(
        f'hof: {individual.fitness.values[0]:.3f} << {individual}')


# In[19]:


from vulexp.explanation.subgraphx import SubgraphX

reveal_subgraphx = SubgraphX(model=saved_model, min_nodes=5)
subgraph = reveal_subgraphx.explain(x=foo_sample.x.to(device), edge_index=foo_sample.edge_index.to(device), max_nodes=5)

print(subgraph.coalition)


# In[ ]:


def remove_nodes(raw_graph: nx.Graph, graph_label, node_set: Optional[Set[int]]):
    from vulexp.data_models.helpers import from_networkx, convert_single_graph
    cp_graph = raw_graph.copy()
    cp_graph.remove_nodes_from(node_set)
    new_graph = from_networkx(convert_single_graph(cp_graph), group_node_attrs=['feat'])
    new_graph.y = torch.LongTensor([graph_label])
    new_graph.x = new_graph.x.float()
    return new_graph


# In[ ]:


ga_mask = torch.zeros(foo_sample.num_nodes).type(torch.float32).to(foo_sample.x.device)
ga_mask[hof[-1].get_nodes()] = 1 


# In[ ]:


sub_mask = torch.zeros(foo_sample.num_nodes).type(torch.float32).to(foo_sample.x.device)
sub_mask[list(subgraph.coalition)] = 1 


# In[ ]:


ga_result = graph_build_split(foo_sample.x, foo_sample.edge_index, ga_mask)
sub_result = graph_build_split(foo_sample.x, foo_sample.edge_index, sub_mask)


# In[ ]:


output = saved_model(ga_result[0].to(device), ga_result[1].to(device), None)
pred = torch.sigmoid(output).item()
print(pred)


# In[ ]:


output = saved_model(sub_result[0].to(device), sub_result[1].to(device), None)
pred = torch.sigmoid(output).item()
print(pred)


# In[ ]:




