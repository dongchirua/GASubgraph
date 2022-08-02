import os
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor, Tensor

from torch_geometric.nn import GINConv, global_max_pool

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

# from vulexp.ml_models.pl_train_module_new import TrainingModule
from vulexp.ml_models.gnn import GNN
from vulexp.data_models.pl_data_module import DataModule
from vulexp.data_models.reveal_data import Reveal
from vulexp.visualization.graphs import nx_to_graphviz

from filelock import FileLock

from sklearn.metrics import roc_auc_score, f1_score

from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

from vulexp.ml_models.gin import GIN

def train_model_wrapper(config, nn_network, train_dataset, max_epoch: int = 100, num_gpus: int = 0):

    model = TrainingModule(nn_network, train_dataset, **config)
    trainer = pl.Trainer(max_epochs=max_epoch, enable_progress_bar=False, log_every_n_steps=10,
                         logger=TensorBoardLogger(save_dir="ray_logs", name="GIN"),
                         gpus=math.ceil(num_gpus),
                         callbacks=[
                             TuneReportCallback({
                                 "loss": "epoch/val_loss",
                                 "auc": "epoch/val_auc"}, on="validation_end")
                         ])
    trainer.fit(model)


def tune_with_asha(custom_nn_model, custom_dataset, num_samples=100, num_epochs=10, gpus_per_trial=0):
    config_grid = {
        "in_channels": custom_dataset.feature_dim,
        "hidden_channels": tune.choice([32, 64]),
        "num_layers": tune.choice([2, 3, 5, 7, 9]),
        "out_channels": 1,
        "dropout": tune.uniform(0.05, 0.9),
        "learning_rate": tune.loguniform(1e-5, 1e-1),
        "weight_decay": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([128, 512, 1028])
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["hidden_channels", "num_layers", "dropout", "learning_rate", "weight_decay", "batch_size"],
        metric_columns=["loss", "auc", "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train_model_wrapper, nn_network=custom_nn_model,
                                                    train_dataset=custom_dataset, num_gpus=gpus_per_trial)

    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

    analysis = tune.run(train_fn_with_parameters,
                        resources_per_trial=resources_per_trial,
                        metric="auc",
                        mode="max",
                        config=config_grid,
                        num_samples=num_samples,
                        scheduler=scheduler,
                        progress_reporter=reporter,
                        name="GIN_tune_with_asha")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    tune_with_asha(GIN_SubgraphX, reveal_dataset, gpus_per_trial=0.5)
