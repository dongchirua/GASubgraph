import pytorch_lightning as pl
from pytorch_lightning import Trainer

from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)

from vulexp.data_models.reveal_data import Reveal
from vulexp.ml_models.gnn import GNN
from vulexp.ml_models.pl_train_module_logit import TrainingModule
import subprocess
import os
from pathlib import Path

# n_epoch = 101
# absolute_path = os.getcwd()
# data_dir = 'data/reveal'
# reveal_dataset = Reveal(data_dir, absolute_path=absolute_path)
from datetime import datetime


def gitsha():
    """Get current git commit sha for reproducibility."""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode()
    )


def gitmessage():
    """Get current git commit sha for reproducibility."""
    m = subprocess.check_output(["git", "log", "-1", "--format=%s"]).strip().decode()
    return "_".join(m.lower().split())


def get_run_id(args=None):
    """Generate run ID."""
    if not args:
        ID = datetime.now().strftime("%Y%m%d%H%M_{}".format(gitsha()))
        return ID + "_" + gitmessage()
    ID = datetime.now().strftime(
        "%Y%m%d%H%M_{}_{}".format(
            gitsha(), "_".join([f"{v}" for _, v in vars(args).items()])
        )
    )
    return ID


def train_model(config, custom_nn_model, save_path, custom_dataset, num_epochs=10, num_gpus=1, num_workers=8,
                input_channel=None, n_class=None):
    if input_channel is not None:
        config['input_channel'] = input_channel
    if n_class is not None:
        config['n_class'] = n_class
    model = TrainingModule(custom_nn_model, custom_dataset, num_workers=num_workers, **config)

    metrics = {"loss": "ptl/val_loss",
               "f1": "ptl/val_f1",
               "mean_accuracy": "ptl/val_accuracy"}

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="ptl/val_loss", mode='min')
    raytune_callback = TuneReportCallback(metrics, on="validation_end")
    rtckpt_callback = TuneReportCheckpointCallback(metrics, on="validation_end")

    trainer = Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        default_root_dir=save_path,
        logger=TensorBoardLogger(save_dir=f'{save_path}/tensor_logs', name=custom_nn_model.name),
        callbacks=[raytune_callback, rtckpt_callback, checkpoint_callback]
    )
    trainer.fit(model)
    trainer.test(model)


def tune_ashas_scheduler(config_grid, custom_nn_model, custom_dataset,
                         input_channel, n_class, name,
                         num_samples=250, max_epochs=10,
                         gpus_per_trial=0, **kwargs):
    cwd = os.getcwd()
    run_id = get_run_id()
    store_path = os.path.join(cwd, 'raytune_feature', run_id)
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)

    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=list(config_grid.keys()),
        metric_columns=["loss", "mean_accuracy", "f1"])

    train_fn_with_parameters = tune.with_parameters(train_model,
                                                    save_path=store_path,
                                                    n_class=n_class,
                                                    input_channel=input_channel,
                                                    custom_nn_model=custom_nn_model,
                                                    custom_dataset=custom_dataset,
                                                    num_epochs=max_epochs,
                                                    num_gpus=gpus_per_trial)
    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

    analysis = tune.run(train_fn_with_parameters,
                        resources_per_trial=resources_per_trial,
                        metric="loss",
                        mode="min",
                        config=config_grid,
                        num_samples=num_samples,
                        scheduler=scheduler,
                        local_dir=store_path,
                        progress_reporter=reporter,
                        keep_checkpoints_num=2,
                        checkpoint_score_attr="min-val_loss",
                        name=name)

    print("Best hyperparameters found were: ", analysis.best_config)
