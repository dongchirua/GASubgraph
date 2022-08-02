import statistics
from abc import ABC

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


class TrainingModule(LightningModule):

    def __init__(self, model, dataset, batch_size: int = 100, num_workers: int = 4, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = model(**kwargs)
        self.dataset = dataset
        self.criterion = torch.nn.CrossEntropyLoss()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def forward(self, x, edge_index, batch) -> torch.Tensor:
        return self.model(x, edge_index, batch)

    def evaluate(self, batch, stage):
        y_hat = self(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(y_hat, batch.y)
        preds = torch.argmax(y_hat.softmax(dim=1), dim=1)
        acc = accuracy_score(preds.tolist(), batch.y.tolist())
        f1 = f1_score(preds.tolist(), batch.y.tolist(), average='binary')

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log(f"{stage}_acc", acc, on_step=True, on_epoch=True, logger=True)
        self.log(f"{stage}_f1", f1, on_step=True, on_epoch=True, logger=True)

        return {"loss": loss, f"accuracy": acc, "f1": f1}

    def training_step(self, batch: Batch, batch_idx: int):
        return self.evaluate(batch, 'train')

    def validation_step(self, batch: Batch, batch_idx: int):
        return self.evaluate(batch, 'val')

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = statistics.mean([x["accuracy"] for x in outputs])
        avg_f1 = statistics.mean([x["f1"] for x in outputs])
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)
        self.log("ptl/val_f1", avg_f1)

    def test_step(self, batch: Batch, batch_idx: int):
        return self.evaluate(batch, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

    def prepare_data(self):
        train_dataset, val_dataset, test_dataset = self.dataset.generate_train_test()
        self.train_dataset = train_dataset.shuffle()
        self.val_dataset = val_dataset.shuffle()
        self.test_dataset = test_dataset.shuffle()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
