from abc import ABC

import torch
from torch_geometric.data import Batch
from pytorch_lightning import LightningModule
from sklearn.metrics import f1_score, accuracy_score


class TrainingModule(LightningModule, ABC):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = model(**kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index, batch=None) -> torch.Tensor:
        return self.model(x, edge_index, batch)

    def evaluate(self, batch, stage=None):
        y_hat = self(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(y_hat, batch.y)
        preds = torch.argmax(y_hat.softmax(dim=1), dim=1)
        acc = accuracy_score(preds.tolist(), batch.y.tolist())
        f1 = f1_score(preds.tolist(), batch.y.tolist(), average='binary')

        if stage:
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_acc", acc, on_step=True, on_epoch=True, logger=True)
            self.log(f"{stage}_f1", f1, on_step=True, on_epoch=True, logger=True)

        return loss

    def training_step(self, batch: Batch, batch_idx: int):
        return self.evaluate(batch, 'train')

    def validation_step(self, batch: Batch, batch_idx: int):
        self.evaluate(batch, 'val')

    def test_step(self, batch: Batch, batch_idx: int):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
