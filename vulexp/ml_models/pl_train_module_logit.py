import os
import torch
import torch.nn.functional as F
from filelock import FileLock
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


class TrainingModule(LightningModule):

    def __init__(self, model: torch.nn.Module, dataset, batch_size: int = 100, num_workers: int = 0, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = model(**kwargs)
        self.dataset = dataset
        self.criterion = F.binary_cross_entropy_with_logits

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.weight_decay = kwargs.get('weight_decay', 5e-4)
        self.threshold = kwargs.get('threshold', 0.5)

    def __str__(self):
        str(self.model)

    def forward(self, x, edge_index, batch) -> torch.Tensor:
        return self.model(x, edge_index, batch)

    def loss_and_pred(self, batch, stage=None):
        out = self(batch.x, batch.edge_index, batch.batch)
        out = out.squeeze()

        pred = torch.sigmoid(out)
        target = batch.y.float().squeeze()

        loss = self.criterion(input=out, target=target)

        if stage:
            self.log(f"batch/{stage}/loss", loss, on_step=True, on_epoch=False, logger=True)

        return {
            "loss": loss,
            "preds": pred.detach().cpu().tolist(),
            "targets": target.detach().cpu().tolist()
        }

    def training_step(self, batch: Batch, batch_idx: int):
        return self.loss_and_pred(batch, 'train')

    def validation_step(self, batch: Batch, batch_idx: int):
        return self.loss_and_pred(batch, 'val')

    @staticmethod
    def compute_auc(outputs, thresholds):
        preds = []
        targets = []

        for output in outputs:
            preds += output['preds']
            targets += output['targets']

        fpr, tpr, thresholds = roc_curve(y_true=targets, y_score=preds, pos_label=1)
        auc_positive = auc(fpr, tpr)
        auc_macro = roc_auc_score(y_true=targets, y_score=preds)
        meta_info = {'tpr': tpr, 'fpr': fpr}
        return auc_positive, auc_macro, meta_info

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        preds = []
        targets = []

        for output in outputs:
            preds += output['preds']
            targets += output['targets']

        auc_positive, auc_macro, _ = TrainingModule.compute_auc(outputs, self.threshold)
        f1 = self.use_threshold(preds, targets, self.threshold)

        self.log("epoch/val/loss", avg_loss)
        self.log("epoch/val/auc_positive", auc_positive)
        self.log("epoch/val/auc_macro", auc_macro)
        self.log("epoch/val/f1", f1)

    def test_epoch_end(self, outputs):
        preds = []
        targets = []

        for output in outputs:
            preds += output['preds']
            targets += output['targets']

        auc_positive, auc_macro, _ = TrainingModule.compute_auc(outputs, self.threshold)
        f1 = self.use_threshold(preds, targets, self.threshold)

        self.log("epoch/test/auc_positive", auc_positive)
        self.log("epoch/test/auc_macro", auc_macro)
        self.log("epoch/test/f1", f1)

    def test_step(self, batch: Batch, batch_idx: int):
        return self.loss_and_pred(batch, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @staticmethod
    def serialize(data_set):
        with FileLock(os.path.expanduser("./data_undirected.lock")):
            return data_set.generate_train_test()

    def prepare_data(self):
        self.train_dataset, self.val_dataset, self.test_dataset = self.serialize(self.dataset)
        # self.train_dataset, self.val_dataset, self.test_dataset = self.dataset.generate_train_test()
        # print('size of train, val, test ', len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # note: https://github.com/pytorch/pytorch/issues/66482
        # don't use worker in DataLoader when use ray
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict(self, sample: Data, device='cpu'):
        self.eval()
        batch = torch.zeros(sample.x.shape[0], dtype=int, device=device)
        out = self(sample.x, sample.edge_index, batch)
        pred_prob = torch.sigmoid(out.squeeze())
        return pred_prob

    @staticmethod
    def use_threshold(preds: list, targets: list, threshold):
        y_hat = [1 if i > threshold else 0 > threshold for i in preds]
        f1 = f1_score(y_hat, targets, average='binary')
        return f1

    def predicts(self, samples: Batch, threshold: float = 0.5, device='cpu'):
        self.eval()
        out = self.loss_and_pred(samples)
        preds = out['preds']
        targets = out['targets']
        f1 = self.use_threshold(preds, targets, threshold)
