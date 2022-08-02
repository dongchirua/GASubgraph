from typing import Optional
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch_geometric.loader import DataLoader


class DataModule(LightningDataModule):

    def __init__(self, dataset, batch_size=64, num_workers=1):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def num_features(self) -> int:
        return 64

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        train_dataset, val_dataset, test_dataset = self.dataset.generate_train_test()
        self.train_dataset = train_dataset.shuffle()
        self.val_dataset = val_dataset.shuffle()
        self.test_dataset = test_dataset.shuffle()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
