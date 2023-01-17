#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import torch
from pytorch_lightning import Trainer
from vulexp.data_models.reveal_data import Reveal
from vulexp.ml_models.pl_train_module_logit import TrainingModule
from vulexp.ml_models.gin import GIN

if __name__ == '__main__':
    @dataclass
    class Args:
        seed: int = 27  # Random seed.
        to_undirected = False
        gtype = 'cpg'

    args = Args()

    data_dir = 'data/reveal/'
    reveal_dataset = Reveal(data_dir, args.gtype, to_undirected=args.to_undirected, seed=args.seed)

    reveal_train, reveal_val, reveal_test = reveal_dataset.generate_train_test()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    saved_model = TrainingModule.load_from_checkpoint(model=GIN, map_location=device,
                                                      checkpoint_path="solo_train/202301171249_88cc311_add_click/checkpoint/GIN_CPG-epoch=04-loss=0.57-f1=0.29189189189189185.ckpt")
    saved_model.to(device)
    saved_model.eval()

    trainer = Trainer()
    saved_model.dataset = reveal_dataset
    trainer.test(saved_model)
