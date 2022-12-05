import random
from typing import Sequence, Tuple

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset


class _SimpleDataset(Dataset):
    def __init__(self, data: Sequence, labels: Sequence):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.data)


class ModularAdditionDataModule(LightningDataModule):
    def __init__(
        self,
        p: int,
        frac_train: float = 0.3,
        train_batch_size: int = 256,
        val_batch_size: int = 256,
        full_batch: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.collate_fn = None
        # Input format is x|y|=, d_vocab=114 (integers from 0 to p−1 and =)
        # Note that with this format, the value of p also happens to be the index of the "=" token,
        # hence our input of x|y|= maps to token idxs of x|y|p
        pairs = [(x, y, p) for x in range(p) for y in range(p)]
        random.Random(seed).shuffle(pairs)
        labels = [(-1, -1, (x + y) % p) for x, y, _ in pairs]
        div = int(frac_train * len(pairs))
        train_data = torch.tensor(pairs[:div], dtype=torch.long)
        train_labels = torch.tensor(labels[:div], dtype=torch.long)
        val_data = torch.tensor(pairs[div:], dtype=torch.long)
        val_labels = torch.tensor(labels[div:], dtype=torch.long)

        if full_batch:
            train_data, train_labels = train_data.unsqueeze(0), train_labels.unsqueeze(0)
            val_data, val_labels = val_data.unsqueeze(0), val_labels.unsqueeze(0)
            # override any given batch sizes
            # train_batch_size = len(train_data)
            # val_batch_size = len(val_data)
            train_batch_size = val_batch_size = 1

            def no_op_collate(batch):
                return batch[0]

            self.collate_fn = no_op_collate

        self.train_dataset = _SimpleDataset(train_data, train_labels)
        self.val_dataset = _SimpleDataset(val_data, val_labels)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size, collate_fn=self.collate_fn
        )
