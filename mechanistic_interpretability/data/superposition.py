from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset


class _SuperpositionDataset(IterableDataset):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        feature_probability: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,
        batch_size: int = 1024,
    ):

        self.n_instances = n_instances
        self.n_features = n_features

        if feature_probability is None:
            feature_probability = torch.ones(())
        self.feature_probability = feature_probability
        if importance is None:
            importance = torch.ones(())
        self.importance = importance

        self.batch_size = batch_size

    def __iter__(self) -> Tensor:
        feat = torch.rand((self.batch_size, self.n_instances, self.n_features))
        batch = torch.where(
            torch.rand(
                (self.batch_size, self.n_instances, self.n_features),
            )
            <= self.feature_probability,
            feat,
            torch.zeros(()),
        )
        yield batch, self.importance


class SuperpositionDataModule(LightningDataModule):
    def __init__(
        self,
        n_instances: int,
        n_features: int,
        feature_probability: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,
        batch_size: int = 1024,
    ) -> None:
        super().__init__()

        self.train_dataset = _SuperpositionDataset(
            n_instances=n_instances,
            n_features=n_features,
            feature_probability=feature_probability,
            importance=importance,
            batch_size=batch_size,
        )

        # sloppy way to skip pytorch dataloader batching/collation and just delegate
        # batching to the dataset since it's synthetic data that we can create on the fly
        def no_op_collate(batch):
            return batch[0]

        self.collate_fn = no_op_collate

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=1, collate_fn=self.collate_fn)


def exponential_importance(n_features: int) -> Tensor:
    """code source: https://github.com/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb"""
    # Exponential feature importance curve from 1 to 1/100
    importance = (0.9 ** torch.arange(n_features))[None, :]
    return importance


def feature_frequency_sweep(n_instances: int) -> Tensor:
    """code source: https://github.com/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb"""
    # Sweep feature frequency across the instances from 1 (fully dense) to 1/20
    feature_probability = (20 ** -torch.linspace(0, 1, n_instances))[:, None]
    return feature_probability
