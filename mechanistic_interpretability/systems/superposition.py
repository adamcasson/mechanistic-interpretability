from typing import Any, Callable, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT


class Superposition(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_partial: Optional[Callable] = None,
        lr_scheduler_partial: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.optimizer_partial = optimizer_partial
        self.lr_scheduler_partial = lr_scheduler_partial

    def training_step(self, batch: Tuple, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        """code source: https://github.com/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb"""
        features, importance = batch
        out = self.model(features)
        error = importance * (features.abs() - out) ** 2
        loss = einops.reduce(error, 'b i f -> i', 'mean').sum()

        self.log('loss', loss)

        return loss

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple]:
        optimizer = self.optimizer_partial(self.parameters())
        if self.lr_scheduler_partial is not None:
            scheduler = self.lr_scheduler_partial(optimizer)

            return [optimizer], [scheduler]

        return optimizer
