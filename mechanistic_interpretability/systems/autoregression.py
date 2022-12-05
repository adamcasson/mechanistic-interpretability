from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from mechanistic_interpretability.utils import per_token_cross_entropy


class AutoregressiveDecoderOnly(LightningModule):
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
        loss, per_token_loss, logits = self._shared_step(batch)

        return {'loss': loss, 'per_token_loss': per_token_loss, 'logits': logits}

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: int = 0
    ) -> None:
        self.logger.experiment.add_scalars(
            'loss',
            {'train_loss': outputs['loss']},
            global_step=self.global_step,
        )

    def validation_step(self, batch: Tuple, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        loss, per_token_loss, logits = self._shared_step(batch)

        return {'loss': loss, 'per_token_loss': per_token_loss, 'logits': logits}

    def on_validation_batch_end(
        self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self.logger.experiment.add_scalars(
            'loss',
            {'val_loss': outputs['loss']},
            global_step=self.global_step,
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        self._shared_step(batch)

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple]:
        optimizer = self.optimizer_partial(self.parameters())
        if self.lr_scheduler_partial is not None:
            scheduler = self.lr_scheduler_partial(optimizer)

            return [optimizer], [scheduler]

        return optimizer

    def _shared_step(self, batch: Tuple) -> Tensor:
        logits = self.model(batch[0])
        targets = batch[1]

        loss, per_token_loss = per_token_cross_entropy(logits, targets, ignore_index=-1)

        return loss, per_token_loss, logits
