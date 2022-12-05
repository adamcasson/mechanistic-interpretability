from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT

from mechanistic_interpretability.systems.autoregression import (
    AutoregressiveDecoderOnly,
)


class DigitAddition(AutoregressiveDecoderOnly):
    def __init__(
        self,
        model: nn.Module,
        optimizer_partial: Optional[Callable] = None,
        lr_scheduler_partial: Optional[Callable] = None,
    ) -> None:
        super().__init__(model, optimizer_partial, lr_scheduler_partial)

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, unused: int = 0
    ) -> None:
        per_token_loss = outputs['per_token_loss']

        self.logger.experiment.add_scalars(
            'per_token_train_loss',
            {
                f'token_{i}': per_token_loss[i]
                for i in range(self.model.n_ctx)
                if not torch.isnan(per_token_loss[i]).item()
            },
            global_step=self.global_step,
        )

        self.logger.experiment.add_scalars(
            'loss',
            {'train_loss': outputs['loss']},
            global_step=self.global_step,
        )

        argmax_per_token_logits = outputs['logits'].argmax(-1)
        for i in range(5, 9):
            self.logger.experiment.add_histogram(
                f'token_{i}_train_predictions',
                argmax_per_token_logits[:, i],
                global_step=self.global_step,
                bins=10,
            )

    def on_validation_batch_end(
        self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        per_token_loss = outputs['per_token_loss']

        self.logger.experiment.add_scalars(
            'per_token_val_loss',
            {
                f'token_{i}': per_token_loss[i]
                for i in range(self.model.n_ctx)
                if not torch.isnan(per_token_loss[i]).item()
            },
            global_step=self.global_step,
        )

        self.logger.experiment.add_scalars(
            'loss',
            {'val_loss': outputs['loss']},
            global_step=self.global_step,
        )

        argmax_per_token_logits = outputs['logits'].argmax(-1)
        for i in range(5, 9):
            self.logger.experiment.add_histogram(
                f'token_{i}_val_predictions',
                argmax_per_token_logits[:, i],
                global_step=self.global_step,
                bins=10,
            )
