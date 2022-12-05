from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def cross_entropy_high_precision(
    input: Tensor, targets: Tensor, ignore_index: int = -100
) -> Tensor:
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes
    # and dodgy gradients
    valid_input = input[targets != ignore_index]
    valid_targets = targets[targets != ignore_index]
    logprobs = F.log_softmax(valid_input.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=valid_targets[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss


def per_token_cross_entropy(
    input: Tensor, targets: Tensor, ignore_index: int = -100
) -> Tuple[Tensor, Tensor]:
    # Shapes: batch x sequence x vocab, batch x sequence
    batch_size = input.size(0)
    input = rearrange(input, 'b s v -> (b s) v')
    targets = rearrange(targets, 'b s -> (b s)')

    valid_input = input[targets != ignore_index]
    valid_targets = targets[targets != ignore_index]

    logprobs = F.log_softmax(valid_input.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=valid_targets[:, None], dim=-1)
    mean_loss = -torch.mean(prediction_logprobs)

    positional_loss = torch.zeros_like(targets, dtype=torch.float64)
    mask = targets != -1

    positional_loss[torch.where(mask)[0]] = prediction_logprobs.squeeze()

    positional_loss = rearrange(positional_loss, '(b s) -> b s', b=batch_size)
    mask = rearrange(mask, '(b s) -> b s', b=batch_size)

    # mean_loss = -(positional_loss.sum() / mask.sum())
    mean_positional_loss = -(positional_loss.sum(0) / mask.sum(0))

    return mean_loss, mean_positional_loss


def lr_lambda(step: int) -> int:
    """Gradually ramps up LR by 10% (starting at 10% for the first epoch) each epoch until epoch 10 where the full LR is used onwards.

    code source: https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20#scrollTo=BhhJmRH8IIvy

    Args:
        step: current epoch

    Returns:
        int: LR multiplier
    """
    # step + 1 because otherwise epoch 0 will have a multipler of 0.0 which is pointless
    return min(step + 1 / 10, 1)
