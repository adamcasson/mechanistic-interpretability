import torch.nn as nn
from torch import Tensor


class Add(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return a + b


class Mul(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return a * b


class MatMul(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return a @ b
