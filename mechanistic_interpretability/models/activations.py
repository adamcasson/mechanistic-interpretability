import torch.nn as nn
from torch import Tensor

from mechanistic_interpretability.models import ops


class SoLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.mul = ops.Mul()

    def forward(self, x: Tensor) -> Tensor:
        return self.mul(x, self.softmax(x))


class SoLULN(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.solu = SoLU()
        self.ln = nn.LayerNorm(in_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.ln(self.solu(x))
