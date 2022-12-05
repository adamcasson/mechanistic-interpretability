import math
from typing import Literal

import einops
import torch
import torch.nn as nn
from torch import Tensor

from mechanistic_interpretability.models import activations, ops

SupportedAct = Literal['relu', 'gelu', 'identity', 'solu']

ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'gelu': torch.nn.GELU,
    'identity': torch.nn.Identity,
    'solu': activations.SoLU,
    'solu_ln': activations.SoLULN,
}


class CausalMask(nn.Module):
    def __init__(self, n_ctx: int) -> None:
        super().__init__()
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))

    def forward(self, attn_scores: Tensor, T: int) -> Tensor:
        return torch.tril(attn_scores) - 1e10 * (1 - self.mask[:T, :T])


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, n_ctx: int) -> None:
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_head * n_heads, bias=False)
        self.W_O = nn.Linear(d_head * n_heads, d_model, bias=False)

        self.causal_mask = CausalMask(n_ctx)

        self.n_heads = n_heads
        self.d_head = d_head

        self.dot_prod = ops.MatMul()
        self.attn_softmax = nn.Softmax(dim=-1)
        self.apply_attn = ops.MatMul()

    def forward(self, x: Tensor) -> Tensor:
        T = x.size(1)
        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)

        q, k, v = map(
            lambda t: einops.rearrange(
                t, 'B T (nh dh) -> B nh T dh', nh=self.n_heads, dh=self.d_head
            ),
            (q, k, v),
        )

        attn_scores_pre = self.dot_prod(q, k.transpose(-2, -1))
        attn_scores_masked = self.causal_mask(attn_scores_pre, x.shape[-2])
        attn_matrix = self.attn_softmax(attn_scores_masked / math.sqrt(self.d_head))
        z = self.apply_attn(attn_matrix, v)
        z_flat = einops.rearrange(z, 'B nh T dh -> B T (nh dh)')
        out = self.W_O(z_flat)

        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, act_type: SupportedAct) -> None:
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)

        # TODO: deal with solu_ln/activation signature(s) in a cleaner way
        if act_type == 'solu_ln':
            self.act_func = ACTIVATIONS[act_type](in_features=d_mlp)
        else:
            self.act_func = ACTIVATIONS[act_type]()

    def forward(self, x: Tensor) -> Tensor:
        x = self.W_in(x)
        x = self.act_func(x)
        x = self.W_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        d_head: int,
        n_heads: int,
        n_ctx: int,
        act_type: SupportedAct,
    ) -> None:
        super().__init__()

        self.attn = Attention(d_model, n_heads, d_head, n_ctx)
        self.mlp = MLP(d_model, d_mlp, act_type)

        self.residual_attn_add = ops.Add()
        self.residual_mlp_add = ops.Add()

    def forward(self, x: Tensor) -> Tensor:
        x = self.residual_attn_add(x, self.attn(x))
        x = self.residual_mlp_add(x, self.mlp(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_vocab: int,
        d_model: int,
        d_mlp: int,
        d_head: int,
        n_heads: int,
        n_ctx: int,
        act_type: SupportedAct,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_ctx = n_ctx
        self.embed = nn.Embedding(d_vocab, d_model)
        self.pos_embed = nn.Embedding(n_ctx, d_model)
        self.add_pos_embed = ops.Add()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, d_mlp, d_head, n_heads, n_ctx, act_type)
                for _ in range(n_layers)
            ]
        )

        self.unembed = nn.Linear(d_model, d_vocab)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('embed.weight') or pn.endswith('pos_embed.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(d_model))
            if pn.endswith('unembed.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(d_vocab))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.embed(x)
        x = self.add_pos_embed(x, self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        return x
