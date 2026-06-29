# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional
import warnings

from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

XFORMERS_AVAILABLE = False


@dataclass
class FIAComputeParams:
    q: Tensor
    k: Tensor
    v: Tensor
    causal: bool = False
    scale: Optional[float] = None
    dropout_p: float = 0.0


def attention_probs(q: Tensor, k: Tensor, scale: float, attn_drop: nn.Dropout, training: bool) -> Tensor:
    q = q * scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    return attn_drop(attn) if training else attn


def manual_attention(q: Tensor, k: Tensor, v: Tensor, scale: float, attn_drop: nn.Dropout, training: bool) -> Tensor:
    return attention_probs(q, k, scale, attn_drop, training) @ v


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        is_global_attention: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.is_global_attention = is_global_attention

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, save_vis_attn=False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if save_vis_attn:
            with torch.no_grad():
                self.last_attn = attention_probs(q.detach(), k.detach(), self.scale, self.attn_drop, False)

        if self.fused_attn and self.is_global_attention:
            x = self._compute_attention_with_fia(
                FIAComputeParams(
                    q=q,
                    k=k,
                    v=v,
                    causal=False,
                    scale=self.scale,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            )
        elif self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
                scale=self.scale,
            )
        else:
            x = manual_attention(q, k, v, self.scale, self.attn_drop, self.training)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _compute_attention_with_fia(self, params: FIAComputeParams) -> Tensor:
        q, k, v = params.q, params.k, params.v
        _, num_heads, _, head_dim = q.shape
        scale = params.scale if params.scale is not None else 1.0 / math.sqrt(head_dim)

        if torch_npu is None or q.device.type != "npu":
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=params.dropout_p,
                is_causal=params.causal,
                scale=scale,
            )

        if params.dropout_p > 0.0 and self.training:
            logging.warning(
                "FIA does not support dropout (dropout_p=%s), falling back to SDPA",
                params.dropout_p,
            )
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=params.dropout_p,
                is_causal=params.causal,
                scale=scale,
            )

        num_key_value_heads = k.shape[1]
        if num_heads == num_key_value_heads:
            num_key_value_heads = 0

        try:
            out = torch_npu.npu_fused_infer_attention_score(
                q,
                k,
                v,
                num_heads=num_heads,
                scale=float(scale),
                input_layout="BNSD",
                num_key_value_heads=num_key_value_heads,
                pre_tokens=65535,
                next_tokens=65535 if not params.causal else 0,
                sparse_mode=0,
                inner_precise=0,
            )[0]
        except RuntimeError as exc:
            logging.warning("FIA failed, falling back to SDPA: %s", exc)
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=params.dropout_p,
                is_causal=params.causal,
                scale=scale,
            )
        return out.to(q.dtype)


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, save_vis_attn=False) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x,save_vis_attn=save_vis_attn)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
