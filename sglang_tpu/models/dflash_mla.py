# SPDX-License-Identifier: Apache-2.0
"""MLA DFlash draft attention served as MHA over the expanded latent.

SpecForge MLA drafts (``dflash_config.attention_mode == "mla"``, e.g. the
Kimi-K3.1 DFlash draft) keep DeepSeek MLA weights. The draft context is a short
sliding window and the TPU draft pool is the fused per-layer MHA pool, so the
latent is expanded through ``kv_b_proj`` into per-head K ``[k_nope | k_pe]`` and
V, served by the same RPA backend and fused KV writer as GQA drafts. The MLA
math is unchanged: interleaved (non-neox) YaRN rope on the rope dims and the
mscale^2 softmax scale.

``sglang_tpu.hooks.dflash_mla`` selects this class for MLA drafts and gives the
draft ModelConfig the expanded head dims.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from sglang.srt.configs.model_config import compute_mla_mscale_scaling
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.dflash import _get_dflash_layer_attention_params
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_utils import _get_dflash_config
from sglang.srt.utils.hf_transformers_utils import get_rope_config


def is_dflash_mla_draft(config) -> bool:
    """True when ``dflash_config.attention_mode == "mla"`` (SpecForge MLA draft layers)."""
    mode = _get_dflash_config(config).get("attention_mode", "gqa")
    return str(mode).lower() == "mla"


def mla_draft_head_dims(text_config) -> tuple[int, int]:
    """(qk head_dim, v head_dim) of the expanded MHA view of an MLA draft."""
    return (
        int(text_config.qk_nope_head_dim) + int(text_config.qk_rope_head_dim),
        int(text_config.v_head_dim),
    )


class DFlashMLAAttention(nn.Module):
    """MLA draft attention as MHA: every local head is its own KV head."""

    def __init__(
        self, config, layer_id: int, quant_config=None, prefix: str = ""
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise ValueError("DFLASH MLA drafts require unquantized draft weights.")
        tp_size = int(get_parallel().tp_size)
        hidden_size = int(config.hidden_size)
        total_num_heads = int(config.num_attention_heads)
        assert total_num_heads % tp_size == 0, (total_num_heads, tp_size)
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = self.num_heads
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.head_dim, self.v_head_dim = mla_draft_head_dims(config)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                hidden_size, self.q_lora_rank, bias=False, prefix="q_a_proj"
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                total_num_heads * self.head_dim,
                bias=False,
                prefix="q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                hidden_size,
                total_num_heads * self.head_dim,
                bias=False,
                prefix="q_proj",
            )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            prefix="kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            prefix="kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            total_num_heads * self.v_head_dim, hidden_size, bias=False, prefix="o_proj"
        )

        rope_theta, rope_scaling = get_rope_config(config)
        rope_type = (
            rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_scaling
            else None
        )
        if rope_type in (None, "default"):
            rope_scaling = None
        self.scaling = self.head_dim**-0.5
        if rope_scaling is not None:
            rope_scaling = dict(rope_scaling, rope_type="deepseek_yarn")
            self.scaling = compute_mla_mscale_scaling(rope_scaling, self.scaling)
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=int(config.max_position_embeddings),
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=not bool(getattr(config, "rope_interleave", True)),
        )
        self.sliding_window_size, self.attn_type = _get_dflash_layer_attention_params(
            config, layer_id
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            sliding_window_size=self.sliding_window_size,
            attn_type=self.attn_type,
        )

    def _project_q(self, hidden_states: torch.Tensor):
        if self.q_lora_rank is not None:
            q_lat, _ = self.q_a_proj(hidden_states)
            q, _ = self.q_b_proj(self.q_a_layernorm(q_lat))
        else:
            q, _ = self.q_proj(hidden_states)
        return q.view(-1, self.num_heads, self.head_dim).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

    def _project_kv(self, hidden_states: torch.Tensor):
        """k_nope [n, h, nope], k_pe (unrotated) [n, 1, rope], v [n, h, v_head_dim]."""
        latent, _ = self.kv_a_proj_with_mqa(hidden_states)
        kv_c, k_pe = latent.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv, _ = self.kv_b_proj(self.kv_a_layernorm(kv_c))
        k_nope, v = kv.view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return k_nope, k_pe.unsqueeze(1), v

    def _join_k(self, k_nope: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [k_nope, k_pe.expand(-1, self.num_heads, self.qk_rope_head_dim)], dim=-1
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q_nope, q_pe = self._project_q(hidden_states)
        k_nope, k_pe, v = self._project_kv(hidden_states)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = self._join_k(k_nope, k_pe)
        attn_output = self.attn(
            q.reshape(-1, self.num_heads * self.head_dim),
            k.reshape(-1, self.num_kv_heads * self.head_dim),
            v.reshape(-1, self.num_kv_heads * self.v_head_dim),
            forward_batch,
        )
        output, _ = self.o_proj(attn_output)
        return output

    def project_context_kv_rotated(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotated K [n, h, head_dim] and V [n, h, v_head_dim] of context tokens."""
        k_nope, k_pe, v = self._project_kv(hidden_states)
        # Only k_pe is rotated; the q slot is a placeholder of the same shape.
        _, k_pe = self.rotary_emb(positions, k_pe, k_pe)
        return self._join_k(k_nope, k_pe), v
