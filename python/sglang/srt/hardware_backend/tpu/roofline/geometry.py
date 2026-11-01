# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model geometry and layer-family validator from HuggingFace config.json."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

LAYER_FAMILIES: tuple[str, ...] = (
    "gqa",
    "mla",
    "dsa",
    "swa",
    "gdn",
    "kda",
    "dense_mlp",
    "moe",
    "mtp",
    "ple",
)

_VALID_LAYER_TOKENS: frozenset[str] = frozenset(
    {
        "full_attention",
        "attention",
        "gqa",
        "mla",
        "mla_attention",
        "dsa",
        "dsa_attention",
        "sparse_mla",
        "sliding_attention",
        "swa",
        "sliding_window_attention",
        "linear_attention",
        "gdn",
        "gated_deltanet",
        "mamba",
        "kda",
        "kimi_linear",
        "kda_attention",
        "dense_mlp",
        "mlp",
        "moe",
        "mtp",
        "ple",
    }
)

_UNSUPPORTED_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "clip",
        "siglip",
        "vit",
        "whisper",
        "flux",
        "sdxl",
        "autoencoder_kl",
    }
)

_GDN_TOKENS: frozenset[str] = frozenset(
    {"linear_attention", "gdn", "gated_deltanet", "mamba"}
)
_KDA_TOKENS: frozenset[str] = frozenset({"kda", "kimi_linear", "kda_attention"})


@dataclass(frozen=True)
class ModelGeometry:
    """Architecture dimensions shared across attention, recurrence, and FFN families."""

    hidden: int
    num_layers: int
    layer_types: tuple[str, ...]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    attn_output_gate: bool
    linear_k_heads: int
    linear_v_heads: int
    linear_k_dim: int
    linear_v_dim: int
    conv_kernel: int
    num_experts: int
    top_k: int
    moe_inter: int
    shared_inter: int
    vocab: int
    mtp_layers: int
    dense_inter: int = 0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    sliding_window: int = 0
    num_shared_experts: int = 0
    first_k_dense_replace: int = 0
    ple_dim: int = 0
    has_attention_sink: bool = False
    model_type: str = ""
    architectures: tuple[str, ...] = ()

    @staticmethod
    def from_hf_config(cfg: Mapping[str, Any]) -> ModelGeometry:
        return from_hf_config(cfg)


def _extract_text_config(
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        raw = dict(cfg)
    except Exception:
        return None
    sub = raw.get("text_config")
    if sub is not None and type(sub) is dict:
        return raw, dict(sub)
    return raw, raw


def _resolve_layer_types(
    tc: dict[str, Any],
    num_layers: int,
    kv_lora_rank: int,
    index_topk: int,
    sliding_window: int,
) -> tuple[str, ...] | None:
    raw_types = tc.get("layer_types")
    if raw_types is not None:
        if type(raw_types) not in (list, tuple) or len(raw_types) != num_layers:
            return None
        tokens = tuple(str(x) for x in raw_types)
        if any(tok not in _VALID_LAYER_TOKENS for tok in tokens):
            return None
        return tokens

    model_type = str(tc.get("model_type", "")).lower()
    if model_type in ("gpt_oss", "gpt-oss") and sliding_window > 0:
        return tuple(
            "sliding_attention" if i % 2 == 0 else "full_attention"
            for i in range(num_layers)
        )

    interval = int(tc.get("full_attention_interval", 0) or 0)
    if interval > 0:
        att_tok = (
            "dsa"
            if kv_lora_rank > 0 and index_topk > 0
            else "mla"
            if kv_lora_rank > 0
            else "full_attention"
        )
        lin_tok = "kda" if model_type.startswith("kimi") else "linear_attention"
        return tuple(
            att_tok if (i + 1) % interval == 0 else lin_tok for i in range(num_layers)
        )
    if kv_lora_rank > 0:
        default_mla = "dsa" if index_topk > 0 else "mla"
        return (default_mla,) * num_layers
    return ("full_attention",) * num_layers


def supports(cfg: Mapping[str, Any]) -> bool:
    """Validate whether a HuggingFace config mapping can be priced by the TPU roofline."""
    extracted = _extract_text_config(cfg)
    if extracted is None:
        return False
    raw, tc = extracted
    model_type = str(tc.get("model_type") or raw.get("model_type") or "").lower()
    if model_type in _UNSUPPORTED_MODEL_TYPES:
        return False
    try:
        hidden = int(tc.get("hidden_size", 0) or 0)
        num_layers = int(tc.get("num_hidden_layers", 0) or 0)
        num_heads = int(tc.get("num_attention_heads", 0) or 0)
        if hidden <= 0 or num_layers <= 0 or num_heads <= 0:
            return False
        num_kv_heads = int(tc.get("num_key_value_heads", num_heads) or 0)
        if num_kv_heads <= 0:
            return False
        head_dim = int(tc.get("head_dim") or (hidden // num_heads))
        if head_dim <= 0:
            return False
        kv_lora_rank = int(tc.get("kv_lora_rank", 0) or 0)
        qk_nope_head_dim = int(tc.get("qk_nope_head_dim", 0) or 0)
        qk_rope_head_dim = int(tc.get("qk_rope_head_dim", 0) or 0)
        index_topk = int(tc.get("index_topk", 0) or 0)
        sliding_window = int(tc.get("sliding_window", 0) or 0)
    except (TypeError, ValueError):
        return False

    layer_types = _resolve_layer_types(
        tc, num_layers, kv_lora_rank, index_topk, sliding_window
    )
    if layer_types is None:
        return False

    has_mla = any(
        t in ("mla", "mla_attention", "dsa", "dsa_attention", "sparse_mla")
        or (t in ("full_attention", "attention") and kv_lora_rank > 0)
        for t in layer_types
    )
    if has_mla and (kv_lora_rank <= 0 or qk_nope_head_dim <= 0 or qk_rope_head_dim < 0):
        return False

    lin_cfg_raw = tc.get("linear_attn_config")
    lin_cfg = dict(lin_cfg_raw) if type(lin_cfg_raw) is dict else {}
    raw_conv = (
        tc.get("linear_conv_kernel_dim")
        or tc.get("conv_kernel")
        or tc.get("d_conv")
        or lin_cfg.get("conv_kernel_size")
    )
    if any(t in _GDN_TOKENS for t in layer_types):
        if raw_conv is None or int(raw_conv) <= 0:
            return False

    ple_dim = int(tc.get("per_layer_input_dim") or tc.get("ple_dim") or 0)
    if "ple" in layer_types and ple_dim <= 0:
        return False

    num_experts = int(
        tc.get("num_experts")
        or tc.get("n_routed_experts")
        or tc.get("num_local_experts")
        or 0
    )
    dense_inter = int(
        tc.get("intermediate_size") or tc.get("dense_intermediate_size") or 0
    )
    moe_inter = int(
        tc.get("moe_intermediate_size") or tc.get("latent_moe_dim") or dense_inter or 0
    )
    if num_experts > 0:
        top_k = int(
            tc.get("num_experts_per_tok")
            or tc.get("experts_per_token")
            or tc.get("num_selected_experts")
            or tc.get("top_k")
            or 0
        )
        if top_k <= 0 or moe_inter <= 0:
            return False
    elif dense_inter <= 0:
        return False

    return True


def from_hf_config(cfg: Mapping[str, Any]) -> ModelGeometry:
    """Build ModelGeometry from a raw or nested HuggingFace config mapping."""
    if not supports(cfg):
        raise ValueError(
            "Unsupported or incomplete HuggingFace config for TPU roofline"
        )
    extracted = _extract_text_config(cfg)
    assert extracted is not None
    raw, tc = extracted

    hidden = int(tc["hidden_size"])
    num_layers = int(tc["num_hidden_layers"])
    num_heads = int(tc["num_attention_heads"])
    num_kv_heads = int(tc.get("num_key_value_heads", num_heads))
    head_dim = int(tc.get("head_dim") or (hidden // num_heads))

    kv_lora_rank = int(tc.get("kv_lora_rank", 0) or 0)
    q_lora_rank = int(tc.get("q_lora_rank", 0) or 0)
    qk_nope_head_dim = int(tc.get("qk_nope_head_dim", 0) or 0)
    qk_rope_head_dim = int(tc.get("qk_rope_head_dim", 0) or 0)
    v_head_dim = int(tc.get("v_head_dim", 0) or 0)
    index_n_heads = int(tc.get("index_n_heads", 0) or 0)
    index_head_dim = int(tc.get("index_head_dim", 0) or 0)
    index_topk = int(tc.get("index_topk", 0) or 0)
    sliding_window = int(tc.get("sliding_window", 0) or 0)

    layer_types = _resolve_layer_types(
        tc, num_layers, kv_lora_rank, index_topk, sliding_window
    )
    assert layer_types is not None

    mtp_raw = tc.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = raw.get("mtp_num_hidden_layers")
    if mtp_raw is None:
        mtp_raw = tc.get(
            "num_nextn_predict_layers", raw.get("num_nextn_predict_layers", 0)
        )

    num_experts = int(
        tc.get("num_experts")
        or tc.get("n_routed_experts")
        or tc.get("num_local_experts")
        or 0
    )
    dense_inter = int(
        tc.get("intermediate_size") or tc.get("dense_intermediate_size") or 0
    )
    moe_inter = int(tc.get("moe_intermediate_size") or tc.get("latent_moe_dim") or 0)
    if num_experts > 0 and moe_inter == 0 and dense_inter > 0:
        moe_inter = dense_inter

    num_shared_experts = int(
        tc.get("n_shared_experts") or tc.get("num_shared_experts") or 0
    )
    shared_inter = int(tc.get("shared_expert_intermediate_size", 0) or 0)
    if shared_inter == 0 and num_shared_experts > 0 and moe_inter > 0:
        shared_inter = num_shared_experts * moe_inter

    lin_cfg_raw = tc.get("linear_attn_config")
    lin_cfg = dict(lin_cfg_raw) if type(lin_cfg_raw) is dict else {}
    lin_k_heads = int(
        tc.get("linear_num_key_heads") or lin_cfg.get("num_key_heads") or 0
    )
    lin_v_heads = int(
        tc.get("linear_num_value_heads") or lin_cfg.get("num_value_heads") or 0
    )
    lin_k_dim = int(tc.get("linear_key_head_dim") or lin_cfg.get("key_head_dim") or 0)
    lin_v_dim = int(
        tc.get("linear_value_head_dim") or lin_cfg.get("value_head_dim") or 0
    )
    ple_dim = int(tc.get("per_layer_input_dim") or tc.get("ple_dim") or 0)

    raw_conv = (
        tc.get("linear_conv_kernel_dim")
        or tc.get("conv_kernel")
        or tc.get("d_conv")
        or lin_cfg.get("conv_kernel_size")
    )
    conv_kernel = (
        int(raw_conv) if any(t in _GDN_TOKENS for t in layer_types) and raw_conv else 0
    )

    model_type = str(tc.get("model_type") or raw.get("model_type") or "").lower()
    raw_arch = raw.get("architectures") or tc.get("architectures")
    if raw_arch and type(raw_arch) in (list, tuple):
        architectures = tuple(str(a) for a in raw_arch)
    else:
        arch_fallback = {
            "qwen3_moe": ("Qwen3MoeForCausalLM",),
            "gpt_oss": ("GptOssForCausalLM",),
            "gpt-oss": ("GptOssForCausalLM",),
        }
        architectures = arch_fallback.get(model_type, ())
    has_sink = bool(
        tc.get("has_attention_sink")
        or tc.get("attention_sink")
        or tc.get("use_attention_sink")
        or tc.get("attn_sink")
        or model_type in ("gpt_oss", "gpt-oss")
    )
    top_k = int(
        tc.get("num_experts_per_tok")
        or tc.get("experts_per_token")
        or tc.get("num_selected_experts")
        or tc.get("top_k")
        or 0
    )

    return ModelGeometry(
        hidden=hidden,
        num_layers=num_layers,
        layer_types=layer_types,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        attn_output_gate=bool(tc.get("attn_output_gate", False)),
        linear_k_heads=lin_k_heads,
        linear_v_heads=lin_v_heads,
        linear_k_dim=lin_k_dim,
        linear_v_dim=lin_v_dim,
        conv_kernel=conv_kernel,
        num_experts=num_experts,
        top_k=top_k,
        moe_inter=moe_inter,
        shared_inter=shared_inter,
        vocab=int(tc.get("vocab_size", raw.get("vocab_size", 0)) or 0),
        mtp_layers=int(mtp_raw or 0),
        dense_inter=dense_inter,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        sliding_window=sliding_window,
        num_shared_experts=num_shared_experts,
        first_k_dense_replace=int(tc.get("first_k_dense_replace", 0) or 0),
        ple_dim=ple_dim,
        has_attention_sink=has_sink,
        model_type=model_type,
        architectures=architectures,
    )
