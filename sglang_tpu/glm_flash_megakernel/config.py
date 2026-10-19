"""GLM-5.3-Flash spec from config.json; nothing defaults to a sibling model's value."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Literal

LayerType = Literal["linear_attention", "deepseek_sparse_attention"]
MlpType = Literal["dense", "sparse"]
IndexerType = Literal["full", "shared"]


class SpecError(ValueError):
    """Raised when the checkpoint disagrees with what this implementation supports."""


@dataclasses.dataclass(frozen=True)
class Glm53FlashSpec:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    pad_token_id: int
    eos_token_ids: tuple[int, ...]
    tie_word_embeddings: bool

    layer_types: tuple[LayerType, ...]
    mlp_layer_types: tuple[MlpType, ...]
    indexer_types: tuple[IndexerType, ...]

    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int

    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_kpool_compress: bool
    index_kpool_always_select_tail: bool

    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_dim: int
    linear_lower_bound: float | None

    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    intermediate_size: int
    n_group: int
    topk_group: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    scoring_func: str
    topk_method: str
    swiglu_limit: float
    first_k_dense_replace: int

    num_nextn_predict_layers: int

    quant_fmt: str
    weight_block_size: tuple[int, int]
    quant_exclude: frozenset[str]

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kda_qkv_dim(self) -> int:
        return self.linear_num_heads * self.linear_head_dim

    @property
    def kda_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "linear_attention")

    @property
    def dsa_layers(self) -> tuple[int, ...]:
        return tuple(
            i for i, t in enumerate(self.layer_types) if t == "deepseek_sparse_attention"
        )

    @property
    def mtp_layer_idx(self) -> int:
        return self.num_hidden_layers

    @property
    def index_select_k(self) -> int:
        """Number of pools the indexer selects (index_topk is a token budget)."""
        return self.index_topk // self.index_kpool

    @property
    def index_output_width(self) -> int:
        w = self.index_topk
        if self.index_kpool_always_select_tail:
            w += self.index_kpool - 1
        return w

    def body_pattern(self) -> tuple[int, tuple[tuple[LayerType, MlpType], ...]]:
        """Return (period, pattern) of the repeating layer schedule, after the prefix."""
        prefix = self.first_k_dense_replace
        tail = [
            (self.layer_types[i], self.mlp_layer_types[i])
            for i in range(prefix, self.num_hidden_layers)
        ]
        for period in range(1, len(tail) + 1):
            if all(tail[i] == tail[i % period] for i in range(len(tail))):
                return period, tuple(tail[:period])
        return len(tail), tuple(tail)

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str) -> "Glm53FlashSpec":
        raw = json.load(open(os.path.join(ckpt_dir, "config.json")))
        if raw.get("model_type") != "glm5_next":
            raise SpecError(f"expected model_type=glm5_next, got {raw.get('model_type')!r}")
        arch = raw.get("architectures") or []
        if "Glm5NextForConditionalGeneration" not in arch:
            raise SpecError(f"unexpected architectures: {arch}")
        t = raw["text_config"]
        q = raw["quantization_config"]

        def need(key):
            if key not in t:
                raise SpecError(
                    f"text_config has no {key!r}; this implementation refuses to "
                    "substitute a default from another model"
                )
            return t[key]

        eos = need("eos_token_id")
        eos = tuple(eos) if isinstance(eos, (list, tuple)) else (eos,)

        spec = cls(
            vocab_size=need("vocab_size"),
            hidden_size=need("hidden_size"),
            num_hidden_layers=need("num_hidden_layers"),
            rms_norm_eps=need("rms_norm_eps"),
            pad_token_id=need("pad_token_id"),
            eos_token_ids=eos,
            tie_word_embeddings=t.get("tie_word_embeddings", raw.get("tie_word_embeddings", False)),
            layer_types=tuple(need("layer_types")),
            mlp_layer_types=tuple(need("mlp_layer_types")),
            indexer_types=tuple(need("indexer_types")),
            hc_mult=need("hc_mult"),
            hc_eps=need("hc_eps"),
            hc_sinkhorn_iters=need("hc_sinkhorn_iters"),
            num_attention_heads=need("num_attention_heads"),
            q_lora_rank=need("q_lora_rank"),
            kv_lora_rank=need("kv_lora_rank"),
            qk_nope_head_dim=need("qk_nope_head_dim"),
            qk_rope_head_dim=need("qk_rope_head_dim"),
            v_head_dim=need("v_head_dim"),
            index_n_heads=need("index_n_heads"),
            index_head_dim=need("index_head_dim"),
            index_topk=need("index_topk"),
            index_kpool=need("index_kpool"),
            index_kpool_compress=need("index_kpool_compress"),
            index_kpool_always_select_tail=need("index_kpool_always_select_tail"),
            linear_num_heads=need("linear_attn_config")["num_heads"],
            linear_head_dim=need("linear_attn_config")["head_dim"],
            linear_conv_kernel_dim=need("linear_attn_config")["short_conv_kernel_size"],
            linear_lower_bound=need("linear_attn_config")["gate_lower_bound"],
            n_routed_experts=need("n_routed_experts"),
            n_shared_experts=need("n_shared_experts"),
            num_experts_per_tok=need("num_experts_per_tok"),
            moe_intermediate_size=need("moe_intermediate_size"),
            intermediate_size=need("intermediate_size"),
            n_group=need("n_group"),
            topk_group=need("topk_group"),
            norm_topk_prob=need("norm_topk_prob"),
            routed_scaling_factor=need("routed_scaling_factor"),
            scoring_func=need("scoring_func"),
            topk_method=need("topk_method"),
            swiglu_limit=need("swiglu_limit"),
            first_k_dense_replace=need("first_k_dense_replace"),
            num_nextn_predict_layers=need("num_nextn_predict_layers"),
            quant_fmt=q["fmt"],
            weight_block_size=tuple(q["weight_block_size"]),
            quant_exclude=frozenset(q["modules_to_not_convert"]),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        n = self.num_hidden_layers
        for name, seq in (
            ("layer_types", self.layer_types),
            ("mlp_layer_types", self.mlp_layer_types),
            ("indexer_types", self.indexer_types),
        ):
            if len(seq) != n:
                raise SpecError(f"{name} has {len(seq)} entries, expected {n}")

        bad = set(self.layer_types) - {"linear_attention", "deepseek_sparse_attention"}
        if bad:
            raise SpecError(f"unsupported layer types: {sorted(bad)}")

        if self.qk_rope_head_dim != 0:
            raise SpecError(
                f"qk_rope_head_dim={self.qk_rope_head_dim}; this checkpoint is "
                "documented as NoPE on the DSA path and the implementation relies on it"
            )
        if self.index_topk % self.index_kpool:
            raise SpecError("index_topk must be divisible by index_kpool")
        if self.num_attention_heads * self.v_head_dim % 128:
            raise SpecError("attention output width must be lane-aligned")
        if self.scoring_func != "sigmoid":
            raise SpecError(f"unsupported router scoring_func {self.scoring_func!r}")
        if self.topk_method != "noaux_tc":
            raise SpecError(f"unsupported topk_method {self.topk_method!r}")
        if self.quant_fmt != "e4m3":
            raise SpecError(f"unsupported fp8 format {self.quant_fmt!r}")
        if self.num_nextn_predict_layers != 1:
            raise SpecError(
                f"num_nextn_predict_layers={self.num_nextn_predict_layers}; the "
                "speculative path assumes exactly one MTP layer"
            )
        derived = self.mlp_layer_types.index("sparse") if "sparse" in self.mlp_layer_types else len(self.mlp_layer_types)
        if derived != self.first_k_dense_replace:
            raise SpecError(
                f"first_k_dense_replace={self.first_k_dense_replace} but mlp_layer_types "
                f"turns sparse at {derived}"
            )
