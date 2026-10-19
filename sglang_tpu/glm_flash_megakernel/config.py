"""Validated model spec for GLM-5.3-Flash, derived from the checkpoint.

Nothing here defaults to a value from a sibling model's source.  Every field is read
from `config.json`, and `validate_against_manifest` then cross-checks the derived
shapes against the actual safetensors headers, so a wrong assumption fails loudly at
load time instead of turning into a silent numerical difference 45 layers later.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Literal

LayerType = Literal["linear_attention", "deepseek_sparse_attention"]
MlpType = Literal["dense", "sparse"]
IndexerType = Literal["full", "shared"]

# Weight names in the checkpoint live under this prefix, while the
# `quantization_config.modules_to_not_convert` list is written against `model.layers.*`.
# The two must be reconciled explicitly; they are not the same string.
WEIGHT_PREFIX = "model.language_model."
QUANT_EXCLUDE_PREFIX = "model."


class SpecError(ValueError):
    """Raised when the checkpoint disagrees with what this implementation supports."""


@dataclasses.dataclass(frozen=True)
class Glm53FlashSpec:
    # --- global ---
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    pad_token_id: int
    eos_token_ids: tuple[int, ...]
    tie_word_embeddings: bool

    # --- per-layer schedule ---
    layer_types: tuple[LayerType, ...]
    mlp_layer_types: tuple[MlpType, ...]
    indexer_types: tuple[IndexerType, ...]

    # --- mHC ---
    hc_mult: int
    hc_eps: float
    hc_sinkhorn_iters: int

    # --- DSA / MLA ---
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    # --- DSA indexer ---
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    index_kpool_compress: bool
    index_kpool_always_select_tail: bool

    # --- KDA ---
    linear_num_heads: int
    linear_head_dim: int
    linear_conv_kernel_dim: int
    linear_lower_bound: float | None

    # --- MoE ---
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

    # --- speculative / MTP ---
    num_nextn_predict_layers: int

    # --- quantization ---
    quant_fmt: str
    weight_block_size: tuple[int, int]
    quant_exclude: frozenset[str]

    # ---------------------------------------------------------------- derived
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
        """Index of the single MTP / next-token-prediction draft layer."""
        return self.num_hidden_layers

    @property
    def index_select_k(self) -> int:
        """Number of *pools* the indexer selects (index_topk is a token budget)."""
        return self.index_topk // self.index_kpool

    @property
    def index_output_width(self) -> int:
        """Width of the emitted index vector, including the always-selected tail."""
        w = self.index_topk
        if self.index_kpool_always_select_tail:
            w += self.index_kpool - 1
        return w

    def body_pattern(self) -> tuple[int, tuple[tuple[LayerType, MlpType], ...]]:
        """Return (period, pattern) of the repeating layer schedule, after the prefix.

        The layer loop in the megakernel is rolled over this pattern, so getting it
        from the config rather than assuming `3 x KDA + 1 x DSA` matters.
        """
        prefix = self.first_k_dense_replace
        tail = [
            (self.layer_types[i], self.mlp_layer_types[i])
            for i in range(prefix, self.num_hidden_layers)
        ]
        for period in range(1, len(tail) + 1):
            if all(tail[i] == tail[i % period] for i in range(len(tail))):
                return period, tuple(tail[:period])
        return len(tail), tuple(tail)

    # ------------------------------------------------------------ constructor
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

    # -------------------------------------------------------------- validation
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

        # This checkpoint is NoPE on the attention path.  The HF config's own
        # validate_architecture() refuses anything else, so a nonzero rope dim here
        # would mean we are looking at a different model than we think.
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
        # first_k_dense_replace must agree with mlp_layer_types rather than be trusted.
        derived = self.mlp_layer_types.index("sparse") if "sparse" in self.mlp_layer_types else len(self.mlp_layer_types)
        if derived != self.first_k_dense_replace:
            raise SpecError(
                f"first_k_dense_replace={self.first_k_dense_replace} but mlp_layer_types "
                f"turns sparse at {derived}"
            )

    def is_quantized(self, weight_name: str) -> bool:
        """Whether a checkpoint tensor is fp8-blockwise quantized.

        `modules_to_not_convert` is written against the `model.layers.*` prefix while
        the tensors live under `model.language_model.layers.*`, so the name has to be
        translated before the lookup.  A `.weight` suffix is stripped because the
        exclusion list names modules, not parameters.
        """
        name = weight_name
        if name.startswith(WEIGHT_PREFIX):
            name = QUANT_EXCLUDE_PREFIX + name[len(WEIGHT_PREFIX):]
        for suffix in (".weight", ".bias"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        if name in self.quant_exclude:
            return False
        # also allow a module-level exclusion to cover its parameters
        parts = name.split(".")
        for i in range(len(parts), 0, -1):
            if ".".join(parts[:i]) in self.quant_exclude:
                return False
        return True


# ---------------------------------------------------------------------------
# Cross-check the derived spec against the real safetensors headers.
# ---------------------------------------------------------------------------
def validate_against_manifest(spec: Glm53FlashSpec, manifest: dict) -> list[str]:
    """Return a list of human-readable problems; empty means the spec matches.

    `manifest` maps tensor names to safetensors header metadata, with a
    `shape` list and a `dtype` string such as "BF16" for each tensor.
    """
    problems: list[str] = []
    P = WEIGHT_PREFIX

    def check(name: str, shape: list[int], dtype: str | None = None):
        meta = manifest.get(name)
        if meta is None:
            problems.append(f"missing tensor {name}")
            return
        if list(meta["shape"]) != list(shape):
            problems.append(f"{name}: shape {meta['shape']} != expected {shape}")
        if dtype is not None and meta["dtype"] != dtype:
            problems.append(f"{name}: dtype {meta['dtype']} != expected {dtype}")

    H = spec.hidden_size
    check(f"{P}embed_tokens.weight", [spec.vocab_size, H], "BF16")
    check("lm_head.weight", [spec.vocab_size, H], "BF16")
    check(f"{P}norm.weight", [H], "BF16")

    for i in range(spec.num_hidden_layers):
        L = f"{P}layers.{i}."
        check(L + "input_layernorm.weight", [H], "BF16")
        check(L + "post_attention_layernorm.weight", [H], "BF16")
        # mHC: (2 + N) * N coefficients projected from N * hidden
        nc = (2 + spec.hc_mult) * spec.hc_mult
        for tag in ("attn", "ffn"):
            check(L + f"hc_{tag}_fn", [nc, spec.hc_mult * H], "BF16")
            check(L + f"hc_{tag}_base", [nc], "F32")
            check(L + f"hc_{tag}_scale", [3], "F32")

        if spec.layer_types[i] == "linear_attention":
            Q = spec.kda_qkv_dim
            for p in ("q_proj", "k_proj", "v_proj"):
                check(L + f"self_attn.{p}.weight", [Q, H], "BF16")
            for p in ("q_conv1d", "k_conv1d", "v_conv1d"):
                check(L + f"self_attn.{p}.weight", [Q, 1, spec.linear_conv_kernel_dim], "BF16")
            check(L + "self_attn.b_proj.weight", [spec.linear_num_heads, H], "BF16")
            check(L + "self_attn.f_a_proj.weight", [spec.linear_head_dim, H], "BF16")
            check(L + "self_attn.f_b_proj.weight", [Q, spec.linear_head_dim], "BF16")
            check(L + "self_attn.g_a_proj.weight", [spec.linear_head_dim, H], "BF16")
            check(L + "self_attn.g_b_proj.weight", [Q, spec.linear_head_dim], "BF16")
            check(L + "self_attn.A_log", [spec.linear_num_heads], "F32")
            check(L + "self_attn.dt_bias", [Q], "F32")
            check(L + "self_attn.o_norm.weight", [spec.linear_head_dim], "BF16")
            check(L + "self_attn.o_proj.weight", [H, Q], "BF16")
        else:
            nh = spec.num_attention_heads
            check(L + "self_attn.q_a_proj.weight", [spec.q_lora_rank, H], "F8_E4M3")
            check(L + "self_attn.q_a_layernorm.weight", [spec.q_lora_rank], "BF16")
            check(L + "self_attn.q_b_proj.weight", [nh * spec.qk_head_dim, spec.q_lora_rank], "F8_E4M3")
            check(
                L + "self_attn.kv_a_proj_with_mqa.weight",
                [spec.kv_lora_rank + spec.qk_rope_head_dim, H],
                "F8_E4M3",
            )
            check(L + "self_attn.kv_a_layernorm.weight", [spec.kv_lora_rank], "BF16")
            check(
                L + "self_attn.kv_b_proj.weight",
                [nh * (spec.qk_nope_head_dim + spec.v_head_dim), spec.kv_lora_rank],
                "BF16",
            )
            check(L + "self_attn.o_proj.weight", [H, nh * spec.v_head_dim], "F8_E4M3")
            if spec.indexer_types[i] == "full":
                I = L + "self_attn.indexer."
                check(I + "wq_b.weight", [spec.index_n_heads * spec.index_head_dim, spec.q_lora_rank], "BF16")
                check(I + "wk.weight", [spec.index_head_dim, H], "BF16")
                check(I + "k_norm.weight", [spec.index_head_dim], "BF16")
                check(I + "k_norm.bias", [spec.index_head_dim], "BF16")
                check(I + "weights_proj.weight", [spec.index_n_heads, H], "BF16")
                if spec.index_kpool_compress:
                    check(I + "index_kpool_compress_ape", [spec.index_kpool, spec.index_head_dim], "BF16")
                    check(I + "index_kpool_compress_gate", [spec.index_head_dim, H], "BF16")

        if spec.mlp_layer_types[i] == "sparse":
            M = L + "mlp."
            check(M + "gate.weight", [spec.n_routed_experts, H], "BF16")
            check(M + "gate.e_score_correction_bias", [spec.n_routed_experts], "F32")
            mi = spec.moe_intermediate_size
            bs = spec.weight_block_size
            for e in (0, spec.n_routed_experts - 1):
                check(M + f"experts.{e}.gate_proj.weight", [mi, H], "F8_E4M3")
                check(M + f"experts.{e}.gate_proj.weight_scale_inv",
                      [-(-mi // bs[0]), -(-H // bs[1])], "F32")
                check(M + f"experts.{e}.down_proj.weight", [H, mi], "F8_E4M3")
            for p in ("gate_proj", "up_proj"):
                check(M + f"shared_experts.{p}.weight", [mi * spec.n_shared_experts, H], "F8_E4M3")
            check(M + "shared_experts.down_proj.weight", [H, mi * spec.n_shared_experts], "F8_E4M3")
        else:
            M = L + "mlp."
            for p in ("gate_proj", "up_proj"):
                check(M + f"{p}.weight", [spec.intermediate_size, H], "F8_E4M3")
            check(M + "down_proj.weight", [H, spec.intermediate_size], "F8_E4M3")

    # MTP / next-token-prediction draft layer
    m = spec.mtp_layer_idx
    MT = f"{P}layers.{m}."
    check(MT + "enorm.weight", [H], "BF16")
    check(MT + "hnorm.weight", [H], "BF16")
    check(MT + "eh_proj.weight", [H, 2 * H], "BF16")
    check(MT + "shared_head.norm.weight", [H], "BF16")

    return problems


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", nargs="?", default="./GLM-5.3-Flash")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    spec = Glm53FlashSpec.from_checkpoint(args.ckpt_dir)
    period, pattern = spec.body_pattern()
    print(f"layers               : {spec.num_hidden_layers} (+1 MTP at index {spec.mtp_layer_idx})")
    print(f"  KDA layers         : {len(spec.kda_layers)}  {spec.kda_layers[:8]}...")
    print(f"  DSA layers         : {len(spec.dsa_layers)}  {spec.dsa_layers}")
    print(f"  dense mlp prefix   : {spec.first_k_dense_replace}")
    print(f"  body period        : {period}  pattern={pattern}")
    print(f"  indexer modes      : {set(spec.indexer_types)}")
    print(f"mHC                  : mult={spec.hc_mult} sinkhorn={spec.hc_sinkhorn_iters} eps={spec.hc_eps}")
    print(f"MLA                  : heads={spec.num_attention_heads} q_lora={spec.q_lora_rank} "
          f"kv_lora={spec.kv_lora_rank} nope={spec.qk_nope_head_dim} rope={spec.qk_rope_head_dim} "
          f"v={spec.v_head_dim}")
    print(f"DSA indexer          : heads={spec.index_n_heads} dim={spec.index_head_dim} "
          f"topk={spec.index_topk} kpool={spec.index_kpool} select_k={spec.index_select_k} "
          f"out_width={spec.index_output_width} tail={spec.index_kpool_always_select_tail}")
    print(f"KDA                  : heads={spec.linear_num_heads} dim={spec.linear_head_dim} "
          f"conv={spec.linear_conv_kernel_dim} lower_bound={spec.linear_lower_bound}")
    print(f"MoE                  : {spec.n_routed_experts} routed top-{spec.num_experts_per_tok} "
          f"+{spec.n_shared_experts} shared, mi={spec.moe_intermediate_size}, "
          f"scale={spec.routed_scaling_factor}, norm={spec.norm_topk_prob}, "
          f"group={spec.n_group}/{spec.topk_group}")
    print(f"quant                : {spec.quant_fmt} block={spec.weight_block_size} "
          f"({len(spec.quant_exclude)} excluded modules)")

    if args.manifest:
        manifest = json.load(open(args.manifest))
        problems = validate_against_manifest(spec, manifest)
        if problems:
            print(f"\n{len(problems)} MISMATCHES vs checkpoint tensors:")
            for p in problems[:40]:
                print("   ", p)
            return 1
        print("\nspec matches every checked tensor in the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
