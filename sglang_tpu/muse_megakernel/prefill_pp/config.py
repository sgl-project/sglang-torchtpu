# SPDX-License-Identifier: Apache-2.0
"""The PP prefill stage's static description: config, slot table, weight and pool layouts."""

import os

import jax.numpy as jnp
import msgspec
import numpy as np

LANE = 128
PHASES = ("attn_in", "attention", "route", "experts", "combine")
FP8 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16
F32 = jnp.float32

# slot table: a header, then NF int32 fields per slot
HDR = 8
H_STAGE, H_LAYERS = 0, 1
NF = 4
F_VALID, F_GLOBAL, F_FP8, F_LAYER = range(4)

# rows of the per-layer vector slab [slots, V_ROWS, vec_width] f32
V = dict(
    pre_attn_w=0,
    attn_alpha=1,
    attn_beta=2,
    pre_ffn_w=3,
    post_ffn_w=4,
    ffn_alpha=5,
    ffn_beta=6,
    pre_expert_w=7,
    expert_bias=8,
    post_expert_w=9,
    wqkvg_s=10,
    wo_s=11,
    wpool_s=12,
    wunpool_s=13,
)
V_ROWS = 16


class PPConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Static description of one stage program (every stage runs the same one)."""

    hidden: int
    head_dim: int
    q_heads: int
    kv_heads: int
    n_experts: int
    top_k: int
    expert_hidden: int
    inter: int
    rms_eps: float
    post_eps: float
    scale_query_by: float
    attn_scale: float
    router_mult: float
    route_scale: float
    router_eps: float
    window: int
    # token rows of the chunk
    rows: int
    # layer slots per stage
    slots: int
    page: int
    pages_per_seq: int
    # pool pages of a global / sliding-window slot
    global_pages: int
    swa_pages: int
    max_position: int
    # slots of the global attention layers (the rest are sliding-window)
    global_slots: tuple = (2, 6)
    # slots that may hold a stage's fp8-expert layer
    fp8_slots: tuple = (1, 6)
    # dense row tile and weight column tile
    tm: int = 256
    tn: int = 512

    @property
    def q_dim(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def qkvg_dim(self) -> int:
        return 2 * self.q_dim + 2 * self.kv_dim

    @property
    def vec_width(self) -> int:
        return max(self.qkvg_dim, self.hidden)

    def validate(self) -> None:
        """Raises ValueError for a geometry the stage kernels do not tile."""
        H, He, T = self.hidden, self.expert_hidden, self.rows
        checks = [
            (H >= 2 * He and self.q_dim >= 2 * He, "combine staging halves"),
            (
                self.tm % self.top_k == 0 and (self.tm // self.top_k) % 8 == 0,
                "row tile",
            ),
            (T % self.tm == 0 and T % self.page == 0, "chunk rows"),
            (
                self.qkvg_dim % self.tn == 0 and H % self.tn == 0 and He % self.tn == 0,
                "tn",
            ),
            (self.q_dim % self.tn == 0 and self.kv_dim % self.tn == 0, "head tiles"),
            (self.n_experts <= 2 * LANE and self.n_experts % LANE == 0, "router lanes"),
            (self.top_k <= LANE, "routing lanes"),
            (self.q_heads % self.kv_heads == 0, "GQA"),
        ]
        bad = [why for ok, why in checks if not ok]
        if bad:
            raise ValueError(f"PPConfig {self}: {'; '.join(bad)}")


def stage_layers(n_layers: int, split: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Consecutive layer runs of the given sizes."""
    if sum(split) != n_layers:
        raise ValueError(f"split {split} does not cover {n_layers} layers")
    out, lo = [], 0
    for n in split:
        out.append(tuple(range(lo, lo + n)))
        lo += n
    return out


def slot_table(
    c: PPConfig, stage: int, layers: tuple[int, ...], kinds: tuple, global_layers: tuple
) -> np.ndarray:
    """The stage's slot table [HDR + slots * NF] int32. Its layers take consecutive slots
    from the one that puts its global layers on cfg.global_slots.

    Raises:
      ValueError: The layers do not fit the stage's slots, their global layers cannot sit on
        cfg.global_slots, or their fp8-expert layer (one at most) is not on cfg.fp8_slots.
    """
    globals_at = [i for i, layer in enumerate(layers) if layer in global_layers]
    first = c.global_slots[0] - globals_at[0] if globals_at else 0
    placed = [first + i for i in globals_at]
    if first < 0 or first + len(layers) > c.slots or placed != list(c.global_slots):
        raise ValueError(
            f"stage {stage}: layers {layers} (global {[layers[i] for i in globals_at]}) do not "
            f"fit {c.slots} slots with global attention at {c.global_slots}"
        )
    fp8_at = [first + i for i, layer in enumerate(layers) if kinds[layer] == "fp8"]
    if len(fp8_at) > 1 or not set(fp8_at) <= set(c.fp8_slots):
        raise ValueError(
            f"stage {stage}: fp8 experts on slots {fp8_at}, not one of {c.fp8_slots}"
        )
    t = np.zeros(HDR + c.slots * NF, np.int32)
    t[H_STAGE], t[H_LAYERS] = stage, len(layers)
    for i, layer in enumerate(layers):
        f = t[HDR + (first + i) * NF : HDR + (first + i + 1) * NF]
        f[F_VALID], f[F_GLOBAL] = 1, int(layer in global_layers)
        f[F_FP8], f[F_LAYER] = int(kinds[layer] == "fp8"), layer
    return t


def weight_shapes(c: PPConfig) -> dict[str, tuple[tuple[int, ...], jnp.dtype]]:
    """One stage's dense weights (per device), column-tile major: each [K, tn] tile is one
    contiguous block."""
    S, H, He, E, tn = c.slots, c.hidden, c.expert_hidden, c.n_experts, c.tn
    return {
        "wqkvg": ((S, c.qkvg_dim // tn, H, tn), FP8),
        "wo": ((S, H // tn, c.q_dim, tn), FP8),
        "wpool": ((S, He // tn, H, tn), FP8),
        "wunpool": ((S, H // tn, He, tn), FP8),
        "wrouter": ((S, H, E), BF16),
        "vecs": ((S, V_ROWS, c.vec_width), F32),
    }


WEIGHT_ORDER = ("wqkvg", "wo", "wpool", "wunpool", "wrouter", "vecs")


def expert_shapes(c: PPConfig, kind: str) -> list[tuple[tuple[int, ...], jnp.dtype]]:
    """(w13, w13_scale, w2, w2_scale) of one layer in the fused MoE's serving layouts: NVFP4 as
    e2m1 codes packed two per byte along K with bf16 block scales (16 along K) paired into
    int32 words; fp8 [E, K, N] with f32 channel scales."""
    E, He, I = c.n_experts, c.expert_hidden, c.inter
    if kind == "fp4":
        return [
            ((E, He // 2, 2 * I), jnp.uint8),
            ((E, He // 32, 2 * I), jnp.int32),
            ((E, I // 2, He), jnp.uint8),
            ((E, I // 32, He), jnp.int32),
        ]
    if kind == "fp8":
        return [
            ((E, He, 2 * I), FP8),
            ((E, 1, 1, 2 * I), F32),
            ((E, I, He), FP8),
            ((E, 1, 1, He), F32),
        ]
    raise ValueError(f"expert kind {kind}")


def pool_shape(c: PPConfig, pages: int) -> tuple[int, ...]:
    return (pages, 2 * c.kv_heads, c.head_dim // 4, 4, c.page)


def skipped_phases() -> frozenset[str]:
    """MUSE_PP_SKIP=a,b: timing ablation only (wrong numerics), those phases do not run."""
    skip = frozenset(p for p in os.environ.get("MUSE_PP_SKIP", "").split(",") if p)
    if not skip <= set(PHASES):
        raise ValueError(f"MUSE_PP_SKIP={sorted(skip)}: phases are {PHASES}")
    return skip
