# SPDX-License-Identifier: Apache-2.0
"""The EP decode megakernel: muse_megakernel.kernel's row-tiled program with an EP FFN section
on 32 whole experts per rank, and the layout of its weights. Two arms (config.ep_arm): a2a
(EPExperts: sequence-parallel, two ragged all-to-alls) and gather (EPGatherExperts: every row
on every rank, small steps)."""

import jax
import numpy as np

from sglang_tpu.muse_megakernel.ep_mk.config import (
    ep_arm,
    ep_expert_shapes,
    ep_mk_config,
)
from sglang_tpu.muse_megakernel.ep_mk.experts import (
    chunk_cols,
    fp4_weights_kperm,
    kperm,
    permute_fp8_rows,
    qblock,
    requant_fp4_kperm,
)
from sglang_tpu.muse_megakernel.ep_mk.gather import EPGatherExperts
from sglang_tpu.muse_megakernel.ep_mk.layer import EPExperts
from sglang_tpu.muse_megakernel.kernel import megakernel_call
from sglang_tpu.muse_megakernel.mk.config import MKConfig

__all__ = [
    "ep_arm",
    "ep_expert_shapes",
    "ep_expert_weights",
    "ep_megakernel_call",
    "pooled_perm",
]


def ep_megakernel_call(cfg: MKConfig, axis: str = "model", arm: str | None = None):
    """megakernel_call's f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv), with
    each layer's experts as ep_expert_shapes (ep_expert_weights), wpool / wunpool replicated
    ([L, H, H_e] and [L, H_e, H] fp8, their scales whole in the smalls rows) and the pooled
    width in pooled_perm order (wpool's columns, wpool_s, pre_expert_w). arm: a2a | gather,
    None for ep_arm(cfg.rows); an arm that cannot serve the step raises."""
    cfg = ep_mk_config(cfg)
    arm = ep_arm(cfg.rows) if arm is None else arm
    if arm == "a2a":
        hook: EPExperts | EPGatherExperts = EPExperts(cfg)
    elif arm == "gather":
        hook = EPGatherExperts(cfg)
    else:
        raise ValueError(f"EP arm {arm!r}: a2a or gather")
    return megakernel_call(cfg, axis, ep=hook)


def pooled_perm(he: int) -> np.ndarray:
    """The pooled (expert input) width's order in the EP kernel."""
    return kperm(he)


def ep_expert_weights(
    kind: str,
    w13: jax.Array,
    s13: jax.Array,
    w2: jax.Array,
    s2: jax.Array,
    chunk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One rank's whole experts in the EP kernel's layout, from the natural one ([E, H_e/pack,
    2I] w13, [E, I/pack, H_e] w2, int32 words): hidden rows in pooled_perm order; for fp4
    the gate / up columns and w2's rows in kperm order too (the dense arm's scale layout);
    then w13's columns in chunk_cols order (chunk: the MKConfig's inter_local). Under
    SGLANG_TPU_EP_QBLOCK, fp4 rows are requantized to its blocks of the kperm order."""
    cols = chunk_cols(w13.shape[-1], chunk)
    if kind == "fp4" and qblock():
        u32 = jax.numpy.uint32
        w13k, s13k = requant_fp4_kperm(w13.view(u32), s13, qblock())
        w2k, s2k = requant_fp4_kperm(w2.view(u32), s2, qblock())
        gu = kperm(w13.shape[-1])
        return (
            w13k.view(jax.numpy.int32)[..., gu][..., cols],
            s13k[..., gu][..., cols],
            w2k.view(jax.numpy.int32),
            s2k,
        )
    if kind == "fp4":
        w13k, s13k, w2k, s2k = fp4_weights_kperm(
            w13.view(jax.numpy.uint32), s13, w2.view(jax.numpy.uint32), s2
        )
        return (
            w13k.view(jax.numpy.int32)[..., cols],
            s13k[..., cols],
            w2k.view(jax.numpy.int32),
            s2k,
        )
    return (
        permute_fp8_rows(w13.view(jax.numpy.uint32)).view(jax.numpy.int32)[..., cols],
        s13[..., cols],
        w2,
        s2,
    )
