# SPDX-License-Identifier: Apache-2.0
"""The EP decode megakernel: muse_megakernel.kernel's row-tiled program with the EPExperts
expert section, and the layout its weights take (EP_WEIGHTS)."""

import jax
import numpy as np

from sglang_tpu.muse_megakernel.ep_mk.config import ep_expert_shapes
from sglang_tpu.muse_megakernel.ep_mk.experts import (
    fp4_weights_kperm,
    kperm,
    permute_fp8_rows,
)
from sglang_tpu.muse_megakernel.ep_mk.layer import EPExperts
from sglang_tpu.muse_megakernel.kernel import megakernel_call
from sglang_tpu.muse_megakernel.mk.config import MKConfig

__all__ = ["ep_expert_shapes", "ep_expert_weights", "ep_megakernel_call", "pooled_perm"]


def ep_megakernel_call(cfg: MKConfig, axis: str = "model"):
    """megakernel_call's f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv) with
    each layer's experts as ep_expert_shapes (ep_expert_weights) and the pooled width in
    pooled_perm order (wpool's output columns, wpool_s and pre_expert_w permuted)."""
    return megakernel_call(cfg, axis, ep=EPExperts(cfg))


def pooled_perm(he: int) -> np.ndarray:
    """The pooled (expert input) width's order in the EP kernel."""
    return kperm(he)


def ep_expert_weights(
    kind: str, w13: jax.Array, s13: jax.Array, w2: jax.Array, s2: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One rank's whole experts in the EP kernel's layout, from the natural one ([E, H_e/pack,
    2I] w13, [E, I/pack, H_e] w2, int32 words): hidden rows in pooled_perm order; for fp4
    the gate / up columns and w2's rows in kperm order too (the dense arm's scale layout)."""
    if kind == "fp4":
        w13k, s13k, w2k, s2k = fp4_weights_kperm(
            w13.view(jax.numpy.uint32), s13, w2.view(jax.numpy.uint32), s2
        )
        return w13k.view(jax.numpy.int32), s13k, w2k.view(jax.numpy.int32), s2k
    return (
        permute_fp8_rows(w13.view(jax.numpy.uint32)).view(jax.numpy.int32),
        s13,
        w2,
        s2,
    )
