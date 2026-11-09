# SPDX-License-Identifier: Apache-2.0
"""The FFN-section hook of the row-tiled program: build_kernel(cfg, ep=hook) runs
hook.layer_moe after the post-attention residual in place of the pool, router, expert-TP
loop, unpool and the FFN residual (ep_mk/)."""

from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

import jax

from sglang_tpu.muse_megakernel.mk.config import MKConfig


class EPLayerStep(NamedTuple):
    """One layer's FFN section as the hook sees it: every row's post-attention stream in
    S["h_stream"] and its pre-FFN-normed bf16 in S["lhs"]."""

    c: MKConfig
    S: dict[str, Any]  # scratch refs by plan name
    W: dict[str, Any]  # dense weight refs by WEIGHT_NAMES
    smr: Any  # this layer's smalls slab ref
    layer: jax.Array | int  # traced in the rolled fp4 layer loop
    kind: str  # "fp4" | "fp8"
    # for_layer(fn): fn(refs, off) on this layer's (w13, w13_scale, w2, w2_scale) refs, off its
    # index in stacked refs or None; one branch-tree walk when the layer index is traced
    for_layer: Callable[[Callable[[tuple, Any], None]], None]
    wait_refs: tuple  # (refs, off) of one layer of this kind: DMA wait descriptors
    me: jax.Array | int
    n_live: jax.Array
    router: Callable[
        [], None
    ]  # every row's router logits into S["acc"][:, :Ep] (the ring)
    capture: Callable[[Any, jax.Array], None]  # capture(rows, stream) of row_tile rows


class EPHook(Protocol):
    def scratch_plan(self, cfg: MKConfig, plan: list) -> list:
        """The row-tiled scratch plan with the hook's buffers."""
        ...

    def layer_moe(self, step: EPLayerStep) -> None:
        """Leaves every row's post-FFN stream in S["h_stream"] (captured where asked)."""
        ...
