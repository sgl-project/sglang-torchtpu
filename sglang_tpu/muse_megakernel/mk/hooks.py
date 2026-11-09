# SPDX-License-Identifier: Apache-2.0
"""The expert-section hook of the row-tiled program: build_kernel(cfg, ep=hook) runs
hook.moe in place of the expert-TP loop, y reduce-scatter and combine (ep_mk/)."""

from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

import jax

from sglang_tpu.muse_megakernel.mk.config import MKConfig


class EPMoEStep(NamedTuple):
    """What one layer's expert section sees, after top-k, the pooled all-gather and xfull."""

    c: MKConfig
    S: dict[str, Any]  # scratch refs by plan name
    smr: Any  # this layer's smalls slab ref
    layer: jax.Array | int  # traced in the rolled fp4 layer loop
    kind: str  # "fp4" | "fp8"
    # for_layer(fn): fn(refs, off) on this layer's (w13, w13_scale, w2, w2_scale) refs, off its
    # index in stacked refs or None; one branch-tree walk when the layer index is traced
    for_layer: Callable[[Callable[[tuple, Any], None]], None]
    wait_refs: tuple  # (refs, off) of one layer of this kind: DMA wait descriptors
    me: jax.Array | int
    n_live: jax.Array
    flush_dense: Callable[
        [], None
    ]  # issues the dense refills held during the expert phase


class EPHook(Protocol):
    def scratch_plan(self, cfg: MKConfig, plan: list) -> list:
        """The row-tiled scratch plan with the hook's buffers (and without unused ones)."""
        ...

    def moe(self, step: EPMoEStep) -> None:
        """Leaves this device's rows' combined expert output u in S["u_send"]."""
        ...
