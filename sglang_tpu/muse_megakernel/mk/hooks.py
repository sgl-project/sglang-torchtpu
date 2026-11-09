# SPDX-License-Identifier: Apache-2.0
"""The hooks of the row-tiled program (ep_mk/): build_kernel(cfg, ep=hook) runs hook.o_proj
in place of the o-proj reduce-scatter / all-gather and the post-attention residual, and
hook.layer_moe in place of the pool, router, expert-TP loop, unpool and FFN residual."""

from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

import jax

from sglang_tpu.muse_megakernel.mk.config import MKConfig


class EPLayerStep(NamedTuple):
    """One layer's FFN section as the hook sees it: after hook.o_proj, this rank's rows'
    post-attention stream in S["h_stream"] and their pre-FFN-normed bf16 in S["lhs"]."""

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


class EPOProjStep(NamedTuple):
    """The o-projection as the hook sees it: every row's gated attention output in S["lhs"]
    [:, :q_dim] (this rank's heads)."""

    c: MKConfig
    S: dict[str, Any]
    smr: Any
    layer: jax.Array | int
    me: jax.Array | int
    n_live: jax.Array
    # mv(on_col): every row's o-proj partial through the dense ring, one wo column tile of
    # hidden_local columns at a time into S["acc"][:, :hidden_local]; on_col(ni, col) once
    # tile ni (physical column block col) is final
    mv: Callable[[Callable[[int, Any], None]], None]


class EPHook(Protocol):
    def scratch_plan(self, cfg: MKConfig, plan: list) -> list:
        """The row-tiled scratch plan with the hook's buffers."""
        ...

    def o_proj(self, step: EPOProjStep) -> None:
        """Replaces the o-proj reduce-scatter / all-gather and the post-attention residual."""
        ...

    def layer_moe(self, step: EPLayerStep) -> None:
        """Leaves every row's post-FFN stream in S["h_stream"] (captured where asked)."""
        ...
