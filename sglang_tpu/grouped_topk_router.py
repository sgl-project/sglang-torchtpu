# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V3-style biased grouped top-k routing as one Pallas kernel.

Public entry point (import path ``sglang_tpu.grouped_topk_router``)::

    tpu_biased_grouped_topk(
        hidden_states: Tensor,           # [T, H]; only its row count is used
        gating_output: Tensor,           # [T, E] fp32 router logits
        correction_bias: Tensor,         # [E] fp32 e_score_correction_bias
        topk: int,
        renormalize: bool,
        num_expert_group: int,
        topk_group: int,
        num_fused_shared_experts: int = 0,     # must be 0
        routed_scaling_factor: float | None = None,
        apply_routed_scaling_factor_on_output: bool = False,
    ) -> tuple[Tensor, Tensor]           # (topk_weights fp32 [T, topk], topk_ids int32 [T, topk])

It is a drop-in for ``sglang.srt.layers.moe.topk.biased_grouped_topk_impl``
(same signature and outputs). Like it, ``routed_scaling_factor`` multiplies the
weights only when ``apply_routed_scaling_factor_on_output`` is set; Beam's
default runner applies the 2.5 after the expert sum, so Beam passes False.
Pad-row masking (``num_token_non_padded``) stays in ``select_experts``.

Algorithm (Beam: E=256, 8 groups of 32, keep 4 groups, top-8)::

    s       = sigmoid(logits)                      # fp32
    c       = s + bias                             # choice scores
    g[j]    = top1(c in group j) + top2(c in group j)
    keep    = top-`topk_group` groups by g
    ids     = top-`topk` experts by c among kept groups
    w       = s[ids] / (sum(s[ids]) + 1e-20)       # if renormalize

Tie-breaking: every selection takes the largest value and, among equal values,
the lowest index (expert id or group id). The output columns are in selection
order: descending choice score, ties by ascending expert id. The reference uses
``torch.topk(sorted=False)``, whose column order and tie winner are
unspecified (CPU torch 2.14 happens to keep the lowest-index set too; CUDA's
radix select is not guaranteed to), so compare outputs as (id, weight) sets
per row. The renormalizing
sum adds in selection order, so weights can differ from the reference in the
last fp32 ulp. ``sigmoid`` is the chip's fp32 logistic; on near-ties at the
selection boundary a 1-ulp sigmoid difference can swap an expert.

Layout: the natural one, tokens on sublanes and experts on lanes. The op
consumes the [T, E] logits and a [1, E] bias exactly as torch holds them and
writes [T, topk] outputs, and the exported module is a bare ``pallas_call``:
no transpose, reshape or copy on either side of the custom call. torch_tpu
splices the kernel module into the model graph and can drop or mishandle a
layout-changing op next to it inside a large compiled graph, so none may exist.
Every "max over experts" is a cross-lane reduction per 8-token row tile. A
group is a 32-lane range selected with a full-width mask, not a lane slice, so
no reduction needs an unaligned lane relayout. No gathers, sorts or reshapes.

Mosaic block rules relied on: each block's last two dims must be divisible by
(8, 128) or equal the full array dims. Logits (bm, E): E is the full width
(256 = 2 x 128 lanes); bm is 256 (8 | 256) or the full T. Bias (1, E): both
full. Outputs (bm, topk): topk equals the full last dim, so a [T, 8] output is
legal without lane padding; its stores are lane-masked (8 of 128 lanes).

No ``from __future__ import annotations``: torch_tpu resolves the ``_wrapped``
signature with ``eval_str``, and ``jax`` is only a local name there.
"""

from threading import Lock
from typing import TYPE_CHECKING, Callable, Optional

import torch

if TYPE_CHECKING:
    import jax

# Tokens (sublanes) per program. A [256, 256] fp32 block is 256 KiB and the
# kernel keeps about ten such temporaries live (~2.5 MiB). Smaller token counts
# use one block of the full height, which is always a legal block shape.
_MAX_BLOCK_TOKENS = 256
_RENORMALIZE_SUM_EPSILON = 1e-20


# --------------------------------------------------------------------------
# Torch reference (biased_grouped_topk_impl, without torch.compile)
# --------------------------------------------------------------------------


def biased_grouped_topk_ref(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``biased_grouped_topk_impl`` for ``num_fused_shared_experts == 0``."""
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    if num_fused_shared_experts:
        raise NotImplementedError("fused shared experts are not supported")
    scores = gating_output.sigmoid()
    num_token, num_experts = scores.shape
    scores_for_choice = scores.view(num_token, -1) + correction_bias.view(
        -1, num_experts
    )
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, num_experts // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights_sum = topk_weights.sum(dim=-1, keepdim=True, dtype=torch.float32)
        topk_weights = topk_weights / (topk_weights_sum + _RENORMALIZE_SUM_EPSILON)
        if apply_routed_scaling_factor_on_output:
            topk_weights *= routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def canonicalize_topk(
    weights: torch.Tensor, ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort each row's (id, weight) pairs by id so unordered outputs compare."""
    order = ids.argsort(dim=-1)
    return weights.gather(1, order), ids.gather(1, order)


# --------------------------------------------------------------------------
# Pallas kernel
# --------------------------------------------------------------------------


def _router_kernel(
    logits_ref,
    bias_ref,
    w_ref,
    ids_ref,
    *,
    topk: int,
    num_groups: int,
    topk_group: int,
    renormalize: bool,
    scale: float,
):
    """One block of tokens: logits_ref is [bm, E], bias_ref [1, E].

    Every reduction is over the last axis (experts on lanes) and yields
    [bm, 1]; the outputs are [bm, topk].
    """
    import jax
    import jax.numpy as jnp

    f32 = jnp.float32
    neg_inf = jnp.float32(-jnp.inf)
    logits = logits_ref[...].astype(f32)
    # A NaN makes every `values == best` below false, so argmax_lowest would
    # return num_experts (out of range) for the whole row. Compiled-shape pad
    # rows can hold NaN, and an out-of-range id corrupts the real tokens in the
    # fused MoE. Treat NaN as -inf: that expert scores 0 and is never chosen.
    logits = jnp.where(jnp.isnan(logits), neg_inf, logits)
    bm, num_experts = logits.shape
    per_group = num_experts // num_groups

    scores = jax.nn.sigmoid(logits)
    choice = scores + bias_ref[...].astype(f32)
    col = jax.lax.broadcasted_iota(jnp.int32, (bm, num_experts), 1)

    def argmax_lowest(values):
        """Max over experts and the lowest expert id holding it, both [bm, 1]."""
        best = jnp.max(values, axis=-1, keepdims=True)
        idx = jnp.min(
            jnp.where(values == best, col, num_experts), axis=-1, keepdims=True
        )
        return best, idx

    # Group score: sum of the two largest choice scores in the group. With a
    # tie for the maximum both top values equal it, as in torch.topk(2). A
    # group is a lane range, selected by a full-width mask (see module doc).
    in_groups = []
    group_scores = []
    for g in range(num_groups):
        in_group = (col >= g * per_group) & (col < (g + 1) * per_group)
        c_g = jnp.where(in_group, choice, neg_inf)
        top1, top1_idx = argmax_lowest(c_g)
        top2 = jnp.max(jnp.where(col == top1_idx, neg_inf, c_g), axis=-1, keepdims=True)
        in_groups.append(in_group)
        group_scores.append(top1 + top2)

    # Keep group g iff fewer than topk_group groups beat it, where h beats g
    # when its score is larger, or equal with a lower index: a stable top-k.
    keep = jnp.zeros((bm, num_experts), dtype=jnp.bool_)
    for g in range(num_groups):
        beaten_by = jnp.zeros((bm, 1), dtype=jnp.int32)
        for h in range(num_groups):
            if h == g:
                continue
            if h < g:
                wins = group_scores[h] >= group_scores[g]
            else:
                wins = group_scores[h] > group_scores[g]
            beaten_by = beaten_by + wins.astype(jnp.int32)
        keep = keep | (in_groups[g] & (beaten_by < topk_group))

    candidates = jnp.where(keep, choice, neg_inf)
    slot = jax.lax.broadcasted_iota(jnp.int32, (bm, topk), 1)
    weights = jnp.zeros((bm, topk), f32)
    ids = jnp.zeros((bm, topk), jnp.int32)
    total = jnp.zeros((bm, 1), f32)
    for j in range(topk):
        _, idx = argmax_lowest(candidates)
        chosen = col == idx
        # Exactly one expert is chosen, so the masked sum is an exact gather.
        w = jnp.sum(jnp.where(chosen, scores, 0.0), axis=-1, keepdims=True)
        total = total + w
        weights = jnp.where(slot == j, w, weights)
        ids = jnp.where(slot == j, idx, ids)
        candidates = jnp.where(chosen, neg_inf, candidates)

    if renormalize:
        weights = weights / (total + _RENORMALIZE_SUM_EPSILON)
        if scale != 1.0:
            weights = weights * scale
    w_ref[...] = weights
    ids_ref[...] = ids


def biased_grouped_topk_jax(
    gating_output: "jax.Array",  # [T, E] fp32
    correction_bias: "jax.Array",  # [1, E] fp32
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool = True,
    scale: float = 1.0,
    *,
    interpret: bool = False,
) -> "tuple[jax.Array, jax.Array]":
    """JAX-level router: [T, E] logits in, ([T, topk] fp32, [T, topk] int32) out.

    ``scale`` is the routed scale when it is applied here (1.0 otherwise).
    This is a bare ``pallas_call`` on the arrays as given: with fp32 inputs
    nothing (no transpose, reshape or convert) surrounds the kernel.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    m, num_experts = gating_output.shape
    if correction_bias.shape == (num_experts,):
        # Reshape here rather than in torch, like the sibling Beam kernels, so
        # no reshape of a graph value sits next to the custom call.
        correction_bias = correction_bias.reshape(1, num_experts)
    if correction_bias.shape != (1, num_experts):
        raise ValueError(
            f"correction_bias must be [{num_experts}] or [1, {num_experts}], "
            f"got {tuple(correction_bias.shape)}."
        )
    # Block rows: a multiple of 8 sublanes, or the full token count.
    bm = min(m, _MAX_BLOCK_TOKENS)
    kernel = lambda *refs: _router_kernel(
        *refs,
        topk=topk,
        num_groups=num_expert_group,
        topk_group=topk_group,
        renormalize=renormalize,
        scale=scale,
    )
    return pl.pallas_call(
        kernel,
        grid=(pl.cdiv(m, bm),),
        in_specs=[
            pl.BlockSpec((bm, num_experts), lambda i: (i, 0)),
            pl.BlockSpec((1, num_experts), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((bm, topk), lambda i: (i, 0)),
            pl.BlockSpec((bm, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((m, topk), jnp.float32),
            jax.ShapeDtypeStruct((m, topk), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="biased_grouped_topk",
    )(gating_output.astype(jnp.float32), correction_bias.astype(jnp.float32))


_OP: Callable | None = None
_OP_LOCK = Lock()


def _get_op() -> Callable:
    global _OP
    if _OP is not None:
        return _OP
    with _OP_LOCK:
        if _OP is not None:
            return _OP

        import jax  # noqa: F401  (force loud failure if JAX is missing)

        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        def _wrapped(
            gating_output: jax.Array,
            correction_bias: jax.Array,
            topk: int,
            num_expert_group: int,
            topk_group: int,
            renormalize: int,
            scale: float,
        ) -> tuple[jax.Array, jax.Array]:
            # [T, E] logits and [1, E] bias in, [T, topk] out: the exported
            # module is a bare pallas_call and the wrapper adds no transpose
            # or copy around it (see the module doc).
            return biased_grouped_topk_jax(
                gating_output,
                correction_bias,
                topk,
                num_expert_group,
                topk_group,
                bool(renormalize),
                scale,
            )

        # Rank-local kernel: the router is replicated, no mesh needed.
        op = cpu_export_jax_op("sglang_tpu_pallas::biased_grouped_topk", _wrapped)

        def _fake(gating_output, correction_bias, topk, *_):
            m = gating_output.shape[0]
            return (
                gating_output.new_empty((m, topk), dtype=torch.float32),
                gating_output.new_empty((m, topk), dtype=torch.int32),
            )

        op.register_fake(_fake)
        _OP = op
        return op


def register_tpu_biased_grouped_topk_op() -> None:
    """Register the op before torch.compile traces a model."""
    _get_op()


def can_use_tpu_biased_grouped_topk(
    *,
    num_experts: int,
    topk: int,
    num_expert_group: Optional[int],
    topk_group: Optional[int],
    num_fused_shared_experts: int = 0,
    scoring_func: str = "sigmoid",
) -> bool:
    """Whether a router config is in the kernel's validated envelope."""
    if scoring_func != "sigmoid" or num_fused_shared_experts:
        return False
    if not num_expert_group or not topk_group:
        return False
    if num_experts % num_expert_group:
        return False
    per_group = num_experts // num_expert_group
    # Group masks need no alignment, but only whole 8-expert groups (Beam:
    # 32) are validated on a chip.
    return (
        per_group % 8 == 0
        and 0 < topk_group <= num_expert_group
        and 0 < topk <= topk_group * per_group
    )


def tpu_biased_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for ``biased_grouped_topk_impl`` (see the module doc)."""
    if hidden_states.shape[0] != gating_output.shape[0]:
        raise ValueError("Number of tokens mismatch")
    if gating_output.dim() != 2 or gating_output.shape[0] == 0:
        raise ValueError(
            f"gating_output must be [tokens >= 1, experts], got "
            f"{tuple(gating_output.shape)}."
        )
    num_experts = gating_output.shape[1]
    if correction_bias.shape != (num_experts,):
        # A per-token [T, E] bias (dynamic_expert_bias) takes the torch path.
        raise ValueError(
            f"correction_bias must be [{num_experts}], got "
            f"{tuple(correction_bias.shape)}."
        )
    if not can_use_tpu_biased_grouped_topk(
        num_experts=num_experts,
        topk=topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=num_fused_shared_experts,
    ):
        raise ValueError(
            f"unsupported routing config: E={num_experts} topk={topk} "
            f"groups={num_expert_group} topk_group={topk_group} "
            f"shared={num_fused_shared_experts}"
        )
    if gating_output.device.type != "tpu":
        return biased_grouped_topk_ref(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            num_expert_group,
            topk_group,
            num_fused_shared_experts,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
    scale = (
        float(routed_scaling_factor)
        if renormalize
        and apply_routed_scaling_factor_on_output
        and routed_scaling_factor is not None
        else 1.0
    )
    # Logits and bias go in as torch holds them (the [E] -> [1, E] reshape is
    # inside the op); the outputs are already [T, topk]: nothing surrounds the
    # custom call.
    return _get_op()(
        gating_output.to(torch.float32),
        correction_bias.to(torch.float32),
        topk,
        num_expert_group,
        topk_group,
        int(bool(renormalize)),
        scale,
    )
