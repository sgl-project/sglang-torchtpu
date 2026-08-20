# SPDX-License-Identifier: Apache-2.0
"""Rank-major compact Q layout planner for the PCP streaming kernel.

The PCP streaming kernel expects the Q tensor packed rank-major, with
BATCH-FLAT ownership: the batch's request-major query rows are treated as
one flat token stream, split into ``interleave_size`` chunks, and rank r
owns flat indices
``[k*cycle + r*interleave, k*cycle + (r+1)*interleave)`` where
``cycle = pcp_size * interleave_size``. The owner coordinate of a request
is therefore ``cu_q_lens[req]``. Request-absolute coordinates (causal
positions, KV-cache slots) are a separate axis the kernel derives on
device from ``kv_lens - q_lens``; this planner never needs them. sglang
batches Q request-major, so this planner computes per-forward the reorder
indices needed to convert between the two layouts.

Design points:

1. Pure host/NumPy: the caller passes per-request token counts pulled off
   ``ForwardBatch`` (``extend_seq_lens_cpu``, ``extend_prefix_lens_cpu``,
   ``seq_lens_cpu``), so there is no PyTorch device roundtrip.
2. Uses the ``pcp_layout`` helpers directly
   (``pcp_local_token_counts``, ``build_pcp_rank_major_token_order``) —
   pure NumPy.
3. The plan is a ``msgspec.Struct`` (frozen) attached to
   ``forward_batch.pcp_layout_plan``; consumed by
   ``apply_pcp_local_shard`` at attention time and by
   ``gather_and_unpack_pcp`` at the sampler boundary.
4. Boundary-crossing rejection (a single request straddling the
   prompt/decode split within one forward): kernel-level constraint.

The runner integration hook is
:meth:`sglang.srt.layers.cp.pcp_streaming.PCPStreamingCPStrategy.build_extra_metadata`,
invoked from :func:`sglang.srt.layers.cp.utils.prepare_cp_forward`. It
calls :func:`plan_pcp_sequence_layout` here and attaches the returned
``PcpSequenceLayoutPlan`` to ``forward_batch.pcp_layout_plan`` (plus
plain-scalar mirrors for the compiled forward).
"""

from __future__ import annotations

from typing import Optional

import msgspec
import numpy as np
import torch

from sglang.srt.layers.attention.pallas.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
    pcp_local_token_counts,
)


class PcpSequenceLayoutPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Per-forward-batch reorder plan attached to
    ``forward_batch.pcp_layout_plan`` before the model forward.

    All arrays are NumPy on host. The compiled forward never reads this
    struct (nor ``forward_batch.attn_cp_metadata``) — the load-bearing
    counts are mirrored onto ``ForwardBatch`` as plain ints by
    ``build_extra_metadata``. The reorder indices themselves are applied by
    ``apply_pcp_local_shard`` to the on-device hidden states and positions.
    """

    pcp_size: int
    interleave_size: int
    pcp_rank: int
    # (padded_num_tokens,) int64: for each destination slot in rank-major
    # packed order, the source index in the original request-major batch.
    # -1 for padding slots.
    token_order: np.ndarray
    # (total_num_tokens,) int64: inverse of token_order, for the sampler
    # boundary's rank-major → request-major rewrite.
    inverse_order: np.ndarray
    # (pcp_size,) int32: number of tokens each rank owns after reorder.
    local_token_counts: np.ndarray
    # Global (unsharded) token count = sum(scheduled_tokens_per_req).
    global_num_tokens: int
    # Global padded token count. Must be divisible by pcp_size.
    global_padded_num_tokens: int
    # This rank's local token count (already padded within-rank).
    local_num_tokens: int
    local_padded_num_tokens: int


def resolve_pcp_interleave_size(page_size: int) -> int:
    """Resolve the PCP ring-ownership granularity, in tokens.

    ``SGLANG_TPU_PCP_INTERLEAVE_SIZE=0`` (the default) means "use page_size" —
    the coarsest legal setting, one whole KV page per rank per virtual page.
    That is the fast setting: a ring page group spans one owner cycle
    (``pcp_size * interleave_size``), so the number of ring steps scales as
    ``page_size / interleave_size``. Anything smaller is a deliberate,
    measured choice.
    """
    page_size = int(page_size)
    if page_size <= 0:
        raise ValueError(f"page_size must be > 0 for PCP, got {page_size}.")

    from sglang.srt.environ import envs

    configured = int(envs.SGLANG_TPU_PCP_INTERLEAVE_SIZE.get())
    if configured == 0:
        return page_size
    if configured < 0:
        raise ValueError(
            "SGLANG_TPU_PCP_INTERLEAVE_SIZE must be >= 0 (0 = match "
            f"page_size), got {configured}."
        )
    return configured


# Largest (ring tile x local Q heads) product the current-pass Pallas kernel
# fits on a v7 core. The tile sizes the fp32 online-softmax accumulator and the
# (m, l) state, and past this point the register allocator starts spilling:
# measured on v7-8, q_block=256 at 96 Q heads asks for 17.98 MB of spill slots
# on top of a 16 MB accumulator and fails to compile. Empirically 8 Q heads
# fits 2048, 32 heads fits 512, and 96 heads fits only 128 -- all consistent
# with this product.
_PCP_RING_TILE_HEAD_BUDGET = 16384
_PCP_RING_TILE_MIN = 128
_PCP_RING_TILE_MAX = 1024


def resolve_pcp_ring_tile_tokens(num_q_heads: int) -> int:
    """Resolve how many local Q rows share one trip around the PCP ring.

    The kernel loops (Q tile, ring page group) with the ring on the inside, so
    the whole causal history is re-streamed around the ring once per Q tile:
    ring bytes scale as ``(tiles + 1) / 2`` for
    ``tiles = local_tokens / ring_tile``. Making the tile bigger is therefore
    close to a linear reduction in ring traffic. Measured on v7-8, pcp=8, at
    32 Q heads: ISL 32768 goes 30.3 ms at 128 -> 14.7 ms at 512 (2.07x), and
    ISL 8192 goes 2.93 ms -> 2.07 ms (1.41x).

    The ceiling is VMEM plus register spills, which scale with
    ``tile * num_q_heads`` -- hence the head-count-derived default. Override
    with ``SGLANG_TPU_PCP_RING_TILE_TOKENS`` (0 = auto).

    This value is also the compile-bucket floor: the kernel requires the global
    padded token count to be a multiple of ``pcp_size * ring_tile``, so a
    larger tile means coarser buckets. That is the right trade for the long
    prefills PCP targets and the wrong one for tiny batches.
    """
    num_q_heads = int(num_q_heads)
    if num_q_heads <= 0:
        raise ValueError(f"num_q_heads must be > 0, got {num_q_heads}.")

    from sglang.srt.environ import envs

    configured = int(envs.SGLANG_TPU_PCP_RING_TILE_TOKENS.get())
    if configured < 0:
        raise ValueError(
            "SGLANG_TPU_PCP_RING_TILE_TOKENS must be >= 0 (0 = auto), "
            f"got {configured}."
        )
    if configured:
        return configured

    budget = _PCP_RING_TILE_HEAD_BUDGET // num_q_heads
    tile = _PCP_RING_TILE_MIN
    while tile * 2 <= min(budget, _PCP_RING_TILE_MAX):
        tile *= 2
    return tile


def plan_pcp_sequence_layout(
    *,
    num_scheduled_tokens_per_req: np.ndarray,
    prompt_tokens_per_req: np.ndarray,
    computed_tokens_per_req: np.ndarray,
    padded_num_tokens: int,
    pcp_size: int,
    pcp_rank: int,
    interleave_size: int,
) -> PcpSequenceLayoutPlan:
    """Compute the rank-major reorder plan for a single forward batch.

    Query ownership is batch-flat: the batch's request-major query rows are
    treated as one flat token stream and dealt round-robin in
    ``interleave_size`` chunks, so the owner coordinate of a row is just its
    index in the packed batch. That is independent of the request-absolute
    coordinates (causal positions, KV-cache slots), which the kernel derives
    on device from ``kv_lens - q_lens``.

    Ownership by request-absolute position instead would restart the
    interleave cycle at each request, which piles rows onto whichever ranks
    the requests' absolute starts happen to be congruent to. Since every rank
    pads to the batch-wide maximum local count, one hot rank inflates the
    compile bucket for the whole layer stack. At decode (q_len == 1) the
    absolute scheme keys ownership off each request's history length, which
    for uniform-length traffic puts the entire batch on one rank.

    Rejects boundary-crossing requests (a request whose scheduled tokens
    straddle its prompt/decode boundary within a single forward) — the
    kernel doesn't support this yet.

    Sglang almost never generates boundary-crossing forwards in practice
    (chunked prefill chunks and decode steps are separate), but the
    check is here as a runtime guard.
    """
    if pcp_size < 1:
        raise ValueError(f"pcp_size must be >= 1, got {pcp_size}")
    if interleave_size < 1:
        raise ValueError(f"interleave_size must be >= 1, got {interleave_size}")
    if padded_num_tokens % pcp_size != 0:
        raise ValueError(
            f"padded_num_tokens={padded_num_tokens} must be divisible by "
            f"pcp_size={pcp_size} — increase the compile bucket."
        )
    if pcp_size > 1 and padded_num_tokens % (pcp_size * interleave_size) != 0:
        raise ValueError(
            f"padded_num_tokens={padded_num_tokens} must be divisible by "
            f"pcp_size*interleave_size={pcp_size * interleave_size} — the "
            f"kernel places tokens in {interleave_size}-token chunks across "
            f"{pcp_size} ranks, so any tail smaller than one cycle leaves "
            f"ranks with mismatched slots. Increase the compile bucket."
        )

    num_scheduled_tokens_per_req = np.asarray(
        num_scheduled_tokens_per_req, dtype=np.int64
    )
    prompt_tokens_per_req = np.asarray(prompt_tokens_per_req, dtype=np.int64)
    computed_tokens_per_req = np.asarray(computed_tokens_per_req, dtype=np.int64)
    if not (
        num_scheduled_tokens_per_req.size
        == prompt_tokens_per_req.size
        == computed_tokens_per_req.size
    ):
        raise ValueError(
            "num_scheduled_tokens_per_req, prompt_tokens_per_req, and "
            "computed_tokens_per_req must have the same length."
        )

    # Defensive: reject boundary-crossing (a single request's scheduled
    # span containing both prompt and decode tokens). The sglang driver
    # passes ``prompt_tokens = computed + scheduled`` (see
    # ``PCPStreamingCPStrategy.build_extra_metadata``), so this branch is
    # unreachable from the sglang call site — kept here for standalone
    # callers (unit tests).
    end = computed_tokens_per_req + num_scheduled_tokens_per_req
    crosses = (
        (num_scheduled_tokens_per_req > 0)
        & (computed_tokens_per_req < prompt_tokens_per_req)
        & (end > prompt_tokens_per_req)
    )
    if np.any(crosses):
        raise NotImplementedError(
            "PCP streaming does not support prompt/decode boundary-crossing "
            "schedules: a single request cannot have both prompt tokens "
            "AND decode tokens in the same forward. Offending request "
            f"indices: {list(np.nonzero(crosses)[0])}."
        )

    total_num_tokens = int(np.sum(num_scheduled_tokens_per_req))

    if pcp_size <= 1:
        # Trivial layout: no rank sharding, identity reorder.
        token_order = np.full(padded_num_tokens, -1, dtype=np.int64)
        token_order[:total_num_tokens] = np.arange(total_num_tokens, dtype=np.int64)
        inverse_order = np.arange(total_num_tokens, dtype=np.int64)
        local_counts = np.array([total_num_tokens], dtype=np.int32)
        return PcpSequenceLayoutPlan(
            pcp_size=1,
            interleave_size=interleave_size,
            pcp_rank=0,
            token_order=token_order,
            inverse_order=inverse_order,
            local_token_counts=local_counts,
            global_num_tokens=total_num_tokens,
            global_padded_num_tokens=padded_num_tokens,
            local_num_tokens=total_num_tokens,
            local_padded_num_tokens=padded_num_tokens,
        )

    # Batch-flat owner coordinate: exactly cu_q_lens[:-1].
    token_owner_starts = np.zeros(num_scheduled_tokens_per_req.shape, dtype=np.int64)
    if num_scheduled_tokens_per_req.size > 1:
        np.cumsum(
            num_scheduled_tokens_per_req[:-1],
            dtype=np.int64,
            out=token_owner_starts[1:],
        )

    local_counts = pcp_local_token_counts(
        num_scheduled_tokens_per_req,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        token_owner_start_offsets_per_req=token_owner_starts,
    )
    token_order, inverse_order = build_pcp_rank_major_token_order(
        num_scheduled_tokens_per_req,
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        padded_num_tokens=padded_num_tokens,
        token_owner_start_offsets_per_req=token_owner_starts,
    )
    local_padded = padded_num_tokens // pcp_size
    return PcpSequenceLayoutPlan(
        pcp_size=pcp_size,
        interleave_size=interleave_size,
        pcp_rank=pcp_rank,
        token_order=token_order,
        inverse_order=inverse_order,
        local_token_counts=local_counts,
        global_num_tokens=total_num_tokens,
        global_padded_num_tokens=padded_num_tokens,
        local_num_tokens=int(local_counts[pcp_rank]),
        local_padded_num_tokens=local_padded,
    )


def apply_pcp_local_shard(
    x: torch.Tensor, forward_batch, axis: int = 0
) -> torch.Tensor:
    """Slice ``x`` to this rank's rank-major-packed local rows.

    ``forward_batch.pcp_layout_plan`` must have been set by
    :func:`plan_pcp_sequence_layout` (via the ``ModelRunner`` integration
    hook). ``x`` is expected in request-major order (shape ``(global_num_
    tokens, ...)``); the returned tensor has shape
    ``(local_padded_num_tokens, ...)`` and contains this rank's owned
    rows in rank-major order.
    """
    plan: Optional[PcpSequenceLayoutPlan] = forward_batch.pcp_layout_plan
    if plan is None:
        raise RuntimeError(
            "apply_pcp_local_shard called with no pcp_layout_plan on "
            "forward_batch — ensure prepare_cp_forward() invoked "
            "plan_pcp_sequence_layout before the model forward."
        )
    if plan.pcp_size <= 1:
        return x

    # Compute this rank's slice of the reordered token_order. Ranks are
    # laid out contiguously in the packed order: rank r owns rows
    # [r*local_padded, (r+1)*local_padded).
    r = plan.pcp_rank
    lp = plan.local_padded_num_tokens
    rank_slice_np = plan.token_order[r * lp : (r + 1) * lp]

    # Compute the validity mask + -1-sentinel clamp on the HOST (numpy) so
    # we avoid the D->H sync that ``bool(valid.all())`` on a device tensor
    # triggers. That sync used to run twice per forward (hidden + positions
    # in cp_split_before_forward) and blocked dispatch of the next forward.
    all_valid = bool((rank_slice_np >= 0).all())
    if all_valid:
        safe_idx_np = rank_slice_np
    else:
        safe_idx_np = np.where(rank_slice_np >= 0, rank_slice_np, 0)

    device = x.device
    safe_idx = torch.as_tensor(safe_idx_np, dtype=torch.int64, device=device)
    if axis == 0:
        gathered = x[safe_idx]
        if not all_valid:
            # Zero out padding rows: the kernel masks padded Q rows to zero
            # output, and downstream gathers must not read embedding noise
            # out of a slot that owns no token.
            valid_np = rank_slice_np >= 0
            valid = torch.as_tensor(valid_np, dtype=torch.bool, device=device)
            pad_mask = (~valid).view(-1, *([1] * (gathered.dim() - 1)))
            gathered = torch.where(pad_mask, torch.zeros_like(gathered), gathered)
        return gathered
    raise NotImplementedError("apply_pcp_local_shard currently only supports axis=0.")


def gather_and_unpack_pcp(x: torch.Tensor, forward_batch) -> torch.Tensor:
    """All-gather rank-local hidden states across the PCP ring, then undo
    the rank-major packing so downstream sampler code sees request-major
    rows.

    ``x`` is this rank's local hidden tensor of shape
    ``(local_padded_num_tokens, ...)``. Returned tensor is
    ``(global_num_tokens, ...)`` in request-major order.
    """
    plan: Optional[PcpSequenceLayoutPlan] = forward_batch.pcp_layout_plan
    if plan is None or plan.pcp_size <= 1:
        return x

    from sglang.srt.distributed.pcp_mesh import all_gather_equal_tokens

    # PCP local shards are equal-sized (``local_padded_num_tokens`` per rank
    # by construction), so a plain all-gather along axis 0 concatenates
    # them in ``rank_in_group`` order — matching the rank-major packing
    # ``token_order`` was built against.
    x = x.contiguous()
    gathered = all_gather_equal_tokens(x, dim=0)
    # ``gathered`` now has shape ``(pcp_size * local_padded_num_tokens, ...)
    # == (global_padded_num_tokens, ...)`` in rank-major packed order.
    device = gathered.device
    inv = torch.as_tensor(plan.inverse_order, dtype=torch.int64, device=device)
    return gathered[inv]
