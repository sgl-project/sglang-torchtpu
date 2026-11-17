# SPDX-License-Identifier: Apache-2.0
"""TPU Pallas diffusion ring attention kernel with in-kernel ICI DMA hops and hybrid fixed-m softmax."""

import functools
import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np

LOG2_E = 1.44269504
# Fixed-m heads compute p = exp2(s - m) for base-2 logits s with one m for every
# head. A head is admitted when Cauchy-Schwarz bounds |s| by
# FIXED_M_RING_SAFE_BOUND, and m = floor(bound) - FIXED_M_RECENTER keeps every
# exp2 argument in [-102, 88]: l and o stay finite in float32, and p * v stays a
# normal float (TPU flushes subnormals) down to |v| = 2**-24.
FIXED_M_RECENTER = 88.0
FIXED_M_RING_SAFE_BOUND = 95.0

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
SV_DIM_NUMBERS = (((0,), (0,)), ((), ()))

_COLLECTIVE_ID = 8
VMEM_LIMIT_BYTES = 64 * 2**20
_LANE = 128

_DEFAULT_BLOCK_Q = 512
_DEFAULT_BLOCK_KV = 2048
_UNTUNED_VMEM_BUDGET = 56 * 2**20
_V7X_VMEM_CAP_BYTES = int(63.94 * 2**20)
_HYBRID_RAGGED_SCORE_LIMIT = 8_700_000

# Tuned (block_q, block_kv, block_kv_compute, block_q_sub) tiles on TPU v7x keyed by (ring_size, shard_len).
_TUNED_BLOCKS: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (8, 4950): (5120, 2560, 2560, 2560),
    (8, 9450): (9600, 2560, 2560, 3200),
    (8, 13950): (14080, 3584, 3584, 2816),
    (8, 18450): (18816, 2688, 2688, 2688),
    (8, 22950): (23040, 2560, 2560, 2560),
    (4, 9900): (9984, 2560, 2560, 3328),
    (4, 18900): (19200, 3328, 3328, 3200),
    (4, 27900): (28160, 2816, 2816, 2816),
    (4, 36900): (18560, 8192, 2048, 3712),
    (4, 45900): (23040, 2560, 2560, 2560),
    (8, 2145): (2304, 2048, 2048, 1152),
    (8, 4095): (4096, 4096, 4096, 2048),
    (8, 6045): (6144, 6144, 2048, 3072),
    (8, 7995): (8192, 2048, 2048, 4096),
    (8, 9945): (9984, 2048, 2048, 3328),
    (4, 4290): (4352, 4352, 4352, 2176),
    (4, 8190): (8192, 8192, 4096, 2048),
    (4, 12090): (12160, 2432, 2432, 2432),
    (4, 15990): (16000, 2816, 2816, 3200),
    (4, 19890): (19968, 5632, 5632, 1536),
}

# Pure online-softmax tiles when use_fixed_m is False.
_TUNED_ONLINE_BLOCKS: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (4, 8190): (8448, 4096, 4096, 2816),
    (4, 12090): (12544, 6144, 6144, 1792),
    (4, 15990): (16128, 6144, 2048, 5376),
    (4, 19890): (19968, 3328, 3328, 3328),
    (4, 9900): (9984, 3584, 3584, 3328),
    (4, 18900): (19200, 2816, 2816, 3840),
    (4, 36900): (37632, 1024, 1024, 5376),
    (4, 45900): (23040, 3072, 3072, 2304),
}


def auto_block_q_sub(
    block_q: int, lane: int = _LANE, max_q_sub: int | None = None
) -> int:
    """Return the largest lane-multiple divisor of block_q within ceiling."""
    ceiling = max(lane, (block_q // 4) // lane * lane)
    if max_q_sub is not None:
        ceiling = min(ceiling, max_q_sub)
    q_sub = lane
    d = lane
    while d <= block_q:
        if block_q % d == 0:
            if d <= ceiling:
                q_sub = d
            else:
                break
        d += lane
    return q_sub


_SINGLE_BUFFER_Q_MIN = 4608


def is_single_buffered_q(block_q: int, shard_len: int) -> bool:
    """Return True when Q/Out use pl.Buffered(1) on whole-shard tiles above 4608 rows."""
    return block_q >= shard_len and block_q > _SINGLE_BUFFER_Q_MIN


def estimate_vmem_bytes(
    block_q: int,
    block_q_sub: int,
    block_kv: int,
    block_kv_compute: int | None = None,
    *,
    single_buffered_q: bool = True,
    head_dim: int = 128,
) -> int:
    """Estimate scoped-VMEM usage in bytes on TPU v7x."""
    if block_kv_compute is None:
        block_kv_compute = block_kv
    scale = head_dim / 128.0
    ml_bytes_per_row = 2.0 * NUM_SUBLANES * 4.0
    per_row_128 = 1128.8 if single_buffered_q else 1661.9
    per_row = ml_bytes_per_row + (per_row_128 - ml_bytes_per_row) * scale
    est = (
        per_row * block_q
        + 4.28 * block_q_sub * block_kv_compute
        + 1274.0 * scale * block_kv
        - 5.99 * 2**20 * scale
    )
    return max(0, int(est + 0.999999))


def _untuned_q_sub(
    block_q: int,
    block_kv: int,
    head_dim: int,
    *,
    single_buffered_q: bool = True,
) -> int:
    """Choose block_q_sub so the score tile fits within the VMEM budget."""
    min_vmem = estimate_vmem_bytes(
        block_q,
        _LANE,
        block_kv,
        block_kv,
        single_buffered_q=single_buffered_q,
        head_dim=head_dim,
    )
    if min_vmem > _UNTUNED_VMEM_BUDGET:
        raise ValueError(
            f"ring attention block_q={block_q} exceeds the "
            f"{_UNTUNED_VMEM_BUDGET // 2**20} MiB VMEM budget"
        )
    ceiling = max(_LANE, (block_q // 4) // _LANE * _LANE)
    q_sub = _LANE
    for d in range(_LANE, ceiling + 1, _LANE):
        if block_q % d == 0:
            if (
                estimate_vmem_bytes(
                    block_q,
                    d,
                    block_kv,
                    block_kv,
                    single_buffered_q=single_buffered_q,
                    head_dim=head_dim,
                )
                <= _UNTUNED_VMEM_BUDGET
            ):
                q_sub = d
            else:
                break
    return q_sub


_BKV_LADDER = (
    1024,
    1280,
    1536,
    1792,
    2048,
    2304,
    2560,
    2816,
    3072,
    3328,
    3584,
    3840,
    4096,
    4608,
    5120,
    5632,
    6144,
)


def _untuned_bkv_candidates(shard_len: int, r0_full: int) -> list[int]:
    max_bkv = 4096 if shard_len >= 7500 else 6144
    bkv_ladder = tuple(b for b in _BKV_LADDER if b <= max_bkv)
    bkv_aligned = {-(-(-(-shard_len // n)) // _LANE) * _LANE for n in range(1, 20)}
    bkv_cands = sorted(
        {b for b in bkv_ladder if b <= r0_full}
        | {b for b in bkv_aligned if 1024 <= b <= min(r0_full, max_bkv)}
    )
    return bkv_cands or [min(r0_full, _DEFAULT_BLOCK_KV)]


def _tile_score(
    shard_len: int, n_passes: int, bq: int, bkv: int, bkvc: int, qs: int
) -> float:
    n_kv = -(-shard_len // bkv)
    kv_pad_ratio = (n_kv * bkv) / shard_len
    q_pad_ratio = (n_passes * bq) / shard_len
    n_compute = n_kv * (bkv // bkvc)
    return (
        q_pad_ratio * kv_pad_ratio
        + 0.0060 * n_kv
        + 0.0025 * n_compute
        + 0.0022 * (bq // qs)
    )


def _untuned_tiles_for_q_block(
    shard_len: int, target_q: int, bq: int, bkv_cands: list[int], fixed_m: bool
):
    qs_cap = (3840 if target_q >= 7500 else 4096) if fixed_m else 3840
    max_qs = min(qs_cap, max(_LANE, (bq // 2) // _LANE * _LANE))
    min_qs = min(1536, max(_LANE, (bq // 8) // _LANE * _LANE))
    for qs in range(_LANE, max_qs + 1, _LANE):
        if bq % qs or qs < min_qs:
            continue
        for bkv in bkv_cands:
            bkvc_list = [bkv]
            # Only split bkv when one KV block covers the shard to stay under IMEM limits.
            if bkv >= shard_len and bkv >= 4096:
                for c in (2048, 2560, 3072):
                    if c < bkv and bkv % c == 0:
                        bkvc_list.append(c)
            for bkvc in bkvc_list:
                yield bkv, bkvc, qs


def _best_untuned_tile(
    shard_len: int,
    n_passes: int,
    bkv_cands: list[int],
    *,
    fixed_m: bool,
    head_dim: int,
) -> tuple[int, int, int, int] | None:
    vmem_budget = int(0.94 * _V7X_VMEM_CAP_BYTES)
    target_q = -(-shard_len // n_passes)
    r0 = -(-target_q // _LANE) * _LANE
    max_pad = 0.035 if target_q >= 4096 else 0.06
    r_max = max(r0, -(-int(target_q * (1.0 + max_pad)) // _LANE) * _LANE)

    best_score = float("inf")
    best_tuple: tuple[int, int, int, int] | None = None
    for bq in range(r0, r_max + 1, _LANE):
        single_buf = is_single_buffered_q(bq, shard_len)
        for bkv, bkvc, qs in _untuned_tiles_for_q_block(
            shard_len, target_q, bq, bkv_cands, fixed_m
        ):
            vmem = estimate_vmem_bytes(
                bq, qs, bkv, bkvc, single_buffered_q=single_buf, head_dim=head_dim
            )
            if vmem > vmem_budget:
                continue
            ragged = shard_len % bkv != 0
            if fixed_m and ragged and qs * bkvc > _HYBRID_RAGGED_SCORE_LIMIT:
                continue
            score = _tile_score(shard_len, n_passes, bq, bkv, bkvc, qs)
            if score < best_score:
                best_score = score
                best_tuple = (bq, bkv, bkvc, qs)
    return best_tuple


def select_untuned_blocks(
    shard_len: int,
    *,
    use_fixed_m: bool | None = None,
    head_dim: int = 128,
) -> tuple[int, int, int, int]:
    """Select (block_q, block_kv, block_kv_compute, block_q_sub) for an untuned shard."""
    fixed_m = use_fixed_m is not False
    r0_full = max(_LANE, -(-shard_len // _LANE) * _LANE)
    bkv_cands = _untuned_bkv_candidates(shard_len, r0_full)
    if shard_len <= 34_000:
        pass_counts = [1, max(2, -(-shard_len // 23040))]
    else:
        pass_counts = [max(2, -(-shard_len // 23040))]
    for n_passes in pass_counts:
        best = _best_untuned_tile(
            shard_len, n_passes, bkv_cands, fixed_m=fixed_m, head_dim=head_dim
        )
        if best is not None:
            return best

    bkv = min(r0_full, _DEFAULT_BLOCK_KV)
    qs = _untuned_q_sub(
        r0_full,
        bkv,
        head_dim,
        single_buffered_q=is_single_buffered_q(r0_full, shard_len),
    )
    return (r0_full, bkv, bkv, qs)


def _pad_seq(tensor, multiple: int, axis: int):
    import jax.numpy as jnp

    pad = -tensor.shape[axis] % multiple
    if not pad:
        return tensor
    widths = [(0, 0)] * tensor.ndim
    widths[axis] = (0, pad)
    return jnp.pad(tensor, widths)


def max_row_norm_per_head(x, *, seq_axis: int = 2):
    """Max L2 row norm per head over batch and sequence."""
    import jax.numpy as jnp

    return jnp.sqrt((x.astype(jnp.float32) ** 2).sum(axis=-1)).max(axis=(0, seq_axis))


def ring_fixed_m_operands(qn_max, mk_h):
    """Build the (2, heads) scalar-prefetch array for hybrid fixed-m ring attention.

    ``qn_max`` is the per-head max query row norm times the base-2 softmax scale,
    so ``qn_max * mk_h`` bounds the head's base-2 logits.
    """
    import jax.numpy as jnp

    bound = qn_max * mk_h
    # Keep row 0 constant so scalar prefetch stays off Mosaic's critical path.
    m = jnp.full_like(bound, float(math.floor(FIXED_M_RING_SAFE_BOUND)))
    return jnp.stack([m, (bound <= FIXED_M_RING_SAFE_BOUND).astype(jnp.float32)])


def fixed_m_update(qk, v, m, scale_log2):
    """Compute one KV sub-block's softmax sum and output at a pinned m."""
    import jax.numpy as jnp
    from jax import lax

    s_curr = jnp.exp2(qk * scale_log2 - m)
    s_cast = s_curr.astype(jnp.bfloat16)
    l_curr = s_curr.sum(axis=0, keepdims=True).astype(jnp.float32)
    o_curr = lax.dot_general(
        v, s_cast, SV_DIM_NUMBERS, preferred_element_type=jnp.float32
    )
    return l_curr, o_curr


def online_update(qk, v, m_prev, scale_log2):
    """Compute one KV sub-block's online-softmax terms."""
    import jax.numpy as jnp
    from jax import lax

    # scale_log2 > 0, so scaling the max equals the max of the scaled logits.
    m_curr = qk.max(axis=0)[None, :] * scale_log2
    m_next = jnp.maximum(m_prev, m_curr)
    s_curr = jnp.exp2(qk * scale_log2 - m_next[0:1])
    s_cast = s_curr.astype(jnp.bfloat16)
    l_curr = s_curr.sum(axis=0, keepdims=True)
    alpha = jnp.exp2(m_prev - m_next)
    o_curr = lax.dot_general(
        v, s_cast, SV_DIM_NUMBERS, preferred_element_type=jnp.float32
    )
    return m_next, alpha, l_curr, o_curr


class Blocks(NamedTuple):
    """Resolved tiling for one ring attention launch."""

    block_q: int
    block_kv: int
    block_kv_compute: int
    block_kv_compute_in: int
    block_q_sub: int | None

    @classmethod
    def for_shard(
        cls,
        shard_len: int,
        ring_size: int,
        pinned: Mapping[str, Any] | None = None,
        head_dim: int = 128,
        *,
        use_fixed_m: bool | None = None,
    ) -> "Blocks":
        pinned = {k: v for k, v in (pinned or {}).items() if v is not None}
        key = (ring_size, shard_len)
        tuned = (
            (_TUNED_ONLINE_BLOCKS.get(key) or _TUNED_BLOCKS.get(key))
            if use_fixed_m is False
            else _TUNED_BLOCKS.get(key)
        )
        if tuned is not None:
            block_q, block_kv, tuned_bkvc, block_q_sub = tuned
        elif not pinned.keys() & {
            "block_q",
            "block_kv",
            "block_kv_compute",
            "block_q_sub",
        }:
            block_q, block_kv, tuned_bkvc, block_q_sub = select_untuned_blocks(
                shard_len, use_fixed_m=use_fixed_m, head_dim=head_dim
            )
        else:
            block_q = -(-shard_len // _DEFAULT_BLOCK_Q) * _DEFAULT_BLOCK_Q
            block_kv = _DEFAULT_BLOCK_KV
            tuned_bkvc = block_kv
            block_q_sub = None
        block_q = int(pinned.get("block_q", block_q))
        block_kv = int(pinned.get("block_kv", block_kv))
        default_bkvc = tuned_bkvc if "block_kv" not in pinned else block_kv
        block_kv_compute = int(pinned.get("block_kv_compute", default_bkvc))
        resolved_q_sub = pinned.get("block_q_sub", block_q_sub)
        if "block_q_sub" not in pinned and (
            resolved_q_sub is None or block_q % resolved_q_sub != 0
        ):
            resolved_q_sub = _untuned_q_sub(
                block_q,
                block_kv_compute,
                head_dim,
                single_buffered_q=is_single_buffered_q(block_q, shard_len),
            )
        return cls(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=block_kv_compute,
            block_kv_compute_in=int(
                pinned.get("block_kv_compute_in", block_kv_compute)
            ),
            block_q_sub=resolved_q_sub,
        )

    def apply(
        self,
        q,
        k,
        v,
        *,
        ring_axis: str,
        ring_size: int,
        axis_names: tuple[str, ...],
        scale_log2: float,
        use_fixed_m: bool = False,
        mk=None,
        interpret: Any = False,
    ):
        shard_len = q.shape[2]
        q = _pad_seq(q, self.block_q, axis=2)
        q_sub = self.block_q_sub or auto_block_q_sub(self.block_q)
        if self.block_q % q_sub:
            raise ValueError(
                f"block_q ({self.block_q}) must be a multiple of block_q_sub ({q_sub})"
            )
        return ring_attention_forward(
            q,
            _pad_seq(k, self.block_kv, axis=2),
            _pad_seq(v, self.block_kv, axis=2),
            block_q=self.block_q,
            block_kv=self.block_kv,
            block_kv_compute=self.block_kv_compute,
            block_kv_compute_in=self.block_kv_compute_in,
            q_sub_block=q_sub,
            q_seq_len=shard_len,
            kv_seq_len=shard_len,
            ring_axis=ring_axis,
            ring_size=ring_size,
            axis_names=axis_names,
            scale_log2=scale_log2,
            use_fixed_m=use_fixed_m,
            mk=mk,
            interpret=interpret,
        )


def _neighbor(
    axis_names: tuple[str, ...], ring_axis: str, offset: int
) -> tuple[Any, ...]:
    from jax import lax

    ring_size = lax.axis_size(ring_axis)
    idx = lax.axis_index(ring_axis)
    nxt = lax.rem(idx + offset + ring_size, ring_size)
    return tuple(nxt if a == ring_axis else lax.axis_index(a) for a in axis_names)


class _KernelRefs(NamedTuple):
    mk: Any
    q: Any
    k_hbm: Any
    v_hbm: Any
    o: Any
    k_buf: Any
    v_buf: Any
    m_scratch: Any
    l_scratch: Any
    o_scratch: Any
    local_sem: Any
    send_sem: Any
    recv_sem: Any
    credit_sem: Any


class _KernelConfig(NamedTuple):
    mask_value: float
    grid_width: int
    ring_size: int
    bkv: int
    bkv_compute: int
    bkv_compute_in: int
    head_dim_v: int
    kv_seq_len: int
    q_heads_per_kv_head: int
    num_kv_heads: int
    use_fixed_m: bool
    q_sub: int
    axis_names: tuple[str, ...]
    ring_axis: str
    scale_log2: float


class _GridStep(NamedTuple):
    h: Any
    j: Any
    r: Any
    hk: Any
    q_idx: Any
    num_h: Any
    num_i: Any
    slot: Any
    nslot: Any
    is_first_hop: Any
    is_last_hop: Any
    is_first_step: Any
    is_last_step: Any
    upstream: tuple[Any, ...]
    downstream: tuple[Any, ...]


def _grid_step(cfg: _KernelConfig) -> _GridStep:
    from jax import lax
    from jax.experimental import pallas as pl

    b, h, i, j, r = (pl.program_id(n) for n in range(5))
    num_b, num_h, num_i = (pl.num_programs(n) for n in range(3))

    hk = b * cfg.num_kv_heads + h // cfg.q_heads_per_kv_head
    q_idx = (b * num_h + h) * num_i + i

    t = (q_idx * cfg.grid_width + j) * cfg.ring_size + r
    slot = lax.rem(t, 2)
    nslot = 1 - slot

    is_first_hop = (j == 0) & (r == 0)
    is_last_hop = (j == cfg.grid_width - 1) & (r == cfg.ring_size - 1)
    is_first_step = t == 0
    is_last_step = (q_idx == num_b * num_h * num_i - 1) & is_last_hop

    return _GridStep(
        h=h,
        j=j,
        r=r,
        hk=hk,
        q_idx=q_idx,
        num_h=num_h,
        num_i=num_i,
        slot=slot,
        nslot=nslot,
        is_first_hop=is_first_hop,
        is_last_hop=is_last_hop,
        is_first_step=is_first_step,
        is_last_step=is_last_step,
        upstream=_neighbor(cfg.axis_names, cfg.ring_axis, -1),
        downstream=_neighbor(cfg.axis_names, cfg.ring_axis, +1),
    )


def _local_load(refs: _KernelRefs, cfg: _KernelConfig, dst_slot, kv_head, block):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    src = (pl.ds(kv_head, 1), pl.ds(block * cfg.bkv, cfg.bkv))
    dst = pl.ds(dst_slot, 1)
    return (
        pltpu.make_async_copy(
            refs.k_hbm.at[src], refs.k_buf.at[dst], refs.local_sem.at[0]
        ),
        pltpu.make_async_copy(
            refs.v_hbm.at[src], refs.v_buf.at[dst], refs.local_sem.at[1]
        ),
    )


def _remote_push(refs: _KernelRefs, step: _GridStep, src_slot, dst_slot):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    src, dst = pl.ds(src_slot, 1), pl.ds(dst_slot, 1)
    return (
        pltpu.make_async_remote_copy(
            refs.k_buf.at[src],
            refs.k_buf.at[dst],
            refs.send_sem.at[0],
            refs.recv_sem.at[0],
            device_id=step.downstream,
        ),
        pltpu.make_async_remote_copy(
            refs.v_buf.at[src],
            refs.v_buf.at[dst],
            refs.send_sem.at[1],
            refs.recv_sem.at[1],
            device_id=step.downstream,
        ),
    )


def _kv_dma_prologue(refs: _KernelRefs, cfg: _KernelConfig, step: _GridStep) -> None:
    # Hop r > 0 computes on the block upstream pushed during hop r - 1; the last hop of a
    # KV block prefetches the next local block from HBM instead (1-credit double buffer).
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    r, slot, nslot = step.r, step.slot, step.nslot

    @pl.when(step.is_first_step)
    def _barrier():
        sem = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(sem, 1, device_id=step.upstream)
        pl.semaphore_signal(sem, 1, device_id=step.downstream)
        pl.semaphore_wait(sem, 2)

    @pl.when(step.is_first_step)
    def _prime():
        for dma in _local_load(refs, cfg, 0, step.hk, 0):
            dma.start()
        for dma in _local_load(refs, cfg, 0, step.hk, 0):
            dma.wait()

    @pl.when(jnp.logical_not(step.is_first_step))
    def _await_current():
        @pl.when(r == 0)
        def _await_local():
            for dma in _local_load(refs, cfg, slot, step.hk, step.j):
                dma.wait()

        @pl.when(r > 0)
        def _await_remote():
            for dma in _remote_push(refs, step, nslot, slot):
                dma.wait_recv()

    @pl.when(r > 0)
    def _retire_send():
        for dma in _remote_push(refs, step, nslot, slot):
            dma.wait_send()

    # Release nslot in hop t's prologue so Mosaic cannot hoist the signal before hop t-1's reads.
    @pl.when((r < cfg.ring_size - 1) & (cfg.ring_size > 1))
    def _release():
        pl.semaphore_signal(refs.credit_sem, 1, device_id=step.upstream)

    @pl.when(jnp.logical_not(step.is_last_step))
    def _prefetch_next():
        @pl.when(r < cfg.ring_size - 1)
        def _push():
            pl.semaphore_wait(refs.credit_sem, 1)
            for dma in _remote_push(refs, step, slot, nslot):
                dma.start()

        @pl.when(r == cfg.ring_size - 1)
        def _reload_own():
            at_q_end = step.j == cfg.grid_width - 1
            next_q_idx = step.q_idx + 1
            next_bh = lax.div(next_q_idx, step.num_i)
            next_b = lax.div(next_bh, step.num_h)
            next_h = lax.rem(next_bh, step.num_h)
            next_hk = next_b * cfg.num_kv_heads + lax.div(
                next_h, cfg.q_heads_per_kv_head
            )
            load_hk = jnp.where(at_q_end, next_hk, step.hk)
            load_j = jnp.where(at_q_end, 0, step.j + 1)
            for dma in _local_load(refs, cfg, nslot, load_hk, load_j):
                dma.start()


def _m_fixed(refs: _KernelRefs, step: _GridStep):
    return refs.mk[0, step.h] - FIXED_M_RECENTER


def _for_each_q_chunk(refs: _KernelRefs, cfg: _KernelConfig, fn) -> None:
    from jax import lax
    from jax.experimental import pallas as pl

    n_q = refs.q.shape[0] // cfg.q_sub
    if n_q > 1:

        def _qbody(c, carry):
            fn(pl.multiple_of(c * cfg.q_sub, _LANE))
            return carry

        lax.fori_loop(0, n_q, _qbody, None)
    else:
        for qs in range(0, refs.q.shape[0], cfg.q_sub):
            fn(qs)


def _two_body_last(cfg: _KernelConfig) -> bool:
    # Fixed-m heads get their own last-KV-block body only with a single KV block. Otherwise
    # they resume in the online body from m pinned at the fixed value: one compiled body less
    # keeps the program inside tpu7x's ~64K-bundle instruction memory, past which it runs far slower.
    return cfg.grid_width == 1


def _init_accumulators(
    refs: _KernelRefs, cfg: _KernelConfig, step: _GridStep, is_fixed
) -> None:
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    q_sub = cfg.q_sub

    def _zero_out_and_sum(qsl):
        refs.o_scratch[:, qsl] = jnp.zeros(
            (refs.o_scratch.shape[0], q_sub), refs.o_scratch.dtype
        )
        refs.l_scratch[:, qsl] = jnp.zeros(
            (refs.l_scratch.shape[0], q_sub), refs.l_scratch.dtype
        )

    def _init_fixed_chunk(qs):
        qsl = pl.ds(qs, q_sub)
        _zero_out_and_sum(qsl)
        if not _two_body_last(cfg):
            refs.m_scratch[:, qsl] = jnp.full(
                (refs.m_scratch.shape[0], q_sub),
                _m_fixed(refs, step),
                refs.m_scratch.dtype,
            )

    def _init_online_chunk(qs):
        qsl = pl.ds(qs, q_sub)
        _zero_out_and_sum(qsl)
        refs.m_scratch[:, qsl] = jnp.full(
            (refs.m_scratch.shape[0], q_sub), cfg.mask_value, refs.m_scratch.dtype
        )

    @pl.when(step.is_first_hop)
    def _init():
        if cfg.use_fixed_m:

            @pl.when(is_fixed)
            def _init_fixed():
                _for_each_q_chunk(refs, cfg, _init_fixed_chunk)

            @pl.when(jnp.logical_not(is_fixed))
            def _init_online():
                _for_each_q_chunk(refs, cfg, _init_online_chunk)

        else:
            _for_each_q_chunk(refs, cfg, _init_online_chunk)


def _online_inner(cfg: _KernelConfig, qk, v_chunk, m_prev, l_prev, o_prev):
    chunk = cfg.bkv_compute_in
    for c in range(0, qk.shape[0], chunk):
        m_next, alpha, l_curr, o_curr = online_update(
            qk[c : c + chunk], v_chunk[c : c + chunk], m_prev, cfg.scale_log2
        )
        o_prev = alpha[0:1, ...] * o_prev + o_curr
        m_prev, l_prev = m_next, l_curr + alpha * l_prev
    return m_prev, l_prev, o_prev


def _fixed_inner(cfg: _KernelConfig, qk, v_chunk, m_fix, l_prev, o_prev):
    chunk = cfg.bkv_compute_in
    for c in range(0, qk.shape[0], chunk):
        l_curr, o_curr = fixed_m_update(
            qk[c : c + chunk], v_chunk[c : c + chunk], m_fix, cfg.scale_log2
        )
        l_prev, o_prev = l_prev + l_curr, o_prev + o_curr
    return l_prev, o_prev


def _accumulate_kv_rows(
    refs: _KernelRefs, cfg: _KernelConfig, step: _GridStep, offset, length, fixed
) -> None:
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl

    sl = pl.ds(offset, length)
    k_blk = refs.k_buf[step.slot, sl, :]
    v_chunk = refs.v_buf[step.slot, sl, :]

    def _one_q_chunk(qs):
        qsl = pl.ds(qs, cfg.q_sub)
        qk = lax.dot_general(
            k_blk, refs.q[qsl, :], NT_DIM_NUMBERS, preferred_element_type=jnp.float32
        )
        if fixed:
            l_prev, o_prev = _fixed_inner(
                cfg,
                qk,
                v_chunk,
                _m_fixed(refs, step),
                refs.l_scratch[:, qsl],
                refs.o_scratch[:, qsl],
            )
            refs.l_scratch[:, qsl] = l_prev
        else:
            m_prev, l_prev, o_prev = _online_inner(
                cfg,
                qk,
                v_chunk,
                refs.m_scratch[:, qsl],
                refs.l_scratch[:, qsl],
                refs.o_scratch[:, qsl],
            )
            refs.m_scratch[:, qsl] = m_prev
            refs.l_scratch[:, qsl] = l_prev
        refs.o_scratch[:, qsl] = o_prev

    _for_each_q_chunk(refs, cfg, _one_q_chunk)


def _run_full(refs: _KernelRefs, cfg: _KernelConfig, step: _GridStep, fixed) -> None:
    from jax import lax

    def body(kv_compute_index, _):
        _accumulate_kv_rows(
            refs, cfg, step, kv_compute_index * cfg.bkv_compute, cfg.bkv_compute, fixed
        )

    lax.fori_loop(0, cfg.bkv // cfg.bkv_compute, body, None)


def _run_tail(refs: _KernelRefs, cfg: _KernelConfig, step: _GridStep, fixed) -> None:
    # Only the real rows of the last KV block; the rest is padding.
    from jax import lax

    def body(kv_compute_index, _):
        _accumulate_kv_rows(
            refs, cfg, step, kv_compute_index * cfg.bkv_compute, cfg.bkv_compute, fixed
        )

    if cfg.kv_seq_len % cfg.bkv == 0:
        lax.fori_loop(0, cfg.bkv // cfg.bkv_compute, body, None)
        return
    remain = cfg.kv_seq_len % cfg.bkv
    iter_num = (remain + cfg.bkv_compute - 1) // cfg.bkv_compute
    if remain % cfg.bkv_compute == 0:
        lax.fori_loop(0, iter_num, body, None)
    else:
        if iter_num > 1:
            lax.fori_loop(0, iter_num - 1, body, None)
        _accumulate_kv_rows(
            refs,
            cfg,
            step,
            (iter_num - 1) * cfg.bkv_compute,
            remain % cfg.bkv_compute,
            fixed,
        )


def _accumulate_hop(
    refs: _KernelRefs, cfg: _KernelConfig, step: _GridStep, is_fixed
) -> None:
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    j, last_j = step.j, cfg.grid_width - 1
    if not cfg.use_fixed_m:
        if cfg.grid_width > 1:

            @pl.when(j != last_j)
            def _body():
                _run_full(refs, cfg, step, False)

        @pl.when(j == last_j)
        def _last_body():
            _run_tail(refs, cfg, step, False)

        return

    if cfg.grid_width > 1:

        @pl.when((j != last_j) & is_fixed)
        def _body_fixed():
            _run_full(refs, cfg, step, True)

        @pl.when((j != last_j) & jnp.logical_not(is_fixed))
        def _body_online():
            _run_full(refs, cfg, step, False)

    if _two_body_last(cfg):

        @pl.when((j == last_j) & is_fixed)
        def _last_body_fixed():
            _run_tail(refs, cfg, step, True)

        @pl.when((j == last_j) & jnp.logical_not(is_fixed))
        def _last_body_online():
            _run_tail(refs, cfg, step, False)

    else:

        @pl.when(j == last_j)
        def _last_body():
            _run_tail(refs, cfg, step, False)


def _write_output(refs: _KernelRefs, cfg: _KernelConfig, step: _GridStep) -> None:
    import jax.numpy as jnp
    from jax.experimental import pallas as pl

    head_dim_v_repeats = cfg.head_dim_v // NUM_SUBLANES

    @pl.when(step.is_last_hop)
    def _write_out():
        def _write_out_chunk(qs):
            qsl = pl.ds(qs, cfg.q_sub)
            l = refs.l_scratch[:, qsl]
            l_inv = jnp.tile(1.0 / l, (head_dim_v_repeats, 1))
            refs.o[:, qsl] = (refs.o_scratch[:, qsl] * l_inv).astype(refs.o.dtype)

        _for_each_q_chunk(refs, cfg, _write_out_chunk)


def _ring_attention_kernel(*refs: Any, **static: Any) -> None:
    refs = _KernelRefs(*refs)
    cfg = _KernelConfig(**static)
    if cfg.head_dim_v % NUM_SUBLANES != 0:
        raise NotImplementedError(
            f"head_dim_v={cfg.head_dim_v} should be a multiple of {NUM_SUBLANES}"
        )
    step = _grid_step(cfg)
    _kv_dma_prologue(refs, cfg, step)
    is_fixed = refs.mk[1, step.h] > 0.5 if cfg.use_fixed_m else None
    _init_accumulators(refs, cfg, step, is_fixed)
    _accumulate_hop(refs, cfg, step, is_fixed)
    _write_output(refs, cfg, step)


def _ring_grid_spec(
    *,
    grid: tuple[int, ...],
    bq: int,
    bkv: int,
    head_dim_qk: int,
    head_dim_v: int,
    kv_dtype: tuple[Any, Any],
    single_buffered_q: bool,
):
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    resident = {"pipeline_mode": pl.Buffered(1)} if single_buffered_q else {}

    def q_index_map(b, h, i, j, r, *_):
        del j, r
        return (b, h, i, 0)

    def out_index_map(b, h, i, j, r, *_):
        del j, r
        return (b, h, 0, i)

    in_specs = [
        pl.BlockSpec((None, None, bq, head_dim_qk), q_index_map, **resident),
        pl.BlockSpec(memory_space=pl.ANY),
        pl.BlockSpec(memory_space=pl.ANY),
    ]
    out_specs = pl.BlockSpec((None, None, head_dim_v, bq), out_index_map, **resident)
    scratch_shapes = [
        pltpu.VMEM((2, bkv, head_dim_qk), kv_dtype[0]),
        pltpu.VMEM((2, bkv, head_dim_v), kv_dtype[1]),
        pltpu.VMEM((NUM_SUBLANES, bq), jnp.float32),
        pltpu.VMEM((NUM_SUBLANES, bq), jnp.float32),
        pltpu.VMEM((head_dim_v, bq), jnp.float32),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.REGULAR,
    ]
    return pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
        scratch_shapes=scratch_shapes,
    )


def ring_attention_forward(
    q,
    k,
    v,
    *,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
    block_kv_compute_in: int,
    q_sub_block: int,
    q_seq_len: int,
    kv_seq_len: int,
    ring_axis: str,
    ring_size: int,
    axis_names: tuple[str, ...],
    scale_log2: float,
    use_fixed_m: bool = False,
    mk=None,
    vmem_limit_bytes: int | None = VMEM_LIMIT_BYTES,
    mask_value: float = DEFAULT_MASK_VALUE,
    interpret: Any = False,
):
    """Launch single-call Pallas ring attention over [B, H, S, D] shards.

    ``q`` is unscaled; ``scale_log2`` (softmax scale times log2(e)) multiplies
    the float32 logits inside the kernel.
    """
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    batch, num_q_heads, _, head_dim_qk = q.shape
    num_kv_heads = k.shape[1]
    kv_pad = k.shape[2]
    head_dim_v = v.shape[-1]
    q_heads_per_kv_head = num_q_heads // num_kv_heads

    bq, bkv = block_q, block_kv
    q_sub = q_sub_block
    if bq % q_sub != 0 or q_sub % _LANE != 0:
        raise ValueError(
            f"q_sub_block ({q_sub}) must be a multiple of {_LANE} dividing block_q ({bq})"
        )
    if bkv % block_kv_compute != 0:
        raise ValueError(
            f"block_kv_compute ({block_kv_compute}) must divide block_kv ({bkv})"
        )
    if not use_fixed_m:
        mk = jnp.zeros((2, 1), jnp.float32)
    elif mk is None:
        raise ValueError("use_fixed_m requires mk")

    grid_width = (kv_seq_len + bkv - 1) // bkv
    grid_height = (q_seq_len + bq - 1) // bq
    grid_spec = _ring_grid_spec(
        grid=(batch, num_q_heads, grid_height, grid_width, ring_size),
        bq=bq,
        bkv=bkv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        kv_dtype=(k.dtype, v.dtype),
        single_buffered_q=is_single_buffered_q(bq, q_seq_len),
    )
    out_shape = jax.ShapeDtypeStruct(
        (batch, num_q_heads, head_dim_v, q_seq_len), q.dtype
    )

    out = pl.pallas_call(
        functools.partial(
            _ring_attention_kernel,
            mask_value=mask_value,
            grid_width=grid_width,
            ring_size=ring_size,
            bkv=bkv,
            bkv_compute=block_kv_compute,
            bkv_compute_in=block_kv_compute_in,
            head_dim_v=head_dim_v,
            kv_seq_len=kv_seq_len,
            q_heads_per_kv_head=q_heads_per_kv_head,
            num_kv_heads=num_kv_heads,
            use_fixed_m=use_fixed_m,
            q_sub=q_sub,
            axis_names=tuple(axis_names),
            ring_axis=ring_axis,
            scale_log2=scale_log2,
        ),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",) * 5,
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
            disable_bounds_checks=True,
            vmem_limit_bytes=vmem_limit_bytes,
            collective_id=_COLLECTIVE_ID,
            has_side_effects=False,
        ),
        out_shape=out_shape,
        interpret=interpret,
        name="diffusion_ring_attention",
    )(
        mk,
        q,
        k.reshape(batch * num_kv_heads, kv_pad, head_dim_qk),
        v.reshape(batch * num_kv_heads, kv_pad, head_dim_v),
    )
    return jnp.swapaxes(out, 2, 3)


def ring_attention_sharded(
    q,
    k,
    v,
    *,
    ring_axis: str,
    axis_names: tuple[str, ...],
    scale: float,
    use_fixed_m: bool = True,
    pinned: Mapping[str, Any] | None = None,
    interpret: Any = False,
):
    """Run shard_map body on [1, ..., B, S_shard, H, D] local shards."""
    import jax
    import jax.numpy as jnp

    lead = len(axis_names)
    lead_axes = tuple(range(lead))
    q_s = jnp.squeeze(q, axis=lead_axes)
    k_s = jnp.squeeze(k, axis=lead_axes)
    v_s = jnp.squeeze(v, axis=lead_axes)

    ring_size = jax.lax.axis_size(ring_axis)
    shard_len = q_s.shape[1]
    head_dim = q_s.shape[-1]
    if head_dim != 128:
        # The tuned tiles and the VMEM model are measured at head_dim 128.
        raise ValueError(f"ring attention supports head_dim 128, got {head_dim}")
    # Scaling the bf16 query before QK^T would round it again; the kernel
    # scales the float32 logits instead.
    scale_log2 = scale * LOG2_E

    mk = None
    if use_fixed_m:
        # Prevent XLA from duplicating Q/K producers into the norm reductions.
        q_s, k_s = jax.lax.optimization_barrier((q_s, k_s))
        qn_max = max_row_norm_per_head(q_s, seq_axis=1) * scale_log2
        mk_h = max_row_norm_per_head(k_s, seq_axis=1)
        if q_s.shape[2] != k_s.shape[2]:
            mk_h = jnp.repeat(mk_h, q_s.shape[2] // k_s.shape[2], axis=0)
        qn_max, mk_h = jax.lax.pmax((qn_max, mk_h), ring_axis)
        mk = ring_fixed_m_operands(qn_max, mk_h)

    blocks = Blocks.for_shard(
        shard_len,
        ring_size,
        pinned,
        head_dim=head_dim,
        use_fixed_m=use_fixed_m,
    )
    out = blocks.apply(
        jnp.swapaxes(q_s, 1, 2),
        jnp.swapaxes(k_s, 1, 2),
        jnp.swapaxes(v_s, 1, 2),
        ring_axis=ring_axis,
        ring_size=ring_size,
        axis_names=axis_names,
        scale_log2=scale_log2,
        use_fixed_m=use_fixed_m,
        mk=mk,
        interpret=interpret,
    )
    return jnp.expand_dims(jnp.swapaxes(out, 1, 2), axis=lead_axes)
