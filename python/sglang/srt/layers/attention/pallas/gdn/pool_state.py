# SPDX-License-Identifier: Apache-2.0
"""Describe a pooled Gated-DeltaNet state buffer to the fused kernel.

The fused context-parallel entry point does not take dense conv and recurrent
tensors. It reads and writes both through an indexed byte-addressed pool and
needs a static copy-plan saying where each region sits inside one state
index's window: the recurrent region is whole kernel blocks at the start of
the window, the conv region is a few token rows in the block after it.

This module builds that plan. It is reachable only from the fused kernel; the
dense path leaves ``GDNConfig.state_plan`` as None.
"""

import dataclasses
import math

import jax.numpy as jnp

from sglang.srt.layers.attention.pallas.gdn import config


@dataclasses.dataclass(frozen=True)
class PooledGDNStateLayout:
    ssm_bytes: int
    conv_bytes: int
    token_bytes: int
    ssm_tokens: int
    conv_tokens: int

    @property
    def required_tokens(self) -> int:
        return self.ssm_tokens + self.conv_tokens

    @property
    def required_bytes(self) -> int:
        return self.required_tokens * self.token_bytes


def derive_pooled_gdn_state_layout(
    *,
    ssm_bytes: int,
    conv_bytes: int,
    token_bytes: int,
) -> PooledGDNStateLayout:
    """Derive the physical token extent of pooled GDN state regions."""
    if min(ssm_bytes, conv_bytes, token_bytes) <= 0:
        raise ValueError("pooled GDN byte sizes must be positive")
    if ssm_bytes % token_bytes:
        raise ValueError("pooled GDN SSM state must occupy complete FA token rows")

    ssm_tokens = ssm_bytes // token_bytes
    conv_tokens = 1
    while conv_tokens * token_bytes < conv_bytes or ssm_tokens % conv_tokens != 0:
        conv_tokens *= 2
        if conv_tokens > ssm_tokens:
            raise ValueError(
                "pooled GDN conv tile cannot be placed after the SSM region"
            )

    return PooledGDNStateLayout(
        ssm_bytes=ssm_bytes,
        conv_bytes=conv_bytes,
        token_bytes=token_bytes,
        ssm_tokens=ssm_tokens,
        conv_tokens=conv_tokens,
    )


def _conv_qk_pair_rows_perm(
    taps: int, conv_dim: int, n_v: int, d_v: int, lanes: int
) -> tuple[int, ...] | None:
    """Typed-row permutation for the QK pair-blocked pooled conv layout.

    Stored order per tap interleaves Q and K row-pairs —
    ``[Q0, K0, Q1, K1, ..., V...]`` — so that one whole pool token holds
    one head-pair's Q and K rows and a full-width state can receive a
    TP-rank shard's first conv token as a single contiguous token copy.
    Returns the load-order permutation (``logical[i] = stored[perm[i]]``),
    or ``None`` when the stored order equals the logical order (head-shard
    states whose Q segment is a single typed row).
    """
    v_elems = n_v * d_v
    qk_elems = conv_dim - v_elems
    q_elems = qk_elems // 2
    if 2 * q_elems != qk_elems or q_elems % lanes:
        raise NotImplementedError(
            "QK pair-blocked conv layout requires lane-aligned equal Q/K "
            f"segments: conv_dim={conv_dim}, n_v*d_v={v_elems}, "
            f"lanes={lanes}"
        )
    if v_elems % lanes:
        raise NotImplementedError(
            f"QK pair-blocked conv layout requires lane-aligned V: "
            f"{v_elems} % {lanes}"
        )
    q_rows = q_elems // lanes
    v_rows = v_elems // lanes
    rows_per_tap = conv_dim // lanes
    if q_rows <= 1:
        return None
    perm: list[int] = []
    for tap in range(taps):
        base = tap * rows_per_tap
        perm.extend(base + 2 * i for i in range(q_rows))
        perm.extend(base + 2 * i + 1 for i in range(q_rows))
        perm.extend(base + 2 * q_rows + j for j in range(v_rows))
    return tuple(perm)


def _pool_geometry(pool):
    """(block_size, payload dims between token and lane, lanes)."""
    assert pool.ndim >= 3, pool.shape
    return pool.shape[1], tuple(pool.shape[2:-1]), pool.shape[-1]


def v3_state_source(
    pool,
    *,
    split: int,
    ssm_ntok: int,
    conv_tok0: int,
    conv_ntok: int,
    conv_dim: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    qk_pair_layout: bool = False,
) -> config.StateSourcePlan:
    """Static copy-plan letting the fused GDN kernel stream the mamba
    state regions directly between this pool and its double-buffered
    pipeline, without an external round trip through dense tensors.
    for these regions, without the external round trip.

    The ssm region is the whole kernel blocks at the start of the manager
    window (one contiguous DMA per slot), viewed f32 with the ``d_v``
    lane split; the conv region is ``conv_ntok`` token rows inside a
    single kernel block, viewed bf16 with the tail rows zero-padded.
    Arguments mirror the region geometry computed by the pooled GDN
    caller; ``pool`` contributes only its shape and dtype.

    The plan describes one checkpoint. With speculative decoding the caller
    names one source block per checkpoint via
    ``MetadataRef.s_idx_to_ckpt_indices``, and every checkpoint reuses this
    same intra-block geometry, so the block size stays independent of
    ``num_speculative_tokens``.
    """
    block_size, payload, lanes = _pool_geometry(pool)
    tok_bytes = math.prod(payload) * lanes * jnp.dtype(pool.dtype).itemsize

    assert lanes % d_v == 0, (lanes, d_v)
    ssm_rows = n_v * d_k
    # The caller sizes the ssm region to exactly the f32 state bytes, so
    # the kernel's write covers the whole region (no padding rows).
    assert ssm_ntok * tok_bytes == ssm_rows * d_v * 4, (
        ssm_ntok,
        tok_bytes,
        n_v,
        d_k,
        d_v,
    )
    if ssm_ntok % block_size == 0:
        ssm_nblocks, ssm_nrows = ssm_ntok // block_size, block_size
    else:
        assert ssm_ntok < block_size, (ssm_ntok, block_size)
        ssm_nblocks, ssm_nrows = 1, ssm_ntok
    ssm = config.StateRegion(
        kb0=0,
        nblocks=ssm_nblocks,
        row0=0,
        nrows=ssm_nrows,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=lanes // d_v,
        rows_used=ssm_rows,
    )

    kb0, row0 = divmod(conv_tok0, block_size)
    if row0 + conv_ntok > block_size:
        raise NotImplementedError(
            "conv region partially straddles kernel blocks: "
            f"tok0={conv_tok0} ntok={conv_ntok} kernel_block={block_size}"
        )
    assert conv_tok0 + conv_ntok <= split * block_size, (
        conv_tok0,
        conv_ntok,
        split,
        block_size,
    )
    assert (kernel_size - 1) * conv_dim % lanes == 0, (kernel_size, conv_dim, lanes)
    # The kernel regroups conv rows into (kernel_size - 1, conv_dim) via
    # 128-aligned lane concat, which needs whole rows per conv row.
    assert conv_dim % lanes == 0, (conv_dim, lanes)
    conv_rows = (kernel_size - 1) * conv_dim // lanes
    assert conv_rows * 2 * lanes <= conv_ntok * tok_bytes, (
        conv_rows,
        conv_ntok,
        tok_bytes,
    )
    conv_rows_perm = None
    if qk_pair_layout:
        conv_rows_perm = _conv_qk_pair_rows_perm(
            kernel_size - 1, conv_dim, n_v, d_v, lanes
        )
    conv = config.StateRegion(
        kb0=kb0,
        nblocks=1,
        row0=row0,
        nrows=conv_ntok,
        view_dtype=jnp.dtype(jnp.bfloat16),
        lane_split=1,
        rows_used=conv_rows,
        rows_perm=conv_rows_perm,
    )

    return config.StateSourcePlan(stride=split, conv=conv, recurrent=ssm)


def build_pool_state_plan(
    pool: jnp.ndarray,
    *,
    conv_dim: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pool_block_tokens: int,
    qk_pair_layout: bool,
):
    """Describe this rank's GDN state regions in the unified pool."""
    if pool_block_tokens % pool.shape[1] != 0:
        raise ValueError(
            "Manager block size must be divisible by the unified "
            f"pool kernel block size: {pool_block_tokens} and "
            f"{pool.shape[1]}."
        )
    split = pool_block_tokens // pool.shape[1]
    per_tok_elems = math.prod(pool.shape[2:])
    tok_bytes = per_tok_elems * jnp.dtype(pool.dtype).itemsize
    state_layout = derive_pooled_gdn_state_layout(
        ssm_bytes=n_v * d_k * d_v * 4,
        conv_bytes=(kernel_size - 1) * conv_dim * 2,
        token_bytes=tok_bytes,
    )
    if state_layout.required_tokens > pool_block_tokens:
        raise ValueError(
            "GDN state does not fit in the unified pool manager "
            f"block: {state_layout.required_tokens} > "
            f"{pool_block_tokens}."
        )
    return v3_state_source(
        pool,
        split=split,
        ssm_ntok=state_layout.ssm_tokens,
        conv_tok0=state_layout.ssm_tokens,
        conv_ntok=state_layout.conv_tokens,
        conv_dim=conv_dim,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        qk_pair_layout=qk_pair_layout,
    )
