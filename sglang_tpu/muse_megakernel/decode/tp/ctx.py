# SPDX-License-Identifier: Apache-2.0
"""Trace-time state of one decode megakernel body, shared by its split-out sections
(attn, moe, tiled_*): kernel() binds each field once, each section unpacks what it reads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

from sglang_tpu.muse_megakernel.common.coll import Coll
from sglang_tpu.muse_megakernel.common.config import MKConfig
from sglang_tpu.muse_megakernel.common.hooks import EPHook
from sglang_tpu.muse_megakernel.common.stream import _WeightStream

# A Pallas kernel ref (HBM, VMEM, SMEM or semaphore), or a view of one.
Ref = Any
Fn = Callable[..., Any]


class KernelCtx:
    # ---- config and static geometry (build_kernel)
    c: MKConfig
    ep: EPHook | None
    R: int
    K: int
    H: int
    L: int
    He: int
    Hel: int
    Hl: int
    Il: int
    hd: int
    hq: int
    hkv: int
    g: int
    ph: int
    pd: int
    ps: int
    q: int
    ncap: int
    roll_lo: int | None
    roll_hi: int | None
    kind_layers: dict[str, list[int]]
    # ---- static per-kernel values (kernel)
    F: int
    NBR: int
    NT: int
    RT: int
    Rc: int
    yw: int
    bf16: Any
    f32: Any
    o_rs_streamed: bool
    use_pair_o_rs: bool
    use_pair_y: bool
    group_lo: dict[int, int]
    layer_ex: dict[int, tuple[list[Ref], int | None]]
    tc_rmap: dict
    tc_smap: dict
    # ---- refs and traced scalars
    S: dict[str, Ref]
    W: dict[str, Ref]
    KV: tuple[Ref, ...]
    emb_ref: Ref
    pf_ref: Ref
    pos_ref: Ref
    hid_out: Ref
    aux_out: Ref
    newkv_out: Ref
    n_live: jax.Array
    tp_rank: jax.Array
    me: jax.Array | int
    # untiled steps only
    cos_x: jax.Array | None = None
    sin_x: jax.Array | None = None
    row_live: jax.Array | None = None
    # ---- collaborators
    coll: Coll
    ws: _WeightStream
    # ---- kernel() helpers
    allgather_sum: Fn
    allreduce: Fn
    blk_owner: Fn
    capture_tile: Fn
    chunk_dst: Fn
    divmod_nn: Fn
    dma: Fn
    f_blocks: Fn
    flags: Fn
    for_tiles: Fn
    gathered_tile: Fn
    kv_prologue: Fn
    mv_tiled: Fn
    o_blocks: Fn
    pad_view: Fn
    page_src: Fn
    per_geom: Fn
    pipe_u: Fn
    prefetch_smalls: Fn
    rcopy: Fn
    req_geom: Fn
    rget: Fn
    ring_words: Fn
    rows_at: Fn
    rows_of: Fn
    single_row: Fn
    smalls_copy: Fn
    stream_pair_y: Fn
    stream_view: Fn
    tile_live: Fn
    tile_pass: Fn
    tiled_fold_lhs: Fn
    tsem: Fn
    u_finish: Fn
    u_start: Fn
    unique_experts_vec: Fn
    vpads_tiles: Fn
    wait_chunk: Fn
    y_block: Fn
    ag_start: Fn
    ag_tile_ready: Fn
    ag_tile_sent: Fn
    ag_tile_start: Fn
    hag_forward: Fn
    hag_ready: Fn
    hag_sent: Fn
    hag_start: Fn
    # ---- sections (partials of the split-out modules)
    request_attention_body: Fn
    unique_experts: Fn
    route_tiled: Fn
    attention_tiled: Fn
    expert_io_tiled: Fn
    moe_tiled: Fn
    layer_tiled: Fn
