# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 Ulysses exchanges pipelined over head groups inside the block program.

Runtime patch, tracked debt:
- Why: on TPU the H3 attention core runs exchange -> attention -> exchange as
  one dependency chain inside each regionally compiled DiT block, so both
  all-to-alls (q|k|v in, attention output back) sit fully exposed next to
  the attention kernel. The GPU head-group pipeline
  (``ulysses_head_group_pipelined_attention``) uses streams and async
  process-group handles, which do not trace into a torch_tpu program.
- How: :func:`install` rebinds ``_minimax_h3_attention_core_impl``. Eligible
  calls (Ulysses only, the TPU live-length kernel, no host bounds) split the
  local heads into groups and write the exchange as independent per-group
  collectives, so XLA's latency-hiding scheduler can run group g + LEAD's
  q|k|v exchange while group g attends. XLA:TPU keeps one all-to-all in
  flight at a time, so group g's output exchange rides in the same
  collective as group g + LEAD's input (one buffer, destination-major on dim
  0, then parts: attention output, then q, k, v per head, each part one
  contiguous [rows, 128] run), and the live-length kernel reads q|k|v from
  the received buffer in place. With FUSED_PACK the forward norms and rotates
  q/k straight into those buffers (minimax_h3_ulysses_pack). When the
  kernel's blocks do not tile each rank's rows (at Ulysses 8, S % 16384 != 0
  with 2048-row blocks) or it pads the sequence to whole blocks, a block
  straddles two source ranks' runs, so the group's
  q|k|v are first copied out into ``[S, heads_g * d]`` rows (one cheap copy
  per part) and the kernel runs exactly as in the full call. Heads are
  independent and the kernel runs each head with the full call's block sizes,
  so the result is bit-identical; only the schedule and data movement change.
  Async all-to-alls need ``--xla_tpu_enable_async_all_to_all`` (see
  :data:`LIBTPU_FLAGS`, set by the worker before libtpu loads, only for an H3
  pipeline with Ulysses > 1 and no ring); without them the layers keep the
  serial path and the first layer built logs a warning.
- Upstream PR or issue: none.
- Remove when: torch_tpu schedules the Ulysses exchange asynchronously by
  itself, or the H3 core grows a traceable head-group pipeline.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# One local head per group (H3 at Ulysses 8: 7 heads per rank): the first
# input exchange and the last output exchange stay exposed, each one head's
# slice.
HEADS_PER_GROUP = 1
# Input exchanges issued before the first attention; group g's output then
# rides with group g + LEAD's input.
LEAD = 2

# Norm q/k and pack q|k|v into the exchange buffers in one kernel
# (minimax_h3_ulysses_pack) instead of the QK-norm + RoPE kernel
# (minimax_h3_qk_norm_rope) plus XLA's strided head gathers.
FUSED_PACK = True

# libtpu flags the pipeline needs: without them every all-to-all is
# synchronous and the per-group split only adds launches.
LIBTPU_FLAGS = (
    "--xla_tpu_enable_async_all_to_all=true",
    "--xla_tpu_force_async_all_to_all=true",
)

_INSTALLED = False
# Whether LIBTPU_INIT_ARGS enables LIBTPU_FLAGS (read once at install, after
# the worker set them and libtpu loaded). Without async all-to-alls the
# per-group exchanges run one after another next to the attention calls, so
# the pipeline would only add collectives: the layers keep the original path.
_ASYNC_ALL_TO_ALL = False
_warned_no_async = False


def async_all_to_all_enabled() -> bool:
    """Whether LIBTPU_INIT_ARGS sets every flag of :data:`LIBTPU_FLAGS` true
    (the last occurrence of a flag wins, as libtpu parses them)."""
    import os

    values = {}
    for arg in os.environ.get("LIBTPU_INIT_ARGS", "").split():
        name, _, value = arg.partition("=")
        values[name] = value.lower()
    return all(
        values.get(flag.partition("=")[0]) in ("", "true", "1")
        and flag.partition("=")[0] in values
        for flag in LIBTPU_FLAGS
    )


def _group_sizes(heads: int, per_group: int) -> list[int]:
    return [min(per_group, heads - start) for start in range(0, heads, per_group)]


def _exchange(x: torch.Tensor) -> torch.Tensor:
    """All-to-all over the Ulysses group, dim 0 split evenly (one chunk per
    rank). A functional collective: the in-place form leaves a copy_ into a
    fresh buffer that AOT autograd rejects once an opaque op reads the buffer
    directly."""
    import torch.distributed._functional_collectives as ft_c

    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_group
    from sglang.multimodal_gen.runtime.layers.usp import _maybe_wait

    group = get_sp_group().ulysses_group
    return _maybe_wait(ft_c.all_to_all_single(x.contiguous(), None, None, group))


def _blocks_divide(rows: int, world: int) -> bool:
    """Whether the live-length kernel's blocks tile each rank's rows, so it
    can read q|k|v from the part-major exchange buffer in place (the buffer
    holds each source rank's rows as a separate run)."""
    from sglang_tpu.multimodal_gen import tpu_splash_attn

    return tpu_splash_attn._live_length_parts_fit(world, rows)


def _copy_path_pad(seq: int) -> int:
    """Zero rows the copy path appends so the kernel runs whole 2048-row
    blocks. At 720P (S = 33280, a 512-key last block) the one-head calls of the
    copy path took 2.08 ms per head on tpu7x next to the in-flight exchanges,
    against 1.18 ms per head in the serial seven-head call; padded, the request
    ran 49.7 s instead of 62.2 s. Only when the padded call keeps the same
    blocks: the added keys are masked and the added query rows dropped, so
    every head's result is unchanged."""
    from sglang_tpu.multimodal_gen import tpu_splash_attn as tsa

    blocks, lazy_offset, rows = tsa._live_length_kernel(seq, seq)
    if not lazy_offset or rows != seq or seq <= tsa._LIVE_LENGTH_PAD_MIN_SEQ:
        return 0
    padded = tsa._round_up(seq, blocks[1])
    if padded == seq or tsa._live_length_kernel(padded, padded)[0] != blocks:
        return 0
    return padded - seq


def _eligible(attention, q, k, v, kwargs) -> bool:
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_ulysses_parallel_world_size,
    )
    from sglang_tpu.multimodal_gen import tpu_splash_attn

    impl = attention._attention_impl
    world_size = get_ulysses_parallel_world_size()
    return (
        _ASYNC_ALL_TO_ALL
        and kwargs.get("ulysses_active", False)
        and not kwargs.get("ring_active", False)
        and kwargs.get("gate_compress") is None
        and kwargs.get("subblock_sparse_query_block_mask") is None
        and kwargs.get("cu_seqlens_host") is None
        and isinstance(impl, tpu_splash_attn.TpuSplashAttentionImpl)
        and impl.trailing_padding_on_device
        and tpu_splash_attn._FLASH_LIVE_LENGTH_OP is not None
        and tpu_splash_attn._FLASH_LIVE_LENGTH_PARTS_OP is not None
        and not impl.causal
        and kwargs["cu_seqlens"].shape == (3,)
        and q.device.type == "tpu"
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.shape[-1] == 128
        and q.shape[0] * world_size >= 512
        and q.shape[0] % 16 == 0
        and world_size > 1
        and q.dim() == 3
        and q.shape == k.shape == v.shape
        and q.shape[1] % world_size == 0
        and q.shape[1] // world_size > 1
    )


def _exchange_attend(attention, cu_seqlens, *, world, rows, sizes, first, combine):
    """The head-group pipeline over prepared exchange buffers.

    Buffers are part-major, ``[rank, parts, rows, d]``: ``first(g)`` is group
    g's send buffer (parts q, k, v per head) for the first LEAD groups;
    ``combine(out_g, nxt)`` the send buffer carrying group g's attention
    output ``[rank, heads_g, rows, d]`` in its first parts and group nxt's
    q|k|v after them. Returns ``[rows, heads, d]``.
    """
    from sglang_tpu.multimodal_gen import tpu_splash_attn

    impl = attention._attention_impl
    live = cu_seqlens[1:2].to(torch.int32)
    groups = len(sizes)
    in_place = _blocks_divide(rows, world)
    pad = 0 if in_place else _copy_path_pad(world * rows)

    def attend(buffer, size, col):
        """Group attention over a received buffer ``[source rank, parts,
        rows, d]``; returns ``[source rank, size, rows, d]``."""
        if in_place:
            # source rank w's rows are the gathered sequence's rows w * rows ..
            return tpu_splash_attn._FLASH_LIVE_LENGTH_PARTS_OP(
                buffer, live, impl.softmax_scale, size, col
            )
        # A kernel block straddles source ranks: copy each part out into the
        # gathered sequence's [S, size * d] rows, as the full call reads them.
        d = buffer.shape[-1]
        q_g, k_g, v_g = (
            torch.nn.functional.pad(
                buffer[:, col + p : col + 3 * size : 3]
                .permute(0, 2, 1, 3)
                .reshape(world * rows, size * d),
                (0, 0, 0, pad),
            )
            for p in range(3)
        )
        out = tpu_splash_attn._FLASH_LIVE_LENGTH_OP(
            q_g, k_g, v_g, live, impl.softmax_scale, size
        )[: world * rows]
        return out.view(world, rows, size, d).permute(0, 2, 1, 3)

    # group -> (received buffer, column block where its q|k|v start)
    received = {}
    for g in range(min(LEAD, groups)):
        received[g] = (_exchange(first(g)), 0)
    outs = []
    for g in range(groups):
        size = sizes[g]
        buffer, col = received.pop(g)
        out_g = attend(buffer, size, col)
        nxt = g + LEAD
        if nxt < groups:
            both = _exchange(combine(out_g, nxt))
            outs.append(both[:, :size])
            received[nxt] = (both, size)
        else:
            outs.append(_exchange(out_g))
    # [source rank = head owner, local head, rows, d] -> [rows, heads, d]
    out = torch.cat(outs, dim=1)
    return out.permute(2, 0, 1, 3).reshape(rows, world * sum(sizes), out.shape[-1])


def _pipelined_core(attention, q, k, v, *, cu_seqlens):
    """q/k/v ``[rows, heads, d]`` row shards -> attention output, same layout."""
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_ulysses_parallel_world_size,
    )

    world = get_ulysses_parallel_world_size()
    rows, heads, d = q.shape
    h_local = heads // world
    sizes = _group_sizes(h_local, HEADS_PER_GROUP)
    starts = [sum(sizes[:g]) for g in range(len(sizes))]
    # [rows, rank, local head, d] views of the projections
    q4, k4, v4 = (t.unflatten(1, (world, h_local)) for t in (q, k, v))

    def pack(g):
        """Group g's q|k|v, destination-major: ``[rank, 3 * heads_g, rows, d]``."""
        return torch.stack(
            [
                t[:, :, h].transpose(0, 1)
                for h in range(starts[g], starts[g] + sizes[g])
                for t in (q4, k4, v4)
            ],
            dim=1,
        )

    return _exchange_attend(
        attention,
        cu_seqlens,
        world=world,
        rows=rows,
        sizes=sizes,
        first=pack,
        combine=lambda out_g, nxt: torch.cat([out_g, pack(nxt)], dim=1),
    )


def _pack_op_counts(h_local: int) -> tuple[int, ...]:
    """Group counts of the pack ops a rank with ``h_local`` heads calls:
    group 0 alone, then the others (see :func:`_fused_attend`)."""
    from sglang_tpu.multimodal_gen import minimax_h3_ulysses_pack as pack_mod

    sizes, _ = pack_mod.group_layout(h_local, HEADS_PER_GROUP, LEAD)
    return (1, len(sizes) - 1) if len(sizes) > 1 else (1,)


def _register_pack_ops(attention) -> None:
    """Register the pack ops this layer's forward will call, while the model
    is built: never inside a compiled region. A layer whose ops are missing
    (e.g. registration failed) takes the original forward."""
    global _warned_no_async
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_ulysses_parallel_world_size,
        model_parallel_is_initialized,
    )
    from sglang_tpu.multimodal_gen import minimax_h3_ulysses_pack as pack_mod

    if not model_parallel_is_initialized():
        return
    world = get_ulysses_parallel_world_size()
    heads = attention.num_heads
    if world <= 1 or heads % world or heads // world <= 1:
        return
    if not _ASYNC_ALL_TO_ALL:
        if not _warned_no_async:
            _warned_no_async = True
            logger.warning(
                "MiniMax-H3 Ulysses head-group pipeline is off: LIBTPU_INIT_ARGS "
                "does not enable %s (the worker appends them only when the "
                "server arguments name an H3 pipeline with Ulysses > 1 and no "
                "ring). Without async all-to-alls the pipeline only adds "
                "collectives, so attention keeps the serial exchange.",
                " ".join(LIBTPU_FLAGS),
            )
        return
    if not FUSED_PACK:
        return
    try:
        for count in _pack_op_counts(heads // world):
            pack_mod.ensure_op(count)
    except Exception as exc:  # the layer keeps the original forward
        logger.warning(
            "MiniMax-H3 Ulysses pack op registration failed (%s); the fused "
            "pack is off for %d heads at Ulysses %d.",
            exc,
            heads,
            world,
        )


def _fused_pack_eligible(attention, x, rope_cache, kwargs) -> bool:
    """Whether the forward can norm + rotate q/k straight into the exchange
    buffers: the attention core would take the pipeline, and the QK-norm +
    RoPE kernel (minimax_h3_qk_norm_rope) would serve the norm (same
    arithmetic)."""
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_ulysses_parallel_world_size,
    )
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang_tpu.multimodal_gen import (
        minimax_h3_qk_norm_rope,
        minimax_h3_ulysses_pack,
        tpu_splash_attn,
    )

    if not (
        FUSED_PACK
        and _ASYNC_ALL_TO_ALL
        and minimax_h3_qk_norm_rope._INSTALLED
        and rope_cache is not None
        and x.device.type == "tpu"
        and x.dim() == 2
        and kwargs.get("ulysses_active", False)
        and not kwargs.get("ring_active", False)
        and kwargs.get("cu_seqlens_host") is None
        and kwargs.get("subblock_sparse_query_block_mask") is None
        and not attention._use_fused_qknorm_rope
        and (
            attention.hybrid is None
            or attention._attention_backend_enum
            is not AttentionBackendEnum.HYBRID_WINDOW_ATTN_H3
        )
    ):
        return False
    impl = attention._attention_impl
    world = get_ulysses_parallel_world_size()
    rows = x.shape[0]
    cos_sin_cache = rope_cache[0]
    q_norm, k_norm = attention.q_norm, attention.k_norm
    heads = attention.num_heads
    return (
        isinstance(impl, tpu_splash_attn.TpuSplashAttentionImpl)
        and impl.trailing_padding_on_device
        and tpu_splash_attn._FLASH_LIVE_LENGTH_OP is not None
        and tpu_splash_attn._FLASH_LIVE_LENGTH_PARTS_OP is not None
        and not impl.causal
        and kwargs["cu_seqlens"].shape == (3,)
        and attention.head_dim == minimax_h3_ulysses_pack.HEAD_DIM
        and world > 1
        and heads % world == 0
        and heads // world > 1
        # registered while the model was built (_register_pack_ops)
        and all(
            minimax_h3_ulysses_pack.registered_op(count) is not None
            for count in _pack_op_counts(heads // world)
        )
        and rows * world >= 512
        and rows % 16 == 0
        # every pack call's blocks fit the kernel's VMEM budget
        and all(
            minimax_h3_ulysses_pack.block_rows_for(
                rows,
                minimax_h3_ulysses_pack.pack_row_blocks(
                    heads // world, HEADS_PER_GROUP, LEAD, first_group, count
                ),
            )
            is not None
            for first_group, count in zip((0, 1), _pack_op_counts(heads // world))
        )
        and cos_sin_cache.dim() == 2
        and cos_sin_cache.shape == (rows, minimax_h3_ulysses_pack.ROTARY_DIM)
        and q_norm.weight is not None
        and k_norm.weight is not None
        and q_norm.eps is not None
        and q_norm.eps == k_norm.eps
    )


def _tensors(result) -> list:
    return list(result) if isinstance(result, (list, tuple)) else [result]


def _fused_attend(attention, qkv, cos_sin_cache, cu_seqlens):
    """qkv ``[rows, 3 * heads * d]`` -> attention output ``[rows, heads, d]``
    with QK-norm + RoPE written into the exchange buffers by one kernel."""
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_ulysses_parallel_world_size,
    )
    from sglang_tpu.multimodal_gen import minimax_h3_ulysses_pack as pack_mod

    world = get_ulysses_parallel_world_size()
    rows = qkv.shape[0]
    h_local = attention.num_heads // world
    sizes, blocks = pack_mod.group_layout(h_local, HEADS_PER_GROUP, LEAD)
    args = (
        qkv.contiguous(),
        attention.q_norm.weight,
        attention.k_norm.weight,
        cos_sin_cache.contiguous(),
        float(attention.q_norm.eps),
        world,
        HEADS_PER_GROUP,
        LEAD,
    )
    # Group 0 alone first: its exchange, the one no attention can hide, then
    # runs next to the pass over the other groups.
    packs = [
        t
        for first_group, count in zip((0, 1), _pack_op_counts(h_local))
        for t in _tensors(pack_mod.registered_op(count)(*args, first_group))
    ]

    def combine(out_g, nxt):
        # the kernel left blocks[nxt] empty parts for out_g
        return packs[nxt].slice_scatter(out_g, dim=1, start=0, end=blocks[nxt])

    return _exchange_attend(
        attention,
        cu_seqlens,
        world=world,
        rows=rows,
        sizes=sizes,
        first=lambda g: packs[g],
        combine=combine,
    )


def _original_tail(self, x, qkv, *, rope_cache, cu_seqlens, max_seqlen, **kwargs):
    """``MiniMaxH3Attention.forward`` from the qkv projection on, for a
    projection the fused pack does not take (e.g. a quantized one). The hybrid
    window backend and the MPS path never reach here (:func:`_fused_pack_eligible`)."""
    from sglang.multimodal_gen.runtime.platforms import current_platform

    total = x.shape[0]
    q, k, v = (
        t.view(total, self.num_heads, self.head_dim)
        for t in qkv.split(self.local_inner_dim, dim=-1)
    )
    attend = (
        self._attend_eager
        if torch.compiler.is_compiling()
        and current_platform.runs_dit_attention_eager_under_compile()
        else self._attend
    )
    out = attend(
        x,
        q,
        k,
        v,
        rope_cache=rope_cache,
        cu_seqlens=cu_seqlens,
        cu_seqlens_host=kwargs.get("cu_seqlens_host"),
        max_seqlen=max_seqlen,
        subblock_sparse_query_block_mask=kwargs.get("subblock_sparse_query_block_mask"),
        ulysses_active=kwargs.get("ulysses_active", False),
        ring_active=kwargs.get("ring_active", False),
    )
    out, _ = self.out_proj(out)
    return out


def install() -> None:
    global _INSTALLED, _ASYNC_ALL_TO_ALL
    if _INSTALLED:
        return
    _ASYNC_ALL_TO_ALL = async_all_to_all_enabled()
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit

    original = dit._minimax_h3_attention_core_impl

    def attention_core(attention, q, k, v, **kwargs):
        dit._minimax_h3_bind_attention_backend(attention, q.dtype)
        if not _eligible(attention, q, k, v, kwargs):
            return original(attention, q, k, v, **kwargs)
        return _pipelined_core(
            attention,
            q,
            k,
            v,
            cu_seqlens=kwargs["cu_seqlens"],
        )

    dit._minimax_h3_attention_core_impl = attention_core

    original_init = dit.MiniMaxH3Attention.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _register_pack_ops(self)

    original_forward = dit.MiniMaxH3Attention.forward

    def forward(
        self, x, *, rope_cache, cu_seqlens, max_seqlen, x_prequant=None, **kwargs
    ):
        if x.dtype == torch.bfloat16:
            dit._minimax_h3_bind_attention_backend(self, x.dtype)
        if not (
            x.dtype == torch.bfloat16
            and _fused_pack_eligible(
                self, x, rope_cache, {"cu_seqlens": cu_seqlens, **kwargs}
            )
        ):
            return original_forward(
                self,
                x,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                x_prequant=x_prequant,
                **kwargs,
            )
        total = x.shape[0]
        qkv, _ = self.qkv_proj(x if x_prequant is None else x_prequant)
        if qkv.dtype != torch.bfloat16 or qkv.shape != (
            total,
            3 * self.num_heads * self.head_dim,
        ):
            # e.g. a quantized projection: the original forward from here on
            return _original_tail(
                self,
                x,
                qkv,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                **kwargs,
            )
        out = _fused_attend(self, qkv, rope_cache[0], cu_seqlens)
        out, _ = self.out_proj(out.reshape(total, self.num_heads * self.head_dim))
        return out

    dit.MiniMaxH3Attention.__init__ = init
    dit.MiniMaxH3Attention.forward = forward
    _INSTALLED = True
