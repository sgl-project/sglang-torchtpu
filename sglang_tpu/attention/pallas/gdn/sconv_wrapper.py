# SPDX-License-Identifier: Apache-2.0
"""Causal short convolution (sconv) on the GDN kernel's tile pipeline.

Per channel c and token t, with width W and the previous W - 1 inputs per
request kept in a dense state pool ([slots, W - 1, (1,) dim], oldest first)::

    static:  y_t = (x_t + sum_k w[c, k] * x_{t-W+1+k}) * (W + 1) ** -0.5
    dynamic: y_t = (x_t + sum_lag (s[lag, c] + d[t, c // hd, lag]) * x_{t-lag})
                   * (2W + 1) ** -0.5

Positions before a request's start read zero unless it has an initial state.
Reuses the GDN skeleton (metadata, buffered refs, state selection, conv1d) with
`GDNConfig.conv_only`: no recurrent state, gates or activation.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.gdn import (
    compute_conv1d,
    config,
    memory_ref,
    metadata,
    vmem_ldst,
)
from sglang_tpu.attention.pallas.util import get_tpu_version


def inner_kernel(*refs, cfg: config.GDNConfig):
    """One tile: conv over [carried state, x], state write-back, residual, scale."""
    if cfg.dynamic:
        x_ref, dyn_ref, state_in_ref, out_ref, state_out_ref, *scratch = refs
    else:
        x_ref, state_in_ref, out_ref, state_out_ref, *scratch = refs
        dyn_ref = None
    metadata_ref, weight_ref, carry_ref = scratch
    p_id = pl.program_id(0)

    real_sizes, prev_conv, _ = vmem_ldst.load_and_select_states(
        metadata_ref=metadata_ref,
        p_id=p_id,
        conv_state_in_slot_ref=state_in_ref,
        recurrent_in_slot_ref=None,
        carry_conv_scratch_ref=carry_ref,
        carry_recurrent_scratch_ref=None,
        cfg=cfg,
    )
    x = x_ref[...].astype(jnp.float32)  # [seq, chunk, 1, dim]
    out, new_state = compute_conv1d.causal_conv1d(
        real_sizes=real_sizes,
        lhs=jnp.concat([prev_conv, x], axis=1),
        conv_weight=weight_ref[...],
        conv_bias=None,
        cfg=cfg,
        dyn=None if dyn_ref is None else dyn_ref[...].astype(jnp.float32),
    )
    state_out_ref[...] = new_state.astype(state_out_ref.dtype)
    if carry_ref is not None:
        carry_ref[...] = new_state[:, -1]
    if cfg.use_residual:
        out = out + x
    out_ref[...] = (out * cfg.output_scale).astype(out_ref.dtype)


def outer_kernel(
    metadata_ref: memory_ref.MetadataRef,
    x_ref: jax.Array,
    dyn_ref: jax.Array | None,
    conv_state_ref: jax.Array,
    _: jax.Array,
    weight_ref: jax.Array,
    out_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: None,
    *,
    cfg: config.GDNConfig,
    metadata_layout: tuple | None,
):
    del carry_recurrent_scratch_ref
    if metadata_layout is not None:
        metadata_ref = memory_ref.unpack_metadata(metadata_ref, metadata_layout)
    x_alloc, out_alloc, state_in_alloc, state_out_alloc = memory_ref.create_conv_allocs(
        metadata_ref=metadata_ref,
        qkv_ref=x_ref,
        out_ref=out_ref,
        conv_state_ref=conv_state_ref,
        cfg=cfg,
    )
    in_allocs = [x_alloc]
    in_refs = [x_ref]
    if cfg.dynamic:
        in_allocs.append(
            memory_ref.create_input_alloc(
                metadata_ref,
                dyn_ref,
                (cfg.seq_tile_size, cfg.chunk_size, 1, cfg.kernel_size * cfg.dim_size),
                cfg,
            )
        )
        in_refs.append(dyn_ref)
    in_allocs.append(state_in_alloc)
    in_refs.append(conv_state_ref)
    allocs = (*in_allocs, out_alloc, state_out_alloc)

    pipeline_func = pltpu.emit_pipeline(
        body=functools.partial(inner_kernel, cfg=cfg),
        grid=(metadata_ref.num_tiles[...],),
        in_specs=tuple(a.spec for a in in_allocs),
        out_specs=(out_alloc.spec, state_out_alloc.spec),
    )

    @pl.with_scoped(allocations=allocs)
    def _run(allocations):
        pipeline_func(
            *in_refs,
            out_ref,
            conv_state_out_ref,
            scratches=(metadata_ref, weight_ref, carry_conv_scratch_ref),
            allocations=allocations,
        )

    _run()


def _kernel_weight(weight: jax.Array, dynamic: bool, act_dtype) -> jax.Array:
    """[kernel_size, 1, dim] f32, kernel tap k weighing x_{t - (K-1-k)}."""
    if dynamic:
        # Lag-major [K, dim], lag 0 = current token; used at its stored precision.
        return weight[::-1].astype(jnp.float32)[:, None, :]
    # [dim, (1,) K], oldest tap first; rounded to the activation dtype first.
    weight = weight.reshape(weight.shape[0], -1).astype(act_dtype)
    return weight.astype(jnp.float32).T[:, None, :]


def _kernel_dyn(dyn: jax.Array, dim: int) -> jax.Array:
    """[T, n_heads, K] lag-major -> [T, 1, K * dim] f32 in kernel tap order."""
    num_tokens, n_heads, kernel_size = dyn.shape
    taps = jnp.swapaxes(dyn[..., ::-1].astype(jnp.float32), 1, 2)
    taps = jnp.broadcast_to(
        taps[..., None], (num_tokens, kernel_size, n_heads, dim // n_heads)
    )
    return taps.reshape(num_tokens, 1, kernel_size * dim)


def _decode_tile_size(
    *, requested: int, num_seqs: int, dim: int, kernel_size: int, window: int
) -> int:
    # Budget mirrors fused_conv1d_gdn: double-buffered tiles in half the limit.
    bytes_per_seq = (1 + window) * (kernel_size - 1) * dim * 4 + window * dim * (
        4 + 4 + kernel_size * 4
    )
    budget = int(
        config.GDNConfig.DEFAULT_VMEM_FRACTION
        * pltpu.get_tpu_info().vmem_capacity_bytes
    )
    tile_budget = (budget // 2) // config.GDNConfig.__dataclass_fields__[
        "num_buffers"
    ].default
    return max(1, min(requested, num_seqs, tile_budget // bytes_per_seq))


@jax.jit(
    donate_argnames=("conv_state",),
    static_argnames=(
        "use_residual",
        "num_spec_tokens",
        "batched_only",
        "prefill_only",
        "decode_tile_size",
        "mixed_tile_size",
    ),
)
def fused_sconv(
    x: jax.Array,  # [T, dim]
    conv_state: jax.Array,  # [slots, K - 1, (1,) dim], slot 0 is the null slot
    weight: jax.Array,  # static [dim, (1,) K]; dynamic [K, dim] lag-major
    query_start_loc: jax.Array,  # [num_seqs + 1]
    state_indices: jax.Array,  # [num_seqs]
    distribution: jax.Array,  # [3]: decode_end, prefill_end, mixed_end
    seq_lens: jax.Array,  # [num_seqs], including this call's tokens
    dyn: jax.Array | None = None,  # [T, n_heads, K] lag-major
    read_offsets: jax.Array | None = None,  # [num_seqs]
    *,
    use_residual: bool = True,
    num_spec_tokens: int = 0,
    batched_only: bool = False,
    prefill_only: bool = False,
    decode_tile_size: int = 16,
    mixed_tile_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Returns (new conv_state, y [T, dim] in x.dtype); conv_state is donated.

    Sequences `[0, distribution[0])` run batched (one token, or a verify
    window of `num_spec_tokens + 1` writing one state per position); the rest
    up to `distribution[2]` run one sequence per tile. Rows past them are 0.
    """
    if batched_only and prefill_only:
        raise ValueError("batched_only and prefill_only are mutually exclusive")
    dynamic = dyn is not None
    batch_size, dim = x.shape
    kernel_size = weight.shape[0] if dynamic else weight.shape[-1]
    num_seqs = state_indices.size
    assert query_start_loc.shape == (num_seqs + 1,)
    assert distribution.shape == (3,)
    if num_spec_tokens > 0:
        assert read_offsets is not None, "num_spec_tokens > 0 needs read_offsets"
    if read_offsets is None:
        read_offsets = jnp.zeros((num_seqs,), jnp.int32)
    if dynamic:
        assert weight.shape == (kernel_size, dim), weight.shape
        assert dyn.shape[0] == batch_size and dyn.shape[2] == kernel_size
        assert dim % dyn.shape[1] == 0
    weights = _kernel_weight(weight, dynamic, x.dtype)

    state_shape = conv_state.shape
    conv_state = conv_state.reshape(-1, kernel_size - 1, 1, dim)
    x = x.reshape(batch_size, 1, dim)
    dyn = None if dyn is None else _kernel_dyn(dyn, dim)
    scale = (kernel_size * (2 if dynamic else 1) + int(use_residual)) ** -0.5
    window = num_spec_tokens + 1
    decode_tile_size = _decode_tile_size(
        requested=decode_tile_size,
        num_seqs=num_seqs,
        dim=dim,
        kernel_size=kernel_size,
        window=window,
    )

    def call_kernel(state, in_act, mode):
        cfg = config.GDNConfig(
            mode=mode,
            dtypes=config.Dtypes(
                act_in=x.dtype,
                act_out=x.dtype,
                compute=jnp.float32.dtype,
                recurrent_state=jnp.float32.dtype,
                conv_state=state.dtype,
            ),
            batch_size=batch_size,
            dim_size=dim,
            kernel_size=kernel_size,
            tile_size=(
                min(mixed_tile_size, batch_size)
                if mode == config.GDNMode.PER_SEQ
                else decode_tile_size
            ),
            num_kq_heads=0,
            num_v_heads=1,
            kq_head_dim=0,
            v_head_dim=dim,
            window_size=1 if mode == config.GDNMode.PER_SEQ else window,
            conv_only=True,
            use_residual=use_residual,
            output_scale=scale,
            dynamic=dynamic,
        )
        return _call_kernel(
            cfg=cfg,
            x=x,
            dyn=dyn,
            state=state,
            in_act=in_act,
            weights=weights,
            metadata_obj=_metadata(
                cfg=cfg,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                state_indices=state_indices,
                read_offsets=read_offsets.astype(jnp.int32),
                distribution=distribution,
            ),
        )

    out, state = None, conv_state
    if not prefill_only:
        out, state = call_kernel(state, None, config.GDNMode.BATCHED)
    if not batched_only:
        out, state = call_kernel(state, out, config.GDNMode.PER_SEQ)
    # Rows past the last sequence (bucket padding) are never written.
    num_valid = query_start_loc[distribution[-1]]
    out = jnp.where(jnp.arange(batch_size)[:, None, None] < num_valid, out, 0)
    return state.reshape(state_shape), out.reshape(batch_size, dim)


def _metadata(
    *,
    cfg: config.GDNConfig,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    read_offsets: jax.Array,
    distribution: jax.Array,
) -> memory_ref.MetadataRef:
    if cfg.mode == config.GDNMode.PER_SEQ:
        return metadata.compute_per_seq_metadata(
            cfg=cfg,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            read_offsets=read_offsets,
            start_seq=distribution[0],
            end_seq=distribution[-1],
        )
    return metadata.compute_batched_seq_metadata(
        cfg=cfg,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        read_offsets=read_offsets,
        end_seq=distribution[0],
    )


def _call_kernel(
    *,
    cfg: config.GDNConfig,
    x: jax.Array,
    dyn: jax.Array | None,
    state: jax.Array,
    in_act: jax.Array | None,
    weights: jax.Array,
    metadata_obj: memory_ref.MetadataRef,
) -> tuple[jax.Array, jax.Array]:
    metadata_obj, metadata_layout = memory_ref.pack_metadata(metadata_obj)
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    # The kernel never reads in_act; it only names the buffer the output aliases.
    # Without one, the output is a fresh buffer and fused_sconv zeroes the
    # uncovered rows: aliasing a zeros constant into the output let XLA reuse
    # that buffer while y was still live inside compiled graphs.
    fresh_out = in_act is None
    if fresh_out:
        in_act = jnp.zeros((1,), cfg.dtypes.act_out)
    out_struct = cfg.get_out_shape() if fresh_out else in_act
    # Flat inputs: packed metadata, x, [dyn], state, in_act, weights.
    state_input = 2 + int(dyn is not None)
    # The kernel DMAs these as HBM; unpinned, XLA may place small ones in VMEM
    # and the HBM DMAs then hit unrelated buffers.
    pin = get_tpu_version() >= 7
    if pin:
        x, state, in_act = (
            pltpu.with_memory_space_constraint(a, pltpu.HBM) for a in (x, state, in_act)
        )
        if dyn is not None:
            dyn = pltpu.with_memory_space_constraint(dyn, pltpu.HBM)
    out_shape = tuple(
        pltpu.HBM(shape=a.shape, dtype=a.dtype) if pin else a
        for a in (out_struct, state)
    )
    return pl.pallas_call(
        functools.partial(outer_kernel, cfg=cfg, metadata_layout=metadata_layout),
        out_shape=out_shape,
        in_specs=(
            smem_spec,
            hbm_spec,
            None if dyn is None else hbm_spec,
            hbm_spec,
            hbm_spec,
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        out_specs=(hbm_spec, hbm_spec),
        scratch_shapes=cfg.get_scratch_shape_dict(),
        input_output_aliases=(
            {state_input: 1} if fresh_out else {state_input: 1, state_input + 1: 0}
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=cfg.get_vmem_limit_bytes(),
        ),
        name=cfg.get_kernel_name(),
        metadata=cfg.get_metadata(),
    )(metadata_obj, x, dyn, state, in_act, weights)
