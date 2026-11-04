# SPDX-License-Identifier: Apache-2.0
"""Two-level one-shot Pallas all-reduce for single-token TP decode tensors.

XLA's all-reduce of a [1, H] bf16 decode activation over a multi-host TP group
is latency bound: on a tpu7x 2x2x4 slice (TP32, 8 TensorCores per host) it took
21 us for H=7168. This kernel does it in two one-shot phases: every rank sends
its whole input to the other ranks of its host and sums what it receives, then
sends that host partial to the rank with the same local index on every other
host and sums again. Each phase sends one DMA per peer: the inputs are packed
as dense (rows, 128) tiles at 16-row aligned offsets of one VMEM buffer, so
several tensors (e.g. Kimi-K3's routed latent and shared-expert output) share
the exchange. The cross-host partial travels as bf16 (half the bytes of f32).
Measured on that slice: 13.0 us at H=7168, 15.3 us at H=10752.

Opt-in via SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE. It applies to the TP group
only, when TP spans the whole world (no DP / PP / extra ranks) so TP ranks are
host-major, for bf16 [1, H] tensors with H a multiple of 128; everything else
keeps the original all-reduce.
"""

import functools
import logging

import torch

from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

# Largest packed payload (elements) one call reduces.
MAX_ELEMS = 16 * 16384
# Rows of 128 lanes each tensor's block is aligned to in the packed buffer.
_ALIGN_ROWS = 16
# Barrier-semaphore namespace of this kernel. The fused GMM reduce-scatter uses
# collective_id=0; a distinct id keeps the two from sharing a barrier.
_COLLECTIVE_ID = 1
# Ops keyed by tensors per call: 1 (plain all-reduce) or 2 (Kimi-K3 latent +
# shared). A process has one TP world, registered once.
_OPS: dict = {}
# Plain bool, not dict truthiness: Dynamo traces the eligibility checks.
_READY = False
# Set (while Dynamo traces / Python runs the decode forward) when a tagged
# attention o_proj skipped its all-reduce so that the following attention-residual
# aggregation runs it fused (all_reduce_attn_res); consumed by that aggregation.
_AR1_PENDING = False


def _comm(world: int, local: int, split_barrier: bool, scratch):
    """The exchange of one call: ``signal()`` at kernel entry, ``wait()``
    before the first remote write, ``reduce()`` -> f32 sum of ``packed``
    over all ranks (identical on every rank): an in-host one-shot, then the
    host partial cross-host."""
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    packed, gather1, part, gather2, s1, r1, s2, r2 = scratch[:8]
    groups = world // local
    me = lax.axis_index("model")
    grp, loc = me // local, lax.rem(me, local)
    local_peers = [grp * local + lax.rem(loc + d, local) for d in range(1, local)]
    remote_peers = [lax.rem(grp + d, groups) * local + loc for d in range(1, groups)]
    mesh_id = pl.DeviceIdType.MESH
    # Split barrier: the in-host phase waits only for the host's peers and the
    # cross-host barrier latency overlaps it. One barrier semaphore counts both
    # kinds of signal, so they carry different weights: a remote peer signals
    # 1, a local peer `m` = len(remote_peers) + 1 (more than all remote
    # signals together). Waiting for len(local_peers) * m therefore needs every
    # local peer. What is left is the remote signals that have arrived plus `m`
    # per local peer that already entered the NEXT call (a local peer can
    # finish this call once it has this rank's in-host block); a remote peer
    # cannot enter the next call before this rank sent its host partial, and a
    # local peer cannot enter it before this rank's in-host block is sent.
    # So the remaining count modulo m is the number of remote peers in this
    # call: spin until it reaches len(remote_peers), then take exactly that.
    # Every op sharing the barrier semaphore must use the same barrier
    # (register()).
    split = split_barrier and local > 1 and groups > 1
    nr = len(remote_peers)
    m = nr + 1 if split else 1
    barrier = pltpu.get_barrier_semaphore()
    local_wait = (local - 1) * m

    def signal():
        # Every peer must have entered the kernel (its VMEM buffers exist)
        # before anyone writes into them.
        for p in local_peers:
            pltpu.semaphore_signal(barrier, m, device_id=(p,), device_id_type=mesh_id)
        for p in remote_peers:
            pltpu.semaphore_signal(barrier, 1, device_id=(p,), device_id_type=mesh_id)

    def wait():
        if split:
            pltpu.semaphore_wait(barrier, local_wait)
        else:
            pltpu.semaphore_wait(barrier, local_wait + nr)

    def exchange(src, gather, slot, peers, send_sem, recv_sem):
        copies = []
        for p in peers:
            c = pltpu.make_async_remote_copy(
                src,
                gather.at[slot],
                send_sem,
                recv_sem,
                device_id=(p,),
                device_id_type=mesh_id,
            )
            c.start()
            copies.append(c)
        gather[slot] = src[...]
        for c in copies:
            c.wait_send()
        for c in copies:
            c.wait_recv()

    def reduce():
        if local > 1:
            exchange(packed, gather1, loc, local_peers, s1, r1)
            part[...] = jnp.sum(gather1[...].astype(jnp.float32), axis=0).astype(
                part.dtype
            )
        else:
            part[...] = packed[...].astype(part.dtype)
        if groups > 1:
            if split:
                lax.while_loop(
                    lambda c: lax.rem(pltpu.semaphore_read(barrier), m) != nr,
                    lambda c: c + 1,
                    jnp.int32(0),
                )
                pltpu.semaphore_wait(barrier, nr)
            exchange(part, gather2, grp, remote_peers, s2, r2)
            return jnp.sum(gather2[...].astype(jnp.float32), axis=0)
        return part[...].astype(jnp.float32)

    return signal, wait, reduce


def _kernel(
    *refs,
    n: int,
    rows: tuple,
    offs: tuple,
    world: int,
    local: int,
    split_barrier: bool = False,
    norm_eps: float | None = None,
):
    import jax.numpy as jnp
    from jax import lax

    xs = refs[:n]
    norm_w = None
    if norm_eps is not None:
        norm_w = refs[n]
        refs = refs[:n] + refs[n + 1 :]
    outs = refs[n : 2 * n]
    scratch = refs[2 * n :]
    packed = scratch[0]
    signal, wait, reduce = _comm(world, local, split_barrier, scratch)
    signal()
    # The alignment gaps between tensors stay unwritten: the sums are
    # elementwise and each output reads only its own rows.
    for x, r, off in zip(xs, rows, offs):
        packed[off : off + r, :] = x[...].reshape(r, 128)
    wait()
    total = reduce()
    for i, (o, r, off) in enumerate(zip(outs, rows, offs)):
        y = total[off : off + r, :].astype(o.dtype)
        if i == 0 and norm_w is not None:
            # RMSNorm.forward_native of the reduced bf16 tensor (Kimi-K3's
            # routed latent norm) on the [1, H] row, the layout XLA reduces:
            # the f32 sum of squares then runs in XLA's order (bit-exact).
            x = y.reshape(o.shape).astype(jnp.float32)
            var = jnp.mean(x * x, axis=-1, keepdims=True)
            x = x * lax.rsqrt(var + norm_eps)
            w = norm_w[...].reshape(o.shape).astype(jnp.float32)
            o[...] = (x * w).astype(o.dtype)
            continue
        o[...] = y.reshape(o.shape)


def _scratch(total: int, local: int, groups: int, dtype):
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    return [
        pltpu.VMEM((total, 128), dtype),
        pltpu.VMEM((local, total, 128), dtype),
        pltpu.VMEM((total, 128), jnp.bfloat16),
        pltpu.VMEM((groups, total, 128), jnp.bfloat16),
    ] + [pltpu.SemaphoreType.DMA] * 4


def twolevel_all_reduce_local(
    *xs,
    world: int,
    local: int,
    split_barrier: bool = False,
    norm_weight=None,
    norm_eps: float | None = None,
):
    """Per-rank body; call inside ``shard_map`` over a 1-D mesh axis "model"
    whose index order is the TP rank order (host-major). Each x is [1, H].
    With ``norm_weight`` ([H0] of the first tensor) the first output is the
    RMSNorm (``norm_eps``) of the first tensor's sum."""
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    n = len(xs)
    groups = world // local
    # One token: each [1, H] row is exchanged as dense (H / 128, 128) tiles; a
    # (1, H) row measured 1-4 us slower to exchange. The kernel relayouts in
    # VMEM: as XLA reshapes around the call it added two ~0.8 us ops per tensor.
    rows = tuple(x.shape[0] * x.shape[1] // 128 for x in xs)
    offs, total = [], 0
    for r in rows:
        offs.append(total)
        total += -(-r // _ALIGN_ROWS) * _ALIGN_ROWS
    total = max(total, _ALIGN_ROWS)
    dtype = xs[0].dtype
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    outs = pl.pallas_call(
        functools.partial(
            _kernel,
            n=n,
            rows=rows,
            offs=tuple(offs),
            world=world,
            local=local,
            split_barrier=split_barrier,
            norm_eps=None if norm_weight is None else float(norm_eps),
        ),
        out_shape=tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in xs),
        in_specs=[vmem] * (n + (norm_weight is not None)),
        out_specs=tuple([vmem] * n),
        scratch_shapes=_scratch(total, local, groups, dtype),
        compiler_params=pltpu.CompilerParams(collective_id=_COLLECTIVE_ID),
        name=f"twolevel_allreduce{n}{'n' if norm_weight is not None else ''}_w{world}_l{local}",
    )(*xs, *(() if norm_weight is None else (norm_weight.reshape(rows[0], 128),)))
    return outs if n > 1 else outs[0]


def _ar_res_kernel(
    x_ref,
    a_ref,
    bank_ref,
    cw_ref,
    w_ref,
    out_ref,
    prefix_ref,
    *scratch,
    rows: int,
    world: int,
    local: int,
    split_barrier: bool,
    nvb: int,
    eps: float,
    out_eps: float,
):
    """All-reduce of the attention output x, then the Kimi-K3 attention-residual
    decode aggregation of the MLP side (kimi_attn_residual._decode_kernel with
    the pending addend a): prefix = a + sum(x); out = normed mixture."""
    from sglang_tpu.kimi_attn_residual import add_prefix, decode_math

    packed = scratch[0]
    signal, wait, reduce = _comm(world, local, split_barrier, scratch)
    signal()
    packed[:rows, :] = x_ref[...].reshape(rows, 128)
    wait()
    total = reduce()
    b = total[:rows, :].astype(x_ref.dtype).reshape(x_ref.shape)
    prefix = add_prefix(a_ref[...], b)
    prefix_ref[...] = prefix
    out_ref[...] = decode_math(
        prefix,
        bank_ref,
        cw_ref[...],
        w_ref[...],
        nvb=nvb,
        eps=eps,
        out_eps=out_eps,
        out_dtype=out_ref.dtype,
    )


def twolevel_all_reduce_attn_res_local(
    x,
    a,
    bank,
    cw,
    w,
    *,
    world: int,
    local: int,
    split_barrier: bool,
    nvb: int,
    eps: float,
    out_eps: float,
):
    """Per-rank body (inside shard_map): x, a bf16 [1, H] (x the TP-partial
    attention output, a the pending prefix), bank [1, NB, H], cw / w f32
    [1, H]. Returns (normed, prefix) like attention_residual_decode_jax(a,
    all_reduce(x), ...)."""
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    rows = x.shape[0] * x.shape[1] // 128
    total = max(-(-rows // _ALIGN_ROWS) * _ALIGN_ROWS, _ALIGN_ROWS)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    row = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return pl.pallas_call(
        functools.partial(
            _ar_res_kernel,
            rows=rows,
            world=world,
            local=local,
            split_barrier=split_barrier,
            nvb=nvb,
            eps=eps,
            out_eps=out_eps,
        ),
        out_shape=(row, row),
        in_specs=[vmem] * 5,
        out_specs=(vmem, vmem),
        scratch_shapes=_scratch(total, local, world // local, x.dtype),
        compiler_params=pltpu.CompilerParams(collective_id=_COLLECTIVE_ID),
        name=f"twolevel_allreduce1r_nvb{nvb}_w{world}_l{local}",
    )(x, a, bank, cw, w)


def _build_ar_res(world: int, local: int, split_barrier: bool):
    """Op (x, a, bank, cw [1, H] f32, w [1, H] f32, nvb, eps, out_eps) ->
    (normed, prefix)."""
    import jax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = get_pallas_mesh(("model",), (world,))
    spec = P("model")

    def _wrapped(
        x: jax.Array,
        a: jax.Array,
        bank: jax.Array,
        cw: jax.Array,
        w: jax.Array,
        nvb: int,
        eps: float,
        out_eps: float,
    ) -> tuple[jax.Array, jax.Array]:
        body = functools.partial(
            twolevel_all_reduce_attn_res_local,
            world=world,
            local=local,
            split_barrier=split_barrier,
            nvb=nvb,
            eps=eps,
            out_eps=out_eps,
        )
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * 5,
            out_specs=(spec, spec),
            check_vma=False,
        )(x, a, bank, cw, w)

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::twolevel_allreduce1r_w{world}_l{local}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(spec,) * 5 + (P(),) * 3,
        output_partition_specs=(spec, spec),
    )


def _build(world: int, local: int, n: int, split_barrier: bool = False, norm=False):
    """n tensors per call; ``norm``: n=2 with the first output RMSNorm'd
    (op args: latent, shared, norm weight [1, H0], eps)."""
    import jax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = get_pallas_mesh(("model",), (world,))
    spec = P("model")
    body = functools.partial(
        twolevel_all_reduce_local,
        world=world,
        local=local,
        split_barrier=split_barrier,
    )
    n_in = n
    if norm:
        n_in = 3

        def _wrapped(
            x: jax.Array, y: jax.Array, w: jax.Array, eps: float
        ) -> tuple[jax.Array, jax.Array]:
            return jax.shard_map(
                lambda a, b, c: body(a, b, norm_weight=c, norm_eps=eps),
                mesh=mesh,
                in_specs=(spec,) * 3,
                out_specs=(spec,) * 2,
                check_vma=False,
            )(x, y, w)

    else:
        sharded = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * n,
            out_specs=spec if n == 1 else (spec,) * n,
            check_vma=False,
        )
        if n == 1:

            def _wrapped(x: jax.Array) -> jax.Array:
                return sharded(x)

        else:

            def _wrapped(x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
                return sharded(x, y)

    return cpu_export_jax_op(
        f"sglang_tpu_pallas::twolevel_allreduce{n}{'n' if norm else ''}_w{world}_l{local}",
        _wrapped,
        mesh=mesh,
        # The static eps rides along with a placeholder spec (non-tensor
        # arguments pass through; the specs are zipped with the arguments).
        input_partition_specs=(spec,) * n_in + ((P(),) if norm else ()),
        output_partition_specs=(spec,) * n,
    )


# (hosts, ranks per host) the kernel has been measured on and beats XLA's
# all-reduce, bf16 [1, 7168]:
# - tpu7x 2x2x4, TP32 = 4 hosts x 8 ranks: 13.0 vs 21.0 us;
# - one tpu7x host, TP8 = 1 host x 8 ranks (host-local phase only): 10.2 vs
#   15.0 us.
# A one-shot phase with more peers approaches the flat one-shot that measured
# slower than XLA, so other shapes keep XLA's all-reduce until measured.
VALIDATED_TOPOLOGIES = frozenset({(4, 8), (1, 8)})


def _check_ranks_agree(enabled: bool) -> None:
    """Every rank must take the same path: if one host missed the switch, its
    ranks would stay on XLA's all-reduce while the others wait in the Pallas
    barrier, and the first single-token decode would hang. Fail at startup
    instead."""
    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import get_world_group

    world_group = get_world_group()
    if world_group.world_size <= 1:
        return
    seen = [None] * world_group.world_size
    dist.all_gather_object(seen, bool(enabled), group=world_group.cpu_group)
    if len(set(seen)) > 1:
        on = [rank for rank, flag in enumerate(seen) if flag]
        raise RuntimeError(
            "SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE must be set the same on every "
            f"rank; it is on for ranks {on} only."
        )
    # The barrier protocol (signal weights) must match too.
    split = bool(tpu_envs.SGLANG_TPU_TWOLEVEL_ALLREDUCE_SPLIT_BARRIER.get())
    dist.all_gather_object(seen, split, group=world_group.cpu_group)
    if len(set(seen)) > 1:
        raise RuntimeError(
            f"SGLANG_TPU_TWOLEVEL_ALLREDUCE_SPLIT_BARRIER differs between ranks: {seen}"
        )


def register(world: int, local: int) -> None:
    """Build the ops; must run before torch.compile traces the model. Called
    once, from prepare_model_parallel_groups."""
    global _READY
    # All ops share one barrier semaphore, so all use the same barrier (_comm).
    split = tpu_envs.SGLANG_TPU_TWOLEVEL_ALLREDUCE_SPLIT_BARRIER.get()
    _OPS.update({n: _build(world, local, n, split) for n in (1, 2)})
    if tpu_envs.SGLANG_TPU_TWOLEVEL_ALLREDUCE_FUSE_LATENT_NORM.get():
        _OPS["2n"] = _build(world, local, 2, split, norm=True)
    if tpu_envs.SGLANG_TPU_TWOLEVEL_ALLREDUCE_FUSE_ATTN_RES.get():
        _OPS["ar1r"] = _build_ar_res(world, local, split)
    _READY = True


def register_from_parallel_state() -> None:
    """Called from TpuPlatform.prepare_model_parallel_groups (every rank, after
    the groups exist, before the first collective)."""
    enabled = tpu_envs.SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE.get()
    _check_ranks_agree(enabled)
    if not enabled:
        return
    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import (
        get_tensor_model_parallel_world_size,
    )
    from sglang.srt.runtime_context import get_server_args

    world = get_tensor_model_parallel_world_size()
    if world <= 1 or world != dist.get_world_size():
        logger.info(
            "Two-level Pallas all-reduce off: TP size %d does not span the "
            "world of %d ranks.",
            world,
            dist.get_world_size(),
        )
        return
    server_args = get_server_args()
    nnodes = server_args.nnodes
    # TP spans the world, so TP ranks are host-major with world / nnodes per host.
    if world % nnodes or (nnodes, world // nnodes) not in VALIDATED_TOPOLOGIES:
        logger.warning(
            "Two-level Pallas all-reduce off: TP %d over %d hosts is not a "
            "measured topology (hosts, ranks per host in %s); XLA's all-reduce "
            "stays.",
            world,
            nnodes,
            sorted(VALIDATED_TOPOLOGIES),
        )
        return
    local = world // nnodes
    register(world, local)
    logger.info(
        "Two-level Pallas all-reduce on: TP %d = %d hosts x %d ranks. It takes "
        "single-token bf16 TP all-reduces (and Kimi-K3's merged MoE reduce); "
        "everything else keeps XLA's all-reduce.",
        world,
        world // local,
        local,
    )
    compile_bs = server_args.tpu_compile_bs
    if compile_bs and 1 not in compile_bs:
        logger.warning(
            "Two-level Pallas all-reduce is on but the TPU compile batch sizes "
            "%s have no batch size 1, so no compiled decode step takes it.",
            compile_bs,
        )


def eligible(t: torch.Tensor) -> bool:
    # One decode token only: at 4 rows XLA's all-reduce measured faster.
    return (
        t.dim() == 2
        and t.shape[0] == 1
        and t.dtype == torch.bfloat16
        and t.shape[1] % 128 == 0
        and t.numel() <= MAX_ELEMS
    )


def all_reduce_around(original, group, input_: torch.Tensor) -> torch.Tensor:
    """AROUND hook for ``GroupCoordinator.all_reduce``."""
    from sglang.srt.distributed.parallel_state import get_tp_group

    if not (_READY and group is get_tp_group() and eligible(input_)):
        return original(group, input_)
    return _OPS[1](input_)


def _fusable_norm(norm, latent: torch.Tensor) -> bool:
    """A default-config RMSNorm (forward_native semantics) over the latent."""
    w = getattr(norm, "weight", None)
    return (
        w is not None
        and w.dim() == 1
        and w.shape[0] == latent.shape[1]
        and w.dtype in (torch.float32, torch.bfloat16)
        and not getattr(norm, "cast_x_before_out_mul", True)
        and getattr(norm, "override_orig_dtype", 0) is None
        and getattr(norm, "variance_size_override", 0) is None
        and not getattr(norm, "fp32_residual", False)
    )


def merged_moe_all_reduce_around(original, latent, shared, norm=None):
    """AROUND hook for ``kimi_k3.merged_moe_all_reduce``: both TP-partial
    tensors in one packed exchange, with no concat / split copies. With
    SGLANG_TPU_TWOLEVEL_ALLREDUCE_FUSE_LATENT_NORM the latent's RMSNorm runs
    in the same kernel (saves the norm's two XLA fusions per layer)."""
    if not (
        _READY
        and eligible(latent)
        and eligible(shared)
        and latent.numel() + shared.numel() <= MAX_ELEMS
    ):
        if norm is None:
            return original(latent, shared)
        return original(latent, shared, norm)
    if norm is not None:
        if "2n" in _OPS and _fusable_norm(norm, latent):
            return _OPS["2n"](
                latent,
                shared,
                norm.weight.view(1, -1),
                float(norm.variance_epsilon),
            )
        latent, shared = _OPS[2](latent, shared)
        return norm(latent), shared
    return _OPS[2](latent, shared)


def _set_ar1_pending(value: bool) -> None:
    global _AR1_PENDING
    _AR1_PENDING = value


def take_ar1_pending() -> bool:
    """True once after a tagged o_proj deferred its all-reduce (see
    tag_attn_res_layers); the caller must then reduce it."""
    pending = _AR1_PENDING
    if pending:
        _set_ar1_pending(False)
    return pending


def attn_res_fusion_enabled() -> bool:
    return _READY and "ar1r" in _OPS


def tag_attn_res_layer(layer) -> bool:
    """Called at post-load for each Kimi-K3 decoder layer whose MLP-side
    aggregation takes the pending add right after the o_proj all-reduce:
    marks its o_proj so the reduce is deferred into that aggregation."""
    o_proj = getattr(layer.self_attn, "o_proj", None)
    if o_proj is None or not attn_res_fusion_enabled():
        return False
    if (
        getattr(layer, "_sp_moe", True)
        or getattr(layer, "all_reduce_fusion", True)
        or not getattr(layer, "use_attn_residuals", False)
        or getattr(layer, "is_block_write_layer", True)
        or not getattr(o_proj, "reduce_results", False)
        or getattr(o_proj, "tp_size", 1) <= 1
        or getattr(o_proj, "use_decode_attn_tp", False)
        or getattr(o_proj, "skip_bias_add", False)
        or getattr(o_proj, "bias", None) is not None
    ):
        return False
    o_proj._twolevel_defer_ar1 = True
    return True


def all_reduce_attn_res(partial, prefix, bank, nvb, cw, eps, norm_w, out_eps):
    """(normed, new_prefix) = attention-residual decode aggregation of
    prefix + all_reduce(partial)."""
    return _OPS["ar1r"](
        partial,
        prefix,
        bank,
        cw.view(1, -1),
        norm_w.view(1, -1),
        int(nvb),
        float(eps),
        float(out_eps),
    )


def _defer_ar1(layer, input_, skip_all_reduce, output_tensor) -> bool:
    if not (
        getattr(layer, "_twolevel_defer_ar1", False)
        and "ar1r" in _OPS
        and _READY
        and not skip_all_reduce
        and output_tensor is None
        and input_.dim() == 2
        and input_.shape[0] == 1
        and input_.dtype == torch.bfloat16
        and layer.output_size % 128 == 0
        and layer.output_size <= MAX_ELEMS
    ):
        return False
    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
    from sglang.srt.runtime_context import get_forward, get_parallel

    group = (
        get_parallel().attn_tp_group
        if layer.use_dp_attention_reduce
        else get_tp_group()
    )
    return (
        group is get_tp_group()
        and not get_forward().sp_active
        and not should_skip_mlp_all_reduce()
    )


def row_parallel_linear_forward_around(
    original,
    layer,
    input_,
    skip_all_reduce=False,
    forward_batch=None,
    output_tensor=None,
):
    """AROUND hook for ``RowParallelLinear.forward``: a Kimi-K3 o_proj tagged
    by tag_attn_res_layer returns its TP-partial output and leaves the reduce
    to the following attention-residual aggregation (all_reduce_attn_res);
    every other call runs the original."""
    if _defer_ar1(layer, input_, skip_all_reduce, output_tensor):
        out = original(
            layer,
            input_,
            skip_all_reduce=True,
            forward_batch=forward_batch,
            output_tensor=output_tensor,
        )
        _set_ar1_pending(True)
        return out
    return original(
        layer,
        input_,
        skip_all_reduce=skip_all_reduce,
        forward_batch=forward_batch,
        output_tensor=output_tensor,
    )
