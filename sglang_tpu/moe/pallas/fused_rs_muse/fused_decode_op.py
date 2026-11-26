# SPDX-License-Identifier: Apache-2.0
"""Distributed decode from the model's precomputed, scaled FP32 router logits."""

# PEP 563 must stay off so TorchTPU jax_op / cpu_export_jax_op sees real jax.Array annotations.

from functools import cache, partial
import inspect

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.pallas import jax_op

from sglang_tpu.moe.pallas.fused_moe_rs_op import _get_mesh
from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_experts import (
    fused_decode_from_logits,
)

SUPPORTED_FUSE_ALL_MODES = ("sparse", "sparse_v2", "sparse_fp8", "fp8_sparse", "ep_tp2", "chipmate")
SUPPORTED_SHARDED_FUSE_ALL_MODES = ("sparse", "sparse_v2", "sparse_fp8", "fp8_sparse", "ep_tp2")
SUPPORTED_FULL_MOE_MODES = ("full", "stage_a", "stage_b")
SUPPORTED_FUSE_ALL_BATCHES = (16, 32, 64, 128, 256)
SUPPORTED_FUSE_ALL_BATCHES_B384 = (16, 32, 64, 128, 256, 384)
SUPPORTED_FUSE_ALL_FP8_BATCHES = (16, 32, 64, 128)
SUPPORTED_FUSE_ALL_BLOCKS = (16, 256, 512)
SUPPORTED_REPLICA_SLOTS = (0, 1, 2, 4, 6, 8, 12, 16)
SUPPORTED_REPLICA_COST_MODES = ("active", "tiles", "weighted")


def supported_fuse_all_batches(ep: int | None = None) -> tuple[int, ...]:
    """Return active Muse fuse-all decode batch buckets (adds 8 under DECODE_B8, 384 under DECODE_B384)."""
    from sglang_tpu.envs import envs as tpu_envs

    b384_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_DECODE_B384", None)
    batches = (
        SUPPORTED_FUSE_ALL_BATCHES_B384
        if (b384_env is not None and b384_env.get())
        else SUPPORTED_FUSE_ALL_BATCHES
    )
    b8_env = getattr(tpu_envs, "SGLANG_TPU_MUSE_KERNEL_DECODE_B8", None)
    if b8_env is not None and b8_env.get() and ep in (None, 8):
        return (8, *batches)
    return batches


def get_fused_decode_op(ep=16):
    return _get_fused_decode_op(ep)


@cache
def _get_fused_decode_op(ep):
    mesh = _get_mesh()
    specs = (P(),) * 9
    mapped = jax.shard_map(
        partial(fused_decode_from_logits, ep=ep),
        mesh=mesh,
        in_specs=specs,
        out_specs=P(),
        check_vma=False,
    )

    def wrapped(
        rank: jax.Array,
        logits: jax.Array,
        bias: jax.Array,
        x: jax.Array,
        w1: jax.Array,
        s1: jax.Array,
        w2: jax.Array,
        s2: jax.Array,
        norm: jax.Array,
    ) -> jax.Array:
        return mapped(rank, logits, bias, x, w1, s1, w2, s2, norm)

    op = jax_op(
        "sglang_tpu_pallas::muse_from_logits_ep8_native"
        if ep == 8
        else "sglang_tpu_pallas::muse_from_logits_ep16_compact",
        wrapped,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=(P(),),
    )

    @op.register_fake
    def abstract(rank, logits, bias, x, w1, s1, w2, s2, norm):
        return torch.empty_like(x)

    return op


def muse_fuse_all_decode_op_name(
    *,
    mode: str,
    ep: int,
    tp: int = 1,
    block: int = 256,
    num_replicas: int = 0,
    replica_cost_mode: str = "tiles",
    chipmate_xor: int | None = None,
    return_sharded: bool = False,
    has_w3_mask: bool = False,
    has_spec_slot: bool = False,
    has_free_params: bool = False,
) -> str:
    """Validate fuse-all decode op parameters and return the canonical TorchTPU op name."""
    if mode not in SUPPORTED_FUSE_ALL_MODES:
        raise ValueError(
            f"Unsupported Muse fuse-all decode mode {mode!r}; "
            f"expected one of {SUPPORTED_FUSE_ALL_MODES}"
        )
    if has_w3_mask and mode != "sparse_v2":
        raise ValueError(
            f"has_w3_mask=True requires mode='sparse_v2', got {mode!r}"
        )
    if has_spec_slot and mode != "sparse":
        raise ValueError(
            f"has_spec_slot=True requires mode='sparse', got {mode!r}"
        )
    if has_free_params and mode != "sparse":
        raise ValueError(
            f"has_free_params=True requires mode='sparse', got {mode!r}"
        )
    if has_w3_mask and (has_spec_slot or has_free_params):
        raise ValueError("has_w3_mask cannot be combined with has_spec_slot or has_free_params")
    if return_sharded and mode not in SUPPORTED_SHARDED_FUSE_ALL_MODES:
        raise ValueError(
            f"return_sharded=True requires mode in {SUPPORTED_SHARDED_FUSE_ALL_MODES}, "
            f"got {mode!r}"
        )
    if block not in SUPPORTED_FUSE_ALL_BLOCKS or (
        block == 512 and mode == "chipmate"
    ):
        raise ValueError(
            f"Unsupported FP4 scale block {block} for mode={mode!r}; "
            f"expected one of {SUPPORTED_FUSE_ALL_BLOCKS}"
        )
    if num_replicas not in SUPPORTED_REPLICA_SLOTS:
        raise ValueError(
            f"Unsupported replica slot count {num_replicas}; "
            f"expected one of {SUPPORTED_REPLICA_SLOTS}"
        )
    if replica_cost_mode not in SUPPORTED_REPLICA_COST_MODES:
        raise ValueError(
            f"Unsupported replica cost mode {replica_cost_mode!r}; "
            f"expected one of {SUPPORTED_REPLICA_COST_MODES}"
        )
    if mode == "fp8_sparse":
        if ep not in (8, 16) or tp != 1:
            raise ValueError(
                f"mode='fp8_sparse' requires ep in (8, 16) and tp=1, got ep={ep}, tp={tp}"
            )
        stem = f"muse_decode_fp8_sparse_ep{ep}_r{num_replicas}"
    elif mode == "ep_tp2":
        if ep not in (4, 8) or tp != 2:
            raise ValueError(
                f"mode='ep_tp2' requires ep in (4, 8) and tp=2, got ep={ep}, tp={tp}"
            )
        stem = f"muse_decode_ep{ep}_tp{tp}_k{block}_r{num_replicas}"
    else:
        if ep not in (8, 16) or tp != 1:
            raise ValueError(
                f"mode={mode!r} requires ep in (8, 16) and tp=1, got ep={ep}, tp={tp}"
            )
        if mode == "sparse_fp8":
            stem = f"muse_decode_sparse_fp8_ep{ep}_r{num_replicas}"
        else:
            stem = f"muse_decode_{mode}_ep{ep}_k{block}_r{num_replicas}"
    if num_replicas > 0:
        stem = f"{stem}_{replica_cost_mode}"
    if chipmate_xor is not None and mode in ("ep_tp2", "chipmate"):
        if chipmate_xor <= 0 or (chipmate_xor & (chipmate_xor - 1)) != 0:
            raise ValueError(
                f"chipmate_xor must be a positive power of two, got {chipmate_xor}"
            )
        stem = f"{stem}_x{chipmate_xor}"
    if has_w3_mask:
        stem = f"{stem}_w3"
    if has_spec_slot:
        stem = f"{stem}_ss"
    if has_free_params:
        stem = f"{stem}_dfp"
    if return_sharded:
        stem = f"{stem}_sh"
    return f"sglang_tpu_pallas::{stem}"


def _validate_fuse_all_shapes(
    op_name: str,
    mode: str,
    block: int,
    x: jax.Array,
    w1: jax.Array,
    s1: jax.Array,
    *,
    ep: int = 8,
    tp: int = 1,
    return_sharded: bool = False,
) -> None:
    sh_div = ep * tp
    if return_sharded and (x.shape[0] < sh_div or x.shape[0] % sh_div != 0):
        raise ValueError(
            f"{op_name} return_sharded=True requires batch_size divisible by world_size={sh_div}, "
            f"got {x.shape[0]}"
        )
    if mode in ("sparse_fp8", "fp8_sparse"):
        if str(w1.dtype) != "float8_e4m3fn":
            raise ValueError(
                f"{op_name} expected float8_e4m3fn weights, got {w1.dtype}"
            )
        return
    if x.shape[1] // s1.shape[1] != block:
        raise ValueError(
            f"{op_name} expected K-block {block}, "
            f"got {x.shape[1] // s1.shape[1]}"
        )


def _resolve_fuse_all_kernel(
    *,
    mode: str,
    ep: int,
    tp: int,
    block: int,
    replica_cost_mode: str,
    chipmate_xor: int | None,
    return_sharded: bool = False,
):
    sh_kw = {"return_sharded": True} if return_sharded else {}
    if mode == "fp8_sparse":
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
            fused_decode_sparse_fp8_from_logits,
        )

        return partial(
            fused_decode_sparse_fp8_from_logits,
            ep=ep,
            replica_cost_mode=replica_cost_mode,
            **sh_kw,
        )
    if mode == "sparse":
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
            fused_decode_sparse_from_logits,
        )

        return partial(
            fused_decode_sparse_from_logits,
            ep=ep,
            replica_cost_mode=replica_cost_mode,
            **sh_kw,
        )
    if mode == "sparse_v2":
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse_v2 import (
            fused_decode_sparse_v2_from_logits,
        )

        return partial(
            fused_decode_sparse_v2_from_logits,
            ep=ep,
            replica_cost_mode=replica_cost_mode,
            **sh_kw,
        )
    if mode == "sparse_fp8":
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse_v2 import (
            fused_decode_sparse_fp8_from_logits,
        )

        return partial(
            fused_decode_sparse_fp8_from_logits,
            ep=ep,
            replica_cost_mode=replica_cost_mode,
            **sh_kw,
        )
    if mode == "ep_tp2":
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_ep_tp2 import (
            fused_decode_ep_tp2_from_logits,
        )

        kwargs = {"ep": ep, "tp": tp, "replica_cost_mode": replica_cost_mode, **sh_kw}
        if chipmate_xor is not None:
            kwargs["chipmate_xor"] = chipmate_xor
        return partial(fused_decode_ep_tp2_from_logits, **kwargs)
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_chipmate import (
        fused_decode_chipmate_from_logits,
    )

    kwargs = {"ep": ep, "replica_cost_mode": replica_cost_mode}
    if block == 16:
        kwargs["chipmate_prefer_max_rows"] = True
    if chipmate_xor is not None:
        kwargs["chipmate_xor"] = chipmate_xor
    return partial(fused_decode_chipmate_from_logits, **kwargs)


def get_muse_fuse_all_decode_op(
    *,
    mode: str,
    ep: int,
    tp: int = 1,
    block: int = 256,
    num_replicas: int = 0,
    replica_cost_mode: str = "tiles",
    chipmate_xor: int | None = None,
    return_sharded: bool = False,
    has_w3_mask: bool = False,
    has_spec_slot: bool = False,
    has_free_params: bool = False,
):
    """Return a cached TorchTPU jax_op for Muse Spark fuse-all decode."""
    op_name = muse_fuse_all_decode_op_name(
        mode=mode,
        ep=ep,
        tp=tp,
        block=block,
        num_replicas=num_replicas,
        replica_cost_mode=replica_cost_mode,
        chipmate_xor=chipmate_xor,
        return_sharded=return_sharded,
        has_w3_mask=has_w3_mask,
        has_spec_slot=has_spec_slot,
        has_free_params=has_free_params,
    )
    return _get_muse_fuse_all_decode_op(
        op_name,
        mode,
        ep,
        tp,
        block,
        num_replicas,
        replica_cost_mode,
        chipmate_xor,
        return_sharded,
        has_w3_mask,
        has_spec_slot,
        has_free_params,
    )


@cache
def _get_muse_fuse_all_decode_op(
    op_name: str,
    mode: str,
    ep: int,
    tp: int,
    block: int,
    num_replicas: int,
    replica_cost_mode: str,
    chipmate_xor: int | None,
    return_sharded: bool = False,
    has_w3_mask: bool = False,
    has_spec_slot: bool = False,
    has_free_params: bool = False,
):
    mesh = _get_mesh()
    kernel_fn = _resolve_fuse_all_kernel(
        mode=mode,
        ep=ep,
        tp=tp,
        block=block,
        replica_cost_mode=replica_cost_mode,
        chipmate_xor=chipmate_xor,
        return_sharded=return_sharded,
    )
    extra_keys: list[str] = []
    if has_w3_mask:
        extra_keys.append("w3_mask")
    if has_spec_slot:
        extra_keys.append("spec_home_slot")
    if has_free_params:
        extra_keys.append("free_params")

    base_names = ("rank", "logits", "bias", "x", "w1", "s1", "w2", "s2", "norm")
    rep_names = ("rep_w1", "rep_s1", "rep_w2", "rep_s2", "replica_meta")
    arg_names = list(base_names) + (list(rep_names) if num_replicas > 0 else []) + extra_keys
    specs = (P(),) * len(arg_names)

    def _call_local(*t_args):
        n_core = len(base_names) + (len(rep_names) if num_replicas > 0 else 0)
        ex_kw = {
            k: v for k, v in zip(extra_keys, t_args[n_core:])
        }
        return kernel_fn(*t_args[:n_core], **ex_kw)

    mapped = jax.shard_map(
        _call_local,
        mesh=mesh,
        in_specs=specs,
        out_specs=P(),
        check_vma=False,
    )

    def wrapped(*args) -> jax.Array:
        x_val = args[3]
        w1_val = args[4]
        s1_val = args[5]
        _validate_fuse_all_shapes(
            op_name, mode, block, x_val, w1_val, s1_val, ep=ep, tp=tp, return_sharded=return_sharded
        )
        if num_replicas > 0:
            rep_w2_val = args[11]
            if rep_w2_val.shape[0] != num_replicas:
                raise ValueError(
                    f"{op_name} expected {num_replicas} replica slots, "
                    f"got {rep_w2_val.shape[0]}"
                )
        return mapped(*args)

    wrapped.__name__ = op_name.split("::")[-1]
    wrapped.__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=jax.Array)
            for n in arg_names
        ],
        return_annotation=jax.Array,
    )

    op = jax_op(
        op_name,
        wrapped,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=(P(),),
    )

    @op.register_fake
    def abstract(*args, **kwargs):
        x_val = args[3]
        if return_sharded:
            return x_val.new_empty((x_val.shape[0] // (ep * tp), x_val.shape[1]))
        return torch.empty_like(x_val)

    return op


def _ep_tp2_virtual_ids(topk_ids, *, e_chip: int, chipmate_xor: int):
    """Map real expert IDs [T, 8] in [0, 256) to paired virtual IDs [T, 16] in [0, 512)."""
    import jax.numpy as jnp

    chip = topk_ids // e_chip
    local_e = topk_ids % e_chip
    bit = chipmate_xor.bit_length() - 1
    r0 = ((chip >> bit) << (bit + 1)) | (chip & (chipmate_xor - 1))
    r1 = r0 | chipmate_xor
    v0 = r0 * e_chip + local_e
    v1 = r1 * e_chip + local_e
    return jnp.concatenate([v0, v1], axis=1).astype(jnp.int32)


def ep_tp2_fallback_from_topk(
    rank,
    hidden_states,
    topk_weights,
    topk_ids,
    w1,
    s1,
    w2,
    s2,
    norm,
    *,
    ep: int = 4,
    tp: int = 2,
    chipmate_xor: int = 1,
):
    """Mixed-mode EP(P/2)xTP2 fallback (EXTEND prefill & non-bucketed decode)
    running directly on the halved-intermediate float4_e2m1fn weights with zero extra HBM.

    Maps each real (token, slot) expert choice to both physical chipmates' virtual expert
    IDs, runs ``moe_gmm_local_rs_nodedup`` with ``topk=16, return_raw_out_buf=True``,
    sums the two TP2 partials in FP32 before Muse's ``scale_input=True`` RMSNorm,
    combines over top-8, and all-gathers the replicated ``[T, 4096]`` output.
    """
    import jax
    import jax.numpy as jnp

    from sglang_tpu.moe.pallas.fused_moe_rs_op import (
        fused_moe_rs_padded_tokens,
        pad_moe_tokens,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_moe import (
        moe_gmm_local_rs_nodedup,
    )

    if ep not in (4, 8) or tp != 2:
        raise ValueError(f"ep_tp2 fallback requires ep in (4, 8) and tp=2, got {ep}, {tp}")
    p_world = ep * tp
    e_chip = 256 // ep
    num_tokens, hidden_size = hidden_states.shape
    topk = topk_ids.shape[1]
    kernel_topk = topk * tp

    valid = (topk_ids >= 0) & (topk_ids < 256)
    safe_ids = jnp.where(valid, topk_ids.astype(jnp.int32), jnp.int32(0))
    safe_w = jnp.where(valid, topk_weights.astype(jnp.float32), jnp.float32(0.0))

    padded_tokens = fused_moe_rs_padded_tokens(num_tokens, kernel_topk, p_world)
    hs_pad, w_pad, ids_pad = pad_moe_tokens(
        hidden_states, safe_w, safe_ids, padded_tokens=padded_tokens
    )
    virt_ids = _ep_tp2_virtual_ids(
        ids_pad, e_chip=e_chip, chipmate_xor=chipmate_xor
    )
    virt_w = jnp.ones((padded_tokens, kernel_topk), dtype=hidden_states.dtype)
    group_offset = (rank.reshape((1,)) * jnp.int32(e_chip)).astype(jnp.int32)
    if hidden_size // s1.shape[1] == 512:
        s1 = jnp.repeat(s1, 2, axis=1)
        s2 = jnp.repeat(s2, 2, axis=1)

    out_buf = moe_gmm_local_rs_nodedup(
        hidden_states_local=hs_pad,
        w1=w1,
        w1_scale=s1,
        w1_bias=None,
        w2=w2,
        w2_scale=s2,
        w2_bias=None,
        w1_global_scale=None,
        w2_global_scale=None,
        group_offset=group_offset,
        topk_weights=virt_w,
        topk_indices=virt_ids,
        post_expert_norm_weight_input=None,
        activation="silu",
        topk=kernel_topk,
        ep_size=p_world,
        ep_axis_name="model",
        has_post_norm=False,
        sp_enabled=False,
        fp8_direct_write=False,
        expert_tp=False,
        fp4_w4a16=False,
        return_raw_out_buf=True,
    )
    chunk_size = padded_tokens // p_world
    out_4d = out_buf.reshape(tp, topk, chunk_size, -1)[:, :, :, :hidden_size]
    z = out_4d[0].astype(jnp.float32) + out_4d[1].astype(jnp.float32)
    pnw = norm.reshape((hidden_size,)).astype(jnp.float32) + 1.0
    u = z * pnw[None, None, :]
    v = u * jax.lax.rsqrt(jnp.mean(u * u, axis=-1, keepdims=True) + 1e-8)

    local_w = jax.lax.dynamic_slice_in_dim(
        w_pad, rank[0] * chunk_size, chunk_size, axis=0
    )
    token_shard = jnp.sum(
        v * jnp.swapaxes(local_w, 0, 1)[:, :, None], axis=0
    ).astype(hidden_states.dtype)
    full = jax.lax.all_gather(token_shard, axis_name="model", axis=0, tiled=True)
    return full[:num_tokens, :hidden_size]


def get_muse_ep_tp2_fallback_op(
    *,
    ep: int = 4,
    tp: int = 2,
    block: int = 256,
    chipmate_xor: int | None = None,
):
    """Return cached TorchTPU jax_op for mixed-mode EPxTP2 prefill/fallback."""
    if ep not in (4, 8) or tp != 2 or block not in (16, 256, 512):
        raise ValueError(
            f"Unsupported EPxTP2 fallback config: ep={ep}, tp={tp}, block={block}"
        )
    if chipmate_xor is None:
        try:
            from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_ep_tp2 import (
                TPU7X_CHIPMATE_XOR,
            )

            xor = int(TPU7X_CHIPMATE_XOR)
        except ImportError:
            xor = 1
    else:
        xor = int(chipmate_xor)
    if xor <= 0 or (xor & (xor - 1)) != 0:
        raise ValueError(f"chipmate_xor must be a positive power of two, got {xor}")
    op_name = f"sglang_tpu_pallas::muse_ep{ep}_tp{tp}_k{block}_fallback_x{xor}"
    return _get_muse_ep_tp2_fallback_op(op_name, ep, tp, block, xor)


@cache
def _get_muse_ep_tp2_fallback_op(
    op_name: str,
    ep: int,
    tp: int,
    block: int,
    chipmate_xor: int,
):
    mesh = _get_mesh()
    specs = (P(),) * 9
    mapped = jax.shard_map(
        partial(ep_tp2_fallback_from_topk, ep=ep, tp=tp, chipmate_xor=chipmate_xor),
        mesh=mesh,
        in_specs=specs,
        out_specs=P(),
        check_vma=False,
    )

    def wrapped(
        rank: jax.Array,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        w1: jax.Array,
        s1: jax.Array,
        w2: jax.Array,
        s2: jax.Array,
        norm: jax.Array,
    ) -> jax.Array:
        if hidden_states.shape[1] // s1.shape[1] != block:
            raise ValueError(
                f"{op_name} expected K-block {block}, "
                f"got {hidden_states.shape[1] // s1.shape[1]}"
            )
        return mapped(
            rank, hidden_states, topk_weights, topk_ids, w1, s1, w2, s2, norm
        )

    op = jax_op(
        op_name,
        wrapped,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=(P(),),
    )

    @op.register_fake
    def abstract(rank, hidden_states, topk_weights, topk_ids, w1, s1, w2, s2, norm):
        return torch.empty_like(hidden_states)

    return op


def muse_full_moe_decode_op_name(
    *,
    full_mode: str = "full",
    ep: int,
    block: int = 256,
    num_replicas: int = 0,
    replica_cost_mode: str = "tiles",
) -> str:
    """Validate full-layer decode MoE op parameters and return canonical TorchTPU op name."""
    if full_mode not in SUPPORTED_FULL_MOE_MODES:
        raise ValueError(
            f"Unsupported Muse full-MoE decode mode {full_mode!r}; "
            f"expected one of {SUPPORTED_FULL_MOE_MODES}"
        )
    if ep not in (8, 16):
        raise ValueError(f"full_mode={full_mode!r} requires ep in (8, 16), got ep={ep}")
    if block not in SUPPORTED_FUSE_ALL_BLOCKS:
        raise ValueError(
            f"Unsupported FP4 scale block {block}; expected one of {SUPPORTED_FUSE_ALL_BLOCKS}"
        )
    if num_replicas not in SUPPORTED_REPLICA_SLOTS:
        raise ValueError(
            f"Unsupported replica slot count {num_replicas}; "
            f"expected one of {SUPPORTED_REPLICA_SLOTS}"
        )
    if replica_cost_mode not in SUPPORTED_REPLICA_COST_MODES:
        raise ValueError(
            f"Unsupported replica cost mode {replica_cost_mode!r}; "
            f"expected one of {SUPPORTED_REPLICA_COST_MODES}"
        )
    stem = f"muse_decode_full_{full_mode}_ep{ep}_k{block}_r{num_replicas}"
    if num_replicas > 0:
        stem = f"{stem}_{replica_cost_mode}"
    return f"sglang_tpu_pallas::{stem}"


def get_muse_full_moe_decode_op(
    *,
    full_mode: str = "full",
    ep: int,
    block: int = 256,
    num_replicas: int = 0,
    replica_cost_mode: str = "tiles",
    router_multiplier: float = 0.17677669529663687,
    pre_norm_eps: float = 1e-5,
):
    """Return a cached TorchTPU jax_op for Muse Spark full-layer boundary-fused decode MoE."""
    op_name = muse_full_moe_decode_op_name(
        full_mode=full_mode,
        ep=ep,
        block=block,
        num_replicas=num_replicas,
        replica_cost_mode=replica_cost_mode,
    )
    return _get_muse_full_moe_decode_op(
        op_name,
        full_mode,
        ep,
        block,
        num_replicas,
        replica_cost_mode,
        float(router_multiplier),
        float(pre_norm_eps),
    )


@cache
def _get_muse_full_moe_decode_op(
    op_name: str,
    full_mode: str,
    ep: int,
    block: int,
    num_replicas: int,
    replica_cost_mode: str,
    router_multiplier: float,
    pre_norm_eps: float,
):
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_full_layer import (
        fused_decode_full_moe,
        fused_decode_stage_a_from_logits,
        fused_decode_stage_b_moe,
    )

    mesh = _get_mesh()

    if full_mode == "stage_a":
        kernel_fn = partial(
            fused_decode_stage_a_from_logits,
            ep=ep,
            replica_cost_mode=replica_cost_mode,
        )
        if num_replicas == 0:
            specs = (P(),) * 10
            mapped = jax.shard_map(
                kernel_fn, mesh=mesh, in_specs=specs, out_specs=P(), check_vma=False
            )

            def wrapped_sa(
                rank: jax.Array,
                logits: jax.Array,
                bias: jax.Array,
                x: jax.Array,
                w1: jax.Array,
                s1: jax.Array,
                w2: jax.Array,
                s2: jax.Array,
                norm: jax.Array,
                w_post: jax.Array,
            ) -> jax.Array:
                if x.shape[1] // s1.shape[1] != block:
                    raise ValueError(
                        f"{op_name} expected K-block {block}, got {x.shape[1] // s1.shape[1]}"
                    )
                return mapped(rank, logits, bias, x, w1, s1, w2, s2, norm, w_post)

            op = jax_op(
                op_name,
                wrapped_sa,
                mesh=mesh,
                input_partition_specs=specs,
                output_partition_specs=(P(),),
            )

            @op.register_fake
            def abstract_sa(rank, logits, bias, x, w1, s1, w2, s2, norm, w_post):
                return x.new_empty((x.shape[0], 8192))

            return op

        specs = (P(),) * 15
        mapped = jax.shard_map(
            kernel_fn, mesh=mesh, in_specs=specs, out_specs=P(), check_vma=False
        )

        def wrapped_sa_rep(
            rank: jax.Array,
            logits: jax.Array,
            bias: jax.Array,
            x: jax.Array,
            w1: jax.Array,
            s1: jax.Array,
            w2: jax.Array,
            s2: jax.Array,
            norm: jax.Array,
            w_post: jax.Array,
            rep_w1: jax.Array,
            rep_s1: jax.Array,
            rep_w2: jax.Array,
            rep_s2: jax.Array,
            replica_meta: jax.Array,
        ) -> jax.Array:
            if x.shape[1] // s1.shape[1] != block:
                raise ValueError(
                    f"{op_name} expected K-block {block}, got {x.shape[1] // s1.shape[1]}"
                )
            if rep_w1.shape[0] != num_replicas:
                raise ValueError(
                    f"{op_name} expected {num_replicas} replica slots, got {rep_w1.shape[0]}"
                )
            # Reorder so w_post precedes rep_args per fused_decode_stage_a_from_logits.
            return mapped(
                rank,
                logits,
                bias,
                x,
                w1,
                s1,
                w2,
                s2,
                norm,
                w_post,
                rep_w1,
                rep_s1,
                rep_w2,
                rep_s2,
                replica_meta,
            )

        op = jax_op(
            op_name,
            wrapped_sa_rep,
            mesh=mesh,
            input_partition_specs=specs,
            output_partition_specs=(P(),),
        )

        @op.register_fake
        def abstract_sa_rep(
            rank,
            logits,
            bias,
            x,
            w1,
            s1,
            w2,
            s2,
            norm,
            w_post,
            rep_w1,
            rep_s1,
            rep_w2,
            rep_s2,
            replica_meta,
        ):
            return x.new_empty((x.shape[0], 8192))

        return op

    if full_mode == "stage_b":
        kernel_fn = partial(
            fused_decode_stage_b_moe,
            ep=ep,
            router_multiplier=router_multiplier,
            pre_norm_eps=pre_norm_eps,
            replica_cost_mode=replica_cost_mode,
        )
        if num_replicas == 0:
            specs = (P(),) * 12
            mapped = jax.shard_map(
                kernel_fn, mesh=mesh, in_specs=specs, out_specs=P(), check_vma=False
            )

            def wrapped_sb(
                rank: jax.Array,
                hidden_states: jax.Array,
                router_input: jax.Array,
                gate_weight: jax.Array,
                bias: jax.Array,
                w_pre: jax.Array,
                pre_gain: jax.Array,
                w1: jax.Array,
                s1: jax.Array,
                w2: jax.Array,
                s2: jax.Array,
                norm: jax.Array,
            ) -> jax.Array:
                if 4096 // s1.shape[1] != block:
                    raise ValueError(
                        f"{op_name} expected K-block {block}, got {4096 // s1.shape[1]}"
                    )
                return mapped(
                    rank,
                    hidden_states,
                    router_input,
                    gate_weight,
                    bias,
                    w_pre,
                    pre_gain,
                    w1,
                    s1,
                    w2,
                    s2,
                    norm,
                )

            op = jax_op(
                op_name,
                wrapped_sb,
                mesh=mesh,
                input_partition_specs=specs,
                output_partition_specs=(P(),),
            )

            @op.register_fake
            def abstract_sb(
                rank,
                hidden_states,
                router_input,
                gate_weight,
                bias,
                w_pre,
                pre_gain,
                w1,
                s1,
                w2,
                s2,
                norm,
            ):
                return hidden_states.new_empty((router_input.shape[0], 4096))

            return op

        specs = (P(),) * 17
        mapped = jax.shard_map(
            kernel_fn, mesh=mesh, in_specs=specs, out_specs=P(), check_vma=False
        )

        def wrapped_sb_rep(
            rank: jax.Array,
            hidden_states: jax.Array,
            router_input: jax.Array,
            gate_weight: jax.Array,
            bias: jax.Array,
            w_pre: jax.Array,
            pre_gain: jax.Array,
            w1: jax.Array,
            s1: jax.Array,
            w2: jax.Array,
            s2: jax.Array,
            norm: jax.Array,
            rep_w1: jax.Array,
            rep_s1: jax.Array,
            rep_w2: jax.Array,
            rep_s2: jax.Array,
            replica_meta: jax.Array,
        ) -> jax.Array:
            if 4096 // s1.shape[1] != block:
                raise ValueError(
                    f"{op_name} expected K-block {block}, got {4096 // s1.shape[1]}"
                )
            if rep_w1.shape[0] != num_replicas:
                raise ValueError(
                    f"{op_name} expected {num_replicas} replica slots, got {rep_w1.shape[0]}"
                )
            return mapped(
                rank,
                hidden_states,
                router_input,
                gate_weight,
                bias,
                w_pre,
                pre_gain,
                w1,
                s1,
                w2,
                s2,
                norm,
                rep_w1,
                rep_s1,
                rep_w2,
                rep_s2,
                replica_meta,
            )

        op = jax_op(
            op_name,
            wrapped_sb_rep,
            mesh=mesh,
            input_partition_specs=specs,
            output_partition_specs=(P(),),
        )

        @op.register_fake
        def abstract_sb_rep(
            rank,
            hidden_states,
            router_input,
            gate_weight,
            bias,
            w_pre,
            pre_gain,
            w1,
            s1,
            w2,
            s2,
            norm,
            rep_w1,
            rep_s1,
            rep_w2,
            rep_s2,
            replica_meta,
        ):
            return hidden_states.new_empty((router_input.shape[0], 4096))

        return op

    kernel_fn = partial(
        fused_decode_full_moe,
        ep=ep,
        router_multiplier=router_multiplier,
        pre_norm_eps=pre_norm_eps,
        replica_cost_mode=replica_cost_mode,
    )
    if num_replicas == 0:
        specs = (P(),) * 13
        mapped = jax.shard_map(
            kernel_fn, mesh=mesh, in_specs=specs, out_specs=P(), check_vma=False
        )

        def wrapped_full(
            rank: jax.Array,
            hidden_states: jax.Array,
            router_input: jax.Array,
            gate_weight: jax.Array,
            bias: jax.Array,
            w_pre: jax.Array,
            pre_gain: jax.Array,
            w1: jax.Array,
            s1: jax.Array,
            w2: jax.Array,
            s2: jax.Array,
            norm: jax.Array,
            w_post: jax.Array,
        ) -> jax.Array:
            if 4096 // s1.shape[1] != block:
                raise ValueError(
                    f"{op_name} expected K-block {block}, got {4096 // s1.shape[1]}"
                )
            return mapped(
                rank,
                hidden_states,
                router_input,
                gate_weight,
                bias,
                w_pre,
                pre_gain,
                w1,
                s1,
                w2,
                s2,
                norm,
                w_post,
            )

        op = jax_op(
            op_name,
            wrapped_full,
            mesh=mesh,
            input_partition_specs=specs,
            output_partition_specs=(P(),),
        )

        @op.register_fake
        def abstract_full(
            rank,
            hidden_states,
            router_input,
            gate_weight,
            bias,
            w_pre,
            pre_gain,
            w1,
            s1,
            w2,
            s2,
            norm,
            w_post,
        ):
            return hidden_states.new_empty((router_input.shape[0], 8192))

        return op

    specs = (P(),) * 18
    mapped = jax.shard_map(
        kernel_fn, mesh=mesh, in_specs=specs, out_specs=P(), check_vma=False
    )

    def wrapped_full_rep(
        rank: jax.Array,
        hidden_states: jax.Array,
        router_input: jax.Array,
        gate_weight: jax.Array,
        bias: jax.Array,
        w_pre: jax.Array,
        pre_gain: jax.Array,
        w1: jax.Array,
        s1: jax.Array,
        w2: jax.Array,
        s2: jax.Array,
        norm: jax.Array,
        w_post: jax.Array,
        rep_w1: jax.Array,
        rep_s1: jax.Array,
        rep_w2: jax.Array,
        rep_s2: jax.Array,
        replica_meta: jax.Array,
    ) -> jax.Array:
        if 4096 // s1.shape[1] != block:
            raise ValueError(
                f"{op_name} expected K-block {block}, got {4096 // s1.shape[1]}"
            )
        if rep_w1.shape[0] != num_replicas:
            raise ValueError(
                f"{op_name} expected {num_replicas} replica slots, got {rep_w1.shape[0]}"
            )
        return mapped(
            rank,
            hidden_states,
            router_input,
            gate_weight,
            bias,
            w_pre,
            pre_gain,
            w1,
            s1,
            w2,
            s2,
            norm,
            w_post,
            rep_w1,
            rep_s1,
            rep_w2,
            rep_s2,
            replica_meta,
        )

    op = jax_op(
        op_name,
        wrapped_full_rep,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=(P(),),
    )

    @op.register_fake
    def abstract_full_rep(
        rank,
        hidden_states,
        router_input,
        gate_weight,
        bias,
        w_pre,
        pre_gain,
        w1,
        s1,
        w2,
        s2,
        norm,
        w_post,
        rep_w1,
        rep_s1,
        rep_w2,
        rep_s2,
        replica_meta,
    ):
        return hidden_states.new_empty((router_input.shape[0], 8192))

    return op
