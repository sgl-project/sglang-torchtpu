# SPDX-License-Identifier: Apache-2.0
"""TorchTPU custom ops for Wave-9 decode superkernels (Step (a) b1+MoE and Step (b) b1+MoE+b2)."""

# PEP 563 must stay off so cpu_export_jax_op sees concrete jax.Array annotations.

import inspect
import re
from threading import Lock
from typing import Callable, Iterable, Optional

import torch

from sglang_tpu.kernels.decode_collective_bridges import (
    SUPPORTED_WORLD_SIZES,
    _get_tp_mesh,
)
from sglang_tpu.muse_megakernel.decode.b1_moe_b2_b16_kernel import (
    b1_moe_b2_replicated_b16_jax,
)
from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
    b1_moe_b2_chip_sharded_jax,
    b1_moe_b2_rpad_chip_sharded_jax,
)
from sglang_tpu.muse_megakernel.decode.b1_moe_kernel import (
    b1_moe_chip_sharded_jax,
)

_B1_MOE_LOCK = Lock()
_B1_MOE_B2_OPS: dict[tuple, Callable] = {}
_B1_MOE_B2_B16_OPS: dict[tuple, Callable] = {}
_B1_MOE_B2_RPAD_OPS: dict[tuple, Callable] = {}
_B1_MOE_OPS: dict[tuple, Callable] = {}

_B1_BASE_TENSOR_NAMES = (
    "rank",
    "stream_chip",
    "gated_attn",
    "wo_w8k",
    "wo_scale",
    "pa_alpha",
    "pa_beta",
    "pre_ff_gain",
    "w_pre_pair_w8k",
    "w_pre_pair_scale",
    "w_router_km",
    "pre_exp_gain",
    "bias",
    "w1",
    "s1",
    "w2",
    "s2",
    "post_exp_norm",
)

_B2_TENSOR_NAMES = (
    "w_post_tp2_w8k",
    "s_post_tp2",
    "post_ff_gain",
    "pf_alpha",
    "pf_beta",
    "next_in_gain",
    "next_w_qkv_w8k",
    "next_s_qkv",
    "next_w_gate_w8k",
    "next_s_gate",
)

_REPLICA_TENSOR_NAMES = (
    "rep_w1",
    "rep_s1",
    "rep_w2",
    "rep_s2",
    "replica_meta",
)

_B1_STATIC_SPECS: tuple[tuple[str, type], ...] = (
    ("pa_eps", float),
    ("pre_ff_eps", float),
    ("pre_ff_scale_input", bool),
    ("router_multiplier", float),
    ("pre_exp_eps", float),
    ("pre_exp_scale_input", bool),
)

_B2_STATIC_SPECS: tuple[tuple[str, type], ...] = (
    ("post_ff_eps", float),
    ("post_ff_scale_input", bool),
    ("next_in_eps", float),
    ("next_in_scale_input", bool),
)

_EPI_STATIC_SPECS: tuple[tuple[str, type], ...] = (
    ("qkv_norm_eps", float),
    ("query_scale", float),
    ("k_scale", float),
    ("v_scale", float),
)


def _custom_op_device_types() -> Optional[list[str]]:
    try:
        torch._C._dispatch_key_for_device("tpu")
        return ["tpu"]
    except RuntimeError:
        return None


def _export_superkernel_op(
    op_name: str,
    world_size: int,
    tensor_names: tuple[str, ...],
    static_specs: tuple[tuple[str, type], ...],
    num_outputs: int,
    local_kernel_fn: Callable,
):
    """Export a shard_map-wrapped decode superkernel op via ``cpu_export_jax_op``."""
    import jax
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    mesh = _get_tp_mesh(world_size)
    n_t = len(tensor_names)
    t_specs = (P(),) * n_t
    out_specs = (P(),) * num_outputs
    in_specs = t_specs + (None,) * len(static_specs)
    params = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=jax.Array)
        for name in tensor_names
    ] + [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typ)
        for name, typ in static_specs
    ]
    ret_ann = tuple[(jax.Array,) * num_outputs]  # type: ignore[valid-type]

    def _wrapped(*args):
        tensors = args[:n_t]
        scalars = args[n_t:]

        def _local(*t_args):
            return local_kernel_fn(*t_args, *scalars)

        return jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=t_specs,
            out_specs=out_specs,
            check_vma=False,
        )(*tensors)

    _wrapped.__name__ = op_name.split("::")[-1]
    _wrapped.__signature__ = inspect.Signature(parameters=params, return_annotation=ret_ann)

    return cpu_export_jax_op(
        op_name,
        _wrapped,
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=out_specs,
    )


def _export_step_c_mutating_op(
    op_name: str,
    world_size: int,
    tensor_names: tuple[str, ...],
    static_specs: tuple[tuple[str, type], ...],
    local_kernel_fn: Callable,
):
    """Export Step (c) op with 3-level zero-copy in-place ``kv_cache_fused`` aliasing."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from torch_tpu._internal.pallas import pallas as pallas_impl

    from sglang_tpu.pallas_export import tpu_info_for_cpu_export
    from sglang_tpu.pallas_jax_op import (
        _CpuExportJaxCallable,
        _jax_placeholders,
        _torch_placeholder,
    )

    class _StepCExportJaxCallable(_CpuExportJaxCallable):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__signature__ = self.__signature__.replace(
                return_annotation=tuple[torch.Tensor, torch.Tensor]
            )

        def __call__(self, *args, **kwargs):
            res = super().__call__(*args, **kwargs)
            return res[:2]

    mesh = _get_tp_mesh(world_size)
    n_t = len(tensor_names)
    t_specs = (P(),) * n_t
    out_specs = (P(),) * 3
    in_specs = t_specs + (None,) * len(static_specs)
    params = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=jax.Array)
        for name in tensor_names
    ] + [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typ)
        for name, typ in static_specs
    ]

    def _wrapped(*args):
        tensors = args[:n_t]
        scalars = args[n_t:]

        def _local(*t_args):
            return local_kernel_fn(*t_args, *scalars)

        return jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=t_specs,
            out_specs=out_specs,
            check_vma=False,
        )(*tensors)

    _wrapped.__name__ = op_name.split("::")[-1]
    _wrapped.__signature__ = inspect.Signature(
        parameters=params, return_annotation=tuple[jax.Array, jax.Array, jax.Array]
    )

    signature = _wrapped.__signature__
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    kv_idx = list(signature.parameters.keys()).index("kv_cache_fused")
    donate_argnums_tuple = (kv_idx,)
    aliases = {kv_idx: 2}
    output_shardings = tuple(NamedSharding(mesh, spec) for spec in out_specs)
    jit_function = jax.jit(
        _wrapped,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums_tuple,
        keep_unused=True,
        out_shardings=output_shardings,
    )
    trace_key = pallas_impl._get_kernel_invocation_key(
        op_name,
        [],
        {
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums_tuple,
            "output_partition_specs": tuple(map(str, out_specs)),
        },
    )
    wrapped = _StepCExportJaxCallable(
        name=op_name,
        jit_fn=jit_function,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=in_specs,
        static_argnums=static_argnums,
        donate_argnums=list(donate_argnums_tuple),
        input_output_aliases=aliases,
    )
    wrapped.sglang_output_shardings = output_shardings
    custom_op = torch.library.custom_op(
        op_name,
        wrapped,
        mutates_args=("kv_cache_fused",),
        device_types=_custom_op_device_types(),
    )

    def fake_function(*args, **kwargs):
        jax_args = _jax_placeholders(
            args,
            mesh=mesh,
            partition_specs=in_specs,
        )
        with (
            tpu_info_for_cpu_export(),
            jax._src.config.export_ignore_forward_compatibility(True),
        ):
            lowered = wrapped.exported(*jax_args, **kwargs)
        shardings = output_shardings or (None,) * len(lowered.out_avals)
        unflat = lowered.out_tree.unflatten(
            _torch_placeholder(aval, sharding, mesh)
            for aval, sharding in zip(lowered.out_avals, shardings)
        )
        return unflat[:2]

    custom_op.register_fake(fake_function)
    return custom_op


def get_b1_moe_b2_chip_sharded_op(
    world_size: int,
    *,
    bf16_router: bool = True,
    num_replicas: int = 0,
    has_spec_slot: bool = False,
    replica_cost_mode: str = "tiles",
    fuse_qkv_epi: bool = True,
    has_rope: bool = True,
    has_free_params: bool = False,
) -> Callable:
    """Return cached TorchTPU custom op for Step (b) ``[chip_b1 + muse_sparse + chip_b2]``."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"b1_moe_b2 superkernel requires world_size in {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    eff_rope = bool(fuse_qkv_epi and has_rope)
    key = (
        int(world_size),
        bool(bf16_router),
        int(num_replicas),
        bool(has_spec_slot),
        str(replica_cost_mode),
        bool(fuse_qkv_epi),
        eff_rope,
        bool(has_free_params),
    )
    op = _B1_MOE_B2_OPS.get(key)
    if op is not None:
        return op

    with _B1_MOE_LOCK:
        op = _B1_MOE_B2_OPS.get(key)
        if op is None:
            has_rep = num_replicas > 0
            tensor_names = (
                _B1_BASE_TENSOR_NAMES
                + _B2_TENSOR_NAMES
                + (("cos_sin_step",) if eff_rope else ())
                + (_REPLICA_TENSOR_NAMES if has_rep else ())
                + (("spec_home_slot",) if has_spec_slot else ())
                + (("free_params",) if has_free_params else ())
            )
            static_specs = (
                _B1_STATIC_SPECS
                + _B2_STATIC_SPECS
                + (_EPI_STATIC_SPECS if fuse_qkv_epi else ())
            )
            n_base = len(_B1_BASE_TENSOR_NAMES) + len(_B2_TENSOR_NAMES)
            num_outputs = 4 if fuse_qkv_epi else 3

            def _local_b1_moe_b2(*args):
                n_t = len(tensor_names)
                t_args = args[:n_t]
                s_args = args[n_t:]
                pos = n_base
                cs_arg = None
                if eff_rope:
                    cs_arg = t_args[pos]
                    pos += 1
                rep_args = (None, None, None, None, None)
                if has_rep:
                    rep_args = t_args[pos : pos + 5]
                    pos += 5
                spec_arg = None
                if has_spec_slot:
                    spec_arg = t_args[pos]
                    pos += 1
                fp_arg = t_args[pos] if has_free_params else None
                (
                    pa_eps,
                    pre_ff_eps,
                    pre_ff_sc,
                    rmul,
                    pre_exp_eps,
                    pre_exp_sc,
                    post_ff_eps,
                    post_ff_sc,
                    next_in_eps,
                    next_in_sc,
                ) = s_args[:10]
                epi_kw = {}
                if fuse_qkv_epi:
                    qkv_eps, q_sc, k_sc, v_sc = s_args[10:14]
                    epi_kw = {
                        "qkv_norm_eps": float(qkv_eps),
                        "query_scale": float(q_sc),
                        "k_scale": float(k_sc),
                        "v_scale": float(v_sc),
                    }
                return b1_moe_b2_chip_sharded_jax(
                    *t_args[:n_base],
                    cos_sin_step=cs_arg,
                    rep_w1=rep_args[0],
                    rep_s1=rep_args[1],
                    rep_w2=rep_args[2],
                    rep_s2=rep_args[3],
                    replica_meta=rep_args[4],
                    spec_home_slot=spec_arg,
                    free_params=fp_arg,
                    world_size=world_size,
                    pa_eps=float(pa_eps),
                    pre_ff_eps=float(pre_ff_eps),
                    pre_ff_scale_input=bool(pre_ff_sc),
                    router_multiplier=float(rmul),
                    pre_exp_eps=float(pre_exp_eps),
                    pre_exp_scale_input=bool(pre_exp_sc),
                    post_ff_eps=float(post_ff_eps),
                    post_ff_scale_input=bool(post_ff_sc),
                    next_in_eps=float(next_in_eps),
                    next_in_scale_input=bool(next_in_sc),
                    fuse_qkv_epi=bool(fuse_qkv_epi),
                    replica_cost_mode=str(replica_cost_mode),
                    **epi_kw,
                )

            rtag = "r16" if bf16_router else "r32"
            etag = f"_epi_{'rope' if eff_rope else 'nope'}" if fuse_qkv_epi else "_plain"
            stag = "_sp" if has_spec_slot else ""
            ftag = "_dfp" if has_free_params else ""
            op_name = (
                f"sglang_tpu_pallas::muse_w9_b1_moe_b2_tp{world_size}"
                f"_{rtag}_rep{num_replicas}_{replica_cost_mode}{stag}{ftag}{etag}"
            )
            op = _export_superkernel_op(
                op_name,
                world_size,
                tensor_names,
                static_specs,
                num_outputs,
                _local_b1_moe_b2,
            )
            _B1_MOE_B2_OPS[key] = op
    return op


def get_b1_moe_b2_replicated_b16_op(
    world_size: int,
    *,
    bf16_router: bool = True,
    num_replicas: int = 0,
    has_spec_slot: bool = False,
    replica_cost_mode: str = "tiles",
    has_rope: bool = True,
    has_free_params: bool = False,
) -> Callable:
    """Return cached TorchTPU custom op for replicated ``B=16`` Step (b) superkernel."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"b1_moe_b2_b16 superkernel requires world_size in {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    key = (
        int(world_size),
        bool(bf16_router),
        int(num_replicas),
        bool(has_spec_slot),
        str(replica_cost_mode),
        bool(has_rope),
        bool(has_free_params),
    )
    op = _B1_MOE_B2_B16_OPS.get(key)
    if op is not None:
        return op

    with _B1_MOE_LOCK:
        op = _B1_MOE_B2_B16_OPS.get(key)
        if op is None:
            has_rep = num_replicas > 0
            tensor_names = (
                _B1_BASE_TENSOR_NAMES
                + _B2_TENSOR_NAMES
                + (("cos_sin_step",) if has_rope else ())
                + (_REPLICA_TENSOR_NAMES if has_rep else ())
                + (("spec_home_slot",) if has_spec_slot else ())
                + (("free_params",) if has_free_params else ())
            )
            static_specs = _B1_STATIC_SPECS + _B2_STATIC_SPECS + _EPI_STATIC_SPECS
            n_base = len(_B1_BASE_TENSOR_NAMES) + len(_B2_TENSOR_NAMES)

            def _local_rep_b16(*args):
                n_t = len(tensor_names)
                t_args = args[:n_t]
                s_args = args[n_t:]
                pos = n_base
                cs_arg = None
                if has_rope:
                    cs_arg = t_args[pos]
                    pos += 1
                rep_args = (None, None, None, None, None)
                if has_rep:
                    rep_args = t_args[pos : pos + 5]
                    pos += 5
                spec_arg = None
                if has_spec_slot:
                    spec_arg = t_args[pos]
                    pos += 1
                fp_arg = t_args[pos] if has_free_params else None
                (
                    pa_eps,
                    pre_ff_eps,
                    pre_ff_sc,
                    rmul,
                    pre_exp_eps,
                    pre_exp_sc,
                    post_ff_eps,
                    post_ff_sc,
                    next_in_eps,
                    next_in_sc,
                    qkv_eps,
                    q_sc,
                    k_sc,
                    v_sc,
                ) = s_args[:14]
                return b1_moe_b2_replicated_b16_jax(
                    *t_args[:n_base],
                    cos_sin_step=cs_arg,
                    rep_w1=rep_args[0],
                    rep_s1=rep_args[1],
                    rep_w2=rep_args[2],
                    rep_s2=rep_args[3],
                    replica_meta=rep_args[4],
                    spec_home_slot=spec_arg,
                    free_params=fp_arg,
                    world_size=world_size,
                    pa_eps=float(pa_eps),
                    pre_ff_eps=float(pre_ff_eps),
                    pre_ff_scale_input=bool(pre_ff_sc),
                    router_multiplier=float(rmul),
                    pre_exp_eps=float(pre_exp_eps),
                    pre_exp_scale_input=bool(pre_exp_sc),
                    post_ff_eps=float(post_ff_eps),
                    post_ff_scale_input=bool(post_ff_sc),
                    next_in_eps=float(next_in_eps),
                    next_in_scale_input=bool(next_in_sc),
                    fuse_qkv_epi=True,
                    qkv_norm_eps=float(qkv_eps),
                    query_scale=float(q_sc),
                    k_scale=float(k_sc),
                    v_scale=float(v_sc),
                    replica_cost_mode=str(replica_cost_mode),
                )

            rtag = "r16" if bf16_router else "r32"
            etag = f"_{'rope' if has_rope else 'nope'}"
            stag = "_sp" if has_spec_slot else ""
            ftag = "_dfp" if has_free_params else ""
            op_name = (
                f"sglang_tpu_pallas::muse_w9_b1_moe_b2_rep_b16_tp{world_size}"
                f"_{rtag}_rep{num_replicas}_{replica_cost_mode}{stag}{ftag}{etag}"
            )
            op = _export_superkernel_op(
                op_name,
                world_size,
                tensor_names,
                static_specs,
                4,
                _local_rep_b16,
            )
            _B1_MOE_B2_B16_OPS[key] = op
    return op


def get_b1_moe_b2_rpad_chip_sharded_op(
    world_size: int,
    *,
    bf16_router: bool = True,
    num_replicas: int = 0,
    has_spec_slot: bool = False,
    replica_cost_mode: str = "tiles",
    has_rope: bool = True,
    has_free_params: bool = False,
    has_layer_idx: bool = False,
    kv_pack_dense2: bool = False,
    rpa_mode_str: str = "",
) -> Callable:
    """Return cached TorchTPU custom op for Step (c) ``[chip_b1 + muse_sparse(3d) + chip_b2 + RPAd]``."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"Step (c) superkernel requires world_size in {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    key = (
        int(world_size),
        bool(bf16_router),
        int(num_replicas),
        bool(has_spec_slot),
        str(replica_cost_mode),
        bool(has_rope),
        bool(has_free_params),
        bool(has_layer_idx),
        bool(kv_pack_dense2),
        str(rpa_mode_str),
    )
    op = _B1_MOE_B2_RPAD_OPS.get(key)
    if op is not None:
        return op

    with _B1_MOE_LOCK:
        op = _B1_MOE_B2_RPAD_OPS.get(key)
        if op is None:
            has_rep = num_replicas > 0
            rpad_names = (
                "kv_cache_fused",
                "kv_lens",
                "page_indices",
                "cu_q_lens",
                "cu_kv_lens",
                "distribution",
            )
            tensor_names = (
                _B1_BASE_TENSOR_NAMES
                + _B2_TENSOR_NAMES
                + rpad_names
                + (("cos_sin_step",) if has_rope else ())
                + (_REPLICA_TENSOR_NAMES if has_rep else ())
                + (("spec_home_slot",) if has_spec_slot else ())
                + (("free_params",) if has_free_params else ())
                + (("layer_idx",) if has_layer_idx else ())
            )
            rpad_static_specs = (
                ("rpa_sm_scale", float),
                ("rpa_sliding_window", int),
                ("rpa_k_scale", float),
                ("rpa_v_scale", float),
                ("rpa_out_norm_eps", float),
            )
            static_specs = (
                _B1_STATIC_SPECS + _B2_STATIC_SPECS + _EPI_STATIC_SPECS + rpad_static_specs
            )
            n_base = len(_B1_BASE_TENSOR_NAMES) + len(_B2_TENSOR_NAMES)

            def _local_step_c(*args):
                n_t = len(tensor_names)
                t_args = args[:n_t]
                s_args = args[n_t:]
                pos = n_base
                kv_cache_arg, kv_lens_arg, pg_idx_arg, cu_q_arg, cu_kv_arg, dist_arg = t_args[
                    pos : pos + 6
                ]
                pos += 6
                cs_arg = None
                if has_rope:
                    cs_arg = t_args[pos]
                    pos += 1
                rep_args = (None, None, None, None, None)
                if has_rep:
                    rep_args = t_args[pos : pos + 5]
                    pos += 5
                spec_arg = None
                if has_spec_slot:
                    spec_arg = t_args[pos]
                    pos += 1
                fp_arg = None
                if has_free_params:
                    fp_arg = t_args[pos]
                    pos += 1
                lidx_arg = t_args[pos] if has_layer_idx else None
                (
                    pa_eps,
                    pre_ff_eps,
                    pre_ff_sc,
                    rmul,
                    pre_exp_eps,
                    pre_exp_sc,
                    post_ff_eps,
                    post_ff_sc,
                    next_in_eps,
                    next_in_sc,
                    qkv_eps,
                    q_sc,
                    k_sc,
                    v_sc,
                    rpa_sm_sc,
                    rpa_sw,
                    rpa_k_sc,
                    rpa_v_sc,
                    rpa_norm_eps,
                ) = s_args[:19]
                return b1_moe_b2_rpad_chip_sharded_jax(
                    *t_args[:n_base],
                    kv_cache_arg,
                    kv_lens_arg,
                    pg_idx_arg,
                    cu_q_arg,
                    cu_kv_arg,
                    dist_arg,
                    cos_sin_step=cs_arg,
                    rep_w1=rep_args[0],
                    rep_s1=rep_args[1],
                    rep_w2=rep_args[2],
                    rep_s2=rep_args[3],
                    replica_meta=rep_args[4],
                    spec_home_slot=spec_arg,
                    free_params=fp_arg,
                    layer_idx=lidx_arg,
                    world_size=world_size,
                    pa_eps=float(pa_eps),
                    pre_ff_eps=float(pre_ff_eps),
                    pre_ff_scale_input=bool(pre_ff_sc),
                    router_multiplier=float(rmul),
                    pre_exp_eps=float(pre_exp_eps),
                    pre_exp_scale_input=bool(pre_exp_sc),
                    post_ff_eps=float(post_ff_eps),
                    post_ff_scale_input=bool(post_ff_sc),
                    next_in_eps=float(next_in_eps),
                    next_in_scale_input=bool(next_in_sc),
                    qkv_norm_eps=float(qkv_eps),
                    query_scale=float(q_sc),
                    k_scale=float(k_sc),
                    v_scale=float(v_sc),
                    rpa_sm_scale=float(rpa_sm_sc),
                    rpa_sliding_window=(int(rpa_sw) if int(rpa_sw) > 0 else None),
                    rpa_k_scale=float(rpa_k_sc),
                    rpa_v_scale=float(rpa_v_sc),
                    rpa_out_norm_eps=float(rpa_norm_eps),
                    rpa_mode_str=(str(rpa_mode_str) if rpa_mode_str else None),
                    kv_pack_dense2=bool(kv_pack_dense2),
                    replica_cost_mode=str(replica_cost_mode),
                )

            rtag = "r16" if bf16_router else "r32"
            etag = f"_{'rope' if has_rope else 'nope'}"
            stag = "_sp" if has_spec_slot else ""
            ftag = "_dfp" if has_free_params else ""
            ltag = "_li" if has_layer_idx else ""
            d2tag = "_d2" if kv_pack_dense2 else ""
            mtag = (
                f"_{re.sub(r'[^a-zA-Z0-9_]+', '_', rpa_mode_str).strip('_')}"
                if rpa_mode_str
                else ""
            )
            op_name = (
                f"sglang_tpu_pallas::muse_w9_b1_moe_b2_rpad_tp{world_size}"
                f"_{rtag}_rep{num_replicas}_{replica_cost_mode}{stag}{ftag}{ltag}{d2tag}{mtag}{etag}"
            )
            op = _export_step_c_mutating_op(
                op_name,
                world_size,
                tensor_names,
                static_specs,
                _local_step_c,
            )
            _B1_MOE_B2_RPAD_OPS[key] = op
    return op


def get_b1_moe_chip_sharded_op(
    world_size: int,
    *,
    bf16_router: bool = True,
    num_replicas: int = 0,
    has_spec_slot: bool = False,
    replica_cost_mode: str = "tiles",
    return_sharded: bool = True,
    has_free_params: bool = False,
) -> Callable:
    """Return cached TorchTPU custom op for Step (a) ``[chip_b1 + muse_sparse]``."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"b1_moe superkernel requires world_size in {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    if world_size == 16 and not return_sharded:
        raise ValueError("b1_moe superkernel on world_size=16 requires return_sharded=True")
    key = (
        int(world_size),
        bool(bf16_router),
        int(num_replicas),
        bool(has_spec_slot),
        str(replica_cost_mode),
        bool(return_sharded),
        bool(has_free_params),
    )
    op = _B1_MOE_OPS.get(key)
    if op is not None:
        return op

    with _B1_MOE_LOCK:
        op = _B1_MOE_OPS.get(key)
        if op is None:
            has_rep = num_replicas > 0
            tensor_names = (
                _B1_BASE_TENSOR_NAMES
                + (_REPLICA_TENSOR_NAMES if has_rep else ())
                + (("spec_home_slot",) if has_spec_slot else ())
                + (("free_params",) if has_free_params else ())
            )
            n_base = len(_B1_BASE_TENSOR_NAMES)

            def _local_b1_moe(*args):
                n_t = len(tensor_names)
                t_args = args[:n_t]
                pa_eps, pre_ff_eps, pre_ff_sc, rmul, pre_exp_eps, pre_exp_sc = args[
                    n_t : n_t + 6
                ]
                pos = n_base
                rep_args = (None, None, None, None, None)
                if has_rep:
                    rep_args = t_args[pos : pos + 5]
                    pos += 5
                spec_arg = None
                if has_spec_slot:
                    spec_arg = t_args[pos]
                    pos += 1
                fp_arg = t_args[pos] if has_free_params else None
                return b1_moe_chip_sharded_jax(
                    *t_args[:n_base],
                    rep_w1=rep_args[0],
                    rep_s1=rep_args[1],
                    rep_w2=rep_args[2],
                    rep_s2=rep_args[3],
                    replica_meta=rep_args[4],
                    spec_home_slot=spec_arg,
                    free_params=fp_arg,
                    world_size=world_size,
                    pa_eps=float(pa_eps),
                    pre_ff_eps=float(pre_ff_eps),
                    pre_ff_scale_input=bool(pre_ff_sc),
                    router_multiplier=float(rmul),
                    pre_exp_eps=float(pre_exp_eps),
                    pre_exp_scale_input=bool(pre_exp_sc),
                    return_sharded=bool(return_sharded),
                    replica_cost_mode=str(replica_cost_mode),
                )

            rtag = "r16" if bf16_router else "r32"
            sh_tag = "_sh" if return_sharded else "_rep"
            stag = "_sp" if has_spec_slot else ""
            ftag = "_dfp" if has_free_params else ""
            op_name = (
                f"sglang_tpu_pallas::muse_w9_b1_moe_tp{world_size}"
                f"_{rtag}_rep{num_replicas}_{replica_cost_mode}{stag}{ftag}{sh_tag}"
            )
            op = _export_superkernel_op(
                op_name,
                world_size,
                tensor_names,
                _B1_STATIC_SPECS,
                2,
                _local_b1_moe,
            )
            _B1_MOE_OPS[key] = op
    return op


def prewarm_b1_moe_superkernel_ops(
    world_size: int,
    layer_configs: Iterable[tuple[int, bool, str, bool]],
    *,
    bridge_qkv_epi: bool = False,
    has_rope_variants: tuple[bool, ...] = (True,),
    has_free_params: bool = False,
    include_rep_b16: bool = False,
    step_c_configs: Iterable[tuple[bool, bool, str]] = (),
) -> None:
    """Eagerly register Step (a), Step (b), B=16, and Step (c) custom ops outside ``torch.compile``."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        return
    cfgs = list(layer_configs)
    step_c_list = list(step_c_configs)
    seen_b2: set[tuple] = set()
    seen_b1: set[tuple] = set()
    seen_b16: set[tuple] = set()
    seen_stepc: set[tuple] = set()
    for num_replicas, has_spec_slot, cost_mode, bf16_router in cfgs:
        b1_key = (
            int(num_replicas),
            bool(has_spec_slot),
            str(cost_mode),
            bool(bf16_router),
            bool(has_free_params),
        )
        if b1_key not in seen_b1:
            seen_b1.add(b1_key)
            get_b1_moe_chip_sharded_op(
                world_size,
                bf16_router=bf16_router,
                num_replicas=num_replicas,
                has_spec_slot=has_spec_slot,
                replica_cost_mode=cost_mode,
                return_sharded=True,
                has_free_params=has_free_params,
            )
        epi_modes = (False, True) if bridge_qkv_epi else (False,)
        for use_epi in epi_modes:
            rope_opts = has_rope_variants if use_epi else (False,)
            for r_flag in rope_opts:
                b2_key = (*b1_key, bool(use_epi), bool(use_epi and r_flag))
                if b2_key not in seen_b2:
                    seen_b2.add(b2_key)
                    get_b1_moe_b2_chip_sharded_op(
                        world_size,
                        bf16_router=bf16_router,
                        num_replicas=num_replicas,
                        has_spec_slot=has_spec_slot,
                        replica_cost_mode=cost_mode,
                        fuse_qkv_epi=use_epi,
                        has_rope=r_flag,
                        has_free_params=has_free_params,
                    )
        if include_rep_b16 and bridge_qkv_epi:
            for r_flag in has_rope_variants:
                b16_key = (*b1_key, bool(r_flag))
                if b16_key not in seen_b16:
                    seen_b16.add(b16_key)
                    get_b1_moe_b2_replicated_b16_op(
                        world_size,
                        bf16_router=bf16_router,
                        num_replicas=num_replicas,
                        has_spec_slot=has_spec_slot,
                        replica_cost_mode=cost_mode,
                        has_rope=r_flag,
                        has_free_params=has_free_params,
                    )
        if step_c_list and bridge_qkv_epi:
            for has_lidx, dense2, rpa_m in step_c_list:
                for r_flag in has_rope_variants:
                    sc_key = (
                        *b1_key,
                        bool(r_flag),
                        bool(has_lidx),
                        bool(dense2),
                        str(rpa_m),
                    )
                    if sc_key in seen_stepc:
                        continue
                    seen_stepc.add(sc_key)
                    get_b1_moe_b2_rpad_chip_sharded_op(
                        world_size,
                        bf16_router=bf16_router,
                        num_replicas=num_replicas,
                        has_spec_slot=has_spec_slot,
                        replica_cost_mode=cost_mode,
                        has_rope=r_flag,
                        has_free_params=has_free_params,
                        has_layer_idx=has_lidx,
                        kv_pack_dense2=dense2,
                        rpa_mode_str=rpa_m,
                    )
