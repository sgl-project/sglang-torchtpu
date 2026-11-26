# SPDX-License-Identifier: Apache-2.0
"""TorchTPU Custom Ops & Opt-In Decode Runner for Wave-9 `M2M` Decode Megakernel."""

# PEP 563 must stay off so cpu_export_jax_op sees concrete jax.Array annotations.

import functools
from threading import Lock
from typing import Callable, Optional

import torch

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kernels.decode_collective_bridges import (
    SUPPORTED_WORLD_SIZES,
    _get_tp_mesh,
    get_attn_to_moe_bridge_op,
    get_moe_to_attn_bridge_op,
)
from sglang_tpu.kernels.decode_collective_bridges_w8 import (
    get_attn_to_moe_chip_sharded_bridge_op,
    get_attn_to_moe_tok_sharded_bridge_op,
    get_moe_post_ag_final_norm_op,
    get_moe_post_chip_sharded_final_norm_op,
    get_moe_post_tok_sharded_final_norm_op,
    get_moe_to_attn_chip_sharded_bridge_op,
    get_moe_to_attn_qkv_epi_bridge_op,
    get_moe_to_attn_tok_sharded_bridge_op,
)
from sglang_tpu.muse_megakernel.decode.m2m.chip_sh_kernel import (
    l0_a2m_chip_sh_jax,
    m2m_chip_sh_jax,
)
from sglang_tpu.muse_megakernel.decode.m2m.kernel import (
    kv_write_batch_jax,
    l0_a2m_tok_sh_tp8_jax,
    m2m_tok_sh_tp8_jax,
)
from sglang_tpu.muse_megakernel.decode.m2m.rep_kernel import (
    l0_a2m_rep_jax,
    m2m_rep_jax,
)

_M2M_LOCK = Lock()
_M2M_TOK_OPS: dict[tuple[bool, bool], Callable] = {}
_L0_TOK_OPS: dict[tuple[bool, bool], Callable] = {}
_M2M_CHIP_OPS: dict[tuple[int, bool, bool, bool], Callable] = {}
_L0_CHIP_OPS: dict[tuple[int, bool, bool], Callable] = {}
_M2M_REP_OPS: dict[tuple[int, bool, bool, bool], Callable] = {}
_L0_REP_OPS: dict[tuple[int, bool, bool], Callable] = {}
_KVW_CALLABLES: dict[tuple, object] = {}
_KVW_REGISTERED: list[bool] = []


def _custom_op_device_types() -> Optional[list[str]]:
    try:
        torch._C._dispatch_key_for_device("tpu")
        return ["tpu"]
    except RuntimeError:
        return None


def _contig(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous() or (x._base is not None and x.storage_offset() != 0):
        return x.contiguous()
    return x


def _cpu_export_m2m_op(
    name: str,
    function: Callable[..., object],
    *,
    mesh,
    input_partition_specs,
    output_partition_specs,
):
    """Export an M2M custom op with zero-copy in-place `kv_cache_ro` aliasing (`#2615`)."""
    import inspect

    import jax
    from jax.sharding import NamedSharding
    from torch_tpu._internal.pallas import pallas as pallas_impl

    from sglang_tpu.pallas_export import tpu_info_for_cpu_export
    from sglang_tpu.pallas_jax_op import (
        _CpuExportJaxCallable,
        _jax_placeholders,
        _torch_placeholder,
    )

    class _M2MExportJaxCallable(_CpuExportJaxCallable):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__signature__ = self.__signature__.replace(
                return_annotation=tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            )

        def __call__(self, *args, **kwargs):
            res = super().__call__(*args, **kwargs)
            return res[:3]

    signature = inspect.signature(function, follow_wrapped=False)
    pallas_impl._verify_signature(signature)
    static_argnums = pallas_impl._infer_static_argnums(signature)
    kv_idx = list(signature.parameters.keys()).index("kv_cache_ro")
    donate_argnums_tuple = (kv_idx,)
    aliases = {kv_idx: 3}

    input_partition_specs = tuple(input_partition_specs or ())
    output_partition_specs = tuple(output_partition_specs or ())
    output_shardings = tuple(
        NamedSharding(mesh, spec) for spec in output_partition_specs
    )
    jit_function = jax.jit(
        function,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums_tuple,
        keep_unused=True,
        out_shardings=output_shardings,
    )
    trace_key = pallas_impl._get_kernel_invocation_key(
        name,
        [],
        {
            "static_argnums": static_argnums,
            "donate_argnums": donate_argnums_tuple,
            "output_partition_specs": tuple(map(str, output_partition_specs)),
        },
    )
    wrapped = _M2MExportJaxCallable(
        name=name,
        jit_fn=jit_function,
        trace_key=trace_key,
        mesh=mesh,
        input_partition_specs=input_partition_specs or None,
        static_argnums=static_argnums,
        donate_argnums=list(donate_argnums_tuple),
        input_output_aliases=aliases,
    )
    wrapped.sglang_output_shardings = output_shardings
    custom_op = torch.library.custom_op(
        name,
        wrapped,
        mutates_args=("kv_cache_ro",),
        device_types=_custom_op_device_types(),
    )

    def fake_function(*args, **kwargs):
        jax_args = _jax_placeholders(
            args,
            mesh=mesh,
            partition_specs=tuple(input_partition_specs),
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
        return unflat[:3]

    custom_op.register_fake(fake_function)
    return custom_op


def get_m2m_tok_sh_tp8_op(*, has_rope: bool, bf16_router: bool) -> Callable:
    key = (bool(has_rope), bool(bf16_router))
    op = _M2M_TOK_OPS.get(key)
    if op is not None:
        return op
    with _M2M_LOCK:
        op = _M2M_TOK_OPS.get(key)
        if op is not None:
            return op
        import jax
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        mesh = _get_tp_mesh(8)
        pre_specs = (
            P(),
            P(),
            P(),
            P(),
            P(),
            P("model", None),
            P("model", None),
            P(None, "model"),
            P("model"),
            P(),
            P(),
            P(),
            P(),
            P(None, "model"),
            P("model"),
            P(None, "model"),
            P("model"),
        )
        post_specs = (
            P("model"),
            P("model", None),
            P("model"),
            P(),
            P(),
            P(),
            P(None, "model"),
            P("model"),
            P(),
            P(),
        )
        out_specs = (P("model", None), P(), P(), P("model"))
        r_tag = "rope" if has_rope else "nope"
        rt_tag = "r16" if bf16_router else "r32"
        n_scalars = 15

        if has_rope:
            t_specs = (*pre_specs, P(), *post_specs)
            in_specs = (*t_specs, *([None] * n_scalars))

            def _wrapped_r(
                kv_lens: jax.Array,
                page_indices: jax.Array,
                cu_q_lens: jax.Array,
                cu_kv_lens: jax.Array,
                distribution: jax.Array,
                stream_shard: jax.Array,
                moe_out: jax.Array,
                w_post_w8k: jax.Array,
                s_post: jax.Array,
                post_ff_gain: jax.Array,
                pf_alpha: jax.Array,
                pf_beta: jax.Array,
                next_in_gain: jax.Array,
                next_w_qkv_w8k: jax.Array,
                next_s_qkv: jax.Array,
                next_w_gate_w8k: jax.Array,
                next_s_gate: jax.Array,
                cos_sin_step: jax.Array,
                kv_cache_ro: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_pair_w8k: jax.Array,
                w_pre_pair_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                post_ff_eps: float,
                post_ff_scale_input: bool,
                next_in_eps: float,
                next_in_scale_input: bool,
                qkv_norm_eps: float,
                query_scale: float,
                sm_scale: float,
                sliding_window_val: int,
                k_scale: float,
                v_scale: float,
                kv_pack_dense2: bool,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                sw = sliding_window_val if sliding_window_val > 0 else None
                fn = functools.partial(
                    m2m_tok_sh_tp8_jax,
                    world_size=8,
                    post_ff_eps=float(post_ff_eps),
                    post_ff_scale_input=bool(post_ff_scale_input),
                    next_in_eps=float(next_in_eps),
                    next_in_scale_input=bool(next_in_scale_input),
                    qkv_norm_eps=float(qkv_norm_eps),
                    query_scale=float(query_scale),
                    sm_scale=float(sm_scale),
                    sliding_window=sw,
                    k_scale=float(k_scale),
                    v_scale=float(v_scale),
                    kv_pack_dense2=bool(kv_pack_dense2),
                    pa_eps=float(pa_eps),
                    pre_ff_eps=float(pre_ff_eps),
                    pre_ff_scale_input=bool(pre_ff_scale_input),
                    router_multiplier=float(router_multiplier),
                    pre_exp_eps=float(pre_exp_eps),
                    pre_exp_scale_input=bool(pre_exp_scale_input),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    stream_shard,
                    moe_out,
                    w_post_w8k,
                    s_post,
                    post_ff_gain,
                    pf_alpha,
                    pf_beta,
                    next_in_gain,
                    next_w_qkv_w8k,
                    next_s_qkv,
                    next_w_gate_w8k,
                    next_s_gate,
                    cos_sin_step,
                    kv_cache_ro,
                    wo_w8k,
                    wo_scale,
                    pa_alpha,
                    pa_beta,
                    pre_ff_gain,
                    w_pre_pair_w8k,
                    w_pre_pair_scale,
                    w_router_km,
                    pre_exp_gain,
                )

            # Note: 17 static scalar args at tail
            in_specs = (*t_specs, *([None] * 17))
            op = _cpu_export_m2m_op(
                f"sglang_tpu_pallas::muse_w9_m2m_tok_{r_tag}_{rt_tag}_tp8",
                _wrapped_r,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
        else:
            t_specs = (*pre_specs, *post_specs)
            in_specs = (*t_specs, *([None] * 17))

            def _wrapped_n(
                kv_lens: jax.Array,
                page_indices: jax.Array,
                cu_q_lens: jax.Array,
                cu_kv_lens: jax.Array,
                distribution: jax.Array,
                stream_shard: jax.Array,
                moe_out: jax.Array,
                w_post_w8k: jax.Array,
                s_post: jax.Array,
                post_ff_gain: jax.Array,
                pf_alpha: jax.Array,
                pf_beta: jax.Array,
                next_in_gain: jax.Array,
                next_w_qkv_w8k: jax.Array,
                next_s_qkv: jax.Array,
                next_w_gate_w8k: jax.Array,
                next_s_gate: jax.Array,
                kv_cache_ro: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_pair_w8k: jax.Array,
                w_pre_pair_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                post_ff_eps: float,
                post_ff_scale_input: bool,
                next_in_eps: float,
                next_in_scale_input: bool,
                qkv_norm_eps: float,
                query_scale: float,
                sm_scale: float,
                sliding_window_val: int,
                k_scale: float,
                v_scale: float,
                kv_pack_dense2: bool,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                sw = sliding_window_val if sliding_window_val > 0 else None

                def _local(*t_args):
                    return m2m_tok_sh_tp8_jax(
                        *t_args[:17],
                        None,
                        *t_args[17:],
                        world_size=8,
                        post_ff_eps=float(post_ff_eps),
                        post_ff_scale_input=bool(post_ff_scale_input),
                        next_in_eps=float(next_in_eps),
                        next_in_scale_input=bool(next_in_scale_input),
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        sm_scale=float(sm_scale),
                        sliding_window=sw,
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                        kv_pack_dense2=bool(kv_pack_dense2),
                        pa_eps=float(pa_eps),
                        pre_ff_eps=float(pre_ff_eps),
                        pre_ff_scale_input=bool(pre_ff_scale_input),
                        router_multiplier=float(router_multiplier),
                        pre_exp_eps=float(pre_exp_eps),
                        pre_exp_scale_input=bool(pre_exp_scale_input),
                    )

                return jax.shard_map(
                    _local, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    stream_shard,
                    moe_out,
                    w_post_w8k,
                    s_post,
                    post_ff_gain,
                    pf_alpha,
                    pf_beta,
                    next_in_gain,
                    next_w_qkv_w8k,
                    next_s_qkv,
                    next_w_gate_w8k,
                    next_s_gate,
                    kv_cache_ro,
                    wo_w8k,
                    wo_scale,
                    pa_alpha,
                    pa_beta,
                    pre_ff_gain,
                    w_pre_pair_w8k,
                    w_pre_pair_scale,
                    w_router_km,
                    pre_exp_gain,
                )

            op = _cpu_export_m2m_op(
                f"sglang_tpu_pallas::muse_w9_m2m_tok_{r_tag}_{rt_tag}_tp8",
                _wrapped_n,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
        _M2M_TOK_OPS[key] = op
    return op


def get_l0_a2m_tok_sh_tp8_op(*, has_rope: bool, bf16_router: bool) -> Callable:
    key = (bool(has_rope), bool(bf16_router))
    op = _L0_TOK_OPS.get(key)
    if op is not None:
        return op
    with _M2M_LOCK:
        op = _L0_TOK_OPS.get(key)
        if op is not None:
            return op
        import jax
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        mesh = _get_tp_mesh(8)
        pre_specs = (
            P(),
            P(),
            P(),
            P(),
            P(),
            P("model", None),
            P(),
            P(None, "model"),
            P("model"),
            P(None, "model"),
            P("model"),
        )
        post_specs = (
            P("model"),
            P("model", None),
            P("model"),
            P(),
            P(),
            P(),
            P(None, "model"),
            P("model"),
            P(),
            P(),
        )
        out_specs = (P("model", None), P(), P(), P("model"))
        r_tag = "rope" if has_rope else "nope"
        rt_tag = "r16" if bf16_router else "r32"

        if has_rope:
            t_specs = (*pre_specs, P(), *post_specs)
            in_specs = (*t_specs, *([None] * 13))

            def _wrapped_r(
                kv_lens: jax.Array,
                page_indices: jax.Array,
                cu_q_lens: jax.Array,
                cu_kv_lens: jax.Array,
                distribution: jax.Array,
                stream_shard: jax.Array,
                attn_in: jax.Array,
                w_qkv_w8k: jax.Array,
                s_qkv: jax.Array,
                w_gate_w8k: jax.Array,
                s_gate: jax.Array,
                cos_sin_step: jax.Array,
                kv_cache_ro: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_pair_w8k: jax.Array,
                w_pre_pair_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                qkv_norm_eps: float,
                query_scale: float,
                sm_scale: float,
                sliding_window_val: int,
                k_scale: float,
                v_scale: float,
                kv_pack_dense2: bool,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                sw = sliding_window_val if sliding_window_val > 0 else None
                fn = functools.partial(
                    l0_a2m_tok_sh_tp8_jax,
                    world_size=8,
                    qkv_norm_eps=float(qkv_norm_eps),
                    query_scale=float(query_scale),
                    sm_scale=float(sm_scale),
                    sliding_window=sw,
                    k_scale=float(k_scale),
                    v_scale=float(v_scale),
                    kv_pack_dense2=bool(kv_pack_dense2),
                    pa_eps=float(pa_eps),
                    pre_ff_eps=float(pre_ff_eps),
                    pre_ff_scale_input=bool(pre_ff_scale_input),
                    router_multiplier=float(router_multiplier),
                    pre_exp_eps=float(pre_exp_eps),
                    pre_exp_scale_input=bool(pre_exp_scale_input),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    stream_shard,
                    attn_in,
                    w_qkv_w8k,
                    s_qkv,
                    w_gate_w8k,
                    s_gate,
                    cos_sin_step,
                    kv_cache_ro,
                    wo_w8k,
                    wo_scale,
                    pa_alpha,
                    pa_beta,
                    pre_ff_gain,
                    w_pre_pair_w8k,
                    w_pre_pair_scale,
                    w_router_km,
                    pre_exp_gain,
                )

            op = _cpu_export_m2m_op(
                f"sglang_tpu_pallas::muse_w9_l0_a2m_tok_{r_tag}_{rt_tag}_tp8",
                _wrapped_r,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
        else:
            t_specs = (*pre_specs, *post_specs)
            in_specs = (*t_specs, *([None] * 13))

            def _wrapped_n(
                kv_lens: jax.Array,
                page_indices: jax.Array,
                cu_q_lens: jax.Array,
                cu_kv_lens: jax.Array,
                distribution: jax.Array,
                stream_shard: jax.Array,
                attn_in: jax.Array,
                w_qkv_w8k: jax.Array,
                s_qkv: jax.Array,
                w_gate_w8k: jax.Array,
                s_gate: jax.Array,
                kv_cache_ro: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_pair_w8k: jax.Array,
                w_pre_pair_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                qkv_norm_eps: float,
                query_scale: float,
                sm_scale: float,
                sliding_window_val: int,
                k_scale: float,
                v_scale: float,
                kv_pack_dense2: bool,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                sw = sliding_window_val if sliding_window_val > 0 else None

                def _local(*t_args):
                    return l0_a2m_tok_sh_tp8_jax(
                        *t_args[:11],
                        None,
                        *t_args[11:],
                        world_size=8,
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        sm_scale=float(sm_scale),
                        sliding_window=sw,
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                        kv_pack_dense2=bool(kv_pack_dense2),
                        pa_eps=float(pa_eps),
                        pre_ff_eps=float(pre_ff_eps),
                        pre_ff_scale_input=bool(pre_ff_scale_input),
                        router_multiplier=float(router_multiplier),
                        pre_exp_eps=float(pre_exp_eps),
                        pre_exp_scale_input=bool(pre_exp_scale_input),
                    )

                return jax.shard_map(
                    _local, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    stream_shard,
                    attn_in,
                    w_qkv_w8k,
                    s_qkv,
                    w_gate_w8k,
                    s_gate,
                    kv_cache_ro,
                    wo_w8k,
                    wo_scale,
                    pa_alpha,
                    pa_beta,
                    pre_ff_gain,
                    w_pre_pair_w8k,
                    w_pre_pair_scale,
                    w_router_km,
                    pre_exp_gain,
                )

            op = _cpu_export_m2m_op(
                f"sglang_tpu_pallas::muse_w9_l0_a2m_tok_{r_tag}_{rt_tag}_tp8",
                _wrapped_n,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
        _L0_TOK_OPS[key] = op
    return op


def get_m2m_rep_op(
    world_size: int, *, sharded_moe_in: bool, has_rope: bool, bf16_router: bool
) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    key = (int(world_size), bool(sharded_moe_in), bool(has_rope), bool(bf16_router))
    op = _M2M_REP_OPS.get(key)
    if op is not None:
        return op
    with _M2M_LOCK:
        op = _M2M_REP_OPS.get(key)
        if op is not None:
            return op
        import jax
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        mesh = _get_tp_mesh(world_size)
        moe_sp = P("model", None) if sharded_moe_in else P()
        pre_specs = (
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            moe_sp,
            P(None, "model"),
            P("model"),
            P(),
            P(),
            P(),
            P(),
            P(None, "model"),
            P("model"),
            P(None, "model"),
            P("model"),
        )
        post_specs = (
            P("model"),
            P("model", None),
            P("model"),
            P(),
            P(),
            P(),
            P(None, "model"),
            P("model"),
            P(),
            P(),
        )
        out_specs = (P(), P(), P(), P("model"))
        sh_tag = "sh" if sharded_moe_in else "rep"
        r_tag = "rope" if has_rope else "nope"
        rt_tag = "r16" if bf16_router else "r32"

        if has_rope:
            t_specs = (*pre_specs, P(), *post_specs)
            in_specs = (*t_specs, *([None] * 17))

            def _wrapped_r(
                kv_lens: jax.Array,
                page_indices: jax.Array,
                cu_q_lens: jax.Array,
                cu_kv_lens: jax.Array,
                distribution: jax.Array,
                stream: jax.Array,
                moe_out: jax.Array,
                w_post_w8k: jax.Array,
                s_post: jax.Array,
                post_ff_gain: jax.Array,
                pf_alpha: jax.Array,
                pf_beta: jax.Array,
                next_in_gain: jax.Array,
                next_w_qkv_w8k: jax.Array,
                next_s_qkv: jax.Array,
                next_w_gate_w8k: jax.Array,
                next_s_gate: jax.Array,
                cos_sin_step: jax.Array,
                kv_cache_ro: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_w8k: jax.Array,
                w_pre_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                post_ff_eps: float,
                post_ff_scale_input: bool,
                next_in_eps: float,
                next_in_scale_input: bool,
                qkv_norm_eps: float,
                query_scale: float,
                sm_scale: float,
                sliding_window_val: int,
                k_scale: float,
                v_scale: float,
                kv_pack_dense2: bool,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                sw = sliding_window_val if sliding_window_val > 0 else None
                fn = functools.partial(
                    m2m_rep_jax,
                    world_size=world_size,
                    sharded_moe_in=bool(sharded_moe_in),
                    post_ff_eps=float(post_ff_eps),
                    post_ff_scale_input=bool(post_ff_scale_input),
                    next_in_eps=float(next_in_eps),
                    next_in_scale_input=bool(next_in_scale_input),
                    qkv_norm_eps=float(qkv_norm_eps),
                    query_scale=float(query_scale),
                    sm_scale=float(sm_scale),
                    sliding_window=sw,
                    k_scale=float(k_scale),
                    v_scale=float(v_scale),
                    kv_pack_dense2=bool(kv_pack_dense2),
                    pa_eps=float(pa_eps),
                    pre_ff_eps=float(pre_ff_eps),
                    pre_ff_scale_input=bool(pre_ff_scale_input),
                    router_multiplier=float(router_multiplier),
                    pre_exp_eps=float(pre_exp_eps),
                    pre_exp_scale_input=bool(pre_exp_scale_input),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    stream,
                    moe_out,
                    w_post_w8k,
                    s_post,
                    post_ff_gain,
                    pf_alpha,
                    pf_beta,
                    next_in_gain,
                    next_w_qkv_w8k,
                    next_s_qkv,
                    next_w_gate_w8k,
                    next_s_gate,
                    cos_sin_step,
                    kv_cache_ro,
                    wo_w8k,
                    wo_scale,
                    pa_alpha,
                    pa_beta,
                    pre_ff_gain,
                    w_pre_w8k,
                    w_pre_scale,
                    w_router_km,
                    pre_exp_gain,
                )

            op = _cpu_export_m2m_op(
                f"sglang_tpu_pallas::muse_w9_m2m_{sh_tag}_{r_tag}_{rt_tag}_tp{world_size}",
                _wrapped_r,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
        else:
            t_specs = (*pre_specs, *post_specs)
            in_specs = (*t_specs, *([None] * 17))

            def _wrapped_n(
                kv_lens: jax.Array,
                page_indices: jax.Array,
                cu_q_lens: jax.Array,
                cu_kv_lens: jax.Array,
                distribution: jax.Array,
                stream: jax.Array,
                moe_out: jax.Array,
                w_post_w8k: jax.Array,
                s_post: jax.Array,
                post_ff_gain: jax.Array,
                pf_alpha: jax.Array,
                pf_beta: jax.Array,
                next_in_gain: jax.Array,
                next_w_qkv_w8k: jax.Array,
                next_s_qkv: jax.Array,
                next_w_gate_w8k: jax.Array,
                next_s_gate: jax.Array,
                kv_cache_ro: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_w8k: jax.Array,
                w_pre_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                post_ff_eps: float,
                post_ff_scale_input: bool,
                next_in_eps: float,
                next_in_scale_input: bool,
                qkv_norm_eps: float,
                query_scale: float,
                sm_scale: float,
                sliding_window_val: int,
                k_scale: float,
                v_scale: float,
                kv_pack_dense2: bool,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                sw = sliding_window_val if sliding_window_val > 0 else None

                def _local(*t_args):
                    return m2m_rep_jax(
                        *t_args[:17],
                        None,
                        *t_args[17:],
                        world_size=world_size,
                        sharded_moe_in=bool(sharded_moe_in),
                        post_ff_eps=float(post_ff_eps),
                        post_ff_scale_input=bool(post_ff_scale_input),
                        next_in_eps=float(next_in_eps),
                        next_in_scale_input=bool(next_in_scale_input),
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        sm_scale=float(sm_scale),
                        sliding_window=sw,
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                        kv_pack_dense2=bool(kv_pack_dense2),
                        pa_eps=float(pa_eps),
                        pre_ff_eps=float(pre_ff_eps),
                        pre_ff_scale_input=bool(pre_ff_scale_input),
                        router_multiplier=float(router_multiplier),
                        pre_exp_eps=float(pre_exp_eps),
                        pre_exp_scale_input=bool(pre_exp_scale_input),
                    )

                return jax.shard_map(
                    _local, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    stream,
                    moe_out,
                    w_post_w8k,
                    s_post,
                    post_ff_gain,
                    pf_alpha,
                    pf_beta,
                    next_in_gain,
                    next_w_qkv_w8k,
                    next_s_qkv,
                    next_w_gate_w8k,
                    next_s_gate,
                    kv_cache_ro,
                    wo_w8k,
                    wo_scale,
                    pa_alpha,
                    pa_beta,
                    pre_ff_gain,
                    w_pre_w8k,
                    w_pre_scale,
                    w_router_km,
                    pre_exp_gain,
                )

            op = _cpu_export_m2m_op(
                f"sglang_tpu_pallas::muse_w9_m2m_{sh_tag}_{r_tag}_{rt_tag}_tp{world_size}",
                _wrapped_n,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
        _M2M_REP_OPS[key] = op
    return op


def get_l0_a2m_rep_op(world_size: int, *, has_rope: bool, bf16_router: bool) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    key = (int(world_size), bool(has_rope), bool(bf16_router))
    op = _L0_REP_OPS.get(key)
    if op is not None:
        return op
    with _M2M_LOCK:
        op = _L0_REP_OPS.get(key)
        if op is not None:
            return op
        import jax
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        mesh = _get_tp_mesh(world_size)
        pre_specs = (
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(None, "model"),
            P("model"),
            P(None, "model"),
            P("model"),
        )
        post_specs = (
            P("model"),
            P("model", None),
            P("model"),
            P(),
            P(),
            P(),
            P(None, "model"),
            P("model"),
            P(),
            P(),
        )
        out_specs = (P(), P(), P(), P("model"))
        r_tag = "rope" if has_rope else "nope"
        rt_tag = "r16" if bf16_router else "r32"

        if has_rope:
            t_specs = (*pre_specs, P(), *post_specs)
            in_specs = (*t_specs, *([None] * 13))

            def _wrapped_r(
                kv_lens: jax.Array,
                page_indices: jax.Array,
                cu_q_lens: jax.Array,
                cu_kv_lens: jax.Array,
                distribution: jax.Array,
                stream: jax.Array,
                attn_in: jax.Array,
                w_qkv_w8k: jax.Array,
                s_qkv: jax.Array,
                w_gate_w8k: jax.Array,
                s_gate: jax.Array,
                cos_sin_step: jax.Array,
                kv_cache_ro: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_w8k: jax.Array,
                w_pre_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                qkv_norm_eps: float,
                query_scale: float,
                sm_scale: float,
                sliding_window_val: int,
                k_scale: float,
                v_scale: float,
                kv_pack_dense2: bool,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                sw = sliding_window_val if sliding_window_val > 0 else None
                fn = functools.partial(
                    l0_a2m_rep_jax,
                    world_size=world_size,
                    qkv_norm_eps=float(qkv_norm_eps),
                    query_scale=float(query_scale),
                    sm_scale=float(sm_scale),
                    sliding_window=sw,
                    k_scale=float(k_scale),
                    v_scale=float(v_scale),
                    kv_pack_dense2=bool(kv_pack_dense2),
                    pa_eps=float(pa_eps),
                    pre_ff_eps=float(pre_ff_eps),
                    pre_ff_scale_input=bool(pre_ff_scale_input),
                    router_multiplier=float(router_multiplier),
                    pre_exp_eps=float(pre_exp_eps),
                    pre_exp_scale_input=bool(pre_exp_scale_input),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    stream,
                    attn_in,
                    w_qkv_w8k,
                    s_qkv,
                    w_gate_w8k,
                    s_gate,
                    cos_sin_step,
                    kv_cache_ro,
                    wo_w8k,
                    wo_scale,
                    pa_alpha,
                    pa_beta,
                    pre_ff_gain,
                    w_pre_w8k,
                    w_pre_scale,
                    w_router_km,
                    pre_exp_gain,
                )

            op = _cpu_export_m2m_op(
                f"sglang_tpu_pallas::muse_w9_l0_a2m_rep_{r_tag}_{rt_tag}_tp{world_size}",
                _wrapped_r,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
        else:
            t_specs = (*pre_specs, *post_specs)
            in_specs = (*t_specs, *([None] * 13))

            def _wrapped_n(
                kv_lens: jax.Array,
                page_indices: jax.Array,
                cu_q_lens: jax.Array,
                cu_kv_lens: jax.Array,
                distribution: jax.Array,
                stream: jax.Array,
                attn_in: jax.Array,
                w_qkv_w8k: jax.Array,
                s_qkv: jax.Array,
                w_gate_w8k: jax.Array,
                s_gate: jax.Array,
                kv_cache_ro: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_w8k: jax.Array,
                w_pre_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                qkv_norm_eps: float,
                query_scale: float,
                sm_scale: float,
                sliding_window_val: int,
                k_scale: float,
                v_scale: float,
                kv_pack_dense2: bool,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                sw = sliding_window_val if sliding_window_val > 0 else None

                def _local(*t_args):
                    return l0_a2m_rep_jax(
                        *t_args[:11],
                        None,
                        *t_args[11:],
                        world_size=world_size,
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        sm_scale=float(sm_scale),
                        sliding_window=sw,
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                        kv_pack_dense2=bool(kv_pack_dense2),
                        pa_eps=float(pa_eps),
                        pre_ff_eps=float(pre_ff_eps),
                        pre_ff_scale_input=bool(pre_ff_scale_input),
                        router_multiplier=float(router_multiplier),
                        pre_exp_eps=float(pre_exp_eps),
                        pre_exp_scale_input=bool(pre_exp_scale_input),
                    )

                return jax.shard_map(
                    _local, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    stream,
                    attn_in,
                    w_qkv_w8k,
                    s_qkv,
                    w_gate_w8k,
                    s_gate,
                    kv_cache_ro,
                    wo_w8k,
                    wo_scale,
                    pa_alpha,
                    pa_beta,
                    pre_ff_gain,
                    w_pre_w8k,
                    w_pre_scale,
                    w_router_km,
                    pre_exp_gain,
                )

            op = _cpu_export_m2m_op(
                f"sglang_tpu_pallas::muse_w9_l0_a2m_rep_{r_tag}_{rt_tag}_tp{world_size}",
                _wrapped_n,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
        _L0_REP_OPS[key] = op
    return op


def get_m2m_chip_sh_op(
    world_size: int, *, sharded_moe_in: bool, has_rope: bool, bf16_router: bool
) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    key = (int(world_size), bool(sharded_moe_in), bool(has_rope), bool(bf16_router))
    op = _M2M_CHIP_OPS.get(key)
    if op is not None:
        return op
    with _M2M_LOCK:
        op = _M2M_CHIP_OPS.get(key)
        if op is not None:
            return op
        import jax
        from jax.sharding import PartitionSpec as P

        mesh = _get_tp_mesh(world_size)
        moe_sp = P("model", None) if sharded_moe_in else P()
        pre_specs = (
            P(),
            P(),
            P(),
            P(),
            P(),
            P("model", None),
            moe_sp,
            P(None, "model"),
            P("model"),
            P(),
            P(),
            P(),
            P(),
            P(None, "model"),
            P("model"),
            P(None, "model"),
            P("model"),
        )
        post_specs = (
            P("model"),
            P("model", None),
            P("model"),
            P(),
            P(),
            P(),
            P(None, "model"),
            P("model"),
            P(),
            P(),
        )
        out_specs = (P("model", None), P(), P(), P("model"))
        sh_tag = "sh" if sharded_moe_in else "full"
        r_tag = "rope" if has_rope else "nope"
        rt_tag = "r16" if bf16_router else "r32"
        n_scalars = 17

        if has_rope:
            t_specs = (*pre_specs, P(), *post_specs)
            in_specs = (*t_specs, *([None] * n_scalars))

            def _wrapped_r(
                kv_lens: jax.Array,
                page_indices: jax.Array,
                cu_q_lens: jax.Array,
                cu_kv_lens: jax.Array,
                distribution: jax.Array,
                stream_chip: jax.Array,
                moe_out: jax.Array,
                w_post_w8k: jax.Array,
                s_post: jax.Array,
                post_ff_gain: jax.Array,
                pf_alpha: jax.Array,
                pf_beta: jax.Array,
                next_in_gain: jax.Array,
                next_w_qkv_w8k: jax.Array,
                next_s_qkv: jax.Array,
                next_w_gate_w8k: jax.Array,
                next_s_gate: jax.Array,
                cos_sin_step: jax.Array,
                kv_cache_ro: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_pair_w8k: jax.Array,
                w_pre_pair_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                post_ff_eps: float,
                post_ff_scale_input: bool,
                next_in_eps: float,
                next_in_scale_input: bool,
                qkv_norm_eps: float,
                query_scale: float,
                sm_scale: float,
                sliding_window_val: int,
                k_scale: float,
                v_scale: float,
                kv_pack_dense2: bool,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                sw = sliding_window_val if sliding_window_val > 0 else None
                fn = functools.partial(
                    m2m_chip_sh_jax,
                    world_size=world_size,
                    sharded_moe_in=bool(sharded_moe_in),
                    post_ff_eps=float(post_ff_eps),
                    post_ff_scale_input=bool(post_ff_scale_input),
                    next_in_eps=float(next_in_eps),
                    next_in_scale_input=bool(next_in_scale_input),
                    qkv_norm_eps=float(qkv_norm_eps),
                    query_scale=float(query_scale),
                    sm_scale=float(sm_scale),
                    sliding_window=sw,
                    k_scale=float(k_scale),
                    v_scale=float(v_scale),
                    kv_pack_dense2=bool(kv_pack_dense2),
                    pa_eps=float(pa_eps),
                    pre_ff_eps=float(pre_ff_eps),
                    pre_ff_scale_input=bool(pre_ff_scale_input),
                    router_multiplier=float(router_multiplier),
                    pre_exp_eps=float(pre_exp_eps),
                    pre_exp_scale_input=bool(pre_exp_scale_input),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    stream_chip,
                    moe_out,
                    w_post_w8k,
                    s_post,
                    post_ff_gain,
                    pf_alpha,
                    pf_beta,
                    next_in_gain,
                    next_w_qkv_w8k,
                    next_s_qkv,
                    next_w_gate_w8k,
                    next_s_gate,
                    cos_sin_step,
                    kv_cache_ro,
                    wo_w8k,
                    wo_scale,
                    pa_alpha,
                    pa_beta,
                    pre_ff_gain,
                    w_pre_pair_w8k,
                    w_pre_pair_scale,
                    w_router_km,
                    pre_exp_gain,
                )

            op = _cpu_export_m2m_op(
                f"sglang_tpu_pallas::muse_w9_m2m_chip_{sh_tag}_{r_tag}_{rt_tag}_tp{world_size}",
                _wrapped_r,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
        else:
            t_specs = (*pre_specs, *post_specs)
            in_specs = (*t_specs, *([None] * n_scalars))

            def _wrapped_n(
                kv_lens: jax.Array,
                page_indices: jax.Array,
                cu_q_lens: jax.Array,
                cu_kv_lens: jax.Array,
                distribution: jax.Array,
                stream_chip: jax.Array,
                moe_out: jax.Array,
                w_post_w8k: jax.Array,
                s_post: jax.Array,
                post_ff_gain: jax.Array,
                pf_alpha: jax.Array,
                pf_beta: jax.Array,
                next_in_gain: jax.Array,
                next_w_qkv_w8k: jax.Array,
                next_s_qkv: jax.Array,
                next_w_gate_w8k: jax.Array,
                next_s_gate: jax.Array,
                kv_cache_ro: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_pair_w8k: jax.Array,
                w_pre_pair_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                post_ff_eps: float,
                post_ff_scale_input: bool,
                next_in_eps: float,
                next_in_scale_input: bool,
                qkv_norm_eps: float,
                query_scale: float,
                sm_scale: float,
                sliding_window_val: int,
                k_scale: float,
                v_scale: float,
                kv_pack_dense2: bool,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                sw = sliding_window_val if sliding_window_val > 0 else None

                def _local(*t_args):
                    return m2m_chip_sh_jax(
                        *t_args[:17],
                        None,
                        *t_args[17:],
                        world_size=world_size,
                        sharded_moe_in=bool(sharded_moe_in),
                        post_ff_eps=float(post_ff_eps),
                        post_ff_scale_input=bool(post_ff_scale_input),
                        next_in_eps=float(next_in_eps),
                        next_in_scale_input=bool(next_in_scale_input),
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        sm_scale=float(sm_scale),
                        sliding_window=sw,
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                        kv_pack_dense2=bool(kv_pack_dense2),
                        pa_eps=float(pa_eps),
                        pre_ff_eps=float(pre_ff_eps),
                        pre_ff_scale_input=bool(pre_ff_scale_input),
                        router_multiplier=float(router_multiplier),
                        pre_exp_eps=float(pre_exp_eps),
                        pre_exp_scale_input=bool(pre_exp_scale_input),
                    )

                return jax.shard_map(
                    _local, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    stream_chip,
                    moe_out,
                    w_post_w8k,
                    s_post,
                    post_ff_gain,
                    pf_alpha,
                    pf_beta,
                    next_in_gain,
                    next_w_qkv_w8k,
                    next_s_qkv,
                    next_w_gate_w8k,
                    next_s_gate,
                    kv_cache_ro,
                    wo_w8k,
                    wo_scale,
                    pa_alpha,
                    pa_beta,
                    pre_ff_gain,
                    w_pre_pair_w8k,
                    w_pre_pair_scale,
                    w_router_km,
                    pre_exp_gain,
                )

            op = _cpu_export_m2m_op(
                f"sglang_tpu_pallas::muse_w9_m2m_chip_{sh_tag}_{r_tag}_{rt_tag}_tp{world_size}",
                _wrapped_n,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
        _M2M_CHIP_OPS[key] = op
    return op


def get_l0_a2m_chip_sh_op(world_size: int, *, has_rope: bool, bf16_router: bool) -> Callable:
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(f"Unsupported world_size={world_size}")
    key = (int(world_size), bool(has_rope), bool(bf16_router))
    op = _L0_CHIP_OPS.get(key)
    if op is not None:
        return op
    with _M2M_LOCK:
        op = _L0_CHIP_OPS.get(key)
        if op is not None:
            return op
        import jax
        from jax.sharding import PartitionSpec as P

        mesh = _get_tp_mesh(world_size)
        pre_specs = (
            P(),
            P(),
            P(),
            P(),
            P(),
            P("model", None),
            P(),
            P(None, "model"),
            P("model"),
            P(None, "model"),
            P("model"),
        )
        post_specs = (
            P("model"),
            P("model", None),
            P("model"),
            P(),
            P(),
            P(),
            P(None, "model"),
            P("model"),
            P(),
            P(),
        )
        out_specs = (P("model", None), P(), P(), P("model"))
        r_tag = "rope" if has_rope else "nope"
        rt_tag = "r16" if bf16_router else "r32"
        n_scalars = 13

        if has_rope:
            t_specs = (*pre_specs, P(), *post_specs)
            in_specs = (*t_specs, *([None] * n_scalars))

            def _wrapped_r(
                kv_lens: jax.Array,
                page_indices: jax.Array,
                cu_q_lens: jax.Array,
                cu_kv_lens: jax.Array,
                distribution: jax.Array,
                stream_chip: jax.Array,
                attn_in: jax.Array,
                w_qkv_w8k: jax.Array,
                s_qkv: jax.Array,
                w_gate_w8k: jax.Array,
                s_gate: jax.Array,
                cos_sin_step: jax.Array,
                kv_cache_ro: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_pair_w8k: jax.Array,
                w_pre_pair_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                qkv_norm_eps: float,
                query_scale: float,
                sm_scale: float,
                sliding_window_val: int,
                k_scale: float,
                v_scale: float,
                kv_pack_dense2: bool,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                sw = sliding_window_val if sliding_window_val > 0 else None
                fn = functools.partial(
                    l0_a2m_chip_sh_jax,
                    world_size=world_size,
                    qkv_norm_eps=float(qkv_norm_eps),
                    query_scale=float(query_scale),
                    sm_scale=float(sm_scale),
                    sliding_window=sw,
                    k_scale=float(k_scale),
                    v_scale=float(v_scale),
                    kv_pack_dense2=bool(kv_pack_dense2),
                    pa_eps=float(pa_eps),
                    pre_ff_eps=float(pre_ff_eps),
                    pre_ff_scale_input=bool(pre_ff_scale_input),
                    router_multiplier=float(router_multiplier),
                    pre_exp_eps=float(pre_exp_eps),
                    pre_exp_scale_input=bool(pre_exp_scale_input),
                )
                return jax.shard_map(
                    fn, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    stream_chip,
                    attn_in,
                    w_qkv_w8k,
                    s_qkv,
                    w_gate_w8k,
                    s_gate,
                    cos_sin_step,
                    kv_cache_ro,
                    wo_w8k,
                    wo_scale,
                    pa_alpha,
                    pa_beta,
                    pre_ff_gain,
                    w_pre_pair_w8k,
                    w_pre_pair_scale,
                    w_router_km,
                    pre_exp_gain,
                )

            op = _cpu_export_m2m_op(
                f"sglang_tpu_pallas::muse_w9_l0_a2m_chip_{r_tag}_{rt_tag}_tp{world_size}",
                _wrapped_r,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
        else:
            t_specs = (*pre_specs, *post_specs)
            in_specs = (*t_specs, *([None] * n_scalars))

            def _wrapped_n(
                kv_lens: jax.Array,
                page_indices: jax.Array,
                cu_q_lens: jax.Array,
                cu_kv_lens: jax.Array,
                distribution: jax.Array,
                stream_chip: jax.Array,
                attn_in: jax.Array,
                w_qkv_w8k: jax.Array,
                s_qkv: jax.Array,
                w_gate_w8k: jax.Array,
                s_gate: jax.Array,
                kv_cache_ro: jax.Array,
                wo_w8k: jax.Array,
                wo_scale: jax.Array,
                pa_alpha: jax.Array,
                pa_beta: jax.Array,
                pre_ff_gain: jax.Array,
                w_pre_pair_w8k: jax.Array,
                w_pre_pair_scale: jax.Array,
                w_router_km: jax.Array,
                pre_exp_gain: jax.Array,
                qkv_norm_eps: float,
                query_scale: float,
                sm_scale: float,
                sliding_window_val: int,
                k_scale: float,
                v_scale: float,
                kv_pack_dense2: bool,
                pa_eps: float,
                pre_ff_eps: float,
                pre_ff_scale_input: bool,
                router_multiplier: float,
                pre_exp_eps: float,
                pre_exp_scale_input: bool,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                sw = sliding_window_val if sliding_window_val > 0 else None

                def _local(*t_args):
                    return l0_a2m_chip_sh_jax(
                        *t_args[:11],
                        None,
                        *t_args[11:],
                        world_size=world_size,
                        qkv_norm_eps=float(qkv_norm_eps),
                        query_scale=float(query_scale),
                        sm_scale=float(sm_scale),
                        sliding_window=sw,
                        k_scale=float(k_scale),
                        v_scale=float(v_scale),
                        kv_pack_dense2=bool(kv_pack_dense2),
                        pa_eps=float(pa_eps),
                        pre_ff_eps=float(pre_ff_eps),
                        pre_ff_scale_input=bool(pre_ff_scale_input),
                        router_multiplier=float(router_multiplier),
                        pre_exp_eps=float(pre_exp_eps),
                        pre_exp_scale_input=bool(pre_exp_scale_input),
                    )

                return jax.shard_map(
                    _local, mesh=mesh, in_specs=t_specs, out_specs=out_specs, check_vma=False
                )(
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    distribution,
                    stream_chip,
                    attn_in,
                    w_qkv_w8k,
                    s_qkv,
                    w_gate_w8k,
                    s_gate,
                    kv_cache_ro,
                    wo_w8k,
                    wo_scale,
                    pa_alpha,
                    pa_beta,
                    pre_ff_gain,
                    w_pre_pair_w8k,
                    w_pre_pair_scale,
                    w_router_km,
                    pre_exp_gain,
                )

            op = _cpu_export_m2m_op(
                f"sglang_tpu_pallas::muse_w9_l0_a2m_chip_{r_tag}_{rt_tag}_tp{world_size}",
                _wrapped_n,
                mesh=mesh,
                input_partition_specs=in_specs,
                output_partition_specs=out_specs,
            )
        _L0_CHIP_OPS[key] = op
    return op


def _build_kvw_callable(n_layers: int, n_bufs: int, layer_to_buf_idx: tuple[int, ...]):
    key = (n_layers, n_bufs, layer_to_buf_idx)
    cached = _KVW_CALLABLES.get(key)
    if cached is not None:
        return cached

    import jax
    from torch_tpu._internal.pallas import pallas as pallas_impl

    from sglang_tpu.pallas_jax_op import _CpuExportJaxCallable

    donate = tuple(range(2 * n_layers, 2 * n_layers + n_bufs))
    aliases = {2 * n_layers + j: j for j in range(n_bufs)}

    def fn(*xs: jax.Array) -> tuple:
        dsts = xs[:n_layers]
        rows = xs[n_layers : 2 * n_layers]
        bufs = xs[2 * n_layers : 2 * n_layers + n_bufs]
        return tuple(kv_write_batch_jax(rows, dsts, bufs, layer_to_buf_idx=layer_to_buf_idx))

    jit_fn = jax.jit(fn, donate_argnums=donate, keep_unused=True)
    tag = f"muse_w9_m2m_kvw_l{n_layers}_u{n_bufs}"
    trace_key = pallas_impl._get_kernel_invocation_key(
        f"{tag}_{hash(layer_to_buf_idx)}", [], {"donate": donate}
    )
    callable_ = _CpuExportJaxCallable(
        name=f"jax_{tag}",
        jit_fn=jit_fn,
        trace_key=trace_key,
        mesh=None,
        input_partition_specs=None,
        static_argnums=(),
        donate_argnums=donate,
        input_output_aliases=aliases,
    )
    callable_.sglang_output_shardings = ()
    _KVW_CALLABLES[key] = callable_
    return callable_


def register_m2m_kv_write_op() -> None:
    if _KVW_REGISTERED:
        return
    with _M2M_LOCK:
        if _KVW_REGISTERED:
            return

        @torch.library.custom_op(
            "sglang_tpu_pallas::muse_w9_m2m_kv_write",
            mutates_args=("kv_buffers",),
            device_types=_custom_op_device_types(),
        )
        def muse_w9_m2m_kv_write(
            kvw_rows: list[torch.Tensor],
            kvw_dst: list[torch.Tensor],
            kv_buffers: list[torch.Tensor],
            layer_to_buf_idx: list[int],
        ) -> None:
            from sglang_tpu.pallas_export import tpu_info_for_cpu_export

            callable_ = _build_kvw_callable(
                len(kvw_rows), len(kv_buffers), tuple(int(x) for x in layer_to_buf_idx)
            )
            with tpu_info_for_cpu_export():
                callable_(*kvw_dst, *kvw_rows, *kv_buffers)

        @muse_w9_m2m_kv_write.register_fake
        def _(kvw_rows, kvw_dst, kv_buffers, layer_to_buf_idx):
            return None

        _KVW_REGISTERED.append(True)


def prewarm_m2m_decode_ops(
    world_size: int, *, tok_sharded: bool = False, chip_sharded: bool = True
) -> None:
    """Eagerly register all M2M custom ops outside `torch.compile` (zero Dynamo lock risk!)."""
    if world_size not in SUPPORTED_WORLD_SIZES:
        return
    from sglang_tpu.decode_qkv_epilogue_tpu import get_decode_rope_step_op

    get_decode_rope_step_op()
    register_m2m_kv_write_op()
    for r_flag in (False, True):
        for r16 in (False, True):
            get_l0_a2m_rep_op(world_size, has_rope=r_flag, bf16_router=r16)
            for sh in (False, True):
                get_m2m_rep_op(
                    world_size, sharded_moe_in=sh, has_rope=r_flag, bf16_router=r16
                )
            if chip_sharded:
                get_l0_a2m_chip_sh_op(world_size, has_rope=r_flag, bf16_router=r16)
                for sh in (False, True):
                    get_m2m_chip_sh_op(
                        world_size, sharded_moe_in=sh, has_rope=r_flag, bf16_router=r16
                    )
            if world_size == 8 and tok_sharded:
                get_l0_a2m_tok_sh_tp8_op(has_rope=r_flag, bf16_router=r16)
                get_m2m_tok_sh_tp8_op(has_rope=r_flag, bf16_router=r16)


def maybe_forward_m2m_decode(
    model,
    *,
    positions: torch.Tensor,
    stream: torch.Tensor,
    carried_attn_input: Optional[torch.Tensor],
    cos_sin_step: Optional[torch.Tensor],
    forward_batch,
    tok_sharded_stream: bool,
    chip_sharded_stream: bool = False,
) -> Optional[torch.Tensor]:
    """Execute per-layer `M2M` decode across SWA (all ctx) & Full (`<= max_full_ctx`) layers."""
    if (
        not model.pp_group.is_first_rank
        or not model.pp_group.is_last_rank
        or bool(model.layers_to_capture)
        or int(model.__dict__.get("_decode_col_bridges", 0)) != 3
    ):
        return None

    from sglang_tpu.attention_pallas_rpa_base import _fused_kv_operands
    from sglang_tpu.decode_qkv_epilogue_tpu import (
        _resolve_decode_attn_backend,
        get_decode_rope_step_op,
    )

    backend = _resolve_decode_attn_backend(forward_batch)
    md = getattr(backend, "forward_metadata", None) if backend is not None else None
    if (
        backend is None
        or md is None
        or getattr(md, "decode_query_size", 0) != 1
        or getattr(backend, "_dcp", 0) != 1
        or getattr(md, "cp_write_segments", None) is not None
        or getattr(backend, "kv_cache_quant_dtype", None) != torch.float8_e4m3fn
    ):
        return None

    from sglang_tpu.models.muse_spark import _layer_tp_size

    n_tok = int(positions.shape[0])
    layer0 = model.layers[model.start_layer]
    tp = _layer_tp_size(layer0.self_attn.o_proj)
    if tp not in (8, 16) or n_tok not in (16, 32, 64, 128):
        return None
    if not chip_sharded_stream and not tok_sharded_stream:
        if int(stream.shape[0]) * (tp // 2) == n_tok:
            chip_sharded_stream = True
    if tok_sharded_stream and chip_sharded_stream:
        return None
    expected_stream_rows = (
        (n_tok // tp)
        if tok_sharded_stream
        else ((n_tok // (tp // 2)) if chip_sharded_stream else n_tok)
    )
    if int(stream.shape[0]) != expected_stream_rows:
        return None
    if tok_sharded_stream and (tp != 8 or n_tok not in (64, 128)):
        return None
    if chip_sharded_stream and n_tok not in (64, 128):
        return None
    if (tok_sharded_stream or chip_sharded_stream) and carried_attn_input is None:
        return None
    need_pair_w = tok_sharded_stream or chip_sharded_stream

    # Verify all layers carry prepared Bridge-1b/2b FP8 buffers and 5D non-bundle KV pools.
    kv_lens = _contig(backend._kv_lens(forward_batch))
    cu_q = _contig(md.cu_q_lens)
    cu_kv = _contig(md.cu_kv_lens)
    dense2 = bool(getattr(backend, "_kv_pack_dense2", False))
    if bool(getattr(backend, "_kv_pack_bd4", False)):
        return None
    max_full_ctx = int(tpu_envs.SGLANG_TPU_MUSE_MK_M2M_MAX_FULL_CTX.get())
    has_shared_prefix = getattr(md, "decode_shared_prefix_pages", None) is not None

    layer_meta: list[tuple] = []
    pool_obj = backend.token_to_kv_pool
    for l_idx in range(model.start_layer, model.end_layer):
        lyr = model.layers[l_idx]
        sa = lyr.self_attn
        mlp = lyr.mlp
        has_wpost = "_w7_wpost_w8k" in mlp._buffers or (
            chip_sharded_stream and "_w9_wpost_pair_w8k" in mlp._buffers
        )
        if (
            "_w7_wqkv_w8k" not in sa._buffers
            or "_w7_wgate_w8k" not in sa._buffers
            or sa._o_proj_w8k is None
            or not has_wpost
            or "_w7_wrouter_km" not in mlp._buffers
            or (need_pair_w and "_w8_wpre_pair_w8k" not in mlp._buffers)
            or (not need_pair_w and "_w7_wpre_w8k" not in mlp._buffers)
        ):
            return None
        kv_buf, kv_l_idx = _fused_kv_operands(pool_obj, sa.attn.layer_id)
        if kv_l_idx is not None or kv_buf.ndim != 5:
            return None
        sw_raw = getattr(sa.attn, "sliding_window_size", None)
        if sw_raw is not None and sw_raw <= 0:
            sw = None
        elif sw_raw is not None:
            sw = int(sw_raw) + 1
        else:
            sw = None
        is_swa = sw is not None
        sub_pool = (
            getattr(pool_obj, "swa_kv_pool" if is_swa else "full_kv_pool", None)
            if pool_obj is not None
            else None
        )
        if sub_pool is not None and bool(getattr(sub_pool, "kv_pack_bd4", False)):
            return None
        lyr_dense2 = (
            bool(sub_pool.kv_pack_dense2)
            if sub_pool is not None and hasattr(sub_pool, "kv_pack_dense2")
            else dense2
        )
        pg_idx = _contig(backend._layer_page_indices(sa.attn.layer_id, is_swa_layer=is_swa))
        dist = _contig(backend._layer_distribution(is_swa_layer=is_swa))
        phys_rows = int(kv_buf.shape[1])
        page_size = phys_rows * 2 if lyr_dense2 else phys_rows
        pages_per_seq = int(pg_idx.shape[0]) // max(1, int(kv_lens.shape[0]))
        per_seq_cap = pages_per_seq * page_size
        m2m_ok = is_swa or (per_seq_cap <= max_full_ctx and not has_shared_prefix)
        layer_meta.append((lyr, sa, mlp, kv_buf, sw, pg_idx, dist, lyr_dense2, m2m_ok))

    if not any(m[-1] for m in layer_meta):
        return None

    if cos_sin_step is None and getattr(model.rotary_emb, "cos_sin_pairs", None) is not None:
        cos_sin_step = get_decode_rope_step_op()(
            positions.to(torch.int32), model.rotary_emb.cos_sin_pairs
        )
    cs_kw = {"cos_sin_step": cos_sin_step} if cos_sin_step is not None else {}

    # Step 1: Layer-0 Attention -> MoE(0) inputs (in-place overlapped KV scatter!)
    lyr0, sa0, mlp0, kv0, sw0, pg0, dist0, d2_0, m2m0 = layer_meta[0]
    if carried_attn_input is not None:
        attn_in0 = _contig(carried_attn_input)
    else:
        attn_in0 = _contig(lyr0.input_layernorm(stream).to(torch.bfloat16))
    stream = _contig(stream)

    has_rope0 = sa0.rotary_emb is not None
    r16_0 = mlp0._buffers["_w7_wrouter_km"].dtype == torch.bfloat16
    k_sc0 = float(getattr(sa0.attn, "k_scale_float", None) or 1.0)
    v_sc0 = float(getattr(sa0.attn, "v_scale_float", None) or 1.0)
    pa_a0 = lyr0.post_attention_residual_gate.alpha
    pa_b0 = lyr0.post_attention_residual_gate.beta
    pff_g0 = lyr0.pre_feedforward_layernorm.gain()
    peg_0 = mlp0.pre_expert_norm.gain()

    if m2m0:
        sw_val0 = int(sw0) if sw0 is not None else -1
        cs_t0 = (_contig(cos_sin_step),) if has_rope0 else ()
        if chip_sharded_stream:
            l0_op = get_l0_a2m_chip_sh_op(tp, has_rope=has_rope0, bf16_router=r16_0)
            w_pre0, s_pre0 = mlp0._buffers["_w8_wpre_pair_w8k"], mlp0._buffers["_w8_spre_pair"]
        elif tok_sharded_stream:
            l0_op = get_l0_a2m_tok_sh_tp8_op(has_rope=has_rope0, bf16_router=r16_0)
            w_pre0, s_pre0 = mlp0._buffers["_w8_wpre_pair_w8k"], mlp0._buffers["_w8_spre_pair"]
        else:
            l0_op = get_l0_a2m_rep_op(tp, has_rope=has_rope0, bf16_router=r16_0)
            w_pre0, s_pre0 = mlp0._buffers["_w7_wpre_w8k"], mlp0._buffers["_w7_spre"]
        stream, scaled_logits, expert_in = l0_op(
            kv_lens,
            pg0,
            cu_q,
            cu_kv,
            dist0,
            stream,
            attn_in0,
            sa0._buffers["_w7_wqkv_w8k"],
            sa0._buffers["_w7_sqkv"],
            sa0._buffers["_w7_wgate_w8k"],
            sa0._buffers["_w7_sgate"],
            *cs_t0,
            kv0,
            sa0._o_proj_w8k,
            sa0._o_proj_scale,
            pa_a0,
            pa_b0,
            pff_g0,
            w_pre0,
            s_pre0,
            mlp0._buffers["_w7_wrouter_km"],
            peg_0,
            float(sa0.norm_eps),
            float(sa0.query_scale),
            float(sa0.attn.scaling),
            sw_val0,
            k_sc0,
            v_sc0,
            d2_0,
            float(lyr0.post_attention_layernorm.eps),
            float(lyr0.pre_feedforward_layernorm.eps),
            bool(lyr0.pre_feedforward_layernorm.scale_input),
            float(mlp0.router_multiplier),
            float(mlp0.pre_expert_norm.eps),
            bool(mlp0.pre_expert_norm.scale_input),
        )
    else:
        gated0 = sa0(positions, attn_in0, forward_batch, return_gated_attn=True, **cs_kw)
        if chip_sharded_stream:
            b1_0_op = get_attn_to_moe_chip_sharded_bridge_op(tp, bf16_router=r16_0)
            w_pre0, s_pre0 = mlp0._buffers["_w8_wpre_pair_w8k"], mlp0._buffers["_w8_spre_pair"]
        elif tok_sharded_stream:
            b1_0_op = get_attn_to_moe_tok_sharded_bridge_op(tp, bf16_router=r16_0)
            w_pre0, s_pre0 = mlp0._buffers["_w8_wpre_pair_w8k"], mlp0._buffers["_w8_spre_pair"]
        else:
            b1_0_op = get_attn_to_moe_bridge_op(tp, bf16_router=r16_0)
            w_pre0, s_pre0 = mlp0._buffers["_w7_wpre_w8k"], mlp0._buffers["_w7_spre"]
        stream, scaled_logits, expert_in = b1_0_op(
            stream,
            gated0,
            sa0._o_proj_w8k,
            sa0._o_proj_scale,
            pa_a0,
            pa_b0,
            pff_g0,
            w_pre0,
            s_pre0,
            mlp0._buffers["_w7_wrouter_km"],
            peg_0,
            float(lyr0.post_attention_layernorm.eps),
            float(lyr0.pre_feedforward_layernorm.eps),
            bool(lyr0.pre_feedforward_layernorm.scale_input),
            float(mlp0.router_multiplier),
            float(mlp0.pre_expert_norm.eps),
            bool(mlp0.pre_expert_norm.scale_input),
        )

    # Step 2: Inter-layer MoE(i) -> M2M(i -> i+1) relay across all 62 layers
    n_layers = len(layer_meta)
    for i in range(n_layers):
        lyr, sa, mlp, _kv_i, _sw_i, _pg_i, _dist_i, _d2_i, _m2m_i = layer_meta[i]
        can_sh_fn = getattr(mlp, "can_return_sharded_moe_out", None)
        has_sh_op = (
            bool(can_sh_fn(n_tok, decode=True))
            if callable(can_sh_fn)
            else (
                mlp.__dict__.get("_fused_decode_sharded_op") is not None
                or mlp.__dict__.get("_fused_decode_v2_sharded_op") is not None
            )
        )
        want_sharded = n_tok >= 8 * tp and (n_tok // tp) % 8 == 0 and has_sh_op
        moe_out = mlp(
            None,
            None,
            decode=True,
            return_unprojected_moe_out=True,
            return_sharded_moe_out=want_sharded,
            precomputed_moe_inputs=(scaled_logits, expert_in),
        )
        sharded_in = (
            moe_out.shape[0] * tp == n_tok
            and moe_out.shape[0] >= 8
            and moe_out.shape[0] % 8 == 0
        )
        post_ff_gain = lyr.post_feedforward_layernorm.gain()
        pf_a = lyr.post_feedforward_residual_gate.alpha
        pf_b = lyr.post_feedforward_residual_gate.beta
        if chip_sharded_stream:
            w_post = mlp._buffers.get(
                "_w9_wpost_pair_w8k", mlp._buffers.get("_w7_wpost_w8k")
            )
            s_post = mlp._buffers.get("_w9_spost_pair", mlp._buffers.get("_w7_spost"))
        else:
            w_post = mlp._buffers["_w7_wpost_w8k"]
            s_post = mlp._buffers["_w7_spost"]

        if i + 1 == n_layers:
            fin_gain = model.norm.gain()
            if chip_sharded_stream:
                final_hidden = get_moe_post_chip_sharded_final_norm_op(
                    tp, sharded_moe_in=sharded_in, return_stream_too=False
                )(
                    stream,
                    moe_out,
                    w_post,
                    s_post,
                    post_ff_gain,
                    pf_a,
                    pf_b,
                    fin_gain,
                    float(lyr.post_feedforward_layernorm.eps),
                    bool(lyr.post_feedforward_layernorm.scale_input),
                    float(model.norm.eps),
                    bool(model.norm.scale_input),
                )
            elif tok_sharded_stream:
                final_hidden = get_moe_post_tok_sharded_final_norm_op(
                    tp, return_stream_too=False
                )(
                    stream,
                    moe_out,
                    w_post,
                    s_post,
                    post_ff_gain,
                    pf_a,
                    pf_b,
                    fin_gain,
                    float(lyr.post_feedforward_layernorm.eps),
                    bool(lyr.post_feedforward_layernorm.scale_input),
                    float(model.norm.eps),
                    bool(model.norm.scale_input),
                )
            else:
                final_hidden = get_moe_post_ag_final_norm_op(
                    tp, sharded_moe_in=sharded_in
                )(
                    stream,
                    moe_out,
                    w_post,
                    s_post,
                    post_ff_gain,
                    pf_a,
                    pf_b,
                    fin_gain,
                    float(lyr.post_feedforward_layernorm.eps),
                    bool(lyr.post_feedforward_layernorm.scale_input),
                    float(model.norm.eps),
                    bool(model.norm.scale_input),
                )
            return final_hidden

        n_lyr, n_sa, n_mlp, n_kv, n_sw, n_pg, n_dist, n_d2, n_m2m = layer_meta[i + 1]
        next_in_gain = n_lyr.input_layernorm.gain()
        n_has_rope = n_sa.rotary_emb is not None
        n_r16 = n_mlp._buffers["_w7_wrouter_km"].dtype == torch.bfloat16
        n_k_sc = float(getattr(n_sa.attn, "k_scale_float", None) or 1.0)
        n_v_sc = float(getattr(n_sa.attn, "v_scale_float", None) or 1.0)
        n_pa_a = n_lyr.post_attention_residual_gate.alpha
        n_pa_b = n_lyr.post_attention_residual_gate.beta
        n_pff_g = n_lyr.pre_feedforward_layernorm.gain()
        n_peg = n_mlp.pre_expert_norm.gain()

        if n_m2m:
            n_sw_val = int(n_sw) if n_sw is not None else -1
            cs_tn = (_contig(cos_sin_step),) if n_has_rope else ()
            if chip_sharded_stream:
                m2m_op = get_m2m_chip_sh_op(
                    tp, sharded_moe_in=sharded_in, has_rope=n_has_rope, bf16_router=n_r16
                )
                n_wpre, n_spre = (
                    n_mlp._buffers["_w8_wpre_pair_w8k"],
                    n_mlp._buffers["_w8_spre_pair"],
                )
            elif tok_sharded_stream:
                m2m_op = get_m2m_tok_sh_tp8_op(has_rope=n_has_rope, bf16_router=n_r16)
                n_wpre, n_spre = (
                    n_mlp._buffers["_w8_wpre_pair_w8k"],
                    n_mlp._buffers["_w8_spre_pair"],
                )
            else:
                m2m_op = get_m2m_rep_op(
                    tp, sharded_moe_in=sharded_in, has_rope=n_has_rope, bf16_router=n_r16
                )
                n_wpre, n_spre = (
                    n_mlp._buffers["_w7_wpre_w8k"],
                    n_mlp._buffers["_w7_spre"],
                )
            stream, scaled_logits, expert_in = m2m_op(
                kv_lens,
                n_pg,
                cu_q,
                cu_kv,
                n_dist,
                stream,
                moe_out,
                w_post,
                s_post,
                post_ff_gain,
                pf_a,
                pf_b,
                next_in_gain,
                n_sa._buffers["_w7_wqkv_w8k"],
                n_sa._buffers["_w7_sqkv"],
                n_sa._buffers["_w7_wgate_w8k"],
                n_sa._buffers["_w7_sgate"],
                *cs_tn,
                n_kv,
                n_sa._o_proj_w8k,
                n_sa._o_proj_scale,
                n_pa_a,
                n_pa_b,
                n_pff_g,
                n_wpre,
                n_spre,
                n_mlp._buffers["_w7_wrouter_km"],
                n_peg,
                float(lyr.post_feedforward_layernorm.eps),
                bool(lyr.post_feedforward_layernorm.scale_input),
                float(n_lyr.input_layernorm.eps),
                bool(n_lyr.input_layernorm.scale_input),
                float(n_sa.norm_eps),
                float(n_sa.query_scale),
                float(n_sa.attn.scaling),
                n_sw_val,
                n_k_sc,
                n_v_sc,
                n_d2,
                float(n_lyr.post_attention_layernorm.eps),
                float(n_lyr.pre_feedforward_layernorm.eps),
                bool(n_lyr.pre_feedforward_layernorm.scale_input),
                float(n_mlp.router_multiplier),
                float(n_mlp.pre_expert_norm.eps),
                bool(n_mlp.pre_expert_norm.scale_input),
            )
            continue

        # Long-context Full layer (`streamdyn` / `shared16`): run standalone b2 -> RPAd -> b1!
        if chip_sharded_stream:
            tp16_epi_ok = tp == 16 and bool(
                getattr(tpu_envs, "SGLANG_TPU_MUSE_DECODE_TP16_QKV_EPI", None)
                is not None
                and tpu_envs.SGLANG_TPU_MUSE_DECODE_TP16_QKV_EPI.get()
            )
            use_epi = (
                (tp == 8 or tp16_epi_ok)
                and bool(lyr.__dict__.get("_w8_bridge_qkv_epi", False))
                and n_sa.decode_qkv_epilogue
            )
            if use_epi:
                epi_op = get_moe_to_attn_chip_sharded_bridge_op(
                    tp, sharded_moe_in=sharded_in, fuse_qkv_epi=True, has_rope=n_has_rope
                )
                cs_t = (_contig(cos_sin_step),) if n_has_rope else ()
                stream, nq5, nkv, ng = epi_op(
                    stream,
                    moe_out,
                    w_post,
                    s_post,
                    post_ff_gain,
                    pf_a,
                    pf_b,
                    next_in_gain,
                    n_sa._buffers["_w7_wqkv_w8k"],
                    n_sa._buffers["_w7_sqkv"],
                    n_sa._buffers["_w7_wgate_w8k"],
                    n_sa._buffers["_w7_sgate"],
                    *cs_t,
                    float(lyr.post_feedforward_layernorm.eps),
                    bool(lyr.post_feedforward_layernorm.scale_input),
                    float(n_lyr.input_layernorm.eps),
                    bool(n_lyr.input_layernorm.scale_input),
                    float(n_sa.norm_eps),
                    float(n_sa.query_scale),
                    n_k_sc,
                    n_v_sc,
                )
                n_gated = n_sa(
                    positions,
                    None,
                    forward_batch,
                    precomputed_prepared_qkvg=(nq5, nkv, ng),
                    return_gated_attn=True,
                    **cs_kw,
                )
            else:
                b2_op = get_moe_to_attn_chip_sharded_bridge_op(
                    tp, sharded_moe_in=sharded_in, fuse_qkv_epi=False
                )
                stream, nqkv, ng = b2_op(
                    stream,
                    moe_out,
                    w_post,
                    s_post,
                    post_ff_gain,
                    pf_a,
                    pf_b,
                    next_in_gain,
                    n_sa._buffers["_w7_wqkv_w8k"],
                    n_sa._buffers["_w7_sqkv"],
                    n_sa._buffers["_w7_wgate_w8k"],
                    n_sa._buffers["_w7_sgate"],
                    float(lyr.post_feedforward_layernorm.eps),
                    bool(lyr.post_feedforward_layernorm.scale_input),
                    float(n_lyr.input_layernorm.eps),
                    bool(n_lyr.input_layernorm.scale_input),
                )
                n_gated = n_sa(
                    positions,
                    None,
                    forward_batch,
                    precomputed_qkv_gate=(nqkv, ng),
                    return_gated_attn=True,
                    **cs_kw,
                )
            stream, scaled_logits, expert_in = get_attn_to_moe_chip_sharded_bridge_op(
                tp, bf16_router=n_r16
            )(
                stream,
                n_gated,
                n_sa._o_proj_w8k,
                n_sa._o_proj_scale,
                n_pa_a,
                n_pa_b,
                n_pff_g,
                n_mlp._buffers["_w8_wpre_pair_w8k"],
                n_mlp._buffers["_w8_spre_pair"],
                n_mlp._buffers["_w7_wrouter_km"],
                n_peg,
                float(n_lyr.post_attention_layernorm.eps),
                float(n_lyr.pre_feedforward_layernorm.eps),
                bool(n_lyr.pre_feedforward_layernorm.scale_input),
                float(n_mlp.router_multiplier),
                float(n_mlp.pre_expert_norm.eps),
                bool(n_mlp.pre_expert_norm.scale_input),
            )
        elif tok_sharded_stream:
            use_epi = bool(lyr.__dict__.get("_w8_bridge_qkv_epi", False)) and n_sa.decode_qkv_epilogue
            if use_epi:
                epi_op = get_moe_to_attn_tok_sharded_bridge_op(
                    tp, fuse_qkv_epi=True, has_rope=n_has_rope
                )
                cs_t = (_contig(cos_sin_step),) if n_has_rope else ()
                stream, nq5, nkv, ng = epi_op(
                    stream,
                    moe_out,
                    w_post,
                    s_post,
                    post_ff_gain,
                    pf_a,
                    pf_b,
                    next_in_gain,
                    n_sa._buffers["_w7_wqkv_w8k"],
                    n_sa._buffers["_w7_sqkv"],
                    n_sa._buffers["_w7_wgate_w8k"],
                    n_sa._buffers["_w7_sgate"],
                    *cs_t,
                    float(lyr.post_feedforward_layernorm.eps),
                    bool(lyr.post_feedforward_layernorm.scale_input),
                    float(n_lyr.input_layernorm.eps),
                    bool(n_lyr.input_layernorm.scale_input),
                    float(n_sa.norm_eps),
                    float(n_sa.query_scale),
                    n_k_sc,
                    n_v_sc,
                )
                n_gated = n_sa(
                    positions,
                    None,
                    forward_batch,
                    precomputed_prepared_qkvg=(nq5, nkv, ng),
                    return_gated_attn=True,
                    **cs_kw,
                )
            else:
                b2_op = get_moe_to_attn_tok_sharded_bridge_op(tp, fuse_qkv_epi=False)
                stream, nqkv, ng = b2_op(
                    stream,
                    moe_out,
                    w_post,
                    s_post,
                    post_ff_gain,
                    pf_a,
                    pf_b,
                    next_in_gain,
                    n_sa._buffers["_w7_wqkv_w8k"],
                    n_sa._buffers["_w7_sqkv"],
                    n_sa._buffers["_w7_wgate_w8k"],
                    n_sa._buffers["_w7_sgate"],
                    float(lyr.post_feedforward_layernorm.eps),
                    bool(lyr.post_feedforward_layernorm.scale_input),
                    float(n_lyr.input_layernorm.eps),
                    bool(n_lyr.input_layernorm.scale_input),
                )
                n_gated = n_sa(
                    positions,
                    None,
                    forward_batch,
                    precomputed_qkv_gate=(nqkv, ng),
                    return_gated_attn=True,
                    **cs_kw,
                )
            stream, scaled_logits, expert_in = get_attn_to_moe_tok_sharded_bridge_op(
                tp, bf16_router=n_r16
            )(
                stream,
                n_gated,
                n_sa._o_proj_w8k,
                n_sa._o_proj_scale,
                n_pa_a,
                n_pa_b,
                n_pff_g,
                n_mlp._buffers["_w8_wpre_pair_w8k"],
                n_mlp._buffers["_w8_spre_pair"],
                n_mlp._buffers["_w7_wrouter_km"],
                n_peg,
                float(n_lyr.post_attention_layernorm.eps),
                float(n_lyr.pre_feedforward_layernorm.eps),
                bool(n_lyr.pre_feedforward_layernorm.scale_input),
                float(n_mlp.router_multiplier),
                float(n_mlp.pre_expert_norm.eps),
                bool(n_mlp.pre_expert_norm.scale_input),
            )
        else:
            tp16_epi_ok = tp == 16 and bool(
                getattr(tpu_envs, "SGLANG_TPU_MUSE_DECODE_TP16_QKV_EPI", None)
                is not None
                and tpu_envs.SGLANG_TPU_MUSE_DECODE_TP16_QKV_EPI.get()
            )
            use_epi = (
                (tp == 8 or tp16_epi_ok)
                and bool(lyr.__dict__.get("_w8_bridge_qkv_epi", False))
                and n_sa.decode_qkv_epilogue
            )
            if use_epi:
                epi_op = get_moe_to_attn_qkv_epi_bridge_op(
                    tp, sharded_moe_in=sharded_in, has_rope=n_has_rope
                )
                cs_t = (_contig(cos_sin_step),) if n_has_rope else ()
                stream, nq5, nkv, ng = epi_op(
                    stream,
                    moe_out,
                    w_post,
                    s_post,
                    post_ff_gain,
                    pf_a,
                    pf_b,
                    next_in_gain,
                    n_sa._buffers["_w7_wqkv_w8k"],
                    n_sa._buffers["_w7_sqkv"],
                    n_sa._buffers["_w7_wgate_w8k"],
                    n_sa._buffers["_w7_sgate"],
                    *cs_t,
                    float(lyr.post_feedforward_layernorm.eps),
                    bool(lyr.post_feedforward_layernorm.scale_input),
                    float(n_lyr.input_layernorm.eps),
                    bool(n_lyr.input_layernorm.scale_input),
                    float(n_sa.norm_eps),
                    float(n_sa.query_scale),
                    n_k_sc,
                    n_v_sc,
                )
                n_gated = n_sa(
                    positions,
                    None,
                    forward_batch,
                    precomputed_prepared_qkvg=(nq5, nkv, ng),
                    return_gated_attn=True,
                    **cs_kw,
                )
            else:
                b2_op = get_moe_to_attn_bridge_op(tp, sharded_moe_in=sharded_in)
                stream, nqkv, ng = b2_op(
                    stream,
                    moe_out,
                    w_post,
                    s_post,
                    post_ff_gain,
                    pf_a,
                    pf_b,
                    next_in_gain,
                    n_sa._buffers["_w7_wqkv_w8k"],
                    n_sa._buffers["_w7_sqkv"],
                    n_sa._buffers["_w7_wgate_w8k"],
                    n_sa._buffers["_w7_sgate"],
                    float(lyr.post_feedforward_layernorm.eps),
                    bool(lyr.post_feedforward_layernorm.scale_input),
                    float(n_lyr.input_layernorm.eps),
                    bool(n_lyr.input_layernorm.scale_input),
                )
                n_gated = n_sa(
                    positions,
                    None,
                    forward_batch,
                    precomputed_qkv_gate=(nqkv, ng),
                    return_gated_attn=True,
                    **cs_kw,
                )
            stream, scaled_logits, expert_in = get_attn_to_moe_bridge_op(
                tp, bf16_router=n_r16
            )(
                stream,
                n_gated,
                n_sa._o_proj_w8k,
                n_sa._o_proj_scale,
                n_pa_a,
                n_pa_b,
                n_pff_g,
                n_mlp._buffers["_w7_wpre_w8k"],
                n_mlp._buffers["_w7_spre"],
                n_mlp._buffers["_w7_wrouter_km"],
                n_peg,
                float(n_lyr.post_attention_layernorm.eps),
                float(n_lyr.pre_feedforward_layernorm.eps),
                bool(n_lyr.pre_feedforward_layernorm.scale_input),
                float(n_mlp.router_multiplier),
                float(n_mlp.pre_expert_norm.eps),
                bool(n_mlp.pre_expert_norm.scale_input),
            )

    return model.norm(stream.to(torch.bfloat16))
