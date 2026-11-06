# SPDX-License-Identifier: Apache-2.0
"""Torch ops for the Kimi-K3 host-split GEMV (sglang_tpu/kernels/ops/gemm/host_split_gemv.py).

The static fp8 binder (static_fp8.py) asks for one op per split shape at bind
time, before torch.compile traces the model.
"""

# PEP 563 must stay off here: cpu_export_jax_op reads the wrappers' jax.Array
# annotations, and ``from __future__ import annotations`` turns them into
# strings it rejects (static_fp8.py has it, so the op build cannot live there).

import functools


def _build(
    world: int,
    local: int,
    cols: int,
    used: int,
    used_e: int,
    prefetch: bool,
    tail: int = 0,
):
    """The host-split GEMV for one split shape, as a torch op over the TP mesh.
    ``prefetch`` (latent up-proj only): the kernel also copies the next
    layer's weight (``pf``, HBM) into a VMEM output, returned last.
    ``tail`` (latent up-proj only): 1 = fused MoE tail add (up + shared) and
    next-layer attention-residual aggregation, 2 = the same with the pending
    prefix add ((up + shared) + prefix); see host_split_gemv_local."""
    import jax
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang_tpu.kernels.ops.gemm import host_split_gemv
    from sglang_tpu.pallas_jax_op import cpu_export_jax_op

    npf = int(prefetch)
    mesh = get_pallas_mesh(("model",), (world,))
    spec = P("model")
    body = functools.partial(
        host_split_gemv.host_split_gemv_local,
        local=local,
        cols=cols,
        used=used,
        used_e=used_e,
    )
    if tail:
        assert not used_e
        has_prefix = tail == 2
        n_t = 4 + has_prefix  # shared, [prefix], bank, cw, norm_w

        def _tail_wrapped(x, w, out_scale, inv_x, *rest):
            # rest = [pf], shared, [prefix], bank, cw, norm_w, nvb, eps, out_eps
            nvb, eps, out_eps = rest[-3:]
            tensors = rest[:-3]
            pfs = list(tensors[:npf])
            t = list(tensors[npf:])
            shared = t.pop(0)
            prefix = t.pop(0) if has_prefix else None
            bank, cw, norm_w = t

            def per_rank(x, w, s, inv, *tr):
                tr = list(tr)
                pf_l = tr.pop(0) if npf else None
                sh = tr.pop(0)
                pre = tr.pop(0) if has_prefix else None
                bk_, cw_, nw_ = tr
                outs = body(
                    x,
                    w,
                    s,
                    inv,
                    None,
                    prefetch=pf_l,
                    tail=(sh, pre, bk_, cw_, nw_, nvb, eps, out_eps),
                )
                return tuple(outs)

            ins = [x, w, out_scale, inv_x] + pfs
            ins += [shared] + ([prefix] if has_prefix else []) + [bank, cw, norm_w]
            n_out_t = 2 + npf
            return jax.shard_map(
                per_rank,
                mesh=mesh,
                in_specs=(spec,) * len(ins),
                out_specs=(spec,) * n_out_t,
                check_vma=False,
            )(*ins)

        n_tensors = 4 + npf + n_t
        # cpu_export_jax_op reads the wrapper's annotations: spell them out.
        names = ["x", "w", "out_scale", "inv_x"] + (["pf"] if npf else [])
        names += (
            ["shared"] + (["prefix"] if has_prefix else []) + ["bank", "cw", "norm_w"]
        )
        src = (
            "def _wrapped("
            + ", ".join(f"{a}: jax.Array" for a in names)
            + ", nvb: int, eps: float, out_eps: float) -> tuple["
            + ", ".join(["jax.Array"] * (2 + npf))
            + "]:\n"
            + "    return _tail_wrapped("
            + ", ".join(names)
            + ", nvb, eps, out_eps)\n"
        )
        ns = {"jax": jax, "_tail_wrapped": _tail_wrapped}
        exec(src, ns)
        _wrapped = ns["_wrapped"]
        tag = f"c{cols}_u{used}" + ("_pf" if npf else "") + f"_tail{tail}"
        return cpu_export_jax_op(
            f"sglang_tpu_pallas::host_split_gemv_w{world}_l{local}_{tag}",
            _wrapped,
            mesh=mesh,
            input_partition_specs=(spec,) * n_tensors + (P(),) * 3,
            output_partition_specs=(spec,) * (2 + npf),
        )
    if used_e:
        sharded = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * 5,
            out_specs=(spec,) * 3,
            check_vma=False,
        )

        def _wrapped(
            x: jax.Array,
            w: jax.Array,
            out_scale: jax.Array,
            inv_x: jax.Array,
            router: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return sharded(x, w, out_scale, inv_x, router)

        n_in, n_out = 5, 3
    elif npf:
        assert not used_e
        # The latent up-proj also copies the next layer's weight (``pf``,
        # HBM) into a VMEM output, returned second (host_split_gemv.py).
        sharded = jax.shard_map(
            lambda x, w, s, inv, pf: body(x, w, s, inv, None, prefetch=pf),
            mesh=mesh,
            in_specs=(spec,) * 5,
            out_specs=(spec,) * 2,
            check_vma=False,
        )

        def _wrapped(
            x: jax.Array,
            w: jax.Array,
            out_scale: jax.Array,
            inv_x: jax.Array,
            pf: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return sharded(x, w, out_scale, inv_x, pf)

        n_in, n_out = 5, 2
    else:
        sharded = jax.shard_map(
            lambda *a: body(*a)[0],
            mesh=mesh,
            in_specs=(spec,) * 4,
            out_specs=spec,
            check_vma=False,
        )

        def _wrapped(
            x: jax.Array, w: jax.Array, out_scale: jax.Array, inv_x: jax.Array
        ) -> jax.Array:
            return sharded(x, w, out_scale, inv_x)

        n_in, n_out = 4, 1
    tag = (
        f"c{cols}_u{used}" + (f"_r{used_e}" if used_e else "") + ("_pf" if npf else "")
    )
    return cpu_export_jax_op(
        f"sglang_tpu_pallas::host_split_gemv_w{world}_l{local}_{tag}",
        _wrapped,
        mesh=mesh,
        input_partition_specs=(spec,) * n_in,
        output_partition_specs=(spec,) * n_out,
    )


# One op per split shape (the front and the latent up-proj), built at bind
# time, before torch.compile traces the model.
_OPS: dict = {}


def host_split_op(
    world: int,
    local: int,
    cols: int,
    used: int,
    used_e: int = 0,
    prefetch: bool = False,
    tail: int = 0,
):
    key = (world, local, cols, used, used_e, bool(prefetch), tail)
    if key not in _OPS:
        _OPS[key] = _build(*key)
    return _OPS[key]
