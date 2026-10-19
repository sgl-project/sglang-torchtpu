from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from . import decode as D

# Must match the argument order of `decode.kernel`.
WEIGHT_ORDER = [
    "embed_tokens",
    "lm_head",
    "final_norm",
    "smalls",
    "smalls_f32",
    "kda_qkv",
    "kda_o",
    "kda_fg_b",
    "kda_dt_bias",
    "kda_conv",
    "kda_b",
    "kda_a_log",
    "dsa_qakv",
    "dsa_q_b",
    "dsa_kv_b",
    "dsa_o",
    "dsa_scale",
    "indexer_wq_b",
    "moe_router",
    "expert_gate_up",
    "expert_gate_up_scale",
    "expert_down",
    "expert_down_scale",
    "shared_gate_up",
    "shared_gate_up_scale",
    "shared_down",
    "shared_down_scale",
    "dense_gate_up",
    "dense_down",
    "dense_scale",
    "kda_scale",
]

# K-major packed weights ([L, slots, rows, cols]) hold each rank's blocks on the slot axis.
_REPLICATED = {
    "final_norm": 2,
    "smalls": 3,
    "smalls_f32": 3,
    "moe_router": 3,
    "dsa_qakv": 4,
    "indexer_wq_b": 4,
}
_SPLIT = {
    "embed_tokens": (2, 1),
    "lm_head": (4, 1),
    "kda_qkv": (4, 1),
    "kda_o": (4, 1),
    "kda_fg_b": (3, 1),
    "kda_dt_bias": (3, 2),
    "kda_conv": (3, 2),
    "kda_b": (3, 1),
    "kda_a_log": (3, 2),
    "kda_scale": (3, 2),
    "dsa_q_b": (4, 1),
    "dsa_kv_b": (3, 1),
    "dsa_o": (4, 1),
    "dsa_scale": (3, 2),
    "expert_gate_up": (3, 2),
    "expert_gate_up_scale": (2, 1),
    "expert_down": (3, 1),
    "expert_down_scale": (2, 1),
    "shared_gate_up": (3, 2),
    "shared_gate_up_scale": (2, 1),
    "shared_down": (3, 1),
    "shared_down_scale": (2, 1),
    "dense_gate_up": (4, 1),
    "dense_down": (4, 1),
    "dense_scale": (3, 2),
}


def weight_spec(name: str) -> P:
    if name in _REPLICATED:
        return P(*([None] * _REPLICATED[name]))
    ndim, axis = _SPLIT[name]
    parts = [None] * ndim
    parts[axis] = D.AXIS
    return P(*parts)


# n_query > 1: recur/conv are [L, Q, ...], the state after each query; a step reads pos[1].
CACHE_ORDER = ["kv", "idx_key", "idx_gate", "recur", "conv"]
N_DSA_CACHES = 3
KERNEL_CACHES = CACHE_ORDER[:N_DSA_CACHES] + ["idx_pool"] + CACHE_ORDER[N_DSA_CACHES:]
N_KDSA = N_DSA_CACHES + 1


def make_decode(spec, layout, small_map, small_f32_map, mesh, cfg: D.KernelConfig):
    """`shard(tok, pos, init_streams, weights, dsa_caches, kda_caches)` and its helpers."""
    tp = layout.tp
    slab = D.SlabIndex(small_map, small_f32_map, spec)
    kernel = D.build(spec, layout, cfg, slab)
    write_rows = D.row_write_call(spec, cfg)
    n_dsa = len(spec.dsa_layers)
    n_w = len(WEIGHT_ORDER)
    in_specs = (
        [pl.BlockSpec(memory_space=pltpu.SMEM)] * 2
        + [pl.BlockSpec(memory_space=pltpu.VMEM)]
        + [pl.BlockSpec(memory_space=pl.ANY)] * (n_w + len(KERNEL_CACHES))
    )

    def make_call(cache_structs):
        # XLA copies operands aliased by a kernel with remote DMA, so KDA states are
        # fresh outputs; a capped run (n_layers) aliases them to keep skipped layers.
        n_lead = 3 + n_w
        aliases = (
            {n_lead + i: 3 + i for i in range(N_KDSA, len(KERNEL_CACHES))}
            if cfg.n_layers is not None
            else {}
        )
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct(
                    (cfg.n_query, layout.vocab_padded // tp), jnp.float32
                ),
                jax.ShapeDtypeStruct(
                    (cfg.n_query * spec.hc_mult, spec.hidden_size), jnp.bfloat16
                ),
                jax.ShapeDtypeStruct((cfg.n_query, spec.hidden_size), jnp.bfloat16),
            )
            + tuple(
                jax.ShapeDtypeStruct((n_dsa, cfg.n_query, c.shape[-1]), c.dtype)
                for c in cache_structs[:N_KDSA]
            )
            + tuple(cache_structs[N_KDSA:]),
            in_specs=in_specs,
            out_specs=(pl.BlockSpec(memory_space=pltpu.VMEM),) * (3 + N_KDSA)
            + tuple(pl.BlockSpec(memory_space=pl.ANY) for _ in KERNEL_CACHES[N_KDSA:]),
            scratch_shapes=D.scratch_shapes(spec, layout, cfg),
            input_output_aliases=aliases,
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=cfg.vmem_mib * 2**20),
        )

    def body(tok, pos, init_streams, warrays, dsa_caches, kda_caches):
        caches = {**dsa_caches, **kda_caches}
        ws = [warrays[n] for n in WEIGHT_ORDER]
        cs = [caches[n] for n in KERNEL_CACHES]
        call = make_call([jax.ShapeDtypeStruct(c.shape, c.dtype) for c in cs])
        out = call(tok, pos, init_streams, *ws, *cs)
        new_rows = out[3 : 3 + N_KDSA]
        dsa = write_rows(pos, *new_rows, *cs[:N_KDSA])
        new = dict(zip(KERNEL_CACHES, (*dsa, *out[3 + N_KDSA :])))
        return out[0], out[1], out[2], new

    wspec = {n: weight_spec(n) for n in WEIGHT_ORDER}
    cspec = {
        "kv": P(None, None, None),
        "idx_key": P(None, None, None),
        "idx_gate": P(None, None, None),
        "idx_pool": P(None, None, None),
        "recur": P(None, D.AXIS, None, None),
        "conv": P(None, None, D.AXIS),
    }
    if cfg.n_query > 1:
        cspec["recur"] = P(None, None, D.AXIS, None, None)
        cspec["conv"] = P(None, None, None, D.AXIS)
    dsa_names = KERNEL_CACHES[:N_KDSA]
    kda_names = KERNEL_CACHES[N_KDSA:]
    dspec = {n: cspec[n] for n in dsa_names}
    kspec = {n: cspec[n] for n in kda_names}
    shard = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(P(), P(), P(None, None), wspec, dspec, kspec),
        out_specs=(P(None, D.AXIS), P(None, None), P(None, None), cspec),
        check_vma=False,
    )

    # Every DSA layer plus the MTP layer (index num_hidden_layers).
    ape_off, ape_rows, ape_shape = small_map["indexer_ape"]
    ape_layers = list(spec.dsa_layers) + [spec.num_hidden_layers]
    ape_n = 1
    for d_ in ape_shape:
        ape_n *= d_

    def init_ip(ik, ig, smalls):
        apes = [
            smalls[L, ape_off : ape_off + ape_rows]
            .reshape(-1)[:ape_n]
            .reshape(ape_shape)
            .astype(jnp.float32)
            for L in ape_layers
        ]
        return D.init_pool_keys(spec, ik, ig, apes)

    return _Step(
        shard=shard,
        init_ip=init_ip,
        wspec=wspec,
        dspec=dspec,
        kspec=kspec,
        dsa_names=dsa_names,
        kda_names=kda_names,
        step_caches=KERNEL_CACHES,
    )


class _Step:
    def __init__(self, **kw):
        self.__dict__.update(kw)
