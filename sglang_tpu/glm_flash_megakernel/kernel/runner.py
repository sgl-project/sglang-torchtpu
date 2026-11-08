from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from . import decode as D
from . import pools as PL
from .writeback import row_write_call

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

# sglang's per-layer expert tensors: lists in `weights`, passed after the pools.
EXPERT_LISTS = ("expert_gate_up", "expert_down")
PACKED_ORDER = [n for n in WEIGHT_ORDER if n not in EXPERT_LISTS]

# Serving's per-layer pools, in the order the kernel takes them (one array per layer).
POOL_ORDER = ("nope", "raw", "records", "temporal", "conv")

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
    "kda_conv": (4, 3),
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
# Each rank holds its own shard of the mamba state; the DSA pools are replicated.
_POOL_SPECS = {
    "nope": P(None, None, None, None),
    "raw": P(None, None, None),
    "records": P(None, None, None, None),
    "temporal": P(None, D.AXIS, None, None),
    "conv": P(None, None, None, D.AXIS),
}


def weight_spec(name: str) -> P:
    if name in _REPLICATED:
        return P(*([None] * _REPLICATED[name]))
    ndim, axis = _SPLIT[name]
    parts = [None] * ndim
    parts[axis] = D.AXIS
    return P(*parts)


def _scatter_index_rows(spec, cfg, meta, pos, tab, nraw, nrec, raw, rec):
    """Write the step's raw index rows and the records of the pools it closes with
    XLA scatters (sub-tile slices a DMA cannot address); dropped rows get page -1."""
    R, Q, B = cfg.rows, cfg.n_query, cfg.n_req
    ps, kp, pps = cfg.page_size, spec.index_kpool, cfg.pages_per_seq
    r = jnp.arange(R, dtype=jnp.int32)
    b, qi = r // Q, r % Q
    live = b < meta[0]
    tok = pos[b] + qi
    tpage = jnp.where(live, tab[b * pps + jnp.minimum(tok // ps, pps - 1)], -1)
    fp0, fresh = PL.closing_pools(pos[b], Q, kp)
    ptok = (fp0 + qi) * kp
    rpage = jnp.where(
        live & (qi < fresh), tab[b * pps + jnp.minimum(ptok // ps, pps - 1)], -1
    )
    ri = (ptok % ps) // kp
    raw_out = [
        raw[d].at[tpage, tok % ps].set(nraw[d], mode="drop") for d in range(len(raw))
    ]
    rec_out = [
        rec[d].at[rpage, ri // 4, ri % 4].set(nrec[d], mode="drop")
        for d in range(len(rec))
    ]
    return raw_out, rec_out


def make_decode(spec, layout, small_map, small_f32_map, mesh, cfg: D.KernelConfig):
    """`shard(step_inputs, init_streams, weights, pools) -> (logits, hidden, pools)`."""
    tp = layout.tp
    slab = D.SlabIndex(small_map, small_f32_map, spec)
    kernel = D.build(spec, layout, cfg, slab)
    write_rows = row_write_call(spec, cfg)
    n_dsa, n_kda = len(spec.dsa_layers), len(spec.kda_layers)
    n_moe = sum(t == "sparse" for t in spec.mlp_layer_types)
    n_w = len(PACKED_ORDER)
    n_pool = 3 * n_dsa + 2 * n_kda
    R, Q, B = cfg.rows, cfg.n_query, cfg.n_req
    H = spec.hidden_size
    hkda, dh = spec.linear_num_heads // tp, spec.linear_head_dim
    k1, qsh = spec.linear_conv_kernel_dim - 1, spec.kda_qkv_dim // tp
    rec_bytes = PL.record_bytes(spec.index_head_dim)
    # tok, pos, meta, page table, state rows, k_scales
    n_smem = 6
    in_specs = (
        [pl.BlockSpec(memory_space=pltpu.SMEM)] * n_smem
        + [pl.BlockSpec(memory_space=pltpu.VMEM)]
        + [pl.BlockSpec(memory_space=pl.ANY)]
        * (n_w + n_pool + len(EXPERT_LISTS) * n_moe)
    )
    out_shape = (
        jax.ShapeDtypeStruct((R, layout.vocab_padded // tp), jnp.float32),
        jax.ShapeDtypeStruct((R * spec.hc_mult, H), jnp.bfloat16),
        jax.ShapeDtypeStruct((R, H), jnp.bfloat16),
        jax.ShapeDtypeStruct((n_dsa, R, spec.kv_lora_rank), jnp.bfloat16),
        jax.ShapeDtypeStruct((n_dsa, R, 2 * spec.index_head_dim), jnp.bfloat16),
        jax.ShapeDtypeStruct((n_dsa, B, 8, spec.index_head_dim), jnp.float32),
        jax.ShapeDtypeStruct((n_kda, R, hkda, dh, dh), jnp.float32),
        jax.ShapeDtypeStruct((n_kda, R, k1, 1, 3 * qsh), jnp.float32),
    )
    out_specs = (pl.BlockSpec(memory_space=pltpu.VMEM),) * 6 + (
        pl.BlockSpec(memory_space=pl.ANY),
    ) * 2
    call = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=D.scratch_shapes(spec, layout, cfg),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=cfg.vmem_mib * 2**20),
        name=f"glm_flash_mk_r{R}_q{Q}",
    )

    def body(tok, pos, meta, tab, srow, ckrow, k_scales, init_streams, warrays, pools):
        ws = [warrays[n] for n in PACKED_ORDER]
        ps = [p for n in POOL_ORDER for p in pools[n]]
        experts = [w for n in EXPERT_LISTS for w in warrays[n]]
        out = call(
            tok, pos, meta, tab, srow, k_scales, init_streams, *ws, *ps, *experts
        )
        logits, _, hidden, nkv, nraw, npk, recur, conv = out
        # fp8 bytes as the nope pool holds them: kv / k_scale per layer.
        nkv8 = (nkv.astype(jnp.float32) / k_scales[:, None, None]).astype(
            jnp.float8_e4m3fn
        )
        nkv8 = jax.lax.bitcast_convert_type(nkv8, jnp.uint8).reshape(n_dsa, R, 4, 128)
        npk = npk[:, :, :Q].reshape(n_dsa, R, spec.index_head_dim)
        nrec = jnp.stack([PL.pool_records(npk[d], rec_bytes) for d in range(n_dsa)])
        raw_out, rec_out = _scatter_index_rows(
            spec, cfg, meta, pos, tab, nraw, nrec, pools["raw"], pools["records"]
        )
        nope_out = write_rows(meta, pos, tab, nkv8, *pools["nope"])
        # KDA checkpoints (17 MB per row) go in as XLA scatters on the donated
        # pools: an in-kernel HBM->HBM DMA runs at ~33 GB/s on v7x, 2 ms per 4 rows.
        live = jnp.arange(R, dtype=jnp.int32) // Q < meta[0]
        ck = jnp.where(live, ckrow, -1)
        temporal_out = [
            pools["temporal"][k].at[ck].set(recur[k], mode="drop") for k in range(n_kda)
        ]
        conv_out = [
            pools["conv"][k].at[ck].set(conv[k], mode="drop") for k in range(n_kda)
        ]
        return (
            logits,
            hidden,
            {
                "nope": list(nope_out),
                "raw": raw_out,
                "records": rec_out,
                "temporal": temporal_out,
                "conv": conv_out,
            },
        )

    wspec = {n: weight_spec(n) for n in PACKED_ORDER}
    wspec.update({n: [weight_spec(n)] * n_moe for n in EXPERT_LISTS})
    pspec = {
        n: [_POOL_SPECS[n]] * (n_dsa if n in ("nope", "raw", "records") else n_kda)
        for n in POOL_ORDER
    }
    shard = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(P(), P(), P(), P(), P(), P(), P(), P(None, None), wspec, pspec),
        out_specs=(P(None, D.AXIS), P(None, None), pspec),
        check_vma=False,
    )
    return _Step(shard=shard, wspec=wspec, pspec=pspec)


class _Step:
    def __init__(self, **kw):
        self.__dict__.update(kw)
