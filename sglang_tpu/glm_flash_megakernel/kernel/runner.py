"""Wire the decode megakernel into a callable: one `pallas_call` per step.

Keeps the argument order in one place, so the kernel signature and the call site
cannot drift apart silently, and reports the compiled memory accounting -- argument,
output, alias and temp sizes -- because donation being honoured and a full extra
copy of the cache still being materialised are both true at once often enough that
neither answers the question on its own.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.layout import Format, Layout
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from ..weights import CH_REPLACES, SHARD_AXIS
from . import decode as D


# The kernel reads its weights in this order; `runner` and `decode.kernel` share it.
WEIGHT_ORDER = [
    "embed_tokens", "lm_head", "final_norm", "smalls", "smalls_f32",
    "kda_qkv", "kda_o", "kda_fg_b", "kda_dt_bias",
    "kda_conv", "kda_b", "kda_alog",
    "dsa_q_a", "dsa_q_a_scale", "dsa_q_b", "dsa_q_b_scale",
    "dsa_kv_a", "dsa_kv_a_scale", "dsa_kv_b", "dsa_o", "dsa_o_scale",
    "indexer_wq_b", "moe_router",
    "expert_gate_up", "expert_gate_up_scale", "expert_down", "expert_down_scale",
    "shared_gate_up", "shared_gate_up_scale", "shared_down", "shared_down_scale",
    "dense_gate_up", "dense_gate_up_scale", "dense_down", "dense_down_scale",
    "kda_scale",
]
# Weights the runner derives from the packed set at first use rather than reads.
DERIVED = ("kda_qkv", "kda_o", "kda_scale", "indexer_wq_b", "lm_head", "kda_conv")
# With n_query > 1, recur / conv carry a checkpoint axis after the layer axis
# ([L, Q, ...]): the state after each query, and the step reads checkpoint pos[1].
# The kernel's conv state is channels-last, [L, (Q,) K-1, C]; a caller's
# [L, C, K-1] state (the reference layout) is transposed on the way in.
CACHE_ORDER = ["kv", "idx_key", "idx_gate", "recur", "conv"]
N_DSA_CACHES = 3   # the leading CACHE_ORDER entries, read by the kernel, written after it
# The kernel's own cache order: the pooled indexer keys (idx_pool, f32 [n_dsa,
# L/kpool, Dh]) follow the raw DSA caches.  The runner derives idx_pool from the
# caller's idx_key / idx_gate and carries it between its own steps.
KERNEL_CACHES = CACHE_ORDER[:N_DSA_CACHES] + ["idx_pool"] + CACHE_ORDER[N_DSA_CACHES:]
N_KDSA = N_DSA_CACHES + 1


def make_decode(spec, W, mesh, cfg: D.KernelConfig):
    layout = W.layout
    tp = layout.tp
    slab = D.SlabIndex(W.small_map, W.small_f32_map, spec)
    kernel = D.build(spec, layout, cfg, slab)
    write_rows = D.row_write_call(spec, cfg)
    step_caches = list(KERNEL_CACHES)
    n_dsa = len(spec.dsa_layers)

    n_w = len(WEIGHT_ORDER)
    in_specs = (
        [pl.BlockSpec(memory_space=pltpu.SMEM)] * 2        # token, pos (+ checkpoint)
        + [pl.BlockSpec(memory_space=pltpu.VMEM)]          # init streams
        + [pl.BlockSpec(memory_space=pl.ANY)] * (n_w + len(KERNEL_CACHES))
    )
    def cache_struct(c):
        return jax.ShapeDtypeStruct(_local_shape(c, mesh), c.dtype)

    def make_call(cache_structs):
        # The KDA states are written in place (aliased); the DSA caches are only
        # read, and their new rows come out as three small outputs for
        # `row_write_call`.  `memory_report` shows whether a copy remains.
        # A kernel with remote DMA is a collective to XLA, which copies every
        # aliased operand; the KDA states are therefore fresh outputs, written
        # whole by every layer.  A capped run (n_layers) skips layers, so it
        # aliases them to keep the skipped layers' state.
        n_lead = 3 + n_w
        aliases = ({n_lead + i: 3 + i for i in range(N_KDSA, len(KERNEL_CACHES))}
                   if cfg.n_layers is not None else {})
        return pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((cfg.n_query,
                                            layout.vocab_padded // tp),
                                            jnp.float32),
                       jax.ShapeDtypeStruct((cfg.n_query * spec.hc_mult,
                                             spec.hidden_size),
                                            jnp.bfloat16),
                       jax.ShapeDtypeStruct((cfg.n_query, spec.hidden_size),
                                            jnp.bfloat16),)
                      + tuple(jax.ShapeDtypeStruct((n_dsa, cfg.n_query, c.shape[-1]),
                                                   c.dtype)
                              for c in cache_structs[:N_KDSA])
                      + tuple(cache_structs[N_KDSA:]),
            in_specs=in_specs,
            out_specs=(pl.BlockSpec(memory_space=pltpu.VMEM),) * 3
                      + (pl.BlockSpec(memory_space=pltpu.VMEM),) * N_KDSA
                      + tuple(pl.BlockSpec(memory_space=pl.ANY)
                              for _ in KERNEL_CACHES[N_KDSA:]),
            scratch_shapes=D.scratch_shapes(spec, layout, cfg),
            input_output_aliases=aliases,
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=cfg.vmem_mib * 2**20),
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
        return out[0], out[1], out[2], {n: new[n] for n in step_caches}

    # With ch_experts the expert slots of the kernel read the *_ch arrays.
    src = {n: (CH_REPLACES[n] if cfg.ch_experts and n in CH_REPLACES else n)
           for n in WEIGHT_ORDER}
    wspec = {n: P(*([None] * warr_ndim(W, src[n]))) if SHARD_AXIS[src[n]] is None
             else shard_spec(W, src[n]) for n in WEIGHT_ORDER if n != "kda_scale"}
    # The bf16-ring weights stream K-major (decode.pack_kmajor), packed per
    # shard once; with fp8_kda the KDA qkv / o projections are also
    # quantized to fp8 e4m3 with per-output-channel scales (per shard: o is
    # row-parallel, so its scales differ between ranks).
    for n in DERIVED:
        wspec[n] = P(None, D.AXIS, None, None)
    wspec["indexer_wq_b"] = P(None, None, None, None)
    wspec["kda_scale"] = P(None, None, D.AXIS)
    wspec["kda_conv"] = P(None, None, D.AXIS)
    if cfg.embed_2d:
        wspec["embed_tokens"] = P(None, D.AXIS)

    def prep_local(qkv, o, iwq, lm, conv):
        TR, Hd = cfg.km_cols, spec.hidden_size

        def quant(w):
            s_ = jnp.maximum(jnp.max(jnp.abs(w.astype(jnp.float32)), axis=2) / 448.0, 1e-12)
            return (w.astype(jnp.float32) / s_[..., None]).astype(jnp.float8_e4m3fn), s_

        if cfg.fp8_kda:
            q8, s_q = quant(qkv)
            o8, s_o = quant(o)
            kqkv, ko = D.pack_kmajor(q8, 2 * Hd, TR), D.pack_kmajor(o8, 2 * Hd, TR)
            ksc = jnp.concatenate([s_q, s_o], axis=1)[:, None, :]
        else:
            kqkv, ko = D.pack_kmajor(qkv, Hd, TR), D.pack_kmajor(o, Hd, TR)
            ksc = jnp.ones((qkv.shape[0], 1, qkv.shape[1] + Hd), jnp.float32)
        return (kqkv, ko, ksc, D.pack_kmajor(iwq, Hd, TR),
                D.pack_kmajor(lm[None], Hd, TR), jnp.swapaxes(conv, 1, 2))

    prep = jax.jit(jax.shard_map(
        prep_local, mesh=mesh,
        in_specs=(shard_spec(W, "kda_qkv"), shard_spec(W, "kda_o"),
                  P(None, None, None), shard_spec(W, "lm_head"), shard_spec(W, "kda_conv")),
        out_specs=tuple(wspec[n] for n in DERIVED),
        check_vma=False))
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

    cspec = {n: cspec[n] for n in step_caches}
    # Only the DSA caches are donated: row_write_call updates them in place.  The
    # KDA states come out of the kernel as fresh buffers, and donating their
    # inputs makes XLA copy each one aside and the new state back every step.
    dsa_names = KERNEL_CACHES[:N_KDSA]
    kda_names = [n for n in step_caches if n not in dsa_names]
    dspec = {n: cspec[n] for n in dsa_names}
    kspec = {n: cspec[n] for n in kda_names}
    shard = jax.shard_map(
        body, mesh=mesh,
        in_specs=(P(), P(), P(None, None), wspec, dspec, kspec),
        out_specs=(P(None, D.AXIS), P(None, None), P(None, None), cspec),
        check_vma=False,
    )
    # Donate the caches: the kernel writes them in place, and without donation XLA
    # must keep the caller's buffers alive and copies every cache on each step.
    jshard = jax.jit(shard, donate_argnums=(4,))

    # Default parameter layouts do not match what the kernel declares for several
    # arrays (embed_tokens [V, 1, D] arrives with its unit dim major), and XLA then
    # re-lays them out on every call.  Compiling with AUTO input layouts and moving
    # the weights and caches into the chosen layouts once removes those copies.
    ns = lambda s_: NamedSharding(mesh, s_)
    auto = lambda s_: Format(Layout.AUTO, ns(s_))
    # The caches are pinned row-major, which is what the kernel declares: left
    # to XLA, the [L, 3072, 3] conv state gets its 3072 axis minor and costs a
    # conversion copy on each side of the kernel every step.
    def row_major(n):
        return Format(Layout(major_to_minor=tuple(range(len(cspec[n])))), ns(cspec[n]))

    in_sh = (ns(P()), ns(P()), ns(P(None, None)),
             {n: auto(wspec[n]) for n in WEIGHT_ORDER},
             {n: row_major(n) for n in dsa_names}, {n: row_major(n) for n in kda_names})
    out_sh = (ns(P(None, D.AXIS)), ns(P(None, None)), ns(P(None, None)),
              {n: row_major(n) for n in step_caches})
    state = {}

    def _compile(args):
        def struct(sh, x):
            return jax.ShapeDtypeStruct(
                x.shape, x.dtype, sharding=sh.sharding if isinstance(sh, Format) else sh)

        structs = jax.tree.map(struct, in_sh, args,
                               is_leaf=lambda v: isinstance(v, (Format, NamedSharding)))
        exe = jax.jit(shard, in_shardings=in_sh, out_shardings=out_sh,
                      donate_argnums=(4,)).lower(*structs).compile()
        fin = exe.input_formats[0]
        # Re-jit on the concrete layouts: a plain jit call takes the C++ dispatch
        # path, while calling the AOT executable re-checks every argument in Python.
        exe = jax.jit(shard, in_shardings=fin, out_shardings=exe.output_formats,
                      donate_argnums=(4,))
        state.update(exe=exe, fin=fin, w={}, c={})

    # The pooled-key cache is derived per DSA layer (and the MTP slot) with its
    # layer's indexer_ape from the slab.
    ape_off, ape_rows, ape_shape = W.small_map["indexer_ape"]
    ape_layers = list(spec.dsa_layers) + [spec.num_hidden_layers]

    @jax.jit
    def init_ip(ik, ig, smalls):
        n = 1
        for d_ in ape_shape:
            n *= d_
        apes = [smalls[L, ape_off : ape_off + ape_rows].reshape(-1)[:n]
                .reshape(ape_shape).astype(jnp.float32) for L in ape_layers]
        return D.init_pool_keys(spec, ik, ig, apes)

    def finish_caches(held, out):
        state["ip"], state["ip_for"] = out["idx_pool"], out["idx_key"]
        return {**held, **{n: c for n, c in out.items() if n != "idx_pool"}}

    def run(tok, pos, warrays, caches, init_streams=None):
        # The packed set also carries `mtp_eh_proj`, which the non-speculative
        # decode kernel does not read; pass exactly what the kernel declares.
        if init_streams is None:
            if "init" not in state:
                state["init"] = jnp.zeros(
                    (cfg.n_query * spec.hc_mult, spec.hidden_size), jnp.bfloat16)
            init_streams = state["init"]
        if state.get("ko_src") is not warrays["kda_o"]:
            # The reference forward keeps the original arrays in the caller's dict.
            state["ko_src"] = warrays["kda_o"]
            state["derived"] = dict(zip(DERIVED, prep(
                warrays["kda_qkv"], warrays["kda_o"], warrays["indexer_wq_b"],
                warrays["lm_head"], warrays["kda_conv"])))
        if state.get("ip_for") is not caches["idx_key"]:
            # Caches this runner did not produce (a new request, a reference
            # check): derive the pooled keys from the raw indexer caches.
            state["ip"] = init_ip(caches["idx_key"], caches["idx_gate"], warrays["smalls"])
        caches = {**caches, "idx_pool": state["ip"]}
        k1 = spec.linear_conv_kernel_dim - 1
        if caches["conv"].shape[-1] == k1:      # the reference's [L, C, K-1]
            caches = {**caches, "conv": jnp.swapaxes(caches["conv"], 1, 2)}
        if cfg.n_query > 1:
            # A caller's single state (a fresh request, a reference) becomes every
            # checkpoint; a bare position resumes from checkpoint 0.
            if pos.shape[0] == 1:
                pos = jnp.concatenate([pos, jnp.zeros((1,), pos.dtype)])
            ndim = {"recur": 4, "conv": 3}
            caches = {n: jnp.repeat(c[:, None], cfg.n_query, axis=1)
                      if ndim.get(n) == c.ndim else c for n, c in caches.items()}
        held = {n: c for n, c in caches.items() if n not in step_caches}
        caches = {n: caches[n] for n in step_caches}
        if cfg.embed_2d and warrays["embed_tokens"].ndim == 3:
            e_ = warrays["embed_tokens"]
            warrays["embed_tokens"] = e_.reshape(e_.shape[0], e_.shape[2])

        def wget(n):
            return state["derived"][n] if n in DERIVED else warrays[src[n]]

        args = (tok, pos, init_streams, {n: wget(n) for n in WEIGHT_ORDER},
                {n: caches[n] for n in dsa_names}, {n: caches[n] for n in kda_names})
        if isinstance(tok, jax.core.Tracer):
            logits, streams, state["hidden"], out = jshard(*args)
            return logits, streams, finish_caches(held, out)
        if "exe" not in state:
            _compile(args)
        fin = state["fin"]
        # Re-lay a weight once, in the caller's dict, so the old copy is freed.
        for n in WEIGHT_ORDER:
            if state["w"].get(n) is not wget(n):
                if wget(n).format != fin[3][n]:
                    if n in DERIVED:
                        state["derived"][n] = jax.device_put(state["derived"][n], fin[3][n])
                    else:
                        warrays[src[n]] = jax.device_put(warrays[src[n]], fin[3][n])
                state["w"][n] = wget(n)
        cfmt = {**fin[4], **fin[5]}
        caches = {n: c if state["c"].get(n) is c or c.format == cfmt[n]
                  else jax.device_put(c, cfmt[n]) for n, c in caches.items()}
        dc = {n: caches[n] for n in dsa_names}
        kc = {n: caches[n] for n in kda_names}
        if init_streams is state.get("init"):
            if "init_put" not in state:
                state["init_put"] = jax.device_put(init_streams, fin[2])
            init_streams = state["init_put"]
        else:
            init_streams = jax.device_put(init_streams, fin[2])
        wargs = {n: wget(n) for n in WEIGHT_ORDER}
        if not state.get("put_scalars"):
            try:
                logits, streams, state["hidden"], out = state["exe"](
                    tok, pos, init_streams, wargs, dc, kc)
            except ValueError:   # the executable wants the replicated placement
                state["put_scalars"] = True
        if state.get("put_scalars"):
            logits, streams, state["hidden"], out = state["exe"](
                jax.device_put(tok, fin[0]), jax.device_put(pos, fin[1]),
                init_streams, wargs, dc, kc)
        state["c"] = dict(out)
        return logits, streams, finish_caches(held, out)

    # For a caller that holds its own arrays (serving): the one-time weight
    # preparation and the pooled-key derivation, with their input specs.
    run.prep, run.init_ip = prep, init_ip
    run.prep_in_specs = (shard_spec(W, "kda_qkv"), shard_spec(W, "kda_o"),
                         P(None, None, None), shard_spec(W, "lm_head"),
                         shard_spec(W, "kda_conv"))
    run.src = src
    run.shard = shard
    run.wspec, run.dspec, run.kspec = wspec, dspec, kspec
    run.dsa_names, run.kda_names, run.step_caches = dsa_names, kda_names, step_caches
    run.lower_args = lambda tok, pos, warrays, caches, s_: (
        tok, pos, s_, {n: warrays[n] for n in WEIGHT_ORDER},
        {n: caches[n] for n in dsa_names}, {n: caches[n] for n in kda_names})
    run.jitted = jshard
    run.state = state      # the compiled AUTO-layout step, once run
    return run


def _local_shape(x, mesh):
    return x.shape


def warr_ndim(W, name):
    return W.arrays[name].ndim


def shard_spec(W, name):
    ax = SHARD_AXIS[name]
    parts = [None] * W.arrays[name].ndim
    parts[ax] = D.AXIS
    return P(*parts)


def memory_report(fn, *args) -> dict:
    """Compiled argument / output / alias / temp sizes.

    Donation can be honoured and a full extra copy of a large buffer still be
    materialised as a temporary; both facts show up here together, and neither one
    alone answers whether the layout is paying for itself.
    """
    m = fn.lower(*args).compile().memory_analysis()
    return {
        "argument_GiB": m.argument_size_in_bytes / 2**30,
        "output_GiB": m.output_size_in_bytes / 2**30,
        "alias_GiB": m.alias_size_in_bytes / 2**30,
        "temp_GiB": m.temp_size_in_bytes / 2**30,
    }
