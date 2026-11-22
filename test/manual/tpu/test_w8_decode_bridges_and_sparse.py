# SPDX-License-Identifier: Apache-2.0
"""CPU 8-device Pallas & numerical unit tests for Wave-8 decode bridges & muse_sparse opts."""

import functools
import os
import py_compile
import sys
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_force_host_platform_device_count=8 --xla_backend_optimization_level=0",
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _install_lightweight_pkg_stubs() -> None:
    """Allow running pure-JAX/Pallas multi-device tests without TorchTPU installed."""
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = object
        sys.modules["torch"] = torch_stub

    for pkg, rel in (
        ("sglang_tpu", "sglang_tpu"),
        ("sglang_tpu.kernels", "sglang_tpu/kernels"),
        ("sglang_tpu.attention", "sglang_tpu/attention"),
        ("sglang_tpu.attention.pallas", "sglang_tpu/attention/pallas"),
        ("sglang_tpu.moe", "sglang_tpu/moe"),
        ("sglang_tpu.moe.pallas", "sglang_tpu/moe/pallas"),
        (
            "sglang_tpu.moe.pallas.fused_rs_muse",
            "sglang_tpu/moe/pallas/fused_rs_muse",
        ),
    ):
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [os.path.join(ROOT, rel)]
            sys.modules[pkg] = mod


def test_syntax_and_annotations() -> None:
    targets = (
        "sglang_tpu/envs.py",
        "sglang_tpu/kernels/decode_collective_bridges.py",
        "sglang_tpu/kernels/decode_collective_bridges_w8.py",
        "sglang_tpu/models/muse_spark.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/fused_decode_routing.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/fused_decode_sparse.py",
    )
    import tempfile
    with tempfile.TemporaryDirectory() as tmpd:
        for i, rel in enumerate(targets):
            path = os.path.join(ROOT, rel)
            py_compile.compile(path, cfile=os.path.join(tmpd, f"f{i}.pyc"), doraise=True)
    # Verify PEP 563 stays off in custom-op export modules.
    with open(os.path.join(ROOT, "sglang_tpu/kernels/decode_collective_bridges_w8.py")) as f:
        src = f.read()
    assert "from __future__ import annotations" not in src
    print("[PASS] py_compile + PEP-563 annotation guard")


def test_sparse_routing_and_k256_opts() -> None:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing import (
        parse_free_expert_rows_spec,
        route_weights,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        _fp4_gmm_chunked,
        default_fp8_sparse_decode_config,
        default_sparse_decode_config,
    )

    logits = jax.random.normal(jax.random.PRNGKey(42), (64, 256), dtype=jnp.float32) * 1.4 - 3.2
    bias = jax.random.normal(jax.random.PRNGKey(43), (1, 256), dtype=jnp.float32) * 0.15

    w_exact = route_weights(logits, bias)
    w_zero = route_weights(logits, bias, min_p_topk=0.0, free_max_rows=0)
    assert jnp.array_equal(w_exact, w_zero)

    for mp in (0.04, 0.72, 6.0):
        w_mp = route_weights(logits, bias, min_p_topk=mp)
        assert jnp.allclose(jnp.sum(w_mp, axis=1), 1.0, atol=1e-6)
        assert float(jnp.mean(jnp.sum(w_mp > 0, axis=1))) < 8.0

    r, mw, ms = parse_free_expert_rows_spec("2")
    assert (r, mw, ms) == (2, 0.12, 6)
    r2, mw2, ms2 = parse_free_expert_rows_spec("2:0.12:4")
    w_f = route_weights(logits, bias, free_max_rows=r2, free_max_w=mw2, free_min_slot=ms2)
    dropped = (w_exact > 0) & (w_f == 0)
    still_active = jnp.any(w_f > 0, axis=0, keepdims=True)
    assert not bool(jnp.any(dropped & still_active)), "Dropped hit without freeing expert!"
    assert jnp.allclose(jnp.sum(w_f, axis=1), 1.0, atol=1e-6)

    # Verify env overrides for DECODE_M8, CONTIG_PARTS, MIN_P_TOPK, FREE_EXPERT_ROWS.
    saved = {k: os.environ.get(k) for k in (
        "SGLANG_TPU_MUSE_KERNEL_DECODE_M8",
        "SGLANG_TPU_MUSE_KERNEL_DECODE_CONTIG_PARTS",
        "SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK",
        "SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS",
    )}
    try:
        os.environ["SGLANG_TPU_MUSE_KERNEL_DECODE_M8"] = "1"
        os.environ["SGLANG_TPU_MUSE_KERNEL_DECODE_CONTIG_PARTS"] = "1"
        os.environ["SGLANG_TPU_MUSE_DECODE_MIN_P_TOPK"] = "0.75"
        os.environ["SGLANG_TPU_MUSE_DECODE_FREE_EXPERT_ROWS"] = "2:0.11:6"
        cfg = default_sparse_decode_config(64, 8, 256)
        assert cfg.m_buckets == (8, 32)
        assert cfg.weight_mode == "stream3_k2"
        assert cfg.min_p_topk == 0.75
        assert (cfg.free_max_rows, cfg.free_max_w, cfg.free_min_slot) == (2, 0.11, 6)
        cfg_fp8 = default_fp8_sparse_decode_config(64, 8)
        assert cfg_fp8.m_buckets == (8, 32)
        assert cfg_fp8.weight_mode == "w1_k2_w2_1x"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # Verify bit-exactness of scale-hoist & stream3_k2 2-half K-accumulation inside Pallas!
    x = jax.random.normal(jax.random.PRNGKey(50), (16, 4096), dtype=jnp.bfloat16)
    w_u32 = jax.random.randint(jax.random.PRNGKey(51), (512, 512), 0, 0x7FFFFFFF, dtype=jnp.uint32)
    s_bf = (jax.random.uniform(jax.random.PRNGKey(52), (16, 512), dtype=jnp.float32) * 0.05).astype(jnp.bfloat16)
    v_sp = pl.BlockSpec(memory_space=pltpu.VMEM)

    def _run_gmm_mode(mode: str):
        def _k(x_ref, w_ref, s_ref, out_ref):
            if mode == "base":
                out_ref[...] = _fp4_gmm_chunked(
                    x_ref, w_ref, s_ref, 8, 4096, 0, 512, True, 256, hoist_scales=False
                )
            elif mode == "hoist":
                out_ref[...] = _fp4_gmm_chunked(
                    x_ref, w_ref, s_ref, 8, 4096, 0, 512, True, 256, hoist_scales=True
                )
            else:
                p0 = _fp4_gmm_chunked(
                    x_ref, w_ref.at[pl.ds(0, 256), :], s_ref.at[pl.ds(0, 8), :],
                    8, 2048, 0, 512, True, 256, hoist_scales=True, k_offset=0,
                )
                out_ref[...] = _fp4_gmm_chunked(
                    x_ref, w_ref.at[pl.ds(256, 256), :], s_ref.at[pl.ds(8, 8), :],
                    8, 2048, 0, 512, True, 256, hoist_scales=True, k_offset=2048, init_acc=p0,
                )

        return pl.pallas_call(
            _k,
            out_shape=jax.ShapeDtypeStruct((8, 512), jnp.float32),
            in_specs=[v_sp, v_sp, v_sp],
            out_specs=v_sp,
            interpret=pltpu.InterpretParams(detect_races=False),
        )(x, w_u32, s_bf)

    y_base = _run_gmm_mode("base")
    y_hoist = _run_gmm_mode("hoist")
    y_k2 = _run_gmm_mode("k2")
    assert jnp.array_equal(y_base, y_hoist), "Scale hoist is not bit-exact!"
    assert jnp.array_equal(y_base, y_k2), "stream3_k2 K-half split is not bit-exact!"
    print("[PASS] muse_sparse routing breakers + M8 + bit-exact scale hoist + stream3_k2")


def test_w8_collective_bridges_8rank() -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.attention.pallas.decode_qkv_epilogue import (
        build_decode_rope_step_tables,
        run_decode_qkv_epilogue,
    )
    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
        attn_to_moe_tok_sharded_bridge_jax,
        decode_vocab_embed_norm_jax,
        decode_vocab_embed_tok_sharded_jax,
        fp8_dual_qkv_epi_gate_jax,
        moe_post_ag_final_norm_jax,
        moe_post_tok_sharded_final_norm_jax,
        moe_to_attn_qkv_epi_bridge_jax,
        moe_to_attn_tok_sharded_bridge_jax,
    )

    assert len(jax.devices()) >= 8
    mesh = Mesh(jax.devices()[:8], ("model",))
    W, B, H, H_MOE = 8, 64, 2048, 1024
    half_h, q_cols, half_moe = H // 2, H // 4, H_MOE // 2

    def q_fp8(seed: int, k: int, n: int):
        w = jax.random.normal(jax.random.PRNGKey(seed), (k, n), dtype=jnp.float32) * 0.04
        sc = jnp.maximum(jnp.max(jnp.abs(w), axis=0, keepdims=True) / 448.0, 1e-12)
        return (w / sc).astype(jnp.float8_e4m3fn), sc

    cs_step = build_decode_rope_step_tables(
        jnp.arange(B, dtype=jnp.int32) + 7,
        jax.random.normal(jax.random.PRNGKey(4), (512, 128), dtype=jnp.float32) * 0.5,
    )

    # 1. L0 fp8_dual_qkv_epi_gate_jax (RoPE + NoPE)
    attn_in = jax.random.normal(jax.random.PRNGKey(1), (B, H), dtype=jnp.bfloat16) * 0.5
    wq, sq = q_fp8(2, H, 1280)
    wg, sg = q_fp8(3, H, 1024)
    qkv_ref = ((attn_in.astype(jnp.float32) @ wq.astype(jnp.float32)) * sq).astype(jnp.bfloat16)
    gate_ref = ((attn_in.astype(jnp.float32) @ wg.astype(jnp.float32)) * sg).astype(jnp.bfloat16)
    for has_r in (True, False):
        q_r3, kv_r = run_decode_qkv_epilogue(
            qkv_ref, cs_step if has_r else None,
            norm_eps=1e-5, query_scale=5.47, k_scale=0.8, v_scale=1.2, interpret=True,
        )
        q5, kvm, g_out = fp8_dual_qkv_epi_gate_jax(
            attn_in, wq, sq, wg, sg, cs_step if has_r else None,
            qkv_norm_eps=1e-5, query_scale=5.47, k_scale=0.8, v_scale=1.2, interpret=True,
        )
        assert jnp.array_equal(q5, q_r3.reshape(1, B, 8, 2, 128))
        assert jnp.array_equal(kvm.view(jnp.uint8), kv_r.view(jnp.uint8))
        assert jnp.allclose(g_out.astype(jnp.float32), gate_ref.astype(jnp.float32), rtol=1e-2, atol=2e-2)

    # 2. Pallas Vocab Embed Norm + Token-Sharded Entry
    emb_full = jax.random.normal(jax.random.PRNGKey(5), (W * 256, H), dtype=jnp.bfloat16)
    ids = jax.random.randint(jax.random.PRNGKey(6), (B,), 0, W * 256, dtype=jnp.int32)
    l0_gain = 1.0 + jax.random.normal(jax.random.PRNGKey(7), (1, H), dtype=jnp.float32) * 0.1
    raw_emb = emb_full[ids].astype(jnp.float32)
    stream0_ref = (raw_emb * jax.lax.rsqrt(jnp.mean(raw_emb ** 2, axis=-1, keepdims=True) + 1e-5)).astype(jnp.bfloat16).astype(jnp.float32)
    attn0_ref = ((stream0_ref * jax.lax.rsqrt(jnp.mean(stream0_ref ** 2, axis=-1, keepdims=True) + 1e-5)) * l0_gain).astype(jnp.bfloat16)

    st_rep = jax.shard_map(
        functools.partial(decode_vocab_embed_norm_jax, world_size=W, norm_eps=1e-5, interpret=True),
        mesh=mesh, in_specs=(P(), P("model", None)), out_specs=P(), check_vma=False,
    )(ids, emb_full)
    st_sh0, attn0_out = jax.shard_map(
        functools.partial(decode_vocab_embed_tok_sharded_jax, world_size=W, embed_eps=1e-5, l0_eps=1e-5, interpret=True),
        mesh=mesh, in_specs=(P(), P("model", None), P()), out_specs=(P("model", None), P()), check_vma=False,
    )(ids, emb_full, l0_gain)
    assert jnp.allclose(st_rep, stream0_ref, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(st_sh0, stream0_ref, rtol=1e-3, atol=1e-3)
    assert jnp.allclose(attn0_out.astype(jnp.float32), attn0_ref.astype(jnp.float32), rtol=1e-2, atol=2e-2)

    # 3. Token-Sharded Bridge 1b
    gated_full = jax.random.normal(jax.random.PRNGKey(10), (B, W * 1024), dtype=jnp.bfloat16) * 0.4
    wo_full, swo = q_fp8(11, W * 1024, H)
    wpre_full, spre_full = q_fp8(12, H, H_MOE)
    wpre_pair_g = jnp.concatenate([wpre_full[:, :half_moe] if (r & 1) == 0 else wpre_full[:, half_moe:] for r in range(W)], axis=1)
    spre_pair_g = jnp.concatenate([spre_full[:, :half_moe] if (r & 1) == 0 else spre_full[:, half_moe:] for r in range(W)], axis=1)
    pa_a = jnp.ones((1, H), dtype=jnp.float32) * 0.92
    pa_b = jnp.ones((1, H), dtype=jnp.float32) * 0.35
    pff_g = 1.0 + jax.random.normal(jax.random.PRNGKey(13), (1, H), dtype=jnp.float32) * 0.1
    pexp_g = 1.0 + jax.random.normal(jax.random.PRNGKey(14), (1, H_MOE), dtype=jnp.float32) * 0.1
    wr_f32 = jax.random.normal(jax.random.PRNGKey(15), (H, 256), dtype=jnp.float32) * 0.05

    o_parts = [((gated_full[:, r*1024:(r+1)*1024].astype(jnp.float32) @ wo_full[r*1024:(r+1)*1024, :].astype(jnp.float32)) * swo).astype(jnp.bfloat16) for r in range(W)]
    c_pairs = [(o_parts[2*c].astype(jnp.float32) + o_parts[2*c+1].astype(jnp.float32)).astype(jnp.bfloat16) for c in range(4)]
    tree_a = ((c_pairs[0].astype(jnp.float32) + c_pairs[2].astype(jnp.float32)).astype(jnp.bfloat16).astype(jnp.float32) + (c_pairs[1].astype(jnp.float32) + c_pairs[3].astype(jnp.float32)).astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    tree_b = ((c_pairs[0].astype(jnp.float32) + c_pairs[1].astype(jnp.float32)).astype(jnp.bfloat16).astype(jnp.float32) + (c_pairs[2].astype(jnp.float32) + c_pairs[3].astype(jnp.float32)).astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16)
    o_bf = jnp.concatenate([tree_a[:, :q_cols], tree_b[:, q_cols:2*q_cols], tree_a[:, 2*q_cols:3*q_cols], tree_b[:, 3*q_cols:4*q_cols]], axis=1).astype(jnp.float32)
    st1_ref = pa_a * stream0_ref + pa_b * (o_bf * jax.lax.rsqrt(jnp.mean(o_bf ** 2, axis=-1, keepdims=True) + 1e-8))
    rin_ref = (st1_ref * jax.lax.rsqrt(jnp.mean(st1_ref ** 2, axis=-1, keepdims=True) + 1e-5)) * pff_g
    pre_ref = ((rin_ref.astype(jnp.bfloat16).astype(jnp.float32) @ wpre_full.astype(jnp.float32)) * spre_full).astype(jnp.bfloat16)
    exp_in_ref = ((pre_ref.astype(jnp.float32) * jax.lax.rsqrt(jnp.mean(pre_ref.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1e-5)) * pexp_g).astype(jnp.bfloat16)
    logits_ref = (rin_ref @ wr_f32) * 0.1767767

    st1_sh, log_out, exp_out = jax.shard_map(
        functools.partial(attn_to_moe_tok_sharded_bridge_jax, world_size=W, pa_eps=1e-8, pre_ff_eps=1e-5, router_multiplier=0.1767767, pre_exp_eps=1e-5, interpret=True),
        mesh=mesh,
        in_specs=(P("model", None), P(None, "model"), P("model", None), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(), P()),
        out_specs=(P("model", None), P(), P()), check_vma=False,
    )(stream0_ref, gated_full, wo_full, swo, pa_a, pa_b, pff_g, wpre_pair_g, spre_pair_g, wr_f32, pexp_g)
    assert jnp.allclose(st1_sh, st1_ref, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(exp_out.astype(jnp.float32), exp_in_ref.astype(jnp.float32), rtol=1e-2, atol=2e-2)
    assert jnp.allclose(log_out, logits_ref, rtol=1e-5, atol=1e-5)

    # 4. Token-Sharded Bridge 2b, Replicated Bridge 2b (+ Fused QKV/RoPE Epi), & L61 Final Norm
    moe_out_full = jax.random.normal(jax.random.PRNGKey(20), (B, H_MOE), dtype=jnp.bfloat16) * 0.4
    wpost_full, spost_full = q_fp8(21, H_MOE, H)
    wpost_pair_g = jnp.concatenate([wpost_full[:, :half_h] if (r & 1) == 0 else wpost_full[:, half_h:] for r in range(W)], axis=1)
    spost_pair_g = jnp.concatenate([spost_full[:, :half_h] if (r & 1) == 0 else spost_full[:, half_h:] for r in range(W)], axis=1)
    pff_post_g = 1.0 + jax.random.normal(jax.random.PRNGKey(22), (1, H), dtype=jnp.float32) * 0.1
    pf_a = jnp.ones((1, H), dtype=jnp.float32) * 0.94
    pf_b = jnp.ones((1, H), dtype=jnp.float32) * 0.31
    nxt_g = 1.0 + jax.random.normal(jax.random.PRNGKey(23), (1, H), dtype=jnp.float32) * 0.1
    wq_all, sq_all = q_fp8(24, H, W * 1280)
    wg_all, sg_all = q_fp8(25, H, W * 1024)

    post_ref = (jnp.dot(moe_out_full, wpost_full, preferred_element_type=jnp.float32) * spost_full).astype(jnp.bfloat16)
    post_scaled = post_ref.astype(jnp.float32) * pff_post_g
    st2_ref = pf_a * st1_ref + pf_b * (post_scaled * jax.lax.rsqrt(jnp.mean(post_scaled ** 2, axis=-1, keepdims=True) + 1e-8))
    nxt_in_ref = ((st2_ref * jax.lax.rsqrt(jnp.mean(st2_ref ** 2, axis=-1, keepdims=True) + 1e-5)) * nxt_g).astype(jnp.bfloat16)
    qkv_nxt_r0 = (jnp.dot(nxt_in_ref, wq_all[:, :1280], preferred_element_type=jnp.float32) * sq_all[:, :1280]).astype(jnp.bfloat16)
    gate_nxt_r0 = (jnp.dot(nxt_in_ref, wg_all[:, :1024], preferred_element_type=jnp.float32) * sg_all[:, :1024]).astype(jnp.bfloat16)
    q_nxt_ref, kv_nxt_ref = run_decode_qkv_epilogue(
        qkv_nxt_r0, cs_step, norm_eps=1e-5, query_scale=5.47, k_scale=1.0, v_scale=1.0, interpret=True
    )

    st2_sh, q5_b2, kvm_b2, gate_b2 = jax.shard_map(
        functools.partial(
            moe_to_attn_tok_sharded_bridge_jax,
            world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True, next_in_eps=1e-5,
            fuse_qkv_epi=True, qkv_norm_eps=1e-5, query_scale=5.47, k_scale=1.0, v_scale=1.0, interpret=True,
        ),
        mesh=mesh,
        in_specs=(P("model", None), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model"), P()),
        out_specs=(P("model", None), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
        check_vma=False,
    )(st1_ref, moe_out_full, wpost_full, spost_full, pff_post_g.reshape(-1), pf_a.reshape(-1), pf_b.reshape(-1), nxt_g.reshape(-1), wq_all, sq_all, wg_all, sg_all, cs_step)
    assert jnp.allclose(st2_sh, st2_ref, rtol=2e-3, atol=2e-3)
    assert jnp.allclose(q5_b2[:, :, :8, :, :].astype(jnp.float32), q_nxt_ref.reshape(1, B, 8, 2, 128).astype(jnp.float32), rtol=2e-2, atol=8e-2)
    assert jnp.allclose(gate_b2[:, :1024].astype(jnp.float32), gate_nxt_r0.astype(jnp.float32), rtol=1e-2, atol=2e-2)
    assert kvm_b2.shape == (B, W, 4, 128)

    # Also verify plain (non-epi) 4-MiB TP=8 Token-Sharded Bridge 2b & legacy W_post_pair path!
    st2_sh_plain, qkv_b2_plain, gate_b2_plain = jax.shard_map(
        functools.partial(
            moe_to_attn_tok_sharded_bridge_jax,
            world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True, next_in_eps=1e-5,
            fuse_qkv_epi=False, interpret=True,
        ),
        mesh=mesh,
        in_specs=(P("model", None), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model")),
        out_specs=(P("model", None), P(None, "model"), P(None, "model")),
        check_vma=False,
    )(st1_ref, moe_out_full, wpost_full, spost_full, pff_post_g.reshape(-1), pf_a.reshape(-1), pf_b.reshape(-1), nxt_g.reshape(-1), wq_all, sq_all, wg_all, sg_all)
    assert jnp.allclose(st2_sh_plain, st2_ref, rtol=2e-3, atol=2e-3)
    assert jnp.allclose(qkv_b2_plain[:, :1280].astype(jnp.float32), qkv_nxt_r0.astype(jnp.float32), rtol=1e-2, atol=2e-2)
    assert jnp.allclose(gate_b2_plain[:, :1024].astype(jnp.float32), gate_nxt_r0.astype(jnp.float32), rtol=1e-2, atol=2e-2)

    st2_w7, q5_w7, kvm_w7, gate_w7 = jax.shard_map(
        functools.partial(
            moe_to_attn_qkv_epi_bridge_jax,
            world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True, next_in_eps=1e-5,
            sharded_moe_in=True, qkv_norm_eps=1e-5, query_scale=5.47, k_scale=1.0, v_scale=1.0, interpret=True,
        ),
        mesh=mesh,
        in_specs=(P(), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model"), P()),
        out_specs=(P(), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
        check_vma=False,
    )(st1_ref, moe_out_full, wpost_full, spost_full, pff_post_g.reshape(-1), pf_a.reshape(-1), pf_b.reshape(-1), nxt_g.reshape(-1), wq_all, sq_all, wg_all, sg_all, cs_step)
    assert jnp.allclose(st2_w7, st2_ref, rtol=2e-3, atol=2e-3)
    assert jnp.allclose(q5_w7[:, :, :8, :, :].astype(jnp.float32), q_nxt_ref.reshape(1, B, 8, 2, 128).astype(jnp.float32), rtol=2e-2, atol=8e-2)
    assert jnp.allclose(gate_w7[:, :1024].astype(jnp.float32), gate_nxt_r0.astype(jnp.float32), rtol=1e-2, atol=2e-2)
    assert kvm_w7.shape == (B, W, 4, 128)

    st2_bf = st2_ref.astype(jnp.bfloat16).astype(jnp.float32)
    hid_ref = ((st2_bf * jax.lax.rsqrt(jnp.mean(st2_bf ** 2, axis=-1, keepdims=True) + 1e-5)) * nxt_g).astype(jnp.bfloat16)
    hid_w7 = jax.shard_map(
        functools.partial(moe_post_ag_final_norm_jax, world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True, final_eps=1e-5, sharded_moe_in=True, interpret=True),
        mesh=mesh, in_specs=(P(), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P()), out_specs=P(), check_vma=False,
    )(st1_ref, moe_out_full, wpost_full, spost_full, pff_post_g.reshape(-1), pf_a.reshape(-1), pf_b.reshape(-1), nxt_g.reshape(-1))
    hid_w8 = jax.shard_map(
        functools.partial(moe_post_tok_sharded_final_norm_jax, world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True, final_eps=1e-5, return_stream_too=False, interpret=True),
        mesh=mesh, in_specs=(P("model", None), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P()), out_specs=P(), check_vma=False,
    )(st1_ref, moe_out_full, wpost_full, spost_full, pff_post_g.reshape(-1), pf_a.reshape(-1), pf_b.reshape(-1), nxt_g.reshape(-1))
    hid_w8_pair = jax.shard_map(
        functools.partial(moe_post_tok_sharded_final_norm_jax, world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True, final_eps=1e-5, return_stream_too=False, interpret=True),
        mesh=mesh, in_specs=(P("model", None), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P()), out_specs=P(), check_vma=False,
    )(st1_ref, moe_out_full, wpost_pair_g, spost_pair_g, pff_post_g, pf_a, pf_b, nxt_g)
    st2_fn_st, hid_w8_st = jax.shard_map(
        functools.partial(
            moe_post_tok_sharded_final_norm_jax,
            world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True, final_eps=1e-5,
            return_stream_too=True, interpret=True,
        ),
        mesh=mesh,
        in_specs=(P("model", None), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P()),
        out_specs=(P("model", None), P()),
        check_vma=False,
    )(st1_ref, moe_out_full, wpost_full, spost_full, pff_post_g.reshape(-1), pf_a.reshape(-1), pf_b.reshape(-1), nxt_g.reshape(-1))
    hid_ref_exact_st = ((st2_ref * jax.lax.rsqrt(jnp.mean(st2_ref ** 2, axis=-1, keepdims=True) + 1e-5)) * nxt_g).astype(jnp.bfloat16)
    assert jnp.allclose(st2_fn_st, st2_ref, rtol=2e-3, atol=2e-3)
    assert jnp.allclose(hid_w8_st.astype(jnp.float32), hid_ref_exact_st.astype(jnp.float32), rtol=1e-2, atol=2e-2)
    assert jnp.allclose(hid_w7.astype(jnp.float32), hid_ref.astype(jnp.float32), rtol=1e-2, atol=2e-2)
    assert jnp.allclose(hid_w8.astype(jnp.float32), hid_ref.astype(jnp.float32), rtol=1e-2, atol=2e-2)
    assert jnp.allclose(hid_w8_pair.astype(jnp.float32), hid_ref.astype(jnp.float32), rtol=1e-2, atol=2e-2)

    # 5. Also verify B=128 (r_tok=16, r_pair=32, half=64 multi-strip path) end-to-end across all 8 ranks!
    st1_128 = jnp.concatenate([st1_ref, st1_ref * 0.85], axis=0)
    moe_128 = jnp.concatenate([moe_out_full, moe_out_full * -0.7], axis=0)
    post_128 = (jnp.dot(moe_128, wpost_full, preferred_element_type=jnp.float32) * spost_full).astype(jnp.bfloat16)
    ps_128 = post_128.astype(jnp.float32) * pff_post_g
    st2_128_ref = pf_a * st1_128 + pf_b * (ps_128 * jax.lax.rsqrt(jnp.mean(ps_128 ** 2, axis=-1, keepdims=True) + 1e-8))
    nxt_128_ref = ((st2_128_ref * jax.lax.rsqrt(jnp.mean(st2_128_ref ** 2, axis=-1, keepdims=True) + 1e-5)) * nxt_g).astype(jnp.bfloat16)
    qkv_128_r0 = (jnp.dot(nxt_128_ref, wq_all[:, :1280], preferred_element_type=jnp.float32) * sq_all[:, :1280]).astype(jnp.bfloat16)
    q_128_ref, _ = run_decode_qkv_epilogue(
        qkv_128_r0, None, norm_eps=1e-5, query_scale=5.47, k_scale=1.0, v_scale=1.0, interpret=True
    )
    st2_128_out, q5_128_out, _, _ = jax.shard_map(
        functools.partial(
            moe_to_attn_tok_sharded_bridge_jax,
            world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True, next_in_eps=1e-5,
            fuse_qkv_epi=True, qkv_norm_eps=1e-5, query_scale=5.47, k_scale=1.0, v_scale=1.0, interpret=True,
        ),
        mesh=mesh,
        in_specs=(P("model", None), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model")),
        out_specs=(P("model", None), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
        check_vma=False,
    )(st1_128, moe_128, wpost_full, spost_full, pff_post_g.reshape(-1), pf_a.reshape(-1), pf_b.reshape(-1), nxt_g.reshape(-1), wq_all, sq_all, wg_all, sg_all)
    assert jnp.allclose(st2_128_out, st2_128_ref, rtol=2e-3, atol=2e-3)
    assert jnp.allclose(q5_128_out[:, :, :8, :, :].astype(jnp.float32), q_128_ref.reshape(1, 128, 8, 2, 128).astype(jnp.float32), rtol=2e-2, atol=8e-2)
    print("[PASS] 8-rank Token-Sharded Bridge 1b + Zero-Post-Norm-ICI Bridge 2b (B=64/128, RoPE/NoPE) + L61 Final Norm + Pallas Embed")


def test_tpu7x_spmd_mlir_export_clean() -> None:
    """Export all 8 Wave-8 Pallas ops to TPU7x StableHLO MLIR and assert zero SPMD partitionId (#1494)."""
    from contextlib import contextmanager

    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic import tpu_info as mosaic_tpu_info
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
        attn_to_moe_tok_sharded_bridge_jax,
        decode_vocab_embed_norm_jax,
        decode_vocab_embed_tok_sharded_jax,
        fp8_dual_qkv_epi_gate_jax,
        moe_post_ag_final_norm_jax,
        moe_post_tok_sharded_final_norm_jax,
        moe_to_attn_qkv_epi_bridge_jax,
        moe_to_attn_tok_sharded_bridge_jax,
    )

    @contextmanager
    def _tpu7x_ctx():
        if hasattr(mosaic_tpu_info, "chip_version_from_device_kind"):
            import sglang_tpu.pallas_export as pe

            orig_kind = pe.get_pallas_tpu_device_kind
            pe.get_pallas_tpu_device_kind = lambda: "TPU7x"
            try:
                with pe.tpu_info_for_cpu_export():
                    yield
            finally:
                pe.get_pallas_tpu_device_kind = orig_kind
        else:
            orig_gk = mosaic_tpu_info.core.get_device_kind
            orig_nc = mosaic_tpu_info.core.get_num_device_cores
            orig_is = mosaic_tpu_info.is_tpu_device
            mosaic_tpu_info.core.get_device_kind = lambda: "TPU7x"
            mosaic_tpu_info.core.get_num_device_cores = lambda: 1
            mosaic_tpu_info.is_tpu_device = lambda: True
            try:
                yield
            finally:
                mosaic_tpu_info.core.get_device_kind = orig_gk
                mosaic_tpu_info.core.get_num_device_cores = orig_nc
                mosaic_tpu_info.is_tpu_device = orig_is

    mesh = Mesh(jax.devices()[:8], ("model",))
    W, B, H, H_MOE = 8, 64, 8192, 4096

    def sd(shape, dtype, spec):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, spec))

    def _assert_spmd_export_clean(
        name: str, fn, in_specs, out_specs, shapes_dtypes, *, check_t16_bf16: bool = False
    ):
        from jax._src import tpu_custom_call

        mosaic_asms: list[str] = []
        orig_lower = tpu_custom_call._lower_to_custom_call_config

        def _spy_lower(module, **kw):
            mosaic_asms.append(module.operation.get_asm(enable_debug_info=False))
            return orig_lower(module, **kw)

        tpu_custom_call._lower_to_custom_call_config = _spy_lower
        try:
            out_sh = (
                tuple(NamedSharding(mesh, sp) for sp in out_specs)
                if isinstance(out_specs, tuple)
                else NamedSharding(mesh, out_specs)
            )
            wrapped = jax.jit(
                lambda *xs: jax.shard_map(
                    fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
                )(*xs),
                out_shardings=out_sh,
            )
            avals = [sd(s, dt, sp) for (s, dt), sp in zip(shapes_dtypes, in_specs, strict=True)]
            exp = jax.export.export(wrapped, platforms=("tpu",))(*avals)
            mlir = exp.mlir_module()
        finally:
            tpu_custom_call._lower_to_custom_call_config = orig_lower

        assert "stablehlo.partition_id" not in mlir, (
            f"{name} leaked stablehlo.partition_id into SPMD MLIR!"
        )
        if check_t16_bf16:
            import re

            assert mosaic_asms, f"{name} did not emit a Mosaic TPU kernel!"
            m_txt = mosaic_asms[0]
            for line in m_txt.splitlines():
                if "tpu.concatenate" in line:
                    assert "vector<1x" not in line, f"{name} emitted 1-row tpu.concatenate: {line}"
                if ("vector.load" in line or "vector.store" in line) and "xbf16>" in line:
                    assert re.search(r"vector<(?:\d+x)*(?:1|8)x\d+xbf16>", line) is None, (
                        f"{name} emitted non-T(16,128) bf16 vector op (#1531 e2003): {line}"
                    )

    with _tpu7x_ctx():
        # 1. decode_vocab_embed_norm_jax across ALL buckets incl. exact #1531 b32 reproducer!
        for b_emb in (8, 16, 32, 64, 128, 256, 384):
            _assert_spmd_export_clean(
                f"decode_vocab_embed_norm_b{b_emb}",
                functools.partial(decode_vocab_embed_norm_jax, world_size=W, norm_eps=1e-5),
                (P(), P("model", None)),
                P(),
                [((b_emb,), jnp.int32), ((W * 25008, H), jnp.bfloat16)],
                check_t16_bf16=True,
            )
        # 2. decode_vocab_embed_tok_sharded_jax across B=64..384 (DFlash B=256/384!)
        for b_sh in (64, 128, 256, 384):
            _assert_spmd_export_clean(
                f"decode_vocab_embed_tok_sharded_b{b_sh}",
                functools.partial(
                    decode_vocab_embed_tok_sharded_jax, world_size=W, embed_eps=1e-5, l0_eps=1e-5
                ),
                (P(), P("model", None), P()),
                (P("model", None), P()),
                [((b_sh,), jnp.int32), ((W * 25008, H), jnp.bfloat16), ((1, H), jnp.float32)],
                check_t16_bf16=(b_sh >= 128),
            )
        # 3. fp8_dual_qkv_epi_gate_jax (RoPE at B=64 and DFlash B=384)
        for b_q in (64, 384):
            _assert_spmd_export_clean(
                f"fp8_dual_qkv_epi_gate_rope_b{b_q}",
                functools.partial(
                    fp8_dual_qkv_epi_gate_jax,
                    qkv_norm_eps=1e-5, query_scale=5.47, k_scale=1.0, v_scale=1.0,
                ),
                (P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model"), P()),
                (P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
                [
                    ((b_q, H), jnp.bfloat16),
                    ((H, W * 1280), jnp.float8_e4m3fn),
                    ((1, W * 1280), jnp.float32),
                    ((H, W * 1024), jnp.float8_e4m3fn),
                    ((1, W * 1024), jnp.float32),
                    ((b_q, 2, 128), jnp.float32),
                ],
            )
        # 4. attn_to_moe_tok_sharded_bridge_jax (B=64 & DFlash B=384, full H=8192, H_MOE=4096)
        for b_ts in (64, 384):
            _assert_spmd_export_clean(
                f"attn_to_moe_tok_sharded_bridge_b{b_ts}",
                functools.partial(
                    attn_to_moe_tok_sharded_bridge_jax,
                    world_size=W, pa_eps=1e-8, pre_ff_eps=1e-5,
                    router_multiplier=0.1767767, pre_exp_eps=1e-5,
                ),
                (P("model", None), P(None, "model"), P("model", None), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(), P()),
                (P("model", None), P(), P()),
                [
                    ((b_ts, H), jnp.float32),
                    ((b_ts, W * 1024), jnp.bfloat16),
                    ((W * 1024, H), jnp.float8_e4m3fn),
                    ((1, H), jnp.float32),
                    ((1, H), jnp.float32),
                    ((1, H), jnp.float32),
                    ((1, H), jnp.float32),
                    ((H, W * 2048), jnp.float8_e4m3fn),
                    ((1, W * 2048), jnp.float32),
                    ((H, 256), jnp.bfloat16),
                    ((1, H_MOE), jnp.float32),
                ],
            )
        # 5. moe_to_attn_tok_sharded_bridge_jax (4-MiB TP=8 W_post + 1D gains, B=64..384)
        for b_ts in (64, 128, 256, 384):
            _assert_spmd_export_clean(
                f"moe_to_attn_tok_sharded_bridge_tp8_epi_b{b_ts}",
                functools.partial(
                    moe_to_attn_tok_sharded_bridge_jax,
                    world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True, next_in_eps=1e-5,
                    fuse_qkv_epi=True, qkv_norm_eps=1e-5, query_scale=5.47, k_scale=1.0, v_scale=1.0,
                ),
                (P("model", None), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model"), P()),
                (P("model", None), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
                [
                    ((b_ts, H), jnp.float32),
                    ((b_ts, H_MOE), jnp.bfloat16),
                    ((H_MOE, H), jnp.float8_e4m3fn),
                    ((1, H), jnp.float32),
                    ((H,), jnp.float32),
                    ((H,), jnp.float32),
                    ((H,), jnp.float32),
                    ((H,), jnp.float32),
                    ((H, W * 1280), jnp.float8_e4m3fn),
                    ((1, W * 1280), jnp.float32),
                    ((H, W * 1024), jnp.float8_e4m3fn),
                    ((1, W * 1024), jnp.float32),
                    ((b_ts, 2, 128), jnp.float32),
                ],
                check_t16_bf16=(b_ts >= 128),
            )
            if b_ts >= 128:
                _assert_spmd_export_clean(
                    f"moe_to_attn_tok_sharded_bridge_tp8_plain_b{b_ts}",
                    functools.partial(
                        moe_to_attn_tok_sharded_bridge_jax,
                        world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True, next_in_eps=1e-5,
                        fuse_qkv_epi=False,
                    ),
                    (P("model", None), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model")),
                    (P("model", None), P(None, "model"), P(None, "model")),
                    [
                        ((b_ts, H), jnp.float32),
                        ((b_ts, H_MOE), jnp.bfloat16),
                        ((H_MOE, H), jnp.float8_e4m3fn),
                        ((1, H), jnp.float32),
                        ((H,), jnp.float32),
                        ((H,), jnp.float32),
                        ((H,), jnp.float32),
                        ((H,), jnp.float32),
                        ((H, W * 1280), jnp.float8_e4m3fn),
                        ((1, W * 1280), jnp.float32),
                        ((H, W * 1024), jnp.float8_e4m3fn),
                        ((1, W * 1024), jnp.float32),
                    ],
                    check_t16_bf16=True,
                )
        # 6. moe_to_attn_qkv_epi_bridge_jax (B=64 & DFlash B=384) & 7/8. L61 Final Norm
        for b_q in (64, 384):
            _assert_spmd_export_clean(
                f"moe_to_attn_qkv_epi_bridge_b{b_q}",
                functools.partial(
                    moe_to_attn_qkv_epi_bridge_jax,
                    world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True, next_in_eps=1e-5,
                    sharded_moe_in=True, qkv_norm_eps=1e-5, query_scale=5.47, k_scale=1.0, v_scale=1.0,
                ),
                (P(), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model"), P()),
                (P(), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
                [
                    ((b_q, H), jnp.float32),
                    ((b_q, H_MOE), jnp.bfloat16),
                    ((H_MOE, H), jnp.float8_e4m3fn),
                    ((1, H), jnp.float32),
                    ((H,), jnp.float32),
                    ((H,), jnp.float32),
                    ((H,), jnp.float32),
                    ((H,), jnp.float32),
                    ((H, W * 1280), jnp.float8_e4m3fn),
                    ((1, W * 1280), jnp.float32),
                    ((H, W * 1024), jnp.float8_e4m3fn),
                    ((1, W * 1024), jnp.float32),
                    ((b_q, 2, 128), jnp.float32),
                ],
            )
        _assert_spmd_export_clean(
            "moe_post_ag_final_norm_b384",
            functools.partial(moe_post_ag_final_norm_jax, world_size=W, sharded_moe_in=True),
            (P(), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P()),
            P(),
            [
                ((384, H), jnp.float32),
                ((384, H_MOE), jnp.bfloat16),
                ((H_MOE, H), jnp.float8_e4m3fn),
                ((1, H), jnp.float32),
                ((H,), jnp.float32),
                ((H,), jnp.float32),
                ((H,), jnp.float32),
                ((H,), jnp.float32),
            ],
        )
        for b_fn in (64, 384):
            _assert_spmd_export_clean(
                f"moe_post_tok_sharded_final_norm_tp8_b{b_fn}",
                functools.partial(moe_post_tok_sharded_final_norm_jax, world_size=W, return_stream_too=False),
                (P("model", None), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P()),
                P(),
                [
                    ((b_fn, H), jnp.float32),
                    ((b_fn, H_MOE), jnp.bfloat16),
                    ((H_MOE, H), jnp.float8_e4m3fn),
                    ((1, H), jnp.float32),
                    ((H,), jnp.float32),
                    ((H,), jnp.float32),
                    ((H,), jnp.float32),
                    ((H,), jnp.float32),
                ],
                check_t16_bf16=(b_fn >= 128),
            )
        for b_lc in (8, 16):
            _assert_spmd_export_clean(
                f"moe_to_attn_qkv_epi_b{b_lc}",
                functools.partial(
                    moe_to_attn_qkv_epi_bridge_jax,
                    world_size=W, sharded_moe_in=False, query_scale=5.47,
                ),
                (P(), P(), P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model"), P()),
                (P(), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
                [
                    ((b_lc, H), jnp.float32),
                    ((b_lc, H_MOE), jnp.bfloat16),
                    ((H_MOE, H), jnp.float8_e4m3fn),
                    ((1, H), jnp.float32),
                    ((1, H), jnp.float32),
                    ((1, H), jnp.float32),
                    ((1, H), jnp.float32),
                    ((1, H), jnp.float32),
                    ((H, W * 1280), jnp.float8_e4m3fn),
                    ((1, W * 1280), jnp.float32),
                    ((H, W * 1024), jnp.float8_e4m3fn),
                    ((1, W * 1024), jnp.float32),
                    ((b_lc, 2, 128), jnp.float32),
                ],
                check_t16_bf16=(b_lc >= 16),
            )
        _assert_spmd_export_clean(
            "tok_b1_2d_gains_ws_direct_b64",
            functools.partial(attn_to_moe_tok_sharded_bridge_jax, world_size=W),
            (P("model", None), P(None, "model"), P("model", None), P("model"), P(), P(), P(), P(None, "model"), P(None, "model"), P(), P()),
            (P("model", None), P(), P()),
            [
                ((64, H), jnp.float32),
                ((64, W * 1024), jnp.bfloat16),
                ((W * 1024, H), jnp.float8_e4m3fn),
                ((W, H), jnp.float32),
                ((1, H), jnp.float32),
                ((1, H), jnp.float32),
                ((1, H), jnp.float32),
                ((H, W * 2048), jnp.float8_e4m3fn),
                ((1, W * 2048), jnp.float32),
                ((H, 256), jnp.bfloat16),
                ((1, H_MOE), jnp.float32),
            ],
            check_t16_bf16=False,
        )
    print("[PASS] TPU7x Mosaic T(16,128) + SPMD StableHLO export clean across B=8..384")


def test_w10_tok_b2_fast_races_and_llo_invariants() -> None:
    """Verify 0 DMA races (`detect_races=True`), legacy env fallback, and 13-matmul/0-scf.for MLIR."""
    import re
    import jax
    import jax.numpy as jnp
    from jax._src import tpu_custom_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import sglang_tpu.pallas_export as pe
    from sglang_tpu.attention.pallas.decode_qkv_epilogue import build_decode_rope_step_tables
    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
        moe_post_tok_sharded_final_norm_jax,
        moe_to_attn_qkv_epi_bridge_jax,
        moe_to_attn_tok_sharded_bridge_jax,
    )

    mesh = Mesh(jax.devices()[:8], ("model",))
    W, B, H, H_MOE = 8, 64, 2048, 1024
    eager_race = pltpu.InterpretParams(dma_execution_mode="eager", detect_races=True)

    def q_fp8(seed: int, k: int, n: int):
        w = jax.random.normal(jax.random.PRNGKey(seed), (k, n), dtype=jnp.float32) * 0.04
        sc = jnp.maximum(jnp.max(jnp.abs(w), axis=0, keepdims=True) / 448.0, 1e-12)
        return (w / sc).astype(jnp.float8_e4m3fn), sc

    st = jax.device_put(jax.random.normal(jax.random.PRNGKey(70), (B, H), dtype=jnp.float32), NamedSharding(mesh, P("model", None)))
    st_rep = jax.device_put(jax.random.normal(jax.random.PRNGKey(70), (B, H), dtype=jnp.float32), NamedSharding(mesh, P()))
    mo = jax.device_put(jax.random.normal(jax.random.PRNGKey(71), (B, H_MOE), dtype=jnp.bfloat16), NamedSharding(mesh, P("model", None)))
    wp, sp = q_fp8(72, H_MOE, H)
    wp_d = jax.device_put(wp, NamedSharding(mesh, P(None, "model")))
    sp_d = jax.device_put(sp, NamedSharding(mesh, P(None, "model")))
    wq, sq = q_fp8(73, H, W * 1280)
    wg, sg = q_fp8(74, H, W * 1024)
    wq_d = jax.device_put(wq, NamedSharding(mesh, P(None, "model")))
    sq_d = jax.device_put(sq, NamedSharding(mesh, P(None, "model")))
    wg_d = jax.device_put(wg, NamedSharding(mesh, P(None, "model")))
    sg_d = jax.device_put(sg, NamedSharding(mesh, P(None, "model")))
    g1 = jax.device_put(jnp.ones((H,), dtype=jnp.float32) * 1.05, NamedSharding(mesh, P()))
    al = jax.device_put(jnp.ones((H,), dtype=jnp.float32) * 0.95, NamedSharding(mesh, P()))
    be = jax.device_put(jnp.ones((H,), dtype=jnp.float32) * 0.85, NamedSharding(mesh, P()))
    g2 = jax.device_put(jnp.ones((H,), dtype=jnp.float32) * 1.10, NamedSharding(mesh, P()))
    cs = jax.device_put(
        build_decode_rope_step_tables(
            jnp.arange(B, dtype=jnp.int32) + 3,
            jax.random.normal(jax.random.PRNGKey(75), (256, 128), dtype=jnp.float32) * 0.5,
        ),
        NamedSharding(mesh, P()),
    )

    in_sp_b2 = (P("model", None), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model"), P())
    out_sp_b2 = (P("model", None), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model"))

    def _run_b2(interp_arg):
        return jax.shard_map(
            functools.partial(
                moe_to_attn_tok_sharded_bridge_jax,
                world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True,
                next_in_eps=1e-5, next_in_scale_input=True,
                fuse_qkv_epi=True, qkv_norm_eps=1e-5, query_scale=5.47, k_scale=0.9, v_scale=1.1,
                interpret=interp_arg,
            ),
            mesh=mesh, in_specs=in_sp_b2, out_specs=out_sp_b2, check_vma=False,
        )(st, mo, wp_d, sp_d, g1, al, be, g2, wq_d, sq_d, wg_d, sg_d, cs)

    # 1. Eager DMA race detection on fast tok_b2_p8_epi + fast tok_fnorm_p8 + w7 qkv_epi
    fast_out = _run_b2(eager_race)
    jax.block_until_ready(fast_out)

    saved_env = os.environ.get("SGLANG_TPU_MUSE_DECODE_TOK_B2_FAST")
    try:
        os.environ["SGLANG_TPU_MUSE_DECODE_TOK_B2_FAST"] = "0"
        leg_out = _run_b2(True)
    finally:
        if saved_env is None:
            os.environ.pop("SGLANG_TPU_MUSE_DECODE_TOK_B2_FAST", None)
        else:
            os.environ["SGLANG_TPU_MUSE_DECODE_TOK_B2_FAST"] = saved_env

    assert jnp.allclose(fast_out[0], leg_out[0], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(fast_out[1].astype(jnp.float32), leg_out[1].astype(jnp.float32), rtol=1e-2, atol=2e-2)
    assert jnp.array_equal(fast_out[2].view(jnp.uint8), leg_out[2].view(jnp.uint8))
    assert jnp.allclose(fast_out[3].astype(jnp.float32), leg_out[3].astype(jnp.float32), rtol=1e-2, atol=2e-2)

    for ret_st in (False, True):
        fn_out = jax.shard_map(
            functools.partial(
                moe_post_tok_sharded_final_norm_jax,
                world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True,
                final_eps=1e-5, final_scale_input=True,
                return_stream_too=ret_st, interpret=eager_race,
            ),
            mesh=mesh,
            in_specs=(P("model", None), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P()),
            out_specs=(P("model", None), P()) if ret_st else P(),
            check_vma=False,
        )(st, mo, wp_d, sp_d, g1, al, be, g2)
        jax.block_until_ready(fn_out)

    w7_epi_out = jax.shard_map(
        functools.partial(
            moe_to_attn_qkv_epi_bridge_jax,
            world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True,
            next_in_eps=1e-5, next_in_scale_input=True, sharded_moe_in=True,
            qkv_norm_eps=1e-5, query_scale=5.47, k_scale=0.9, v_scale=1.1,
            interpret=eager_race,
        ),
        mesh=mesh,
        in_specs=(P(), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model"), P()),
        out_specs=(P(), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
        check_vma=False,
    )(st_rep, mo, wp_d, sp_d, g1, al, be, g2, wq_d, sq_d, wg_d, sg_d, cs)
    jax.block_until_ready(w7_epi_out)

    # 2. Verify TPU7x B=64 H=8192 Mosaic MLIR hits theoretical floor: 13 tpu.matmul & 0 scf.for!
    asms: list[str] = []
    orig_lower = tpu_custom_call._lower_to_custom_call_config
    orig_kind = pe.get_pallas_tpu_device_kind
    pe.get_pallas_tpu_device_kind = lambda: "TPU7x"
    tpu_custom_call._lower_to_custom_call_config = lambda m, **kw: (
        asms.append(m.operation.get_asm(enable_debug_info=False)),
        orig_lower(m, **kw),
    )[1]
    try:
        with pe.tpu_info_for_cpu_export():
            H8, HMOE8 = 8192, 4096
            sd = lambda s, dt, sp: jax.ShapeDtypeStruct(s, dt, sharding=NamedSharding(mesh, sp))
            avals = [
                sd((64, H8), jnp.float32, P("model", None)),
                sd((64, HMOE8), jnp.bfloat16, P("model", None)),
                sd((HMOE8, H8), jnp.float8_e4m3fn, P(None, "model")),
                sd((1, H8), jnp.float32, P(None, "model")),
                sd((H8,), jnp.float32, P()),
                sd((H8,), jnp.float32, P()),
                sd((H8,), jnp.float32, P()),
                sd((H8,), jnp.float32, P()),
                sd((H8, W * 1280), jnp.float8_e4m3fn, P(None, "model")),
                sd((1, W * 1280), jnp.float32, P(None, "model")),
                sd((H8, W * 1024), jnp.float8_e4m3fn, P(None, "model")),
                sd((1, W * 1024), jnp.float32, P(None, "model")),
                sd((64, 2, 128), jnp.float32, P()),
            ]
            wrapped = jax.jit(
                lambda *xs: jax.shard_map(
                    functools.partial(
                        moe_to_attn_tok_sharded_bridge_jax,
                        world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True,
                        next_in_eps=1e-5, fuse_qkv_epi=True,
                        qkv_norm_eps=1e-5, query_scale=5.47, k_scale=1.0, v_scale=1.0,
                    ),
                    mesh=mesh, in_specs=in_sp_b2, out_specs=out_sp_b2, check_vma=False,
                )(*xs),
                out_shardings=tuple(NamedSharding(mesh, sp) for sp in out_sp_b2),
            )
            jax.export.export(wrapped, platforms=("tpu",))(*avals)
    finally:
        tpu_custom_call._lower_to_custom_call_config = orig_lower
        pe.get_pallas_tpu_device_kind = orig_kind

    m_txt = asms[0]
    n_mm = len(re.findall(r"tpu\.matmul", m_txt))
    n_for = len(re.findall(r"scf\.for", m_txt))
    assert n_mm == 13, f"Expected 13 tpu.matmul on B=64 tok_b2_p8_epi, got {n_mm}"
    assert n_for == 0, f"Expected 0 scf.for on B=64 tok_b2_p8_epi, got {n_for}"
    print("[PASS] Wave-10 tok_b2_p8 eager detect_races=True + legacy fallback + 13-matmul/0-scf.for")


def test_w11_norm_before_s3_b8_b16_and_2d_gains() -> None:
    """Verify Wave-11 Normalize-Before-s3 bit-exactness + 0 races, B=8/16 QKV_EPI, and 2D gains."""
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as tpu_interp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import sglang_tpu.pallas_export as pe
    from sglang_tpu.attention.pallas.decode_qkv_epilogue import (
        build_decode_rope_step_tables,
        run_decode_qkv_epilogue,
    )
    from sglang_tpu.kernels.decode_collective_bridges_w8 import (
        attn_to_moe_tok_sharded_bridge_jax,
        fp8_dual_qkv_epi_gate_jax,
        moe_post_ag_final_norm_jax,
        moe_to_attn_qkv_epi_bridge_jax,
    )

    mesh = Mesh(jax.devices()[:8], ("model",))
    W, H, H_MOE = 8, 2048, 1024
    eager_race = pltpu.InterpretParams(dma_execution_mode="eager", detect_races=True)

    def q_fp8(seed: int, k: int, n: int):
        w = jax.random.normal(jax.random.PRNGKey(seed), (k, n), dtype=jnp.float32) * 0.04
        sc = jnp.maximum(jnp.max(jnp.abs(w), axis=0, keepdims=True) / 448.0, 1e-12)
        return (w / sc).astype(jnp.float8_e4m3fn), sc

    wp, sp = q_fp8(90, H_MOE, H)
    wq, sq = q_fp8(91, H, W * 1280)
    wg, sg = q_fp8(92, H, W * 1024)
    wp_d = jax.device_put(wp, NamedSharding(mesh, P(None, "model")))
    sp_d = jax.device_put(sp, NamedSharding(mesh, P(None, "model")))
    wq_d = jax.device_put(wq, NamedSharding(mesh, P(None, "model")))
    sq_d = jax.device_put(sq, NamedSharding(mesh, P(None, "model")))
    wg_d = jax.device_put(wg, NamedSharding(mesh, P(None, "model")))
    sg_d = jax.device_put(sg, NamedSharding(mesh, P(None, "model")))
    # Use pre-shaped (1, H) float32 gains & residual gate alphas/betas!
    g1_2d = jax.device_put(
        jax.random.uniform(jax.random.PRNGKey(93), (1, H), dtype=jnp.float32) + 0.5,
        NamedSharding(mesh, P()),
    )
    al_2d = jax.device_put(
        jax.random.uniform(jax.random.PRNGKey(94), (1, H), dtype=jnp.float32) + 0.5,
        NamedSharding(mesh, P()),
    )
    be_2d = jax.device_put(
        jax.random.uniform(jax.random.PRNGKey(95), (1, H), dtype=jnp.float32) + 0.5,
        NamedSharding(mesh, P()),
    )
    g2_2d = jax.device_put(
        jax.random.uniform(jax.random.PRNGKey(96), (1, H), dtype=jnp.float32) + 0.5,
        NamedSharding(mesh, P()),
    )
    rope_tbl = jax.random.normal(jax.random.PRNGKey(97), (256, 128), dtype=jnp.float32) * 0.4

    # 1. Bit-exact parity (max_abs_diff == 0.0) & 0 DMA races for Normalize-Before-s3 (B=32, 64)
    for b_sz, sh_in in [(32, False), (64, True)]:
        st_rep = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(100 + b_sz), (b_sz, H), dtype=jnp.float32),
            NamedSharding(mesh, P()),
        )
        mo_sp = P("model", None) if sh_in else P()
        mo_d = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(110 + b_sz), (b_sz, H_MOE), dtype=jnp.bfloat16),
            NamedSharding(mesh, mo_sp),
        )
        cs_d = jax.device_put(
            build_decode_rope_step_tables(jnp.arange(b_sz, dtype=jnp.int32) + 5, rope_tbl),
            NamedSharding(mesh, P()),
        )
        in_sp = (P(), mo_sp, P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model"), P())
        out_sp = (P(), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model"))

        def _run_qepi(nbs3_flag: bool, interp_arg):
            return jax.shard_map(
                functools.partial(
                    moe_to_attn_qkv_epi_bridge_jax,
                    world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True,
                    next_in_eps=1e-5, next_in_scale_input=False, sharded_moe_in=sh_in,
                    qkv_norm_eps=1e-5, query_scale=5.47, k_scale=0.8, v_scale=1.2,
                    norm_before_s3=nbs3_flag, interpret=interp_arg,
                ),
                mesh=mesh, in_specs=in_sp, out_specs=out_sp, check_vma=False,
            )(st_rep, mo_d, wp_d, sp_d, g1_2d, al_2d, be_2d, g2_2d, wq_d, sq_d, wg_d, sg_d, cs_d)

        out_nbs3 = _run_qepi(True, eager_race)
        jax.block_until_ready(out_nbs3)
        assert not tpu_interp.races.races_found, f"DMA race in qepi NBS3 B={b_sz}"
        out_leg = _run_qepi(False, True)
        for idx, (a, b) in enumerate(zip(out_nbs3, out_leg, strict=True)):
            af = a.view(jnp.uint8) if a.dtype == jnp.float8_e4m3fn else a.astype(jnp.float32)
            bf = b.view(jnp.uint8) if b.dtype == jnp.float8_e4m3fn else b.astype(jnp.float32)
            diff = float(jnp.max(jnp.abs(af.astype(jnp.float32) - bf.astype(jnp.float32))))
            assert diff == 0.0, f"qepi NBS3 vs legacy mismatch at B={b_sz} out[{idx}]: {diff}"

        in_fn = (P(), mo_sp, P(None, "model"), P(None, "model"), P(), P(), P(), P())
        fn_nbs3 = jax.shard_map(
            functools.partial(
                moe_post_ag_final_norm_jax,
                world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True,
                final_eps=1e-5, final_scale_input=True, sharded_moe_in=sh_in,
                norm_before_s3=True, interpret=eager_race,
            ),
            mesh=mesh, in_specs=in_fn, out_specs=P(), check_vma=False,
        )(st_rep, mo_d, wp_d, sp_d, g1_2d, al_2d, be_2d, g2_2d)
        jax.block_until_ready(fn_nbs3)
        assert not tpu_interp.races.races_found, f"DMA race in fnorm NBS3 B={b_sz}"
        fn_leg = jax.shard_map(
            functools.partial(
                moe_post_ag_final_norm_jax,
                world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True,
                final_eps=1e-5, final_scale_input=True, sharded_moe_in=sh_in,
                norm_before_s3=False, interpret=True,
            ),
            mesh=mesh, in_specs=in_fn, out_specs=P(), check_vma=False,
        )(st_rep, mo_d, wp_d, sp_d, g1_2d, al_2d, be_2d, g2_2d)
        fn_diff = float(jnp.max(jnp.abs(fn_nbs3.astype(jnp.float32) - fn_leg.astype(jnp.float32))))
        assert fn_diff <= 1.6e-2, f"fnorm NBS3 vs legacy diff at B={b_sz}: {fn_diff}"

    # 2. Small-batch B=8 and B=16 QKV_EPI (c=8, 16 Long Context!) numerical parity + TPU7x MLIR
    for b_lc in (8, 16):
        st_lc = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(200 + b_lc), (b_lc, H), dtype=jnp.float32),
            NamedSharding(mesh, P()),
        )
        mo_lc = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(210 + b_lc), (b_lc, H_MOE), dtype=jnp.bfloat16),
            NamedSharding(mesh, P()),
        )
        cs_lc = jax.device_put(
            build_decode_rope_step_tables(jnp.arange(b_lc, dtype=jnp.int32) + 2, rope_tbl),
            NamedSharding(mesh, P()),
        )
        ain_lc = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(220 + b_lc), (b_lc, H), dtype=jnp.bfloat16),
            NamedSharding(mesh, P()),
        )
        # L0 fp8_dual_qkv_epi_gate_jax parity vs standalone run_decode_qkv_epilogue
        q_l0, kv_l0, g_l0 = jax.shard_map(
            functools.partial(
                fp8_dual_qkv_epi_gate_jax,
                qkv_norm_eps=1e-5, query_scale=5.47, k_scale=0.8, v_scale=1.2, interpret=True,
            ),
            mesh=mesh,
            in_specs=(P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model"), P()),
            out_specs=(P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
            check_vma=False,
        )(ain_lc, wq_d, sq_d, wg_d, sg_d, cs_lc)
        qkv_ref_0 = ((ain_lc.astype(jnp.float32) @ wq[:, :1280].astype(jnp.float32)) * sq[:, :1280]).astype(jnp.bfloat16)
        q_ref_0, kv_ref_0 = run_decode_qkv_epilogue(
            qkv_ref_0, cs_lc,
            norm_eps=1e-5, query_scale=5.47, k_scale=0.8, v_scale=1.2, interpret=True,
        )
        assert jnp.array_equal(q_l0[:, :, :1], q_ref_0.reshape(1, b_lc, 8, 2, 128)[:, :, :1])
        assert jnp.array_equal(kv_l0[:, :1].view(jnp.uint8), kv_ref_0.view(jnp.uint8))

        # L1..60 moe_to_attn_qkv_epi_bridge_jax at B=8 and B=16 (0 races!)
        out_lc = jax.shard_map(
            functools.partial(
                moe_to_attn_qkv_epi_bridge_jax,
                world_size=W, post_ff_eps=1e-8, post_ff_scale_input=True,
                next_in_eps=1e-5, next_in_scale_input=False, sharded_moe_in=False,
                qkv_norm_eps=1e-5, query_scale=5.47, k_scale=0.8, v_scale=1.2,
                interpret=eager_race,
            ),
            mesh=mesh,
            in_specs=(P(), P(), P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model"), P()),
            out_specs=(P(), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
            check_vma=False,
        )(st_lc, mo_lc, wp_d, sp_d, g1_2d, al_2d, be_2d, g2_2d, wq_d, sq_d, wg_d, sg_d, cs_lc)
        jax.block_until_ready(out_lc)
        assert not tpu_interp.races.races_found, f"DMA race in qepi B={b_lc}"

    print("[PASS] Wave-11 Normalize-Before-s3 bit-exact + 0 races + B=8/16 QKV_EPI + 2D gains")


if __name__ == "__main__":
    test_syntax_and_annotations()
    _install_lightweight_pkg_stubs()
    test_tpu7x_spmd_mlir_export_clean()
    test_sparse_routing_and_k256_opts()
    test_w8_collective_bridges_8rank()
    test_w10_tok_b2_fast_races_and_llo_invariants()
    test_w11_norm_before_s3_b8_b16_and_2d_gains()
    print("ALL WAVE-8 DECODE BRIDGE & SPARSE TESTS PASSED!")
