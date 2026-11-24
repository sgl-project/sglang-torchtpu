# SPDX-License-Identifier: Apache-2.0
"""Unit, Pallas interpret=True parity, and TPU7x zero-copy AOT tests for Wave-9 M2M Decode Megakernel."""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]


def test_w9_m2m_env_defaults_and_ast_safety():
    """SGLANG_TPU_MUSE_MK_M2M must default off, use .get(), and keep PEP 563 off in custom-op files."""
    from sglang_tpu.envs import envs

    assert envs.SGLANG_TPU_MUSE_MK_M2M.default is False
    assert envs.SGLANG_TPU_MUSE_MK_M2M_MAX_FULL_CTX.default == 4096

    for rel in (
        "sglang_tpu/muse_megakernel/decode/m2m/kernel.py",
        "sglang_tpu/muse_megakernel/decode/m2m/rep_kernel.py",
        "sglang_tpu/muse_megakernel/decode/m2m/chip_sh_kernel.py",
        "sglang_tpu/muse_megakernel/decode/m2m/ops.py",
    ):
        src_txt = (_REPO_ROOT / rel).read_text()
        assert "from __future__ import annotations" not in src_txt, f"PEP 563 must stay off in {rel}"
        tree = ast.parse(src_txt)
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While)):
                cond = ast.unparse(node.test)
                for fn in ("SGLANG_TPU_MUSE_MK_M2M", "SGLANG_TPU_MUSE_MK_M2M_MAX_FULL_CTX"):
                    if f"envs.{fn}" in cond or f"tpu_envs.{fn}" in cond:
                        assert ".get()" in cond, f"{rel}: bare EnvField bool in `{cond}`"


def test_w9_m2m_in_specs_and_prewarm_lock_free():
    """Verify exact input_partition_specs length and lock-free lookup after prewarm across TP=8 & TP=16."""
    from sglang_tpu.muse_megakernel.decode.m2m import ops as m2m_ops

    captured: dict[str, tuple] = {}

    def _fake_export(
        name,
        fn,
        *,
        mesh=None,
        input_partition_specs=None,
        output_partition_specs=None,
    ):
        sig = inspect.signature(fn)
        captured[name] = (len(sig.parameters), input_partition_specs, output_partition_specs)
        return fn

    class _ExplodingLock:
        def __enter__(self):
            raise AssertionError("_M2M_LOCK entered after prewarm!")

        def __exit__(self, *args):
            return False

    fake_mesh = object()
    with (
        mock.patch.object(m2m_ops, "_get_tp_mesh", return_value=fake_mesh),
        mock.patch.object(m2m_ops, "_cpu_export_m2m_op", side_effect=_fake_export),
        mock.patch("sglang_tpu.decode_qkv_epilogue_tpu.get_decode_rope_step_op", return_value=lambda *a: None),
        mock.patch.dict(m2m_ops._M2M_TOK_OPS, {}, clear=True),
        mock.patch.dict(m2m_ops._L0_TOK_OPS, {}, clear=True),
        mock.patch.dict(m2m_ops._M2M_CHIP_OPS, {}, clear=True),
        mock.patch.dict(m2m_ops._L0_CHIP_OPS, {}, clear=True),
        mock.patch.dict(m2m_ops._M2M_REP_OPS, {}, clear=True),
        mock.patch.dict(m2m_ops._L0_REP_OPS, {}, clear=True),
    ):
        m2m_ops.prewarm_m2m_decode_ops(8, tok_sharded=True)
        m2m_ops.prewarm_m2m_decode_ops(16, tok_sharded=False)

        with mock.patch.object(m2m_ops, "_M2M_LOCK", _ExplodingLock()):
            for r_flag in (False, True):
                for r16 in (False, True):
                    assert callable(m2m_ops.get_l0_a2m_tok_sh_tp8_op(has_rope=r_flag, bf16_router=r16))
                    assert callable(m2m_ops.get_m2m_tok_sh_tp8_op(has_rope=r_flag, bf16_router=r16))
                    for tp in (8, 16):
                        assert callable(m2m_ops.get_l0_a2m_rep_op(tp, has_rope=r_flag, bf16_router=r16))
                        assert callable(m2m_ops.get_l0_a2m_chip_sh_op(tp, has_rope=r_flag, bf16_router=r16))
                        for sh in (False, True):
                            assert callable(
                                m2m_ops.get_m2m_rep_op(tp, sharded_moe_in=sh, has_rope=r_flag, bf16_router=r16)
                            )
                            assert callable(
                                m2m_ops.get_m2m_chip_sh_op(tp, sharded_moe_in=sh, has_rope=r_flag, bf16_router=r16)
                            )

    assert len(captured) == 4 + 4 + 2 * (4 + 8 + 4 + 8)
    for op_name, (n_params, in_specs, out_specs) in captured.items():
        assert in_specs is not None and len(in_specs) == n_params, (
            f"{op_name}: len(in_specs)={len(in_specs)} != n_params={n_params}"
        )
        assert out_specs is not None and len(out_specs) == 4


def test_w9_m2m_hybrid_swa_and_long_full_dispatch():
    """Verify maybe_forward_m2m_decode fuses SWA layers via M2M and routes >4K Full layers to standalone RPAd."""
    from types import SimpleNamespace

    from sglang_tpu.muse_megakernel.decode.m2m import ops as m2m_ops

    B, TP, H, HE = 64, 8, 8192, 4096
    calls: list[str] = []

    def _fake_l0(*args):
        calls.append("l0_m2m")
        return (
            torch.zeros((B // TP, H), dtype=torch.float32),
            torch.zeros((B, 256), dtype=torch.float32),
            torch.zeros((B, HE), dtype=torch.bfloat16),
        )

    def _fake_m2m(*args):
        calls.append("m2m")
        return (
            torch.zeros((B // TP, H), dtype=torch.float32),
            torch.zeros((B, 256), dtype=torch.float32),
            torch.zeros((B, HE), dtype=torch.bfloat16),
        )

    def _fake_b2(*args):
        calls.append("standalone_b2")
        return (
            torch.zeros((B // TP, H), dtype=torch.float32),
            torch.zeros((1, B, 8, 2, 128), dtype=torch.bfloat16),
            torch.zeros((B, 1, 4, 128), dtype=torch.float8_e4m3fn),
            torch.zeros((B, 1024), dtype=torch.bfloat16),
        )

    def _fake_b1(*args):
        calls.append("standalone_b1")
        return (
            torch.zeros((B // TP, H), dtype=torch.float32),
            torch.zeros((B, 256), dtype=torch.float32),
            torch.zeros((B, HE), dtype=torch.bfloat16),
        )

    def _fake_fnorm(*args):
        calls.append("fnorm")
        return torch.ones((B, H), dtype=torch.bfloat16)

    def _make_norm():
        return SimpleNamespace(eps=1e-5, scale_input=False, gain=lambda: torch.ones((H,)))

    swa_kv = torch.zeros((64, 512, 1, 4, 128), dtype=torch.float8_e4m3fn)
    full_kv = torch.zeros((256, 512, 1, 4, 128), dtype=torch.float8_e4m3fn)

    def _make_layer(l_id: int, is_swa: bool):
        def _sa_call(*a, **kw):
            calls.append(f"standalone_rpa_l{l_id}")
            return torch.zeros((B, 1024), dtype=torch.bfloat16)

        sa = SimpleNamespace(
            o_proj=SimpleNamespace(tp_size=TP),
            rotary_emb=(object() if is_swa else None),
            decode_qkv_epilogue=True,
            norm_eps=1e-5,
            query_scale=1.0,
            attn=SimpleNamespace(
                layer_id=l_id,
                sliding_window_size=(2047 if is_swa else -1),
                scaling=0.125,
                k_scale_float=1.0,
                v_scale_float=1.0,
            ),
            _o_proj_w8k=torch.zeros((1024, H), dtype=torch.float8_e4m3fn),
            _o_proj_scale=torch.ones((H,)),
            _buffers={
                "_w7_wqkv_w8k": torch.zeros((H, 1280), dtype=torch.float8_e4m3fn),
                "_w7_sqkv": torch.ones((1280,)),
                "_w7_wgate_w8k": torch.zeros((H, 1024), dtype=torch.float8_e4m3fn),
                "_w7_sgate": torch.ones((1024,)),
            },
        )

        class _SA:
            def __init__(self, inner):
                self.__dict__.update(inner.__dict__)

            def __call__(self, *a, **kw):
                return _sa_call(*a, **kw)

        class _MLP:
            def __init__(self):
                self._fused_decode_sharded_op = object()
                self.router_multiplier = 1.0
                self.pre_expert_norm = SimpleNamespace(
                    eps=1e-5, scale_input=False, gain=lambda: torch.ones((HE,))
                )
                self._buffers = {
                    "_w7_wpost_w8k": torch.zeros((HE, 1024), dtype=torch.float8_e4m3fn),
                    "_w7_spost": torch.ones((1024,)),
                    "_w8_wpre_pair_w8k": torch.zeros((H, 2048), dtype=torch.float8_e4m3fn),
                    "_w8_spre_pair": torch.ones((2048,)),
                    "_w7_wrouter_km": torch.zeros((H, 256), dtype=torch.bfloat16),
                }

            def can_return_sharded_moe_out(self, n_tok: int, *, decode: bool = True) -> bool:
                return True

            def __call__(self, *a, **kw):
                calls.append(f"moe_l{l_id}")
                return torch.zeros((B // TP, HE), dtype=torch.bfloat16)

        lyr = SimpleNamespace(
            self_attn=_SA(sa),
            mlp=_MLP(),
            input_layernorm=_make_norm(),
            post_attention_layernorm=_make_norm(),
            pre_feedforward_layernorm=_make_norm(),
            post_feedforward_layernorm=_make_norm(),
            post_attention_residual_gate=SimpleNamespace(alpha=torch.ones((H,)), beta=torch.ones((H,))),
            post_feedforward_residual_gate=SimpleNamespace(alpha=torch.ones((H,)), beta=torch.ones((H,))),
            _w8_bridge_qkv_epi=True,
        )
        return lyr

    # Layer 0 (SWA), Layer 1 (Full with 16 pages * 512 = 8192 > 4096 -> standalone!), Layer 2 (SWA -> M2M!)
    layers = [_make_layer(0, True), _make_layer(1, False), _make_layer(2, True)]
    model = SimpleNamespace(
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        layers_to_capture=(),
        _decode_col_bridges=3,
        start_layer=0,
        end_layer=3,
        layers=layers,
        norm=_make_norm(),
        rotary_emb=SimpleNamespace(cos_sin_pairs=torch.zeros((1024, 128))),
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(
            decode_query_size=1,
            cp_write_segments=None,
            cu_q_lens=torch.arange(B + 1, dtype=torch.int32),
            cu_kv_lens=torch.arange(B + 1, dtype=torch.int32) * 8192,
            decode_shared_prefix_pages=None,
        ),
        _dcp=1,
        kv_cache_quant_dtype=torch.float8_e4m3fn,
        _kv_pack_dense2=False,
        token_to_kv_pool=object(),
        _kv_lens=lambda fb: torch.ones((B,), dtype=torch.int32) * 6000,
        _layer_page_indices=lambda lid, is_swa_layer: (
            torch.ones((B * 5,), dtype=torch.int32)
            if is_swa_layer
            else torch.ones((B * 16,), dtype=torch.int32)
        ),
        _layer_distribution=lambda is_swa_layer: torch.tensor([B, B, B], dtype=torch.int32),
    )

    kvw_flushes: list[tuple[int, int]] = []
    m2m_ops.register_m2m_kv_write_op()

    with (
        mock.patch(
            "sglang_tpu.decode_qkv_epilogue_tpu._resolve_decode_attn_backend",
            return_value=backend,
        ),
        mock.patch(
            "sglang_tpu.attention_pallas_rpa_base._fused_kv_operands",
            side_effect=lambda pool, lid: ((swa_kv if lid != 1 else full_kv), None),
        ),
        mock.patch.object(m2m_ops, "get_l0_a2m_tok_sh_tp8_op", return_value=_fake_l0),
        mock.patch.object(m2m_ops, "get_m2m_tok_sh_tp8_op", return_value=_fake_m2m),
        mock.patch.object(m2m_ops, "get_moe_to_attn_tok_sharded_bridge_op", return_value=_fake_b2),
        mock.patch.object(m2m_ops, "get_attn_to_moe_tok_sharded_bridge_op", return_value=_fake_b1),
        mock.patch.object(m2m_ops, "get_moe_post_tok_sharded_final_norm_op", return_value=_fake_fnorm),
        mock.patch(
            "torch.ops.sglang_tpu_pallas.muse_w9_m2m_kv_write",
            side_effect=lambda rows, dsts, bufs, l2b: kvw_flushes.append((len(rows), len(bufs))),
        ),
    ):
        out = m2m_ops.maybe_forward_m2m_decode(
            model,
            positions=torch.zeros((B,), dtype=torch.int32),
            stream=torch.zeros((B // TP, H), dtype=torch.float32),
            carried_attn_input=torch.zeros((B, H), dtype=torch.bfloat16),
            cos_sin_step=torch.zeros((B, 2, 128), dtype=torch.float32),
            forward_batch=object(),
            tok_sharded_stream=True,
        )

    assert out is not None and out.shape == (B, H)
    # Also verify #2838 TP=8 R1 (`chip_sharded_stream=True`) >4K context guard falls back cleanly (`None`):
    with (
        mock.patch(
            "sglang_tpu.decode_qkv_epilogue_tpu._resolve_decode_attn_backend",
            return_value=backend,
        ),
        mock.patch(
            "sglang_tpu.attention_pallas_rpa_base._fused_kv_operands",
            side_effect=lambda pool, lid: ((swa_kv if lid != 1 else full_kv), None),
        ),
    ):
        out_r1_long = m2m_ops.maybe_forward_m2m_decode(
            model,
            positions=torch.zeros((B,), dtype=torch.int32),
            stream=torch.zeros((B // 4, H), dtype=torch.float32),
            carried_attn_input=torch.zeros((B, H), dtype=torch.bfloat16),
            cos_sin_step=torch.zeros((B, 2, 128), dtype=torch.float32),
            forward_batch=object(),
            tok_sharded_stream=False,
            chip_sharded_stream=True,
        )
    assert out_r1_long is None
    assert calls == [
        "l0_m2m",
        "moe_l0",
        "standalone_b2",
        "standalone_rpa_l1",
        "standalone_b1",
        "moe_l1",
        "m2m",
        "moe_l2",
        "fnorm",
    ]


def test_w9_m2m_tp8_and_tp16_interpret_and_aot_subproc():
    """Run 16-CPU-device interpret=True parity + tpu7x 2x2x1/2x2x2 zero-copy AOT export."""
    script = textwrap.dedent(
        """
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.experimental.topologies import get_topology_desc
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa
        from sglang_tpu.kernels.decode_collective_bridges_w8 import (
            attn_to_moe_chip_sharded_bridge_jax,
            attn_to_moe_tok_sharded_bridge_jax,
            moe_to_attn_chip_sharded_bridge_jax,
            moe_to_attn_tok_sharded_bridge_jax,
        )
        from sglang_tpu.muse_megakernel.decode.m2m import (
            kv_write_batch_jax,
            l0_a2m_chip_sh_jax,
            l0_a2m_rep_jax,
            m2m_chip_sh_jax,
            m2m_rep_jax,
            m2m_tok_sh_tp8_jax,
        )

        devs = jax.devices("cpu")
        mesh8 = Mesh(np.array(devs[:8]), ("model",))
        rng = np.random.default_rng(7)
        B, TP, H, HE = 64, 8, 8192, 4096
        phys_rows, page_size = 256, 512
        n_pages_seq = 5
        num_pages = B * n_pages_seq + 4
        # Inject 0x7F (float8_e4m3fn NaN!) into unused pages (pg 0 and tail pages) to prove VREG V-sanitizer NaN immunity!
        kv_raw_u8 = jax.lax.bitcast_convert_type(
            jnp.asarray(rng.normal(0, 0.2, (TP, num_pages, phys_rows, 1, 4, 128)).astype(np.float32), dtype=jnp.float8_e4m3fn),
            jnp.uint8,
        )
        kv_raw_np = np.array(kv_raw_u8, copy=True)
        kv_raw_np[:, 0, :, :, :, :] = 0x7F
        kv_raw_np[:, 1 + 4, :, :, :, :] = 0x7F  # seq 0 unused page 4 filled with 0x7F NaN!
        kv_init = jax.device_put(
            jax.lax.bitcast_convert_type(jnp.asarray(kv_raw_np), jnp.float8_e4m3fn),
            NamedSharding(mesh8, P("model")),
        )
        n_act = 6
        kv_lens_np = np.zeros((B,), dtype=np.int32)
        kv_lens_np[:n_act] = [1, 2, 3, 513, 1024, 1537]
        cu_q_np = np.array([min(i, n_act) for i in range(B + 1)], dtype=np.int32)
        cu_kv_np = np.arange(B + 1, dtype=np.int32) * (n_pages_seq * page_size)
        pg_np = np.arange(1, 1 + B * n_pages_seq, dtype=np.int32)
        dist_np = np.array([n_act, n_act, n_act], dtype=np.int32)

        st_sh = jax.device_put(jnp.asarray(rng.normal(0, 0.1, (B, H)).astype(np.float32)), NamedSharding(mesh8, P("model", None)))
        moe_sh = jax.device_put(jnp.asarray(rng.normal(0, 0.1, (B, HE)).astype(np.float32), dtype=jnp.bfloat16), NamedSharding(mesh8, P("model", None)))
        w_post = jax.device_put(jnp.asarray(rng.normal(0, 0.2, (HE, H)).astype(np.float32), dtype=jnp.float8_e4m3fn), NamedSharding(mesh8, P(None, "model")))
        s_post = jax.device_put(jnp.asarray(rng.uniform(0.01, 0.05, (H,)).astype(np.float32)), NamedSharding(mesh8, P("model")))
        pfg = jnp.asarray(rng.uniform(0.8, 1.2, (H,)).astype(np.float32))
        pf_a = jnp.asarray(rng.uniform(0.9, 1.0, (H,)).astype(np.float32))
        pf_b = jnp.asarray(rng.uniform(0.1, 0.3, (H,)).astype(np.float32))
        nig = jnp.asarray(rng.uniform(0.8, 1.2, (H,)).astype(np.float32))
        w_qkv = jax.device_put(jnp.asarray(rng.normal(0, 0.2, (H, TP * 1280)).astype(np.float32), dtype=jnp.float8_e4m3fn), NamedSharding(mesh8, P(None, "model")))
        s_qkv = jax.device_put(jnp.asarray(rng.uniform(0.01, 0.05, (TP * 1280,)).astype(np.float32)), NamedSharding(mesh8, P("model")))
        w_gate = jax.device_put(jnp.asarray(rng.normal(0, 0.2, (H, TP * 1024)).astype(np.float32), dtype=jnp.float8_e4m3fn), NamedSharding(mesh8, P(None, "model")))
        s_gate = jax.device_put(jnp.asarray(rng.uniform(0.01, 0.05, (TP * 1024,)).astype(np.float32)), NamedSharding(mesh8, P("model")))
        cs = jnp.asarray(rng.normal(0, 0.5, (B, 2, 128)).astype(np.float32))
        wo = jax.device_put(jnp.asarray(rng.normal(0, 0.2, (TP * 1024, H)).astype(np.float32), dtype=jnp.float8_e4m3fn), NamedSharding(mesh8, P("model", None)))
        s_wo = jnp.asarray(rng.uniform(0.01, 0.05, (H,)).astype(np.float32))
        pa_a = jnp.asarray(rng.uniform(0.9, 1.0, (H,)).astype(np.float32))
        pa_b = jnp.asarray(rng.uniform(0.1, 0.3, (H,)).astype(np.float32))
        pff_g = jnp.asarray(rng.uniform(0.8, 1.2, (H,)).astype(np.float32))
        w_pre_full = rng.normal(0, 0.2, (H, HE)).astype(np.float32)
        s_pre_full = rng.uniform(0.01, 0.05, (HE,)).astype(np.float32)
        wpp = jax.device_put(jnp.asarray(np.stack([w_pre_full[:, :2048] if r % 2 == 0 else w_pre_full[:, 2048:] for r in range(TP)]), dtype=jnp.float8_e4m3fn), NamedSharding(mesh8, P("model", None, None)))
        spp = jax.device_put(jnp.asarray(np.stack([s_pre_full[:2048] if r % 2 == 0 else s_pre_full[2048:] for r in range(TP)]), dtype=jnp.float32), NamedSharding(mesh8, P("model", None)))
        wr = jnp.asarray(rng.normal(0, 0.1, (H, 256)).astype(np.float32), dtype=jnp.bfloat16)
        peg = jnp.asarray(rng.uniform(0.8, 1.2, (HE,)).astype(np.float32))
        kvl, pgi, cuq, cukv, dst = map(jnp.asarray, (kv_lens_np, pg_np, cu_q_np, cu_kv_np, dist_np))

        in_sp = (
            P("model", None), P("model", None), P(None, "model"), P("model"), P(), P(), P(), P(),
            P(None, "model"), P("model"), P(None, "model"), P("model"), P(), P("model"),
            P("model", None), P(), P(), P(), P(), P("model", None, None), P("model", None), P(), P(),
            P(), P(), P(), P(), P(),
        )
        out_sp = (P("model", None), P(), P(), P("model"))

        def _ref(*a):
            st_m, q5, kvm, g_o = moe_to_attn_tok_sharded_bridge_jax(
                *a[:13], world_size=TP, post_ff_eps=1e-5, post_ff_scale_input=True,
                fuse_qkv_epi=True, query_scale=1.25, k_scale=1.0, v_scale=1.0, interpret=True,
            )
            gd, kvu = run_decode_fast_rpa(
                q5, kvm, None, a[13][0], *a[23:], sm_scale=0.125, sliding_window=2048,
                k_scale=1.0, v_scale=1.0, mode_str="w8,pbf16,noz,batchfull",
                fused_out_pack=2, prepared_inputs=True, gate=g_o, out_norm_eps=1e-5,
                kv_pack_dense2=True, interpret=True,
            )
            st_f, lg, ei = attn_to_moe_tok_sharded_bridge_jax(
                st_m, gd, *a[14:19], a[19][0], a[20][0], a[21], a[22],
                world_size=TP, pa_eps=1e-8, router_multiplier=1.5, interpret=True,
            )
            return st_f, lg, ei, kvu[None]

        from jax.experimental.pallas import tpu as pltpu
        nan_interp = pltpu.InterpretParams(
            detect_races=False, uninitialized_memory="nan"
        )

        def _cand(*a):
            st_f, lg, ei, kvu = m2m_tok_sh_tp8_jax(
                *a[23:], *a[:13], a[13][0], *a[14:19], a[19][0], a[20][0], a[21], a[22],
                world_size=TP, post_ff_eps=1e-5, post_ff_scale_input=True,
                query_scale=1.25, sm_scale=0.125, sliding_window=2048,
                kv_pack_dense2=True, pa_eps=1e-8, router_multiplier=1.5,
                interpret=nan_interp,
            )
            return st_f, lg, ei, kvu[None]

        args = (st_sh, moe_sh, w_post, s_post, pfg, pf_a, pf_b, nig, w_qkv, s_qkv, w_gate, s_gate, cs, kv_init, wo, s_wo, pa_a, pa_b, pff_g, wpp, spp, wr, peg, kvl, pgi, cuq, cukv, dst)
        r_st, r_lg, r_ei, r_kv = jax.jit(jax.shard_map(_ref, mesh=mesh8, in_specs=in_sp, out_specs=out_sp, check_vma=False))(*args)
        c_st, c_lg, c_ei, c_kv = jax.jit(jax.shard_map(_cand, mesh=mesh8, in_specs=in_sp, out_specs=out_sp, check_vma=False))(*args)
        np.testing.assert_array_equal(np.asarray(c_st), np.asarray(r_st))
        np.testing.assert_array_equal(np.asarray(c_lg), np.asarray(r_lg))
        np.testing.assert_array_equal(np.asarray(c_ei, dtype=np.float32), np.asarray(r_ei, dtype=np.float32))
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(c_kv, jnp.uint8)),
            np.asarray(jax.lax.bitcast_convert_type(r_kv, jnp.uint8)),
        )

        # Also verify R1 chip-sharded M2M (`[B // 4, 8192]`) 0-ULP bit-exact parity (with 0x7F NaNs in KV cache)!
        st_chip = jax.device_put(
            jnp.asarray(np.repeat(rng.normal(0, 0.1, (4, B // 4, H)).astype(np.float32), 2, axis=0).reshape(TP * (B // 4), H)),
            NamedSharding(mesh8, P("model", None)),
        )
        def _ref_chip(*a):
            st_m, q5, kvm, g_o = moe_to_attn_chip_sharded_bridge_jax(
                *a[:13], world_size=TP, sharded_moe_in=True, post_ff_eps=1e-5, post_ff_scale_input=True,
                fuse_qkv_epi=True, query_scale=1.25, k_scale=1.0, v_scale=1.0, interpret=True,
            )
            gd, kvu = run_decode_fast_rpa(
                q5, kvm, None, a[13][0], *a[23:], sm_scale=0.125, sliding_window=2048,
                k_scale=1.0, v_scale=1.0, mode_str="w8,pbf16,noz,batchfull",
                fused_out_pack=2, prepared_inputs=True, gate=g_o, out_norm_eps=1e-5,
                kv_pack_dense2=True, interpret=True,
            )
            st_f, lg, ei = attn_to_moe_chip_sharded_bridge_jax(
                st_m, gd, *a[14:19], a[19][0], a[20][0], a[21], a[22],
                world_size=TP, pa_eps=1e-8, router_multiplier=1.5, interpret=True,
            )
            return st_f, lg, ei, kvu[None]

        def _cand_chip(*a):
            st_f, lg, ei, kvu = m2m_chip_sh_jax(
                *a[23:], *a[:13], a[13][0], *a[14:19], a[19][0], a[20][0], a[21], a[22],
                world_size=TP, sharded_moe_in=True, post_ff_eps=1e-5, post_ff_scale_input=True,
                query_scale=1.25, sm_scale=0.125, sliding_window=2048,
                kv_pack_dense2=True, pa_eps=1e-8, router_multiplier=1.5,
                interpret=nan_interp,
            )
            return st_f, lg, ei, kvu[None]

        args_chip = (st_chip, *args[1:])
        rc_st, rc_lg, rc_ei, rc_kv = jax.jit(jax.shard_map(_ref_chip, mesh=mesh8, in_specs=in_sp, out_specs=out_sp, check_vma=False))(*args_chip)
        cc_st, cc_lg, cc_ei, cc_kv = jax.jit(jax.shard_map(_cand_chip, mesh=mesh8, in_specs=in_sp, out_specs=out_sp, check_vma=False))(*args_chip)
        assert not np.isnan(np.asarray(cc_st)).any() and not np.isnan(np.asarray(cc_ei, dtype=np.float32)).any()
        np.testing.assert_array_equal(np.asarray(cc_st), np.asarray(rc_st))
        np.testing.assert_array_equal(np.asarray(cc_lg), np.asarray(rc_lg))
        np.testing.assert_array_equal(np.asarray(cc_ei, dtype=np.float32), np.asarray(rc_ei, dtype=np.float32))
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(cc_kv, jnp.uint8)),
            np.asarray(jax.lax.bitcast_convert_type(rc_kv, jnp.uint8)),
        )

        # Also exercise `_fast_uniform_path` (`n_act == 64`, `w_pg == 2`, 4-seq static-slot `@pl.loop(0, 15)`)!
        kvl_u = jnp.asarray(520 + (np.arange(B, dtype=np.int32) * 7) % 480)
        cuq_u = jnp.arange(B + 1, dtype=np.int32)
        dst_u = jnp.array([B, B, B], dtype=jnp.int32)
        args_u = (*args_chip[:23], kvl_u, pgi, cuq_u, cukv, dst_u)
        ru_st, ru_lg, ru_ei, ru_kv = jax.jit(jax.shard_map(_ref_chip, mesh=mesh8, in_specs=in_sp, out_specs=out_sp, check_vma=False))(*args_u)
        cu_st, cu_lg, cu_ei, cu_kv = jax.jit(jax.shard_map(_cand_chip, mesh=mesh8, in_specs=in_sp, out_specs=out_sp, check_vma=False))(*args_u)
        assert not np.isnan(np.asarray(cu_st)).any() and not np.isnan(np.asarray(cu_ei, dtype=np.float32)).any()
        np.testing.assert_array_equal(np.asarray(cu_st), np.asarray(ru_st))
        np.testing.assert_array_equal(np.asarray(cu_lg), np.asarray(ru_lg))
        np.testing.assert_array_equal(np.asarray(cu_ei, dtype=np.float32), np.asarray(ru_ei, dtype=np.float32))
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(cu_kv, jnp.uint8)),
            np.asarray(jax.lax.bitcast_convert_type(ru_kv, jnp.uint8)),
        )

        # Also verify `nod2` Full-attention (`sliding_window=None`, `hoistkv`) with `uninitialized_memory="nan"`!
        kv_nod2 = jax.device_put(
            jax.lax.bitcast_convert_type(jnp.asarray(kv_raw_np), jnp.float8_e4m3fn),
            NamedSharding(mesh8, P("model")),
        )
        cukv_nod2 = jnp.arange(B + 1, dtype=jnp.int32) * (n_pages_seq * phys_rows)
        kvl_nod2 = jnp.asarray([1, 2, 3, 257, 512, 769] + [0] * (B - n_act), dtype=jnp.int32)
        args_nod2 = (*args_chip[:13], kv_nod2, *args_chip[14:23], kvl_nod2, pgi, cuq, cukv_nod2, dst)
        def _ref_nod2(*a):
            st_m, q5, kvm, g_o = moe_to_attn_chip_sharded_bridge_jax(
                *a[:13], world_size=TP, sharded_moe_in=True, post_ff_eps=1e-5, post_ff_scale_input=True,
                fuse_qkv_epi=True, query_scale=1.25, k_scale=1.0, v_scale=1.0, interpret=True,
            )
            gd, kvu = run_decode_fast_rpa(
                q5, kvm, None, a[13][0], *a[23:], sm_scale=0.125, sliding_window=None,
                k_scale=1.0, v_scale=1.0, mode_str="w8,pbf16,noz,batchfull",
                fused_out_pack=2, prepared_inputs=True, gate=g_o, out_norm_eps=1e-5,
                kv_pack_dense2=False, interpret=True,
            )
            st_f, lg, ei = attn_to_moe_chip_sharded_bridge_jax(
                st_m, gd, *a[14:19], a[19][0], a[20][0], a[21], a[22],
                world_size=TP, pa_eps=1e-8, router_multiplier=1.5, interpret=True,
            )
            return st_f, lg, ei, kvu[None]
        def _cand_nod2(*a):
            st_f, lg, ei, kvu = m2m_chip_sh_jax(
                *a[23:], *a[:13], a[13][0], *a[14:19], a[19][0], a[20][0], a[21], a[22],
                world_size=TP, sharded_moe_in=True, post_ff_eps=1e-5, post_ff_scale_input=True,
                query_scale=1.25, sm_scale=0.125, sliding_window=None,
                kv_pack_dense2=False, pa_eps=1e-8, router_multiplier=1.5,
                interpret=nan_interp,
            )
            return st_f, lg, ei, kvu[None]
        rn_st, rn_lg, rn_ei, rn_kv = jax.jit(jax.shard_map(_ref_nod2, mesh=mesh8, in_specs=in_sp, out_specs=out_sp, check_vma=False))(*args_nod2)
        cn_st, cn_lg, cn_ei, cn_kv = jax.jit(jax.shard_map(_cand_nod2, mesh=mesh8, in_specs=in_sp, out_specs=out_sp, check_vma=False))(*args_nod2)
        assert not np.isnan(np.asarray(cn_st)).any() and not np.isnan(np.asarray(cn_ei, dtype=np.float32)).any()
        np.testing.assert_array_equal(np.asarray(cn_st), np.asarray(rn_st))
        np.testing.assert_array_equal(np.asarray(cn_lg), np.asarray(rn_lg))
        np.testing.assert_array_equal(np.asarray(cn_ei, dtype=np.float32), np.asarray(rn_ei, dtype=np.float32))
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(cn_kv, jnp.uint8)),
            np.asarray(jax.lax.bitcast_convert_type(rn_kv, jnp.uint8)),
        )

        # Verify tpu7x:2x2x2 TP=16 AOT compiles with ZERO KV HLO copies (`0` `copy.*body_refs`)!
        dev16 = get_topology_desc("tpu7x:2x2x2", platform="tpu").devices
        tmesh16 = Mesh(np.array(sorted(dev16, key=lambda d: (d.coords[2], d.coords[1], d.coords[0], d.core_on_chip))), ("model",))
        sp_kv16 = P("model", None, None, None, None)
        def sd(sh, dt, sp=P()):
            return jax.ShapeDtypeStruct(sh, dt, sharding=NamedSharding(tmesh16, sp))
        def step16(kvl, pgi, cuq, cukv, dst, st, mo, wp, sp, pfg, pfa, pfb, nig, wq, sq, wg, sg, cs, wo, swo, paa, pab, pffg, wpr, spr, wr, peg, kv):
            def b(*x):
                st2, lg2, ei2, kv2 = m2m_rep_jax(
                    *x[:18], x[27], *x[18:27], world_size=16, sharded_moe_in=False, sliding_window=2048, kv_pack_dense2=False, collective_id=54
                )
                st3, lg3, ei3, kv3 = m2m_rep_jax(
                    *x[:5], st2, mo, *x[7:18], kv2, *x[18:27], world_size=16, sharded_moe_in=False, sliding_window=2048, kv_pack_dense2=False, collective_id=55
                )
                return st3, lg3, ei3, kv3
            isp = (P(), P(), P(), P(), P(), P(), P(), P(None, "model"), P("model"), P(), P(), P(), P(), P(None, "model"), P("model"), P(None, "model"), P("model"), P(), P("model", None), P("model"), P(), P(), P(), P(None, "model"), P("model"), P(), P(), sp_kv16)
            return jax.shard_map(b, mesh=tmesh16, in_specs=isp, out_specs=(P(), P(), P(), sp_kv16), check_vma=False)(
                kvl, pgi, cuq, cukv, dst, st, mo, wp, sp, pfg, pfa, pfb, nig, wq, sq, wg, sg, cs, wo, swo, paa, pab, pffg, wpr, spr, wr, peg, kv
            )
        a16 = (
            sd((64,), jnp.int32), sd((320,), jnp.int32), sd((65,), jnp.int32), sd((65,), jnp.int32), sd((3,), jnp.int32),
            sd((64, H), jnp.float32), sd((64, HE), jnp.bfloat16), sd((HE, H), jnp.float8_e4m3fn, P(None, "model")), sd((H,), jnp.float32, P("model")),
            sd((H,), jnp.float32), sd((H,), jnp.float32), sd((H,), jnp.float32), sd((H,), jnp.float32),
            sd((H, 16 * 640), jnp.float8_e4m3fn, P(None, "model")), sd((16 * 640,), jnp.float32, P("model")),
            sd((H, 16 * 512), jnp.float8_e4m3fn, P(None, "model")), sd((16 * 512,), jnp.float32, P("model")),
            sd((64, 2, 128), jnp.float32), sd((16 * 512, H), jnp.float8_e4m3fn, P("model", None)), sd((16 * H,), jnp.float32, P("model")),
            sd((H,), jnp.float32), sd((H,), jnp.float32), sd((H,), jnp.float32),
            sd((H, HE), jnp.float8_e4m3fn, P(None, "model")), sd((HE,), jnp.float32, P("model")),
            sd((H, 256), jnp.bfloat16), sd((HE,), jnp.float32),
            sd((16 * 256, 512, 1, 4, 128), jnp.float8_e4m3fn, sp_kv16),
        )
        comp = jax.jit(step16, donate_argnums=(27,), out_shardings=(NamedSharding(tmesh16, P()), NamedSharding(tmesh16, P()), NamedSharding(tmesh16, P()), NamedSharding(tmesh16, sp_kv16))).lower(*a16).compile()
        hlo = comp.as_text()
        kv_copies = [ln for ln in hlo.splitlines() if " copy(" in ln and "[256,512,1,4,128]" in ln]
        body_ref_copies = [ln for ln in hlo.splitlines() if " copy(" in ln and "body_refs" in ln]
        assert len(kv_copies) == 0, f"Expected ZERO KV pool copies across 2 layers, got {len(kv_copies)}: {kv_copies}"
        assert len(body_ref_copies) == 0, f"Expected ZERO body_refs copies, got {body_ref_copies}"

        # Also verify ops.py TorchTPU StableHLO export (_jax_placeholders + wrapped.exported) end-to-end!
        from unittest import mock
        import torch
        from sglang_tpu.muse_megakernel.decode.m2m import ops as m2m_ops
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export
        from sglang_tpu.pallas_jax_op import _jax_placeholders

        dev8 = get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
        tmesh8 = Mesh(np.array(sorted(dev8, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))), ("model",))
        with mock.patch.object(m2m_ops, "_get_tp_mesh", side_effect=lambda ws: tmesh8 if ws == 8 else tmesh16):
            op_tok = m2m_ops.get_m2m_tok_sh_tp8_op(has_rope=True, bf16_router=True)
            t_args_tok = (
                torch.empty((64,), dtype=torch.int32),
                torch.empty((320,), dtype=torch.int32),
                torch.empty((65,), dtype=torch.int32),
                torch.empty((65,), dtype=torch.int32),
                torch.empty((3,), dtype=torch.int32),
                torch.empty((8, H), dtype=torch.float32),
                torch.empty((8, HE), dtype=torch.bfloat16),
                torch.empty((HE, 1024), dtype=torch.float8_e4m3fn),
                torch.empty((1024,), dtype=torch.float32),
                torch.empty((H,), dtype=torch.float32),
                torch.empty((H,), dtype=torch.float32),
                torch.empty((H,), dtype=torch.float32),
                torch.empty((H,), dtype=torch.float32),
                torch.empty((H, 1280), dtype=torch.float8_e4m3fn),
                torch.empty((1280,), dtype=torch.float32),
                torch.empty((H, 1024), dtype=torch.float8_e4m3fn),
                torch.empty((1024,), dtype=torch.float32),
                torch.empty((64, 2, 128), dtype=torch.float32),
                torch.empty((256, 256, 1, 4, 128), dtype=torch.float8_e4m3fn),
                torch.empty((1024, H), dtype=torch.float8_e4m3fn),
                torch.empty((H,), dtype=torch.float32),
                torch.empty((H,), dtype=torch.float32),
                torch.empty((H,), dtype=torch.float32),
                torch.empty((H,), dtype=torch.float32),
                torch.empty((H, 2048), dtype=torch.float8_e4m3fn),
                torch.empty((2048,), dtype=torch.float32),
                torch.empty((H, 256), dtype=torch.bfloat16),
                torch.empty((HE,), dtype=torch.float32),
                1e-5, True, 1e-5, False, 1e-5, 1.0, 0.125, 2048, 1.0, 1.0, True,
                1e-8, 1e-5, False, 1.0, 1e-5, False,
            )
            w_tok = op_tok._init_fn
            j_args_tok = _jax_placeholders(t_args_tok, mesh=w_tok.mesh, partition_specs=w_tok.input_partition_specs)
            with mock.patch("sglang_tpu.pallas_export.get_pallas_tpu_device_kind", return_value="TPU7x"), tpu_info_for_cpu_export(), jax._src.config.export_ignore_forward_compatibility(True):
                lowered_tok = w_tok.exported(*j_args_tok)
                assert len(lowered_tok.out_avals) == 4
                assert w_tok.input_output_aliases == {18: 3}
                assert "kv_cache_ro" in str(op_tok._schema) and "!" in str(op_tok._schema)

                kvw_c = m2m_ops._build_kvw_callable(2, 1, (0, 0))
                j_kvw = _jax_placeholders(
                    (
                        torch.empty((8, 128), dtype=torch.int32),
                        torch.empty((8, 128), dtype=torch.int32),
                        torch.empty((64, 1, 4, 128), dtype=torch.float8_e4m3fn),
                        torch.empty((64, 1, 4, 128), dtype=torch.float8_e4m3fn),
                        torch.empty((256, 256, 1, 4, 128), dtype=torch.float8_e4m3fn),
                    ),
                    mesh=None,
                    partition_specs=None,
                )
                lowered_kvw = kvw_c.exported(*j_kvw)
                assert len(lowered_kvw.out_avals) == 1
        print("SUBPROC_M2M_OK")
        """
    )
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = f"{_REPO_ROOT}:{_REPO_ROOT / 'python'}:" + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=480,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "SUBPROC_M2M_OK" in proc.stdout
