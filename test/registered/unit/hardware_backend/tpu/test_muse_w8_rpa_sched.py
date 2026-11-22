# SPDX-License-Identifier: Apache-2.0
"""Unit, CPU interpret-mode, and TPU7x Mosaic export tests for Wave-8 RPA/Sched/L61/DFlash."""

from __future__ import annotations

import os
from array import array
from contextlib import contextmanager
from test.registered.unit.hardware_backend.tpu.test_muse_w2_attn_rpa import (
    _cosine,
    _exact_decode_ref,
    _fp8_from_u8,
)
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import export as jax_export
from torch import nn

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental.speculative.block_info import HostPaddedBlockDraftInput


@contextmanager
def _temp_env(**kwargs: str):
    old = {k: os.environ.get(k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            os.environ[k] = v
        yield
    finally:
        for k, prev in old.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def test_write_kv_only_rpa_v3_cpu_interpret_and_export():
    from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
        ragged_paged_attention,
        write_kv_only_rpa_v3,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    rng = np.random.default_rng(801)
    page_size = 128
    pps = 4
    b_sz = 4
    n_pages = b_sz * pps + 2
    t_total = 512

    # Poisoned KV cache with 0x7F
    cache_u8 = np.full((n_pages, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
    # Prefix lengths [100, 130, 0, 0], extend lengths [150, 200, 0, 0] -> sum 350 <= 512
    q_lens = np.array([150, 200, 0, 0], dtype=np.int32)
    pref_lens = np.array([100, 130, 0, 0], dtype=np.int32)
    kv_lens = pref_lens + q_lens
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    cu_kv_lens = np.arange(b_sz + 1, dtype=np.int32) * (pps * page_size)
    page_indices = np.arange(b_sz * pps, dtype=np.int32)
    dist = np.array([0, 0, 2], dtype=np.int32)

    kv_new_u8 = rng.integers(0, 0x70, size=(t_total, 1, 4, 128), dtype=np.uint8)
    ref_cache = cache_u8.copy()
    for s in range(2):
        qs = int(cu_q_lens[s])
        ql = int(q_lens[s])
        kv_st = int(kv_lens[s]) - ql
        pg_base = int(cu_kv_lens[s]) // page_size
        for j in range(ql):
            pos = kv_st + j
            pg = int(page_indices[pg_base + pos // page_size])
            ref_cache[pg, pos % page_size] = kv_new_u8[qs + j]

    upd = write_kv_only_rpa_v3(
        _fp8_from_u8(kv_new_u8),
        _fp8_from_u8(cache_u8),
        jnp.asarray(kv_lens),
        jnp.asarray(page_indices),
        jnp.asarray(cu_q_lens),
        jnp.asarray(cu_kv_lens),
        jnp.asarray(dist),
        interpret=True,
    )
    upd_u8 = np.asarray(jax.lax.bitcast_convert_type(upd, jnp.uint8))
    assert np.array_equal(upd_u8, ref_cache)

    with (
        patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
            return_value="TPU v7",
        ),
        tpu_info_for_cpu_export(),
    ):
        hlo_w = jax_export.export(
            jax.jit(
                lambda kv, c, kl, pi, cq, ckv, d: write_kv_only_rpa_v3(
                    kv, c, kl, pi, cq, ckv, d
                )
            ),
            platforms=["tpu"],
        )(
            jax.ShapeDtypeStruct((16384, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1048, 128, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((4,), jnp.int32),
            jax.ShapeDtypeStruct((4 * 129,), jnp.int32),
            jax.ShapeDtypeStruct((5,), jnp.int32),
            jax.ShapeDtypeStruct((5,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        ).mlir_module()
        assert "RPA_write_kv_only" in hlo_w

        # Pruned T=128 RPAm export
        hlo_128 = jax_export.export(
            jax.jit(
                lambda q, kv, c, kl, pi, cq, ckv, d: ragged_paged_attention(
                    q,
                    kv,
                    None,
                    c,
                    kl,
                    pi,
                    cq,
                    ckv,
                    d,
                    None,
                    None,
                    sm_scale=0.125,
                    k_scale=1.0,
                    v_scale=1.0,
                    _prepared_inputs=True,
                )
            ),
            platforms=["tpu"],
        )(
            jax.ShapeDtypeStruct((1, 128, 8, 2, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((128, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1048, 128, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((4,), jnp.int32),
            jax.ShapeDtypeStruct((4 * 129,), jnp.int32),
            jax.ShapeDtypeStruct((5,), jnp.int32),
            jax.ShapeDtypeStruct((5,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        ).mlir_module()
        assert "RPAm" in hlo_128

    # TorchDynamo + AOTAutograd tracing check on prewarmed write_kv_only custom_op
    from torch._dynamo.backends.common import aot_autograd

    from sglang_tpu.attention_pallas_rpa_v3 import (
        TpuPallasRpaV3AttnBackend,
        _get_write_kv_only_op,
    )

    op = _get_write_kv_only_op()

    class _FakeKvPool:
        def __init__(self, buf):
            self.buf = buf

        def get_fused_kv_buffer(self, _lid):
            return self.buf

    cache_t = torch.zeros((4, 128, 1, 4, 128), dtype=torch.float8_e4m3fn)
    pool = _FakeKvPool(cache_t)
    be = TpuPallasRpaV3AttnBackend.__new__(TpuPallasRpaV3AttnBackend)
    be._dcp = 1
    be.kv_cache_quant_dtype = torch.float8_e4m3fn
    be._write_kv_only_op = op
    be._swa_layer_groups = {}
    be.token_to_kv_pool = pool
    be.forward_metadata = SimpleNamespace(
        kv_lens=torch.tensor([16, 0], dtype=torch.int32),
        cu_q_lens=torch.tensor([0, 16, 16], dtype=torch.int32),
        cu_kv_lens=torch.zeros(3, dtype=torch.int32),
        page_indices=torch.zeros(8, dtype=torch.int32),
        swa_page_indices=None,
        swa_group_page_indices=None,
        cp_write_segments=None,
    )
    lyr = SimpleNamespace(
        layer_id=61,
        sliding_window_size=None,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )
    fb_empty = SimpleNamespace()
    kv_t = torch.zeros((16, 1, 4, 128), dtype=torch.float8_e4m3fn)
    c_meta = SimpleNamespace(
        kv_lens=be.forward_metadata.kv_lens,
        cu_q_lens=be.forward_metadata.cu_q_lens,
        cu_kv_lens=be.forward_metadata.cu_kv_lens,
        page_indices=be.forward_metadata.page_indices,
        swa_page_indices=None,
        swa_group_page_indices=None,
        distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )

    def _trace_fn(kv_arg, cache_arg):
        pool.buf = cache_arg
        be.write_kv_only(kv_arg, lyr, fb_empty, chunk_meta=c_meta)
        return kv_arg

    gm, _ = torch._dynamo.export(_trace_fn)(kv_t, cache_t)
    targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
    assert any("rpa_v3_write_kv_only" in t for t in targets), targets

    aot_graphs = []
    aot_b = aot_autograd(fw_compiler=lambda g, _ex: (aot_graphs.append(g), g)[1])
    try:
        torch.compile(_trace_fn, backend=aot_b, fullgraph=True)(kv_t, cache_t)
    except Exception:
        # CPU runtime execution after AOT graph capture may reject jax_kernel without TPU
        pass
    assert len(aot_graphs) == 1
    aot_str = str(aot_graphs[0].graph)
    assert "auto_functionalized_v2" in aot_str and "rpa_v3_write_kv_only" in aot_str, (
        aot_str
    )


def test_rpad_adaptive_c1024_and_dense_kv_probe_cpu_interpret_and_export():
    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    rng = np.random.default_rng(802)
    b_sz = 8
    pps = 129
    n_pages = b_sz * pps + 4
    cache_u8 = np.full((n_pages, 128, 1, 4, 128), 0x7F, dtype=np.uint8)
    valid_u8 = rng.integers(0, 0x70, size=cache_u8.shape, dtype=np.uint8)
    # Short contexts <= 1024 (exercises adaptive c1024 mp=8 branch with pps=129)
    kv_lens = np.array([300, 1024, 1, 512, 768, 410, 999, 600], dtype=np.int32)
    page_indices = np.arange(b_sz * pps, dtype=np.int32)
    cu_q_lens = np.arange(b_sz + 1, dtype=np.int32)
    cu_kv_lens = np.arange(b_sz + 1, dtype=np.int32) * (pps * 128)
    dist = np.array([b_sz, b_sz, b_sz], dtype=np.int32)

    for s in range(b_sz):
        for pos in range(int(kv_lens[s]) - 1):
            pg = int(page_indices[s * pps + pos // 128])
            cache_u8[pg, pos % 128] = valid_u8[pg, pos % 128]

    q_3d = rng.standard_normal((b_sz, 16, 128)).astype(np.float32) * 0.3
    q_3d[:, :8, 64:] = 0.0
    q_3d[:, 8:, :64] = 0.0
    q_j_3d = jnp.asarray(q_3d, dtype=jnp.bfloat16)
    q_j_prep = q_j_3d.reshape(1, b_sz, 8, 2, 128)
    kv_new_j = _fp8_from_u8(
        rng.integers(0, 0x70, size=(b_sz, 1, 4, 128), dtype=np.uint8)
    )

    for sw in (None, 2048):
        ref_out, ref_cache_u8 = _exact_decode_ref(
            np.asarray(q_j_3d, dtype=np.float32),
            np.asarray(kv_new_j),
            cache_u8,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            b_sz,
            sm_scale=0.125,
            sliding_window=sw,
        )
        ref_pack2 = np.concatenate(
            [
                ref_out[:, :8, :64].reshape(b_sz, 512),
                ref_out[:, 8:, 64:].reshape(b_sz, 512),
            ],
            axis=-1,
        )
        out_2d, upd_cache = run_decode_fast_rpa(
            q_j_prep,
            kv_new_j,
            None,
            _fp8_from_u8(cache_u8),
            jnp.asarray(kv_lens),
            jnp.asarray(page_indices),
            jnp.asarray(cu_q_lens),
            jnp.asarray(cu_kv_lens),
            jnp.asarray(dist),
            sm_scale=0.125,
            sliding_window=sw,
            k_scale=1.0,
            v_scale=1.0,
            mode_str="w7",
            fused_out_pack=2,
            prepared_inputs=True,
            interpret=True,
        )
        out_np = np.asarray(out_2d, dtype=np.float32)
        assert not np.isnan(out_np).any()
        upd_u8 = np.asarray(jax.lax.bitcast_convert_type(upd_cache, jnp.uint8))
        assert np.array_equal(upd_u8, ref_cache_u8)
        assert _cosine(out_np, ref_pack2) > 0.9998

    # Dense KV probe runs cleanly without NaN
    out_probe, _ = run_decode_fast_rpa(
        q_j_prep,
        kv_new_j,
        None,
        _fp8_from_u8(cache_u8),
        jnp.asarray(kv_lens),
        jnp.asarray(page_indices),
        jnp.asarray(cu_q_lens),
        jnp.asarray(cu_kv_lens),
        jnp.asarray(dist),
        sm_scale=0.125,
        sliding_window=None,
        k_scale=1.0,
        v_scale=1.0,
        mode_str="w7",
        fused_out_pack=2,
        prepared_inputs=True,
        probe="dense_kv",
        interpret=True,
    )
    assert not np.isnan(np.asarray(out_probe, dtype=np.float32)).any()

    with (
        patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
            return_value="TPU v7",
        ),
        tpu_info_for_cpu_export(),
    ):
        for sw, long_tok in ((None, "w7auto-mp160"), (2048, "swabatch-mp17")):
            hlo = jax_export.export(
                jax.jit(
                    lambda q, kv, c, kl, pi, cq, ckv, d, g, _sw=sw: run_decode_fast_rpa(
                        q,
                        kv,
                        None,
                        c,
                        kl,
                        pi,
                        cq,
                        ckv,
                        d,
                        sm_scale=0.125,
                        sliding_window=_sw,
                        k_scale=1.0,
                        v_scale=1.0,
                        mode_str="w7",
                        fused_out_pack=2,
                        prepared_inputs=True,
                        gate=g,
                        out_norm_eps=1e-5,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((1, 64, 8, 2, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((64, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((16400, 128, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((64,), jnp.int32),
                jax.ShapeDtypeStruct((64 * 129,), jnp.int32),
                jax.ShapeDtypeStruct((65,), jnp.int32),
                jax.ShapeDtypeStruct((65,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
                jax.ShapeDtypeStruct((64, 1024), jnp.bfloat16),
            ).mlir_module()
            assert "c1024-stream0b-mp8" in hlo
            assert long_tok in hlo


def test_rpam_fp8_qk_hoisted_prescale_accuracy():
    import math

    from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
        ragged_paged_attention,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    # 1. Verify mathematical accuracy of hoisting 1/32 into Q pre-scale (no per-logit rescale)
    rng = np.random.default_rng(803)
    sm_scale = 0.125
    exp_scale = sm_scale * math.log2(math.e)
    q = rng.standard_normal((32, 16, 128)).astype(np.float32) * 1.5
    k = rng.standard_normal((128, 128)).astype(np.float32) * 0.5
    v = rng.standard_normal((128, 128)).astype(np.float32) * 0.5
    k_fp8 = jnp.asarray(k, dtype=jnp.float8_e4m3fn)
    v_fp8 = jnp.asarray(v, dtype=jnp.float8_e4m3fn)

    # Reference: exact fp32/bf16 softmax
    qk_ref = (
        jnp.asarray(q, dtype=jnp.bfloat16).astype(jnp.float32)
        @ k_fp8.astype(jnp.float32).T
    ) * exp_scale
    p_ref = jnp.exp2(qk_ref.astype(jnp.bfloat16))
    p_ref = p_ref / jnp.sum(p_ref.astype(jnp.float32), axis=-1, keepdims=True)
    out_ref = np.asarray(p_ref @ v_fp8.astype(jnp.float32), dtype=np.float32)

    # Hoisted fp8_qk: q_fp8 = clip(q * exp_scale, -448, 448).astype(fp8); qk = (q_fp8 @ k_fp8.T).astype(bf16)
    q_hoisted_fp8 = jnp.clip(jnp.asarray(q) * exp_scale, -448.0, 448.0).astype(
        jnp.float8_e4m3fn
    )
    qk_hoisted = (
        q_hoisted_fp8.astype(jnp.float32) @ k_fp8.astype(jnp.float32).T
    ).astype(jnp.bfloat16)
    p_hoisted = jnp.exp2(qk_hoisted)
    p_hoisted = p_hoisted / jnp.sum(
        p_hoisted.astype(jnp.float32), axis=-1, keepdims=True
    )
    out_hoisted = np.asarray(p_hoisted @ v_fp8.astype(jnp.float32), dtype=np.float32)
    assert _cosine(out_hoisted, out_ref) > 0.999

    # 2. Verify TPU v7 Mosaic export of RPAm fp8_qk has no 1/32 (0.03125) rescale in kernel
    with (
        patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
            return_value="TPU v7",
        ),
        tpu_info_for_cpu_export(),
    ):
        opts = frozenset(
            {
                "stationary_q_m0_ring_fuse",
                "skip_empty_cdma",
                "mixed_s3",
                "sw2048_fuse",
                "fp8_qk",
                "mrb3",
            }
        )
        hlo = jax_export.export(
            jax.jit(
                lambda q_in, kv_in, c, kl, pi, cq, ckv, d: ragged_paged_attention(
                    q_in,
                    kv_in,
                    None,
                    c,
                    kl,
                    pi,
                    cq,
                    ckv,
                    d,
                    None,
                    None,
                    sm_scale=0.125,
                    sliding_window=2048,
                    k_scale=1.0,
                    v_scale=1.0,
                    m_block_sizes=(256, 2048, 256, 256),
                    _prefill_opts=opts,
                    _fused_out_pack=2,
                    _prepared_inputs=True,
                    _skip_decode=True,
                    _a3_out_layout=True,
                )
            ),
            platforms=["tpu"],
        )(
            jax.ShapeDtypeStruct((1, 16384, 8, 2, 128), jnp.bfloat16),
            jax.ShapeDtypeStruct((16384, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1048, 128, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((8,), jnp.int32),
            jax.ShapeDtypeStruct((8 * 129,), jnp.int32),
            jax.ShapeDtypeStruct((9,), jnp.int32),
            jax.ShapeDtypeStruct((9,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        ).mlir_module()
        assert "RPAm" in hlo
        assert "3.125000e-02" not in hlo and "0.03125" not in hlo


def test_dflash_piggyback_single_token_prefill_standalone_and_merged():
    from sglang.srt.managers.overlap_utils import FutureMap

    with _temp_env(SGLANG_TPU_PIGGYBACK_SINGLE_TOKEN_PREFILL="1"):
        pool = SimpleNamespace(
            size=16,
            page_size=128,
            req_to_token=torch.arange(16 * 128, dtype=torch.int32).reshape(16, 128),
            host_resident=True,
        )
        fmap = FutureMap("cpu", SpeculativeAlgorithm.DFLASH, pool)

        alloc = MagicMock()
        alloc.page_size = 128
        alloc.available_size.return_value = 1000

        sched = Scheduler.__new__(Scheduler)
        sched.is_generation = True
        sched.enable_overlap = True
        sched.spec_algorithm = SpeculativeAlgorithm.DFLASH
        sched.require_mlp_sync = False
        sched.enable_hisparse = False
        sched.dllm_config = None
        sched.chunked_reqs = []
        sched.max_running_requests = 8
        sched.future_map = fmap
        sched.token_to_kv_pool_allocator = alloc

        def _make_req(rid: str, tok: int, *, prefix_len: int = 0):
            return SimpleNamespace(
                rid=rid,
                origin_input_ids=array("q", [1] * prefix_len + [tok]),
                output_ids=array("q"),
                prefix_indices=torch.zeros(prefix_len, dtype=torch.int32),
                num_matched_prefix_tokens=prefix_len,
                extend_end=prefix_len + 1,
                extend_len=1,
                inflight_middle_chunks=0,
                beam_group=None,
                return_sampling_mask=False,
                token_indices_to_pool=None,
                return_logprob=False,
                return_hidden_states=False,
                grammar=None,
                is_retracted=False,
                lora_id=None,
                decode_batch_idx=0,
                kv=SimpleNamespace(
                    holds_mamba=False,
                    holds_kv=True,
                    kv_allocated_len=prefix_len + 1,
                    kv_committed_len=prefix_len + 1,
                ),
                finished=lambda: False,
            )

        def _make_nb(reqs, pool_ids, seq_lens, out_loc, toks):
            return SimpleNamespace(
                reqs=list(reqs),
                forward_mode=ForwardMode.EXTEND,
                is_prefill_only=False,
                return_logprob=False,
                return_hidden_states=False,
                return_hidden_states_mode=CaptureHiddenMode.NULL,
                has_grammar=False,
                beam_tail=None,
                input_embeds=None,
                replace_embeds=None,
                prefill_input_ids_cpu=torch.tensor(toks, dtype=torch.int64),
                multimodal_inputs=None,
                req_pool_indices=torch.tensor(pool_ids, dtype=torch.int64),
                req_pool_indices_cpu=torch.tensor(pool_ids, dtype=torch.int64),
                seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
                seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
                orig_seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
                out_cache_loc=torch.tensor(out_loc, dtype=torch.int32),
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=alloc,
                tree_cache=None,
                device="cpu",
                enable_overlap=True,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=None,
                sampling_info=SimpleNamespace(
                    penalizer_orchestrator=SimpleNamespace(is_required=False),
                    merge_batch=lambda other: None,
                ),
                batch_size=lambda: len(reqs),
                maybe_evict_swa=lambda: None,
            )

        fake_parallel = SimpleNamespace(pp_size=1, attn_cp_size=1, attn_dcp_size=1)
        fake_exec = SimpleNamespace(
            mamba=SimpleNamespace(enable_mamba_extra_buffer=False)
        )
        fake_spec = SimpleNamespace(speculative_draft_window_size=None)
        with (
            patch("sglang.srt.managers.scheduler.current_platform") as mock_plat,
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=fake_parallel,
            ),
            patch("sglang.srt.managers.scheduler.get_exec", return_value=fake_exec),
            patch(
                "sglang.srt.runtime_context.max_speculative_num_draft_tokens",
                return_value=5,
            ),
            patch(
                "sglang_tpu.experimental.speculative.draft_info.max_speculative_num_draft_tokens",
                return_value=5,
            ),
            patch(
                "sglang_tpu.experimental.speculative.draft_info.get_spec",
                return_value=fake_spec,
            ),
            patch(
                "sglang_tpu.experimental.speculative.draft_info.allocate_and_assign_decode_cache",
            ),
        ):
            mock_plat.is_tpu.return_value = True

            # 1) Standalone 1-token prefill in DFlash
            r0 = _make_req("r0", 42, prefix_len=0)
            nb0 = _make_nb([r0], [1], [1], [10], [42])
            empty_rb = SimpleNamespace(
                reqs=[],
                is_empty=lambda: True,
                is_prefill_only=False,
                return_logprob=False,
                return_hidden_states=False,
                return_hidden_states_mode=CaptureHiddenMode.NULL,
                has_grammar=False,
                beam_tail=None,
            )
            nb_out, rb_out, ret = sched._maybe_piggyback_single_token_prefill(
                nb0, empty_rb
            )
            assert nb_out is None and ret is rb_out
            assert ret.forward_mode == ForwardMode.DECODE
            assert r0._tpu_piggybacked_prefill is True
            assert r0.kv.kv_committed_len == 0
            assert isinstance(ret.spec_info, HostPaddedBlockDraftInput)
            assert ret.spec_info.tpu_verify_window_rows == 5
            assert ret.spec_info.tpu_committed_lens.tolist() == [0]
            assert fmap.output_tokens_buf[1].item() == 42
            assert fmap.new_seq_lens_buf[1].item() == 0

            # 2) Merge into active DFlash running_batch (with out_cache_loc=None!)
            r_run = _make_req("run", 99, prefix_len=10)
            r_run.kv.kv_committed_len = 11
            r_run.kv.kv_allocated_len = 128
            rb_active = ScheduleBatch.__new__(ScheduleBatch)
            rb_active.reqs = [r_run]
            rb_active.forward_mode = ForwardMode.DECODE
            rb_active.is_prefill_only = False
            rb_active.return_logprob = False
            rb_active.return_hidden_states = False
            rb_active.return_hidden_states_mode = CaptureHiddenMode.NULL
            rb_active.has_grammar = False
            rb_active.beam_tail = None
            rb_active.req_pool_indices = torch.tensor([2], dtype=torch.int64)
            rb_active.req_pool_indices_cpu = torch.tensor([2], dtype=torch.int64)
            rb_active.seq_lens = torch.tensor([11], dtype=torch.int64)
            rb_active.seq_lens_cpu = torch.tensor([11], dtype=torch.int64)
            rb_active.orig_seq_lens = torch.tensor([11], dtype=torch.int32)
            rb_active.out_cache_loc = None
            rb_active.input_ids = None
            rb_active.multimodal_inputs = None
            rb_active.model_config = SimpleNamespace(is_encoder_decoder=False)
            rb_active.req_to_token_pool = pool
            rb_active.token_to_kv_pool_allocator = alloc
            rb_active.tree_cache = SimpleNamespace(supports_swa=lambda: False)
            rb_active.device = "cpu"
            rb_active.enable_overlap = True
            rb_active.spec_algorithm = SpeculativeAlgorithm.DFLASH
            rb_active.sampling_info = SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=False),
                merge_batch=lambda other: None,
            )
            rb_active.spec_info = HostPaddedBlockDraftInput(
                bonus_tokens=torch.tensor([99], dtype=torch.int64),
                new_seq_lens=torch.tensor([11], dtype=torch.int64),
                future_indices=rb_active.req_pool_indices,
                tpu_host_seq_lens_is_committed=True,
            )

            sched.update_running_batch = lambda b: (
                b.spec_info.prepare_for_decode(b),
                b,
            )[1]

            r1 = _make_req("r1", 77, prefix_len=5)
            nb1 = _make_nb([r1], [3], [6], [30], [77])
            nb_out2, rb_out2, ret2 = sched._maybe_piggyback_single_token_prefill(
                nb1, rb_active
            )
            assert nb_out2 is None and ret2 is rb_out2
            assert [r.rid for r in ret2.reqs] == ["run", "r1"]
            assert ret2.out_cache_loc is None
            assert ret2.spec_info.tpu_committed_lens.tolist() == [11, 5]
            assert ret2.spec_info.tpu_verify_cache_candidates.shape == (2, 10)
            assert fmap.output_tokens_buf[3].item() == 77
            assert fmap.new_seq_lens_buf[3].item() == 5


def test_dq5_target_verify_bridges_and_gate_fuse_and_extend_skip_guard():
    from sglang_tpu.compilation.compiled_forward_shim import CompiledForwardShim
    from sglang_tpu.kernels import decode_collective_bridges as dcb
    from sglang_tpu.models.muse_spark import (
        MuseSparkAttention,
        MuseSparkDecoderLayer,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    # 1. decode_attn_gate_fuse lifts dq=1 restriction on TARGET_VERIFY when VERIFY_RPA_FAST=1
    attn = MuseSparkAttention.__new__(MuseSparkAttention)
    nn.Module.__init__(attn)
    attn.head_dim = 128
    attn.num_heads = 2
    attn.num_kv_heads = 1
    attn.q_size = 256
    attn.kv_size = 128
    attn.norm_eps = 1e-5
    attn.decode_qkv_epilogue = True
    attn.decode_attn_gate_fuse = True
    attn.query_scale = 1.0
    attn.rotary_emb = None
    attn._qkvg_w8k = None
    attn._qkvg_weight = None
    attn._o_proj_w8k = None
    attn.qkv_proj = Mock(
        side_effect=lambda x: (torch.zeros(x.shape[0], 512, dtype=x.dtype), None)
    )
    attn.gate_proj = Mock(
        side_effect=lambda x: (torch.ones(x.shape[0], 256, dtype=x.dtype), None)
    )
    attn.o_proj = Mock(side_effect=lambda x, skip_all_reduce=False: (x, None))
    attn.attn = Mock()
    attn._per_head_normalize = Mock(side_effect=lambda x: x)
    seen_gate = []

    def _fake_qkv_attn(*args, gate=None, return_post_fused=False, **kw):
        seen_gate.append((gate is not None, return_post_fused))
        out = torch.ones((args[1].shape[0], 256), dtype=torch.bfloat16)
        return (out, gate is not None) if return_post_fused else out

    with (
        tpu_envs.SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST.override(True),
        patch(
            "sglang_tpu.models.muse_spark.can_use_muse_decode_qkv_epilogue",
            return_value=True,
        ),
        patch(
            "sglang_tpu.models.muse_spark.muse_decode_fused_qkv_attn",
            side_effect=_fake_qkv_attn,
        ),
    ):
        x_tv = torch.ones((320, 512), dtype=torch.bfloat16)
        pos_tv = torch.arange(320, dtype=torch.int32)
        attn(pos_tv, x_tv, SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY))
        assert seen_gate == [(True, True)]

    # 2. _active_w7_bridges allows n_tok=320 (B=64, dq=5) under DECODE_B384=1
    layer = object.__new__(MuseSparkDecoderLayer)
    nn.Module.__init__(layer)
    layer._decode_col_bridges = 3
    layer.self_attn = SimpleNamespace(
        o_proj=MagicMock(),
        _o_proj_w8k=torch.zeros(1),
    )
    # Use RowParallelLinear check bypass via spec
    from sglang.srt.layers.linear import RowParallelLinear

    op = object.__new__(RowParallelLinear)
    op.tp_size = 8
    op.reduce_results = True
    op.bias = None
    layer.self_attn.o_proj = op
    layer.mlp = SimpleNamespace(
        tp_size=8,
        _buffers={
            "_w7_wpre_w8k": torch.zeros(1),
            "_w7_wrouter_km": torch.zeros(1),
            "_w7_wpost_w8k": torch.zeros(1),
        },
        fuses_pre_gather=lambda n, decode: False,
        fuses_post_gather=lambda n, decode: False,
    )
    next_attn = SimpleNamespace(
        _buffers={"_w7_wqkv_w8k": torch.zeros(1), "_w7_wgate_w8k": torch.zeros(1)}
    )
    with (
        tpu_envs.SGLANG_TPU_MUSE_DECODE_B384.override(True),
        patch.dict(os.environ, {"WORLD_SIZE": "8"}),
        patch(
            "sglang.srt.runtime_context.get_forward",
            return_value=SimpleNamespace(sp_active=False),
        ),
    ):
        b1k, b2k = layer._active_w7_bridges(
            320,
            SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
            is_dec_step=True,
            next_input_layernorm=object(),
            next_layer_attn=next_attn,
        )
        assert (b1k, b2k) == (2, 3)

    # 3. Export decode_collective_bridges at rows=320 (B=64, dq=5) for TPU v7
    with (
        patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
            return_value="TPU v7",
        ),
        tpu_info_for_cpu_export(),
    ):
        hlo_qkvg = jax_export.export(
            jax.jit(dcb.fp8_dual_qkv_gate_jax),
            platforms=["tpu"],
        )(
            jax.ShapeDtypeStruct((320, 8192), jnp.bfloat16),
            jax.ShapeDtypeStruct((8192, 1280), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 1280), jnp.float32),
            jax.ShapeDtypeStruct((8192, 1024), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 1024), jnp.float32),
        ).mlir_module()
        assert "muse_w7_dual_qkvg_b320" in hlo_qkvg

    # 4. SGLANG_TPU_EXTEND_SKIP_GUARD_EVAL enters FakeStance on bucketed EXTEND
    entered = []

    class FakeStance:
        def armed(self):
            return True

        def __enter__(self):
            entered.append("enter")

        def __exit__(self, *exc):
            entered.append("exit")

    policy = SimpleNamespace(
        t_bucketer=object(),
        model_runner=SimpleNamespace(
            is_draft_worker=False,
            pp_group=SimpleNamespace(is_first_rank=True),
            token_to_kv_pool=SimpleNamespace(layer_transfer_counter=None),
            model=SimpleNamespace(),
        ),
        keep_outputs_padded=lambda fb: True,
        finalize_forward_batch=lambda out, fb: out,
        keeps_padded_mm_embeds=lambda fb: False,
    )
    fb_ext = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        mm_input_embeds=None,
        spec_info=None,
        input_embeds=None,
        is_prefill_only=False,
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_input_logprob_token_ids_gpu=None,
        extend_logprob_start_lens_cpu=None,
        input_ids=torch.zeros(128, dtype=torch.int64),
        seq_lens=torch.ones(2, dtype=torch.int64),
    )
    with _temp_env(SGLANG_TPU_EXTEND_SKIP_GUARD_EVAL="1"):
        shim = CompiledForwardShim(lambda *a, **kw: "ok", policy=policy)
        shim._skip_guard_stance = FakeStance()
        shim(
            torch.zeros(128, dtype=torch.int64),
            torch.zeros(128, dtype=torch.int64),
            fb_ext,
        )
        assert entered == ["enter", "exit"]


def test_l61_pruned_model_and_causal_lm_dispatch():
    from sglang_tpu.models.muse_spark import (
        MuseSparkAttention,
        MuseSparkDecoderLayer,
        MuseSparkForConditionalGeneration,
        MuseSparkGatedResidual,
        MuseSparkRMSNorm,
        MuseSparkTextModel,
    )

    # Verify MuseSparkForConditionalGeneration routes pruned hidden_states [bs, H] to _forward_pruned_logits
    lm = MuseSparkForConditionalGeneration.__new__(MuseSparkForConditionalGeneration)
    nn.Module.__init__(lm)
    lm._megakernel = None
    lm.pp_group = SimpleNamespace(is_last_rank=True)
    lm.capture_aux_hidden_states = False
    lm.model = Mock(return_value=torch.zeros((2, 16), dtype=torch.bfloat16))
    lm.model.prefill_kernels = None
    lm._forward_pruned_logits = Mock(return_value="pruned_logits")
    lm.logits_processor = Mock(return_value="full_logits")
    lm.lm_head = object()

    fb_null = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        return_logprob=False,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    out = lm.forward(
        torch.zeros(256, dtype=torch.int64),
        torch.arange(256, dtype=torch.int64),
        fb_null,
    )
    assert out == "pruned_logits"
    lm._forward_pruned_logits.assert_called_once()

    # With return_logprob=True or capture_hidden_mode=FULL, must NOT use _forward_pruned_logits
    lm._forward_pruned_logits.reset_mock()
    fb_lp = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        return_logprob=True,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    out_lp = lm.forward(
        torch.zeros(256, dtype=torch.int64),
        torch.arange(256, dtype=torch.int64),
        fb_lp,
    )
    assert out_lp == "full_logits"
    lm._forward_pruned_logits.assert_not_called()

    # Verify _maybe_forward_l61_pruned slices residual, q, kv, gate to last_token_indices padded to 128
    hidden = 16
    layer = MuseSparkDecoderLayer.__new__(MuseSparkDecoderLayer)
    nn.Module.__init__(layer)
    layer.input_layernorm = MuseSparkRMSNorm(hidden, eps=1e-5, gain_center=1.0)
    layer.post_attention_layernorm = MuseSparkRMSNorm(
        hidden, eps=1e-8, has_weight=False
    )
    layer.post_attention_residual_gate = MuseSparkGatedResidual(hidden, temperature=1.0)
    layer.post_attention_residual_gate.precompute_coefficients()
    layer.pre_feedforward_layernorm = MuseSparkRMSNorm(
        hidden, eps=1e-5, gain_center=1.0
    )
    layer.post_feedforward_layernorm = MuseSparkRMSNorm(
        hidden, eps=1e-8, gain_center=1.0, scale_input=True
    )
    layer.post_feedforward_residual_gate = MuseSparkGatedResidual(
        hidden, temperature=1.0
    )
    layer.post_feedforward_residual_gate.precompute_coefficients()

    attn = MuseSparkAttention.__new__(MuseSparkAttention)
    nn.Module.__init__(attn)
    attn.head_dim = 64
    attn.num_heads = 16
    attn.num_kv_heads = 2
    attn.q_size = 1024
    attn.kv_size = 128
    attn.norm_eps = 1e-5
    attn.query_scale = 1.0
    attn.rotary_emb = None
    attn._qkvg_weight = None
    attn._o_proj_w8k = None
    attn.qkv_proj = Mock(
        side_effect=lambda x: (
            torch.zeros((x.shape[0], 1024 + 256), dtype=x.dtype),
            None,
        )
    )
    attn.gate_proj = Mock(
        side_effect=lambda x: (torch.zeros((x.shape[0], 1024), dtype=x.dtype), None)
    )
    attn.o_proj = Mock(
        side_effect=lambda x, skip_all_reduce=False: (
            torch.zeros((x.shape[0], hidden), dtype=x.dtype),
            None,
        )
    )
    attn.attn = SimpleNamespace(
        layer_id=61,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )
    layer.self_attn = attn

    mlp_seen = []

    def _fake_mlp(x_bf16, router_in, decode=False):
        mlp_seen.append((x_bf16.shape[0], decode))
        return torch.zeros((x_bf16.shape[0], hidden), dtype=x_bf16.dtype)

    _fake_mlp._fused_decode_op = object()
    _fake_mlp._fused_decode_batches = (16, 32, 64, 128)
    _fake_mlp.e_score_correction_bias = torch.zeros(256)
    _fake_mlp.pre_expert_norm = SimpleNamespace(weight=torch.ones(hidden))
    _fake_mlp.post_expert_norm = SimpleNamespace(weight=torch.ones(hidden))
    _fake_mlp.experts = SimpleNamespace(
        w13_weight=torch.empty(0, dtype=torch.float4_e2m1fn_x2),
        w2_weight=torch.empty(0, dtype=torch.float4_e2m1fn_x2),
        w13_weight_scale=torch.empty(0),
        w2_weight_scale=torch.empty(0),
        _tpu_fused_rs_group_offset=torch.tensor([0], dtype=torch.int32),
    )
    layer.mlp = _fake_mlp

    kv_written = []
    be = SimpleNamespace(
        _prepared_op=object(),
        kv_cache_quant_dtype=torch.float8_e4m3fn,
        _dcp=1,
        token_to_kv_pool=SimpleNamespace(
            get_fused_kv_operands=lambda lid: (
                torch.zeros((4, 128, 1, 4, 128), dtype=torch.float8_e4m3fn),
                None,
            ),
            get_fused_kv_buffer=lambda lid: torch.zeros(
                (4, 128, 1, 4, 128), dtype=torch.float8_e4m3fn
            ),
        ),
        forward_metadata=SimpleNamespace(
            kv_lens=torch.tensor([100, 156, 0, 0], dtype=torch.int32),
            cu_kv_lens=torch.zeros(5, dtype=torch.int32),
            page_indices=torch.zeros(16, dtype=torch.int32),
            swa_page_indices=None,
            swa_group_page_indices=None,
            cp_write_segments=None,
        ),
        write_kv_only=lambda kv, lyr, fb: kv_written.append(kv.shape[0]),
        forward_prepared=lambda q, kv, lyr, fb, chunk_meta=None: torch.zeros(
            (q.shape[1], 1024), dtype=torch.bfloat16
        ),
    )

    fb_ext = SimpleNamespace(
        extend_seq_lens=torch.tensor([100, 156, 0, 0], dtype=torch.int32)
    )
    exit_idx = torch.cumsum(fb_ext.extend_seq_lens, dim=0, dtype=torch.int32) - 1
    with patch(
        "sglang.srt.model_executor.forward_context.get_attn_backend",
        return_value=be,
    ):
        res = MuseSparkTextModel._maybe_forward_l61_pruned(
            layer,
            torch.arange(256, dtype=torch.int64),
            torch.randn((256, hidden), dtype=torch.float32),
            fb_ext,
            exit_idx,
        )
    assert res is not None and res.shape == (4, hidden)
    assert kv_written == [256]
    assert mlp_seen == [(16, True)]

    # Without _fused_decode_op, _maybe_forward_l61_pruned must return None (no 250ms fallback loop)
    _fake_mlp._fused_decode_op = None
    with patch(
        "sglang.srt.model_executor.forward_context.get_attn_backend",
        return_value=be,
    ):
        assert (
            MuseSparkTextModel._maybe_forward_l61_pruned(
                layer,
                torch.arange(256, dtype=torch.int64),
                torch.randn((256, hidden), dtype=torch.float32),
                fb_ext,
                exit_idx,
            )
            is None
        )

    # Verify MusePrefillKernels._layer_l61_pruned CCS inv_cm_perm alignment:
    # q/gate/kv from a1_chunk are in original [0..T-1] order while stream_shard is chunk-major.
    from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
        from_chunk_major_perm,
        to_chunk_major_perm,
    )
    from sglang_tpu.muse_prefill_kernels.integration import MusePrefillKernels

    tp_ccs = 2
    chunks_ccs = (
        2,
        4,
    )  # t_shard = 6, t_total = 12 -> cm=[0,1, 4,5,6,7 | 2,3, 8,9,10,11]
    t_tot_ccs = tp_ccs * sum(chunks_ccs)
    perm_cm = torch.tensor(to_chunk_major_perm(tp_ccs, chunks_ccs), dtype=torch.long)
    inv_cm = torch.tensor(from_chunk_major_perm(tp_ccs, chunks_ccs), dtype=torch.long)
    # Use a sentinel tensor where token i carries value float(i)
    orig_stream = (
        torch.arange(t_tot_ccs, dtype=torch.float32).unsqueeze(1).repeat(1, hidden)
    )
    cm_stream = orig_stream.index_select(0, perm_cm)
    rank0_shard = cm_stream[: sum(chunks_ccs)]

    pk = MusePrefillKernels.__new__(MusePrefillKernels)
    nn.Module.__init__(pk)
    pk.tp_size = tp_ccs
    pk.tp_rank = 0
    pk.a1_variant = "v6"
    pk.a1_rpa_layout = True
    pk._a1_skip = False
    pk.tp16_owner_a1 = False
    pk.causal_chunk_stream = True
    pk.causal_chunk_spec = ""
    pk.ep_tp2_mode = False
    pk.eplb_moe_op = None
    pk._cur_real_tokens_t = None
    pk.no_next_norm = torch.zeros(hidden)

    def _fake_a1_chunk(ag_k, wqkvg, w_sc, cos_k, sin_k):
        # Infer chunk offset by matching ag_k[:, :, 0] token IDs
        tok_ids = ag_k.reshape(-1, hidden)[:, 0].long()
        q_k = (
            tok_ids.to(torch.bfloat16)
            .view(1, tok_ids.numel(), 1, 1, 1)
            .expand(1, tok_ids.numel(), 8, 2, 128)
        )
        gate_k = tok_ids.to(torch.bfloat16).unsqueeze(1).expand(tok_ids.numel(), 1024)
        kv_k = (
            tok_ids.to(torch.uint8)
            .view(-1, 1, 1, 1)
            .expand(tok_ids.numel(), 1, 4, 128)
            .view(torch.float8_e4m3fn)
        )
        return q_k, gate_k, kv_k

    # For rank 0 owning cm[:6], simulate AG(chunk k) supplying both ranks' rows for chunk k
    def _fake_ag_chunk(local_chunk):
        c_k = local_chunk.shape[0]
        off = 0 if c_k == chunks_ccs[0] else chunks_ccs[0]
        r1 = cm_stream[sum(chunks_ccs) + off : sum(chunks_ccs) + off + c_k]
        return torch.stack([local_chunk, r1], dim=0)

    moe_seen = {}

    def _fake_moe_op(attn_s, res_s, *args):
        moe_seen["shape"] = tuple(res_s.shape)
        moe_seen["vals"] = res_s[:2, 0].tolist()
        return res_s, res_s.to(torch.bfloat16)

    pk.ops = SimpleNamespace(
        a1_chunk=_fake_a1_chunk,
        ag_chunk=_fake_ag_chunk,
        sum_stream=lambda local_rows: (
            local_rows
        ),  # rank 0 sum in test (target tokens on rank 0)
        moe=_fake_moe_op,
        moe_fp8=None,
    )
    seen_pruned = {}

    def _fake_fwd_prep(q_p, kv_p, lyr, fb, chunk_meta=None):
        seen_pruned["q_tok"] = q_p[0, :2, 0, 0, 0].long().tolist()
        seen_pruned["kv_tok"] = kv_p.view(torch.uint8)[:2, 0, 0, 0].long().tolist()
        return torch.zeros((q_p.shape[1], 1024), dtype=torch.bfloat16)

    ccs_kv_chunks = []
    be_ccs = SimpleNamespace(
        forward_metadata=be.forward_metadata,
        write_kv_only=lambda kv_k, lyr, fb, chunk_meta=None: ccs_kv_chunks.append(
            kv_k.shape[0]
        ),
        forward_prepared=_fake_fwd_prep,
    )
    # Tokens 1 (rank 0 chunk 0) and 5 (rank 0 chunk 1) are last tokens of seqs 0 and 1
    ext_ccs = torch.tensor([2, 4, 0, 0], dtype=torch.int32)
    exit_ccs = torch.tensor([1, 5, 5, 5], dtype=torch.int32)
    rope_chunks = [
        (torch.ones((tp_ccs * c, 128)), torch.zeros((tp_ccs * c, 128)))
        for c in chunks_ccs
    ]
    c_metas = [SimpleNamespace(), SimpleNamespace()]
    weights_ccs = SimpleNamespace(
        wqkvg=torch.empty(0),
        wqkvg_scale=torch.empty(0),
        wrouter=torch.empty(0),
        wpool=torch.empty(0),
        wpool_scale=torch.empty(0),
        wunpool=torch.empty(0),
        wunpool_scale=torch.empty(0),
    )
    res_ccs = []

    def _capture_o(x, skip_all_reduce=False):
        return torch.zeros((x.shape[0], hidden), dtype=x.dtype), None

    attn.o_proj = Mock(side_effect=_capture_o)
    with patch(
        "sglang_tpu.muse_prefill_kernels.integration.get_attn_backend",
        return_value=be_ccs,
    ):
        res_ccs_out = pk._layer_l61_pruned(
            layer=layer,
            weights=weights_ccs,
            rope=(torch.empty(0), torch.empty(0)),
            stream_shard=rank0_shard,
            attn_in=rank0_shard,
            forward_batch=SimpleNamespace(extend_seq_lens=ext_ccs),
            exit_last_index=exit_ccs,
            t_shard=sum(chunks_ccs),
            t_total=t_tot_ccs,
            inv_cm_perm=inv_cm,
            chunks=chunks_ccs,
            rope_chunks=rope_chunks,
            chunk_metas=c_metas,
        )
    assert res_ccs_out.shape == (4, hidden)
    assert ccs_kv_chunks == [4, 8]
    assert seen_pruned["q_tok"] == [1, 5]
    assert seen_pruned["kv_tok"] == [1, 5]
    assert moe_seen["shape"] == (128, hidden)
    assert moe_seen["vals"] == [1.0, 5.0]
    assert res_ccs_out[:2, 0].tolist() == [1.0, 5.0]
