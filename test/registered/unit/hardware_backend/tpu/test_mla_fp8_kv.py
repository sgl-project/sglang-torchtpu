"""CPU guards for the fp8 KV path of the Pallas MLA backend.

Covers the pieces that do not need a chip: the pool allocates genuine fp8
base tensors (no uint8 view), the per-tensor quantizer, and the backend's
fp8 branch handing quantized latents plus one shared scale to the kernel op.
"""

import functools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_mla import (
    TpuPallasMlaAttnBackend,
)
from sglang_tpu.attention_pallas_mla_sparse import (
    TpuPallasSparseMlaAttnBackend,
)
from sglang_tpu.attention_pallas_rpa_base import (
    _quantize_kv_to_fp8,
)
from sglang_tpu.kv_pool import (
    TpuFusedMLATokenToKVPool,
    TpuSplitMLATokenToKVPool,
    _zeros_kv_buffer,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FP8 = torch.float8_e4m3fn


class TestZerosKvBuffer(CustomTestCase):
    def test_fp8_buffer_is_a_zeroed_base_tensor(self):
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16):
            with self.subTest(dtype=dtype):
                buf = _zeros_kv_buffer((3, 8), dtype, "cpu")
                self.assertEqual(buf.dtype, dtype)
                self.assertIsNone(buf._base)  # not a view over other storage
                self.assertEqual(
                    buf.untyped_storage().nbytes(), buf.numel() * buf.element_size()
                )
                self.assertEqual(float(buf.to(torch.float32).abs().sum()), 0.0)

    def test_mla_pool_layers_are_fp8_bases(self):
        pool = TpuFusedMLATokenToKVPool(
            size=64,
            page_size=32,
            dtype=FP8,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        self.assertEqual(pool.store_dtype, FP8)
        for layer_id in range(2):
            buf = pool.get_fused_kv_buffer(layer_id)
            self.assertEqual(buf.dtype, FP8)
            self.assertIsNone(buf._base)

    def test_split_pool_allocates_the_buffers_the_sparse_kernels_read(self):
        pool = TpuSplitMLATokenToKVPool(
            size=64,
            page_size=32,
            dtype=FP8,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
            index_head_dim=128,
            index_key_layers=[True, False, True],
        )
        # Per token: 512 NoPE bytes, 64 RoPE bytes padded to 128, and a
        # 256-byte index-key record. Each buffer must be a whole uint8 tensor,
        # not a view, or the ops cannot update it in place and copy the pool.
        expected = {
            pool.get_nope_cache: (pool.num_pages, 32, 4, 128),
            pool.get_rope_cache: (pool.num_pages, 8, 4, 128),
            pool.get_index_key_cache: (pool.num_pages, 8, 4, 256),
        }
        total_bytes = 0
        for accessor, shape in expected.items():
            layers = (0, 2) if accessor == pool.get_index_key_cache else (0, 1, 2)
            for layer_id in layers:
                buf = accessor(layer_id)
                self.assertEqual(tuple(buf.shape), shape)
                self.assertEqual(buf.dtype, torch.uint8)
                self.assertIsNone(buf._base)
                total_bytes += buf.nbytes
        # A layer that reuses the previous layer's result must have no cache.
        with self.assertRaisesRegex(ValueError, "no index-key cache"):
            pool.get_index_key_cache(1)
        self.assertEqual(pool.get_kv_size_bytes(), total_bytes)


class TestQuantizeKvToFp8(CustomTestCase):
    def test_divides_by_scale_clamps_and_casts(self):
        t = torch.tensor([1.0, -2.0, 1000.0, -1000.0], dtype=torch.bfloat16)
        q = _quantize_kv_to_fp8(t, 2.0, FP8)
        self.assertEqual(q.dtype, FP8)
        fmax = torch.finfo(FP8).max
        expected = torch.tensor([0.5, -1.0, fmax, -fmax])
        torch.testing.assert_close(q.to(torch.float32), expected, rtol=0, atol=0)

    def test_scale_one_roundtrips_representable_values(self):
        t = torch.tensor([0.0, 0.5, -1.5, 3.0], dtype=torch.bfloat16)
        q = _quantize_kv_to_fp8(t, 1.0, FP8)
        torch.testing.assert_close(q.to(torch.bfloat16), t, rtol=0, atol=0)


def _run_forward_mla(*, quant_dtype, k_scale_float):
    """Drive ``_forward_mla`` on a stand-in backend with a mocked kernel op."""
    T, H, R, D = 3, 4, 512, 64
    op = MagicMock(return_value=torch.zeros(T, H, R))
    backend = SimpleNamespace(
        kv_cache_quant_dtype=quant_dtype,
        _op=op,
        forward_metadata=SimpleNamespace(
            kv_lens=torch.zeros(1, dtype=torch.int32),
            page_indices=torch.zeros(1, dtype=torch.int32),
            cu_q_lens=torch.zeros(2, dtype=torch.int32),
            cu_kv_lens=torch.zeros(2, dtype=torch.int32),
            distribution=torch.zeros(3, dtype=torch.int32),
            segments="mixed",
        ),
        token_to_kv_pool=SimpleNamespace(get_fused_kv_buffer=lambda _: "kv"),
    )
    # _forward_mla builds the op's arguments through these methods.
    backend._mla_kernel_args = functools.partial(
        TpuPallasMlaAttnBackend._mla_kernel_args, backend
    )
    backend._kv_cache_buffers = functools.partial(
        TpuPallasMlaAttnBackend._kv_cache_buffers, backend
    )
    layer = SimpleNamespace(
        v_head_dim=R,
        qk_head_dim=R + D,
        tp_q_head_num=H,
        scaling=0.125,
        layer_id=0,
        k_scale_float=k_scale_float,
    )
    q = torch.randn(T, H, R, dtype=torch.bfloat16)
    q_rope = torch.randn(T, H, D, dtype=torch.bfloat16)
    k = torch.randn(T, 1, R, dtype=torch.bfloat16) * 4
    k_rope = torch.randn(T, 1, D, dtype=torch.bfloat16) * 4
    out = TpuPallasMlaAttnBackend._forward_mla(
        backend, q, k, layer, forward_batch=None, q_rope=q_rope, k_rope=k_rope
    )
    return out, op.call_args.args, k, k_rope


class TestForwardMlaFp8Branch(CustomTestCase):
    def test_bf16_cache_passes_unquantized_latents_and_no_scales(self):
        out, args, k, k_rope = _run_forward_mla(quant_dtype=None, k_scale_float=None)
        self.assertEqual(tuple(out.shape), (3, 4 * 512))
        new_kv_c, new_k_pe, k_scale, v_scale = args[2], args[3], args[11], args[12]
        self.assertEqual(new_kv_c.dtype, torch.bfloat16)
        torch.testing.assert_close(new_kv_c, k.reshape(3, 512))
        torch.testing.assert_close(new_k_pe, k_rope.reshape(3, 64))
        self.assertIsNone(k_scale)
        self.assertIsNone(v_scale)

    def test_fp8_cache_quantizes_with_layer_scale_shared_by_k_and_v(self):
        _, args, k, k_rope = _run_forward_mla(quant_dtype=FP8, k_scale_float=0.5)
        new_kv_c, new_k_pe, k_scale, v_scale = args[2], args[3], args[11], args[12]
        self.assertEqual(new_kv_c.dtype, FP8)
        self.assertEqual(new_k_pe.dtype, FP8)
        self.assertEqual(k_scale, 0.5)
        self.assertEqual(v_scale, 0.5)  # MLA's V is the same latent tensor as K
        torch.testing.assert_close(
            new_kv_c, _quantize_kv_to_fp8(k.reshape(3, 512), 0.5, FP8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            new_k_pe,
            _quantize_kv_to_fp8(k_rope.reshape(3, 64), 0.5, FP8),
            rtol=0,
            atol=0,
        )

    def test_fp8_cache_without_checkpoint_scale_uses_one(self):
        _, args, _, _ = _run_forward_mla(quant_dtype=FP8, k_scale_float=None)
        self.assertEqual(args[11], 1.0)
        self.assertEqual(args[12], 1.0)
        # Inputs are made dense and contiguous before the pallas bridge.
        self.assertTrue(args[0].is_contiguous() and args[1].is_contiguous())


class TestSparseForwardMlaCacheBuffers(CustomTestCase):
    def test_op_gets_two_cache_buffers_in_place_of_the_fused_one(self):
        """The sparse op takes the dense op's arguments, with the pool's NoPE
        and RoPE buffers in place of the fused buffer and the top-k rows last."""
        T, H, R, D = 3, 4, 512, 64
        op = MagicMock(return_value=torch.zeros(T, H, R))
        backend = SimpleNamespace(
            kv_cache_quant_dtype=FP8,
            _op=op,
            # _mla_kernel_args passes these through, so names stand in for them.
            forward_metadata=SimpleNamespace(
                kv_lens="kv_lens",
                page_indices="page_indices",
                cu_q_lens="cu_q_lens",
                cu_kv_lens="cu_kv_lens",
                distribution="distribution",
            ),
            token_to_kv_pool=SimpleNamespace(
                get_nope_cache=lambda layer_id: f"nope{layer_id}",
                get_rope_cache=lambda layer_id: f"rope{layer_id}",
            ),
        )
        backend._mla_kernel_args = functools.partial(
            TpuPallasMlaAttnBackend._mla_kernel_args, backend
        )
        backend._kv_cache_buffers = functools.partial(
            TpuPallasSparseMlaAttnBackend._kv_cache_buffers, backend
        )
        layer = SimpleNamespace(
            v_head_dim=R,
            qk_head_dim=R + D,
            tp_q_head_num=H,
            scaling=0.125,
            layer_id=7,
            k_scale_float=0.5,
        )
        TpuPallasSparseMlaAttnBackend._forward_mla(
            backend,
            torch.randn(T, H, R, dtype=torch.bfloat16),
            torch.randn(T, 1, R, dtype=torch.bfloat16),
            layer,
            forward_batch=None,
            q_rope=torch.randn(T, H, D, dtype=torch.bfloat16),
            k_rope=torch.randn(T, 1, D, dtype=torch.bfloat16),
            topk_indices="topk",
        )
        # After the four query and new-token tensors:
        self.assertEqual(
            op.call_args.args[4:],
            (
                "nope7",
                "rope7",
                "kv_lens",
                "page_indices",
                "cu_q_lens",
                "cu_kv_lens",
                "distribution",
                0.125,  # sm_scale
                0.5,  # k_scale
                0.5,  # v_scale
                "topk",
            ),
        )
        self.assertEqual(op.call_args.kwargs, {})


if __name__ == "__main__":
    unittest.main()
