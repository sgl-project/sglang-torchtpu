"""Regression tests for the TPU Pallas attention backend.

Covers the things that actually broke during bring-up so they cannot
silently regress:

  * kernel numerics vs a float32 SDPA reference (the quality guard) —
    flash (prefill/extend/decode shapes, incl. the q<128 -> pad path) and
    paged_attention (GQA, page_size variants, q pre-scale, length mask);
  * _bucket_kv_len boundaries (bounds JAX recompiles);
  * _maybe_expand_kv_heads GQA expansion;
  * _patch_support_triton_for_tpu (else the Triton path crashes on TPU
    with "0 active drivers");
  * paged-decode slot padding stays local to pallas_tpu_flash and uses the
    allocator's reserved physical KV slot 0;
  * Both Pallas backend names appear in the attention registry (else
    "pallas_tpu_flash" / "pallas_tpu_rpa_v3" are unknown).

The numeric tests need TPU + torch_tpu and are skipped otherwise. The
helper / patch tests run anywhere.

Usage:
    SGLANG_PLATFORM=tpu python test/manual/tpu/test_pallas_attention.py
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import math
import unittest

from sglang.test.test_utils import CustomTestCase


def _skip_if_no_tpu() -> bool:
    try:
        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


class TestPallasHelpers(CustomTestCase):
    """Pure logic — no TPU required."""

    def test_bucket_kv_len_boundaries(self):
        from sglang_tpu.layers.attention.attention_pallas import (
            _PAGE_SIZE,
            _bucket_kv_len,
        )

        self.assertEqual(_PAGE_SIZE, 64)
        # <= page size rounds up to one page; powers-of-two * page_size.
        self.assertEqual(_bucket_kv_len(1), 64)
        self.assertEqual(_bucket_kv_len(57), 64)
        self.assertEqual(_bucket_kv_len(64), 64)
        self.assertEqual(_bucket_kv_len(65), 128)
        self.assertEqual(_bucket_kv_len(128), 128)
        self.assertEqual(_bucket_kv_len(129), 256)
        self.assertEqual(_bucket_kv_len(200), 256)
        # Bucketing must always cover the real length (no truncation).
        for n in (1, 33, 64, 65, 130, 257, 1000):
            self.assertGreaterEqual(_bucket_kv_len(n), n)

    def test_maybe_expand_kv_heads(self):
        import torch

        from sglang_tpu.layers.attention.attention_pallas import (
            _maybe_expand_kv_heads,
        )

        # Equal head counts -> identity (no copy / no-op semantics).
        kv = torch.randn(8, 5, 16)
        self.assertIs(_maybe_expand_kv_heads(kv, 8, 8), kv)

        # GQA: 2 kv heads -> 8 q heads, q-head h maps to kv-head h // group.
        kv = torch.randn(2, 4, 3)
        out = _maybe_expand_kv_heads(kv, 8, 2)
        self.assertEqual(tuple(out.shape), (8, 4, 3))
        group = 8 // 2
        for h in range(8):
            self.assertTrue(torch.equal(out[h], kv[h // group]))


class TestSupportTritonPatch(CustomTestCase):
    """_patch_support_triton_for_tpu must treat both Pallas TPU backend names
    like non-Triton backends (no Triton driver exists on TPU)."""

    def test_pallas_tpu_is_non_triton_after_patch(self):
        import sglang.srt.utils.common as uc
        from sglang_tpu.patches import (
            _patch_support_triton_for_tpu,
        )

        # Before: stock allowlist treats an unknown name as Triton-capable.
        self.assertTrue(uc.support_triton("pallas_tpu_flash"))
        self.assertTrue(uc.support_triton("pallas_tpu_rpa_v3"))

        _patch_support_triton_for_tpu()

        # After: both Pallas backend names are non-Triton; others unchanged.
        self.assertFalse(uc.support_triton("pallas_tpu_flash"))
        self.assertFalse(uc.support_triton("pallas_tpu_rpa_v3"))
        self.assertFalse(uc.support_triton("torch_native"))
        self.assertTrue(uc.support_triton("flashinfer"))

        # Idempotent.
        _patch_support_triton_for_tpu()
        self.assertFalse(uc.support_triton("pallas_tpu_flash"))
        self.assertFalse(uc.support_triton("pallas_tpu_rpa_v3"))


class TestPallasFlashTokenSlotPadding(CustomTestCase):
    def test_pool_width_is_not_power_of_two_rounded(self):
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        pool = ReqToTokenPool(
            size=2,
            max_context_len=16388,
            device="cpu",
            enable_memory_saver=False,
        )
        self.assertEqual(pool.req_to_token.shape[1], 16388)
        self.assertEqual(pool.max_context_len, 16388)

    def test_padding_uses_reserved_slot_zero(self):
        import torch

        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
        from sglang_tpu.layers.attention.attention_pallas import (
            _pad_token_slots,
        )

        allocator = TokenToKVPoolAllocator(
            size=8,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.assertNotIn(0, allocator.free_pages.tolist())

        slots = torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int32)
        padded = _pad_token_slots(slots, 8)
        self.assertEqual(tuple(padded.shape), (2, 8))
        self.assertTrue(torch.equal(padded[:, :3], slots))
        self.assertTrue(
            torch.equal(padded[:, 3:], torch.zeros((2, 5), dtype=torch.int32))
        )

    def test_padding_helper_also_slices(self):
        import torch

        from sglang_tpu.layers.attention.attention_pallas import (
            _pad_token_slots,
        )

        slots = torch.tensor([[11, 12, 13]], dtype=torch.int32)
        self.assertTrue(torch.equal(_pad_token_slots(slots, 2), slots[:, :2]))


class TestPallasRegistration(CustomTestCase):
    """Both Pallas backend names must be registered in the attention
    registry at module import time (same convention as ascend / intel_xpu)."""

    def test_registered(self):
        from sglang.srt.layers.attention.attention_registry import (
            ATTENTION_BACKENDS,
        )

        for name in ("pallas_tpu_flash", "pallas_tpu_rpa_v3"):
            self.assertIn(name, ATTENTION_BACKENDS)
            self.assertTrue(callable(ATTENTION_BACKENDS[name]))


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestFlashKernelNumerics(CustomTestCase):
    """flash_attention (prefill/extend/decode shapes) vs float32 SDPA.

    Exercises the real backend path including the q<128 -> pad-and-slice
    in _attend, so a regression in padding or scaling is caught.
    """

    def _check(self, H, Sq, Skv, D, causal, tol=2e-2):
        import torch

        from sglang_tpu.layers.attention.attention_pallas import (
            _attend,
            _get_flash_op,
        )

        torch.manual_seed(0)
        scale = 1.0 / math.sqrt(D)
        q = torch.randn(1, H, Sq, D, dtype=torch.float32)
        k = torch.randn(1, H, Skv, D, dtype=torch.float32)
        v = torch.randn(1, H, Skv, D, dtype=torch.float32)
        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=causal
        )
        op = _get_flash_op(bool(causal), float(scale))
        out = _attend(op, q.to("tpu"), k.to("tpu"), v.to("tpu")).to("cpu")
        self.assertEqual(tuple(out.shape), (1, H, Sq, D))
        self.assertLess((out - ref).abs().max().item(), tol)

    def test_prefill_short(self):
        self._check(H=8, Sq=27, Skv=27, D=128, causal=True)

    def test_extend_redundant(self):
        self._check(H=8, Sq=41, Skv=41, D=128, causal=True)

    def test_decode_single_query(self):
        self._check(H=8, Sq=1, Skv=40, D=128, causal=False)


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestPagedKernelNumerics(CustomTestCase):
    """jax pallas paged_attention vs float32 SDPA reference. Guards GQA,
    page_size variants, the required q pre-scale, and length masking."""

    def _check(self, batch, Hq, Hkv, D, page_size, pps, lengths, tol=3e-2):
        import numpy as np
        import torch

        from sglang_tpu.layers.attention.attention_pallas import (
            _get_paged_op,
        )

        total_pages = batch * pps
        rng = np.random.default_rng(0)
        q = rng.standard_normal((batch, Hq, D)).astype(np.float32)
        kp = rng.standard_normal((Hkv, total_pages, page_size, D)).astype(np.float32)
        vp = rng.standard_normal((Hkv, total_pages, page_size, D)).astype(np.float32)
        page_indices = np.arange(total_pages, dtype=np.int32).reshape(batch, pps)
        scale = 1.0 / math.sqrt(D)

        ppb = max(1, min(8, pps))
        op = _get_paged_op(ppb)
        out = (
            op(
                torch.from_numpy((q * scale).astype(np.float32)).to("tpu"),
                torch.from_numpy(kp).to("tpu"),
                torch.from_numpy(vp).to("tpu"),
                torch.from_numpy(np.array(lengths, np.int32)).to("tpu"),
                torch.from_numpy(page_indices).to("tpu"),
            )
            .to("cpu")
            .numpy()
        )
        g = Hq // Hkv
        for b in range(batch):
            kv = kp[:, page_indices[b]].reshape(Hkv, pps * page_size, D)
            vv = vp[:, page_indices[b]].reshape(Hkv, pps * page_size, D)
            for h in range(Hq):
                kh = kv[h // g][: lengths[b]]
                vh = vv[h // g][: lengths[b]]
                s = (q[b, h] @ kh.T) * scale
                s -= s.max()
                w = np.exp(s)
                w /= w.sum()
                exp = w @ vh
                d = np.abs(out[b, h] - exp).max()
                self.assertLess(d, tol, f"b{b} h{h} diff={d}")

    def test_gqa_page16(self):
        self._check(2, 8, 2, 128, 16, 4, [40, 23])

    def test_gqa_page1(self):
        self._check(2, 8, 2, 128, 1, 64, [40, 23])

    def test_mha_page64(self):
        self._check(1, 8, 8, 128, 64, 1, [50])


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestPallasFlashDecodePaddingNumerics(CustomTestCase):
    def test_non_power_of_two_pool_width(self):
        import numpy as np
        import torch

        from sglang_tpu.layers.attention.attention_pallas import (
            TpuPallasAttnBackend,
            _bucket_kv_len,
        )

        batch, hq, hkv, dim = 2, 8, 2, 128
        pool_width = 69
        lengths = np.array([65, 23], dtype=np.int32)
        bucket_len = _bucket_kv_len(int(lengths.max()))
        self.assertEqual(bucket_len, 128)
        self.assertLess(pool_width, bucket_len)

        slots = np.stack(
            (
                np.arange(1, pool_width + 1, dtype=np.int32),
                np.arange(pool_width + 1, 2 * pool_width + 1, dtype=np.int32),
            )
        )
        rng = np.random.default_rng(0)
        query = rng.standard_normal((batch, hq, dim)).astype(np.float32)
        k_cache = rng.standard_normal((2 * pool_width + 1, hkv, dim)).astype(np.float32)
        v_cache = rng.standard_normal((2 * pool_width + 1, hkv, dim)).astype(np.float32)
        # Padding must remain masked even when reserved slot 0 is conspicuous.
        k_cache[0] = 1000
        v_cache[0] = -1000
        scale = 1.0 / math.sqrt(dim)

        output = torch.empty((batch, hq, dim), dtype=torch.float32, device="tpu")
        TpuPallasAttnBackend._run_paged_forward_decode(
            None,
            torch.from_numpy(query).to("tpu"),
            output,
            torch.from_numpy(k_cache).to("tpu"),
            torch.from_numpy(v_cache).to("tpu"),
            torch.from_numpy(slots).to("tpu"),
            torch.arange(batch, dtype=torch.int32, device="tpu"),
            torch.from_numpy(lengths).to("tpu"),
            scaling=scale,
        )
        actual = output.cpu().numpy()

        groups = hq // hkv
        for b in range(batch):
            for h in range(hq):
                token_slots = slots[b, : lengths[b]]
                kh = k_cache[token_slots, h // groups]
                vh = v_cache[token_slots, h // groups]
                scores = (query[b, h] @ kh.T) * scale
                scores -= scores.max()
                weights = np.exp(scores)
                weights /= weights.sum()
                expected = weights @ vh
                diff = np.abs(actual[b, h] - expected).max()
                self.assertLess(diff, 3e-2, f"b{b} h{h} diff={diff}")


if __name__ == "__main__":
    unittest.main()
