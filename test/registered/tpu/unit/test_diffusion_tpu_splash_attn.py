"""Numerics of the multimodal_gen TPU splash attention backend.

Every dispatch branch of ``tpu_attention`` is checked against a float32
reference: the custom Pallas kernel (non-causal attention over at least 512
keys, including ragged KV tails, text cross-attention and a short
sequence-parallel query shard), stock ``make_splash_mha`` for causal shapes,
short query shards and fewer than 512 aligned keys (including zero-padded KV,
whose padded keys must not take softmax weight), and the SDPA fallback.

The numeric tests need the Pallas kernels (TPU v7x, torch_tpu, jax) and are
skipped otherwise; each asserts which path ran, so a lane without the kernels
cannot pass by comparing SDPA with SDPA.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=170, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.test.test_utils import CustomTestCase


def _skip_if_no_tpu() -> bool:
    try:
        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


def _reference(q, k, v, scale, is_causal):
    """float32 attention on CPU, chunked over queries."""
    q, k, v = q.float(), k.float(), v.float()
    q_seq, kv_seq = q.shape[2], k.shape[2]
    out = torch.empty_like(q)
    for start in range(0, q_seq, 1024):
        stop = min(start + 1024, q_seq)
        scores = torch.matmul(q[:, :, start:stop], k.transpose(-2, -1)) * scale
        if is_causal:
            rows = torch.arange(start, stop)[:, None]
            cols = torch.arange(kv_seq)[None, :]
            scores = scores.masked_fill(cols > rows, float("-inf"))
        out[:, :, start:stop] = torch.matmul(torch.softmax(scores, dim=-1), v)
    return out


def _rel_l2(got, want):
    return ((got.float() - want).norm() / want.norm()).item()


def _inputs(
    batch,
    heads,
    q_seq,
    kv_seq,
    *,
    kv_heads=None,
    q_gain=1.0,
    negative_scores=False,
):
    gen = torch.Generator().manual_seed(q_seq * 7919 + kv_seq)
    shape_q = (batch, heads, q_seq, 128)
    shape_kv = (batch, kv_heads or heads, kv_seq, 128)
    q = torch.randn(shape_q, generator=gen) * q_gain
    k = torch.randn(shape_kv, generator=gen)
    v = torch.randn(shape_kv, generator=gen)
    if negative_scores:
        # Every real logit is strongly negative (mean -64 / sqrt(128) = -5.7),
        # so a zero-padded key (logit 0) would dominate the softmax if it were
        # not masked.
        direction = torch.ones(128) / 128**0.5
        q = q * 0.1 + 8.0 * direction
        k = k * 0.1 - 8.0 * direction
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def _kernel_calls(graphs):
    """``(flash, make_splash_mha)`` op calls in the FX graphs Dynamo compiled."""
    names = [
        str(node.target).split(".")[1]
        for graph in graphs
        for node in graph.graph.nodes
        if node.op == "call_function"
        and str(node.target).startswith("sglang_tpu_pallas.")
    ]
    return (
        names.count("diffusion_flash_attention"),
        names.count("diffusion_splash_mha"),
    )


# (flash, make_splash_mha) kernel calls of each path.
_PATH_CALLS = {"flash": (1, 0), "splash": (0, 1), "sdpa": (0, 0)}


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuSplashAttentionNumerics(CustomTestCase):
    def _kernels(self):
        """The backend module with its ops registered; skips without them."""
        from sglang_tpu.multimodal_gen import tpu_splash_attn as tsa

        if not tsa._check_splash_available():
            self.skipTest("Pallas kernels unavailable (needs TPU v7x and jax)")
        tsa._register_ops()
        return tsa

    def _check(
        self,
        batch,
        heads,
        q_seq,
        kv_seq,
        *,
        expect,
        is_causal=False,
        scale=128**-0.5,
        attend=None,
        **kwargs,
    ):
        """``expect`` is the path that must run: "flash", "splash" or "sdpa".

        ``attend(q, k, v)`` computes ``[B, H, S, D]`` attention from device
        tensors; the default is ``tpu_attention``.
        """
        tsa = self._kernels()
        if attend is None:

            def attend(q, k, v):
                return tsa.tpu_attention(q, k, v, scale=scale, is_causal=is_causal)

        q, k, v = _inputs(batch, heads, q_seq, kv_seq, **kwargs)
        with (
            patch.object(tsa, "_FLASH_OP", wraps=tsa._FLASH_OP) as flash,
            patch.object(tsa, "_SPLASH_MHA_OP", wraps=tsa._SPLASH_MHA_OP) as splash,
        ):
            got = attend(q.to("tpu"), k.to("tpu"), v.to("tpu"))
        calls = (flash.call_count, splash.call_count)
        self.assertEqual(calls, _PATH_CALLS[expect], expect)
        self._assert_accurate(got.to("cpu"), q, k, v, scale, is_causal)

    def _assert_accurate(self, got, q, k, v, scale, is_causal):
        """``got`` against float32 attention, within twice bf16 SDPA's error."""
        if k.shape[1] != q.shape[1]:
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        want = _reference(q, k, v, scale, is_causal)
        sdpa = F.scaled_dot_product_attention(
            q.to("tpu"), k.to("tpu"), v.to("tpu"), scale=scale, is_causal=is_causal
        ).to("cpu")
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(got).any().item())
        rel, sdpa_rel = _rel_l2(got, want), _rel_l2(sdpa, want)
        err = (got.float() - want).abs().max().item()
        sdpa_err = (sdpa.float() - want).abs().max().item()
        print(
            f"{self._testMethodName}: rel-L2 {rel:.3e} (sdpa {sdpa_rel:.3e}), "
            f"max abs {err:.3e} (sdpa {sdpa_err:.3e})"
        )
        # Relative L2 over the whole output: a padded key taking softmax weight
        # stays under a max-abs bound but moves rel-L2 by ~1e-1. bf16 inputs
        # bound the achievable accuracy, so allow twice bf16 SDPA's error.
        self.assertLess(rel, max(1e-2, 2 * sdpa_rel), "rel-L2")
        self.assertLess(err, max(2e-2, 2 * sdpa_err), "max abs")

    # Custom kernel: non-causal, q_seq >= 256, kv_seq >= 512.
    def test_custom_two_blocks_and_a_ragged_tail(self):
        # 4608 = 2 * 2048 + 512.
        self._check(1, 4, 4608, 4608, expect="flash")

    def test_custom_whole_kv_blocks(self):
        # 4096 = 2 * 2048: no partial last block.
        self._check(1, 2, 4096, 4096, expect="flash")

    def test_custom_single_kv_block(self):
        # One KV block: the first, last and only grid step coincide.
        self._check(1, 2, 2048, 2048, expect="flash")

    def test_custom_softmax_scale(self):
        # A scale other than head_dim ** -0.5, which equals the default.
        self._check(1, 2, 4128, 4128, expect="flash", scale=0.05)

    def test_custom_ragged_kv_tail(self):
        # 4128 = 2 * 2048 + 32: the last KV block is a 32-row partial block.
        self._check(1, 2, 4128, 4128, expect="flash")

    def test_custom_odd_kv_tail(self):
        # 8191 = 3 * 2048 + 2047: an odd-length partial last block.
        self._check(1, 2, 8191, 8191, expect="flash")

    def test_custom_short_query_shard(self):
        # A sequence-parallel query shard attending the gathered KV.
        self._check(1, 2, 880, 3520, expect="flash")

    def test_custom_batched(self):
        # B=2, as with batched CFG.
        self._check(2, 2, 8190, 8190, expect="flash")

    def test_custom_high_logit_gain(self):
        # Large logits make the softmax peaked and amplify rounding.
        self._check(1, 2, 8190, 8190, expect="flash", q_gain=8.0)

    def test_custom_negative_scores(self):
        self._check(1, 2, 4128, 4128, expect="flash", negative_scores=True)

    def test_custom_largest_block_q(self):
        # q = kv >= 12800 picks block_q=6400, the top of the VMEM model; the
        # second query block starts at row 6400.
        self._check(1, 1, 12800, 12800, expect="flash")

    # Text cross-attention on the custom kernel: long query, short KV.
    def test_cross_attention_aligned_kv(self):
        self._check(1, 4, 4096, 512, expect="flash")

    def test_cross_attention_ragged_query(self):
        # No block size divides 4100 query rows; the edge block is masked.
        self._check(1, 4, 4100, 512, expect="flash")

    def test_cross_attention_unaligned_kv(self):
        # 600 keys: a ragged last KV block.
        self._check(1, 2, 18900, 600, expect="flash", negative_scores=True)

    def test_cross_attention_long_query(self):
        # Short KV leaves VMEM for the largest query block.
        self._check(1, 2, 37800, 512, expect="flash")

    def test_cross_attention_capped_block_q_wide_kv(self):
        # 18944 = 2 x 9472 rows at block_kv=1280: block_q at the cap, the
        # tightest VMEM point of the short-KV regime.
        self._check(1, 2, 18944, 1280, expect="flash")

    # make_splash_mha, long query against fewer than 512 aligned keys.
    def test_splash_mha_cross_attention(self):
        for kv_seq in (256, 384):
            with self.subTest(kv_seq=kv_seq):
                self._check(1, 2, 4096, kv_seq, expect="splash")

    def test_cross_attention_kv_not_multiple_of_1024(self):
        self._check(1, 2, 4096, 1536, expect="flash", negative_scores=True)

    # Medium shapes (custom kernel) and causal (make_splash_mha).
    def test_medium_ragged_kv(self):
        # 1000 keys: a ragged last KV block.
        self._check(1, 4, 1000, 1000, expect="flash", negative_scores=True)

    def test_medium_aligned(self):
        self._check(1, 2, 1024, 1024, expect="flash")

    def test_short_query_long_padded_kv(self):
        # q_seq < 256 with kv > 512: make_splash_mha with segment-id masking.
        self._check(1, 2, 200, 4128, expect="splash", negative_scores=True)

    def test_short_query_medium_padded_kv(self):
        # q_seq < 256 with 512 < kv <= 1024: 700 keys pad to 768, masked.
        self._check(1, 2, 200, 700, expect="splash", negative_scores=True)

    def test_long_query_unaligned_kv(self):
        # 1024 < kv < 2048, not 128-aligned: a ragged last KV block.
        self._check(1, 2, 4096, 1500, expect="flash", negative_scores=True)

    def test_causal_padded(self):
        self._check(1, 2, 1000, 1000, expect="splash", is_causal=True)

    def test_splash_high_logit_gain(self):
        # make_splash_mha pre-scales q in bf16; peaked softmaxes amplify that.
        self._check(1, 2, 1000, 1000, expect="splash", is_causal=True, q_gain=8.0)

    # Short or unaligned-short KV stays on SDPA.
    def test_short_sequences(self):
        self._check(1, 4, 300, 77, expect="sdpa")

    def test_unaligned_short_kv(self):
        self._check(1, 2, 4096, 300, expect="sdpa")

    def test_gqa(self):
        self._check(1, 4, 4608, 4608, kv_heads=2, expect="flash")

    def test_gqa_make_splash_mha(self):
        self._check(1, 4, 4096, 256, kv_heads=2, expect="splash")

    def test_varlen_segments_use_sdpa(self):
        from sglang_tpu.multimodal_gen import tpu_splash_attn

        impl = tpu_splash_attn.TpuSplashAttentionImpl(
            num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5
        )
        lens = (3000, 2500)
        q, k, v = (
            torch.randn(sum(lens), 2, 128, dtype=torch.bfloat16) for _ in range(3)
        )
        cu = torch.tensor([0, lens[0], sum(lens)], dtype=torch.int32)
        with patch.object(
            tpu_splash_attn, "tpu_attention", side_effect=AssertionError("kernel")
        ):
            got = impl.forward_varlen(
                q.to("tpu"),
                k.to("tpu"),
                v.to("tpu"),
                cu_seqlens=cu.to("tpu"),
                max_seqlen=max(lens),
                cu_seqlens_host=(0, lens[0], sum(lens)),
            ).to("cpu")
        start = 0
        for n in lens:
            seg = slice(start, start + n)
            want = _reference(
                q[None, seg].transpose(1, 2),
                k[None, seg].transpose(1, 2),
                v[None, seg].transpose(1, 2),
                128**-0.5,
                False,
            ).transpose(1, 2)[0]
            self.assertLess((got[seg].float() - want).abs().max().item(), 2e-2)
            start += n

    def _live_length_impl(self):
        tsa = self._kernels()
        impl = tsa.TpuSplashAttentionImpl(
            num_heads=7,
            head_size=128,
            causal=False,
            softmax_scale=128**-0.5,
            packed_trailing_padding=True,
        )
        self.assertTrue(impl.trailing_padding_on_device)
        return tsa, impl

    def test_live_length_varlen(self):
        # MiniMax H3's packed [0, used, total] layout, read on device: one
        # kernel call, padding rows zero, live rows attend live keys only.
        tsa, impl = self._live_length_impl()
        for total, used in ((512, 300), (2560, 1500), (2560, 2048), (4608, 4100)):
            q, k, v = (
                torch.randn(total, 7, 128, dtype=torch.bfloat16) for _ in range(3)
            )
            cu = torch.tensor([0, used, total], dtype=torch.int32)
            with patch.object(
                tsa, "_FLASH_KV_LEN_OP", wraps=tsa._FLASH_KV_LEN_OP
            ) as kernel:
                got = impl.forward_varlen(
                    q.to("tpu"),
                    k.to("tpu"),
                    v.to("tpu"),
                    cu_seqlens=cu.to("tpu"),
                    max_seqlen=total,
                ).to("cpu")
            self.assertEqual(kernel.call_count, 1, (total, used))
            self.assertEqual(got.shape, q.shape)
            self.assertTrue((got[used:] == 0).all(), (total, used))
            live = slice(0, used)
            want = _reference(
                q[None, live].transpose(1, 2),
                k[None, live].transpose(1, 2),
                v[None, live].transpose(1, 2),
                128**-0.5,
                False,
            ).transpose(1, 2)[0]
            err = (got[live].float() - want).abs().max().item()
            self.assertLess(err, 2e-2, (total, used))

    def test_live_length_varlen_with_host_bounds_uses_sdpa(self):
        # A caller that passes host bounds (H3's token refiner) keeps SDPA.
        tsa, impl = self._live_length_impl()
        q, k, v = (
            torch.randn(1024, 7, 128, dtype=torch.bfloat16).to("tpu") for _ in range(3)
        )
        cu = torch.tensor([0, 1024, 1024], dtype=torch.int32).to("tpu")
        with patch.object(
            tsa, "_FLASH_KV_LEN_OP", side_effect=AssertionError("kernel")
        ):
            impl.forward_varlen(
                q, k, v, cu_seqlens=cu, max_seqlen=1024, cu_seqlens_host=(0, 1024, 1024)
            )

    def test_backend_impl_layout(self):
        # The attention layer calls the impl with [B, S, H, D] tensors.
        tsa = self._kernels()
        impl = tsa.TpuSplashAttentionImpl(
            num_heads=4, head_size=128, causal=False, softmax_scale=128**-0.5
        )

        def attend(q, k, v):
            out = impl.forward(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None
            )
            return out.transpose(1, 2)

        self._check(1, 4, 4608, 4608, expect="flash", attend=attend)

    def test_torch_compile_output_layout(self):
        # Dynamo traces the op through its fake impl, whose strides must match
        # the kernel's contiguous output for the layer's reshape to lower.
        import torch_tpu._internal.compile  # noqa: F401
        from torch._dynamo.testing import CompileCounterWithBackend

        tsa = self._kernels()
        impl = tsa.TpuSplashAttentionImpl(
            num_heads=4, head_size=128, causal=False, softmax_scale=128**-0.5
        )

        def attn(q, k, v):
            out = impl.forward(q, k, v, None)
            return out.reshape(out.shape[0], out.shape[1], -1)

        # Custom kernel, and make_splash_mha with a padded query.
        for q_seq, kv_seq, expect in ((4608, 4608, "flash"), (200, 4128, "splash")):
            with self.subTest(q_seq=q_seq, kv_seq=kv_seq):
                q, k, v = _inputs(1, 4, q_seq, kv_seq)
                torch._dynamo.reset()
                # The tpu backend, recording the graphs it compiles: the
                # kernel op must be in them, traced through its fake.
                counter = CompileCounterWithBackend("tpu")
                compiled = torch.compile(attn, backend=counter, dynamic=False)
                # Contiguous [B, S, H, D], as the layer's projections produce.
                got = compiled(
                    *(t.transpose(1, 2).contiguous().to("tpu") for t in (q, k, v))
                ).to("cpu")
                self.assertEqual(
                    _kernel_calls(counter.graphs), _PATH_CALLS[expect], expect
                )
                self.assertEqual(got.shape, (1, q_seq, 4 * 128))
                got = got.reshape(1, q_seq, 4, 128).transpose(1, 2)
                self._assert_accurate(got, q, k, v, 128**-0.5, False)

    def test_registration_failure_falls_back_to_sdpa(self):
        from sglang_tpu import pallas_jax_op
        from sglang_tpu.multimodal_gen import tpu_splash_attn as tsa

        with (
            patch.object(tsa, "_FLASH_OP", None),
            patch.object(tsa, "_SPLASH_MHA_OP", None),
            patch.object(tsa, "_SPLASH_AVAILABLE", True),
            patch.object(
                pallas_jax_op, "cpu_export_jax_op", side_effect=RuntimeError("boom")
            ),
        ):
            tsa._register_ops()
            self.assertFalse(tsa._SPLASH_AVAILABLE)
            self.assertIsNone(tsa._FLASH_OP)
            impl = tsa.TpuSplashAttentionImpl(
                num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5
            )
            self.assertFalse(impl.use_splash)

    def test_float32_falls_back_to_sdpa(self):
        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            tpu_attention,
        )

        q, k, v = (t.float() for t in _inputs(1, 2, 2048, 2048))
        got = tpu_attention(q.to("tpu"), k.to("tpu"), v.to("tpu")).to("cpu")
        self.assertEqual(got.dtype, torch.float32)
        want = _reference(q, k, v, 128**-0.5, False)
        self.assertLess((got - want).abs().max().item(), 2e-2)


class TestTpuSplashDispatch(CustomTestCase):
    """Kernel selection logic: pure Python over shapes, no kernel runs.

    Lives in the TPU lane because importing sglang_tpu needs the TPU stack.
    """

    def test_autotune_block_sizes_fit_the_vmem_budget(self):
        from sglang_tpu.multimodal_gen import tpu_splash_attn as tsa

        budget = int(tsa._V7X_VMEM_CAPACITY_BYTES * tsa._VMEM_BUDGET_FRACTION)
        for q_seq, kv_seq in [
            (880, 3520),
            (4128, 4128),
            (8191, 8191),
            (12800, 12800),
            (75600, 75600),
        ]:
            block_q, block_kv, block_kv_compute = tsa._autotune_block_sizes(
                q_seq, kv_seq
            )
            self.assertLessEqual(
                tsa._score_tile_bytes(block_q, block_kv_compute, 128, block_kv),
                budget,
                (q_seq, kv_seq),
            )
        # Pinned choices: a change here changes the compiled kernels.
        self.assertEqual(tsa._autotune_block_sizes(880, 3520), (512, 2048, 2048))
        self.assertEqual(tsa._autotune_block_sizes(4128, 4128), (2304, 2048, 2048))
        self.assertEqual(tsa._autotune_block_sizes(8191, 8191), (4096, 2048, 2048))
        self.assertEqual(tsa._autotune_block_sizes(12800, 12800), (6400, 2048, 2048))
        self.assertEqual(tsa._autotune_block_sizes(75600, 75600), (6400, 2048, 2048))
        # The budget scales with the capacity it is given.
        self.assertLess(
            tsa._autotune_block_sizes(75600, 75600, 128, 32 * 2**20)[0], 6400
        )

    def test_rejects_non_positive_scale_and_ragged_causal(self):
        from sglang_tpu.multimodal_gen import tpu_splash_attn as tsa

        q = torch.empty((1, 2, 4096, 128), dtype=torch.bfloat16, device="meta")
        k = torch.empty((1, 2, 2048, 128), dtype=torch.bfloat16, device="meta")
        with self.assertRaisesRegex(ValueError, "positive scale"):
            tsa.tpu_attention(q, q, q, scale=-0.1)
        with self.assertRaisesRegex(NotImplementedError, "ragged causal"):
            tsa.tpu_attention(q, k, k, is_causal=True)

    def test_custom_kernel_selection(self):
        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            _use_custom_kernel,
        )

        self.assertTrue(_use_custom_kernel(75600, 75600, False))
        self.assertTrue(_use_custom_kernel(880, 3520, False))
        self.assertTrue(_use_custom_kernel(18900, 512, False))
        self.assertFalse(_use_custom_kernel(75600, 256, False))
        self.assertFalse(_use_custom_kernel(200, 4128, False))
        self.assertFalse(_use_custom_kernel(4096, 4096, True))
        # The edges of the custom-kernel regime.
        self.assertFalse(_use_custom_kernel(4096, 511, False))
        self.assertTrue(_use_custom_kernel(4096, 512, False))
        self.assertFalse(_use_custom_kernel(255, 4128, False))
        self.assertTrue(_use_custom_kernel(256, 4128, False))

    def test_cross_attn_block_kv_divides_kv(self):
        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            _cross_attn_block_kv,
        )

        # The reachable inputs: 128-aligned KV below 512 keys.
        for kv_seq in (128, 256, 384):
            block_kv = _cross_attn_block_kv(kv_seq)
            self.assertEqual(kv_seq % block_kv, 0)
            self.assertLessEqual(block_kv, 256)

    def test_short_kv_query_blocks_are_capped(self):
        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            _autotune_block_sizes,
        )

        for q_seq, kv_seq in ((75600, 512), (37800, 512), (55800, 1024)):
            self.assertLessEqual(_autotune_block_sizes(q_seq, kv_seq)[0], 9472)
        # Long KV: the VMEM model binds first.
        self.assertEqual(_autotune_block_sizes(75600, 75600), (6400, 2048, 2048))
        self.assertEqual(_autotune_block_sizes(18944, 1280)[0], 9472)

    def test_splash_mha_blocks(self):
        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            _splash_mha_block_sizes,
        )

        self.assertEqual(_splash_mha_block_sizes(4096, 384, False), (2048, 128))
        self.assertEqual(_splash_mha_block_sizes(4096, 256, False), (2048, 256))
        self.assertEqual(_splash_mha_block_sizes(4096, 4096, True), (1024, 1024))
        self.assertEqual(_splash_mha_block_sizes(1000, 1000, True), (512, 512))
        self.assertEqual(_splash_mha_block_sizes(200, 4128, False), (256, 256))

    def test_sdpa_fallback_reasons(self):
        from sglang_tpu.multimodal_gen import tpu_splash_attn

        def reason(q_shape, kv_shape, *, is_causal=False, dtype=torch.bfloat16):
            q = torch.empty(q_shape, dtype=dtype, device="meta")
            k = torch.empty(kv_shape, dtype=dtype, device="meta")
            return tpu_splash_attn._sdpa_fallback_reason(q, k, k, is_causal)

        with patch.object(
            tpu_splash_attn, "_check_splash_available", return_value=True
        ):
            self.assertIsNone(reason((1, 4, 4608, 128), (1, 4, 4608, 128)))
            self.assertIsNone(reason((1, 4, 4096, 128), (1, 2, 512, 128)))
            self.assertEqual(reason((1, 4, 300, 128), (1, 4, 77, 128)), "short")
            self.assertEqual(reason((1, 4, 4096, 128), (1, 4, 300, 128)), "short")
            self.assertIsNone(reason((1, 4, 18900, 128), (1, 4, 600, 128)))
            # q < 256 with 512 < kv <= 1024: make_splash_mha, not SDPA.
            self.assertIsNone(reason((1, 4, 200, 128), (1, 4, 700, 128)))
            self.assertEqual(
                reason((1, 4, 1000, 128), (1, 4, 2000, 128), is_causal=True),
                "ragged causal",
            )
            self.assertEqual(
                reason((1, 4, 4608, 64), (1, 4, 4608, 64)), "head_dim=64/64"
            )
            self.assertEqual(reason((1, 3, 4608, 128), (1, 2, 4608, 128)), "heads=3/2")
            self.assertEqual(
                reason((1, 4, 4608, 128), (1, 4, 4608, 128), dtype=torch.float32),
                "dtype=torch.float32/torch.float32/torch.float32",
            )
            self.assertEqual(reason((1, 4, 0, 128), (1, 4, 4608, 128)), "empty")
            q = torch.empty((1, 4, 4608, 128), dtype=torch.bfloat16, device="meta")
            v = torch.empty((1, 4, 4096, 128), dtype=torch.bfloat16, device="meta")
            self.assertTrue(
                tpu_splash_attn._sdpa_fallback_reason(q, q, v, False).startswith(
                    "value shape"
                )
            )
        with patch.object(
            tpu_splash_attn, "_check_splash_available", return_value=False
        ):
            self.assertEqual(
                reason((1, 4, 4608, 128), (1, 4, 4608, 128)), "pallas unavailable"
            )


if __name__ == "__main__":
    unittest.main()
