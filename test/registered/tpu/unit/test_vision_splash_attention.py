"""Tests for segment-ID vision attention with block skipping (``VisionTpuSplashAttention``).

* On CPU, the inherited ``VisionTpuAttention.forward`` with the Pallas op swapped for a
  torch reference: ``segment_ids`` and ``cu_seqlens`` give per-segment attention,
  sequences pad to whole 1,024-token blocks with the pads as one more segment, KV
  heads expand, and a dense batch (bsz > 1) goes to the flash path unchanged.
* On the JAX CPU device: the block map marks every block pair that holds a same-segment
  query-key pair (and only overlapping ones); block sizes reject lengths off the 128
  grid; the op rejects a batch of more than one sequence; and the op's JAX body,
  splash in interpret mode, equals the reference for contiguous and interleaved
  segments and for full attention over a padded sequence (real tokens, then a pad
  tail).
* On the TPU in bf16, the registered op through ``VisionTpuSplashAttention``, eager and
  compiled, against an fp32 CPU reference: tiles of a grid that is not a multiple of
  the tile size, in tile order and in row-major order, with pad tokens; lengths that
  are not multiples of 1,024 (16 to 3,200 tokens) at head dims 64, 72, 80 and 128; and
  ``cu_seqlens``; and a padded batch of two, bitwise equal to ``VisionTpuAttention``.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=60,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "infra", "multimodal", "torch-compile"],
)

import math
import unittest

import torch
from einops import rearrange

from sglang.srt.layers.attention.vision import VisionTpuAttention
from sglang.srt.utils import is_tpu
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.vision_splash import VisionTpuSplashAttention


def _ref_token_attention(q, k, v, seg, scale):
    """Per-segment non-causal attention in fp32. q, k, v: [s, h, d]; seg: [s]."""
    qr, kr, vr = (rearrange(x.float(), "s h d -> h s d") for x in (q, k, v))
    scores = torch.einsum("hqd,hkd->hqk", qr, kr) * scale
    scores = scores.masked_fill(~(seg[:, None] == seg[None, :]), float("-inf"))
    out = torch.einsum("hqk,hkd->hqd", torch.softmax(scores, dim=-1), vr)
    return rearrange(out, "h s d -> s h d")


def _ref_batch_attention(q, k, v, seg, scale):
    """``_ref_token_attention`` per row in the op layout: q, k, v [b, h, s, d]; seg [b, s]."""
    return torch.stack(
        [
            rearrange(
                _ref_token_attention(
                    *(rearrange(x[i], "h s d -> s h d") for x in (q, k, v)),
                    seg[i],
                    scale,
                ),
                "s h d -> h s d",
            )
            for i in range(q.shape[0])
        ]
    )


def _segments(lengths):
    return torch.cat(
        [torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lengths)]
    )


def _tiles(grid_h, grid_w, tile):
    """Row-major tile ids of a grid cut into tile x tile tiles (edge tiles partial)."""
    r = torch.arange(grid_h * grid_w) // grid_w
    c = torch.arange(grid_h * grid_w) % grid_w
    return ((r // tile) * -(-grid_w // tile) + c // tile).to(torch.int32)


class _CpuSplashAttn(VisionTpuSplashAttention):
    """``VisionTpuSplashAttention`` with the Pallas op swapped for a torch reference."""

    def __init__(self, head_dim, num_heads, num_kv_heads, scale, flatten_batch=True):
        torch.nn.Module.__init__(self)
        self.head_size = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.flatten_batch = flatten_batch
        self.scale = scale
        self.seen_ids = []

        self.batched_calls = []

        def splash_op(q, k, v, ids):
            self.seen_ids.append(ids)
            return _ref_batch_attention(q, k, v, ids, scale).to(q.dtype)

        def batched_attention(q, k, v, bsz, **kwargs):
            self.batched_calls.append((bsz, kwargs))
            return torch.zeros_like(q)

        self._flash_op = splash_op
        self._batched_attention = batched_attention


class TestVisionTpuSplashAttentionCpu(CustomTestCase):
    """The inherited forward returns per-segment attention of the inputs."""

    def _check(self, seg, h=2, kv_h=None, d=16, seed=0, **forward_kwargs):
        torch.manual_seed(seed)
        kv_h = kv_h or h
        scale = 1.0 / math.sqrt(d)
        s = seg.shape[0]
        q = torch.randn(s, h, d)
        k, v = (torch.randn(s, kv_h, d) for _ in range(2))
        attn = _CpuSplashAttn(d, h, kv_h, scale)
        if not forward_kwargs:
            forward_kwargs = {"segment_ids": seg[None].clone()}
        out = attn.forward(q, k, v, bsz=1, **forward_kwargs)
        rep = h // kv_h
        ref = _ref_token_attention(
            q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), seg, scale
        )
        torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)
        return attn

    def test_contiguous_segments(self):
        self._check(_segments([128, 100, 28, 256]))

    def test_cu_seqlens(self):
        # Packed images described by cu_seqlens instead of segment ids.
        self._check(
            _segments([300, 200, 524]),
            seed=10,
            cu_seqlens=torch.tensor([0, 300, 500, 1024], dtype=torch.int32),
        )

    def test_full_attention_with_a_pad_tail(self):
        # A full-attention layer: one segment for the real tokens, one for the pads.
        self._check(_segments([400, 112]), seed=7)

    def test_interleaved_segments(self):
        self._check(_tiles(32, 32, 16), seed=1)

    def test_pads_to_whole_1024_blocks_as_one_more_segment(self):
        for lengths, padded in (([60, 40], 1024), ([2000, 1200], 4096)):
            with self.subTest(seq_len=sum(lengths)):
                attn = self._check(_segments(lengths), seed=2)
                (ids,) = attn.seen_ids
                self.assertEqual(tuple(ids.shape), (1, padded))
                self.assertTrue(bool((ids[0, sum(lengths) :] == 2).all()))

    def test_kv_heads_expand_to_query_heads(self):
        self._check(_segments([100, 60, 96]), h=4, kv_h=2, seed=3)

    def test_dense_batch_keeps_the_flash_path(self):
        # bsz > 1 goes to the parent's flash path with its arguments unchanged; the
        # splash op is not called.
        q = torch.randn(2 * 256, 2, 16)
        attn = _CpuSplashAttn(16, 2, 2, 0.25, flatten_batch=False)
        cu = torch.tensor([0, 200, 456], dtype=torch.int32)
        attn.forward(q, q, q, bsz=2, cu_seqlens=cu)
        self.assertEqual(len(attn.batched_calls), 1)
        bsz, kwargs = attn.batched_calls[0]
        self.assertEqual(bsz, 2)
        self.assertIs(kwargs["cu_seqlens"], cu)
        self.assertEqual(attn.seen_ids, [])


class TestVisionSplashJax(CustomTestCase):
    """The op's JAX body on the JAX CPU device (exact fp32; splash in interpret mode)."""

    @staticmethod
    def _cpu():
        import jax

        return jax.default_device(jax.devices("cpu")[0])

    def _block_map(self, seg, block_q, block_kv):
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu.attention_pallas import _vision_splash_block_map

        with self._cpu():
            mask, nxt = _vision_splash_block_map(
                jnp.asarray(seg.numpy()), block_q, block_kv
            )
        return np.asarray(mask)[0], np.asarray(nxt)[0]

    def test_block_map_covers_every_same_segment_pair(self):
        for name, seg in (
            ("contiguous", _segments([700, 300, 1024, 50, 998, 1024])),
            ("interleaved", _tiles(64, 64, 32)),
            ("real and pad", _segments([3500, 596])),
        ):
            with self.subTest(order=name):
                mask, nxt = self._block_map(seg, 1024, 512)
                q_blocks, kv_blocks = seg.reshape(-1, 1024), seg.reshape(-1, 512)
                for i, qb in enumerate(q_blocks):
                    for j, kb in enumerate(kv_blocks):
                        meets = bool((qb[:, None] == kb[None, :]).any())
                        if meets:
                            self.assertEqual(mask[i, j], 2, (i, j))
                        if mask[i, j]:
                            self.assertEqual(nxt[i, j], j)
                        else:
                            self.assertEqual(mask[i, nxt[i, j]], 2)

    def test_out_of_order_segments_are_exact_but_denser(self):
        # Contiguous segments numbered out of order: q block 0 holds ids {0, 5}, an id
        # range that meets every kv block. Numbered in order, each q block meets only
        # its own two kv blocks.
        out_of_order = torch.tensor([0, 5, 1, 4], dtype=torch.int32)
        in_order = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        dense, _ = self._block_map(out_of_order.repeat_interleave(1024), 2048, 1024)
        sparse, _ = self._block_map(in_order.repeat_interleave(1024), 2048, 1024)
        self.assertEqual(int((dense > 0).sum()), 4 + 2)
        self.assertEqual(int((sparse > 0).sum()), 2 + 2)
        self._check_body(out_of_order.repeat_interleave(1024), h=1, d=16, seed=9)

    def test_block_sizes_reject_lengths_off_the_128_grid(self):
        from sglang_tpu.attention_pallas import _vision_splash_block_sizes

        for bad in (0, 100, 1000):
            with self.subTest(seq_len=bad):
                with self.assertRaises(ValueError):
                    _vision_splash_block_sizes(bad)

    def test_op_runs_one_sequence(self):
        # Dense batches never reach splash (VisionTpuSplashAttention hands them to the
        # flash op), so the op takes batch 1 only.
        import jax.numpy as jnp

        from sglang_tpu.attention_pallas import _vision_splash_attention

        q = jnp.zeros((2, 1, 1024, 16), jnp.float32)
        with self._cpu():
            with self.assertRaises(ValueError):
                _vision_splash_attention(
                    q, q, q, jnp.zeros((2, 1024), jnp.int32), sm_scale=0.25
                )

    def test_block_map_is_sparse_for_contiguous_segments(self):
        mask, _ = self._block_map(_segments([1024] * 16), 2048, 1024)
        # Each 2048-token q block holds two segments, i.e. two of 16 kv blocks.
        self.assertEqual(int((mask > 0).sum()), 8 * 2)

    def _check_body(self, seg, h, d, seed):
        """seg: [s]; q, k, v in the op layout [1, h, s, d]."""
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu.attention_pallas import _vision_splash_attention

        torch.manual_seed(seed)
        seg = seg[None]
        scale = 1.0 / math.sqrt(d)
        q, k, v = (torch.randn(1, h, seg.shape[1], d) for _ in range(3))
        with self._cpu():
            out = _vision_splash_attention(
                *(jnp.asarray(x.numpy()) for x in (q, k, v)),
                jnp.asarray(seg.numpy()),
                sm_scale=scale,
                interpret=True,
            )
        torch.testing.assert_close(
            torch.from_numpy(np.asarray(out)),
            _ref_batch_attention(q, k, v, seg, scale),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_splash_body_contiguous_segments(self):
        # 4,096 tokens: 2 q blocks x 4 kv blocks, most pairs skipped.
        self._check_body(_segments([1024, 700, 324, 1024, 1000, 24]), h=2, d=16, seed=5)

    def test_splash_body_interleaved_segments(self):
        self._check_body(_tiles(64, 64, 32), h=1, d=16, seed=6)

    def test_splash_body_full_attention_with_a_pad_tail(self):
        # Dense: the 3,500 real tokens reach into every q and kv block (2 x 4 live).
        mask, _ = self._block_map(_segments([3500, 596]), 2048, 1024)
        self.assertEqual(int((mask > 0).sum()), 2 * 4)
        self._check_body(_segments([3500, 596]), h=2, d=16, seed=8)


def _tpu_available() -> bool:
    """The TPU platform (what VisionTpuAttention requires) with a usable device.

    torch.accelerator alone is also true on CUDA hosts.
    """
    return (
        is_tpu() and hasattr(torch, "accelerator") and torch.accelerator.is_available()
    )


@unittest.skipUnless(_tpu_available(), "requires the TPU platform and a TPU device")
class TestVisionSplashOnTpu(CustomTestCase):
    """The registered Pallas op in bf16, eager and compiled, against fp32 on the CPU."""

    HEADS, HEAD_DIM = 2, 96
    # bf16 inputs and outputs against an fp32 reference of the same bf16 inputs.
    BF16_TOL = 2e-2

    def test_backend_registration(self):
        from sglang.srt.layers.attention.vision import QKV_BACKEND_IMPL
        from sglang_tpu.tpu_platform import _register_attention_backends

        _register_attention_backends()
        self.assertIs(QKV_BACKEND_IMPL["tpu"], VisionTpuSplashAttention)

    def test_unaligned_lengths_and_head_dims(self):
        # Lengths that are not multiples of 1,024 (padded inside forward), down to a
        # few patches, and head dims that are not multiples of 32 or 128; one segment,
        # and three packed segments.
        for n, d in ((16, 64), (100, 80), (1000, 72), (3200, 72), (3200, 128)):
            for name, seg in (
                ("dense", _segments([n])),
                ("packed", _segments([n // 2, n // 3, n - n // 2 - n // 3])),
            ):
                with self.subTest(seq_len=n, head_dim=d, pattern=name):
                    attn = VisionTpuSplashAttention(d, 2, 2)
                    g = torch.Generator().manual_seed(n + d)
                    q, k, v = (
                        torch.randn(n, 2, d, generator=g).bfloat16() for _ in range(3)
                    )
                    ref = _ref_token_attention(q, k, v, seg, attn.scale)
                    out = attn(
                        *(x.to("tpu") for x in (q, k, v)),
                        1,
                        segment_ids=seg[None].to("tpu"),
                    )
                    self.assertEqual(tuple(out.shape), (n, 2, d))
                    torch.testing.assert_close(
                        out.float().cpu(), ref, atol=self.BF16_TOL, rtol=self.BF16_TOL
                    )

    def test_cu_seqlens_and_padded_batch(self):
        # Three packed images by cu_seqlens (splash); then a dense batch of two rows
        # (700 and 1,000 real tokens of 1,000), which stays on the flash path: bitwise
        # equal to VisionTpuAttention.
        attn = VisionTpuSplashAttention(
            self.HEAD_DIM, self.HEADS, self.HEADS, flatten_batch=True
        )
        attn_batch = VisionTpuSplashAttention(self.HEAD_DIM, self.HEADS, self.HEADS)
        g = torch.Generator().manual_seed(1)
        q, k, v = (
            torch.randn(2000, self.HEADS, self.HEAD_DIM, generator=g).bfloat16()
            for _ in range(3)
        )
        dev = [x.to("tpu") for x in (q, k, v)]
        for name, module, bsz, cu, refs in (
            ("cu_seqlens", attn, 1, [0, 900, 1500, 2000], [_segments([900, 600, 500])]),
            (
                "padded batch",
                attn_batch,
                2,
                [0, 700, 1700],
                [_segments([700, 300]), _segments([1000])],
            ),
        ):
            with self.subTest(case=name):
                out = (
                    module(*dev, bsz, cu_seqlens=torch.tensor(cu, dtype=torch.int32))
                    .float()
                    .cpu()
                )
                rows = 2000 // bsz
                for i, seg in enumerate(refs):
                    r = slice(i * rows, (i + 1) * rows)
                    ref = _ref_token_attention(q[r], k[r], v[r], seg, attn.scale)
                    torch.testing.assert_close(
                        out[r], ref, atol=self.BF16_TOL, rtol=self.BF16_TOL
                    )
                if bsz > 1:
                    flash = VisionTpuAttention(self.HEAD_DIM, self.HEADS, self.HEADS)
                    cu_t = torch.tensor(cu, dtype=torch.int32)
                    torch.testing.assert_close(
                        out,
                        flash(*dev, bsz, cu_seqlens=cu_t).float().cpu(),
                        atol=0,
                        rtol=0,
                    )

    def test_splash_op_matches_reference_eager_and_compiled(self):
        # 32x32 tiles of a 60x68 grid (full, 32x4, 28x32 and 28x4 tiles): 4,080
        # tokens, padded to 4,096 inside the op.
        raster = _tiles(60, 68, 32)
        tile_order = torch.argsort(raster.to(torch.int64), stable=True)
        attn = VisionTpuSplashAttention(self.HEAD_DIM, self.HEADS, self.HEADS)

        def run(q, k, v, ids):
            return attn(q, k, v, 1, segment_ids=ids)

        compiled_run = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
        for name, seg in (("tile order", raster[tile_order]), ("row-major", raster)):
            with self.subTest(order=name):
                g = torch.Generator().manual_seed(0)
                q, k, v = (
                    torch.randn(
                        seg.shape[0], self.HEADS, self.HEAD_DIM, generator=g
                    ).bfloat16()
                    for _ in range(3)
                )
                ref = _ref_token_attention(q, k, v, seg, attn.scale)
                dev = [x.to("tpu") for x in (q, k, v, seg[None])]
                eager = run(*dev)
                compiled = compiled_run(*dev)
                self.assertEqual(eager.dtype, torch.bfloat16)
                self.assertEqual(tuple(eager.shape), tuple(q.shape))
                torch.testing.assert_close(
                    eager.float().cpu(), ref, atol=self.BF16_TOL, rtol=self.BF16_TOL
                )
                torch.testing.assert_close(compiled.cpu(), eager.cpu(), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
