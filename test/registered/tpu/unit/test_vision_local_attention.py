"""Tests for segment-local vision attention with block skipping (``VisionTpuLocalAttention``).

* On CPU, ``forward(local_segment_ids=...)`` with the Pallas op swapped for a torch
  reference: the output equals the per-segment attention of the inputs, padding and
  KV-head expansion work, and bad inputs are rejected.
* On the JAX CPU device: the block map marks every block pair that holds a same-segment
  query-key pair (and only overlapping ones), and the op's JAX body, splash in
  interpret mode, equals the reference for contiguous and interleaved segments and
  for full attention over a padded sequence (real tokens, then a pad tail).
* On the TPU in bf16, the registered op through ``VisionTpuLocalAttention``, eager and
  compiled, against an fp32 CPU reference: tiles of a grid that is not a multiple of
  the tile size, in tile order and in row-major order, with pad tokens.
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

from sglang.srt.utils import is_tpu
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.vision_local import VisionTpuLocalAttention


def _ref_token_attention(q, k, v, seg, scale):
    """Per-segment non-causal attention in fp32. q, k, v: [s, h, d]; seg: [s]."""
    qr, kr, vr = (rearrange(x.float(), "s h d -> h s d") for x in (q, k, v))
    scores = torch.einsum("hqd,hkd->hqk", qr, kr) * scale
    scores = scores.masked_fill(~(seg[:, None] == seg[None, :]), float("-inf"))
    out = torch.einsum("hqk,hkd->hqd", torch.softmax(scores, dim=-1), vr)
    return rearrange(out, "h s d -> s h d")


def _segments(lengths):
    return torch.cat(
        [torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lengths)]
    )


def _tiles(grid_h, grid_w, tile):
    """Row-major tile ids of a grid cut into tile x tile tiles (edge tiles partial)."""
    r = torch.arange(grid_h * grid_w) // grid_w
    c = torch.arange(grid_h * grid_w) % grid_w
    return ((r // tile) * -(-grid_w // tile) + c // tile).to(torch.int32)


class _CpuLocalAttn(VisionTpuLocalAttention):
    """``VisionTpuLocalAttention`` with both Pallas ops swapped for torch references."""

    def __init__(self, head_dim, num_heads, num_kv_heads, scale):
        torch.nn.Module.__init__(self)
        self.head_size = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.flatten_batch = True
        self.scale = scale
        self._flash_op = lambda q, k, v, seg: rearrange(
            _ref_token_attention(
                *(rearrange(x[0], "h s d -> s h d") for x in (q, k, v)), seg[0], scale
            ),
            "s h d -> 1 h s d",
        ).to(q.dtype)
        self.seen_ids = []

        def local_op(q, k, v, ids):
            self.seen_ids.append(ids)
            return _ref_token_attention(q, k, v, ids, scale).to(q.dtype)

        self._local_op = local_op


class TestVisionTpuLocalAttentionCpu(CustomTestCase):
    """``forward(local_segment_ids=...)`` returns per-segment attention of the inputs."""

    def _check(self, seg, h=2, kv_h=None, d=16, seed=0):
        torch.manual_seed(seed)
        kv_h = kv_h or h
        scale = 1.0 / math.sqrt(d)
        s = seg.shape[0]
        q = torch.randn(s, h, d)
        k, v = (torch.randn(s, kv_h, d) for _ in range(2))
        attn = _CpuLocalAttn(d, h, kv_h, scale)
        out = attn.forward(q, k, v, bsz=1, local_segment_ids=seg[None].clone())
        rep = h // kv_h
        ref = _ref_token_attention(
            q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), seg, scale
        )
        torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)
        return attn

    def test_contiguous_segments(self):
        self._check(_segments([128, 100, 28, 256]))

    def test_full_attention_with_a_pad_tail(self):
        # A full-attention layer: one segment for the real tokens, one for the pads.
        self._check(_segments([400, 112]), seed=7)

    def test_interleaved_segments(self):
        self._check(_tiles(32, 32, 16), seed=1)

    def test_pads_to_the_block_size_as_one_more_segment(self):
        attn = self._check(_segments([60, 40]), seed=2)
        (ids,) = attn.seen_ids
        self.assertEqual(ids.shape[0], 128)
        self.assertTrue(bool((ids[100:] == 2).all()))

    def test_kv_heads_expand_to_query_heads(self):
        self._check(_segments([100, 60, 96]), h=4, kv_h=2, seed=3)

    def test_without_local_ids_is_the_segment_id_path(self):
        torch.manual_seed(4)
        seg = _segments([30, 50, 48])
        q, k, v = (torch.randn(128, 2, 16) for _ in range(3))
        attn = _CpuLocalAttn(16, 2, 2, 0.25)
        out = attn.forward(q, k, v, bsz=1, segment_ids=seg[None].clone())
        torch.testing.assert_close(
            out, _ref_token_attention(q, k, v, seg, 0.25), atol=2e-5, rtol=2e-5
        )
        self.assertEqual(attn.seen_ids, [])

    def test_rejects_other_masks_and_batches(self):
        attn = _CpuLocalAttn(16, 2, 2, 0.25)
        q = torch.randn(128, 2, 16)
        ids = torch.zeros(1, 128, dtype=torch.int32)
        with self.assertRaises(ValueError):
            attn.forward(q, q, q, bsz=1, local_segment_ids=ids, segment_ids=ids)
        with self.assertRaises(ValueError):
            attn.forward(q, q, q, bsz=2, local_segment_ids=ids)

    def test_rejects_bad_id_shapes(self):
        attn = _CpuLocalAttn(16, 2, 2, 0.25)
        q = torch.randn(128, 2, 16)
        for bad in (torch.zeros(128, dtype=torch.int32), torch.zeros(1, 64)):
            with self.subTest(shape=tuple(bad.shape)):
                with self.assertRaises(ValueError):
                    attn.forward(q, q, q, bsz=1, local_segment_ids=bad)

    def test_rejects_v_heads_unlike_k(self):
        attn = _CpuLocalAttn(16, 2, 2, 0.25)
        q = torch.randn(128, 2, 16)
        ids = torch.zeros(1, 128, dtype=torch.int32)
        with self.assertRaises(ValueError):
            attn.forward(q, q, torch.randn(128, 1, 16), bsz=1, local_segment_ids=ids)


class TestVisionLocalJax(CustomTestCase):
    """The op's JAX body on the JAX CPU device (exact fp32; splash in interpret mode)."""

    @staticmethod
    def _cpu():
        import jax

        return jax.default_device(jax.devices("cpu")[0])

    def _block_map(self, seg, block_q, block_kv):
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu.attention_pallas import _vision_local_block_map

        with self._cpu():
            mask, nxt = _vision_local_block_map(
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

    def test_block_map_is_sparse_for_contiguous_segments(self):
        mask, _ = self._block_map(_segments([1024] * 16), 2048, 1024)
        # Each 2048-token q block holds two segments, i.e. two of 16 kv blocks.
        self.assertEqual(int((mask > 0).sum()), 8 * 2)

    def _check_body(self, seg, h, d, seed):
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu.attention_pallas import _vision_local_attention

        torch.manual_seed(seed)
        scale = 1.0 / math.sqrt(d)
        q, k, v = (torch.randn(seg.shape[0], h, d) for _ in range(3))
        with self._cpu():
            out = _vision_local_attention(
                *(jnp.asarray(x.numpy()) for x in (q, k, v)),
                jnp.asarray(seg.numpy()),
                sm_scale=scale,
                interpret=True,
            )
        torch.testing.assert_close(
            torch.from_numpy(np.asarray(out)),
            _ref_token_attention(q, k, v, seg, scale),
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
class TestVisionLocalOnTpu(CustomTestCase):
    """The registered Pallas op in bf16, eager and compiled, against fp32 on the CPU."""

    HEADS, HEAD_DIM = 2, 96
    # bf16 inputs and outputs against an fp32 reference of the same bf16 inputs.
    BF16_TOL = 2e-2

    def test_backend_registration(self):
        from sglang.srt.layers.attention.vision import QKV_BACKEND_IMPL
        from sglang_tpu.tpu_platform import _register_attention_backends

        _register_attention_backends()
        self.assertIs(QKV_BACKEND_IMPL["tpu"], VisionTpuLocalAttention)

    def test_local_op_matches_reference_eager_and_compiled(self):
        # 32x32 tiles of a 60x68 grid (full, 32x4, 28x32 and 28x4 tiles): 4,080
        # tokens, padded to 4,096 inside the op.
        raster = _tiles(60, 68, 32)
        tile_order = torch.argsort(raster.to(torch.int64), stable=True)
        attn = VisionTpuLocalAttention(self.HEAD_DIM, self.HEADS, self.HEADS)

        def run(q, k, v, ids):
            return attn(q, k, v, 1, local_segment_ids=ids)

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
