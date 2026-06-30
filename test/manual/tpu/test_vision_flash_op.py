"""CPU-runnable unit tests for the TPU vision-encoder flash-attention wrapper.

These cover the pure-PyTorch logic in ``VisionTpuAttention.forward``
(``layers/attention/vision.py``): rearrange to ``[b, h, s, d]``, padding the
sequence up to ``_BLOCK_SIZE``, extending ``segment_ids`` with a dedicated
padding segment so pad tokens never attend to real ones, and slicing the output
back to the real length.

The actual pallas ``flash_attention`` kernel numerics (JAX + TPU) are out of
scope here and covered by the kernel tests; we substitute a pure-torch
segment-masked attention reference for ``_flash_op`` so this runs on CPU. The
contract verified is: the wrapper, with padding + a pad segment, returns exactly
the block-diagonal (per-segment) attention of the *unpadded* inputs.
"""

import math
import unittest

import torch
from einops import rearrange

from sglang.srt.layers.attention.vision import VisionTpuAttention
from sglang.test.test_utils import CustomTestCase


def _ref_segment_attention(q, k, v, seg_ids, scale):
    """Pure-torch non-causal attention with block-diagonal (per-segment) mask.

    q, k, v: [b, h, s, d]; seg_ids: [b, s]. A query attends to a key iff they
    share a segment id. Mirrors the pallas SegmentIds semantics.
    """
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
    same = seg_ids[:, None, :, None] == seg_ids[:, None, None, :]  # [b,1,s,s]
    scores = scores.masked_fill(~same, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", attn, v.float()).to(q.dtype)


class _CpuVisionAttn(VisionTpuAttention):
    """``VisionTpuAttention`` with the pallas op swapped for a CPU reference.

    The production ``__init__`` builds the pallas op (imports JAX + torch_tpu);
    we bypass it and inject ``_flash_op`` so ``forward`` runs on CPU.
    """

    def __init__(self, head_dim, num_heads, scale):
        torch.nn.Module.__init__(self)
        self.head_size = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.flatten_batch = True
        self.scale = scale
        self._flash_op = lambda q, k, v, seg: _ref_segment_attention(
            q, k, v, seg, scale
        )


class TestVisionFlashOpWrapper(CustomTestCase):
    def _run(self, seq_lens, h=2, d=16, seed=0):
        torch.manual_seed(seed)
        s = sum(seq_lens)
        scale = 1.0 / math.sqrt(d)
        q = torch.randn(s, h, d)
        k = torch.randn(s, h, d)
        v = torch.randn(s, h, d)
        seg = torch.cat(
            [torch.full((L,), i, dtype=torch.int32) for i, L in enumerate(seq_lens)]
        ).unsqueeze(0)

        attn = _CpuVisionAttn(d, h, scale)
        out = attn.forward(q, k, v, bsz=1, segment_ids=seg.clone())

        qr, kr, vr = (rearrange(x, "(b s) h d -> b h s d", b=1) for x in (q, k, v))
        ref = rearrange(
            _ref_segment_attention(qr, kr, vr, seg, scale), "b h s d -> (b s) h d"
        )
        return out, ref

    def test_single_segment_with_padding(self):
        # 100 -> padded to 128; the 28 pad tokens form their own segment.
        out, ref = self._run([100])
        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)

    def test_multi_segment_block_diagonal(self):
        # Three images packed in one sequence (block-diagonal masking).
        out, ref = self._run([30, 50, 20])
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)

    def test_exact_block_multiple_no_padding(self):
        # 128 is a multiple of _BLOCK_SIZE -> no padding branch.
        out, ref = self._run([128])
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)

    def test_pad_segment_does_not_leak(self):
        # Two unequal segments; correctness here means pad tokens (added to a
        # fresh segment) did not bleed into real tokens' softmax.
        out, ref = self._run([10, 90], h=4, d=32, seed=3)
        torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
