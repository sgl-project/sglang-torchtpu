"""Segment-local vision attention that skips the blocks segments cannot reach.

ViTs with window- or tile-local layers (each token attends only within its own
window or tile) run those layers through ``VisionTpuAttention`` with
``segment_ids``. The Pallas flash kernel masks by segment but still computes
every query-key block of the whole sequence, so a local layer costs as much as
full attention. ``VisionTpuLocalAttention`` additionally takes
``local_segment_ids=``: the same per-token segment ids, run through splash
attention with a block map that skips every query-key block whose segments
cannot meet (``_get_vision_local_op``). The result is exact for any ids; it is
fast only when the ids are non-decreasing in token order, so callers should order
the tokens segment by segment, number the segments in that order, and undo the
order after the encoder. Full-attention
layers over a padded sequence (one segment for the real tokens, one for the pad
tail) can use it too: it then runs dense over the real tokens, and unlike the
segment-ID op it fits a 24 MiB scoped VMEM at long sequences.

``_register_attention_backends`` installs it as the ``"tpu"`` vision attention
backend; without ``local_segment_ids=`` it is ``VisionTpuAttention`` unchanged.

Callers that pass ``local_segment_ids=`` must call this class directly.
``VisionAttention`` does not forward it, and a plain ``VisionTpuAttention`` (the
default of other vision runners, or the ``"tpu"`` entry before registration)
accepts it through ``**kwargs`` and ignores it, running full attention without
an error.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.vision import VisionTpuAttention
from sglang_tpu.attention_pallas import _get_vision_local_op

# Pad the local path to whole 1,024-token blocks. Padding only to 128 leaves odd
# multiples of 128 (e.g. 3,200 = 25 x 128) on 128-token splash blocks, which is up to
# ~16x slower; whole 1,024 blocks were the fastest or within ~13% of it at every
# length measured, including sequences shorter than 1,024.
_PAD_MULTIPLE = 1024


class VisionTpuLocalAttention(VisionTpuAttention):
    """``VisionTpuAttention`` that also runs segment-local attention with block skipping.

    Call it directly with ``local_segment_ids=``; see the module docstring for the
    paths that would drop it silently.
    """

    def __init__(self, head_dim, num_heads, num_kv_heads, **kwargs):
        super().__init__(head_dim, num_heads, num_kv_heads, **kwargs)
        self._local_op = _get_vision_local_op(float(self.scale))

    def forward(
        self,
        q,
        k,
        v,
        bsz,
        cu_seqlens=None,
        attention_mask=None,
        segment_ids=None,
        local_segment_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        """``local_segment_ids``: [1, seq_len] int32, the segment of each token.

        Exact for any ids; blocks are skipped only when the ids are non-decreasing.
        """
        if local_segment_ids is None:
            return super().forward(
                q,
                k,
                v,
                bsz,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                **kwargs,
            )
        if (
            attention_mask is not None
            or segment_ids is not None
            or cu_seqlens is not None
            or bsz != 1
        ):
            raise ValueError(
                "TPU vision local attention takes no attention_mask, segment_ids "
                "or cu_seqlens, and needs bsz == 1"
            )
        s = q.shape[0]
        if tuple(local_segment_ids.shape) != (1, s):
            raise ValueError(
                "TPU vision local attention needs local_segment_ids [1, seq_len]; "
                f"got {tuple(local_segment_ids.shape)} for seq_len {s}"
            )
        # Same KV-head checks and expansion as the segment-id path.
        if k.shape[1] != v.shape[1] or q.shape[1] % k.shape[1] != 0:
            raise ValueError(
                "TPU vision attention requires matching KV heads and an "
                "integer query-to-KV head ratio"
            )
        if k.shape[1] != q.shape[1]:
            repeats = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        ids = local_segment_ids.to(device=q.device, dtype=torch.int32)[0]
        # Pads form one more segment at the end: the ids stay non-decreasing, and
        # blocks that are all padding skip the real-token blocks.
        pad = (-s) % _PAD_MULTIPLE
        if pad > 0:
            q, k, v = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
            ids = torch.cat([ids, (ids.max() + 1).expand(pad)])
        return self._local_op(q, k, v, ids)[:s]
