"""TPU vision attention that skips the query-key blocks segments cannot reach.

``VisionTpuAttention`` masks segment ids with the Pallas flash kernel, which still
computes every query-key block of the whole sequence: a window- or tile-local layer
(each token attends only within its own window or tile) costs as much as full
attention, and above ~16k keys the kernel's single KV step outgrows a 24 MiB scoped
VMEM. ``VisionTpuSplashAttention`` runs the same ``segment_ids`` / ``cu_seqlens``
through splash attention with a block map that skips every query-key block whose
segments cannot meet (``_get_vision_splash_op``); everything else is the parent's
forward. The result is exact for any ids. It is fast only when the ids are
non-decreasing in token order, which ``cu_seqlens`` and the real/pad ids of a
padded batch always are; callers with window- or tile-local layers order the tokens
segment by segment, number the segments in that order, and undo the order after the
encoder.

``_register_attention_backends`` installs it as the ``"tpu"`` vision attention
backend.
"""

from __future__ import annotations

from sglang.srt.layers.attention.vision import VisionTpuAttention


class VisionTpuSplashAttention(VisionTpuAttention):
    """``VisionTpuAttention`` with the flash kernel swapped for block-skipping splash."""

    # The parent pads each sequence to a multiple of this, with the pads as one more
    # segment at the end (so the ids stay non-decreasing and all-pad blocks are
    # skipped). Padding only to 128 leaves odd multiples of 128 (e.g. 3,200 = 25 x
    # 128) on 128-token splash blocks, which is up to ~16x slower; whole 1,024 blocks
    # were the fastest or within ~13% of it at every length measured, including
    # sequences shorter than 1,024.
    _BLOCK_SIZE = 1024

    def __init__(self, head_dim, num_heads, num_kv_heads, **kwargs):
        super().__init__(head_dim, num_heads, num_kv_heads, **kwargs)
        # Imported here, not at module level: _register_attention_backends imports
        # this module on hosts without JAX (CPU CI), and the class is only built on TPU.
        from sglang_tpu.attention_pallas import _get_vision_splash_op

        # Same arguments as the flash op the parent's forward calls.
        self._flash_op = _get_vision_splash_op(float(self.scale))
