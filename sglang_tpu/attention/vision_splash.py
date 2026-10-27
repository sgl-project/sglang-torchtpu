"""TPU vision attention that skips the query-key blocks segments cannot reach.

``VisionTpuAttention`` masks segment ids with the Pallas flash kernel, which still
computes every query-key block of the whole sequence: a window- or tile-local layer
(each token attends only within its own window or tile) costs as much as full
attention, and above ~16k keys the kernel's single KV step outgrows a 24 MiB scoped
VMEM. ``VisionTpuSplashAttention`` runs the same ``segment_ids`` / ``cu_seqlens``
through splash attention with a block map that skips every query-key block whose
segments cannot meet (``_get_vision_splash_op``); everything else is the parent's
forward. The result is exact for any ids. It is fast only when the ids are
non-decreasing in token order, which ``cu_seqlens`` always is; callers with window-
or tile-local layers order the tokens segment by segment, number the segments in
that order, and undo the order after the encoder.

Only one sequence (``bsz == 1``, packed segments) runs on splash. A dense batch of
rows (``bsz > 1``) keeps the flash path: each row would pad to a whole 1,024-token
splash block where flash pads to 128, and with the rows laid end to end the splash
grid grows with the square of the batch, so short rows ran up to ~3.6x slower than
flash at batch 16 (packing the same images as one sequence by ``cu_seqlens`` pads
once and does not pay this).

``_register_attention_backends`` installs it as the ``"tpu"`` vision attention
backend.
"""

from __future__ import annotations

from sglang.srt.layers.attention.vision import VisionTpuAttention


class VisionTpuSplashAttention(VisionTpuAttention):
    """``VisionTpuAttention`` with block-skipping splash for one sequence (``bsz == 1``)."""

    # The parent pads each sequence to a multiple of this, with the pads as one more
    # segment at the end (so the ids stay non-decreasing and all-pad blocks are
    # skipped). Padding only to 128 leaves odd multiples of 128 (e.g. 3,200 = 25 x
    # 128) on 128-token splash blocks, which is up to ~16x slower; whole 1,024 blocks
    # were the fastest or within ~13% of it at every single-sequence length measured,
    # including sequences shorter than 1,024 (dense batches keep the flash path).
    _BLOCK_SIZE = 1024

    def __init__(self, head_dim, num_heads, num_kv_heads, **kwargs):
        super().__init__(head_dim, num_heads, num_kv_heads, **kwargs)
        # Imported here, not at module level: _register_attention_backends imports
        # this module on hosts without JAX (CPU CI), and the class is only built on TPU.
        from sglang_tpu.attention_pallas import _get_vision_splash_op

        # Same arguments as the flash op the parent's forward calls.
        self._flash_op = _get_vision_splash_op(float(self.scale))
        # Dense batches (bsz > 1) run here: the parent unchanged (flash, 128 padding).
        self._batched_attention = VisionTpuAttention(
            head_dim, num_heads, num_kv_heads, **kwargs
        )

    def forward(self, q, k, v, bsz, *args, **kwargs):
        if bsz > 1:
            return self._batched_attention(q, k, v, bsz, *args, **kwargs)
        return super().forward(q, k, v, bsz, *args, **kwargs)
