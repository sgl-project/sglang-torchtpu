# SPDX-License-Identifier: Apache-2.0
"""Lane-aligned TP all-gather of vocab-sharded logits.

gather_logits_lane_aligned
  Why: libtpu miscompiles the TP all-gather of ``[rows, vocab_shard]`` logits
       when the shard width is not a multiple of the 128-lane tile (Qwen3 at
       TP4: 151936 / 4 = 37984). The compiled extend program then returns
       garbage in the first columns of some shards, and the garbage logit is
       sampled as the first output token with logprob 0.0 or NaN.
  How: AROUND MultimemAllGatherer.__call__, the logits processor's TP gatherer
       (on TPU it always takes its tensor_model_parallel_all_gather fallback):
       pad the shard to a lane multiple, gather, drop the padding. An aligned
       shard passes through unchanged.
  Upstream PR or issue: none (libtpu miscompile; the responsible pass is not
       identified yet).
  Remove when: libtpu gathers an unaligned shard correctly in the compiled
       extend program.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F

from sglang.srt.utils.common import ceil_align

# Minor-dim tile width of TPU vector registers.
_LANE_WIDTH = 128


def gather_lane_aligned_shards(
    logits: torch.Tensor, gather: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """``[rows, vocab_shard] -> [rows, tp_size * vocab_shard]`` through ``gather``,
    which only ever sees a lane-aligned shard."""
    vocab_shard = logits.shape[-1]
    padded_shard = ceil_align(vocab_shard, _LANE_WIDTH)
    if padded_shard == vocab_shard:
        return gather(logits)
    gathered = gather(F.pad(logits, (0, padded_shard - vocab_shard)))
    rows = logits.shape[0]
    tp_size = gathered.shape[-1] // padded_shard
    return gathered.reshape(rows, tp_size, padded_shard)[:, :, :vocab_shard].reshape(
        rows, tp_size * vocab_shard
    )


def gather_logits_lane_aligned(
    original, gatherer, logits: torch.Tensor
) -> torch.Tensor:
    """AROUND hook for ``MultimemAllGatherer.__call__``."""
    return gather_lane_aligned_shards(logits, gather=partial(original, gatherer))
