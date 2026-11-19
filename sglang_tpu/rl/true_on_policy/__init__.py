# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant kernels for true on-policy RL: a row's output bytes do not depend on the call's row count.

XLA picks its dot tiling and reduction order per shape, so an XLA matmul, log-softmax or RMSNorm returns
different bytes for the same row at different batch sizes. These Pallas kernels fix the tile of every reduction and
walk it in one order, so a decode step, a prefill chunk and the trainer's packed forward give the same
bytes. The trainer calls the same torch ops (exported here), which is what makes its log-probs equal the
engine's.

The contract is the K step of the matmul (``matmul_kernel._MM_BK``) and the column tile of the log-prob
(``logprob_kernel._LP_COLS``): changing either changes the bytes. Row and column block sizes do not.

The trainer differentiates these ops. The forward is the batch-invariant program; the backward only has to be
the gradient of the same math, so it is plain torch: fp32 for the norm and the log-prob, the gradient's own
dtype (bf16, as for any bf16 linear) for the matmul.
"""

from sglang_tpu.rl.true_on_policy.attention_kernel import (
    PAGE_SIZE,
    packed_causal_attention,
    pin_kv_block,
    refuse_rpa_rounding_flags,
)
from sglang_tpu.rl.true_on_policy.logprob_kernel import logprob
from sglang_tpu.rl.true_on_policy.matmul_kernel import matmul
from sglang_tpu.rl.true_on_policy.norm_kernel import rms_norm

__all__ = [
    "PAGE_SIZE",
    "logprob",
    "matmul",
    "packed_causal_attention",
    "pin_kv_block",
    "refuse_rpa_rounding_flags",
    "rms_norm",
]
