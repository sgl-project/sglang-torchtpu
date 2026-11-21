# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant kernels for true on-policy RL: a row's bytes do not depend on the call's row count.

The contract is the matmul's K step (``matmul_kernel._MM_BK``) and the log-prob's column tile
(``logprob_kernel._LP_COLS``); row and column block sizes do not change the bytes.
"""

from sglang_tpu.rl.true_on_policy.attention_kernel import (
    PAGE_SIZE,
    packed_causal_attention,
    pin_kv_block,
    refuse_rpa_rounding_flags,
)
from sglang_tpu.rl.true_on_policy.gated_delta_net_kernel import (
    GDN_PREFILL_TILE,
    packed_gated_delta_net,
)
from sglang_tpu.rl.true_on_policy.logprob_kernel import logprob
from sglang_tpu.rl.true_on_policy.matmul_kernel import matmul
from sglang_tpu.rl.true_on_policy.moe_kernel import (
    moe_experts,
    refuse_moe_rounding_flags,
)
from sglang_tpu.rl.true_on_policy.norm_kernel import gated_rms_norm, rms_norm

__all__ = [
    "GDN_PREFILL_TILE",
    "PAGE_SIZE",
    "gated_rms_norm",
    "logprob",
    "matmul",
    "moe_experts",
    "packed_causal_attention",
    "packed_gated_delta_net",
    "pin_kv_block",
    "refuse_moe_rounding_flags",
    "refuse_rpa_rounding_flags",
    "rms_norm",
]
