# SPDX-License-Identifier: Apache-2.0
"""Muse Spark prefill program: every decoder layer of one extend chunk as one jitted shard_map.

It reads the decode megakernel's weights (fp8 K-major dense copies, bf16 router, smalls) and
the sglang-owned experts and KV pools. Per layer:

    batched RPA (writes the chunk's K/V)
      -> attn_out.py   o proj + reduce-scatter / all-gather, residual, router, expert input
      -> fused-RS expert-TP MoE (token-sharded output)
      -> post_moe.py   output gather, post-expert proj, residual, next layer's q / kv / gate

ops.MuseMegakernel registers it as the sglang_tpu_pallas::muse_prefill op
(SGLANG_TPU_MUSE_MK_PREFILL).
"""

from sglang_tpu.muse_megakernel.prefill.config import (  # noqa: F401
    PREFILL_TP,
    ROW_ALIGN,
    PFConfig,
)
from sglang_tpu.muse_megakernel.prefill.program import (  # noqa: F401
    N_STEP_ARGS,
    build_forward,
)
