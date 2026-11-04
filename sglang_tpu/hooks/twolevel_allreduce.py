# SPDX-License-Identifier: Apache-2.0
"""Two-level Pallas all-reduce for single-token TP decode tensors (opt-in).

all_reduce
  Why: XLA's all-reduce of a [1, H] bf16 decode activation over a multi-host
       TP group is latency bound (tpu7x 2x2x4, TP32: 21 us at H=7168); a
       host-then-cross-host one-shot Pallas exchange takes 13 us.
  How: AROUND GroupCoordinator.all_reduce; the TP group's eligible tensors go
       to sglang_tpu.distributed.twolevel_allreduce when SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE
       registered the op (every rank on, a measured topology), everything else
       calls the original.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1444 (platform all-reduce
       provider).
  Remove when: GroupCoordinator takes a platform all-reduce provider, per #1444.

merged_moe_all_reduce
  Why: Kimi-K3's latent MoE reduces the routed latent and the shared-expert
       output; with SGLANG_K3_MERGE_MOE_ALLREDUCE core calls one function for
       both, which the TPU op serves as one packed exchange (no concat/split).
  How: AROUND sglang.srt.models.kimi_k3.merged_moe_all_reduce.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1444 (coalesced
       all-reduce).
  Remove when: core has a coalesced TP all-reduce routed through that
       provider, per #1444.
"""

from sglang_tpu.distributed.twolevel_allreduce import all_reduce_around as all_reduce
from sglang_tpu.distributed.twolevel_allreduce import (
    merged_moe_all_reduce_around as merged_moe_all_reduce,
)

__all__ = ["all_reduce", "merged_moe_all_reduce"]
