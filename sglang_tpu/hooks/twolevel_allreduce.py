# SPDX-License-Identifier: Apache-2.0
"""Two-level Pallas all-reduce for single-token TP decode tensors (opt-in).

all_reduce
  Why: XLA's all-reduce of a [1, H] bf16 decode activation over a multi-host
       TP group is latency bound (tpu7x 2x2x4, TP32: 21 us at H=7168); a
       host-then-cross-host one-shot Pallas exchange takes 13 us.
  How: AROUND GroupCoordinator.all_reduce; the TP group's eligible tensors go
       to sglang_tpu.twolevel_allreduce when SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE
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

row_parallel_linear_forward
  Why: in a Kimi-K3 decoder layer with attention residuals, o_proj's TP
       all-reduce is followed directly by the MLP-side attention-residual
       aggregation; one Pallas kernel doing both (all_reduce_attn_res) saves
       a kernel and the aggregation's XLA fusions per layer.
  How: AROUND RowParallelLinear.forward. With the two-level all-reduce on
       and SGLANG_TPU_TWOLEVEL_ALLREDUCE_FUSE_ATTN_RES (default on), an o_proj tagged at
       post-load (tag_attn_res_layer) skips its all-reduce on single-token
       bf16 calls over the TP group and marks the reduce pending; the
       attention-residual hook then runs the fused all-reduce + aggregation.
       Every other call runs the original.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1444 (platform all-reduce
       provider); the deferral itself has no seam issue yet.
  Remove when: core lets a platform fuse a row-parallel layer's all-reduce
       into the op that consumes it.
"""

from sglang_tpu.twolevel_allreduce import all_reduce_around as all_reduce
from sglang_tpu.twolevel_allreduce import (
    merged_moe_all_reduce_around as merged_moe_all_reduce,
)
from sglang_tpu.twolevel_allreduce import (
    row_parallel_linear_forward_around as row_parallel_linear_forward,
)

__all__ = ["all_reduce", "merged_moe_all_reduce", "row_parallel_linear_forward"]
