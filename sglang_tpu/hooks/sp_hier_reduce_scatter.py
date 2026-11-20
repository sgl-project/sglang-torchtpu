# SPDX-License-Identifier: Apache-2.0
"""Hierarchical Pallas reduce-scatter for sequence-parallel stages on tpu7x.

sp_reduce_scatter
  Why: every stage of an input-scattered (sequence-parallel) batch ends with a
       [T, H] reduce-scatter over TP; on one tpu7x 2x2x1 host the row-streaming
       hierarchical kernel (kernels/hier_all_reduce.py) sums each chip's two
       TensorCores before crossing ICI, where XLA runs one flat collective.
  How: AROUND layer_boundary.prepare.sp_reduce_scatter, when
       SGLANG_TPU_SP_HIER_REDUCE_SCATTER armed the kernel at startup
       (sglang_tpu/sp_hier_reduce_scatter.py) and the shape fits it.
  Upstream PR or issue: https://github.com/sgl-project/sglang-torchtpu/issues/2335
  Remove when: never; this is the TPU answer for that seam.
"""

from __future__ import annotations

import torch

from sglang_tpu import sp_hier_reduce_scatter as _hier


def sp_reduce_scatter(original, shard: torch.Tensor, hidden_states: torch.Tensor):
    if not _hier.eligible(hidden_states):
        return original(shard, hidden_states)
    shard.copy_(_hier.hier_reduce_scatter(hidden_states))
