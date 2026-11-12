# SPDX-License-Identifier: Apache-2.0
"""Direct Pallas RDMA all-gather in Dynamo-traced TPU graphs.

GroupCoordinator.all_gather_into_tensor
  Why: under Data Parallelism with DP attention on TPU v7x,
       GroupCoordinator.all_gather_into_tensor lowers through XLA
       all-gather or zero-padded all-reduce fallbacks, consuming ~108 ms per
       prefill step on 8 chips. Direct Pallas RDMA remote DMA writes each
       rank's shard directly into every peer's HBM output buffer in one hop.
  How: AROUND GroupCoordinator.all_gather_into_tensor checks
       torch.compiler.is_dynamo_compiling(), world_size above 1, and
       can_use_rdma_all_gather(group, input), then invokes
       rdma_all_gather_into_tensor with multi-chunk slicing up to 128 MiB.
       Eager calls, direct-tracer graphs, and ineligible shapes keep original.
  Upstream PR or issue: direct Pallas RDMA all-gather optimization for DP
       attention token gathers on Cloud TPU v7x.
  Remove when: XLA's compiled reg_all_gather_into_tensor matches direct
       Pallas RDMA latency and cross-chip rank ordering on multi-chip TPU
       slices.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def all_gather_into_tensor(
    original: Callable[..., Any],
    group: Any,
    output: torch.Tensor,
    input: torch.Tensor,  # noqa: A002 - mirrors GroupCoordinator's signature
) -> None:
    """AROUND hook for ``GroupCoordinator.all_gather_into_tensor``."""
    if not torch.compiler.is_dynamo_compiling() or group.world_size <= 1:
        return original(group, output, input)
    from sglang_tpu.rdma_all_gather import (
        can_use_rdma_all_gather,
        rdma_all_gather_into_tensor,
    )

    if can_use_rdma_all_gather(group, input):
        rdma_all_gather_into_tensor(output, input, group.world_size)
        return None
    return original(group, output, input)
