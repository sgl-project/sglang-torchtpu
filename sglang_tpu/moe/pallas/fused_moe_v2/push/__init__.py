# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/experimental/push_fused_moe/__init__.py
# Revision: ef2599d3d4342f41d2ad223b266e126299862b53
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Push dispatch for the fused EP MoE v2: `fused_ep_moe_v2(dispatch="push")`.

The fused EP MoE v2 kernel with the activation all-gather replaced by
point-to-point pushes of each shard's routed rows: a SparseCore kernel pushes
the first visit positions before the TensorCore kernel starts, and the
TensorCore kernel pushes the rest with in-kernel remote DMAs. See layer.py.
Callers go through `fused_ep_moe_v2`; `fused_ep_moe_push` is exported for the
package's own tests.
"""

from sglang_tpu.moe.pallas.fused_moe_v2.push.layer import fused_ep_moe_push

# The keyword arguments `fused_ep_moe_push` takes beyond v2's operands, which
# `fused_ep_moe_v2(dispatch="push", push_options=...)` forwards. Everything
# else in its signature is a v2 parameter forwarded by name.
PUSH_OPTIONS = (
    "sc_prepush",
    "tc_exchange",
    "height_rungs_override",
    "tail_barrier",
    "onehot_slots",
    "hbm_staging",
    "dispatch_ahead",
    "matmul_scale_repeat",
    "exchange_row_gather",
    "bf16_sort_counts",
    "slim_prepush_barrier",
    "fused_dispatch_meta",
    "table_kernel",
    "exchange_tables",
    "kernel_plan_tables",
    "light_prepush",
    "single_barrier",
    "early_scale_scatter",
)

__all__ = ["PUSH_OPTIONS", "fused_ep_moe_push"]
