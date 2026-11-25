# Adapted from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/experimental/push_fused_moe/__init__.py
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

from sglang_tpu.moe.pallas.fused_moe_v2.host import WeightFormat
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import layer as _layer
from sglang_tpu.moe.pallas.fused_moe_v2.push_fused_moe import layer_k512 as _layer_k512

# Push options only the K512 build (layer_k512) takes.
K512_ONLY_OPTIONS = (
    "split_exchange",
    "onehot_scale_slab",
    "pair_kernel",
    "wide_pair_kernel",
    "decode_combine",
)


def fused_ep_moe_push(*args, weight_format=WeightFormat.FP8, **kwargs):
    """Push dispatch. K32Z builds run layer.py (its K32Z schedule, program
    unchanged); every other format runs layer_k512.py (decode combine, pair
    kernels, one-hot scale slab)."""
    if weight_format == WeightFormat.K32Z:
        given = sorted(k for k in K512_ONLY_OPTIONS if k in kwargs)
        if given:
            raise ValueError(f"push options {given} are not available for K32Z")
        return _layer.fused_ep_moe_push(*args, weight_format=weight_format, **kwargs)
    return _layer_k512.fused_ep_moe_push(*args, weight_format=weight_format, **kwargs)


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
    "split_exchange",
    "onehot_scale_slab",
    "pair_kernel",
    "wide_pair_kernel",
    "decode_combine",
    "fuse_metadata",
)
# K32Z builds on the exchange-table path also take the per-bucket schedule in
# layer.K32Z_PUSH_RECIPES; it is chosen by the layer, not passed in.

__all__ = ["K512_ONLY_OPTIONS", "PUSH_OPTIONS", "fused_ep_moe_push"]
