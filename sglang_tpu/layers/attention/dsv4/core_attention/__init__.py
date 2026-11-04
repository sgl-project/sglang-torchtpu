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
#
# Adapted for SGLang-TorchTPU from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: src/vllm_torchtpu/kernels/deepseek_v4/core_attention/__init__.py

from sglang_tpu.layers.attention.dsv4.core_attention.mla import (
    mla_ragged_paged_attention,
)
from sglang_tpu.layers.attention.dsv4.core_attention.mla_swa import (
    mla_sliding_window_ragged_paged_attention,
)
from sglang_tpu.layers.attention.dsv4.core_attention.sparse_mla import (
    sparse_ragged_paged_attention,
)

__all__ = [
    "mla_ragged_paged_attention",
    "mla_sliding_window_ragged_paged_attention",
    "sparse_ragged_paged_attention",
]
