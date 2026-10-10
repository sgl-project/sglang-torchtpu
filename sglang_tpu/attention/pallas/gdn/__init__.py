# Copyright 2026 The SGLang Team. All rights reserved.
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
"""TPU Pallas kernels for Gated DeltaNet (GDN) linear attention.

Fused Conv1D + gated-delta-rule kernel for hybrid models (Qwen3.5 / Qwen3-Next).
The single public entry point ``fused_conv1d_gdn`` runs conv1d and the gated
delta rule in one Pallas kernel, handling ragged (mixed prefill + decode)
batches via a per-tile metadata plan.

Adapted from vllm-project/tpu-inference ``tpu_inference.kernels.gdn.v3``
(Apache-2.0). Pure JAX/Pallas; call it from torch through
``torch_tpu._internal.pallas``.
"""

from sglang_tpu.attention.pallas.gdn.wrapper import fused_conv1d_gdn

__all__ = ["fused_conv1d_gdn"]
