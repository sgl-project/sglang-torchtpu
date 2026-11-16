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

"""Batched ragged paged attention (RPA) Pallas kernel.

Select this kernel with `--attention-backend pallas_tpu_rpa_batched`. It reads
the token-major KV cache by default; set SGLANG_TPU_USE_SEQ_ALONG_LANE_KV=1 for
the seq-along-lane layout, which also switches the KV pool that allocates it.

The package is a port of google3's
`experimental/users/alynie/batched_rpa/kernel`, synced at google3 cl/1018717699,
and keeps that tree's file split and names (`pipeline`, `scheduler`,
`scheduler_cp`, ...) so later google3 changes can be applied as diffs. SGLang
behaviour that google3 does not have is layered on top; `wrapper` documents the
public contract.
"""
