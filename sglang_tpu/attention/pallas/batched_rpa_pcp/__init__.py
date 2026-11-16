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

"""EXPERIMENTAL: batched RPA kernel package for ring-mode prefill context parallelism.

Used only by ring-mode prefill context parallelism (`SGLANG_TPU_PCP_MODE=ring`):
the fused ring attention prefill op (`attention_pallas_rpa_pcp`, entry point
`wrapper.ragged_paged_attention_pcp_fused`) and its host-side schedule check
(`wrapper.check_pcp_fused_schedule`, via `pp_attention_capacity`). Nothing else
imports it; every other path, including ring-mode decode and short extends
(the DCP `STRIPED` path), uses the production kernel in
`sglang_tpu.attention.pallas.batched_rpa`.

This is a self-contained copy of `batched_rpa` synced to google3's
`experimental/users/alynie/batched_rpa/kernel` at cl/1018717699 (the tree's
file split and names, `pipeline`, `scheduler`, `scheduler_cp`, ..., are kept),
with SGLang behaviour re-applied, plus the fused PCP ring kernel ported from
google3 cl/985434096 (`scheduler_cp` ring schedulers, a step-time model for the
cache-phase tile, the new-token start-mask skip and a memoized host schedule
check). Keeping it separate lets the ring kernel evolve without changing the
production kernel; once the production package is synced to google3 the two
can be merged back. Do not import it from non-ring code.
"""
