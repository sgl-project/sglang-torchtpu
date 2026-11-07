# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""MoE-TP decode MoE layer in one Pallas call, all-reduce included (TPU7x).

For MoE layers served with tensor-parallel experts (`moe_ep_size == 1`: every
rank holds all routed experts, each with a slice of the intermediate dim, and
its slice of the shared expert) at decode token counts up to 64. One Pallas
call per rank streams the experts any token picked, runs this rank's shared
expert slice, combines `shared + routed_scaling_factor * routed` and all-reduces
the result in-kernel, so the decoder layer skips its FFN all-reduce.

It is the MoE-TP counterpart of `fused_moe_decode` (which serves
`moe_ep_size == tp_size`) and is selected by the same
`--moe-runner-backend pallas_tpu_fused_decode`.

Modules:
  - `kernel`: the Pallas kernel (`moe_tp_decode`, `can_use_moe_tp_decode`).
  - `all_reduce`: the in-kernel reduce-scatter + all-gather it ends with.
  - `routing`: SGLang top-k output to the kernel's routing table.
  - `op`: TorchTPU custom op, layer preparation and dispatch.
"""
