"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 128 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_agx/deepseek_v41_flash_dspark_c128.py

deepseek_v41_flash_dspark_agentx.py with max_running_requests 384 (48 per rank).

Why 48 per rank: session-affinity routing makes ranks uneven. At c32 the
busiest rank peaked at 2.75x the average (11 vs 4; canonical c32
j-3e6d0e27-2566-49c7-aca4). At c128 (16 per rank on average) that is ~44; the
base cap of 16 would queue two thirds of the sessions.

Capacity, computed (the boot log and smoke run check it on hardware):
- KV pool: ~20.2M tokens per rank at mem_fraction_static 0.83 and 16 requests
  per rank (canonical c32 j-3e6d0e27-2566-49c7-aca4). The request-scaled SWA
  windows, draft ring and state come out of the same budget, ~640 MiB more per
  rank at 48 requests, so the pool shrinks by a few percent. 48 requests x
  ~140k average context is 6.7M (33%).
- Compile buckets: tpu_compile_bs [8, 16, 32, 64] covers 48 per rank; no new
  compile shapes.
- HBM: 14.7 GB was free after precompile at 16 per rank, with the bs 64
  programs compiled. Headroom at 48 per rank is not measured yet.
"""

from benchmark.tpu.recipe.configs.deepseek_v41_flash_agx.deepseek_v41_flash_dspark_agentx import (
    with_max_running_requests,
)

CONFIG = with_max_running_requests(384)
# Pin each rank's scheduler process to its own CPUs (SGLANG_SET_CPU_AFFINITY): the
# MLP-sync wait across DP ranks halves, per-op costs unchanged. Paired per-request
# interactivity: c32 +2.9% (j-fff55c92 vs j-b29f6ac4), c64 canonical +2.3%
# (j-10b94a15-88dd-40a2-af97 vs j-b598a23c-9c0b-4fef-87d2); neutral at c4
# (j-6643e364-2d36-454b-ae78 vs j-eecd8be6-a630-4b97-879e), so off at c4 (c16: see its recipe).
CONFIG["runtime_env"]["SGLANG_SET_CPU_AFFINITY"] = "1"
# 8 decode rounds after each prefill step (the DSpark base runs 4). AgentX
# canonical c128 on the fused candidate, only this differs
# (j-6c97db0e-00b8-4152-8d93 vs j-9579cdc2-605b-4821-8906): p90 interactivity
# 37.1 -> 50.6, paired p90 ratio 1.546 (95% CI 1.530-1.560; arm export partial
# after evictions), tok/s/chip 95,566 -> 89,073 (-6.8%).
# Raised to 12 by the product rule (tok/s x p90 interactivity): AgentX c128
# real-draft canonical, pdi 12 (j-99bf5b62-c34e-4eb3-a4cb) vs 8
# (j-e728fd7a-11ff-4707-a2a4): paired p90 ratio 1.222 (95% CI 1.210-1.234),
# tok/s/chip 88,537 -> 75,271; product 1.039.
CONFIG["server_args"]["prefill_decode_interval"] = 12
# Fused attention output projection (wo_a kept in VMEM, wo_b streamed; wo_a
# requantized to per-channel FP8), all layers. AgentX canonical c128, only this
# differs (j-461ef6c2-2b2f-49ea-bc9d vs j-0ec4825d-5b1a-4893-bb4b): paired p90
# interactivity ratio 1.025 (95% CI 1.021-1.030), tok/s/chip +2.1%, KV pool
# +4.5%. GSM8K-1319 0.911/0.908 vs 0.910/0.914 off. Off at c4 (slower there).
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_FUSED_OPROJ"] = "1"
# Visit-balanced fused-EP expert layout (packaged JSON, gate routing). AgentX c128
# real-draft canonical, only this differs (j-2b7c9cdf vs j-e728fd7a-11ff-4707-a2a4):
# paired p90 interactivity ratio 1.021 (95% CI 1.011-1.031), tok/s/chip
# 88,537 -> 90,390.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_FUSED_EP_LAYOUT"] = (
    "dsv41_flash_dp8_agentx_c128_visit.json"
)
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_FUSED_EP_LAYOUT_ROUTING"] = "gate"
# G-B-lite on seams of <= 96 rows (128-row seams take the shifted mHC kernel).
# AgentX c128, 96 (j-978d61e1-84dc-4a88-aa99) vs 192 (j-ee9b3ff5-d135-4a6b-8191):
# paired p90 interactivity ratio 1.006 (95% CI 1.001-1.012), tok/s/chip +0.9%.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_MHC_GBLITE_MAX_ROWS"] = "96"
# fused_moe_v2 plan kernel with the tables-only build for t_local 96..192 (the
# full kernel serves t_local <= 64). AgentX c128 real-draft, only this differs
# (j-1920a203 vs j-bac5367d): paired p90 interactivity ratio 1.013 (95% CI
# 1.005-1.021), tok/s/chip +2.6%; product 1.039.
# (plan kernel: superseded by push dispatch below, which turns it off)
CONFIG["runtime_env"]["SGLANG_TPU_FUSED_EP_V2_PLAN_KERNEL_TABLES"] = "1"
# libtpu flags (as at c4/c16/c32/c64/c256): XLA sorts on
# SparseCore, async all-to-all, no deduplicated calls. Replaces the inherited
# string: the flags are part of the compile-cache key. AgentX c128, same-slot pair, only this differs
# (j-f454de18-0aa3-4a67-937e vs control j-9b48ad62-9603-4d43-be65): paired p90
# interactivity ratio 1.019 (95% CI 1.016-1.022), median 1.029 (1.027-1.031),
# tok/s/chip 61,092 -> 61,955; product 1.033. No SparseCore halt.
CONFIG["runtime_env"]["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_dvfs_p_state=7"
    " --xla_tpu_enable_offloading_sort_to_sparsecore=true"
    " --xla_tpu_enable_async_all_to_all=true"
    " --xla_tpu_enable_deduplicated_calls=false"
)

# Fused EP push dispatch (K512 push build, K512 dispatch_ahead 25)
# with the all-gather plan kernel off and push's in-kernel remote DMAs allowed.
# Adopted at every c: same-slot A/B at c4 (no loss: p90 0.995 n.s., median +2.2%) and c256 (merged tip
# 9e490d50b8, j-3838320f-fb7d-4db8-a5c0 vs j-0afca8ae-f063-4e6a-8ff2: p90
# interactivity 1.044 [1.040, 1.047], tok/s +0.8%, product ~1.052); in-between c by
# mechanism (device: t48 ~neutral, t96-512 0.81-0.96x all-gather time). GSM8K 0.903.
CONFIG["runtime_env"]["LIBTPU_INIT_ARGS"] += (
    " --xla_mosaic_unsafe_allow_multicore_remote_dma=true"
)
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_FUSED_EP_DISPATCH"] = "push"
CONFIG["runtime_env"]["SGLANG_TPU_FUSED_EP_V2_PLAN_KERNEL"] = "0"
# Extend SWA window plan sized to this rank's request slots instead of the fixed
# extend batch of 64 (exact; no-op at 64+ slots per rank, so not set at c256+).
# adopted on the mechanism (exact; extends plan fewer padded window rows). GSM8K 0.904 vs 0.903.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_SWA_PLAN_TRIM"] = "1"
