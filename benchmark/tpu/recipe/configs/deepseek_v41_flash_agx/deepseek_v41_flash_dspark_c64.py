"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 64 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_agx/deepseek_v41_flash_dspark_c64.py

deepseek_v41_flash_dspark_agentx.py with max_running_requests 256 (32 per rank).

Why 32 per rank: session-affinity routing pins each conversation and its
subagents to one rank, so ranks are uneven. At c32 the busiest rank peaked at
11 running requests vs 4 on average (2.75x; canonical c32
j-3e6d0e27-2566-49c7-aca4). At c64 (8 per rank on average) that is ~22, above
the base cap of 16, so the base would queue sessions on the busy ranks.

Capacity, computed (the boot log and smoke run check it on hardware):
- KV pool: ~20.2M tokens per rank at mem_fraction_static 0.83 and 16 requests
  per rank (canonical c32 j-3e6d0e27-2566-49c7-aca4). The request-scaled SWA
  windows, draft ring and state come out of the same budget, ~320 MiB more per
  rank at 32 requests, so the pool shrinks by a few percent. 32 requests x
  ~140k average context is 4.5M (22%).
- Compile buckets: tpu_compile_bs [8, 16, 32, 64] covers 32 per rank; no new
  compile shapes.
- HBM: 14.7 GB was free after precompile at 16 per rank, with the bs 64
  programs compiled. Headroom at 32 per rank is not measured yet.
"""

from benchmark.tpu.recipe.configs.deepseek_v41_flash_agx.deepseek_v41_flash_dspark_agentx import (
    with_max_running_requests,
)

CONFIG = with_max_running_requests(256)
# Pin each rank's scheduler process to its own CPUs (SGLANG_SET_CPU_AFFINITY): the
# MLP-sync wait across DP ranks halves, per-op costs unchanged. Paired per-request
# interactivity: c32 +2.9% (j-fff55c92 vs j-b29f6ac4), c64 canonical +2.3%
# (j-10b94a15-88dd-40a2-af97 vs j-b598a23c-9c0b-4fef-87d2); neutral at c4
# (j-6643e364-2d36-454b-ae78 vs j-eecd8be6-a630-4b97-879e), so off at c4 (c16: see its recipe).
CONFIG["runtime_env"]["SGLANG_SET_CPU_AFFINITY"] = "1"
# 8 decode rounds after each prefill step (the DSpark base runs 4), and at most one
# new request per prefill step. AgentX canonical c64, same tree and image:
# pdi 8 + prefill_max_requests 1 (j-7629e917-d728-46f0-b13f) vs pdi 8 alone
# (j-635c72bc-c100-46ea-8f46): paired p90 interactivity ratio 1.033 (95% CI
# 1.026-1.042), tok/s/chip 69,280 -> 70,003. Vs the fused candidate
# (j-47da0ee3): p90 interactivity 50.2 -> 65.1, paired p90 ratio 1.303 (95% CI
# 1.292-1.313), tok/s/chip 67,120 -> 70,003, p90 TTFT 13.1 s. pdi 6 gave 1.214
# (FAST). Policy: tok/s and p90 interactivity decide, TTFT is reported only.
# Raised to 16 by the product rule (tok/s x p90 interactivity): AgentX c64
# real-draft canonical, pdi 16 + cap 1 (j-fccd5074-1ed1-460c-97d8) vs pdi 8 +
# cap 1 (j-625ee78d-2fab-4067-b5f1): paired p90 ratio 1.231 (95% CI
# 1.224-1.240), tok/s/chip 66,825 -> 60,780; product 1.119
# (pdi 12: 1.105).
CONFIG["server_args"]["prefill_decode_interval"] = 16
CONFIG["server_args"]["prefill_max_requests"] = 1
# Fused attention output projection (wo_a requantized to per-channel FP8), all
# layers. AgentX c64 real-draft canonical, only this differs
# (j-48518c8c-9c54-42dd-b883 vs j-44717140-1cc1-4baf-a65d): paired p90
# interactivity ratio 1.054 (95% CI 1.043-1.064), tok/s/chip +1.1%, KV +4.4%.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_FUSED_OPROJ"] = "1"
# Shifted mHC row floor 1, measured with the indexer publication clamp (not in this
# tree yet): AgentX c64 (j-0d42aad8 vs j-d7edaa78): p90 1.0015 (n.s.), median +1.5%,
# TTFT -5%, tok/s +0.5%; product ~1.007, exact.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_SHIFTED_MHC_MIN_ROWS"] = "1"
# libtpu flags (as at c4/c16/c256): XLA sorts on
# SparseCore, async all-to-all, no deduplicated calls. Replaces the inherited
# string: the flags are part of the compile-cache key. AgentX c64, only this
# differs (j-8f83d190-5bae-436c-8a6e vs j-d7edaa78-a9a7-41ac-828b): paired p90
# interactivity ratio 1.041 (95% CI 1.036-1.045), median 1.046 (1.044-1.048),
# p90 TTFT 0.907, tok/s/chip 68,800 -> 71,479; product 1.082. No SparseCore halt.
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
# paired c64 (j-0f04009f / j-d8028ecf): product x1.029, p90 interactivity x1.022 [1.019, 1.025]. GSM8K 0.904 vs 0.903.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_SWA_PLAN_TRIM"] = "1"
