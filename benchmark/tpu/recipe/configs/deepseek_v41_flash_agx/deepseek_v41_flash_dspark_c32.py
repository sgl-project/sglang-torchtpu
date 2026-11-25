"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 32 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_agx/deepseek_v41_flash_dspark_c32.py

Same as deepseek_v41_flash_dspark_agentx.py: 128 running requests (16 per rank).
At c32 the busiest rank peaked at 11 running requests (4 per rank on average;
canonical c32 j-3e6d0e27-2566-49c7-aca4, session-affinity routing), so 16 per
rank did not queue there. c8 and c32 are the validated canonical settings.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash_agx.deepseek_v41_flash_dspark_agentx import (
    CONFIG as _AGENTX,
)

CONFIG = deepcopy(_AGENTX)
# Pin each rank's scheduler process to its own CPUs (SGLANG_SET_CPU_AFFINITY): the
# MLP-sync wait across DP ranks halves, per-op costs unchanged. Paired per-request
# interactivity: c32 +2.9% (j-fff55c92 vs j-b29f6ac4), c64 canonical +2.3%
# (j-10b94a15-88dd-40a2-af97 vs j-b598a23c-9c0b-4fef-87d2); neutral at c4
# (j-6643e364-2d36-454b-ae78 vs j-eecd8be6-a630-4b97-879e), so off at c4 (c16: see its recipe).
CONFIG["runtime_env"]["SGLANG_SET_CPU_AFFINITY"] = "1"
# 16 decode rounds after each prefill step (the DSpark base runs 4): the slowest
# users' decode is interrupted less often. AgentX c32 FAST, same tree and image,
# only this differs (j-015ef2f5-8ed0-450d-b4e8 vs j-f301a224-21e0-4757-931f):
# p90 interactivity 71.4 -> 100.5, paired p90 ratio 1.412 (95% CI 1.392-1.433),
# tok/s/chip 39,925 -> 38,630 (-3%), p90 TTFT 3.0 -> 11.3 s. Beats pdi 8 (1.217)
# and 12 (1.314); prefill_max_requests 1 on top costs 2% p90. Policy: tok/s and
# p90 interactivity decide, TTFT is reported only.
# Raised to 24 by the product rule (tok/s x p90 interactivity): AgentX c32
# real-draft canonical, pdi 24 (j-7f72b75c-8733-4b81-8f7e) vs 16
# (j-710d46e2-33b8-4120-bce4): paired p90 ratio 1.055 (95% CI 1.042-1.065),
# tok/s/chip 39,070 -> 38,592; product 1.042 (pdi 32: 0.986).
CONFIG["server_args"]["prefill_decode_interval"] = 24
# fused_moe_v2 plan kernel: engages at t_local 48 (c32 verify bucket bs 8 x 6),
# self-gates at 96+. AgentX c32 fixed-mode, only this differs
# (j-c8fe8f33-7cf7-48a9-8e18 vs j-504b50ab-836c-4544-acf1): paired p90
# interactivity ratio 1.031 (95% CI 1.019-1.045), tok/s/chip +3.1%.
# (plan kernel: superseded by push dispatch below, which turns it off)
# Visit-balanced fused-EP expert layout (packaged JSON, gate routing). AgentX c32
# real-draft canonical, only this differs (j-7cdeb834 vs j-710d46e2-33b8-4120-bce4):
# paired p90 interactivity ratio 0.999 (95% CI 0.985-1.009), tok/s/chip
# 39,070 -> 40,025; product 1.023.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_FUSED_EP_LAYOUT"] = (
    "dsv41_flash_dp8_agentx_c32_visit.json"
)
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_FUSED_EP_LAYOUT_ROUTING"] = "gate"
# Shifted mHC row floor 1: verify seams below 96 rows take the shifted kernel
# (same basis as c4/c64; exact).
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_SHIFTED_MHC_MIN_ROWS"] = "1"
# SWA publish plan: plan a forward's write records once, pack each layer against it
# (equivalence 16/16). AgentX c32 same-cluster pair: p90 1.0125, median 1.007;
# product 1.016.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_SWA_PUBLISH_PLAN"] = "1"
# libtpu flags (as at c4/c16/c64/c256): XLA sorts on
# SparseCore, async all-to-all, no deduplicated calls. Replaces the inherited
# string: the flags are part of the compile-cache key. AgentX c32,
# only this differs (j-c9da90d4-cbad-45f5-a048 vs control
# j-a7b9ccd3-9646-4235-8891, created 81 min apart): paired p90 interactivity
# ratio 1.066 (95% CI 1.061-1.071), median 1.073 (1.070-1.076), tok/s/chip
# 22,934 -> 23,352; product 1.085. No SparseCore halt.
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
# Fused attention output projection (FP8 wo_a + wo_b, one Pallas kernel), by mechanism:
# it removes the BF16 wo_a stream (~0.4 ms per cycle), measured on at c4 (device
# 42.4 -> 24.7 us per layer, paired median interactivity 1.023) and c64-c256. The
# next full sweep re-measures c32.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_FUSED_OPROJ"] = "1"
