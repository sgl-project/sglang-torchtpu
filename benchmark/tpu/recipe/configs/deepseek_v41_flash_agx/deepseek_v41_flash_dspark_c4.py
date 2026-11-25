"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 4 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_agx/deepseek_v41_flash_dspark_c4.py

deepseek_v41_flash_dspark_agentx.py (128 running requests, 16 per rank) with
three low-concurrency settings, measured together:

- prefill_decode_interval_min_bs 8: under attention DP, the 4-round decode
  interval after any rank's extend made every rank skip prefill (~4 x 51 ms per
  1024-token chunk at c4). Ranks with fewer than 8 running requests now admit
  prefill every round; the interval still applies at high load.
- tpu_compile_bs from 1 (inherited [8, 16, 32, 64]): a rank holds at most ~2
  requests at c4, so every verify/draft step padded to bs 8 (48 rows per rank).
  GSM8K-1319 with bs 1 compiled and used: 0.912 (parallel 8) / 0.906 (parallel
  128), no NaN (j-516f79b6-4fb7-4e75-9ba1).
- chunked prefill 2048 tokens per rank (16384 global, max_prefill_tokens to
  match; the 2048 token bucket is already compiled).

Shared compile owners from a trace-cost profile
(sglang_tpu/compilation/cost_profiles/dsv41_flash_agentx_c4.json, built with
`python -m sglang_tpu.compilation.shared_compile_cost` from the c4 startup logs of
j-fecb900b and j-066b785d): the heavy 2048-token forwards are spread, and graphs
sharing kernel shapes stay on one rank (a repeat traced in ~13% of the first).
Startup 1223 -> 1011 s, slowest rank's tracing 734 -> 438 s, serving unchanged
(j-ee92835e boot 1 vs j-fecb900b). A missing or stale profile falls back to
round-robin.

AgentX canonical c4 on the fused-EP tree, only these settings vs the default
recipe (j-20e8c9d3-e157-40e8-9a18 vs j-a9903e34-edd8-494f-b8dc): 5,107 -> 5,600
tok/s/chip, p90 interactivity 89.1 -> 104.4, TTFT p90 1,457 -> 1,171 ms, TPOT
p50 9.39 -> 7.84 ms; paired per-request interactivity 1.207 (95% CI
1.195-1.215), TTFT 0.882 (0.864-0.893).
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash_agx.deepseek_v41_flash_dspark_agentx import (
    CONFIG as _AGENTX,
)

CONFIG = deepcopy(_AGENTX)
CONFIG["runtime_env"]["SGLANG_TPU_SHARED_COMPILE_COST_PROFILE"] = (
    "dsv41_flash_agentx_c4.json"
)
CONFIG["server_args"].update(
    {
        "prefill_decode_interval_min_bs": 8,
        "tpu_compile_bs": [1, 2, 4, 8, 16, 32, 64],
        "chunked_prefill_size": 16384,
        "max_prefill_tokens": 16384,
    }
)
# fused_moe_v2 plan kernel for the EP dispatch plan: engages at t_local <= 42
# (c4 verify buckets: 6/12/24), self-gates above. AgentX c4, only this differs
# (j-9ebcf26c-8e0b-407e-ae77 vs j-624728b3-4914-4b44-b24d): median per-request
# interactivity ratio 1.0145 (95% CI 1.0107-1.0178), p90 1.012 (n.s.), median
# TTFT 0.977. Superseded at c4 by push dispatch (below), which turns it off.
# Host Engram batch read with int64 indices: numerics-exact (16/16). AgentX c4,
# only this differs (j-402c28f8 vs j-20dcea16): p90 interactivity ratio 1.020,
# median 1.012 (95% CI 1.008-1.017), tok/s/chip +0.2%; product 1.022.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_ENGRAM_BATCH_INT64"] = "1"
# Eager-mode defer-and-fuse for the host-side eager ops. AgentX c4: product
# p90 interactivity x tok/s 1.057; GSM8K 0.905 vs 0.907 (flips 36/34, p=0.905), accept 3.56.
CONFIG["runtime_env"]["SGLANG_TPU_EAGER_MODE"] = "DEFER_AND_FUSE"
# Shifted mHC row floor at 1. Measured together with the indexer publication
# live-length clamp (not in this tree yet): AgentX c4 product 1.093; GSM8K 0.917
# (flips 40/27, p=0.14), accept 3.48.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_SHIFTED_MHC_MIN_ROWS"] = "1"
# DP lockstep host overhead: DP sync rows through shared
# memory instead of gloo, an idle DP rank's spec inputs and attention metadata reused per
# shape, DP token counts kept on device. Decode step wall 15.28 -> 14.34 ms, DP gather
# wait 1.57 -> 0.56 ms (span diag). AgentX c4, only this differs, two pairs: paired
# median interactivity 1.046 (95% CI 1.039-1.053) and 1.008 (1.002-1.011), p90 1.010 and
# 1.014 (n.s.) (j-115506ff-d740-4c11-abfb vs j-3cf9d790-7db7-4ccd-82cd;
# j-27880e24-e456-42fc-a85b vs j-5c1cdf9e-08e7-45c9-9ebf). GSM8K 0.906 (j-45978f7e).
# Gated to c4 by the small->large rule (adopt gated where it helps in any regime):
# c16 product 0.990 (j-0b7878ea-05b5-430a-8a29 vs j-b47e7127-a384-4059-b636), off there.
CONFIG["runtime_env"]["SGLANG_TPU_DP_SYNC_SHM"] = "1"
CONFIG["runtime_env"]["SGLANG_TPU_SPEC_IDLE_INPUT_CACHE"] = "1"
CONFIG["runtime_env"]["SGLANG_TPU_DP_COUNT_CACHE"] = "1"
# libtpu flags: XLA sorts on SparseCore (the indexer's
# candidate sorts run async beside TensorCore work), async all-to-all, and no
# deduplicated calls (no per-layer code reloads). Replaces the inherited string:
# the flags are part of the compile-cache key. AgentX c4, only this differs
# (j-9c4698b0-8b70-4490-a9c2 vs j-19dc4fd2-12df-481a-b85c): paired p90
# interactivity ratio 1.103 (95% CI 1.046-1.135), median 1.076 (1.068-1.082),
# tok/s/chip 6,581 -> 6,729; product 1.128. No SparseCore halt.
CONFIG["runtime_env"]["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_offload_gather_to_sparsecore=false --xla_tpu_dvfs_p_state=7"
    " --xla_tpu_enable_offloading_sort_to_sparsecore=true"
    " --xla_tpu_enable_async_all_to_all=true"
    " --xla_tpu_enable_deduplicated_calls=false"
    # Push dispatch's in-kernel remote DMAs (below).
    " --xla_mosaic_unsafe_allow_multicore_remote_dma=true"
)
# Fused EP push dispatch (K512 push build: decode combine,
# pair kernels, one-hot scale slab; K512 dispatch_ahead 25) with the all-gather
# plan kernel off, as measured. AgentX c4, only this differs (same tree,
# j-1109e6fc-7260-4e8e-8500 vs j-975364bf-bf27-4719-b80d): p90 interactivity
# ratio 0.995 (n.s.), median 1.0225 (95% CI 1.018-1.026): neutral to slightly
# positive (the run's +6.1% tok/s/chip was client completion order, not serving
# speed: matched latency 0.997). GSM8K 0.910 vs base 0.904 (McNemar p=0.39). No
# SparseCore halt.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_FUSED_EP_DISPATCH"] = "push"
CONFIG["runtime_env"]["SGLANG_TPU_FUSED_EP_V2_PLAN_KERNEL"] = "0"
# Extend SWA window plan sized to this rank's request slots instead of the fixed
# extend batch of 64 (exact; no-op at 64+ slots per rank, so not set at c256+).
# paired c4 (j-13a56955 / j-7d1cea83): product x0.997 (flat), median TTFT -2.9%. GSM8K 0.904 vs 0.903.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_SWA_PLAN_TRIM"] = "1"
# Fused attention output projection (FP8 wo_a + wo_b in one Pallas kernel; BF16 wo_a
# released). AgentX c4 real-draft canonical, only this differs (same tree 8521a53e08
# + the host-copy wo_b scale fix, j-549beab4-4ff0-4e16-b660 vs j-73bf53ac-44d7-436e-96c1):
# paired median interactivity 1.023 (95% CI 1.017-1.031), p90 1.062 (0.990-1.114, n.s.),
# median TTFT 0.959 (0.946-0.969). tok/s/chip -6.7% is the client's replay schedule
# (one 300 s idle cap, 26 of 28 missing requests in one trace; matched requests all
# faster), within c4 run-to-run noise. Device: o-proj 42.4 -> 24.7 us
# per layer per verify (XProf j-cf23a401-c328-475d-ac85, ~1.03x the HBM floor).
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_FUSED_OPROJ"] = "1"
