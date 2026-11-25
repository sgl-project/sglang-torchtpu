"""DeepSeek-V4.1-Flash + DSpark for AgentX at concurrency 256 (one tpu7x-8 host).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_agx/deepseek_v41_flash_dspark_c256.py

deepseek_v41_flash_dspark_agentx.py with max_running_requests 512 (64 per rank,
as many sequences per engine as vLLM-TPU runs on this host).

Why 64 per rank: 32 per rank on average; with session-affinity routing the
busiest rank peaked at 2.75x the average at c32 (canonical c32
j-3e6d0e27-2566-49c7-aca4). 64 is the largest compiled bucket
(torch_compile_max_bs 64), so it is the ceiling without new compile shapes; at
this concurrency the relative imbalance should be smaller than at c32.

Capacity, computed (the boot log and smoke run check it on hardware):
- KV pool: ~20.2M tokens per rank at mem_fraction_static 0.83 and 16 requests
  per rank (canonical c32 j-3e6d0e27-2566-49c7-aca4). The request-scaled SWA
  windows, draft ring and state come out of the same budget, ~960 MiB more per
  rank at 64 requests, so the pool shrinks by a few percent. 64 requests x
  ~140k average context is 9.0M (45%).
- Compile buckets: tpu_compile_bs [8, 16, 32, 64] covers 64 per rank; no new
  compile shapes.
- HBM: 14.7 GB was free after precompile at 16 per rank, with the bs 64
  programs compiled. Headroom at 64 per rank is not measured yet.
"""

from benchmark.tpu.recipe.configs.deepseek_v41_flash_agx.deepseek_v41_flash_dspark_agentx import (
    with_max_running_requests,
)

CONFIG = with_max_running_requests(512)
# Pin each rank's scheduler process to its own CPUs (SGLANG_SET_CPU_AFFINITY): the
# MLP-sync wait across DP ranks halves, per-op costs unchanged. Paired per-request
# interactivity: c32 +2.9% (j-fff55c92 vs j-b29f6ac4), c64 canonical +2.3%
# (j-10b94a15-88dd-40a2-af97 vs j-b598a23c-9c0b-4fef-87d2); neutral at c4
# (j-6643e364-2d36-454b-ae78 vs j-eecd8be6-a630-4b97-879e), so off at c4 (c16: see its recipe).
CONFIG["runtime_env"]["SGLANG_SET_CPU_AFFINITY"] = "1"
# Poll the tokenizer for new requests every 8 scheduler steps instead of every
# step: fewer host-side receives in the decode loop. AgentX canonical c256, only
# this differs from the fused candidate (j-b01658da-4e8e-46d2-9f49 vs
# j-b24a56fa-10a1-4e00-a850): p90 interactivity 30.2 -> 30.5, paired p90 ratio
# 1.0085 (95% CI 1.0014-1.0145, 12,956 matched requests), tok/s/chip
# 111,559 -> 112,874. Neutral at c4/c32 with worse TTFT, so c256 only.
CONFIG["server_args"]["scheduler_recv_interval"] = 8
# Stage A mixed prefill + verify (needs the Stage A scheduler and TPU spec-worker
# commits; SGLANG_TPU_MIXED_VERIFY_FUSED unset selects Stage A): a prefill chunk
# and the other requests' verify run in one forward, so a rank's prefill no longer
# stalls decode on every DP rank. AgentX canonical c256 on the fused candidate,
# only this differs (j-06f9ca04-d086-4057-abbb vs j-b24a56fa-10a1-4e00-a850):
# paired p90 interactivity ratio 1.123 (95% CI 1.116-1.130), tok/s/chip -9.2%,
# p90 TTFT x1.19. Adopted at c256 despite the throughput loss (user, 2026-11-24).
CONFIG["server_args"]["enable_mixed_chunk"] = True
# The base recipe's XLA overlap flags (async collective fusion, latency-hiding
# scheduler, collective matmul) on top of the DSV4.1 pair; the SparseCore gather
# offload stays off. AgentX canonical c256 on the fused candidate, only this
# differs (j-0d1ba031-0ed9-41c4-a0cc vs j-b24a56fa-10a1-4e00-a850): paired p90
# interactivity ratio 1.018 (95% CI 1.012-1.025), tok/s/chip 111,559 -> 114,548.
# Neutral at c4/c32. Part of the compile-cache key: first boot compiles cold.
CONFIG["runtime_env"]["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_enable_async_collective_fusion=true"
    " --xla_tpu_overlap_compute_collective_tc=true"
    " --xla_tpu_enable_async_collective_fusion_multiple_steps=true"
    " --xla_tpu_enable_latency_hiding_scheduler=true"
    " --xla_latency_hiding_scheduler_rerun=10"
    " --xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative"
    " --xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative"
    " --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false"
    " --xla_tpu_offload_gather_to_sparsecore=false"
    " --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"
    " --xla_tpu_dvfs_p_state=7"
    # libtpu flags (adopted at c4): XLA sorts on SparseCore, async
    # all-to-all, no deduplicated calls. AgentX c256, only these differ
    # (j-4f877116-38c7-4d98-9e85 vs j-3de577be-cb70-41e4-9684): paired p90
    # interactivity ratio 1.022 (95% CI 1.013-1.031), median 1.018
    # (1.014-1.020), tok/s/chip 154,710 -> 157,951; product 1.044. No
    # SparseCore halt.
    " --xla_tpu_enable_offloading_sort_to_sparsecore=true"
    " --xla_tpu_enable_async_all_to_all=true"
    " --xla_tpu_enable_deduplicated_calls=false"
)
# Finer compile buckets: the busiest rank runs 18 (p50) / 25 (p90) requests, and
# [8, 16, 32, 64] padded 63% of steps to bs 32. AgentX c256 real-draft canonical,
# only this differs (j-38a4957a-116b-4ce9-9762 vs j-70937ca0-946c-4d53-a4f8):
# paired p90 interactivity ratio 1.091 (95% CI 1.081-1.099), tok/s/chip
# 103,221 -> 111,162; product 1.175. Needs the sublane-aligned RoPE token tile.
CONFIG["server_args"]["tpu_compile_bs"] = [8, 16, 24, 32, 40, 48, 64]
# 2048 prefill tokens per rank and step, 6 decode rounds after each prefill: c256
# is prefill-bound, and the larger chunk amortizes the fixed host work around
# each extend. AgentX c256 real-draft canonical, with the finer buckets above
# (j-90290162-3add-43cd-9de0 vs j-70937ca0-946c-4d53-a4f8): tok/s/chip
# 103,221 -> 139,662, paired p90 interactivity ratio 0.995
# (95% CI 0.985-1.002); product 1.346 (buckets alone 1.175).
CONFIG["server_args"]["chunked_prefill_size"] = 16384
CONFIG["server_args"]["max_prefill_tokens"] = 16384
# Decode interval raised from 6 to 12, with the prefill attention-work cap
# (W = 800M token-pairs while any DP rank decodes) and shortest-prefill-first
# ordering. AgentX c256 real-draft canonical, only these differ
# (j-4baee8c1-4a23-43f3-b28d vs j-3de577be-cb70-41e4-9684): paired p90
# interactivity ratio 1.403 (95% CI 1.388-1.419), tok/s/chip 154,710 -> 149,536;
# p90 TTFT 46.6 -> 32.8 s; product 1.356. pdi 12 alone
# (j-5e265a35-4468-41fa-a9bb) lost 15% tok/s; the cap + SPF recover it.
# Raised again to 16 (same cap and SPF). AgentX c256 real-draft canonical, same
# slot and tree (j-8743703e-b4c7-4112-956f vs j-54e3d3de-de67-4e75-9ec8, pdi 12):
# paired p90 interactivity ratio 1.112 (95% CI 1.102-1.121), tok/s/chip
# 161,772 -> 155,300; product 1.067; 0 failed. Far-tail cost of shortest-prefill-first:
# >=32K-prompt TTFT p90 38.0 -> 33.8 s but p99 387 -> 477 s (+23%) and the worst
# request 49 -> 54 min; add a starvation guard if that tail matters.
CONFIG["server_args"]["prefill_decode_interval"] = 16
CONFIG["runtime_env"]["SGLANG_DP_ATTN_WORK_CAP"] = "800000000"
CONFIG["runtime_env"]["SGLANG_PREFILL_SPF"] = "1"
# Fused attention output projection (wo_a requantized to per-channel FP8), all
# layers. AgentX c256 real-draft, only this differs (j-5fa78565-fffd-4f9f-99ed vs
# j-9bcca990-173d-45cb-9ae1): paired p90 interactivity ratio 1.022 (95% CI
# 1.012-1.033), tok/s/chip flat (-0.3%), KV +4.6%; product 1.019.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_FUSED_OPROJ"] = "1"
# Host Engram pack on 4 threads: AgentX c256 paired vs the sweep c256
# point: p90 interactivity 1.015 (significant), tok/s/chip +2.6%; product 1.042.
CONFIG["runtime_env"]["SGLANG_TPU_DSV41_ENGRAM_PACK_THREADS"] = "4"

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
