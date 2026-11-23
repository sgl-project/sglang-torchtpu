# GLM-5.3-Flash AgentX on TPU

GLM-5.3-Flash (fp8) on one tpu7x host: 2x2x1, 4 chips = 8 TensorCores, tp8, 1M context.
Benchmark: AgentX (`agentx-harness==1.0.14`, SemiAnalysis cc traces, 1 h window per
concurrency). MTP verify at InferenceX's golden acceptance: K=3 steps -> AL 3.00, K=7 ->
AL 3.93. One run JSON in [`runs/`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs) per point.

[`agentx.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/agentx.py) holds the server config of every point, one profile per concurrency
(`--conc <cc>` prints it); each profile renders its point's run JSON exactly. The other
AgentX files live in
[`benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx);
the file names below are relative to it.

## Results

Final curve (2026-11-23): c1, c4, c8, c16, c32. Real weights, one 1 h window per point.
Configs are in Per point and Tunable configs.

| cc | tok/s/chip | M tokens/$ | 1/ITL p50 / p90 | TTFT p90 s | E2E latency p90 / limit s | prefix hit / ideal % | run JSON |
|---|---|---|---|---|---|---|---|
| 1 | 6,452 | 22.6 | 460.0 / 414.4 | 0.86 | 11.8 / 24.0 | 98.17 / 98.29 | [`runs/R1b_c1_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/R1b_c1_real_l3_ring_full.json) |
| 4 | 8,876 | 31.0 | 420.0 / 321.5 | 0.79 | 10.1 / 20.4 | 97.67 / 97.88 | [`runs/C4u_c4_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/C4u_c4_real_l3_ring_full.json) |
| 8 | 14,920 | 52.1 | 327.3 / 208.3 | 0.97 | 10.2 / 19.6 | 97.48 / 97.78 | [`runs/K2ts_c8_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/K2ts_c8_real_l3_ring_full.json) |
| 16 | 26,360 | 92.1 | 201.5 / 118.5 | 2.40 | 13.9 / 20.3 | 97.23 / 97.60 | [`runs/C16_S2ts_kda1400_bf16_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/C16_S2ts_kda1400_bf16_real_l3_ring_full.json) |
| 32 | 43,843 | 153.2 | 117.5 / 53.7 | 2.37 | 27.8 / 28.2 | 97.16 / 97.73 | [`runs/R10_c32_mr32_combo_real_l3_ring_full_ixsplit.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/R10_c32_mr32_combo_real_l3_ring_full_ixsplit.json) |

- Latency limit: a point's p90 E2E latency may not exceed GB300's p90 at the same concurrency
  plus 10 s. GB300 p90 (InferenceX run 42026173061): c1 14.0, c4 10.4, c8 9.6, c16 10.3,
  c32 18.2 s. Every point is within it.
- 1/ITL = output tok/s per user = 1000 / inter-token latency (ms) at that percentile.
- Tokens/$ = tok/s/chip x 3600 / $1.03 per chip-hour (InferenceX's TPU7x rate), in millions.
- tok/s/chip counts prompt and completion tokens, cached prompt tokens included.
- Ideal = aiperf's theoretical prefix-cache hit, the figure InferenceX publishes.
- Every point: 0 recompiles after ready, 0 OOM, 0 scheduler exceptions, 1-2 failed requests
  (aiperf `InvalidInferenceResultError`). One window per point; noise is about ±3%. c32
  checked TP-rank agreement every 64 verify steps: 0 mismatches.
- Hosts: one tpu7x host each, libtpu 0.0.50.
  - c1-c16: one host of a 2x2x2 dev pod run alone, torch_tpu 0.2.0, engine `b38e84a11e`.
  - c32: a 2x2x1 tpuq pod, torch_tpu 0.2.1.dev, engine `f977f22c30`.
  - The branch head adds the exact logits all-reduce (Optimizations, Correctness), which the
    timed windows do not run (simulated acceptance).

## Optimizations

Base kernels on at every point: fused mHC, streaming MoE, FP8 small-M linear, q-block
sparse MLA prefill.

Decode:
- MoE ring kernel: FP8 expert weights stream once through a VMEM ring.
- Adaptive MTP: 7 draft steps at 1 running, else 3 (c1/c8).
- Simulated-acceptance accept step compiled into the decode program.
- No eager ops in the decode step.
- Sort-free router top-k and ring routing, bit-exact.
- Split-cache decode attends only each row's own candidates (c8-c32).
- StreamIndex top-k: one DMA per run of consecutive KV pages.

Prefill:
- MoE dispatch of pre-quantized fp8 rows over a SparseCore gather.
- Token-split prefill compacts owned rows in one Pallas pass.
- Fused mHC kernel on input-scattered prefill batches.
- Router top-k kernel for prefill.
- SparseCore top-k width buckets read each row's prefix in place.
- DSA token split: each rank attends only its own cache rows (c8-c32).
- bf16 merge of the split cache (c16/c32).

Cache:
- MLA latent cache split in-page: by 2 (c8/c16), by 8 (c32).
- DSA index-key cache split by 8 with a row-slice prefill exchange (c32); the exact exchange
  needs the latent cache split by 8. The freed HBM goes to a larger device KV pool. KV bytes
  per token per rank: 3,840 at latent split 2, 1,536 at split 8, 864 at split 8 with the
  index split.
- Host KV tier: raiden, `page_first_fused`, `write_through`; it also keeps KDA states.
- Hybrid-cache checkpoints level 3: a KDA checkpoint at every prefill chunk.
- No KDA checkpoints during decode at c32: AgentX does not reuse them, and evicting them
  takes cached KV along.
- KDA slots sized for the conversations AgentX keeps cached.
- Raw-ring pooled indexer; page table in HBM.

Scheduling:
- Prefill-decode interval 16 at c1-c16; 0 at c32, where a fixed interval starves admission.
- Decode buckets: every batch size up to c16; c32 every size to 20, then 24, 28, 32, since
  each decode program costs host RAM.

Correctness:
- c32: `--xla_tpu_enable_sparse_core_collective_offload_all_gather=false` in
  `LIBTPU_INIT_ARGS` keeps all-gathers off the SparseCore, whose offloaded all-gather
  returns wrong data in the split decode graph.
- TP vocab logits and the draft vocab gather use an exact all-reduce of zero-padded per-rank
  slots under the direct tracer (`sglang_tpu/all_gather_via_all_reduce.py`), so TP ranks
  sample the same tokens with real sampling.

## Accuracy

GSM8K, 200 questions, sgl-eval 0.1.3: temperature 1.0, top_p 0.95, thinking, max_tokens
16000. Real acceptance: the point's run JSON without the `SGLANG_SIMULATE_ACC_*` variables
(`agentx.py --eval`), with a TP-rank check every 64 verify steps. Engine `f977f22c30`; c32
also with the exact logits all-reduce.

| config | GSM8K |
|---|---|
| c8 | 98.0% |
| c32 | 98.5% |

## Replicate

### Requirements

- Host: one tpu7x 2x2x1 host (tpu7x-8), libtpu 0.0.50 as measured (Results, Hosts), plus:
  - `libibverbs.so.1` (`apt-get install libibverbs1`) for the raiden host-tier transfers;
  - `transformers==5.19.0 blake3 timm==1.0.25`, which the images lack:
    `pip install --no-deps --target $DEPS ...`, then `$DEPS` first on `PYTHONPATH`.
- Weights: target and MTP draft at `/weights/GLM-5.3-Flash`.
- Caches: put them on a PVC, never on the container overlay. One directory per point holds
  the compile caches, `TMPDIR`, `SGLANG_CACHE_DIR` (`~/.cache/sglang` links to it) and
  `PROMETHEUS_MULTIPROC_DIR`.
- Core dumps: `ulimit -c 0`, because one rank's core dump fills the disk.
- Host memory: the pod peaks at 532-913 GB of the node's ~987 GB allocatable (table below).
  Keep `/dev/shm` and other processes off the node.
- Client: `agentx-harness==1.0.14` in its own venv, on the same host. The trace
  `semianalysis_cc_traces_weka_062126` (HF `semianalysisai/cc-traces-weka-062126`, 393
  entries) must be pre-fetched into `HF_HOME`. This is the full uncapped corpus InferenceX
  uses for GLM.

### Per point

Run one fresh server per point. Stop it (SIGTERM) before the next point.

| point | run JSON | client `CC` | MTP K / AL | max running | decode buckets | KV pool at boot (cap) | KDA slots | latent split / token split | host tier: KV tokens / KDA states | pod memory peak | boot s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c1 | [`R1b_c1_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/R1b_c1_real_l3_ring_full.json) | 1 | `1:7,2:3` / 3.93, 3.00 | 2 | 1..2 | 3,407,872 | 74 | none / off | 6.82M / 149 | 532 GB | 1,056 |
| c4 | [`C4u_c4_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/C4u_c4_real_l3_ring_full.json) | 4 | 3 / 3.00 | 4 | 1..4 | 3,407,872 | 384 | none / off | 6.82M / 769 | 676 GB | 966 |
| c8 | [`K2ts_c8_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/K2ts_c8_real_l3_ring_full.json) | 8 | `1:7,2:3` / 3.93, 3.00 | 8 | 1..8 | 8,604,416 (9,437,184) | 800 | 2 / on | 12.9M / 1,201 | 833 GB | 1,316 |
| c16 | [`C16_S2ts_kda1400_bf16_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/C16_S2ts_kda1400_bf16_real_l3_ring_full.json) | 16 | 3 / 3.00 | 16 | 1..16 | 5,643,136 (9,437,184) | 1,400 | 2 / on | 9.59M / 2,002 | 874 GB | 1,846 |
| c32 | [`R10_c32_mr32_combo_real_l3_ring_full_ixsplit.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/R10_c32_mr32_combo_real_l3_ring_full_ixsplit.json) | 32 | 3 / 3.00 | 32 | 1..20, 24, 28, 32 | 14,155,776 | 1,200 | 8 / on | 17.0M / 3,001 | 913 GB | 1,425 |

- K = `speculative_num_steps`, giving K + 1 draft tokens and topk 1. `1:7,2:3` = 7 draft
  steps at 1 running request, 3 from 2 on; the launch K is 7. Measured mean acceptance:
  c1 3.91, c8 3.45, the others 3.00.
- Acceptance is simulated on the server; each JSON sets it:
  - `SGLANG_SIMULATE_ACC_LEN` = the golden AL of the launch K;
  - `SGLANG_SIMULATE_ACC_METHOD=match-expected`;
  - `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`;
  - c1 and c8: `SGLANG_TPU_SPEC_STEPS_BY_RUNNING=1:7,2:3` and
    `SGLANG_TPU_SPEC_SIMULATE_ACC_LEN_BY_STEPS=3:3.00,7:3.93`.

  The client sets nothing for acceptance.
- Boot = launch to ready: the NFS weight load (10-19 min, slowest rank) plus compiles; c32
  booted from a warm compile cache.
- c1: the JSON compiles with SPMD, which needs torch_tpu 0.2.1.dev or later. The measured c1
  ran on a dev pod (torch_tpu 0.2.0) with `use_spmd_compile` false; that changes only how
  compiles are distributed. The other points compile per rank.
- Same at every point:
  - context 1,048,576, page 128;
  - fp8 weights and fp8_e4m3 KV;
  - `mem_fraction_static` 0.9, ep 1;
  - prefill programs 1k/2k/3k/4k with 4096-token chunks;
  - `hybrid_cache_checkpoints_level` 3;
  - MoE ring kernel on, megakernel off (decode on the model path), direct tracer;
  - host tier raiden / `page_first_fused` / `write_through`;
  - raw-ring pooled indexer, page table in HBM;
  - checkpoint prefetch, page cache dropped after load, 2400 s unbalanced-load timeout;
  - served name `zai-org/GLM-5.3-Flash`, parsers glm45 / glm47, `127.0.0.1:30000`.
- c16 and c32 also set the bf16 split-cache merge (`SGLANG_TPU_DSA_MLA_KV_SHARD_BF16_MERGE=1`).
- All points also set five prefill/decode flags (Optimizations):
  `SGLANG_TPU_ENABLE_MOE_PREQUANT_DISPATCH`, `SGLANG_TPU_DSA_OWNED_ROWS_KERNEL`,
  `SGLANG_TPU_FUSED_MHC_INPUT_SCATTERED`, `SGLANG_TPU_ROUTER_TOPK_KERNEL_PREFILL` = 1 and
  `SGLANG_TPU_DSA_KV_SHARD_RAGGED_BLOCK=128`, with `prefill_decode_interval` 16 at c1-c16 and
  0 at c32.
- c32 also sets `SGLANG_TPU_DSA_INDEX_KV_SHARD=1` (index-key cache split), appends
  `--xla_tpu_enable_sparse_core_collective_offload_all_gather=false` to `LIBTPU_INIT_ARGS`
  and sets `mamba_track_interval` 1,048,576: no KDA checkpoints during decode
  (Optimizations).

### Server

[`launch_agentx.sh <1|4|8|16|32>`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/launch_agentx.sh) runs the steps below on the point's
[`agentx.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/agentx.py) profile (`--dry-run` prints the env, command and server args; the usage
text lists its knobs):

```bash
CACHE_ROOT=/cache/agentx AGENTX_HALF_SLICE=1 \
  benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/launch_agentx.sh 16
```

By hand, from the repo root, with `CC` from the table:

```bash
CC=16
CR=/cache/agentx/c$CC                    # on a PVC
DEPS=/cache/agentx/pydeps                # transformers 5.19.0, blake3, timm 1.0.25
export PYTHONSAFEPATH=1 PYTHONPATH=$DEPS:$PWD/python:$PWD PYTHONFAULTHANDLER=1 HF_HUB_OFFLINE=1 \
  TMPDIR=$CR/tmp PROMETHEUS_MULTIPROC_DIR=$CR/prom SGLANG_CACHE_DIR=$CR/sglang \
  TORCHINDUCTOR_CACHE_DIR=$CR/inductor TORCH_TPU_TIER3_ROOT=$CR/tier3 TORCH_TPU_TIER2_NAME=agentx_c$CC
mkdir -p $TMPDIR $PROMETHEUS_MULTIPROC_DIR $SGLANG_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TORCH_TPU_TIER3_ROOT ~/.cache
ln -sfn $SGLANG_CACHE_DIR ~/.cache/sglang
ulimit -c 0
python3 -u -m benchmark.tpu.launch_server \
  --config recipe/configs/glm_5_3_flash/agentx.py --concurrency $CC   # --dry-run prints the command only
```

`python3 -m benchmark.tpu.recipe.configs.glm_5_3_flash.agentx --conc $CC` prints the config
(`launch_server --config @<file>` takes it); `--eval` drops the simulated acceptance.

- One host of a 2x2x2 slice run alone, as the dev-pod points ran: export this first
  (launcher: `AGENTX_HALF_SLICE=1`).

  ```bash
  export FORCE_SINGLE_HOST=1 TPU_TOPOLOGY=2x2x1 TPU_HOST_BOUNDS=1,1,1 HOST_BOUNDS=1,1,1 \
    TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_WORKER_HOSTNAMES=localhost \
    TPU_WORKER_ID=0 TPU_PROCESS_ADDRESSES=localhost:8471 TPU_ACCELERATOR_TYPE=tpu7x-8
  ```

- c1 on torch_tpu 0.2.0, as measured (launcher: `AGENTX_NO_SPMD=1`): launch with
  `--config @$CR/c1.json` after

  ```bash
  python3 -m benchmark.tpu.recipe.configs.glm_5_3_flash.agentx --conc 1 --no-spmd-compile > $CR/c1.json
  ```

- The server is ready when the log prints `The server is fired up and ready to roll`.

### Client

On the same host, with `CC` from the table:

```bash
A=benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx
URL=http://127.0.0.1:30000 CC=16 OUT=$PWD/agentx-c16
mkdir -p $OUT
# 1. One flush after boot.
curl -s -m 120 -X POST $URL/flush_cache; sleep 5
# 2. Harness warm-up, then the 1 h window. Drop HF_DATASETS_OFFLINE to download the trace instead.
HF_DATASETS_OFFLINE=1 AIPERF_HTTP_TCP_USER_TIMEOUT=900000 AIPERF_REQUIRED_SERVER_METRIC_PREFIX=sglang: \
  TMPDIR=/tmp env -u HF_HUB_OFFLINE -u PYTHONPATH -u PYTHONSAFEPATH agentx profile \
  --url $URL --endpoint /v1/chat/completions --model zai-org/GLM-5.3-Flash \
  --tokenizer /weights/GLM-5.3-Flash --tokenizer-trust-remote-code --concurrency $CC \
  --public-dataset semianalysis_cc_traces_weka_062126 --benchmark-duration 3600 \
  --server-metrics $URL/metrics --output-artifact-dir $OUT/aiperf_artifacts \
  > $OUT/aiperf.log 2>&1
# 3. The table row.
python3 $A/summarize.py --md $OUT
```

- The `agentx` preset (seed 42):
  - replays sessions (`agentic_replay`) with `ignore_eos`;
  - busts the cache on each first turn;
  - caps idle gaps at 10 s.
- The harness warm-up runs before the window (up to ~10 min at c32). Only the profiling
  phase counts. No warm-up prompts or prefill probe are sent.
- Per point: boot, then 62-75 min of client time (warm-up, the 1 h window, drain).

### Metrics

[`summarize.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/summarize.py) (stdlib only) turns one output dir into a table row:
- tok/s/chip = (prompt + completion tokens) / window / 4 chips. The window runs from the
  first request start to the last request end, profiling phase only.
- 1/ITL p50 / p90 = 1000 / inter-token latency (ms).
- TTFT and E2E latency: percentiles over the profiling-phase requests.
- Prefix hit = (device + host hits) / all prefill tokens on TP0, from the
  `--server-metrics` CSV.
- Ideal = aiperf `theoretical_prefix_cache_hit`.
- On the five Results points' output dirs it gives the table's tok/s/chip, TTFT and E2E latency,
  1/ITL within 0.4 and hit within 0.02 pt.

### Submission vs measured

- The submission is [`agentx.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/agentx.py)'s profiles: each renders its run JSON in the table
  exactly ([`check_recipes.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/check_recipes.py) diffs them). The measured runs
  launched those JSONs as they are, except c1's `use_spmd_compile` (see Per point) and the
  c32 window's TP-rank check every 64 verify steps (`SGLANG_TPU_SPEC_RANK_CHECK_INTERVAL=64`;
  default 8192).
- [`launch_agentx.sh`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/launch_agentx.sh) launches the `agentx.py` profiles.

## Pending

- The exact logits all-reduce's cost with real sampling (the timed windows do not run it).

## Tunable configs

Values as the run JSONs set them, c1 / c4 / c8 / c16 / c32; unset = server default.

| knob | where set | c1 / c4 / c8 / c16 / c32 | what it does |
|---|---|---|---|
| prefill-decode interval | `prefill_decode_interval` | 16 / 16 / 16 / 16 / 0 | decode steps between prefill batches; 0 runs prefill whenever it waits |
| latent-cache split | env `SGLANG_TPU_DSA_MLA_KV_SHARDS` | unset (1) / unset / 2 / 2 / 8 | ranks that split the MLA latent cache in-page: fewer KV bytes per rank, a costlier prefill merge |
| index-key cache split | env `SGLANG_TPU_DSA_INDEX_KV_SHARD` plus the libtpu flag (Correctness) | unset / unset / unset / unset / 1 | splits the pooled indexer's key cache like the latent cache, freeing HBM for the KV pool; needs split 8 |
| token split | env `SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT` | unset / unset / 1 / 1 / 1 | each selected cache row gathered by the rank that holds it, plus an all-to-all of queries and outputs |
| device KV pool | `max_total_tokens` (cap; HBM may bound it lower), `mem_fraction_static` 0.9 | 3,407,872 / 3,407,872 / 9,437,184 / 9,437,184 / 14,155,776 (boot sizes in Per point) | device KV tokens; HBM is shared with the KDA slots |
| device KDA slots | `max_mamba_cache_size` | 74 / 384 / 800 / 1,400 / 1,200 | device KDA checkpoints, sized for the conversations AgentX keeps cached |
| checkpoint level | `hybrid_cache_checkpoints_level` | 3 at every point | a KDA checkpoint at every prefill chunk (level 2, the default: only at branch points and request ends) |
| decode KDA checkpoints | `mamba_track_interval` (decode tokens per KDA checkpoint; default 256) | unset / unset / unset / unset / 1,048,576 | 1,048,576: no KDA checkpoints during decode |
| host KV tier | `hicache_ratio` (x device pool) | 2 / 2 / 1.5 / 1.7 / 1.2 | host KV tokens, for host-tier prefix hits |
| host KDA states | env `SGLANG_TPU_HICACHE_STATE_RATIO` (x device KDA slots; unset = `hicache_ratio`) | unset / unset / unset / 1.429 / 2.5 | host KDA states, for host-tier hits after idle gaps |
| max running | `max_running_requests`, `torch_compile_max_bs` | 2 / 4 / 8 / 16 / 32 | decode batch cap |
| decode buckets | `tpu_compile_bs` | 1..max running at c1-c16; c32 1..20, 24, 28, 32 | one compiled decode program per batch size; each costs host RAM |
| MTP draft length | `speculative_num_steps` = K (`speculative_num_draft_tokens` K + 1, `speculative_eagle_topk` 1); env `SGLANG_TPU_SPEC_STEPS_BY_RUNNING` | `1:7,2:3` / 3 / `1:7,2:3` / 3 / 3 | draft steps per decode step; `1:7,2:3` = 7 at 1 running request, 3 from 2 |
| simulated acceptance | env `SGLANG_SIMULATE_ACC_LEN` (`SGLANG_TPU_SPEC_SIMULATE_ACC_LEN_BY_STEPS` per K), `SGLANG_SIMULATE_ACC_METHOD`, `SGLANG_SIMULATE_ACC_TOKEN_MODE` | 3.93 / 3.0 / 3.93 / 3.0 / 3.0; `match-expected`; `real-draft-token` | accepted length = InferenceX's golden AL of K, on real draft tokens |
| MoE ring kernel | env `SGLANG_TPU_ENABLE_MOE_RING_KERNEL` | 1 at every point | each distinct FP8 expert streamed once through a VMEM ring, 2..256 tokens |
| megakernel | env `SGLANG_TPU_GLM_FLASH_MEGAKERNEL` | 0 at every point | off: decode runs on the model path |
| prefill chunk | `chunked_prefill_size`, `max_prefill_tokens`, `tpu_compile_tokens` | 4096, 4096, [1024, 2048, 3072, 4096] at every point | prefill chunk size and the compiled prefill programs |
| top-k width buckets | env `SGLANG_TPU_SCTOPK_LONG_ROWS` | on by default (0 disables) | exact; prefill top-k reads each row's prefix in place |
| bf16 split-cache merge | env `SGLANG_TPU_DSA_MLA_KV_SHARD_BF16_MERGE=1` | unset / unset / unset / 1 / 1 | bf16 instead of fp32 reduce-scatter for the split cache's prefill merge |

## InferenceX notes

1. **Acceptance in the launch config.** InferenceX says not to hard-code acceptance in
   srt-slurm recipes or launchers, because its connector injects it. This TPU launch has no
   connector, so the run JSONs set the `SGLANG_SIMULATE_ACC_*` variables (and the per-step
   map at c1/c8). If the submission goes through InferenceX's srt driver, drop them.
2. **Cache flush.** InferenceX says "do not flush caches or reuse a live server across
   concurrency points". Each point had its own fresh server, with one `/flush_cache` after
   boot, before the client started.
3. **Draft steps per decode step.** c1 and c8 change the MTP step count with the running
   count (7 at 1 running, 3 from 2), each at its golden AL. SemiAnalysis confirmed that
   changing the draft length with the running count is allowed.

## Files

- [`agentx.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/agentx.py): the server config of every Results point, one profile per
  concurrency on top of `mtp.py` (`--conc <cc>` prints it; `get_config(cc)` serves
  `launch_server --concurrency`), the same interface as Muse Spark's AgentX recipe
  (`muse_spark_ax/agentx.py`).
- [`runs/`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs): launch JSONs; the Results table links each point's.
- [`summarize.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/summarize.py): computes the metrics of one client output dir.
- [`launch_agentx.sh`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/launch_agentx.sh): launches one Results point's `agentx.py` profile with
  its caches on a PVC (Replicate, Server).
- [`check_recipes.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/check_recipes.py): diffs each `agentx.py` profile against its run JSON.

Chart: [GLM Flash AgentX Curves](https://claude.ai/artifact/JnGfziw2GBSqDx5LCZArVf)
