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

Chart v47 (artifact version `1795443917-f7f1`), read 2026-11-23.

Real weights, one 1 h window per point.
- c1-c16: this branch's integrated tree (`b38e84a11e`, job 36d0): the final curve plus the
  prefill and decode commits and prefill-decode interval 16 (Optimizations).
- c32: the integrated tree plus the DSA index-key cache split (`f977f22c30`, job f909), with
  the freed HBM spent on a larger device KV pool and libtpu's SparseCore all-gather offload
  off (Accuracy), and the decode-time KDA checkpoints off, with 1,200 KDA slots and host
  tier 1.2x (c32 KDA retention below). It is over the latency limit.

| cc | tok/s/chip | M tokens/$ | 1/ITL p50 / p90 | TTFT p90 s | E2E latency p90 / limit s | prefix hit / ideal % | vs final curve: tok/s / 1/ITL p90 | run JSON |
|---|---|---|---|---|---|---|---|---|
| 1 | 6,452 | 22.6 | 460.0 / 414.4 | 0.86 | 11.8 / 24.0 | 98.17 / 98.29 | +0.7% / -0.3% | [`runs/R1b_c1_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/R1b_c1_real_l3_ring_full.json) |
| 4 | 8,876 | 31.0 | 420.0 / 321.5 | 0.79 | 10.1 / 20.4 | 97.67 / 97.88 | +0.2% / +3.0% | [`runs/C4u_c4_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/C4u_c4_real_l3_ring_full.json) |
| 8 | 14,920 | 52.1 | 327.3 / 208.3 | 0.97 | 10.2 / 19.6 | 97.48 / 97.78 | +0.6% / +11.0% | [`runs/K2ts_c8_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/K2ts_c8_real_l3_ring_full.json) |
| 16 | 26,360 | 92.1 | 201.5 / 118.5 | 2.40 | 13.9 / 20.3 | 97.23 / 97.60 | +4.1% / +28.0% | [`runs/C16_S2ts_kda1400_bf16_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/C16_S2ts_kda1400_bf16_real_l3_ring_full.json) |
| 32 | 42,923 | 150.0 | 112.6 / 53.2 | 2.24 | 29.6 / 28.2 **over** | 97.11 / 97.69 | +21.5% / +52.0% | [`runs/R10_c32_mr32_combo_real_l3_ring_full_ixsplit.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/R10_c32_mr32_combo_real_l3_ring_full_ixsplit.json) |

- Latency limit: a point's p90 E2E latency may not exceed GB300's p90 at the same concurrency
  plus 10 s. GB300 p90 (InferenceX run 42026173061, the chart's GB300 line): c1 14.0, c4 10.4,
  c8 9.6, c16 10.3, c32 18.2 s. c32 is over it (29.6 vs 28.2 s, 1.4 s over; 36.7 s without
  the split). At interval 0 prefill stalls are 10.4 s of c32's p90-band latency (30.1 s).
- Index-key cache split (c32), to enable: set `SGLANG_TPU_DSA_MLA_KV_SHARDS=8` (the
  exact prefill exchange needs the latent cache split by 8),
  `SGLANG_TPU_DSA_INDEX_KV_SHARD=1` and the libtpu flag (Accuracy), then resize the pools.
  Spend the freed HBM on device KV tokens (`max_total_tokens`) and set `hicache_ratio` so host
  KV bytes per rank stay at or above the unsplit run's. KV bytes per token per rank: 3,840 at
  latent split 2, 1,536 at split 8, 864 at split 8 with the index split.
- c32 KDA retention (job f909): `mamba_track_interval` 1,048,576 (no KDA checkpoints during
  decode; the default takes one every 256 tokens), `max_mamba_cache_size` 800 -> 1,200 and
  `hicache_ratio` 2.37 -> 1.2, device KV pool unchanged. Against the same config without
  them (job 7605): 42,923 vs 41,958 tok/s/chip, 1/ITL p90 53.2 vs 50.7, E2E p90 29.6 vs
  31.9 s. Turning the decode checkpoints off gives most of it alone (29.8 s, +1.6% tok/s):
  they were not reused, and their KDA evictions took cached KV along (3.35M -> 0.39M tokens
  per hour); device prefix hit 93.7 -> 96.9%. The host tier trades KV for KDA states: 17.0M
  KV tokens (14.7 GB per rank, below the unsplit run's 29.0 GB, unlike the rule above) and
  3,001 KDA states (57.3 GB per rank).
- c16 stays unsplit. With the split and the flag (job 7605) it measured 26,412 tok/s/chip and
  1/ITL p90 125.2 (unsplit 26,360 and 118.5), but a lower E2E-normalized p90 (60.1 vs 66.0) and
  a higher pod memory peak (985 vs 874 GB). Its JSON is kept as
  [`runs/C16_S2ts_kda1400_bf16_real_l3_ring_full_ixsplit.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/C16_S2ts_kda1400_bf16_real_l3_ring_full_ixsplit.json).
- c8 runs without the split, as measured. Its earlier divergence with the split (job 01cc)
  was the fault in Accuracy.
- 1/ITL = output tok/s per user = 1000 / inter-token latency (ms) at that percentile. The
  chart plots tokens/$ against 1/ITL p90.
- Tokens/$ = tok/s/chip x 3600 / $1.03 per chip-hour (InferenceX's TPU7x rate), in millions.
- tok/s/chip counts prompt and completion tokens, cached prompt tokens included.
- Ideal = aiperf's theoretical prefix-cache hit, the figure InferenceX publishes.
- vs final curve: against the same point's `*_real_l3_ring.json` run (`f568acf704`, job
  e2df). Against dummy weights at the same config, c1-c8 tok/s/chip is +3.7 / -0.9 / +1.5%.
- c1 and c4 are flat because one request runs 98% / 83% of the time: the interval and the
  batch-side kernels have little to act on.
- Every point: 0 recompiles after ready, 0 OOM, 0 scheduler exceptions, 1-2 failed requests
  (aiperf `InvalidInferenceResultError`). One window per point; noise is about ±3%. c32
  checked TP-rank agreement every 64 verify steps: 0 mismatches.
- Hosts: one tpu7x host each, libtpu 0.0.50.
  - c1-c16: one host of a 2x2x2 dev pod, torch_tpu 0.2.0.
  - c32: a tpuq pod, torch_tpu 0.2.1.dev.

## Optimizations

Per-point values are in Per point and Tunable configs. Base kernels on in every run JSON:
fused mHC, streaming MoE, FP8 small-M linear, q-block sparse MLA prefill.

Decode:
- MoE ring kernel: FP8 expert weights stream once through a VMEM ring.
- Adaptive MTP: 7 draft steps at 1 running, else 3 (c1/c8).
- Simulated-acceptance accept step compiled into the decode program.
- No eager ops in the decode step.
- Sort-free router top-k and ring routing, bit-exact.
- Split-cache decode attends only each row's own candidates (c8-c32).
- StreamIndex top-k: one DMA per run of consecutive KV pages.
- Megakernel off: model path plus ring kernel is faster above 1 request.

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
- DSA index-key cache split by 8, row-slice prefill exchange (c32).
- Freed HBM goes to a larger device KV pool (c32).
- Host KV tier: raiden, `page_first_fused`, `write_through`.
- Host tier also keeps KDA states.
- Hybrid-cache checkpoints level 3: a KDA checkpoint at every prefill chunk.
- KDA slots sized for the conversations AgentX keeps cached.
- Raw-ring pooled indexer; page table in HBM.

Scheduling:
- Prefill-decode interval 16 at c1-c16 (0 at c32).
- Decode buckets: every batch size up to c16, sparse at c32.

## Accuracy

Index-key split fix (job 7605). XLA's SparseCore-offloaded all-gathers returned wrong data in
the split decode graph. Under simulated acceptance whole batches turned to NaN and zero drafts,
the same on every rank. The earlier split numbers (c16 26,888, c32 47,243 tok/s/chip) were
measured on that wrong output and are replaced. Two fixes:
- `LIBTPU_INIT_ARGS += --xla_tpu_enable_sparse_core_collective_offload_all_gather=false`, in
  the split run JSONs and `agentx.py`. The c32 window ran with it.
- With real sampling the TP vocab-logits all-gather still lost rank 0's block on the peers,
  so TP ranks sampled different tokens. `TpuLogitsGatherer` and the draft vocab gather now use
  an exact all-reduce of zero-padded per-rank slots under the direct tracer
  (`sglang_tpu/all_gather_via_all_reduce.py`). This does not touch the timed windows, which use
  simulated acceptance (argmax path).

GSM8K, 200 questions, sgl-eval 0.1.3: temperature 1.0, top_p 0.95, thinking, max_tokens
16000. Engine code `f977f22c30`, plus the logits fix where marked. Real acceptance: the run
JSON without the `SGLANG_SIMULATE_ACC_*` variables (`agentx.py --eval`), with a TP-rank check
every 64 verify steps.

| config | flag | logits fix | GSM8K |
|---|---|---|---|
| c8 (unsplit) | no | no | 98.0% |
| c32 unsplit ([`R10_c32_mr32_combo_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/R10_c32_mr32_combo_real_l3_ring_full.json)) | no | no | 96.0% |
| c32 split | yes | no | TP ranks diverged at verify 64 |
| c32 split | yes | yes | 98.5% |
| c8 split, K=7 (not a Results config) | yes | yes | 98.5% |

- Base tree (final curve): GSM8K-200 (`few_shot_gsm8k`, parallel 8) on `f568acf704` with the c8 config,
  simulated acceptance off: 0.930. Gate 0.88; the ring-kernel-only tree scored 0.925. With
  that ring-only run's exact config, its greedy output equals the ring-only tree's on 8/8
  prompts (256 tokens each).

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
- Host memory: the pod peaks at 532-906 GB of the node's ~987 GB allocatable (table below).
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
| c32 | [`R10_c32_mr32_combo_real_l3_ring_full_ixsplit.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs/R10_c32_mr32_combo_real_l3_ring_full_ixsplit.json) | 32 | 3 / 3.00 | 32 | 1, 2, 4, 8, 12..32 by 4 | 14,155,776 | 1,200 | 8 / on | 17.0M / 3,001 | 906 GB | 1,456 |

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
- Boot = launch to ready: the NFS weight load (6-23 min, slowest rank) plus a cold compile
  cache.
- c1: the JSON compiles with SPMD, which needs torch_tpu 0.2.1.dev or later. The measured c1
  ran on a dev pod (torch_tpu 0.2.0) with `use_spmd_compile` false; that changes only how
  compiles are distributed. The other points compile per rank.
- Same at every point:
  - context 1,048,576, page 128;
  - fp8 weights and fp8_e4m3 KV;
  - `mem_fraction_static` 0.9, ep 1;
  - prefill programs 1k/2k/3k/4k with 4096-token chunks;
  - `hybrid_cache_checkpoints_level` 3;
  - MoE ring kernel on, megakernel off, direct tracer;
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
- c32 also sets `SGLANG_TPU_DSA_INDEX_KV_SHARD=1` (index-key cache split, Results), appends
  `--xla_tpu_enable_sparse_core_collective_offload_all_gather=false` to `LIBTPU_INIT_ARGS`
  (Accuracy) and sets `mamba_track_interval` 1,048,576: no KDA checkpoints during decode
  (Results).

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
  phase counts. No warm-up prompts or prefill probe are sent; the previous points sent both.
- Per point: boot, then 62-77 min of client time (warm-up, the 1 h window, drain).

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
- Each run JSON is built in steps:
  - `*_real.json`: the point's dummy-weight config with real weights (no `load_format`,
    checkpoint prefetch, page cache dropped after load, 2400 s unbalanced-load timeout),
    token mode `real-draft-token` and request time stats;
  - `*_real_l3.json`: plus `hybrid_cache_checkpoints_level` 3;
  - `*_real_l3_ring.json`: plus the ring kernel. c1/c4/c8 also get the draft-step changes;
  - `*_real_l3_ring_full.json`: plus the five flags (Per point); c1-c16 also `prefill_decode_interval` 16,
    c32 interval 0;
  - `*_real_l3_ring_full_ixsplit.json` (c32, and c16's unused split): the `*_full` JSON
    plus `SGLANG_TPU_DSA_INDEX_KV_SHARD=1`, the libtpu flag and resized pools
    (`max_total_tokens` / `hicache_ratio`): c32 14,155,776 / 1.2; c16 25,000,000 / 1.71,
    also `SGLANG_TPU_DSA_MLA_KV_SHARDS` 2 -> 8. c32 also sets `max_mamba_cache_size` 1,200
    and `mamba_track_interval` 1,048,576 (job f909).
- [`launch_agentx.sh`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/launch_agentx.sh) launches the `agentx.py` profiles. The recipes
  ([`agentx_cc<N>.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx), [`check_recipes.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/check_recipes.py)) still build and check
  the previous dummy-weight points' configs.

## Pending

- c32 within its latency limit (prefill stalls).
- The logits-gather fix's cost with real sampling (the timed windows do not use it).
- InferenceX question 3 below (draft steps per decode step).

## Tunable configs

Values as the run JSONs set them, c1 / c4 / c8 / c16 / c32; unset = server default.
Evidence is a job ID; 1/ITL numbers are p90. Evidence before job 301e is on dummy weights.

| knob | where set | c1 / c4 / c8 / c16 / c32 | what it trades | evidence |
|---|---|---|---|---|
| prefill-decode interval | `prefill_decode_interval` | 16 / 16 / 16 / 16 / 0 | N decode steps between prefill batches: higher 1/ITL, later TTFT; a fixed N starves admission at high concurrency | integrated tree, N=16: about +17% 1/ITL at c16 on top of the kernels, little effect at c1/c4 (36d0). Earlier: c16 N=16, split 8 before token split: 1/ITL 84.9 -> 91.3, tok/s -0.5%, TTFT p90 2.06 -> 2.95 s (ff06). c32 N=16: 1/ITL +38%, tok/s -1.8%, TTFT p90 12.7 -> 17.8 s; N=64 worse (754a). Muse Spark c10, N=64 vs its shipped profile: 1/ITL 169 vs 116 (Haolin's perf KB, F-0069..71) |
| latent-cache split | env `SGLANG_TPU_DSA_MLA_KV_SHARDS` | unset (1) / unset / 2 / 2 / 8 | fewer shards: cheaper prefill merge, more KV bytes per rank (host KV 30,720 / 18,432 / 12,288 B/token/pod at 2 / 4 / 8; the split-2 pool is HBM-bound below the 9.4M cap) | c16 split 2 vs 8: 1/ITL +18.9%, tok/s +6.7%; split 4: 1/ITL 98.1 (8c26). c32 at max running 32, split 4 vs 8: 38,807 vs 38,149 tok/s, 1/ITL 41.2 vs 39.9 (3ca4, 175a). c4 unsplit vs split 8: 1/ITL 337.7 vs 287.5 (6076) |
| index-key cache split | env `SGLANG_TPU_DSA_INDEX_KV_SHARD` plus the libtpu flag (Accuracy) | unset / unset / unset / unset / 1 | the pooled indexer's key cache split like the latent cache: HBM freed for the KV pool; exact prefill exchange needs split 8 | with the flag and the pools resized: c32 38,539 -> 41,958 tok/s, 1/ITL 42.7 -> 50.7, E2E p90 36.7 -> 31.9 s (c49c, 7605); c16 26,360 -> 26,412, 1/ITL 118.5 -> 125.2, E2E-normalized p90 66.0 -> 60.1 (also split 2 -> 8; 36d0, 7605). Without the flag the output was wrong (01cc, 83cb numbers void) |
| token split | env `SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT` | unset / unset / 1 / 1 / 1 | each selected cache row gathered by one rank, plus an all-to-all of queries and outputs | c16 split 8: 1/ITL 83.7 -> 93.5, tok/s 23,207 -> 24,407; c8: 1/ITL 175.0 -> 188.1; neutral on the unsplit c1/c4 (fee5) |
| device KV pool | `max_total_tokens` (cap; HBM may bound it lower), `mem_fraction_static` 0.9 | 3,407,872 / 3,407,872 / 9,437,184 / 9,437,184 / 14,155,776 (boot sizes in the per-point table) | HBM for KV tokens vs device KDA slots | c16 (8.64M -> 5.66M tokens, 800 -> 1,400 slots): state evictions 6,399 -> 2,192 per h, tok/s and 1/ITL within noise (8c26). c32 pool sized for the index-key split (Results; 01cc) |
| device KDA slots | `max_mamba_cache_size` | 74 / 384 / 800 / 1,400 / 1,200 | HBM for KDA checkpoints vs KV pool; sized for the conversations AgentX keeps cached, not from max running | c16 1,100: 26,004 / 1/ITL 111.6; 1,400: hit gap 0.36 vs 0.50 pt (8c26). c32 1,600: no gain (98c5). c32 with decode checkpoints off, 800 -> 1,200 and host tier 2.37 -> 1.2: 42,618 -> 42,923 tok/s, E2E p90 29.8 -> 29.6 s, within noise (f909) |
| checkpoint level | `hybrid_cache_checkpoints_level` | 3 at every point | KDA checkpoints at every prefill chunk vs only at branch points and request ends (level 2, the default) | real weights, level 2 -> 3: hit gap 1.67-3.80 -> 0.13-1.17 pt, tok/s +3 to +21% (301e) |
| decode KDA checkpoints | `mamba_track_interval` (decode tokens per KDA checkpoint; default 256) | unset / unset / unset / unset / 1,048,576 | checkpoints during decode vs KDA slots and the KV evicted with their states | c32 off: 41,958 -> 42,618 tok/s, 1/ITL p90 50.7 -> 51.4, E2E p90 31.9 -> 29.8 s; KV evicted with its state 3.35M -> 0.43M tokens per hour (f909) |
| host KV tier | `hicache_ratio` (x device pool) | 2 / 2 / 1.5 / 1.7 / 1.2 | host-hit reach vs host RAM | c32 at max running 32 fit with 2.0 (R6c, 175a) and 1.33 at split 4 (R7, 3ca4). c32 at 1.2 with 1,200 KDA slots: 17.0M host KV tokens (14.7 vs 29.0 GB per rank unsplit), host RAM to 3,001 KDA states; host-tier hit 0.27%, total 97.11% (f909) |
| host KDA states | env `SGLANG_TPU_HICACHE_STATE_RATIO` (x device KDA slots; unset = `hicache_ratio`) | unset / unset / unset / 1.429 / 2.5 | host states (~19.6 MB/rank each) give the host-tier hits after idle gaps vs host RAM | c16 2,001 states reach ~15 min; ratio 1.5 would cut them to 1,201, ~9 min (8c26) |
| max running | `max_running_requests`, `torch_compile_max_bs` | 2 / 4 / 8 / 16 / 32 | decode batch vs step time | c32 at 32 vs 16, both with the prefill patches: 40,220 vs 39,770 tok/s, 1/ITL 43.5 vs 59.6 (5196, 15c4); before the patches tok/s flat, 1/ITL -24 to -27% (175a, 3ca4) |
| decode buckets | `tpu_compile_bs` | 1..max running at c1-c16; c32 10 buckets | unpadded decode steps vs host RAM per program | c32 at max running 32: buckets 1..32 ran out of host RAM at boot, 1..16 + 20..32 was OOM-killed in the window, 10 buckets ran (175a) |
| MTP draft length | `speculative_num_steps` = K (`speculative_num_draft_tokens` K + 1, `speculative_eagle_topk` 1); env `SGLANG_TPU_SPEC_STEPS_BY_RUNNING` | `1:7,2:3` / 3 / `1:7,2:3` / 3 / 3 | more tokens per step at low batch vs more verify work at high batch | per token, K=7 vs K=3 with the ring: c8 -2.2 to -2.7% at 1 running, +5 to +14% at 3-8; c4 tie at 1, +6.9 to +13.6% at 2-4 (5795). Dummy, at the golden AL: c4 K=3 vs 8 1/ITL 282.5 vs 236.4; c8 175.0 vs 148.8 (2460) |
| simulated acceptance | env `SGLANG_SIMULATE_ACC_LEN` (`SGLANG_TPU_SPEC_SIMULATE_ACC_LEN_BY_STEPS` per K), `SGLANG_SIMULATE_ACC_METHOD`, `SGLANG_SIMULATE_ACC_TOKEN_MODE` | 3.93 / 3.0 / 3.93 / 3.0 / 3.0; `match-expected`; `real-draft-token` | not a perf knob: AL = `GOLDEN_AL[K]`, never 3.5. Token mode: same accepted count as `fixed`; real tokens cost more per step | before the compiled accept, `real-draft-token` vs `fixed`: +3.1 / +4.4 ms per step at 8 / 16 running (290a) |
| MoE ring kernel | env `SGLANG_TPU_ENABLE_MOE_RING_KERNEL` | 1 at every point | each distinct FP8 expert streamed once through a VMEM ring, 2..256 tokens | decode step -29% at 8 / 32 running; c32 +20.9% (78d8) |
| megakernel | env `SGLANG_TPU_GLM_FLASH_MEGAKERNEL` | 0 at every point | one kernel per decode step vs the model path | vs model + ring: -3% at 1 request with K=3, +15 / +45% at 2 / 4, slower everywhere with K=7 (1454) |
| prefill chunk | `chunked_prefill_size`, `max_prefill_tokens`, `tpu_compile_tokens` | 4096, 4096, [1024, 2048, 3072, 4096] at every point | bigger chunks: less fixed cost per token vs longer decode stalls, more programs | 8192: -2.2% per token, stalls ~2x, +18 programs/rank (dcd8). c16 at 2048: 1/ITL 73.4 vs 83.7 (6076) |
| expert parallel | `ep_size` (`tp_size` 8) | 1 at every point | expert-TP vs expert-parallel MoE | untested; judged not worth it for c32+ |
| top-k width buckets | env `SGLANG_TPU_SCTOPK_LONG_ROWS` | on by default (0 disables) | exact; faster prefill top-k | with the bf16 merge at split 8 + token split: 4k chunk 0.2432 -> 0.2204 s at ctx 0, 0.2795 -> 0.2393 s at 256k (2507) |
| bf16 split-cache merge | env `SGLANG_TPU_DSA_MLA_KV_SHARD_BF16_MERGE=1` | unset / unset / unset / 1 / 1 | bf16 instead of fp32 reduce-scatter for the prefill merge | as above; GSM8K-1319 0.9060 on vs 0.9037 off (2507). c32 with both: 39,770 vs 38,358, 1/ITL 59.6 vs 54.4 (15c4) |

- Host RAM: resident host memory grows ~34-39 MB per compiled program per rank (3ca4), so
  decode buckets cost host RAM. When a config does not fit, shrink the host tier first and
  keep the buckets: host-tier hits are fine as long as nothing is evicted or thrashing.

## Open InferenceX questions

1. **Acceptance in the launch config.** InferenceX says not to hard-code acceptance in
   srt-slurm recipes or launchers, because its connector injects it. This TPU launch has no
   connector, so the run JSONs set the `SGLANG_SIMULATE_ACC_*` variables (and the per-step
   map at c1/c8). If the submission goes through InferenceX's srt driver, drop them.
2. **Cache flush.** InferenceX says "do not flush caches or reuse a live server across
   concurrency points". Each point had its own fresh server, with one `/flush_cache` after
   boot, before the client started.
3. **Draft steps per decode step.** c1 and c8 change the MTP step count with the running
   count (7 at 1 running, 3 from 2), each at its golden AL. InferenceX gives one golden AL
   per step count and does not say whether the step count may change per decode step.

## Files

- [`agentx.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/agentx.py): the server config of every Results point, one profile per
  concurrency on top of `mtp.py` (`--conc <cc>` prints it; `get_config(cc)` serves
  `launch_server --concurrency`), the same interface as Muse Spark's AgentX recipe
  (`muse_spark_ax/agentx.py`).
- [`runs/*_real_l3_ring_full.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs) (c1-c16) and [`runs/*_real_l3_ring_full_ixsplit.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs)
  (c32): the launch JSONs of the Results points.
- [`runs/*.json`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/runs): the launch JSONs of all measured runs, which are the measurement record.
- [`summarize.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/summarize.py): computes the metrics of one client output dir.
- [`launch_agentx.sh`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/launch_agentx.sh): launches one Results point's `agentx.py` profile with the
  caches and env of jobs 36d0 (c1-c16) and 7605 (c32) (Replicate, Server).
- [`_base.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/_base.py), [`agentx_cc<N>.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx), [`check_recipes.py`](benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/check_recipes.py): build and check the previous
  dummy-weight points' configs; `check_recipes.py` also diffs each `agentx.py` profile against its
  run JSON. [`benchmark/tpu/test_launch_server.py`](benchmark/tpu/test_launch_server.py) runs its dummy-weight checks.

Chart: [GLM Flash AgentX Curves](https://claude.ai/artifact/JnGfziw2GBSqDx5LCZArVf)
