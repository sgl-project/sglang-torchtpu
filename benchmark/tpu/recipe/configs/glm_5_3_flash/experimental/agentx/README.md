# GLM-5.3-Flash AgentX on TPU

GLM-5.3-Flash (fp8) on one tpu7x host: 2x2x1, 4 chips = 8 TensorCores, tp8, 1M context.
Benchmark: AgentX (`agentx-harness==1.0.14`, SemiAnalysis cc traces, 1 h window per
concurrency). MTP verify at InferenceX's golden acceptance: K=3 steps -> AL 3.00, K=7 ->
AL 3.93. One run JSON in `runs/` per point.

## Results

Real weights, this branch at `f568acf704`, one 1 h window per point (job e2df).

| cc | tok/s/chip | M tokens/$ | 1/ITL p50 / p90 | TTFT p90 s | E2E latency p90 s | prefix hit / ideal % | vs 301e | vs dummy | run JSON |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6,406 | 22.4 | 456.3 / 415.5 | 0.81 | 12.3 | 98.16 / 98.28 | +7.0% | +3.0% | `runs/R1b_c1_real_l3_ring.json` |
| 4 | 8,855 | 31.0 | 418.2 / 312.1 | 0.76 | 10.3 | 97.68 / 97.89 | +2.0% | -1.1% | `runs/C4u_c4_real_l3_ring.json` |
| 8 | 14,838 | 51.9 | 315.2 / 187.7 | 0.73 | 10.2 | 97.51 / 97.81 | +5.6% | +0.9% | `runs/K2ts_c8_real_l3_ring.json` |
| 16 | 25,327 | 88.5 | 181.2 / 92.6 | 1.17 | 15.1 | 97.14 / 97.52 | +25.3% | -3.9% | `runs/C16_S2ts_kda1400_bf16_real_l3_ring.json` |
| 32 | 35,338 | 123.5 | 73.2 / 35.0 | 3.26 | 43.1 | 96.82 / 97.63 | +22.5% | -12.1% | `runs/R10_c32_mr32_combo_real_l3_ring.json` |
| 48 | 37,460 | 130.9 | 40.5 / 23.1 | 3.55 | 70.2 | 96.16 / 97.23 | +17.2% | -11.3% | `runs/R9_c48_mr48_real_l3_ring.json` |

- 1/ITL = output tok/s per user = 1000 / inter-token latency (ms) at that percentile. The
  chart plots tokens/$ against 1/ITL p90.
- Tokens/$ = tok/s/chip x 3600 / $1.03 per chip-hour (InferenceX's TPU7x rate), in millions.
- tok/s/chip counts prompt and completion tokens, cached prompt tokens included.
- Ideal = aiperf's theoretical prefix-cache hit, the figure InferenceX publishes.
- vs 301e: tok/s/chip against the previous real-weight curve (`*_real_l3.json`, job 301e:
  5,988 / 8,678 / 14,049 / 20,213 / 28,854 / 31,960). 1/ITL p90 rose +23 / +64 / +46 / +42 /
  +23 / +20%. TTFT p90 and E2E latency p90 are equal or lower at every point.
- vs dummy: tok/s/chip against the same run config on dummy weights: the Previous table's
  point at c1/c4/c8/c48, `C16_S2ts_kda1400.json` at c16 (26,359; no bf16 merge), R10 at c32
  (40,220, job 5196).
- c4 and c8 are arrival-bound: at c4 one request runs 79% of the time, at c8 47%.
- Every point: 0 recompiles after ready, 0 OOM, 0 scheduler exceptions, 1-2 failed requests
  (aiperf `InvalidInferenceResultError`). One window per point; noise is about ±3%.
- Hosts:
  - c48: tpuq slice, image `sglang-tpu@sha256:1affcdb…` (jax 0.11.2, libtpu 0.0.50,
    torch_tpu 0.2.1.dev20261028). It ran at `c3157c5ff1`; `f568acf704` only changes the
    c1/c4/c8 JSONs.
  - c1-c32: one host of a 2x2x2 dev pod, image `sha256:5448af1b…` (libtpu 0.0.50,
    torch_tpu 0.2.0). The registry's cleanup policy has since deleted that image.

### Previous (dummy weights)

| cc | run JSON | tok/s/chip | M tokens/$ | 1/ITL p50 / p90 | TTFT p90 s | prefix hit / ideal % |
|---|---|---|---|---|---|---|
| 1 | `runs/R1b_c1.json` | 6,220 | 21.7 | 464.1 / 421.5 | 1.32 | 98.12 / 98.25 |
| 4 | `runs/C4u_c4.json` | 8,958 | 31.3 | 446.5 / 337.7 | 1.37 | 97.71 / 97.92 |
| 8 | `runs/K2ts_c8.json` | 14,705 | 51.4 | 320.9 / 217.7 | 1.16 | 97.51 / 97.81 |
| 16 | `runs/C16_S2ts.json` | 26,037 | 91.0 | 209.0 / 110.9 | 1.62 | 97.07 / 97.57 |
| 32 | R8_combo, job 15c4 | 39,770 | 139.0 | 99.0 / 59.6 | 12.67 | 96.83 / 97.60 |
| 48 | R9, job ccd1 | 42,219 | 147.6 | 47.2 / 26.8 | 2.65 | 96.25 / 97.23 |

- The chart's dummy line: older trees, token mode `fixed`, harness 1.0.12.
- c32 and c48 ran on the prefill-patch tree. c32's config is `R8_c32_combo.json` (commit
  `4ea6597f8b`, not in `runs/`): `R5_c32_ts.json` plus the bf16 split-cache merge. c48's is
  `runs/R9_c48_mr48_real.json` with dummy weights and token mode `fixed`.

## What changed since 301e

- MoE ring kernel for decode (`SGLANG_TPU_ENABLE_MOE_RING_KERNEL=1`, job 78d8): decode step
  -29% at 8 and 32 running; c32 +20.9%.
- Compiled real-draft-token accept (290a): free, -0.3 to -0.5 ms per step.
- Zero eager ops in the decode step (c466): -1 to -2% per step.
- Sort-free router top-k and ring route (0e38): -1 to -4% per step, bit-exact.
- Adaptive MTP (5795): draft steps by running count. c1 and c8 run `1:7,2:3`. c4 is now
  fixed K=3 (was 7), because K=7 costs 7-14% more per token at 2-4 running.
- Megakernel stays off (1454). Against model + ring it is -3% at 1 request with K=3,
  +15% / +45% at 2 / 4 requests, and slower at every request count with K=7.

## Accuracy

- GSM8K-200 (`few_shot_gsm8k`, parallel 8) on `f568acf704` with the c8 config, simulated
  acceptance off: 0.930. Gate 0.88; the ring-kernel-only tree scored 0.925.
- With that ring-only run's exact config, this tree's greedy output equals the ring-only
  tree's on 8/8 prompts (256 tokens each).

## Replicate

### Requirements

- Host: one tpu7x 2x2x1 host (tpu7x-8), with one of the images above, plus:
  - `libibverbs.so.1` (`apt-get install libibverbs1`) for the raiden host-tier transfers;
  - `transformers==5.19.0 blake3 timm==1.0.25`, which the images lack:
    `pip install --no-deps --target $DEPS ...`, then `$DEPS` first on `PYTHONPATH`.
- Weights: target and MTP draft at `/weights/GLM-5.3-Flash`.
- Caches: put them on a PVC, never on the container overlay. One directory per point holds
  the compile caches, `TMPDIR`, `SGLANG_CACHE_DIR` (`~/.cache/sglang` links to it) and
  `PROMETHEUS_MULTIPROC_DIR`.
- Core dumps: `ulimit -c 0`, because one rank's core dump fills the disk.
- Host memory: the pod peaks at 553-984 GB of the node's ~987 GB allocatable (table below).
  Keep `/dev/shm` and other processes off the node.
- Client: `agentx-harness==1.0.14` in its own venv, on the same host. The trace
  `semianalysis_cc_traces_weka_062126` (HF `semianalysisai/cc-traces-weka-062126`, 393
  entries) must be pre-fetched into `HF_HOME`. This is the full uncapped corpus InferenceX
  uses for GLM.

### Per point

Run one fresh server per point. Stop it (SIGTERM) before the next point.

| point | run JSON | client `CC` | MTP K / AL | max running | decode buckets | KV pool at boot (cap) | KDA slots | latent split / token split | host tier: KV tokens / KDA states | pod memory peak | boot s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c1 | `R1b_c1_real_l3_ring.json` | 1 | `1:7,2:3` / 3.93, 3.00 | 2 | 1..2 | 3,407,872 | 74 | none / off | 6.82M / 149 | 553 GB | 1,125 |
| c4 | `C4u_c4_real_l3_ring.json` | 4 | 3 / 3.00 | 4 | 1..4 | 3,407,872 | 384 | none / off | 6.82M / 769 | 690 GB | 1,206 |
| c8 | `K2ts_c8_real_l3_ring.json` | 8 | `1:7,2:3` / 3.93, 3.00 | 8 | 1..8 | 8,604,416 (9,437,184) | 800 | 2 / on | 12.9M / 1,201 | 885 GB | 1,255 |
| c16 | `C16_S2ts_kda1400_bf16_real_l3_ring.json` | 16 | 3 / 3.00 | 16 | 1..16 | 5,643,136 (9,437,184) | 1,400 | 2 / on | 9.59M / 2,002 | 929 GB | 2,196 |
| c32 | `R10_c32_mr32_combo_real_l3_ring.json` | 32 | 3 / 3.00 | 32 | 1, 2, 4, 8, 12..32 by 4 | 9,437,184 | 800 | 8 / on | 18.9M / 2,001 | 917 GB | 1,425 |
| c48 | `R9_c48_mr48_real_l3_ring.json` | 48 | 3 / 3.00 | 48 | 1..16, 20..48 by 4 | 16,000,000 | 800 | 8 / on | 18.0M / 2,001 | 984 GB | 2,116 |

- K = `speculative_num_steps`, giving K + 1 draft tokens and topk 1. `1:7,2:3` = 7 draft
  steps at 1 running request, 3 from 2 on; the launch K is 7. Measured mean acceptance:
  c1 3.90, c8 3.42, the others 3.00.
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
- c16-c48 also set the bf16 split-cache merge (`SGLANG_TPU_DSA_MLA_KV_SHARD_BF16_MERGE=1`).

### Server

From the repo root, with the run JSON from the table:

```bash
A=benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx
CFG=$A/runs/C16_S2ts_kda1400_bf16_real_l3_ring.json
CR=/cache/agentx/c16                     # on a PVC
DEPS=/cache/agentx/pydeps                # transformers 5.19.0, blake3, timm 1.0.25
export PYTHONSAFEPATH=1 PYTHONPATH=$DEPS:$PWD/python:$PWD PYTHONFAULTHANDLER=1 HF_HUB_OFFLINE=1 \
  TMPDIR=$CR/tmp PROMETHEUS_MULTIPROC_DIR=$CR/prom SGLANG_CACHE_DIR=$CR/sglang \
  TORCHINDUCTOR_CACHE_DIR=$CR/inductor TORCH_TPU_TIER3_ROOT=$CR/tier3 TORCH_TPU_TIER2_NAME=agentx_c16
mkdir -p $TMPDIR $PROMETHEUS_MULTIPROC_DIR $SGLANG_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TORCH_TPU_TIER3_ROOT ~/.cache
ln -sfn $SGLANG_CACHE_DIR ~/.cache/sglang
ulimit -c 0
python3 -u -m benchmark.tpu.launch_server --config @$CFG    # --dry-run prints the command only
```

- One host of a 2x2x2 slice run alone, as the dev-pod points ran: export this first.

  ```bash
  export FORCE_SINGLE_HOST=1 TPU_TOPOLOGY=2x2x1 TPU_HOST_BOUNDS=1,1,1 HOST_BOUNDS=1,1,1 \
    TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_WORKER_HOSTNAMES=localhost \
    TPU_WORKER_ID=0 TPU_PROCESS_ADDRESSES=localhost:8471 TPU_ACCELERATOR_TYPE=tpu7x-8
  ```

- c1 on torch_tpu 0.2.0, as measured:

  ```bash
  python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); d["server_args"]["use_spmd_compile"] = False; json.dump(d, open(sys.argv[2], "w"), indent=2)' \
    $A/runs/R1b_c1_real_l3_ring.json $CR/c1.json
  CFG=$CR/c1.json
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
- The harness warm-up runs before the window (up to ~23 min at c48). Only the profiling
  phase counts. No warm-up prompts or prefill probe are sent; the previous points sent both.
- Per point: boot, then 62-83 min of client time (warm-up, the 1 h window, drain).

### Metrics

`summarize.py` (stdlib only) turns one output dir into a table row:
- tok/s/chip = (prompt + completion tokens) / window / 4 chips. The window runs from the
  first request start to the last request end, profiling phase only.
- 1/ITL p50 / p90 = 1000 / inter-token latency (ms).
- TTFT and E2E latency: percentiles over the profiling-phase requests.
- Prefix hit = (device + host hits) / all prefill tokens on TP0, from the
  `--server-metrics` CSV.
- Ideal = aiperf `theoretical_prefix_cache_hit`.
- On the six output dirs it gives the Results table's tok/s/chip, TTFT and E2E latency,
  1/ITL within 0.4 and hit within 0.02 pt.

### Submission vs measured

- The submission is the run JSONs in the table, as measured, except c1's `use_spmd_compile`
  (see Per point).
- Each `*_real_l3_ring.json` is built in steps:
  - `*_real.json`: the point's dummy-weight config with real weights (no `load_format`,
    checkpoint prefetch, page cache dropped after load, 2400 s unbalanced-load timeout),
    token mode `real-draft-token` and request time stats;
  - `*_real_l3.json`: plus `hybrid_cache_checkpoints_level` 3;
  - `*_real_l3_ring.json`: plus the ring kernel. c1/c4/c8 also get the draft-step changes.
- The recipes (`agentx_cc<N>.py`, `launch_agentx.sh`, `check_recipes.py`) still build and
  check the previous dummy-weight points' configs, not these.

## Other measured runs

Each of these files in `runs/` is a run's launch JSON.

Real-weight curves, 1 h windows, harness 1.0.14, real-draft-token (tok/s/chip at c1 / c4 /
c8 / c16 / c32 / c48):

| run JSONs | job | change | tok/s/chip |
|---|---|---|---|
| `*_real.json` | fd97 | real weights; checkpoint level 2 (default) | 5,813 / 8,217 / 12,651 / 18,615 / 24,272 / 26,506 |
| `*_real_l3.json` | 301e | + level 3: hit gap back to the dummy level | 5,988 / 8,678 / 14,049 / 20,213 / 28,854 / 31,960 |
| `*_real_l3_ring.json` | e2df | + this round (What changed) | Results |

The rest are dummy-weight 1 h windows unless stated otherwise.

c32 follow-ups to `R5_c32_ts.json` (38,358 tok/s/chip, 1/ITL p90 54.4), none of which beats it:

| run JSON | job | change vs `R5_c32_ts` | outcome |
|---|---|---|---|
| `R6_c32_ts_mr32.json` | 175a | max running 32, decode buckets 1..32 | no window: boot ran out of host RAM allocating the KDA host pool |
| `R6b_c32_ts_mr32_sparse.json` | 175a | as above, buckets 1..16, 20, 24, 28, 32 | OOM-killed 35 s into the window (host RAM) |
| `R6c_c32_ts_mr32_b10_h20.json` | 175a | max running 32, 10 buckets, host KV 2.0x | 38,149, 1/ITL p50/p90 85.0 / 39.9, TTFT p90 3.03 s; dominated by R5_c32_ts |
| `R7_c32_s4ts_mr32_b10.json` | 3ca4 | as R6c with split 4, host KV 1.33x | 38,807, 1/ITL 87.9 / 41.2, TTFT p90 2.80 s; dominated on the chart axes |
| `R5k_c32_ts_kda1600.json` | 98c5 | 1,600 KDA slots | 37,900, 1/ITL p90 53.2, hit gap 0.83 vs 0.80 pt: no gain |

Earlier arms, superseded by the points above (tok/s/chip, 1/ITL p90):

| run JSON | job | config | outcome |
|---|---|---|---|
| `E1_c1.json` | 2460 | c1, K=8 / AL 3.99 | 5,557, 352.5 |
| `E2_c4.json` | 2460 | c4, split 8, K=8 | 7,996, 236.4 |
| `E3_c8.json` | 2460 | c8, split 8, K=8 | 13,693, 148.8 |
| `E4_c16.json` | 2460 | c16, split 8, K=3 | 23,207, 83.7 |
| `R2_c4.json` | 2460 | E2 with K=3 / AL 3.00 | 8,359, 282.5 |
| `R3_c8.json` | 2460 | E3 with K=3 | 14,337, 175.0 |
| `R5_c32.json` | 2460 | c32, split 8, K=3, max running 16 (ran as R5b after a warm-boot hang) | 35,755, 50.2 |
| `R1b_c1_ts.json` | fee5 | c1 point + token split | 6,269, 416.8 (neutral) |
| `C4u_c4_ts.json` | fee5 | c4 point + token split | 8,939, 336.4 (neutral) |
| `R3_c8_ts.json` | fee5 | R3 + token split | 14,466, 188.1 |
| `E4_c16_ts.json` | fee5 | E4 + token split | 24,407, 93.5 |
| `K2_c8.json` | f366 | c8, split 2, no token split (decode probe) | 14,796, 198.2 |
| `C16_S2ts.json` | 8c26 | c16, split 2 + token split (chart v26 point) | 26,037, 110.9 |
| `C16_S4ts.json` | 8c26 | c16 point with split 4, host KV 1.65x | 25,139, 98.1 |
| `C16_S2ts_kda1100.json` | 8c26 | c16 point with 1,100 KDA slots | 26,004, 111.6 |
| `C16_S4ts_real.json` | 8c26 | C16_S4ts, real weights | GSM8K-200 0.905 (no AgentX window) |

## Pending

- Prefill chunks (in progress). Prefill stalls set the p90 1/ITL tail; without them p90
  1/ITL would be 93 -> 130 at c16, 35 -> 61 at c32, 23 -> 48 at c48.
- Decode step at 4-48 running (in progress): the per-request step cost at real AgentX
  contexts.
- InferenceX question 3 below (draft steps per decode step).

## Tunable configs

Values as the run JSONs set them, c1 / c4 / c8 / c16 / c32 / c48; unset = server default.
Evidence is a job ID; 1/ITL numbers are p90. Evidence before job 301e is on dummy weights.

| knob | where set | c1 / c4 / c8 / c16 / c32 / c48 | what it trades | evidence |
|---|---|---|---|---|
| prefill-decode interval | `prefill_decode_interval` | unset (0, off) at every point | N decode steps between prefill batches: higher 1/ITL, later TTFT; a fixed N starves admission at high concurrency | c16 N=16, split 8 before token split: 1/ITL 84.9 -> 91.3, tok/s -0.5%, TTFT p90 2.06 -> 2.95 s (ff06). c32 N=16: 1/ITL +38%, tok/s -1.8%, TTFT p90 12.7 -> 17.8 s; N=64 worse (754a). c48 N=16: max prefill wait 96 s (ccd1). Muse Spark c10, N=64 vs its shipped profile: 1/ITL 169 vs 116 (Haolin's perf KB, F-0069..71) |
| latent-cache split | env `SGLANG_TPU_DSA_MLA_KV_SHARDS` | unset (1) / unset / 2 / 2 / 8 / 8 | fewer shards: cheaper prefill merge, more KV bytes per rank (host KV 30,720 / 18,432 / 12,288 B/token/pod at 2 / 4 / 8; the split-2 pool is HBM-bound below the 9.4M cap) | c16 split 2 vs 8: 1/ITL +18.9%, tok/s +6.7%; split 4: 1/ITL 98.1 (8c26). c32 at max running 32, split 4 vs 8: 38,807 vs 38,149 tok/s, 1/ITL 41.2 vs 39.9 (3ca4, 175a). c4 unsplit vs split 8: 1/ITL 337.7 vs 287.5 (6076) |
| token split | env `SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT` | unset / unset / 1 / 1 / 1 / 1 | each selected cache row gathered by one rank, plus an all-to-all of queries and outputs | c16 split 8: 1/ITL 83.7 -> 93.5, tok/s 23,207 -> 24,407; c8: 1/ITL 175.0 -> 188.1; neutral on the unsplit c1/c4 (fee5) |
| device KV pool | `max_total_tokens` (cap; HBM may bound it lower), `mem_fraction_static` 0.9 | 3,407,872 / 3,407,872 / 9,437,184 / 9,437,184 / 9,437,184 / 16,000,000 (boot sizes in the per-point table) | HBM for KV tokens vs device KDA slots | c16 (8.64M -> 5.66M tokens, 800 -> 1,400 slots): state evictions 6,399 -> 2,192 per h, tok/s and 1/ITL within noise (8c26) |
| device KDA slots | `max_mamba_cache_size` | 74 / 384 / 800 / 1,400 / 800 / 800 | HBM for KDA checkpoints vs KV pool; sized for the conversations AgentX keeps cached, not from max running | c16 1,100: 26,004 / 1/ITL 111.6; 1,400: hit gap 0.36 vs 0.50 pt (8c26). c32 1,600: no gain (98c5) |
| checkpoint level | `hybrid_cache_checkpoints_level` | 3 at every point | KDA checkpoints at every prefill chunk vs only at branch points and request ends (level 2, the default) | real weights, level 2 -> 3: hit gap 1.67-3.80 -> 0.13-1.17 pt, tok/s +3 to +21% (301e) |
| host KV tier | `hicache_ratio` (x device pool) | 2 / 2 / 1.5 / 1.7 / 2.0 / 1.125 | host-hit reach vs host RAM | c32 at max running 32 fit with 2.0 (R6c, 175a) and 1.33 at split 4 (R7, 3ca4) |
| host KDA states | env `SGLANG_TPU_HICACHE_STATE_RATIO` (x device KDA slots; unset = `hicache_ratio`) | unset / unset / unset / 1.429 / 2.5 / 2.5 | host states (~19.6 MB/rank each) give the host-tier hits after idle gaps vs host RAM | c16 2,001 states reach ~15 min; ratio 1.5 would cut them to 1,201, ~9 min (8c26) |
| max running | `max_running_requests`, `torch_compile_max_bs` | 2 / 4 / 8 / 16 / 32 / 48 | decode batch vs step time | c32 at 32 vs 16, both with the prefill patches: 40,220 vs 39,770 tok/s, 1/ITL 43.5 vs 59.6 (5196, 15c4); before the patches tok/s flat, 1/ITL -24 to -27% (175a, 3ca4) |
| decode buckets | `tpu_compile_bs` | 1..max running at c1-c16; c32 10 buckets; c48 24 buckets | unpadded decode steps vs host RAM per program | c32 at max running 32: buckets 1..32 ran out of host RAM at boot, 1..16 + 20..32 was OOM-killed in the window, 10 buckets ran (175a) |
| MTP draft length | `speculative_num_steps` = K (`speculative_num_draft_tokens` K + 1, `speculative_eagle_topk` 1); env `SGLANG_TPU_SPEC_STEPS_BY_RUNNING` | `1:7,2:3` / 3 / `1:7,2:3` / 3 / 3 / 3 | more tokens per step at low batch vs more verify work at high batch | per token, K=7 vs K=3 with the ring: c8 -2.2 to -2.7% at 1 running, +5 to +14% at 3-8; c4 tie at 1, +6.9 to +13.6% at 2-4 (5795). Dummy, at the golden AL: c4 K=3 vs 8 1/ITL 282.5 vs 236.4; c8 175.0 vs 148.8 (2460) |
| simulated acceptance | env `SGLANG_SIMULATE_ACC_LEN` (`SGLANG_TPU_SPEC_SIMULATE_ACC_LEN_BY_STEPS` per K), `SGLANG_SIMULATE_ACC_METHOD`, `SGLANG_SIMULATE_ACC_TOKEN_MODE` | 3.93 / 3.0 / 3.93 / 3.0 / 3.0 / 3.0; `match-expected`; `real-draft-token` | not a perf knob: AL = `GOLDEN_AL[K]`, never 3.5. Token mode: same accepted count as `fixed`; real tokens cost more per step | before the compiled accept, `real-draft-token` vs `fixed`: +3.1 / +4.4 ms per step at 8 / 16 running (290a) |
| MoE ring kernel | env `SGLANG_TPU_ENABLE_MOE_RING_KERNEL` | 1 at every point | each distinct FP8 expert streamed once through a VMEM ring, 2..256 tokens | decode step -29% at 8 / 32 running; c32 +20.9% (78d8) |
| megakernel | env `SGLANG_TPU_GLM_FLASH_MEGAKERNEL` | 0 at every point | one kernel per decode step vs the model path | vs model + ring: -3% at 1 request with K=3, +15 / +45% at 2 / 4, slower everywhere with K=7 (1454) |
| prefill chunk | `chunked_prefill_size`, `max_prefill_tokens`, `tpu_compile_tokens` | 4096, 4096, [1024, 2048, 3072, 4096] at every point | bigger chunks: less fixed cost per token vs longer decode stalls, more programs | 8192: -2.2% per token, stalls ~2x, +18 programs/rank (dcd8). c16 at 2048: 1/ITL 73.4 vs 83.7 (6076) |
| expert parallel | `ep_size` (`tp_size` 8) | 1 at every point | expert-TP vs expert-parallel MoE | untested; judged not worth it for c32+ |
| top-k width buckets | env `SGLANG_TPU_SCTOPK_LONG_ROWS` | on by default (0 disables) | exact; faster prefill top-k | with the bf16 merge at split 8 + token split: 4k chunk 0.2432 -> 0.2204 s at ctx 0, 0.2795 -> 0.2393 s at 256k (2507) |
| bf16 split-cache merge | env `SGLANG_TPU_DSA_MLA_KV_SHARD_BF16_MERGE=1` | unset / unset / unset / 1 / 1 / 1 | bf16 instead of fp32 reduce-scatter for the prefill merge | as above; GSM8K-1319 0.9060 on vs 0.9037 off (2507). c32 with both: 39,770 vs 38,358, 1/ITL 59.6 vs 54.4 (15c4) |

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

- `runs/*_real_l3_ring.json`: the launch JSONs of the Results points.
- `runs/*.json`: the launch JSONs of all measured runs, which are the measurement record.
- `summarize.py`: computes the metrics of one client output dir.
- `_base.py`, `agentx_cc<N>.py`, `launch_agentx.sh`, `check_recipes.py`: build, launch and
  check the previous dummy-weight points' configs. `benchmark/tpu/test_launch_server.py`
  also runs `check_recipes.py`.

Chart: [GLM Flash AgentX Curves](https://claude.ai/artifact/JnGfziw2GBSqDx5LCZArVf)
