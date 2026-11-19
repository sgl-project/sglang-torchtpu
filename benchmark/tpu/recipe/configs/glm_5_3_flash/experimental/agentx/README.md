# GLM-5.3-Flash AgentX on TPU (branch `qinghan-glm-agentx`)

Serving configs and measured AgentX numbers for GLM-5.3-Flash on one tpu7x host
(4 chips, tp8, 1M context, MTP verify), one recipe per client concurrency (cc).

## Status

- Local branch, not pushed. No PRs to main from it for now.
- Fixed base: main `a28113224f` plus the commits below. It does not track main;
  numbers are compared on this tree, not on current main.
- On top of the base:
  - #2214: sparse-MLA latent cache split across the 8 attention-TP ranks (head `f5d3c8d2fd`).
  - #2261: DSA attention token split with the pooled indexer (`1b15f613e4`).
  - #2251: one AgentX recipe per cc (`6dd36c73fd`).
  - 1k/2k/4k/8k prefill buckets and a gmm_v2 tuned entry for the 4096 bucket (`ab02eaa9bc`, `ddb08e45ee`).
  - Merge of main `69b9ae761b` (#2233, #2262, #2263, #2265, #2269 and others).
  - The latent split moved inside every page, so the host tier (HiCache) works with it for
    the DSA caches and the KDA states (`6f4546f4f9`), eviction-cause counters (`1260fcf114`),
    KDA host pool sizing (`73b341852d`), review fixes (`01ca1a418e`).
  - Recipes: host-tier variants for c16/c32, KDA slots explicit per recipe (`e5305a50a9`).

## Config per cc

Recipe files are in this directory; `_base.py` builds all of them. Values below are from
the recipe source and the launcher's `--dry-run` at `e5305a50a9`.

| cc | recipe | max running | KV pool tokens | KDA slots | latent split | host tier | decode megakernel | compile bs |
|---|---|---|---|---|---|---|---|---|
| 1 | `agentx_cc1.py` | 2 | 3,407,872 | 74 | none | on, ratio 2 | every step (<= 2 running) | 1, 2 |
| 4 | `agentx_cc4.py` | 2 | 3,407,872 | 74 | none | on, ratio 2 | every step (<= 2 running) | 1, 2 |
| 8 | `agentx_cc8.py` | 16 | 3,407,872 | 144 | none | on, ratio 2 | steps with <= 4 running | 1, 4, 8, 16 |
| 16 | `agentx_cc16.py` | 16 | 9,437,184 | 800 | C=8, in-page | off | off (model path) | 1, 4, 8, 16 |
| 16 | `agentx_cc16_host.py` | 16 | 9,437,184 | 800 | C=8, in-page | on, ratio 2.5 | off (model path) | 1, 4, 8, 16 |
| 32 | `agentx_cc32.py` | 16 | 9,437,184 | 800 | C=8, in-page | off | off (model path) | 1, 4, 8, 16 |
| 32 | `agentx_cc32_host.py` | 16 | 9,437,184 | 800 | C=8, in-page | on, ratio 2.5 | off (model path) | 1, 4, 8, 16 |

- **KDA slots are sized for the conversations AgentX keeps cached, never from max running.**
  `agentx_config` requires `mamba_slots`. Measured on the split cache without a host tier (11-16):
  a 16.8M-token pool with 144 slots gave 5,354 tok/s/chip at c16 and 3,876 at c32; a 9.4M pool
  with 800 slots gave 20,093 and 22,813. With 144 slots the evictions came from the KDA slots, at
  KV usage of at most 0.22.
- c32 runs the c16 server config: 16 running, the other 16 sessions queue.
- Host tier sizing:
  - c1/c4/c8: KV host pool 2x the device pool (6,815,872 tokens per rank; the latent is
    replicated, so every rank holds a full copy); KDA host states 2x the device slots.
  - c16_host/c32_host: 2.5x for both. Per rank: the 1/8 latent share plus index keys for
    23.6M tokens (36.2 GB) and 2,000 KDA states (39.3 GB); 604 GB of host RAM on the 8 ranks,
    with 108 GB left available on the 990 GB host (measured). Do not raise the ratio.
  - `state_host_ratio` (`SGLANG_TPU_HICACHE_STATE_RATIO`) sizes the KDA host states on their own;
    no recipe sets it.
  - Common: `hicache_io_backend raiden`, `page_first_fused`, `write_through`.
- Megakernel at c8: the recipe asks for 64 rows (16 running x 4 MTP rows); the kernel caps a
  step at 16 rows (VMEM), so steps with more than 4 running requests run on the model path.
- Since the golden-MTP recipes: MTP (NEXTN, topk 1) 8 steps / 9 draft tokens at c1, c4, c8
  and the `_host` c4 / c8 variants, 3 steps / 4 draft tokens at c16 and c32; acceptance
  forced to InferenceX's golden AL for that length (3.99 / 3.00,
  `glm-5.3-flash__mtp__native__default-default.yaml`, thinking_on) with
  `SGLANG_SIMULATE_ACC_METHOD=match-expected` and fixed committed tokens. The megakernel
  runs on the split cache too, so it is on in every recipe: 9-row steps fit its 16-row cap
  with one running request, 4-row steps with up to four.
- `agentx_cc4_host.py` / `agentx_cc8_host.py`: the c16 host-tier config (split, 9.4M pool,
  800 KDA slots, host 2.5x) at 4 / 8 running.
- `runs/E*.json`: the launch JSONs of the c1 / c4 / c8 / c16 runs on these settings
  (cc1, cc4_host, cc8_host, cc16_host with `SGLANG_TPU_GLM_FLASH_MEGAKERNEL=0`).
  `runs/R*.json`: the same with MTP K=3 at golden AL 3.00 (R2 c4, R3 c8, R5 = cc32_host at
  c32) and K=7 at 3.93 (R1b c1).
- Prefill: chunks of 4096 tokens (`chunked_prefill_size`, `max_prefill_tokens` 4096) and
  buckets `tpu_compile_tokens` 1024, 2048, 3072, 4096 at every cc. The measured rows below
  ran 8192-token chunks with buckets 1024, 2048, 4096, 8192.
- Compile: direct tracer (`tpu_compile_tracer direct`) at every cc. Without the latent split
  (c1/c4/c8) also SPMD compile (`use_spmd_compile`) with scattered prefill input off; the
  split cache refuses SPMD (a rank's share is a compile-time constant of its program). The
  measured rows below ran the Dynamo tracer.
- Decode buckets: one program per running count (`tpu_compile_bs` 1..max running, as Muse's
  AgentX recipe), so no decode step is padded; the megakernel builds the same buckets within
  its 16-row cap. The table's "compile bs" column is what the measured rows ran.
- DSA attention token split (`SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT`): off in every
  recipe. It is refused with the split cache (c16/c32). At c8 it was measured as an extra
  env (see the c8 rows).
- Same at every cc: context 1,048,576, page 128, fp8 KV, ep 1, `SGLANG_TPU_DSA_KPOOL_RAW_RING=1`,
  `SGLANG_TPU_GLM_FLASH_PAGE_TABLE_HBM=1`, megakernel max len 1,048,576,
  scheduler thread pinning, `scheduler_recv_interval 8`.

## Launch and measure

Server, from the repo root. `AGENTX_DUMMY=1` serves random weights (`load_format dummy`):

```bash
AGENTX_DUMMY=1 python3 -m benchmark.tpu.launch_server \
    --config recipe/configs/glm_5_3_flash/experimental/agentx/agentx_cc16_host.py
```

Add `--dry-run` to print the resolved command without starting a server.

Client, a 1 h window (`--concurrency` = the recipe's cc):

```bash
AIPERF_HTTP_TCP_USER_TIMEOUT=900000 agentx profile --url http://127.0.0.1:30000 \
    --endpoint /v1/chat/completions --concurrency 16 \
    --model zai-org/GLM-5.3-Flash --tokenizer /weights/GLM-5.3-Flash \
    --tokenizer-trust-remote-code --public-dataset semianalysis_cc_traces_weka_062126 \
    --benchmark-duration 3600 --server-metrics http://127.0.0.1:30000/metrics \
    --output-artifact-dir OUT/aiperf_artifacts
```

- The client replays agentic sessions (`agentic_replay`) with `ignore_eos`, first-turn cache
  bust and a 10 s idle-gap cap.
- **Warmup cannot be skipped.** Before the window, the client replays the history of every
  session in flight: snapshot primers (17 at c16), then 10 more requests per lane with
  `max_tokens=1` and no idle gaps. This brings the prefix cache to the state the window starts
  from. It took 507 s at c16 (bcf7 armA). Only the profiling phase counts.
- Budget per point: boot (12 min with a warm compile cache, 33-37 min cold), then warmup, then 1 h.

Metrics, from `profile_export.jsonl`: successful profiling-phase records only.
- Window: first request start to last request end.
- tok/s/chip = (prompt + completion tokens) / window / 4 chips. Prompt tokens include cached
  ones. Tokens per $ = tok/s/chip x 3600 / 1.03 ($ per chip-hour).
- Interactivity = 1 / inter-token latency at that percentile (tok/s per user); its p90 is the slow tail.
- e2e = aiperf `request_latency`.
- Prefix hit = cached prompt tokens (`usage.prompt_tokens_details.cached_tokens`; the recipes set
  `enable_cache_report`) / prompt tokens. Ideal = aiperf `theoretical_prefix_cache_hit`, an
  infinite-cache bound over ~64-token blocks.

Dummy weights versus accuracy:
- With `AGENTX_DUMMY=1`, the text is garbage by design. `SGLANG_SIMULATE_ACC_LEN=3.5` forces the
  MTP acceptance to 3.5 tokens per step, matching the GPU reference runs.
- For accuracy, load real weights and turn the simulation off:

```bash
python3 -m benchmark.tpu.launch_server \
    --config recipe/configs/glm_5_3_flash/experimental/agentx/agentx_cc8.py \
    --runtime-envs '{"SGLANG_SIMULATE_ACC_LEN": "-1"}'
python3 -m sglang.test.few_shot_gsm8k --num-questions 200 --parallel 8 --port 30000
```

## Numbers per cc

AgentX, SemiAnalysis cc traces, 1 h window, dummy weights.
- "= recipe": the run's config is identical to the recipe at `e5305a50a9` (checked against its JSON config).
- "older config": the run differs as stated; it is kept for reference. For c1 and c4 the
  only config difference is the prefill program (checked by dumping both recipes).
- Images: "old image" is the tpuq image used before 11-18; "new image" is `1affcdb`
  (jax 0.11.2, libtpu 0.0.50).

| cc | run (date, job, tree, image) | config | tok/s/chip | TTFT p50 / p90 s | interactivity p50 / p90 | e2e p50 / p90 / p99 s | prefix hit % (ideal) | requests (failed) |
|---|---|---|---|---|---|---|---|---|
| 1 | 11-17, cdf1, main `cc781eee6c`, old image | older config: 8192-only prefill | 5,690 | 0.99 / 1.45 | 408.1 / 376.5 | 3.24 / 13.73 / 33.5 | 97.97 (98.12) | 285 (0) |
| 4 | 11-17, cdf1, `cc781eee6c` + recipe `74ee0bb49d`, old image | older config: 8192-only prefill | 8,163 | 0.71 / 1.66 | 397 / 296 | n/a | 96.95 (97.91) | 627 (0) |
| 8 | 11-18, b844 `bk`, `ddb08e45ee`, old image | = `agentx_cc8.py` | 13,827 | 0.456 / 1.298 | 254.8 / 174.2 | 2.48 / 14.68 / 42.1 | 95.76 (97.74) | 1,316 (0) |
| 8 | 11-18, b844 `bkts`, `ddb08e45ee`, old image | `agentx_cc8.py` + token split | 14,190 | 0.430 / 1.152 | 262.4 / 190.5 | 2.35 / 12.89 / 41.1 | 95.96 (97.79) | 1,333 (0) |
| 8 | 11-18, b844 `8k`, `ab02eaa9bc`, old image | older config: 8192-only prefill | 13,596 | 0.656 / 1.449 | 240.2 / 142.7 | 2.82 / 15.05 / 44.5 | 96.02 (97.72) | 1,302 (0) |
| 8 | 11-16, `ring-1116` (d10f), main `55f642b178` + #2193 + raw ring, old image | older config: 8192-only prefill | 13,769 | 0.682 / 1.483 | 239.1 / 142.2 | 2.80 / 15.69 / 46.7 | 96.21 (97.74) | 1,307 (0) |
| 16 | 11-18, bcf7 `armA`, `1260fcf114`, new image | = `agentx_cc16.py` | 22,899 | 0.592 / 2.088 | 183.6 / 76.5 | 3.18 / 17.82 / 77.8 | 96.44 (97.35) | 2,348 (0) |
| 16 | ff06 `B0`, `73b341852d`, new image | = `agentx_cc16_host.py` | pending (ff06 B0) | | | | | |
| 16 | ff06 `P16`, `73b341852d`, new image | `agentx_cc16_host.py` + `prefill_decode_interval` 16 | pending (ff06 P16) | | | | | |
| 16 | ff06 `M8`, `73b341852d`, new image | `agentx_cc16_host.py` with max running 8 (800 slots) | pending (ff06 M8) | | | | | |
| 16 | ff06 `P64`, `73b341852d`, new image | `agentx_cc16_host.py` + `prefill_decode_interval` 64 | pending (ff06 P64) | | | | | |
| 16 | 11-16, `s2c-1116`, `55f642b178` + #2193 + raw ring + whole-page split, old image | older config: split by whole page (g % 8), 8192-only prefill | 20,093 | 0.94 / 2.2 | 128 / 56 | n/a | 96.11 (97.09) | 2,206 (0) |
| 32 | bcf7 `armB32`, `73b341852d`, new image | = `agentx_cc32_host.py` | pending (bcf7 armB32) | | | | | |
| 32 | bcf7 `armA32`, `73b341852d`, new image | = `agentx_cc32.py` | pending (bcf7 armA32) | | | | | |
| 32 | 11-16, `s2c-1116`, as the c16 row above, old image | older config: split by whole page (g % 8), 8192-only prefill | 22,813 | 6.3 / 29.3 | 63 / 32 | n/a | 93.86 (97.14) | 2,461 (0) |

- The recipe docstrings quote 5,690 / 8,163 / 13,769 / 20,093 / 22,813: the older-config
  rows at c1, c4, c8 (11-16), c16 and c32.
- e2e is n/a where no `request_latency` export of that run is kept.
- c16, no host tier, evictions in the window (bcf7 armA): 6.31M tokens evicted because the
  KV pool ran short (no KDA states); 1.83M tokens and 5,137 KDA states because the KDA slots ran short.

Accuracy, GSM8K-200, real weights, `SGLANG_SIMULATE_ACC_LEN=-1`:

| config | run (date, job, tree, image) | accuracy | invalid |
|---|---|---|---|
| `agentx_cc8.py` (buckets, megakernel, host tier) | 11-18, b844, `ddb08e45ee`, old image | 0.935 | 0.000 |
| split C=8 + host tier at ratio 4, pool 131,072, 144 slots (sized to force host reloads) | 11-18, bcf7 `chk2`, `1260fcf114`, new image | 0.925 | 0.000 |

- Same chk config on the old image (bcf7 `chk`): 0.935.
- bcf7 `chk2` reload check, split cache: a 20,205-token prefix (sparse DSA) answered 12/12 from
  device hits and 12/12 after a forced host reload, with identical greedy outputs. The reload
  moved 20,096 KV tokens and 2 KDA states.

## Caveats

- **Dummy weights:** random weights route MoE experts uniformly. Real routing is skewed. The
  megakernel and the MoE kernels see a different load than in production.
- **Simulated acceptance:** every decode number assumes 3.5 accepted tokens per MTP step.
- **One 1 h window per point.** Two runs of the same config on different trees (c8, 11-16 vs
  11-18 `8k`) differ by:
  - tok/s 1.3%;
  - e2e p50 0.7%, e2e p90 4.1%;
  - TTFT p90 2.3%, TTFT p99 65%.

  The client is closed-loop. The TTFT tail is set by a few re-prefills of evicted prefixes,
  which vary run to run.
- **Different trees and tpuq images across rows.** No row ran on the branch head itself:
  - `1260fcf114` (c16 armA) precedes `73b341852d` and `01ca1a418e` (host-pool sizing, validator checks).
  - `ddb08e45ee` (c8) precedes the #2214 and main merges.
- **Split cache refusals:** the latent split refuses:
  - the DSA attention token split;
  - the decode megakernel;
  - L3 storage (`--hicache-storage-backend`);
  - `--use-spmd-compile`, P/D and prefill CP.

  So c16/c32 decode on the model path, without the token split.

## Remaining work

- c16 with host tier: `agentx_cc16_host.py` (ff06 B0).
- c32 with and without host tier on the in-page split (bcf7 armB32, armA32).
- c16 interactivity arms: ff06 P16, M8, P64.
- c1, c4, c8 on the branch head with the prefill buckets (current numbers are older configs or `ddb08e45ee`).
- GSM8K on the exact `agentx_cc16_host.py` sizing; real-weight AgentX (expert routing).

## Changelog

On main `a28113224f`:
- `0c534ebafb`, `c0328b768a`: #2214, sparse-MLA latent cache split by page across attention-TP ranks.
- `1b15f613e4`: #2261, DSA attention token split with the pooled indexer.
- `6dd36c73fd`: #2251, one AgentX recipe per cc.
- `ab02eaa9bc`: 1k/2k/4k/8k prefill buckets.
- `ddb08e45ee`: gmm_v2 tuned tiles for the 4096 bucket.
- `5d5c751474`: merge #2214 head `f5d3c8d2fd`.
- `865eaa489e`: merge main `69b9ae761b`.
- `6f4546f4f9`: latent split inside every page; host tier admitted with the split.
- `1260fcf114`: device evictions counted by the pool that ran short.
- `73b341852d`: `SGLANG_TPU_HICACHE_STATE_RATIO` sizes the KDA host pool.
- `01ca1a418e`: refuse L3 storage with the split, lcm share check, state ratio check.
- `e5305a50a9`: recipes: `agentx_cc16_host.py`, `agentx_cc32_host.py`; `mamba_slots` required.
- This README.
