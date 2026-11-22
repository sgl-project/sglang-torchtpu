# Muse Spark 1P1D TP8 kernel handoff (2026-11-17)

Branch `wyzhang-exp-1117-tp-kernel`, one commit on main `97afaea937`. This is the best measured code.
Worktree: `sglang-torchtpu/tree/wyzhang-exp-1117-tp-kernel`. Experiment records:
`exp/wyzhang-exp-1117-tp-kernel/` (`STATE.md` = event log, `logs/<leg>/`, `scripts/render_2903c.py` legs,
`xprof/QH_OB_C64_C128.md`, `llo/PREFILL_LLO_QZ.md`, `advisor/r12-review-{a,b}.md`). Older branches
(1113-kernel, 1117-best, 1117-agd, 1117-fuse) are in `exp/.../branch-backup/tp-kernel-branches.bundle`.

## Status

Setup: Muse Spark NVFP4 MoE, fp8 KV, 1P1D, each engine one tpu7x 2x2x1 (TP8/EP8), InferenceX 8k/1k random
0.8-1.0, metric = total tok/s / 8 chips. Targets c64 2800, c128 3500.

Best config = this commit + prefill env of leg `sa` + decode server argument `--sampling-backend tpu` + decode env
`SGLANG_TPU_RPA_V3_DECODE_RING=3` (leg `sl`; full env in `exp/.../k8s/sl/b.json`). Prefill flags: the `QE` set in `render_2903c.py`,
`SGLANG_TPU_MUSE_KERNEL_EXPERT_PLACEMENT=/cache/wyz-1113-kernel/placement_nd_c64.json`,
`A1_AG_UNDER_D=1`, `A1_PARTS=2,3,2,1`, `A3_RPA_OUT=1`, `A3_OPT=1`. Decode env: `decode_role.env` + `SGLANG_TPU_RPA_V3_DECODE_RING=3`.

| Leg (11-17) | Config | c64 tok/s/chip / TTFT p50 / mean TPOT | c128 | Quality |
|---|---|---|---|---|
| **sl vs sm** | **sc + decode RING=3 vs sc, same window** | **2060.7 vs 2052.6 (+0.4 %)** | **3063.6 / 629 / 43.68 vs 3000.0 / 628 / 44.70 (+2.1 %)** | sn: GSM8K 0.968, MM pass |
| sj vs sk | first RING=3 pair (partial window overlap) | 2060.5 vs 2055.0 (+0.3 %) | 3081.2 / 616 / 43.43 vs 3016.7 / 619 / 44.44 (+2.1 %) | - |
| sc | this commit + decode tpu sampling | 2056.6 / 537 / 32.71 | 3001.9 / 622 / 44.66 | sf: GSM8K 0.98, MM pass |
| sa | this commit, pytorch sampling | 1831.6 / 747 / 35.60 | 2707.5 / 1139 / 49.02 | sb: GSM8K 0.974, MM pass |
| sh vs sg | repeat pair, tpu vs pytorch sampling | 2049.4 vs 1805.5 (+13.5 %) | 3005.7 / 610 / 44.59 vs 2668.6 / 1219 / 49.59 (+12.6 %) | - |
| sd | sa + `A1_PARTS=2,2,2,1,1` | 1834.6 / 708 | 2679.5 / 1216 | rejected (noise, c128 worse) |
| qz | previous best (same code before cleanup) | ~1836 | ~2710 | GSM8K 0.972 |

- Same input/output token totals in every pair. c64 gain measured three times (+12.3, +12.7, +13.5 %), c128 twice (+10.9, +12.6 %).
- RING=3 c128 gain measured twice (+2.1 %, +2.1 %, TPOT -1.0 ms both times, as predicted); c64 flat.
- Gap to target from sl: c64 needs mean TPOT 32.6 -> ~24.0 ms (-8.6 ms), c128 43.7 -> ~38.2 ms (-5.5 ms).
  Pricing: `tok_after = tok * TPOT / (TPOT - saved)`. Throughput is decode-bound; prefill work moves TTFT.
- Run-to-run noise is about 2 % tok/s; compare only same-window pairs.
- Why tpu sampling helps (xprof `si` + decode server logs, `xprof/si/SI_ANALYSIS.md`): the steady step barely moves
  (step period c64 31.8 -> 31.8 ms, c128 41.9 -> 41.2). The logits gather and its XLA (0.7 / 1.3 ms) are replaced by
  the top-k kernel (0.15 / 0.30 ms), and full attention got 0.3 / 0.7 ms slower in the new graph. The gain is stalls:
  with pytorch sampling, decode steps taken while a KV transfer arrives average 46 / 67 ms (median 34 / 47); with
  tpu sampling 34 / 46 ms. Implied mean step 38.7 -> 31.8 (c64), 54.8 -> 43.8 (c128); median unchanged. Likely the
  eager host round trip of pytorch sampling serialised with transfer handling; not proven (traces hold 4-6 steps).
- Device step period (si, tpu sampling): c64 31.8 ms, c128 41.2 ms. Mean TPOT minus step period, i.e. time
  outside the steady step: 0.9 / 3.4 ms. rc breakdown: FP4 GMM 15.2 / 17.5, SW attention 3.0 / 5.9, full attention
  1.9 / 3.8 (si 2.2 / 4.5), glue 3.1 / 4.3. Prefill forward (6.8k prompt) 177.3 ms.

## Next ROI: kernel improvements

Two independent reviews of the rc xprof, the qz prefill LLO and this commit (`advisor/r12-review-a.md`,
`advisor/r12-review-b.md`). Prices use sc as the base (row 6 is now in the base; sl is ~1 ms TPOT lower at c128). Savings are conditional estimates and not additive.

**Step 0 is done** (leg `si`, `xprof/si/`). It re-prices rows 4, 5 and 7, adds rows 0 and 8, and leaves rows 1-3 and
6 unchanged. No other XLA chain above 0.3 ms/step. Decode LLO for SW RPA and the decode GMM is still not dumped.

Decode (sets tok/s):

| # | Change | Saving c64 / c128 ms/step | tok/s/chip c64 / c128 | Effort | Agreement | First experiment |
|---|---|---|---|---|---|---|
| 0 | Time outside the steady step (mean TPOT - step period 0.9 / 3.4 ms): decode prealloc read-back fix (below) and admission work during KV transfer-in | up to 0.9 / 3.4 | 2116 / 3249 | 1-2 days | si data | Fix A from `glue/DECODE_FIXES.md` as a serving pair on sc |
| 1 | SW decode RPA: process 2-4 one-query requests per inner iteration (`ragged_paged_attention_v3.py:660-745, 2049-2092`) | 1.0-1.9 / 2.0-3.9 | 2121-2183 / 3143-3289 | 2-3 days | both | `rpa/bench_decode_rpa.py --kinds v3p_sw --batches 64,128 --llo` with nocompute/nodma arms, sc decode env |
| 2 | FP4 decode GMM per-expert visit cost: 8-row bucket, earlier W1 tile DMA, batched row DMAs (`gmm_fused_rs_nodedup.py:62-99, 1265-1321, 1503, 1562`) | 1.5-3 / 2-3.5 (B); unpriced (A) | 2155-2264 / 3143-3257 | microbench, then 2-3 days | split | `gmm/bench_gmm_tiles.py` at M512/M1024 with a nocomp removal arm + LLO; decide only after it |
| 3 | Router top-8 + weights + GMM group metadata in one decode kernel (`muse_spark.py:734`, `fused_moe.py:83,411`) | 0.75-0.9 / 0.9 | 2105-2115 / 3064 | 1.5 days | both | router chain microbench at B64/B128 with ties, empty local experts, padding |
| 4 | q/k RMSNorm + RoPE + fp8 cast + KV pack after the qkv GEMM (`muse_spark.py:386`, `attention_pallas_rpa_v3.py:55`) | 0.9 / 0.9 (si) | 2116 / 3064 | 2 days | both | one attention layer, byte-compare Q/K and fp8 cache |
| 5 | Decode RMSNorm layout (c128 input-norm 368 vs 125 us at c64) + expert-output layout copy 205 us (`muse_spark.py:165,844`, `fused_moe.py:551,559`) | 0.2 / 0.6-0.8 (si) | 2069 / 3043-3057 | 0.5-1 day | both | HLO layout dump at B128, single-layer A/B |
| 6 | ~~RPA decode fetch ring `SGLANG_TPU_RPA_V3_DECODE_RING=3`~~ done, adopted (sl vs sm, +2.1 % c128) | 0 / 1.0 measured | 2061 / 3064 measured | - | - | - |
| 7 | top-k ring kernel k=1 greedy branch (`kernels/distributed_topk.py:320-407, 732-739`) | <= 0.1 / 0.2 | 2063 / 3015 | hours | B only | kernel costs 120 / 234 us (si); low priority |
| 8 | Full attention RPAd 16-18 % slower per call in the tpu-sampling graph (same kernel, KV +3 %) | 0.3 / 0.7 | 2076 / 3050 | diagnosis first | si data | standalone RPAd microbench at si shapes vs rc shapes |

Rows 1-5 together: about 3.2-5.5 ms at c128, which gives about 3400 c128. c64 2800 still needs about 3-5 ms
more; more than one token per step (DFlash2 recipe, `benchmark/tpu/recipe/configs/muse_spark/*dflash*`) is
the remaining large lever. Dropped for decode: compact K256 scales (0.06 ms/step measured in isolation),
the logits-gather item (replaced by tpu sampling).

Non-kernel, larger than most rows above: the device read-back in decode prealloc costs about 16 ms of device
idle per admission, about 1.0 / 2.1 ms of TPOT (2121 / 3150). Fix A in `exp/.../glue/DECODE_FIXES.md`;
never benchmarked, not in this commit.

Prefill (ms per forward; sets TTFT, not tok/s):

| # | Change | Saving | Effort | Exact | First experiment |
|---|---|---|---|---|---|
| P1 | One-byte A1 all-gather payload: quantize the communicated Q/gate input, compute K/V from owner-local bf16 (`integration.py:541,602,621`, `kernel_a1.py:3371,4071`). A1 is transport-bound: the gather wait (77 us median) exceeds the copies (48 us) in all 58 layers | 5-9 | 3-5 days | no | SparseCore all-gather microbench bf16 vs s8 at 128/256/384 rows/rank |
| P2 | RPA m0: price the time outside the bundle model with the existing `_decode_probe="nocompute"/"nodma"` arms, then keep K/V tiles across q blocks (`ragged_paged_attention_v3.py:1069-1158`) | 1.2-3.7 | probe, then 2-4 days | yes | `llo/chain_qz.py` probes at T 4096/6144/8192 |
| P3 | A3 serving excess (151 us served vs 109 us bench) | <= 2.6 | diagnosis first | yes | rank/barrier removal arms |
| P4 | Gate-quant prologue tiles 512/256 instead of 1024 (`kernel_a3.py:2718`; 4764 spill stores, XLU bound) | 0.1-0.4 | 0.25 day | yes | `llo/chain_qz.py --chain v5x4p2321:v16:1:m0` |
| P5 | kernel D: refill gathers two tiles ahead (`kernel_d.py:534,648`; archived patch 160.9 -> 145.4 us on the old kernel) | 0.2-0.5 | 0.75 day | yes | `llo/prefill_llo.py --kernels moe`, then D -> AG -> A1 chain |

Suggested order: the prealloc fix (row 0) as a serving pair on sl, then decode #1
and #2 microbenchmarks, then #3-#5. Prefill only after decode, starting with the P1 microbench.

## Already tried, do not repeat without new evidence

- A1: copy removal (transport-bound), weight streaming (+25 us/layer), five parts `2,2,2,1,1` (leg sd), an
  in-kernel A1 collective (slower than the SparseCore all-gather).
- A3: gate/quant fused into Wo (v17, 197 vs 151 us).
- Prefill GMM tile / K_STEP / strip-loop sweeps; post-dot K256 scaling (3331 vs 2280 us); GMM output pre-summing.
- Sort-free scatter routing (`SORTFREE_SCATTER=2`, `SORTFREE_MODE=3`): -2.8 % at c128. Off by default.
- B2 quant under the next tile: B2 has one M tile per half, so it needs a new schedule.
- Per-shard greedy argmax on decode: TPOT -1.2 ms, tok/s flat.

## Running a leg

- Render and submit from the exp dir: `K13_NS=priority-dev K13_TREE=<worktree> python3 scripts/render_2903c.py <legs>`,
  then `scripts/submit_k13.sh <leg>/<b|q|f>` (b = bench, q = quality, f = xprof). Legs are tuples
  (mode, prefill env, decode env, decode server args) in `LEGS`; concurrencies in `CONC`.
- entry.py skips steps with a `.done` marker in `gs://wyzhang-dev-usc1b/cache/wyz-1113-kernel/<leg>-<mode>-<snapshot>/`.
  After a preemption mid-run, delete both jobs, clear that dir and resubmit: a restarted pod runs c128 cold.
- priority-dev capacity is reclaimed by other cohort queues; a half-admitted or preempted P/D pair must be
  deleted and resubmitted. Kueue may relabel the queue to `multislice-queue`; it still admits.
- `pd_profile.py` uploads the xprof only at the end: a preempted profile run loses everything.
