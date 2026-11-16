# GLM-5.2 selected-route MoE port: device and E2E results

The routed and shared-expert port passes functional acceptance. It is slower
than the requested historical baseline in this workload and remains disabled by
default. No fresh baseline A/B was run and the cross-run comparison does not
isolate kernel causality.

## Source and acceptance

- Patch base: #2198 `ffd83db8f6168e0752e0769f8947cffb308e724d`.
- Port source: `sgl-project/sglang-jax@c948bdda95208e68a474c8088937347d08fe8937`.
- Frozen runtime implementation: `73886400f0`; E2E measurement source:
  `9c7eedd3a2ca16ba6f906900fa6b3a1393072259` (same implementation plus harness).
- Candidate: Falcon `exp-0ppqh107uo`, artifact `art-gvokyz9112`, **SUCCEEDED**
  on both hosts; 2026-11-16 17:46:15–18:28:29 UTC, about 42m14s including
  bootstrap, preflight, loading and compilation.
- Result extraction: analysis `an-5ie99xlyfo`, **SUCCEEDED**.
- Historical baseline: `exp-6mtkc6t2qi` / `art-tfqylgfhvw`; its frozen request
  inputs are reused (1,142,656 token ids; SHA256
  `607d4dc372753216266489042597330f2fc66c94b5e0fd6a760181a10162b3c6`).
  The TopK candidate is absent from this measurement.
- Hardware/layout: eight physical v7x chips, two hosts, sixteen workers,
  TP16/DP-attention16/EP16, PP1. MTP stays on (five steps, topk1, six draft
  tokens); the independent NextN draft retains its existing backend.
- Same absolute Python: `/tmp/glm52-baseline-runtime/venv/bin/python`.
  Actual workers record Torch 2.11.0, torch_tpu 0.2.1.dev20261028201649,
  JAX/JAXLIB 0.11.2 and libtpu 0.0.50, including imports/compiler settings and
  mapped libtpu. Setup-only probes are not substituted for worker evidence.

The registered EP8 full sequence (`exp-rjhbw8wnjp`) passes all eight ranks.
The mandatory EP16 preflight passes both hosts in approximately 98 seconds,
with complete records for ranks 0–15: each passes ten small routed/shared and
partially invalid-route cases, compiled 64-row idle DP 16→8→1→16, and the real
H6144/I2048/E256/TopK8 geometry at local 6/24/64/2048 rows. Preflight directly
consumes TPU Parameters after pack/replace; all outputs are finite and the
full-geometry FP32-reference relative L2 is 0.0347–0.0457. Real shared FP8
all-gather/layout and storage replacement checks also pass.

Actual model logs confirm expected=prepared=shared=75 on every worker (the
78-layer checkpoint has three dense layers), sixteen NextN keep-baseline
markers, and the new `glm_ep_moe_v2_ep16_shared1` exports for verify and prefill,
including global 32768 rows. Compiled graphs report no CPU partitions. All
sixteen DP smoke requests answer the basic arithmetic check; generation is
forced to sixteen tokens with `ignore_eos=True`, so this is basic functional
acceptance, not a broad quality-equivalence evaluation.

## Historical comparison

Three measured rounds follow warmup for each scenario. Compilation, prefix
priming and drain tails are excluded from the steady throughput window.

| Metric | Historical baseline | MoE candidate | Change |
| --- | ---: | ---: | ---: |
| C16 cached-prefix TTFT, ms | 6302.242 | 7037.514 | +11.67% |
| C64 client TPOT, ms | 18.849 | 20.793 | +10.31% |
| C64 steady output, tokens/s | 3511.861 | 3209.872 | -8.60% |
| C64 steady output, tokens/s/physical chip | 438.983 | 401.234 | -8.60% |
| C64 accepted drafts/verify | 4.1665 | 4.1979 | observed ratio |

C16 TTFT medians by round are 7025.681 / 7043.348 / 7037.514 ms. C64 TPOT
medians are 20.467 / 20.848 / 20.793 ms, with per-chip steady throughput
403.180 / 394.151 / 401.234 tokens/s. These results establish a roughly 10%
regression in the selected historical comparison, not a performance win. No
post-measurement math or tuning change is made to obscure it.

## Workload and cache checks

| Scenario | Requests / DP | Input / output tokens | Cached tokens, every round |
| --- | ---: | ---: | ---: |
| C16 prefill | 16 / 1 | 53248 / 16 | 49088–49152 |
| C64 shared-prefix decode | 64 / 4 | 90000 / 1024 | 89920–89984 |

Both cache-hit ranges match the historical baseline, including the allowed
one-page (64-token) recomputation. Full-concurrency load snapshots are
28/35/23 for C16 and 784/835/774 for C64. C64 uses the common window after all
requests reach output token 64 and before any reaches token 960; all samples
inside that window show all sixteen DPs with four running requests. Window
lengths are 15.166 / 15.915 / 14.753 seconds. The 6.770 / 5.010 / 6.542 second
drain tails are excluded. Client checks enforce the input/output lengths,
selected DP receipts and non-aborted completion. Independent raw-request audit
confirms all 368 requests have `num_retractions=0`; each of the 240 formal
requests has the stated prompt/completion lengths, its requested DP rank and
length-based completion. The C64 steady windows contain 639/655/604 sampled
load records, all with sixteen DPs at four running requests and zero waiting
requests. Maximum sampled used tokens is 93888, below capacity 168512.

Minimum reported token capacity across DPs decreases from 219008 to 168512
(about 23.1%). Replicating the existing shared weights adds
`75 * 3 * 6144 * 2048 = 2,831,155,200` bytes (about 2.64 GiB), plus scales.
The comparison therefore has a real memory cost even though the shared-prefix
workload still fits. Maximum observed post-drain load-boundary use is
66,815,450,112 bytes. This is not an instantaneous allocation peak. The run
finishes without OOM. The harness did not scrape allocator eviction counters;
matching cache hits and absence of an eviction warning cannot prove those
counters were zero.

## Shutdown evidence

The last formal request finishes at 18:27:58.917 UTC. At 18:28:02 the host-0
server logs SIGTERM and controller termination during the harness shutdown.
The other host then logs two CPU Gloo peer-reset tracebacks at 18:28:03/05 in
`dp_attn.prepare_mlp_sync_batch_raw` / `all_gather_into_tensor`, referencing that
peer. Both application results remain `SUCCEEDED` without a shutdown timeout.
The timing and CPU stack support classifying these as cross-host teardown
errors after measurement, rather than a Pallas failure during a request. The
shutdown is therefore not described as error-free or fully orderly. No port
change or replacement run is made to hide those logs.

## Review status

`SGLANG_TPU_USE_GLM_MOE_V2` remains false by default. The patch is an opt-in
functional port with measured performance/memory drawbacks, not a proposed
default replacement. It adds no new core TPU branch or hook: the existing
plugin loader-finalization entry dispatches the binder after quant processing.
The pure-call contract declares all mutable HBM workspaces as outputs and keeps
all waits/barriers/drains. See `glm52-moe-port-evidence-20261116/compiler-entrypoint.md`
for exact source reconstruction, compiler diagnostics and earlier failure
boundaries. Improving the regression requires separate measured follow-up;
this task does not claim an isolated cause or a speedup.
