# GLM-5.2 indexer radix V2 candidate

This patch is based on `ffd83db8f6168e0752e0769f8947cffb308e724d`
(PR #2198). Set `SGLANG_TPU_USE_DSA_RADIX_TOPK=1` on **each worker host before
startup** to opt in. The choice is frozen when the indexer custom op is
registered. Restart the process to change it; compiled services cannot switch
backends in place. The default stays SparseCore. With the opt-in enabled,
query chunks of at most 64 rows use V2; larger prefill chunks retain the incumbent because the bounded
sweep found a regression. GLM's 92,160-column/K2048 verify shape uses input block
16,384, reconstruction threshold 8,192, and forward scan.

The candidate replaces selection over the existing streamindex FP32 score
buffer. Query/key quantization, index-key cache bytes, writer, score precision,
causal row lengths, router/MoE, and sparse attention are unchanged. Outputs are
sequence-local logical positions; no physical mapping is requested. The existing
indexer custom op remains the public API. The private Pallas package does not
add a second custom-op/buffer singleton or a TPU dependency in shared core.

## Source and differences

Source: `sgl-project/sgl-jax` at
`c948bdda95208e68a474c8088937347d08fe8937`, Apache-2.0:
`python/sgl_jax/srt/kernels/radix_topk/v2/{kernel,tuned_configs}.py`.
The kernel file's last modification was `1377df0e6be2d807b7ba7db1f99addc73f629095`.
The existing `dsv4/sparsecore_topk.py` is a different, cooperative histogram
implementation with shared-VMEM communication; it remains the fallback.

| Item | JAX source | This adapter |
| --- | --- | --- |
| Selection | exact unordered, one subcore per score row | same |
| Candidate storage | bounded K + reconstruction threshold + input block | same |
| Input | FP32 scores | FP32 or existing int32 score bits, bitcast only |
| Masked `-inf` | accepted as a real score inside valid length | excluded, matching the existing streamindex contract |
| Hardware dispatch | `jax.devices()` spelling in DSA wrapper | Pallas target metadata, including TorchTPU CPU export |
| Physical mapping | optional shared page table | unused; logical positions only |
| Eligibility | v7, supported K, source dispatch heuristic | explicit opt-in on v7, K=16/2048, at most 64 rows; larger chunks/CP/score-returning calls use the existing selector |
| Runtime | source package declares JAX TPU 0.11.1 | target declares JAX/JAXLIB 0.11.2 and libtpu 0.0.50 |

The score producer already materializes this buffer. This patch introduces no
additional full score matrix, no new query chunks, and no extra host reads.
The existing chunk/overlap scheduling annotations are preserved. The tuning
table is copied as a starting point, not a claim of TorchTPU performance.

## Contract and validation

Picks are int32, exact as a set, unique, below each row's causal bound, followed
by `-1` padding. Arbitrary legal cutoff ties are allowed. NaNs are unsupported.
The registered DSA test adds selector and full-indexer coverage for masked
holes, negative/equal scores, empty/short rows, partial blocks, uneven requests,
shuffled physical pages, padding, verify and token split. It uses model K=2048.

With the candidate's pinned absolute interpreter and `PYTHONPATH=.:python`:

```sh
/path/to/python test/registered/tpu/unit/test_dsa_index_topk.py TestDsaRadixTopk
/path/to/python test/manual/kernels/benchmark_glm52_radix_topk.py --repeats 20
```

The manual benchmark emits runtime/import/libtpu evidence, same-score exactness,
selector timings and complete indexer timings including quantization and cache
writes. `--source-v2 /path/to/unchanged/jax/kernel.py` additionally measures the
original implementation under the exact same runtime. Compilation/first call is
reported separately from steady state. Use a bounded Falcon allocation.

E2E uses the existing `exp-6mtkc6t2qi` as the historical baseline, per the user's
instruction; only the candidate is rerun. MTP, workload, cache layout, W8A8 and
FP8 KV stay fixed. Historical-versus-candidate differences are observational,
not a controlled same-allocation A/B attribution. The device and E2E results below establish the tested functional scope; local
export checks supplement that hardware evidence.


## Device evidence and bounded tuning

- `exp-ssby71gg0x` / `art-s0thwqmdcf`, measurement source
  `e7a20de3a4e1a9a629f3c04a253878ba4cfb01d9`: both registered raw-JAX device
  tests passed in 23.250 seconds. Same-score selector measurements included
  the unchanged JAX source under the candidate's JAX/JAXLIB 0.11.2 runtime.
  Its complete-indexer fixture used generic 128-token pages, so those timings
  must not be labeled as the E2E's page64 shape.
- `exp-3ahwf0txj4` / `art-ux5izedk1z`, measurement source
  `418db3f4ab124f1cbe74ef20870c3e57d1df47e3`: the real TorchTPU serving bridge
  passed eager, torch.compile, token-split, padding and original-cache-alias
  checks in 8.173 seconds. A separate fresh raw-JAX process completed all 68
  selector/indexer tuning records. No second baseline model was run.
- Runtime evidence was captured inside each actual test process:
  `/tmp/glm52-baseline-runtime/venv/bin/python`, Python 3.12.12,
  Torch 2.11.0+cpu, torch_tpu 0.2.1.dev20261028201649,
  JAX/JAXLIB 0.11.2, libtpu 0.0.50, tpu-sync-torch
  0.0.1.dev20261112081648. Serving-op export used JAX CPU; standalone
  selector/indexer processes used raw JAX TPU. Imports, compiler flags and
  mapped libtpu paths were captured in the same processes.

The second allocation fixed page size 64 and page-table/score-buffer capacity
92,160 for both effective contexts. The benchmark uses the same paged FP8
score semantics as production, with generated inputs and shuffled pages;
these are per-local-device synchronized call latencies, not model timings.
Eight block/threshold/scan choices were tested once, with three warmups and
12 steady-state samples each. Compilation/first call is reported separately.

| Effective context | Workload | Existing selector, complete indexer (us) | Best forced V2 (us) | Difference | Final opt-in route |
| --- | --- | ---: | ---: | ---: | --- |
| 53,248 | verify, 64 padded rows / 24 active | 547.58 | 548.99 | +0.26% | V2, explicitly experimental |
| 92,160 | verify, 64 padded rows / 24 active | 659.30 | 688.02 | +4.36% | V2, explicitly experimental |
| 53,248 | prefill, 2,048 rows | 8,149.17 | 8,369.01 | +2.70% | existing selector |
| 92,160 | prefill, 2,048 rows | 11,809.98 | 13,530.00 | +14.56% | existing selector |

The complete-indexer verify fixture has **64 rows with 24 active queries**.
Startup logs from the actual MTP model show verify graphs with **6/12/18/24
rows**, plus EXTEND graphs with 64/2,048 rows. The fixture is a bounded
selection/indexer experiment, not the exact padding shape of every serving
verify call. Independent selector tests included 6/12/18/24/64-row shapes;
the raw-JAX integration test also exercised a 24-row verify batch.

Both fixture verify minima used `(16384, 8192, forward)`. There was no stable gain;
this is a functional port and reproducible negative performance result, not
an optimization enabled by default. The sweep stopped here. Model E2E ran
this frozen policy with MTP, as recorded below.

The private `radix_config` argument exists only to force an explicit static
configuration during bounded manual measurements. It preserves masks and
mathematics and bypasses the large-prefill fallback for measurement. Reproduce:

```sh
/path/to/python test/registered/kernels/ops/attention/test_dsa_radix_serving_op.py
# Fresh process after TorchTPU releases PJRT:
JAX_PLATFORMS=tpu /path/to/python test/manual/kernels/tune_glm52_radix_topk.py --repeats 12
```

Shared-core debt and registered-test checks pass. The support-matrix strict
check still reports ten pre-existing registrations outside this change.


## Candidate-only MTP model E2E

[Experiment `exp-l3h3amkwm8`, artifact `art-8cnbsifu1t`](https://falcon.infiscale-infra.org/v1/experiments/exp-l3h3amkwm8/artifacts/art-8cnbsifu1t/view)
and both host applications completed **SUCCEEDED** on the first attempt, with
16-DP routed smoke **PASS** and all six formal rounds present. The run completed
at 01:07:56 China Standard Time on 2026-11-17, about 47 minutes 40 seconds after
Falcon reported running; it stayed within the 6,900-second allocation bound and
the platform's two-hour limit. There was no application restart or recovery.
Formal artifact analysis `an-5o5hdp36tf` read back the result and identity files.

The measured code was `6991974e9b5245902a0310a1dd26e985d4d455d3`, composing
kernel policy `125bf7fe4dffa405bf2a1905633e5b97542d2a0b` with the frozen
benchmark harness. This follow-up changes documentation only. The historical
baseline is `exp-6mtkc6t2qi` / `art-tfqylgfhvw`, source
`811865ca48ca23aa14dc74c9dbebaf5b380279dc`. Per the requested scope, only the
candidate model was run; there is no new A/B baseline run.

Configuration: eight physical v7x chips, two hosts, 16 visible devices,
TP16/attention-DP16/EP16; offline channelwise Torch W8A8, original `pallas_tpu`
MoE, `pallas_tpu_mla_sparse`, FP8 KV, native NEXTN resolved to EAGLE with five
steps/topk1/six draft tokens, DP lm_head enabled, page size 64, context ceiling
92,160, mem fraction 0.85, and per-DP prefill chunk 2,048. The candidate env was
set on both hosts before startup. Only chunks of at most 64 query rows take V2;
large prefill chunks use the incumbent, while short prefill tails can take V2.

The candidate reused the historical baseline's frozen natural-code token arrays
through a read-only mount. The corpus contains 1,142,656 token IDs with SHA256
`607d4dc372753216266489042597330f2fc66c94b5e0fd6a760181a10162b3c6`, identical
to the baseline. The checkpoint config hash is
`0735624bbd5666283a1f135c59ff4c15e8b51c56fc50e022181525253f29463b`; the index
hash is `0ee9d61ce4e0ca912c2a5b8220f57b1060fb95ec09304da77c85a43d56aeb2a0`.
Actual-process records from both hosts confirm the same absolute interpreter,
imports, pinned runtime and mapped libtpu listed above. The harness source-tree
hash is `bbf1b6384c71c655c4497670f63eb1ba9edb72e2752fe93c2dad1fc4147ea475`;
its candidate/source/runtime/compiler cache identity key is
`6066dffbee517d306f60`.

C16 uses 49,152 cached tokens plus 4,096 extend, output 16. C64 uses 90,000 input
tokens and output 1,024, with each DP's four requests sharing an 89,984-token
prefix plus individual 16-token suffixes. Capacity was 219,008 slots/DP versus
364,288 needed for independent prefixes, so this remains a shared-prefix
benchmark. Observed cache hits were 49,088–49,152 and 89,920–89,984, respectively;
last-page replay is included in TTFT.

TTFT includes HTTP/scheduling. TPOT is the first-to-last SSE interval divided
by output tokens minus one. C64 throughput counts actual SSE output increments
in the common window after every request reaches 64 outputs and before any
reaches 960; all samples in that window sustain 16 DPs with four active requests
each. Values retain client-visible costs; serving-time compilation is not
subtracted. Each workload has one warmup and three formal rounds.

| Metric | Historical median (range) | Candidate median (range) | Cross-run median change |
| --- | ---: | ---: | ---: |
| C16 HTTP TTFT (ms) | 6,302.24 (6,301.60–6,327.23) | 6,268.74 (6,264.61–6,287.12) | -0.53% |
| C64 client TPOT (ms) | 18.8491 (18.2252–19.3932) | 18.5086 (18.3379–18.8638) | -1.81% |
| C64 common-window output (token/s) | 3,511.86 (3,368.34–3,628.18) | 3,561.84 (3,488.31–3,648.69) | +1.42% |
| C64 output / physical chip (token/s/chip) | 438.98 (421.04–453.52) | 445.23 (436.04–456.09) | +1.42% |

| Formal round | C16 TTFT (ms) | C64 TPOT (ms) | C64 output (token/s/chip) |
| --- | ---: | ---: | ---: |
| 1 | 6,287.1221 | 18.3379 | 456.0863 |
| 2 | 6,268.7394 | 18.8638 | 436.0382 |
| 3 | 6,264.6114 | 18.5086 | 445.2299 |

Accepted drafts per verify, computed from summed reported counters, changed
from 3.01970 to 3.12626 in C16 and from 4.16652 to 4.17710 in C64. These counters
are separate from the verified target token and from per-step latency; changes
in acceptance or batching cannot be attributed to selector speed.

The candidate throughput median lies inside the old run's 421.04–453.52
range. Together with the negative same-runtime kernel/indexer measurements,
these small historical differences do **not** establish stable model speedup
or isolated kernel causality. The result is a working port with a bounded,
reproducible negative microbenchmark result and a functional MTP E2E pass.
The switch remains default-off and this PR remains draft. The smoke and forced
length generation do not constitute a comprehensive model accuracy evaluation.

[Structured E2E results](glm52_radix_topk_e2e_results.json) retain all six round
summaries, reported cache-hit arrays, MTP counters, source/runtime/input hashes,
prefix decision and historical comparison. The full raw request/load/log
records remain in the Falcon artifact. Inspect the frozen run and analysis via:

```sh
falcon exp inspect exp-l3h3amkwm8 --output json
falcon workflow analysis outputs an-5o5hdp36tf --output json
```
