# GLM-5.2 indexer radix V2 candidate

This patch is based on `ffd83db8f6168e0752e0769f8947cffb308e724d`
(PR #2198). Set `SGLANG_TPU_USE_DSA_RADIX_TOPK=1` on **each worker host before
startup** to opt in. The choice is frozen when the indexer custom op is
registered. Restart the process to change it; compiled services cannot switch
backends in place. The default stays SparseCore. With the opt-in enabled, query chunks of at most
64 rows use V2; larger prefill chunks retain the incumbent because the bounded
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
not a controlled same-allocation A/B attribution. Hardware results are recorded
separately when available; this document does not claim model support or gains
from local export checks alone.


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

Both verify minima used `(16384, 8192, forward)`. There was no stable gain;
this is a functional port and reproducible negative performance result, not
an optimization enabled by default. The sweep stops here. Model E2E will run
this frozen policy with MTP and compare against the historical baseline.

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
