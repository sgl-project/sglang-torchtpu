# GLM-5.2 indexer radix V2 candidate

This patch is based on `ffd83db8f6168e0752e0769f8947cffb308e724d`
(PR #2198). Set `SGLANG_TPU_USE_DSA_RADIX_TOPK=1` on **each worker host before
startup** to opt in. The choice is frozen when the indexer custom op is
registered. Restart the process to change it; compiled services cannot switch
backends in place. The default stays SparseCore.

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
| Eligibility | v7, supported K, source dispatch heuristic | explicit opt-in on v7, K=16/2048; CP/score-returning calls use existing selector |
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
