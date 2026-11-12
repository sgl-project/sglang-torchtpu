# Exact block-maximum top-k primitive

This is a **standalone, opt-in native JAX/SparseCore primitive**. It does not
replace `select_fp8_indexer_jax`, change its signature, change model callers,
change candidate metadata, or activate a serving fast path. Existing scoring,
selection, and fallbacks remain unchanged.

## Source and port delta

Base: SGLang `9cb500543ca9dbe059ff8bcc82dbe2e3026d36d3`.
Implementation and private radix helper extracted from Epic
`dc1cd126e71c2248f3fbf2bef0d8d24e1e3768d7`, respectively
`index_selection/block_topk.py` and `index_selection/publication.py`.
Only the required `Radix` class is ported; no additional publication API or
scorer rewrite is included. Existing `streaming_topk.py` is retained intact.

Port-specific changes:

- Check array rank/dtypes and native target admission before hardware access.
- Include actual optional-publication staging and resident fetch scratch in
  the VMEM plan. Admit publication only with valid HBM alignment and capacity.
- Require full SparseCore reduction-span alignment as well as HBM alignment;
  partial final spans otherwise gather beyond the resident row before masking.
- Reduce block maxima by ordered integer keys, preserving `+0 > -0` exactly.
- Give the newest visible block a unique internal priority key above `+inf`
  during publication/screening. This does **not** modify token scores. Thus
  publication is mandatory even if earlier blocks have positive infinity.

## Contract

`scores` is FP32 `[rows,width]`, `lengths` is INT32 `[rows]`. Lengths clip to
`[0,width]`; the caller must mask positions at or past each clipped length to
negative infinity. NaNs are unsupported; positive infinity is supported.

`block_topk(scores, lengths, k=..., block_size=..., publish=0)` returns INT32
`[rows,k]`: selected logical token IDs ascending, lower IDs preferred at equal
ordered FP32 keys, `-1` suffix padding, negative infinity excluded. Signed
zeros are not folded. With `publish > 0`, return `(token_ids, block_ids)`;
block IDs are ascending with `-1` suffix, newest nonempty visible block
mandatory, remaining blocks selected by maximum and low-ID ties. This forced
block never forces tokens into the selected token set.

`is_supported(..., publish=...)` checks native TPU/SparseCore availability,
lane/HBM alignment, useful pruning, and resident/streamable scratch plans.
Callers must retain their existing fallback when it returns false. Unsupported
inputs raise `ValueError` from the primitive; it does not provide a CPU
numerical implementation. There is intentionally no inner JIT: callers may
JIT externally and pass the existing scheduling-group ID.

The primitive is kept under the existing plugin implementation owner. A
logical attention namespace/inventory adapter is deferred: current registry
metadata cannot truthfully represent Pallas provenance and OOT TPU capability.
No shared-core TPU branch or misleading backend registration is added.

## Reproduce

With the repository's pinned TPU dependencies installed, from repository root:

```sh
PYTHONPATH=python:. python test/registered/unit/kernels/test_block_maximum_topk_admission.py
PYTHONPATH=python:. python test/registered/kernels/ops/attention/test_block_maximum_topk.py
PYTHONPATH=. python test/manual/kernels/bench_block_maximum_topk.py --rows 16 --widths 16384 131072 --repeats 20
PYTHONPATH=. python test/manual/kernels/bench_block_maximum_topk.py --rows 16 --widths 65536 65408 --publish 128 --repeats 20
```

The runner compares exact IDs against an independent NumPy ordered-bit oracle
and existing ordered SparseCore selection on identical scores. Compilation,
first execution, and synchronized steady-state median/p95 are separate.
Publication adds work to the new variant; token-only old timings must not be
reported as a like-for-like publication speedup. Score generation, integrated
score-plus-selector latency, TorchCPU-to-TPU fast-path activation, and model
E2E/GSM8K are **not** established by this microbenchmark.

CPU-hosted TPU export is currently deliberately not admitted by
`is_supported`: it returns false on CPU default backend. Existing export
fallback behavior is unchanged. Native numerical/compile evidence cannot be
called proof of serving activation. A target-aware export seam and actual
integrated A/B are prerequisites to a later default selector change.
