# Block-maximum top-k validation — 2026-11-12

## Decision

Ship a standalone native JAX primitive; **do not activate the default FP8
selector**. Exact native numerical tests pass and modest selector-only gains
exist, but integrated scorer latency, equivalent publication baseline, and
CPU-hosted Torch export activation are not established. No model/serving or
GSM8K claim. No changes to scoring, candidates, schemas, allocator, or models.

## Identity and environment

- Main base `9cb500543ca9dbe059ff8bcc82dbe2e3026d36d3`.
- Epic `dc1cd126e71c2248f3fbf2bef0d8d24e1e3768d7`.
- Final modified-file hashes: [source receipt](source-receipt.json).
- Own Falcon experiment `exp-zri0ub2kbw`, artifact `art-g4l733ea0s`;
  SUCCEEDED, active 03:34:23–03:45:39 UTC, 11m16s; holding allocation ended.
- Cluster `gke-tpu-train-us-central1-2-prod`, one v7x 2x2x1 host (4 chips /
  8 visible devices); single local JAX device used per invocation.
- JAX/jaxlib 0.11.2, libtpu 0.0.50, TorchTPU
  0.2.1.dev20261028201649, torch 2.11.0+cpu, Python 3.12.12.
  Exact [runtime receipt](runtime.json).
- Brian historical completed experiment `exp-lgxh0z3ujh` supplied the runtime
  reconstruction recipe, not performance or correctness credit for this port.
  Base `us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.9.0-rev1`
  tag is mutable; immutable image digest was unavailable. A fresh venv installs
  repository TPU pins. Mounted TorchTPU wheel checksum:
  `6e12816e92f9a5d9649358541f332f06583ccdfee5b6a74434dbf2141757fd82`.
- Exact main archive plus hash-checked overlay delivered through Falcon cp;
  no source push required. No other user's resources modified.

## Correctness / compilation

- Final CPU/resource contract suite: **3/3 passed**, executed on the pinned
  remote stack; CPU backend rejection explicitly checked. Local minimal JAX
  oracle/admission assertions passed, but full macOS registered imports remain
  blocked by missing unrelated `datasets` dependency.
- Final native SparseCore suite: **3/3 passed**, 16.98 seconds. Resident dual
  buffer, resident single buffer, streamed dual buffer; 97 rows exercise score
  slots/output DMA reuse. k=128/512, block=8/16, publication=0/128.
- Full-array integer-oracle equality: low-ID ties, signed zeros within and
  across blocks, empty/short/clipped lengths, partial visible blocks,
  masked/all negative infinity, positive infinity, forced latest low score
  independent of token top-k. Unsupported HBM-aligned partial SC-span shape
  (width2176/k128/block16) rejected.
- Final CPU-hosted real `tpu_info_for_cpu_export` context confirms native
  primitive admission remains **false**, intentionally. This is boundary
  verification, not successful exported SparseCore execution or serving use.
- Ruff/Black/syntax checks pass; shared-core TPU debt guard passes without
  baseline increase. `support_matrix.py --strict` has **9 pre-existing
  registration problems**, reproduced against the original checkout; no new
  registration problems. Not reported as strict-green.

## Same-score selector-only A/B

Final source, 97 rows, k512/block8/publish0. Mixed deterministic adversarial
rows (including short lengths, ties, all-inf and signed zeros), seed20261112.
One compile and first execution reported separately; 50 synchronous
`block_until_ready` samples per arm, no profiler. New then old, not interleaved;
these host-observed timings include launch/synchronization overhead and drift.
Old is existing ordered `streaming_topk`, `exclude_neg_inf=True`, scan block8192,
reconstruct_threshold2048. **No claim of universal/default-serving speedup.**

| Width | Plan | Old median / p95 µs | New median / p95 µs | Median ratio old/new |
|---:|---|---:|---:|---:|
| 16,384 | resident, 2 buffers | 223.14 / 236.98 | 216.69 / 230.22 | 1.030x |
| 65,536 | resident, 1 buffer | 308.59 / 335.39 | 257.55 / 273.10 | 1.198x |
| 131,072 | streamed, 2 buffers | 352.81 / 364.95 | 317.83 / 372.34 | 1.110x |

New compile times 1.02–1.09s, old0.90–2.35s. First execution1.66–2.07ms.
131K new tail is worse despite lower median. Earlier rows16 measurements at
16384/65408/65536/131072 also pass exact IDs; retained raw, not substituted
for final source results. Publication128 is extra work (roughly259–301µs in
that initial run) and has no equivalent old-publication benchmark here.

## Memory and limitations

FP32 input HBM is4*rows*width bytes (6.06/24.25/48.5MiB for final widths).
Outputs rows*k*4 bytes. VMEM planner accounts for actual explicit scratch,
publication staging and resident fetch lane; compiler/native success is not a
measured peak-memory profile. Score materialization is unchanged. No measured
scorer+selector total, runtime memory census, 256K/1M full matrix, loop scheduling
metadata integration, or full CPU-to-TPU lowering activation. No hardware
profile captured; allocation's Falcon type PROFILE is not proof of a trace.

## Reproduce and raw evidence

See the implementation contract [documentation](../../../../sglang_tpu/kernels/deepseek_v41/index_selection/BLOCK_TOPK.md).
Raw logs and JSONL here are intentionally small, non-secret task evidence;
compiler dumps/large profiles are not checked in. Re-run on a new pinned source
and runtime after any integration or version change. Full upstream pipeline
benchmark is a separate follow-up, not part of this primitive acceptance.
