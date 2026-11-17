# GLM5.2 complete indexer benchmark

Boundary: resident prepared queries + already-updated index-key cache → exact unordered physical slots. All score/TopK/slot mapping work is inside one compiled module. Projection/RoPE/query quantization/cache writes and attention math are outside this explicitly bounded score module for every arm.

Arms:
- `torch_baseline_full`: production StreamIdxTC score + cooperative SparseCore TopK + original `prepare_gather_indices` mapping. Padded slots normalize to -1 for comparison.
- `torch_selector_only_full`: same production scorer/mapping with the existing opt-in selector-only patch.
- `jax_fp8_pipeline_full`: complete pinned JAX indexer with its score/TopK/mapping pipeline and native V1/V2 dispatch; read the SAME packed FP8 cache bytes after bounded page DMA. No whole-cache conversion, no replacement serial loop.
- `jax_native_bf16_reference`: private original JAX package, imports renamed only; a BF16 cache holds the exact same dequantized FP8 keys. This additional arm isolates cache-ABI adaptation, not serving equivalence.

Source: sgl-project/sglang-jax c948bdda95208e68a474c8088937347d08fe8937, Apache-2.0. source-provenance.json contains original SHA256s; private `glm52_jax` imports are the only original-source edits. fp8-adaptation.patch records the cache ABI changes for `glm52_fp8`; original V1/V2/mapping/TC-SC pipeline math are preserved.

GLM geometry: H32/D128/K2048/page64/capacity92160; context53248/90000; MTP6/12/18/24 rows, decode4x1, prefill2048; shuffled pages. Synthetic fixture data, BF16 query/gate entry values, query FP8 scales folded into gate weights as in production. A NumPy quantizer builds common prepared inputs. No claim that inputs are actual model score dumps. One local JAX device participates per call on a 4-physical-chip allocation.

30 interleaved unprofiled synchronized samples; separate process profiles five complete calls for q6, q24 and prefill2048 at context90000. Compilation and warmup excluded. Correctness: all-row physical membership/causality/unique/cardinality; up to eight NumPy score rows per case with existing 2% cutoff tolerance; ragged/padded tails separately checked. No model E2E.

Run through `run.sh`, using the exact candidate-declared absolute Python environment from the baseline bootstrap. The actual measurement process writes runtime/import/library evidence. Production defaults are unchanged.
