# Private channelwise checkpoint compatibility: proposed implementation

Status: design only. Actual mounted checkpoint headers have not been inspected;
no loader support or TPU correctness is claimed. Source converter evidence is
not a substitute for checking the exact checkpoint used by the experiment.

## Gating and placement

Use the existing `TpuSRTPlatform.get_quantization_config` seam to select a
plugin-owned Fp8Config subclass for `fp8`, delegating ordinary configs unchanged.
Keep the public quantization name and parent Fp8Config contract. No core registry
mutation, extra shared TPU branch, or loader allowlist change is needed.

`weight_block_size=null` + dynamic + e4m3 is NOT a unique private-layout marker.
Require explicit private-layout opt-in plus validated headers/manifest. Scalar
FP8, online FP8, block FP8 and MXFP8 must retain their existing behavior. Do not
change Flash block-to-channel conversion or introduce a misleading fake block
size to make loaders accept per-channel scales.

Header gate: each opted-in matrix is 2D E4M3 [N,K] with paired FP32
`weight_scale_inv[N,1]`; validate ignored/excluded layers and actual model
architecture. Scale is a multiplier (W=Q*S), not a reciprocal. Unsupported
quantized embeddings/head or intentionally unquantized Indexer projections
must fail explicitly rather than load FP8 bytes as BF16 without scales.

## Dense

Allocate canonical `weight_scale_inv` as output-channel scale [N_local,1], with
output_dim=0 and NO input_dim. Preserve normal weight allocation/metadata. The
legacy loader selected for a plugin method can partition column rows and
merged gate/up offsets while replicating channel scales for row-parallel K
shards. Do not add the plugin class to the shared v2 name allowlist blindly.

Expose a same-Parameter `weight_scale` alias before model post_load_weights:
Deepseek's raw kv_b reader runs before quant-method postprocessing and accesses
that name. Register the canonical inv name first, verify named-parameter
identity/dedup behavior, and remove the previous scalar parameter. Alternative
alias designs require equal loader/reader evidence.

After load, transpose FP8 weight [N,K] to [K,N] without requantization; preserve
channel scales and use existing apply_tpu_fp8_linear. input_scale=None.

The existing kv_b post-load reader supports channel scales, but intentionally
converts absorbed-MLA weights to tensor quantization for bmm. Test that operation
rather than incorrectly requiring all downstream kv_b representations to retain
per-channel quantization.

## MoE

Replace scalar scales with canonical inv channel scales [E,2I_local,1] and
[E,H,1], using existing CHANNEL scale loader attrs, not BLOCK. Gate/up scales
partition output I by MoE TP and pack at their destination offsets; down scales
replicate because down weights partition K. Preserve EP global-to-local mapping,
weight_loader attrs and staged expert loading semantics.

Post-load squeeze trailing scale dim and call existing TPU preparation with
block_quant=False. It transposes weights and formats scales for gmm_v2. Bypass
parent scalar FP8 preparation: that path merges gate/up maxima and requantizes.
Existing helper may round scales to BF16 unless its existing FP32-scales flag is
enabled; record this numerical difference, do not silently change the default.

## Required checks before serving

- Positive layout detection; missing/wrong shape/dtype scale rejection; scalar
  null-block checkpoint falls through unchanged.
- Distinct row scales across column TP ranks, merged gate/up and q_a/kv_a concat.
- Row-parallel replicated scales; MoE TP w1/w3 slices and unsliced w2 scales.
- EP local mapping; staged scale/weight arrival permutations.
- Reconstruct W=Q*S before/after packing/transpose; test runtime activation
  quantization separately from a BF16-input mathematical reference.
- kv_b channel-to-tensor conversion and absorbed layout reference.
- Model-ignored projections/exclusions; reject unsupported quantized consumers.
- Existing scalar/block/online and Flash contracts remain unchanged.

CPU packing/reference tests do not establish TPU lowering, collectives, compile
stability or performance. Native device load, model quality and Flash regression
remain gates. OOT sparse-serving limitations remain unchanged by quant support.
