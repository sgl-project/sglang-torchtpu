# GLM5.2 source kernel-route comparison

## Scope and exact baselines

**Conclusion: not kernel-equivalent.** Shared algorithm names/Pallas lineage do
not establish identical dispatch, fusion, activation precision, or accumulator policy.
No dump or extra accelerator run was performed for this source audit.

- Torch original E2E commits: chat `552e93ab66`, long baseline `2160012cc6`;
  final original snapshot `ca8dbe8e62`. Draft rebased snapshot `e5c0854800`
  on main `97afaea93765ce3546806481cab3b5c3bc0beb09`.
- Actual JAX repository: https://github.com/primatrix-glm/sglang-jax-private
  at `3695225808f0293cae422ea7adf55eddacadd43d`, branch
  `develop/pr72-rebased-fp8-benchmark-20260920`. Remote branch SHA checked
  2026-11-16 and equals this commit. The public sglang-jax repo does NOT have
  this object. Earlier evidence metadata naming the public repo is not accurate
  source provenance; the actual Falcon source manifest points to the private repo.
- T paths below are relative to the Torch repository; J paths to the private JAX
  repository. Line ranges refer to those exact inspected trees. They describe
  source/config eligibility, **not an instruction trace or proof of chosen tiles**.
- Torch candidate: `torch_staged_loading_candidate.json`, no opt-in qblock/small-M
  flags. Historical diagnostic candidates must not be substituted for this route.

## Top-level routing

T `python/sglang/srt/models/glm4_moe.py:1285-1286,1423` registers
`GlmMoeDsaForCausalLM`, inheriting `DeepseekV2ForCausalLM`.
T `sglang_tpu/tpu_platform.py:1283-1289` selects `TpuFp8Config` for FP8;
`sglang_tpu/glm52_channelwise_fp8.py:35-95` preserves upstream config unless
explicit `glm52_channelwise_fp8` checkpoint layout is present, then selects
specialized Linear/MoE loading methods. Other consumers, including KV, delegate.
The layout marker changes ABI interpretation, not an implementation parity switch.

## Route summary

| Component | Torch successful candidate, source-selected | JAX pinned delivery | Comparison |
|---|---|---|---|
| Dense quantized linear | `apply_tpu_fp8_linear` -> dynamic FP8 activation -> `torch._scaled_mm` | hybrid W8A16 -> shape-tuned Pallas matmul or explicit dot fallback | **Different activation precision and dispatch** |
| Routed MoE | inherited `Fp8MoEMethod.apply` -> `apply_tpu_fp8_moe` -> `fused_moe_call` -> sort/SC gather -> two `gmm_v2` -> SC combine | `FusedEPMoEV2` -> fused-v2 distributed scatter/compute/gather | **Different kernel and communication pipeline** |
| Shared expert | separate shared MLP; shared fusion disabled | shared weights inside fused-v2 EP kernel | **Different fusion/overlap and dense activation policy** |
| MoE numeric policy | default BF16 stored scales and GMM accumulator, FP8 activation blocks | W8A8 with FP32 gate/up and weighted-gather accumulation | **Not numerically equivalent from dtype labels alone** |
| Sparse attention | default per-query fused sparse MLA for prefill and decode; qblock opt-in off | DSA precision `fp8-qk-pv`, split path with explicit FP8 QK/PV | **Different kernel family / precision path** |
| Indexer/top-k | `dsa_indexer_step` via Torch TPU bridge | private DSA gather/indexer/top-k path | See detailed supplement below; not established equal |
| Norm | native RMSNorm decomposition | shape-gated packed MoE input RMSNorm for fused-v2 | **Different selected optimization** |
| RoPE | native TPU coefficient gather/rotation operations | shape-gated packed prefill RoPE, ordinary fallback otherwise | **Different prefill implementation** |

## Dense linear: W8A8 versus hybrid W8A16

T `sglang_tpu/glm52_channelwise_fp8.py:104-168` loads E4M3 weights and
multiplicative FP32 output-channel scales, transposes weights, and calls
`apply_tpu_fp8_linear`.
T `sglang_tpu/fp8_linear.py:241-256` dynamically row-quantizes activations to FP8;
`:259-322` pads M to16, calls `torch._scaled_mm` with unit scales/FP32 output,
then applies activation and weight scales externally and casts BF16.
Small-M Pallas is optional (`:293-298`), default FALSE
(`sglang_tpu/envs.py:95-97`). Therefore default regular quantized linear is W8A8.

J `benchmark/glm52/delivery/serve/common.sh:90` chooses
`python/sgl_jax/srt/utils/quantization/configs/fp8_glm52_static_per_channel_moe_w8a8_linear_w8a16.yaml:15-26`:
regular Linear activation dtype null (W8A16), MoE activation dtype FP8 (W8A8),
per-channel matmul backend Pallas. J `python/sgl_jax/srt/layers/linear.py:467-506`
passes `quantize_activation=False`.
J `python/sgl_jax/srt/kernels/quantized_matmul/kernel.py:181-223` requires exact
shape/dtype tuned registry hit for Pallas; otherwise dot fallback. J
`python/sgl_jax/srt/kernels/quantized_matmul/quantized_matmul_kernels/kernel.py:99-122`
uses BF16 activation x FP8 weights with FP32 preferred accumulation, then weight
scale and BF16 output. Do not describe all JAX linear shapes as always Pallas.

## MoE: no fused-RS in this specialized Torch route

T specialized `process_weights_after_loading`
(`sglang_tpu/glm52_channelwise_fp8.py:206-224`) only calls
`prepare_tpu_fp8_moe_weights`; it bypasses parent's fused-RS preparation.
`Fp8MoEMethod` defaults `handles_moe_reduction=False`
(T `python/sglang/srt/layers/quantization/fp8.py:1428-1434`), and inherited
`:3049-3070` selects ordinary `apply_tpu_fp8_moe`, not fused-RS.
An auto runner name is not evidence that fused-RS is selected.

T `sglang_tpu/fp8_moe.py:145-179,492-524` formats [E,K,N] weights and
[E,1,1,N] scales, default scales BF16 (`sglang_tpu/envs.py:427-438`).
`:532-600` casts routing weights BF16, IDs int32, then calls `fused_moe_call`.
Standard dispatcher keeps global expert IDs and does not implement JAX-style
A2A (`python/sglang/srt/layers/moe/token_dispatcher/standard.py:114-133,259-271`).
T `sglang_tpu/moe/pallas/fused_moe_op.py:642-748,871-930,963-994` performs fixed-row padding,
expert-ID sort, local EP window, SparseCore ragged gather, W13 and W2 grouped
matmuls, inverse sort, and SparseCore weighted gather/reduce with EP validity mask.
The two matmuls use `gmm_v2`; source FP8 dot preferred FP32 is immediately cast
to accumulator dtype (`sglang_tpu/moe/pallas/gmm_v2.py:625-660`), which defaults BF16
for quantized path (`:1581-1588`). FP32-scale opt-in alone is not FP32 accumulation.
Final cross-rank reduction belongs outside the kernel
(`python/sglang/srt/models/deepseek_v2.py:1475-1533`,
`python/sglang/srt/layers/moe/utils.py:876-925`). Shared experts are separate.

J `python/sgl_jax/srt/models/glm5_moe.py:1100-1133` builds `FusedEPMoEV2`
with shared experts; `python/sgl_jax/srt/layers/fused_moe.py:592-657` configures
W8A8, tuned configs, resharding, and `fused_ep_moe_v2`.
J `python/sgl_jax/srt/kernels/fused_moe/v2/kernel.py:690-885` implements
A2A scatter/send/wait and gather; `:1263-1435` FP32 gate/up accumulators,
`:1663-1754` FP32 weighted gather; `:1789+` shared expert overlaps routed scatter,
`:1938-1984` shared FP32 dots and dynamic FP8 shared-down activation quantization.
This is a separate distributed-kernel port, not a backend-string substitution.

## Norm and RoPE

T `python/sglang/srt/models/deepseek_v2.py:2183,2273,2779-2782` creates
regular RMSNorm; native implementation
`python/sglang/srt/layers/layernorm.py:821-870` computes statistics FP32.
T `python/sglang/srt/layers/rotary_embedding/rope_variant.py:512-545`
uses native TPU coefficient gather and fixed-matrix rotation; subclass still
depends on actual HF rope_scaling, so do not assume an unseen checkpoint setting.

J `python/sgl_jax/srt/layers/layernorm.py:101-143` selects packed
MoEInputRMSNorm only for H6144 BF16/device eligibility; GLM fused-v2 chooses it
at `python/sgl_jax/srt/models/glm5_moe.py:1231-1239`.
J `python/sgl_jax/srt/layers/packed_rope.py:28-54,85-112` selects
`packed_prefill_rope` only for supported BF16/64-dim/interleaved prefill,
otherwise ordinary RotaryEmbedding. This does not imply decode uses packed RoPE.

## Sparse attention: explicit gather and arithmetic routes

T `sglang_tpu/envs.py:106` defaults qblock FALSE;
`sglang_tpu/attention_pallas_mla_sparse.py:373-380,676-694` selects the same
per-query default op for prefill/decode. `:87-93,149-176` imports JAX and bridges
via torch_tpu custom_jax_kernel to `sparse_mla_attention_fused`.
Both implementations therefore use JAX/Pallas; this is not simply Torch versus
JAX arithmetic implementation.

T `sglang_tpu/attention/pallas/mla_sparse_fused.py:207-217,256-275` writes separate
NoPE/RoPE caches and calls sparse ragged paged attention, tensorcore cache layout,
gather chunk64. T `sglang_tpu/attention/pallas/mla_sparse/kernel.py:1234-1247,1332-1359`
batches gather via `dsa_gather`, not `native_sc_gather`, then per-query attention.
T `sglang_tpu/attention/pallas/mla_sparse/dsa_gather.py:73-107,255-281` uses 32-bit
SparseCore gather and independent indirect streams for the split caches.
T kernel `:395-425` unpacks FP8 KV without converting the returned KV to BF16;
`:229-269` performs mixed BF16-Q/FP8-KV einsums, FP32 softmax, unquantized FP32-P
PV with FP32 preferred result, a shared scale and BF16 output. Full selected K is
resident per query, unlike tiled online softmax. Actual MXU lowering of mixed
dtypes remains unverified; do not relabel this native FP8xFP8 QK/PV.

J `python/sgl_jax/srt/layers/attention/dsa_sparse_backend.py:222-274`
resolves DSA config and exact attention with fp8_q/fp8_pv.
J DSA config and kernel path locations are indexed in the audit provenance below.
`dsa_config.py:81-102` forces split (not HLO-overlap) pipeline for fp8-qk-pv even
if the overlap flag was requested. `:41-52` chooses decode tile defaults by Q,
prefill >=128 default (256,16,256); delivery extend override applies only >=128.
`exact_attention.py:295-323,546-618` routes to FP8 attention and bounded pipelined
split gathers. `fp8_attention.py:135-217` quantizes Q, unpacks per-token FP32 scale
bytes, uses native FP8 QK with FP32 accumulation, online softmax across K256 tiles,
absorbs KV scales into P and dynamically quantizes P to FP8 before FP8 PV/FP32
accumulation. Cache encoding, scale granularity, Q/P dtype, gather residency and
softmax tiling all differ. No throughput superiority is inferred from source.

## Indexer and top-k: FP8 indexer versus BF16 reference index cache

T `sglang_tpu/dsa_indexer_tpu.py:322-378` calls backend indexer;
`sglang_tpu/attention_pallas_mla_sparse.py:397-425` selects index_topk bridge.
T `sglang_tpu/attention/pallas/dsa_index_topk.py:217-259` dynamically FP8-quantizes
Q/K with UE8M0 scales, folds Q scale into head weights, writes key+scale-byte
records and calls streamindex_topk. `:35-60` sets page/query tiles and cache padding;
early exit is forcibly false. The streamindex kernel implements native FP8 QK,
then optimized prefill BF16 rounding for relu/weighted reduction; it calls a
separate multi-subcore shared-histogram SparseCore exact radix top-k.

J `python/sgl_jax/srt/models/glm5_moe.py:389-488` owns index-cache/select calls;
`dsa_indexer_ops.py:161-189` writes paged cache then computes/selects scores.
J `python/sgl_jax/srt/mem_cache/memory_pool.py:1259-1265` explicitly requires BF16
indexer cache for FP8 DSA. Thus FP8 attention does NOT imply FP8 indexer.
Private `indexer.py:25-38,450-457` uses Q-block32 at16 logical devices and
key-block4096; decode scorer resolves blocks around target22528 keys /2 sequences.
Private `topk.py:56-61,96-114` selects exact radix V2 on v7 eligible shapes,
otherwise V1. This is not Torch's sparsecore_topk implementation, despite both
being exact unordered radix-family selection.

### Audit provenance for kernel filenames

The following filenames refer to their unique locations in the pinned private
JAX kernel tree: `dsa_config.py`, `exact_attention.py`, `fp8_attention.py`,
`dsa_indexer_ops.py`, `indexer.py`, `paged_score.py`, `topk.py`.
Use the repository file index accompanying this document for exact full paths.
Torch streamindex is in `sglang_tpu/dsv4/streamindex_topk/`; top-k implementation
is indexed alongside its import-resolved source.
Check full paths in that index before navigating: similar public upstream names
are not a substitute for these private source files.

## Evidence / extraction constraints

- Source parity and runtime success are different deliverables. Original runs
  prove functional native Torch E2E, not JAX arithmetic, tile or fusion parity.
- Rebase did not alter the audited core ABI/fp8_linear/fp8_moe/gmm_v2/model paths;
  nevertheless rebased head is not newly hardware-validated.
- Keep checkpoint ABI/loading, dense policy, attention/indexer, MoE numerics,
  distributed fusion and packed norm/RoPE as independent work items.
- Qblock row-budget experimental patch is NOT selected in successful E2E.
- No performance ratio should be published until input/cache/cadence and
  explicitly chosen numerical contracts are comparable.
