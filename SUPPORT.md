# SGLang-torchtpu: supported models and features

This page is the human-curated view of what SGLang-torchtpu supports on TPU. It assigns every model and feature a tier and says what that tier means. The test evidence behind each row is in [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md), which is generated from the CI test registrations on every merge and never edited by hand.

Rule for this page: a row may not claim a higher tier than its tests allow. Anything with code but no green on-device test is **Experimental**, no matter how complete the code looks.

## Tiers

| Tier | Meaning | What backs it |
| --- | --- | --- |
| **Quickstart** | Small model used in the docs and the quickstart. Fast to download and launch. | Runs on every PR (CI tier in the matrix). |
| **Optimized** | Tuned recipe, nightly accuracy and performance tracking, the configurations we recommend for production. | Nightly accuracy and perf suites, both green. |
| **Supported** | Serves correctly in the listed topology. Performance is measured but not tuned, or tuning is in progress. | Green nightly e2e or accuracy test. |
| **Experimental** | Code path exists. Either no on-device test, a disabled test, or a test that is not green. Expect rough edges and no accuracy guarantee. | Code only, or a registered test without a recent green. |
| Unsupported | Rejected at startup with an error, or no code path. Listed at the end so nobody has to guess. | An explicit check in the platform. |

Status words inside the tables use the matrix vocabulary: **CI** = every PR, **Nightly** = nightly on the project cluster, **Nightly (ext)** = nightly on an external cluster, **Manual** = verified by hand on the stated date.

## Summary

The tables below are the short view. The sections that follow give per-row topology, notes, and the matrix status words.

### Models

| Model | Quantization | Topology | Tier |
| --- | --- | --- | --- |
| gpt-oss-120b | MXFP4 | 1 host, 2 hosts, 2P+2D | Optimized |
| Qwen3-Coder-480B-A35B | FP8 | 2 hosts, 1 host, 2P+2D | Optimized |
| Qwen3.5-397B-A17B (text, VL) | FP8 | 1 host | Optimized |
| GLM-5.3-Flash | FP8 | 1 host | Optimized |
| Kimi-K3 | MXFP4 | 4 hosts | Optimized |
| gpt-oss-20b | MXFP4 | 1 host | Quickstart |
| Qwen3-0.6B | bf16 | 1 chip, 1 host | Quickstart |
| Qwen3-30B-A3B | FP8 | 1 host | Supported |
| Qwen3-VL-8B | bf16 | 1 host, 2P+2D | Supported |
| GLM-5.2 / GLM-5.3 | FP8 | 1 host | Supported |
| DeepSeek-V3.1, DeepSeek-R1-0528 | FP8 | 2 hosts | Supported |
| Kimi-K2.6 | INT4 W4A16 | 2 hosts | Supported |
| MiMo-V2.5-Pro | FP8 | 2 hosts | Supported |
| Gemma 4 (31B, 26B-A4B) | bf16 | 1 host | Supported |
| Qwen3-Coder-30B-A3B | FP8 | 1 host | Experimental |
| Qwen3.8-27B | bf16 | 1 host | Experimental |
| DeepSeek-V4 | FP8 | — | Experimental |
| Wan2.2-TI2V-5B (diffusion) | bf16 | 1 chip | Experimental |

### Features

| Area | Feature | Tier |
| --- | --- | --- |
| Attention | Ragged paged attention v3 (default), fp8 KV cache, attention sinks, sliding window | Optimized |
| Attention | Paged MLA (DeepSeek), GDN hybrid linear attention (Qwen3.5) | Supported |
| Attention | Batched RPA, KDA (Kimi-K3) | Experimental |
| MoE | Grouped matmul in bf16 / FP8 / FP4 / INT4, expert parallel | Optimized |
| MoE | Fused reduce-scatter, SparseCore gather/reduce | Supported |
| Quantization | FP8 channelwise, MXFP4 | Optimized |
| Quantization | FP8 blockwise, INT4 W4A16 | Supported |
| Quantization | NVFP4 MoE, MXINT8 KV cache | Experimental |
| Parallelism | Tensor parallel, expert parallel, multi-host launch | Optimized |
| Parallelism | DP attention | Supported |
| Parallelism | Prefill context parallel, decode context parallel, pipeline parallel | Experimental |
| Speculative decoding | EAGLE3 | Optimized |
| Speculative decoding | DFlash | Supported |
| Caching | Prefix cache, chunked prefill, HiCache KV offload to host | Optimized |
| Disaggregation | Prefill/decode over raiden with TP resharding | Optimized |
| Disaggregation | Native TPU transfer, decode-side radix reuse | Supported |
| Runtime | torch.compile with the TPU backend, compile bundles, direct tracer, overlap scheduling | Supported |
| Runtime | Out-of-tree plugin mode, zero-recompile serving check | Experimental |
| Multimodal | Vision-language models (Qwen3-VL, Qwen3.5-VL, Kimi MoonViT, Gemma 4 vision) | Supported |
| Generation | Diffusion (Wan2.2) | Experimental |
| Serving | Prompt and input logprobs | Supported |
| Serving | Structured output, function calling | Experimental |

## Hardware

| Generation | Single-host | Multi-host | Status |
| --- | --- | --- | --- |
| TPU7x (Ironwood) | 4 chips (tp=8 with 2 TensorCores per chip) | 2 hosts, tp=16; 4 hosts for 2P+2D disaggregation | CI + Nightly |

All kernels are tuned for TPU7x. The Pallas kernels check for TPU v4 or newer, but no other generation is tested or tuned.

## Models

| Model | Quantization | Topology verified | Tier | Notes |
| --- | --- | --- | --- | --- |
| gpt-oss-120b | MXFP4 | 1 host tp8+ep8; 2 hosts tp16; 2P+2D | **Optimized** | Flagship. Nightly accuracy (gsm8k, gpqa, aime25, mmlu-pro), four perf configs, PD, spec decode, KV offload, resharding. |
| Qwen3-Coder-480B-A35B | FP8 channelwise | 2 hosts tp16/ep16/dp2; 1 host; 2P+2D | **Optimized** | Flagship. Nightly gsm8k, agentic tool-call workload, PD resharding, benchmark matrix. |
| Qwen3.5-397B-A17B (text and VL) | FP8 | 1 host | **Optimized** | GDN hybrid linear attention. Nightly MMMU-Pro and perf. |
| GLM-5.3-Flash | FP8 | 1 host tp8 | **Optimized** | DeepSeek Sparse Attention. Serving, MTP, and decode-megakernel recipes. No registered test. |
| gpt-oss-20b | MXFP4 | 1 host | **Quickstart** | PR gate: spec decode, spec + DP attention, prefix cache + KV offload, PP smoke. Also nightly accuracy and perf. |
| Qwen3-0.6B | bf16 | 1 chip, 1 host | **Quickstart** | PR gate: smoke, prompt logprobs, compile tracer, compile bundles, plugin-mode smoke. Single-host only (tp>1 OOMs). |
| Qwen3-30B-A3B | FP8 | 1 host (PCP) | **Supported** | PR-gate e2e for prefill context parallel. Multi-host smoke retired. |
| Qwen3-VL-8B | bf16 | 1 host; 2P+2D | **Supported** | EAGLE3 spec decode, MMMU-Pro nightly (report-only). Vision encoder compiled with bucketed shapes. |
| DeepSeek-V3.1, DeepSeek-R1-0528 | FP8 | 2 hosts tp16 | **Supported** | DeepSeek-V3 architecture, paged MLA attention. Nightly accuracy and perf for both; the R1 accuracy suite is intermittently red. |
| Kimi-K2.6 | INT4 W4A16 | 2 hosts | **Supported** | Nightly perf green; MMMU accuracy shards red at the time of writing. |
| MiMo-V2.5-Pro | FP8 | 2 hosts dp2 | **Supported** | Nightly gsm8k. |
| Qwen3-Coder-30B-A3B | FP8 | 1 host | **Experimental** | Recipe available. PR-gate test disabled pending runner capacity; intended Quickstart model. |
| Qwen3.8-27B | bf16 | 1 host tp8 | **Experimental** | Dense hybrid GDN, vision on. Nightly AIME26, GPQA, MMMU-Pro; MTP and DFlash recompile e2e. |
| GLM-5.2 / GLM-5.3 | FP8 | 1 host tp8+ep8 | **Experimental** | DSA sparse-MLA kernels, shared with GLM-5.3-Flash. The `glm_5_3.sh` recipe was removed as an orphan in #1432. No registered test. |
| DeepSeek-V4 | FP8 | — | **Experimental** | Compressor, indexer, sparse attention, SparseCore top-k landed. No on-device test. FP4 indexer and online compression rejected. |
| Kimi-K2.5 | INT4 | shares the K2.6 path | **Experimental** | MoonViT vision tower compiled per image. No K2.5-specific test. |
| Kimi-K3 / K3-Linear | bf16 | 1 host (tiny config) | **Experimental** | Largest per-model TPU code path (KDA attention). Only a dummy-weight compile smoke runs; no real checkpoint served in CI. |
| Gemma 4 (31B, 26B-A4B) | bf16 | 1 host | **Experimental** | Text and vision. Accuracy and perf suites registered; red at the time of writing. Moves to Supported when green. |
| Wan2.2-TI2V-5B (diffusion) | bf16 | 1 chip; multi-host collectives | **Experimental** | Runs through the generic diffusion engine; external nightly only, no recent green. |
| Llama, Qwen3 dense (other sizes), Qwen3-MoE (other sizes), Kimi-Linear | — | — | **Experimental** | Generic TPU path with no model-specific test. |

## Features

### Attention

| Feature | Single-host | Multi-host | Tier | Notes |
| --- | --- | --- | --- | --- |
| Ragged paged attention v3 (`pallas_tpu_rpa_v3`, default) | CI | Nightly | **Optimized** | Fused 5-D KV pool, fp8 KV by default, attention sinks. |
| Batched RPA (`pallas_tpu_rpa_batched`) | CI (unit, PCP e2e) | — | **Experimental** | Required backend for prefill and decode context parallel. No sinks, no spec decode. |
| Paged MLA (`pallas_tpu_mla`) | CI (unit) | Nightly (DeepSeek) | **Supported** | Absorbed path. fp8 KV opt-in until accuracy is validated. |
| GDN / hybrid linear attention | Nightly (Qwen3.5) | — | **Supported** | Mamba-style state pool; no host offload of GDN state. |
| KDA (Kimi-K3) | CI (unit) | — | **Experimental** | Kernel tested; no real-weight e2e. |
| Sliding window, hybrid SWA, attention sinks | Nightly (gpt-oss) | Nightly | **Supported** | Sinks on RPA v3 only. |
| Flash attention (`pallas_tpu_flash`) | — | — | **Experimental** | Original backend, flat pool, no fp8 KV scale. Kept for comparison. |

### MoE and quantization

| Feature | Single-host | Multi-host | Tier | Notes |
| --- | --- | --- | --- | --- |
| MoE grouped matmul (bf16, FP8, FP4, INT4), expert parallel | CI (unit) + Nightly | Nightly | **Optimized** | gmm_v2 kernels. |
| MoE fused reduce-scatter | CI (unit) + Nightly | Nightly | **Supported** | EP only; expert-count and activation constraints. |
| SparseCore gather/reduce | CI (unit) | — | **Supported** | |
| FP8 channelwise and per-tensor | Nightly (480B, MiMo) | Nightly | **Optimized** | |
| FP8 blockwise | Nightly | Nightly | **Supported** | Dynamic activations only. |
| MXFP4 (gpt-oss) | Nightly | Nightly | **Optimized** | Native packed dequant in the kernel. |
| INT4 W4A16 MoE | CI (unit) | Nightly (Kimi-K2.6) | **Supported** | |
| NVFP4 MoE | CI (fold test) | — | **Experimental** | No on-device numerics test. Gated SwiGLU experts only. |
| MXINT8 KV cache | CI (unit, CPU) | — | **Experimental** | |

### Parallelism

| Feature | Single-host | Multi-host | Tier | Notes |
| --- | --- | --- | --- | --- |
| Tensor parallel | CI | Nightly | **Optimized** | |
| Expert parallel | Nightly | Nightly | **Optimized** | |
| DP attention | Nightly | Nightly | **Supported** | attn-TP > 1 needs `--enable-dp-lm-head`. |
| Prefill context parallel (PCP) | CI (Qwen3-30B-A3B e2e) | — | **Experimental** | Batched RPA, zigzag strategy, torch.compile with the dynamo tracer required. No spec decode or DP attention with it. |
| Decode context parallel (DCP) | — | — | **Experimental** | Size 1 or 2; batched RPA; no PP or spec decode with it. |
| Pipeline parallel (PP) | CI (gpt-oss-20b; tests disabled pending torch_tpu) | — | **Experimental** | One device per stage; PP × PCP stages. No DCP, spec decode, or dynamic chunking with it. |
| Multi-host launch (Ray, 2 and 4 hosts) | — | Nightly | **Optimized** | Node normalization and leader forwarding tested nightly. |

### Speculative decoding

| Feature | Single-host | Multi-host | Tier | Notes |
| --- | --- | --- | --- | --- |
| EAGLE3 | CI (20B) + Nightly (120B) | Nightly | **Optimized** | topk=1 only, no rejection sampling. |
| DFlash | Nightly | Nightly | **Supported** | Draft needs an lm_head. |

### KV cache and caching

| Feature | Single-host | Multi-host | Tier | Notes |
| --- | --- | --- | --- | --- |
| Prefix cache, chunked prefill | CI | Nightly | **Optimized** | |
| HiCache KV offload to host (torch_tpu io, raiden) | CI (20B) + Nightly | Nightly | **Optimized** | Batch-granular loads. L3 storage tier not implemented on raiden. |
| fp8 KV cache | CI | Nightly | **Optimized** | Default on the RPA backends. |

### Disaggregated serving

| Feature | Single-host | Multi-host | Tier | Notes |
| --- | --- | --- | --- | --- |
| PD disaggregation, native TPU transfer | Nightly | — | **Supported** | Not for MLA models. |
| PD disaggregation, raiden transfer | Nightly | Nightly (2P+2D) | **Optimized** | TP resharding (16 prefill shards to 8 decode shards), prefix cache and KV offload on both roles. |
| Decode-side radix reuse under PD | Nightly | — | **Supported** | |

### Compilation and runtime

| Feature | Single-host | Multi-host | Tier | Notes |
| --- | --- | --- | --- | --- |
| torch.compile (backend `tpu`), compile bundles | CI | Nightly | **Supported** | Opt-in. Cold start is dominated by per-bucket compilation; use the compile cache. |
| Direct compile tracer | CI | — | **Supported** | |
| Overlap scheduling | CI | Nightly | **Supported** | Single PJRT queue; no multi-stream overlap. |

### Multimodal and generation

| Feature | Single-host | Multi-host | Tier | Notes |
| --- | --- | --- | --- | --- |
| Vision-language (Qwen3-VL, Qwen3.5-VL, Kimi MoonViT, Gemma 4 vision) | Nightly (Qwen3-VL, Qwen3.5) | Nightly (Qwen3-VL PD) | **Supported** | Compiled vision encoders with shape buckets; host-side embedding scatter by default. |
| Diffusion (Wan2.2) | Nightly (ext) | Nightly (ext, collectives) | **Experimental** | Generic diffusion engine; no TPU-specific kernels. |
| Prompt and input logprobs | CI (0.6B) | — | **Supported** | Env-gated. Not under DP attention or PP. |
| Structured output / grammar | — | — | **Experimental** | Works through the eager sampler path; excluded from the spec-decode fast path. No TPU test yet. |
| Function calling | — | — | **Experimental** | Same as structured output. |

## Maintaining this page

- [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md) is regenerated by `scripts/ci/tpu/support_matrix.py` from the `register_tpu_ci(...)` markers. When a row here says CI or Nightly, that file must show a matching cell.
- Raising a tier needs a test: register the test with topology and model labels, let it go green, then edit this page in the same PR.
