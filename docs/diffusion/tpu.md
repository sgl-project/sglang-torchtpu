# TPU (torch_tpu) Support

SGLang Diffusion runs on Google TPU through
[torch_tpu](https://github.com/google-pytorch/torch_tpu), the PyTorch
PrivateUse1/PJRT backend. This page documents how to run, how the
integration is structured, its current validated state, and the known
torch_tpu limitations with their in-tree workarounds.

This is the **torch_tpu** path for the diffusion engine
(`sglang.multimodal_gen`). The separate JAX LLM engine is documented in
`docs/platforms/tpu.md`; the srt (LLM) torch_tpu path is documented in
`benchmark/tpu/README.md`.

## Quick start

```bash
# torch_tpu wheel (with its jax dependency) must be installed; the platform
# auto-detects it — no flag needed. SGLANG_DIFFUSION_PLATFORM_OVERRIDE=tpu forces it.
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "..." \
  --width 1280 --height 720 --num-frames 81 --num-inference-steps 40 \
  --num-gpus 8 --enable-cfg-parallel --ulysses-degree 4 \
  --dit-cpu-offload false --text-encoder-cpu-offload true \
  --vae-cpu-offload true \
  --save-output
```

Recommended environment:

```bash
export TPU_PREMAPPED_BUFFER_SIZE=8589934592   # 8 GiB pinned host pool for transfers
# Optional: torch_tpu eager mode for the denoising loop (default DEFER_AND_FUSE).
# DEFER_NEVER gives op-by-op execution for debugging / baselines.
export SGLANG_DIFFUSION_TPU_EAGER_MODE=DEFER_AND_FUSE
```

`--enable-torch-compile` is supported (see the execution-mode table below).

## Architecture

All TPU behavior lives in two modules, reached exclusively through
`Platform` hooks whose defaults preserve GPU behavior exactly:

| Module | Role |
|---|---|
| `runtime/platforms/tpu.py` | Platform: device lifecycle (one chip per process, slicebuilder env in the parent, rendezvous env at worker birth), real memory stats via the `torch.accelerator` allocator bridge, capability flags, denoising-loop execution-mode scope, DiT-eviction context for VAE decode, xprof profiler session |
| `runtime/layers/attention/backends/tpu_pallas.py` | Flash attention via the srt pallas kernel wrapped as a torch custom op (`jax_op`); non-causal, equal head counts; block sizes from the 720p sweep |

Two invariants the platform enforces (both regression-tested):

1. **Identity/memory queries never initialize PJRT.** Server-args
   processing calls them in the parent process before workers spawn; a
   parent-side PJRT init steals the workers' chips
   (`open(/dev/vfio/N): Device or resource busy`).
2. **GPU behavior is unchanged.** Every hook defaults to the pre-existing
   GPU code path; verified by running the full unit suite with the platform
   forced to the baseline and comparing the failure set against `main`
   (identical).

## Execution modes (validated 2026-07-29)

Wan2.2-A14B, 720p/81f, 8×TPU7x, CFG-parallel × Ulysses-4, torch_tpu built
from head `3597f02f`. Cold 4-step runs; the 4-step average is dominated by
first-step warmup (lazy trace for eager, dynamo + XLA compile storm for
compiled), so it measures warmup cost rather than steady throughput.

| Mode | Avg s/step (cold, 4 steps) | VAE decode | Status |
|---|---|---|---|
| Eager op-by-op (`DEFER_NEVER`) | 39.9 | 74.3 s | works |
| Eager fused (`DEFER_AND_FUSE`, default) | 40.6 | 74.3 s | works — reference output |
| `--enable-torch-compile` (serialization off, in-tree default) | 74.6 | 42.2 s | works — DiT partitions into 323 subprograms/rank at collective boundaries; decode ~1.8× faster after a compiled denoise |
| Compiled with torch_tpu stock serialization | — | — | **crashes** (`e0102: RuntimeProgramInputMismatch`) — see limitations |

Output parity: all green modes produce frame-statistics-identical videos
(compiled numerics differ within tolerance).

## Known torch_tpu limitations and in-tree workarounds

Reported upstream with reproducing unit tests; each row names the in-tree
handling so the workaround can be deleted when the upstream fix lands.

| torch_tpu limitation | In-tree handling |
|---|---|
| Serialized-compilation cache key degenerates to a per-process constant; any store keyed on it returns the wrong executable (e2e: `e0102` crash) | `build_torch_compile_kwargs` constructs `TpuBackend(enable_serialization=False)`; dynamo's in-process guards handle shape dispatch |
| `spmd_safe` regions are no-split zones for SplitCompiler; wrapping a whole 14B forward compile-time-OOMs (`arguments 282G`) | No spmd_safe wrapping; default materialize-collectives partitioning is the supported compiled shape for large models |
| P2P `experimental_send/recv` aborts on iterated ~80 MB messages | `supports_p2p_communication() → False`: spatially-sharded VAE decode disabled, decode runs unsharded |
| torch_tpu context managers graph-break whole frames under dynamo (b/537815330) | Execution-mode scopes applied at pipeline-stage level, never inside module forwards |
| Wrapped Python float scalars materialize as f64 program params (emulated f64) | Fix offered upstream (`is_wrapped_number` dtype handling); until merged, cost is accepted |
| `nn.Module` moves run at 2–10 GB/s (`batch_transfer` reaches 56 GB/s but requires shape-homogeneous pinned buffers) | DiT eviction for VAE decode uses plain `.cpu()`/`.to()` inside the platform decode context (~34 s per request on 2×14B experts) |
| VAE frame-loop recompiles per chunk shape under compile | `supports_vae_decode_torch_compile() → False`: VAE always decodes eagerly |
| torch.profiler cannot see TPU device activity | `create_aux_profiler` runs a torch_tpu xprof session alongside; captures >2 GiB abort (upstream protobuf cap) — keep captures to few steps |

## Testing

- Unit tests: `python/sglang/multimodal_gen/test/unit/test_tpu_*.py`,
  `test_platform_*.py` (CPU-safe; the pallas numeric-parity and eager-mode
  tests self-skip without a TPU).
- CI: registered as `test/registered/tpu/test_diffusion_tpu_platform.py`
  in the `stage-a-test-tpu-unit` suite; multimodal_gen TPU paths are in the
  TPU PR gate filter.
- Run everything locally:
  `pytest python/sglang/multimodal_gen/test/unit/ -k "tpu or platform"`.

## Bring-up record (2026-07)

The enablement was validated by a measurement campaign against torch_tpu
head (`3597f02f`), summarized here so numbers in this page have provenance:

- End-to-end 4-step and 8-step 720p/81f runs per execution mode, frame
  parity checked against reference outputs; performance identical to the
  pre-productionization prototype (40.6062 vs 40.6066 s/step).
- Standalone kernel/transfer benchmarks: pallas attention best block
  (2048, 1024) at 154.6 ms/call (larger query blocks register-spill);
  `batch_transfer` 56 GB/s d2h / 31 GB/s h2d vs 2–10 GB/s module moves.
- Every limitation above was reproduced at torch_tpu head with a drop-in
  absltest before being worked around or reported; the compiled-mode
  crash with stock serialization was demonstrated end-to-end.
- GPU non-regression was verified by running the full unit suite with the
  platform forced to baseline and diffing the failure set against `main`
  (identical, 1001 passed).
