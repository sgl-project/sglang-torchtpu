# Wan2.2-A14B megakernel (tpu7x)

All 40 transformer blocks of a Wan2.2-T2V-A14B forward run as one Pallas call per
denoising step and CFG branch. The Ulysses exchange happens inside the kernel by
remote DMA, so a step has no XLA collectives between blocks.

## Enable

```bash
SGLANG_TPU_ENABLE_WAN_MEGAKERNEL=1 \
PARALLEL_ARGS="--cfg-parallel-size 2 --ulysses-degree 4" \
MODEL_ARGS="--regional-compile --dit-layerwise-offload false --dit-cpu-offload false \
  --warmup-resolutions 832x480 --warmup-num-frames 81" \
bash benchmark/tpu/diffusion/wan22_a14b.sh
```

The two `--warmup-*` flags make the server warmup compile the served shape, so the first
request is already warm. Set them to the shape you serve, for example `1280x720` and `81`
for 720p.

The full configuration also builds the per-step constants once, uses the two layout
switches below, decodes the VAE five latent frames a decoder call with the TPU decode
options of `../wan_vae.py`, spawns the mp4 encoder without forking the worker, and warms
the whole step schedule:

```bash
SGLANG_TPU_ENABLE_WAN_MEGAKERNEL=1 SGLANG_TPU_ENABLE_WAN_MEGAKERNEL_STEP_CONSTANTS=1 \
SGLANG_TPU_OPT_WAN_MEGAKERNEL_PIPELINED_EPILOGUES=1 \
SGLANG_TPU_OPT_WAN_MEGAKERNEL_SKIP_BOUNDS_CHECKS=1 \
SGLANG_DIFFUSION_WAN_VAE_DECODE_CHUNK_FRAMES=5 SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS=128 \
SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS=xla_tpu_enable_offloading_copy_to_sparsecore=false \
IMAGEIO_FFMPEG_NO_PREVENT_SIGINT=1 \
PARALLEL_ARGS="--cfg-parallel-size 2 --ulysses-degree 4" \
MODEL_ARGS="--regional-compile --dit-layerwise-offload false --dit-cpu-offload false \
  --warmup-steps 40 --warmup-resolutions 832x480 --warmup-num-frames 81" \
bash benchmark/tpu/diffusion/wan22_a14b.sh
```

`SGLANG_TPU_ENABLE_WAN_MEGAKERNEL_STEP_CONSTANTS=1` builds each shape's per-step constants
once: the tiling, the op and the real-row count, uploaded without waiting. A denoising step
then makes no blocking host-to-device copy (on torch_tpu one returns only after every queued
program has finished), so the host queues the next step's programs while the kernel runs.
The kernel gets the same inputs.

Two switches change only the compiled code's layout; each gives the same bits:

- `SGLANG_TPU_OPT_WAN_MEGAKERNEL_PIPELINED_EPILOGUES=1` runs each phase C projection tile's
  epilogue, and each MLP up tile's GELU, in the next tile's step.
- `SGLANG_TPU_OPT_WAN_MEGAKERNEL_SKIP_BOUNDS_CHECKS=1` compiles the kernel without Mosaic's
  DMA bounds checks, which add scalar work at every copy.

`SGLANG_DIFFUSION_WAN_VAE_DECODE_CHUNK_FRAMES=5` decodes the first latent frame alone and
then five a call (81 frames: 5 calls instead of 21), the same video up to rounding in
fewer, larger ops. `--vae-decode-precision bf16` shortens the decode further on TPU,
because the worker sets fp32 matmuls to the highest precision and Wan2.2-A14B's config
decodes in fp32 (720p decode: 4.0 s fp32, 2.3 s bf16).

At load, the hook packs both experts' blocks into the kernel's layout and registers one
op per tiling, shared by both experts. Each forward then embeds on the torch side, runs
the kernel, and applies the head to the rank's rows. The kernel takes over when every
rank meets all of these:

- the TPU platform
- ring degree 1 and TP 1, with CFG × Ulysses covering all ranks
- bf16 projections resident on the device (no offload, FSDP or LoRA)
- no CLIP image cross-attention and scalar timesteps

Otherwise it logs the reason, and the blocks run in torch.

## Layout

| File | Contents |
| --- | --- |
| `reference.py` | The specification in plain JAX. It follows diffusers' bf16 rounding points (`mode="bf16"`), with an f32 (`"f32"`) variant. |
| `kernel.py` | The megakernel, with four phases per layer. |
| `tiling.py` | Tile sizes. `for_tokens` picks a request's rows per device (padding) and attention tiles. |
| `pack.py`, `op.py`, `hook.py` | Packing from sglang's modules, the torch op (`cpu_export_jax_op` over a `("cfg", "sp")` mesh), and the `WanTransformer3DModel` patch. |

The four phases of a layer in `kernel.py`:

- **A** — LayerNorm and modulation, QKV, q/k RMSNorm and RoPE. The RMSNorm and RoPE of q's heads run under the k tiles' matmuls and those of k's heads under the v tiles'. Head rows go to their owners after each matmul.
- **B** — Self-attention with transposed scores. The softmax offset comes from |q|·max|k| bounds: a fixed offset when every score fits, a per-query offset otherwise, and online softmax when a row would underflow. The row-sum check sits at the end of the fixed branch, and the per-query-offset path and the online rerun sit in one branch the fixed path skips, so the fixed path (about 98% of items) runs one span of code. QK^T contracts head_dim 128 over the 256-deep MXU, so in bf16 it holds the MXU at half rate and attention is MXU-bound.
- **C** — The output projection, cross-attention over the prompt's distinct text keys, and its projection. Each projection's first weight tile loads during the work before it (a LayerNorm, the cross-attention or the previous row tile's end). A row tile's attention rows and x load during the previous row tile, and its write-backs finish during the next one.
- **C2** — The MLP in two passes: up-projection and GELU into a row tile, then the down-projection in output-column tiles. The next row tile's input loads during the down-projection.

The LayerNorms of A and C take each row's mean and mean square in one sweep (variance E[x²] − mean²); `reference` keeps the two-pass form.

## Cross-attention keys

sglang zero-pads the T5 prompt embedding to 512 rows, and Wan's cross-attention has no
mask, so every padding row has the same K/V. `pack.cross_keys` finds where that repeated
row starts (comparing K/V exactly, on the device), and the kernel attends to the first
64, 128 or 256 text rows (`Tiling.cross_keys`, the smallest count above the prompt's
distinct rows): the prompt's tokens and copies of the padding row, whose softmax weights
add up to the number of padding rows. When there are enough copies, they take that
number's binary digits; each weighted term then rounds to bf16 exactly as the unweighted
one, so the output differs from the all-rows softmax only by the f32 rounding of its
sums. Otherwise the copies share the weight equally. Prompts of 256 tokens or more
attend to all 512 rows.

Each count has its own cross-attention loop in the one compiled kernel. Each device picks
its loop at run time, so the two CFG branches attend to different counts in the same
call. The loops take a group's eight heads in steps of four (`Tiling.cross_step`), which
keeps phase C's row-tile loop, all four loops included, within one instruction overlay
(about 30K bundles). A prompt of 38-54 T5 tokens and an empty negative prompt attend to
64 keys; sglang's default Wan2.2 negative prompt (126 tokens) attends to 128.

## Measurements

Wan2.2-T2V-A14B on tpu7x 2x2x1 (8 cores), 81 frames, CFG 2 × Ulysses 4. Kernel time per
layer, with cross-attention over 64 text keys (a 38-54 token prompt and an empty negative
prompt):

| Request | Kernel time per layer | Rows a device |
| --- | --- | --- |
| 832x480 | 13.50 ms | 8192 (no padding) |
| 1280x720 | 55.14 ms | 19456 (75600 tokens, 2.9% padding) |

Attention is about 58% of a 480p layer and 76% of a 720p layer, and it runs at its MXU
floor. On real activations, about 98% of attention tiles take the fixed-offset softmax.

On identical inputs, the step-0 output matches the torch blocks to the following rel. L2
errors:

| Request | rel. L2 |
| --- | --- |
| 832x480, 81 frames | 8.6e-3 |
| 832x480, 17 frames | 5e-3–7e-3 |
| 1280x720, 81 frames | 1.0e-2–1.1e-2 |

## Tests

`test/registered/unit/plugins/test_tpu_wan_megakernel.py` runs the kernel in Pallas TPU
interpret mode on CPU against `reference`. It covers one and two devices, every softmax
variant and the row-sum floor's reruns, both layout switches, padding, odd tile counts, and
cross-attention over a padded prompt's distinct keys (a different count on each device).
Interpret mode simulates DMAs and detects races, with both eager and deferred DMA
execution. The file also checks `pack` against the reference layout bit for bit, and
`pack.cross_keys`' counts and weights.
