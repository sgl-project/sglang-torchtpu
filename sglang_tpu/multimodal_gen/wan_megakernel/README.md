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

`SGLANG_TPU_ENABLE_WAN_MEGAKERNEL_FP8=1` adds W8A8 projections. Weights get per-column e4m3
scales and activations per-row scales, with f32 accumulation. Attention, norms and
residuals stay as in the bf16 path.

`SGLANG_TPU_ENABLE_WAN_MEGAKERNEL_FP8_QK=1` computes the attention scores of the fixed
softmax (the path about 98% of attention tiles take) from e4m3 copies of q and k, with one
scale per query tile and one per head, and f32 accumulation. PV, the other softmax paths
and everything outside attention keep their precision.

The fastest configuration also decodes the VAE in bf16 and five latent frames a decoder
call, spawns the mp4 encoder without forking the worker, and warms the whole step schedule:

```bash
SGLANG_TPU_ENABLE_WAN_MEGAKERNEL=1 SGLANG_TPU_ENABLE_WAN_MEGAKERNEL_FP8=1 \
SGLANG_TPU_ENABLE_WAN_MEGAKERNEL_FP8_QK=1 IMAGEIO_FFMPEG_NO_PREVENT_SIGINT=1 \
SGLANG_DIFFUSION_WAN_VAE_DECODE_CHUNK_FRAMES=5 \
PARALLEL_ARGS="--cfg-parallel-size 2 --ulysses-degree 4" \
MODEL_ARGS="--regional-compile --dit-layerwise-offload false --dit-cpu-offload false \
  --vae-decode-precision bf16 --warmup-steps 40 \
  --warmup-resolutions 1280x720 --warmup-num-frames 81" \
bash benchmark/tpu/diffusion/wan22_a14b.sh
```

`SGLANG_TPU_ENABLE_WAN_MEGAKERNEL_STEP_CONSTANTS=1` builds each shape's per-step constants
once: the tiling, the op and the real-row count, uploaded without waiting. A denoising step
then makes no blocking host-to-device copy (on torch_tpu one returns only after every queued
program has finished), so the host queues the next step's programs while the kernel runs.
The kernel gets the same inputs.

`SGLANG_TPU_WAN_MEGAKERNEL_FLOOR_GUARD=inline` lays out the bf16 attention items' row-sum
floor so the fixed softmax path (about 98% of items) runs one span of code: its check ends
the fixed branch, and rowfix and the rerun sit in one branch it skips. The checks, reruns
and outputs are the default layout's. `rowfix` checks rowfix sums only, as the kernel did
before the fixed-path floor; it is for measurements.

Three switches change only the compiled code's layout; each gives the same bits:

- `SGLANG_TPU_OPT_WAN_MEGAKERNEL_ONLINE_ONE_LOOP=1` runs the online softmax over every key
  tile, the edge tile included, in one loop of masked sub-blocks. The attention item loop
  then fits one instruction overlay (about 58K bundles at 832x480). Otherwise it spans six,
  and each item reloads the fixed path's overlay on its way back from the item's end.
- `SGLANG_TPU_OPT_WAN_MEGAKERNEL_PIPELINED_EPILOGUES=1` (bf16) runs each phase C projection
  tile's epilogue, and each MLP up tile's GELU, in the next tile's step.
- `SGLANG_TPU_OPT_WAN_MEGAKERNEL_SKIP_BOUNDS_CHECKS=1` compiles the kernel without Mosaic's
  DMA bounds checks, which add scalar work at every copy.

`--vae-decode-precision bf16` matters on TPU because the worker sets fp32 matmuls to the
highest precision, and Wan2.2-A14B's config decodes in fp32 (720p decode: 4.0 s fp32,
2.3 s bf16). `SGLANG_DIFFUSION_WAN_VAE_DECODE_CHUNK_FRAMES=5` decodes the first latent frame
alone and then five a call (81 frames: 5 calls instead of 21), the same video up to
rounding in fewer, larger ops.

At load, the hook packs both experts' blocks into the kernel's layout and registers one
op per tiling, shared by both experts. Each forward then embeds on the torch side, runs
the kernel, and applies the head to the rank's rows. The kernel takes over when every
rank meets all of these:

- the TPU platform
- ring degree 1 and TP 1, with CFG × Ulysses covering all ranks
- bf16 projections resident on the device (no offload, FSDP or LoRA)
- no CLIP image cross-attention and scalar timesteps

Otherwise it logs the reason, and the blocks run in torch.

## Step caching

`SGLANG_TPU_WAN_STEP_CACHE="steps=i,j,..."` runs the kernel only on the listed denoising
steps and on the last step. Every other step adds its expert's last computed block-stack
residual (output minus input) for the same CFG branch to the step's own input. The plan
depends only on the step index, so all ranks skip the same steps. It applies to
single-video requests.

For the 40-step Wan2.2-A14B schedule (flow shift 12, expert switch at step 26), this
15-step plan keeps the videos' sharpness, motion and flicker statistics at the uncached
level:

```bash
SGLANG_TPU_WAN_STEP_CACHE=steps=0,1,3,5,8,11,14,17,20,23,26,32,36,38,39
```

The plan comes from calibration dumps. `SGLANG_TPU_DEBUG_WAN_CACHE_STATS=<prefix>` writes
per-forward residual statistics. `SGLANG_TPU_DEBUG_WAN_CACHE_ROWS=16` additionally saves 16
sampled residual rows per forward and rank. A dynamic program over these dumps picks, for
each step budget, the steps that minimize the guided-prediction error of reusing residuals
(`g·Δr_cond - (g-1)·Δr_uncond`, relative, weighted by the step's sigma decrement to the 0.2
power). Dropping only the unconditional branch costs more error per saved forward than
skipping both branches.

## Layout

| File | Contents |
| --- | --- |
| `reference.py` | The specification in plain JAX. It follows diffusers' bf16 rounding points (`mode="bf16"`), with f32 (`"f32"`) and W8A8 (`"fp8"`) variants. |
| `kernel.py` | The megakernel, with four phases per layer. |
| `tiling.py` | Tile sizes. `for_tokens` picks a request's rows per device (padding) and attention tiles. |
| `pack.py`, `op.py`, `hook.py` | Packing from sglang's modules, the torch op (`cpu_export_jax_op` over a `("cfg", "sp")` mesh), and the `WanTransformer3DModel` patch with step caching. |

The four phases of a layer in `kernel.py`:

- **A** — LayerNorm and modulation, QKV, q/k RMSNorm and RoPE. The RMSNorm and RoPE of q's heads run under the k tiles' matmuls and those of k's heads under the v tiles'. Head rows go to their owners after each matmul.
- **B** — Self-attention with transposed scores. The softmax offset comes from |q|·max|k| bounds: a fixed offset when every score fits, a per-query offset otherwise, and online softmax when a row would underflow. QK^T contracts head_dim 128 over the 256-deep MXU, so in bf16 it holds the MXU at half rate and attention is MXU-bound. With fp8 QK^T, the fixed softmax runs as a software pipeline over 256-lane query chunks: the next chunk's QK^T is issued before this chunk's exp and PV, so e4m3 QK^T runs on one MXU and PV on the other, and a chunk's PV products accumulate in the MXU result buffer over a 2048-key tile. The key tile holding the last real row runs in the same loop: its rows past the last real key are zeroed (score 0, weight exactly 1, value 0), and their count comes off the row sums after the loop. An item whose smallest real-row sum is below 2 reruns on the per-query-offset path, which masks those keys.
- **C** — The output projection, cross-attention over the prompt's distinct text keys, and its projection. Each projection's first weight tile loads during the work before it (a LayerNorm, the cross-attention or the previous row tile's end). A row tile's attention rows and x load during the previous row tile, and its write-backs finish during the next one.
- **C2** — The MLP in two passes: up-projection and GELU into a row tile, then the down-projection in output-column tiles. The next row tile's input loads during the down-projection. With W8A8 the up-projection takes two weight tiles a step from a 4-slot ring, so a tile's GELU overlaps the next tile's matmul.

The LayerNorms of A and C take each row's mean and mean square in one sweep (variance E[x²] − mean²); `reference` keeps the two-pass form. With W8A8, each projection's weight tile multiplies its rows in four parts, so one part's results leave the MXU and get scaled while the next part's matmuls run.

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
(about 30K bundles in bf16). A benchmark prompt (38-54 T5 tokens) and its empty negative
prompt attend to 64 keys; sglang's default Wan2.2 negative prompt (126 tokens) attends
to 128.

## Measurements

Setup: Wan2.2-T2V-A14B on tpu7x 2x2x1 (8 cores), 81 frames, CFG 4.0 / 3.0, on sglang at the
Wan2.2 TPU PR stack (#1845, 3a9469b961) with this megakernel. Latency is from submit to the
downloaded mp4, for warm requests whose prompt is new to the server (a repeated prompt skips
text encoding). The torch path runs the recipe's CFG 2 × Ulysses 2 × ring 2 with the DiT
resident, fp32 VAE decode and sglang's 2-step server warmup at the served shape. The megakernel
uses CFG 2 × Ulysses 4 with W8A8, fp8 QK^T, bf16 VAE decode in five-frame chunks and a
full-schedule warmup (the configuration under Enable).

| Request | sglang torch blocks | megakernel | megakernel + 15-step plan |
| --- | --- | --- | --- |
| 832x480, 40 steps | 33.8 s | 16.6 s | 7.55 s (4.5x) |
| 1280x720, 40 steps | 116 s | 65.2 s | 26.8 s (4.3x) |
| 1280x720, 50 steps | | 81.2 s | |

With the same bf16 VAE decode, vfork mp4 encoder (`IMAGEIO_FFMPEG_NO_PREVENT_SIGINT=1`) and
full-schedule warmup, the torch path takes 32.6 s at 832x480 and 113.4 s at 1280x720.

Over the four 480p benchmark prompts, the mean Laplacian variance (sharpness), mean
absolute frame difference (motion) and mean absolute second frame difference (flicker) is
198 / 4.69 / 7.61 with the 15-step plan and 202 / 4.98 / 7.88 on all 40 steps. In the
reference forward on real weights and a real step input, e4m3 QK^T changes the model
output by 0.9% for the high-noise expert (t = 999) and 4.9% for the low-noise expert
(t = 865).

Kernel time per layer, with cross-attention over 64 text keys (a benchmark prompt and an
empty negative prompt):

| Request | bf16 | W8A8 | W8A8 + fp8 QK^T | Rows a device |
| --- | --- | --- | --- | --- |
| 832x480 | 13.50 ms | 11.55 ms | 9.29 ms | 8192 (no padding) |
| 1280x720 | 55.14 ms | 50.54 ms | 37.20 ms | 19456 (75600 tokens, 2.9% padding) |

Over all 512 text rows, W8A8 + fp8 QK^T takes 9.42 ms at 832x480 and 37.55 ms at 1280x720.
In bf16, attention is about 58% of a 480p layer and 76% of a 720p layer, and it runs at
its MXU floor. On real activations, about 98% of attention tiles take the fixed-offset
softmax. Of those, the zero-padded edge's guard reruns 5 of 6307 in the high-noise
expert's first step at 832x480 and none in the low-noise expert's.

On identical inputs, the step-0 output matches the torch blocks to the following rel. L2
errors:

| Request | bf16 | W8A8 |
| --- | --- | --- |
| 832x480, 81 frames | 8.6e-3 | 2.2e-2–2.7e-2 |
| 832x480, 17 frames | 5e-3–7e-3 | |
| 1280x720, 81 frames | 1.0e-2–1.1e-2 | 3.5e-2–3.6e-2 |

## Tests

`test/registered/unit/plugins/test_tpu_wan_megakernel.py` runs the kernel in Pallas TPU
interpret mode on CPU against `reference`. It covers one and two devices, every softmax
variant, W8A8, fp8 QK^T (with the zero-padded edge tile and its fallback), padding, odd
tile counts, and cross-attention over a padded prompt's distinct keys (a different count
on each device). Interpret mode simulates DMAs and detects races, with both eager and
deferred DMA execution. The file also checks `pack` against the reference layout bit for
bit, fp8 included, and `pack.cross_keys`' counts and weights.
