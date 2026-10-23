# Diffusion recipes on Cloud TPU (`torch_tpu`)

Launch recipes for `sglang.multimodal_gen` on one tpu7x 2x2x1 host (8 TensorCores, `--num-gpus` counts cores).
All recipes use the Pallas splash attention backend, torch.compile with the `tpu` backend, a TP-folded resident
text encoder, the highest TPU clock and a 65536 KiB scoped-VMEM limit for XLA fusions (`LIBTPU_INIT_ARGS`), the full
64 MiB of per-core VMEM on tpu7x (3% faster denoise on Z-Image-Turbo than without it); see `launch.sh`. The Wan
recipes compile per transformer block (`--regional-compile`), which keeps each block's collectives inside its program,
and run Ulysses across the two cores of a chip with a K/V all-gather ring across chips.
That placement relies on Ulysses groups being contiguous ranks (ring groups are strided, see `parallelism.mdx`) and on
ranks 2k and 2k+1 being the two cores of one chip, as in the default `--num-gpus 8` device order.

| Recipe | Model | Cores | Parallelism |
| :--- | :--- | :---: | :--- |
| `flux2_klein_4b.sh` | `black-forest-labs/FLUX.2-klein-4B` | 1 or 4 (2 and 8 divide 24 heads; untested) | Ulysses = cores (24 heads) |
| `zimage_turbo.sh` | `Tongyi-MAI/Z-Image-Turbo` | 1 or 2 | Ulysses = cores (30 heads) |
| `wan22_ti2v_5b.sh` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 8 | CFG 2 x Ulysses 2 x ring 2 |
| `wan22_a14b.sh` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 8 | CFG 2 x Ulysses 2 x ring 2 |
| `ltx2.sh` | `Lightricks/LTX-2` | 8 | CFG 2 x Ulysses 4 |

Override `NUM_GPUS`, `PARALLEL_ARGS`, `MODEL_ARGS`, `HOST` (default `127.0.0.1`) or `PORT` from the environment
(an empty `PARALLEL_ARGS`/`MODEL_ARGS` drops the recipe's flags). `ltx2.sh` fixes its parallel layout to 8 ranks, so
override `NUM_GPUS` and `PARALLEL_ARGS` together there. Flags in `LIBTPU_INIT_ARGS` are kept; the clock and
scoped-VMEM defaults are benchmark settings, added unless it sets them. Extra arguments are passed to the server,
e.g. `./ltx2.sh --warmup-resolutions 1280x720`.

LTX-2 under sequence parallelism skips attention masks only when no stream is padded: the latent frame count
`(num_frames - 1) / 8 + 1` must divide by the SP degree (121 frames at SP 4 qualifies; 97 and 241 do not), and
likewise the audio latent frames. A padded stream keeps the masked attention path.

## Performance

tpu7x 2x2x1, steady-state request (the first request compiles), seed 42. E2E and peak HBM come from a default run;
denoise step and VAE decode from a run with `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1`, which synchronizes the device at
stage boundaries (otherwise queued denoising work is counted in the decode stage). Shapes whose token count does not
divide by the sequence-parallel size (the two I2V rows) pad the sequence; attention masks the padding.

| Model | Request | Cores | E2E | Denoise step | VAE decode | Peak HBM / core |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: |
| FLUX.2-klein-4B | T2I 1024x1024, 4 steps | 1 | 0.90 s | 0.076 s | 0.33 s | 24.8 GB |
| FLUX.2-klein-4B | T2I 1024x1024, 4 steps | 4 | 0.85 s | 0.044 s | 0.33 s | 19.2 GB |
| Z-Image-Turbo | T2I 1024x1024, 9 steps | 1 | 1.36 s | 0.088 s | 0.33 s | 29.2 GB |
| Z-Image-Turbo | T2I 1024x1024, 9 steps | 2 | 1.31 s | 0.069 s | 0.33 s | 25.2 GB |
| Wan2.2-TI2V-5B | T2V 1280x704, 121 frames, 20 steps | 8 | 9.6 s | 0.272 s | 2.5 s | 31.1 GB |
| Wan2.2-TI2V-5B | I2V 960x928 (from a square image), 121 frames, 20 steps | 8 | 11.8 s | 0.331 s | 2.6 s | 67.2 GB |
| Wan2.2-T2V-A14B | T2V 1280x720, 81 frames, 20 steps | 8 | 58.8 s | 2.72 s | 2.2 s | 87.8 GB |
| Wan2.2-I2V-A14B | I2V 624x624, 81 frames, 20 steps | 8 | 17.0 s | 0.698 s | 1.2 s | 71.9 GB |
| LTX-2 | T2V+audio 1280x720, 121 frames, 40 steps | 8 | 23.2 s | 0.433 s | 2.3 s | 60.6 GB |

Known limit: Wan2.2-I2V-A14B at 720p-class sizes with 121 frames fails in VAE decode (one decode program needs about
40.9 GB of HBM; about 35 GB per core is free).

## Example requests

`/v1/images/generations` returns the image; `size`, `num_inference_steps` and `guidance_scale` set the request.

```bash
curl -s -X POST http://localhost:30000/v1/images/generations -H "Content-Type: application/json" -d '{
  "prompt": "A cinematic portrait of a red panda astronaut floating in space, highly detailed",
  "size": "1024x1024", "num_inference_steps": 9, "guidance_scale": 0.0, "seed": 42, "response_format": "b64_json"
}' | python3 -c 'import base64, json, sys; open("out.png", "wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

`/v1/videos` queues a job and returns its `id`; poll `GET /v1/videos/{id}` until `status` is `completed`, then
download `GET /v1/videos/{id}/content`. Image-to-video requests (Wan2.2-I2V-A14B, Wan2.2-TI2V-5B) add
`"input_reference": "<path or URL>"`.

LTX-2 (`ltx2.sh`):

```bash
curl -s -X POST http://localhost:30000/v1/videos -H "Content-Type: application/json" -d '{
  "prompt": "A fluffy golden dog sits centered in frame against a soft pastel background, looking into the camera",
  "size": "1280x720", "num_frames": 121, "num_inference_steps": 40, "guidance_scale": 4.0, "seed": 42
}'
```

Wan2.2-TI2V-5B (`wan22_ti2v_5b.sh`):

```bash
curl -s -X POST http://localhost:30000/v1/videos -H "Content-Type: application/json" -d '{
  "prompt": "A fluffy golden dog sits centered in frame against a soft pastel background, looking into the camera",
  "size": "1280x704", "num_frames": 121, "num_inference_steps": 20, "seed": 42
}'
```
