# Diffusion recipes on Cloud TPU (`torch_tpu`)

Launch recipes for `sglang.multimodal_gen` on one tpu7x 2x2x1 host (8 TensorCores, `--num-gpus` counts cores).
All recipes use the Pallas splash attention backend, torch.compile with the `tpu` backend, a TP-folded resident
text encoder, the highest TPU clock and a 65536 KiB scoped-VMEM limit for XLA fusions (`LIBTPU_INIT_ARGS`), the full
64 MiB of per-core VMEM on tpu7x (3% faster denoise on Z-Image-Turbo than without it); see `launch.sh`. The Wan2.2
recipe compiles per transformer block (`--regional-compile`), which keeps each block's collectives inside its program,
and runs Ulysses across the two cores of a chip with a K/V all-gather ring across chips.

| Recipe | Model | Cores | Parallelism |
| :--- | :--- | :---: | :--- |
| `flux2_klein_4b.sh` | `black-forest-labs/FLUX.2-klein-4B` | 1 or 4 | Ulysses = cores (24 heads) |
| `zimage_turbo.sh` | `Tongyi-MAI/Z-Image-Turbo` | 8 or 1 | Ulysses = cores (30 heads, padded to 32) |
| `wan22_a14b.sh` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers`, `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 8 | CFG 2 x Ulysses 2 x ring 2 |
| `ltx2.sh` | `Lightricks/LTX-2` | 8 | Ulysses 8 (32 heads) |
| `ltx23.sh` | `Lightricks/LTX-2.3` | 8 | Ulysses 8 (32 heads) |

Override `NUM_GPUS`, `PARALLEL_ARGS`, `MODEL_ARGS`, `HOST` (default `127.0.0.1`) or `PORT` from the environment
(an empty `PARALLEL_ARGS`/`MODEL_ARGS` drops the recipe's flags). Flags in `LIBTPU_INIT_ARGS` are kept; the clock and
scoped-VMEM defaults are benchmark settings, added unless it sets them. Extra arguments are passed to the server,
e.g. `./ltx2.sh --warmup-resolutions 1280x704`.

## Performance

tpu7x 2x2x1, torch_tpu 0.2.0 (rows marked † use 0.2.1.dev20261028201649, main's pin), libtpu 0.0.50, steady-state request (the first two requests compile and warm up), seed 42. E2E and peak HBM come from a default run;
denoise step and VAE decode from a run with `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1`, which synchronizes the device at
stage boundaries (otherwise queued denoising work is counted in the decode stage).

| Model | Request | Cores | E2E | Denoise step | VAE decode | Peak HBM / core |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: |
| FLUX.2-klein-4B | T2I 1024x1024, 4 steps | 1 | 0.43 s | 0.075 s | 0.10 s | 24.8 GB |
| FLUX.2-klein-4B | T2I 1024x1024, 4 steps | 4 | 0.32 s | 0.047 s | 0.10 s | 19.2 GB |
| Z-Image-Turbo | T2I 1024x1024, 9 steps | 1 | 0.97 s | 0.094 s | 0.10 s | 29.2 GB |
| Z-Image-Turbo | T2I 1024x1024, 9 steps | 8 | 0.36 s | 0.026 s | 0.10 s | 22.4 GB |
| Wan2.2-T2V-A14B | T2V 1280x720, 81 frames, 20 steps | 8 | 60.3 s | 2.77 s | 4.0 s | 68.8 GB |
| Wan2.2-T2V-A14B | T2V 1280x720, 81 frames, 40 steps | 8 | 115.6 s | 2.77 s | 4.0 s | 68.8 GB |
| Wan2.2-I2V-A14B | I2V 624x624, 81 frames, 20 steps | 8 | 18.4 s | 0.730 s | 1.9 s | 64.3 GB |
| LTX-2 † | T2V+audio 1280x704, 121 frames, 40 steps, one-stage | 8 | 31.1 s | 0.489 s | 8.5 s | 63.1 GB |
| LTX-2.3 † | T2V+audio 1280x704, 121 frames, 30 steps, one-stage | 8 | 35.6 s | 0.975 s | 3.9 s | 66.0 GB |


## Example requests

`/v1/images/generations` returns the image; `size`, `num_inference_steps` and `guidance_scale` set the request.

```bash
curl -s -X POST http://localhost:30000/v1/images/generations -H "Content-Type: application/json" -d '{
  "prompt": "A cinematic portrait of a red panda astronaut floating in space, highly detailed",
  "size": "1024x1024", "num_inference_steps": 9, "guidance_scale": 0.0, "seed": 42, "response_format": "b64_json"
}' | python3 -c 'import base64, json, sys; open("out.png", "wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

`/v1/videos` queues a job and returns its `id`; poll `GET /v1/videos/{id}` until `status` is `completed`, then
download `GET /v1/videos/{id}/content`. Image-to-video requests (Wan2.2-I2V-A14B) add
`"input_reference": "<path or URL>"`.

LTX-2 (`ltx2.sh`):

```bash
curl -s -X POST http://localhost:30000/v1/videos -H "Content-Type: application/json" -d '{
  "prompt": "A street musician with a beard strums an acoustic guitar and sings a cheerful folk song on a busy city sidewalk at dusk, people walking past, his fingers moving on the strings, warm streetlights, clear vocals and guitar.",
  "size": "1280x704", "num_frames": 121, "num_inference_steps": 40, "guidance_scale": 4.0, "seed": 42
}'
```

Wan2.2-T2V-A14B (`wan22_a14b.sh`):

```bash
curl -s -X POST http://localhost:30000/v1/videos -H "Content-Type: application/json" -d '{
  "prompt": "A skateboarder in a red hoodie carves down a sunlit concrete bowl, kicks off the coping into a high air, grabs the board and lands smoothly as dust swirls; tracking camera, golden hour, with the clack of the board and wind.",
  "size": "1280x720", "num_frames": 81, "num_inference_steps": 40, "guidance_scale": 4.0, "seed": 42
}'
```
