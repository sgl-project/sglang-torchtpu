# SGLang-torchtpu

PyTorch-native SGLang serving on Google TPUs.

[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat&labelColor=555555)](LICENSE)
<!-- add once public: PyPI version · nightly single-host · nightly multi-host -->

Docs | [Quickstart](#get-started) | [Supported models and features](SUPPORT.md) | [Cookbook](https://cookbook.sglang.io/) | Slack | [Issues](https://github.com/sgl-project/sglang-torchtpu/issues)

<<<<<<< HEAD
SGLang-torchtpu runs the SGLang engine on TPUs through torch_tpu and Pallas kernels. It is the PyTorch-native sibling of [SGL-JAX](https://github.com/sgl-project/sglang-jax): eager execution, the PyTorch ecosystem, and the same server, OpenAI-compatible API, and scheduler as SGLang on GPUs. It ships as a plugin that installs next to stock SGLang.
=======
SGLang is an open-source inference framework for large language, vision-language, and diffusion models, optimized for agentic workloads, large-scale serving, and RL rollouts. [SGLang Diffusion](https://docs.sglang.io/docs/sglang-diffusion) is its built-in image and video generation engine, included in this repository and the `sglang` Python package.
>>>>>>> upstream/main

## What you get

- **Hardware:** TPU7x (Ironwood), single-host (4 chips) and multi-host (2 hosts at tp=16; 4 hosts for disaggregated prefill/decode).
- **Models:** Gemma 4, GLM, Kimi, gpt-oss, DeepSeek, Qwen, at full size on multi-host TPU. Tiers in [SUPPORT.md](SUPPORT.md).
- **Parallelism:** TP, EP, DPA, PP, CP.
- **Caching:** prefix cache, HiCache KV offload to host memory, prefill/decode disaggregation with TP resharding.
- **Speculative decoding:** EAGLE3, DFlash, DFlash2, DSPark, MTP. Serve with `SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER=1`, which selects the [experimental spec worker](sglang_tpu/experimental/speculative/README.md); DFlash2 and MTP run only on it. The legacy spec worker is in maintenance mode.
- **Quantization:** FP8 channelwise and blockwise, MXFP4, INT4 W4A16, fp8 KV cache.
- **Compilation:** torch.compile/direct_compile with a TPU backend.

Everything above is backed by a registered test. [SUPPORT.md](SUPPORT.md) assigns each model and feature a tier; [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md) is generated from the CI registrations on every merge and shows the test behind every claim.

## Get started

Pull the image, which bundles SGLang, the TPU plugin, torch_tpu, and libtpu:

```bash
docker pull <public registry>/sglang-torchtpu:latest
```

Or install into a Python environment. The plugin pins the torch_tpu and libtpu versions it was validated with:

```bash
pip install sglang sglang-tpu
```

<<<<<<< HEAD
Launch the quickstart model on a single host and send a request:
=======
Next, launch your model:

- [Quickstart](https://docs.sglang.io/docs/get-started/quickstart): Run your first model and send a request.
- [Cookbook](https://cookbook.sglang.io/): Choose your model and hardware to get a ready-to-run launch command.

## Supported Hardware

SGLang supports a wide range of GPUs, TPUs, NPUs, CPUs, and Apple Silicon platforms.

| Platform | Representative hardware |
| --- | --- |
| [NVIDIA](https://docs.sglang.io/docs/hardware-platforms/nvidia-gpus) | B200/B300/GB200/GB300; H100/H200/H800/H20; A100; RTX 30/40/50 series, RTX 6000 Ada / PRO 6000; DGX Spark, Jetson Orin |
| [AMD](https://docs.sglang.io/docs/hardware-platforms/amd_gpu) | Instinct MI300X, MI325X, MI350X, MI355X |
| [Google TPU](https://docs.sglang.io/docs/hardware-platforms/tpu) | v6e, v7; [SGL-JAX](https://github.com/sgl-project/sglang-jax) / [SGL-torchtpu](https://lmsys.org/blog/2026-07-30-sglang-google-tpu/) |
| Intel ([GPU](https://docs.sglang.io/docs/hardware-platforms/xpu) / [CPU](https://docs.sglang.io/docs/hardware-platforms/cpu_server)) | Arc / Arc Pro B-Series GPUs, Xeon CPUs |
| [Apple Silicon](https://docs.sglang.io/docs/hardware-platforms/apple_metal) | Macs via Metal / MLX |
| [Huawei Ascend](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/getting-started/installation) | A2, A3, 950PR/DT NPUs |
| [Moore Threads](https://docs.sglang.io/docs/hardware-platforms/mthreads_gpu) | MTT S5000 GPUs |

Integrations in progress: AWS Trainium, [Alibaba T-Head PPU](https://github.com/sgl-project/sglang/issues/37519), [Cambricon MLU](https://github.com/sgl-project/sglang/pull/26898), Qualcomm QAIC, MetaX, Hygon HCU/DCU, Iluvatar CoreX, and more.

See the [Cookbook](https://cookbook.sglang.io/) and platform guides for model compatibility and setup.

## SGL Ecosystem

| Area | Projects | Purpose |
| --- | --- | --- |
| Diffusion | [SGLang Diffusion](https://docs.sglang.io/docs/sglang-diffusion/installation) | Built into SGLang for image and video generation with diffusion models. |
| Audio | [SGLang Omni](https://github.com/sgl-project/sglang-omni) | Audio model serving for text-to-speech (TTS) and automatic speech recognition (ASR). |
| Speculative Decoding | [SpecForge](https://github.com/sgl-project/SpecForge) | Train draft models for speculative decoding and deploy them with SGLang. |
| RL and Post-Training | [Miles](https://github.com/radixark/miles), [slime](https://github.com/THUDM/slime), [AReaL](https://github.com/inclusionAI/AReaL), [Tunix](https://github.com/google/tunix), [verl](https://github.com/volcengine/verl) | Training frameworks that integrate SGLang for rollout generation. |
| Deployment and Orchestration | [SMG](https://github.com/smg-project/smg), [RBG](https://github.com/sgl-project/rbg), [llm-d](https://llm-d.ai/docs/dev/operations/disaggregation/sglang), [Ray Serve](https://docs.ray.io/en/latest/serve/llm/user-guides/sglang.html), [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/backends/sg-lang/reference-guide) | Deploy and scale SGLang inference services with routing, load balancing, and cluster orchestration. |
| Education | [Mini-SGLang](https://github.com/sgl-project/mini-sglang), [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang), [DeepLearning.AI course](https://www.deeplearning.ai/short-courses/efficient-inference-with-sglang-text-and-image-generation/) | Learn inference engine design and efficient text and image generation through code and hands-on courses. |

## Development and Contributing

Contributions are welcome, from bug fixes and documentation to model support and performance improvements.

### Development setup

Start from the `lmsysorg/sglang:dev` Docker image, which provides development tools and most dependencies. Clone or mount your SGLang checkout inside the container, then install it in editable mode from the repository root so tests use your local Python changes:
>>>>>>> upstream/main

```bash
python -m sglang.launch_server --model-path openai/gpt-oss-20b --device tpu --tp 4

curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "Hello"}]}'
```

Next steps:

- TPU guide: multi-host launch, weights on shared storage, compile cache, and tuning flags.
- [Cookbook](https://cookbook.sglang.io/): per-model launch commands.
- Working from a checkout instead of the wheel:

  ```bash
  git clone https://github.com/sgl-project/sglang-torchtpu.git && cd sglang-torchtpu
  pip install -e python/          # sglang
  pip install --no-deps -e .      # sglang-tpu, this plugin
  ```

## Supported hardware

| Generation | Single-host | Multi-host | Status |
| --- | --- | --- | --- |
| TPU7x (Ironwood) | 4 chips | 2 hosts (tp=16); 4 hosts for 2P+2D | CI + nightly |

## Supported models and features

[SUPPORT.md](SUPPORT.md) lists every model and feature with a tier: **Quickstart** = small, runs on every PR · **Optimized** = tuned recipe with nightly accuracy and performance tracking · **Supported** = serves correctly, tracked nightly · **Experimental** = code present, no green on-device test yet. It also lists what is not supported. The test behind each claim is in the generated [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md).

## Relationship to SGLang

SGLang-torchtpu is an out-of-tree platform plugin for [sgl-project/sglang](https://github.com/sgl-project/sglang). The TPU backend lives in `sglang_tpu/`; SGLang discovers it through the `sglang.srt.platforms` entry point and loads it when `--device tpu` is set. Changes that SGLang needs to host a hardware backend are proposed upstream as small extension points; TPU kernels, memory pools, and recipes stay here. A migration scoreboard tracks the remaining in-tree coupling and is published daily.

Where to file what: TPU-specific bugs and feature requests here; engine bugs that reproduce on GPU upstream, with the `tpu` label if TPU is involved.

## Development

- CPU-only unit tests run on any machine without a TPU:

  ```bash
  python test/run_suite.py --suite base-a-test-cpu
  ```

- Develop speculative decoding in the experimental spec worker, which has no CUDA dependency: see its [developer guide](sglang_tpu/experimental/speculative/README.md). The legacy spec worker only takes fixes.
- Kernel and end-to-end tests need a TPU. A maintainer applies the `run-ci` label and comments `/tag-and-rerun-ci` to run them on your pull request.
- CONTRIBUTING.md covers the branch-and-PR flow, adding a test to the support matrix, and the policy for changes outside `sglang_tpu/`.

## Community

- Slack: `<channel placeholder>` in the SGLang workspace.
- GitHub Discussions for questions and design threads.
- [Issues](https://github.com/sgl-project/sglang-torchtpu/issues) for bugs and feature requests, with chip generation, topology, and torch_tpu version in the template.
- Security: see SECURITY.md.

## License and acknowledgments

Apache-2.0. See NOTICE for code adapted from [sglang-jax](https://github.com/sgl-project/sglang-jax), [Helion](https://github.com/pytorch/helion), [tpu-inference](https://github.com/vllm-project/tpu-inference), and [vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu).

SGLang-torchtpu is built by the SGLang community and its contributors.
