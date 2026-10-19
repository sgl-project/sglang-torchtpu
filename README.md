<p align="center">
  <img src="assets/torch-tpu.svg" alt="SGLang-TorchTPU" width="360">
</p>

<h1 align="center">SGLang-TorchTPU</h1>

<p align="center">
  <strong>PyTorch-native LLM serving on Google TPUs.</strong><br>
  The SGLang serving engine. The PyTorch ecosystem. Built for TPU.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat&amp;labelColor=555555" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Hardware-TPU7x%20Ironwood-orange?style=flat&amp;labelColor=555555" alt="Hardware: TPU7x Ironwood">
</p>

SGLang-TorchTPU brings [SGLang](https://github.com/sgl-project/sglang) to Google TPUs through **torch_tpu** and **Pallas kernels**. Serve large language and vision-language models with SGLang's OpenAI-compatible API and scheduling engine, while developing in PyTorch with eager execution and TPU compilation.

## Highlights

**PyTorch-native TPU serving.** Develop with familiar PyTorch workflows and deploy on TPU7x (Ironwood), with `torch.compile` and direct compilation. Tensor, expert, data, pipeline, and context parallelism cover a range of serving configurations.

**Efficient inference at scale.** Combine prefix caching, HiCache KV offload to host memory, and disaggregated prefill/decode with TP resharding. Low-precision inference includes FP8, MXFP4, INT4 W4A16, and FP8 KV cache.

**Speculative decoding.** Explore EAGLE3, DFlash, DFlash2, DSPark, and MTP for supported model configurations. The speculative decoding developer guide below covers worker selection and model-specific requirements.

## Get started

### Docker

Pull the TPU image, which has the engine and TPU plugin installed:

```bash
docker pull lmsysorg/sglang:v0.1.0-tpu
```

On a TPU7x host, start a container with access to the TPU chips:

```bash
docker run -it --rm --privileged --net=host --shm-size=64g \
  -e SGLANG_PLATFORM=tpu_oot \
  lmsysorg/sglang:v0.1.0-tpu
```

The container starts in the repository root, `/sgl-workspace/sglang`.

### pip

On a TPU7x host (Linux x86_64, Python 3.11 to 3.14), install the engine and TPU plugin wheels from the [v0.1.0 release](https://github.com/sgl-project/sglang-torchtpu/releases/tag/tpu-v0.1.0). Install both together: `sglang` here is this repository's build, not the PyPI package.

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  https://github.com/sgl-project/sglang-torchtpu/releases/download/tpu-v0.1.0/sglang-0.5.22-py3-none-any.whl \
  https://github.com/sgl-project/sglang-torchtpu/releases/download/tpu-v0.1.0/sglang_tpu-0.1.0-py3-none-any.whl
```

The serving recipes below live in the repository, so clone it to run them.

### From source

On a TPU7x host, install the engine with [the TPU dependency manifest](python/pyproject_tpu.toml), then the TPU plugin:

```bash
git clone https://github.com/sgl-project/sglang-torchtpu.git
cd sglang-torchtpu
cp python/pyproject_tpu.toml python/pyproject.toml
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e python/
pip install --no-deps -e .
```

### Launch a server

From the repository root, launch GPT-OSS-20B with the single-host serving recipe, configured for 4 chips and 8 TensorCores:

```bash
python -m benchmark.tpu.launch_server --config recipe/configs/gpt_oss_20b/server.py
```

Once the server is ready, send an OpenAI-compatible request from another terminal:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Supported hardware

Development and testing focus on **TPU7x (Ironwood)**, from a single host to multi-host serving and disaggregated prefill/decode.

| Deployment | Topology |
| --- | --- |
| Single host | 4 chips, 8 TensorCores; TP up to 8 |
| Multi-host | 2 hosts; TP 16 |
| Disaggregated serving | 4 hosts; 2 prefill + 2 decode |

## Models and features

Selected model integrations include the following. Recipe availability and test coverage vary by model; experimental integrations are included.

| Model | Available recipes and tests |
| --- | --- |
| Muse Spark Open | Serving and agentic recipes |
| GLM-5.3-Flash | Serving and MTP recipes |
| Kimi-K3 | Small-model compile test with dummy weights |
| DeepSeek-V4-Flash | Serving recipe |
| Qwen3.8-27B | Accuracy and MTP recipes |
| GPT-OSS (20B, 120B) | 20B recipes, 120B recipes |
| Gemma 4 (31B, 26B-A4B) | 31B recipes, 26B-A4B recipes |

See the [full support guide](SUPPORT.md) for support tiers, model-specific requirements, and experimental capabilities.

Explore [model recipes](benchmark/tpu/recipe/README.md), [TPU benchmarks](benchmark/tpu/README.md), and the [CI support matrix](SUPPORT_MATRIX.md) for configurations and test coverage.

## Part of the SGLang ecosystem

SGLang-TorchTPU is the PyTorch-native TPU backend in the [SGLang](https://github.com/sgl-project/sglang) ecosystem, alongside the JAX-based [SGL-JAX](https://github.com/sgl-project/sglang-jax). It brings TPU execution to the SGLang serving interface, so applications can use familiar OpenAI-compatible clients.

TPU kernels, memory management, and backend integrations live in `sglang_tpu/`.

## Development and contributing

Contributions are welcome in model support, kernel optimizations, serving features, benchmarks, and documentation.

1. Read the [contributor guide](CONTRIBUTING.md) for development setup and the pull request workflow.
2. Explore the [TPU backend](sglang_tpu/README.md) and [speculative decoding developer guide](sglang_tpu/experimental/speculative/README.md).
3. Add coverage for your changes and include accuracy or performance results where relevant.

CPU-only unit tests run without a TPU:

```bash
python test/run_suite.py --suite base-a-test-cpu
```

Kernel and end-to-end tests require TPU hardware. A maintainer applies the `run-ci` label and comments `/tag-and-rerun-ci` to run them on your pull request.

## Issues and feedback

- **Bugs and feature requests:** Open an [issue](https://github.com/sgl-project/sglang-torchtpu/issues), including your TPU generation, topology, and torch_tpu version for bug reports.
- **Security:** See our [security policy](SECURITY.md) for reporting vulnerabilities.

For engine issues that also reproduce on GPUs, use the [upstream SGLang issue tracker](https://github.com/sgl-project/sglang/issues).

## License and acknowledgments

SGLang-TorchTPU is released under the [Apache 2.0 license](LICENSE).

SGLang-TorchTPU is a joint effort of Google, Meta, RadixArk, and the SGLang community. Thanks to all contributors.

See [NOTICE](NOTICE) for attribution details.
