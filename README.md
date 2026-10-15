# SGLang-torchtpu

PyTorch-native SGLang serving on Google TPUs.

[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat&labelColor=555555)](LICENSE)
<!-- add once public: PyPI version · nightly single-host · nightly multi-host -->

[Docs](<docs url>) | [Quickstart](#get-started) | [Supported models and features](SUPPORT.md) | [Cookbook](https://cookbook.sglang.io/) | [Slack](<slack channel placeholder>) | [Issues](https://github.com/sgl-project/sglang-torchtpu/issues)

SGLang-torchtpu runs the SGLang engine on TPUs through [torch_tpu](<torch_tpu url>) and Pallas kernels. It is the PyTorch-native sibling of [SGL-JAX](https://github.com/sgl-project/sglang-jax): eager execution, the PyTorch ecosystem, and the same server, OpenAI-compatible API, and scheduler as SGLang on GPUs. It ships as a plugin that installs next to stock SGLang.

## What you get

- **Hardware:** TPU7x (Ironwood), single-host (4 chips) and multi-host (2 hosts at tp=16; 4 hosts for disaggregated prefill/decode).
- **Models:** Gemma 4, GLM, Kimi, gpt-oss, DeepSeek, Qwen, at full size on multi-host TPU. Tiers in [SUPPORT.md](SUPPORT.md).
- **Parallelism:** TP, EP, DPA, PP, CP.
- **Caching:** prefix cache, HiCache KV offload to host memory, prefill/decode disaggregation with TP resharding.
- **Speculative decoding:** EAGLE3, DFlash, DSPark.
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

Launch the quickstart model on a single host and send a request:

```bash
python -m sglang.launch_server --model-path openai/gpt-oss-20b --device tpu --tp 4

curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "Hello"}]}'
```

Next steps:

- [TPU guide](<docs url>): multi-host launch, weights on shared storage, compile cache, and tuning flags.
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

- Kernel and end-to-end tests need a TPU. A maintainer applies the `run-ci` label and comments `/tag-and-rerun-ci` to run them on your pull request.
- [CONTRIBUTING.md](CONTRIBUTING.md) covers the branch-and-PR flow, adding a test to the support matrix, and the policy for changes outside `sglang_tpu/`.

## Community

- Slack: `<channel placeholder>` in the SGLang workspace.
- [GitHub Discussions](<discussions url placeholder>) for questions and design threads.
- [Issues](https://github.com/sgl-project/sglang-torchtpu/issues) for bugs and feature requests, with chip generation, topology, and torch_tpu version in the template.
- Security: see [SECURITY.md](SECURITY.md).

## License and acknowledgments

Apache-2.0. See [NOTICE](NOTICE) for code adapted from [tpu-inference](https://github.com/vllm-project/tpu-inference), [sglang-jax](https://github.com/sgl-project/sglang-jax), [torchtpu-vllm](<torchtpu-vllm url>), and [Helion](https://github.com/pytorch/helion).

SGLang-torchtpu is built by RadixArk together with the Google TPU and torch_tpu teams and Meta. Thanks to the SGLang maintainers and community for the engine this plugin extends.
