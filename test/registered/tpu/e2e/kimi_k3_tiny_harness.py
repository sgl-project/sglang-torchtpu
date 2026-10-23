"""A tiny Kimi-K3 for TPU e2e tests that compare servers with each other.

Kimi-K3 interleaves MLA layers with KDA (linear-attention) layers. Tests that
compare servers in different layouts (P/D, another TP, DP attention) need the
same model in every one of them, so this builds one: a seeded random
checkpoint of the tiny architecture at the 128-lane head dims the TPU kernels
take, which every pod writes byte for byte the same. Dummy weights would not
do: they depend on each parameter's local shard, so servers at different TP
would run different models.

This module registers itself as a disabled helper: run_suite requires a
registration from every file under test/registered, and this one owns no test.
"""

import json
import os
import sys
import tempfile

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_tpu_ci(est_time=0, suite="tpu-integration", disabled="helper module, no tests")

TINY_MODEL = "inference-optimization/Kimi-K3-0.40B"
TINY_MODEL_REVISION = "d853649387ffe8f48ce0198a29ac1a44205031f7"
PROCESSOR_SOURCE = "moonshotai/Kimi-K3"
PROCESSOR_SOURCE_REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
MODEL_OVERRIDE = {
    "text_config": {
        "v_head_dim": 128,
        "linear_attn_config": {
            "full_attn_layers": [4, 8],
            "head_dim": 128,
            "kda_layers": [1, 2, 3, 5, 6, 7],
            "num_heads": 8,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        },
    }
}
SERVER_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--ep-size",
    "4",
    "--attention-backend",
    "pallas_tpu_mla",
    "--page-size",
    "128",
    "--context-length",
    "2048",
    "--chunked-prefill-size",
    "512",
    "--max-prefill-tokens",
    "512",
    "--max-running-requests",
    "4",
    "--mem-fraction-static",
    "0.5",
    "--disable-radix-cache",
    "--grammar-backend",
    "none",
    "--enable-torch-compile",
    "--torch-compile-max-bs",
    "4",
    "--tpu-compile-tokens",
    "512",
    "--tpu-compile-bs",
    "1",
    "4",
    # One sampler for every compared server; greedy picks the argmax either way.
    "--sampling-backend",
    "pytorch",
]
MAX_NEW_TOKENS = 24
TOP_K = 20


# 1300 tokens span three 512-token prefill chunks, so the KDA state the decode
# receives is the one left after chunked prefill.
PROMPTS = [
    [(i * 7919 + seed * 104729) % 50000 + 100 for i in range(length)]
    for length, seed in [(100, 1), (37, 2), (300, 3), (1300, 4)]
]


def build_model(root: str) -> None:
    """Write a seeded random checkpoint of the tiny Kimi-K3 at the TPU head
    dims (``MODEL_OVERRIDE``) to ``root``: the same bytes on every pod."""
    import importlib
    import shutil
    import types
    import zlib

    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from torch import nn

    hub = snapshot_download(
        TINY_MODEL,
        revision=TINY_MODEL_REVISION,
        allow_patterns=["*.json", "*.py", "*.model"],
        ignore_patterns=["checkpoint-*"],
    )
    package_root = tempfile.mkdtemp(prefix="kimi_k3_tiny_code_")
    package = os.path.join(package_root, "kimi_k3_tiny")
    shutil.copytree(hub, package)
    open(os.path.join(package, "__init__.py"), "w").close()

    # The reference modeling code imports fla (flash-linear-attention), which
    # is CUDA-only. Only two of its modules hold parameters; stand in for them
    # with the same parameter names and shapes, and for its kernels -- used
    # only in forward, which building the weights never runs -- with stubs.
    class ShortConvolution(nn.Conv1d):
        def __init__(self, hidden_size, kernel_size, bias=False, **kwargs):
            super().__init__(
                hidden_size,
                hidden_size,
                kernel_size,
                groups=hidden_size,
                padding=kernel_size - 1,
                bias=bias,
            )

    class FusedRMSNormGated(nn.Module):
        def __init__(self, hidden_size, eps=1e-5, elementwise_affine=True, **kwargs):
            super().__init__()
            self.weight = (
                nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None
            )

    def kernel(*args, **kwargs):
        raise RuntimeError("fla kernel stand-in called")

    stand_ins = {
        "fla": {},
        "fla.modules": dict(
            ShortConvolution=ShortConvolution, FusedRMSNormGated=FusedRMSNormGated
        ),
        "fla.ops": {},
        "fla.ops.kda": dict(chunk_kda=kernel, fused_recurrent_kda=kernel),
        "fla.ops.utils": {},
        "fla.ops.utils.index": dict(
            prepare_cu_seqlens_from_mask=kernel, prepare_lens_from_mask=kernel
        ),
        "fla.utils": dict(tensor_cache=lambda fn: fn),
    }
    saved = {name: sys.modules.get(name) for name in stand_ins}
    sys.path.insert(0, package_root)
    default_dtype = torch.get_default_dtype()
    try:
        for name, attrs in stand_ins.items():
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            sys.modules[name] = module
        configuration = importlib.import_module("kimi_k3_tiny.configuration_kimi_k3")
        modeling = importlib.import_module("kimi_k3_tiny.modeling_kimi_k3")
        config = configuration.KimiK3Config.from_pretrained(package)
        text = MODEL_OVERRIDE["text_config"]
        config.text_config.v_head_dim = text["v_head_dim"]
        config.text_config.linear_attn_config = {
            **(config.text_config.linear_attn_config or {}),
            **text["linear_attn_config"],
        }
        torch.set_default_dtype(torch.bfloat16)
        model = modeling.KimiK3ForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(default_dtype)
        sys.path.remove(package_root)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    # Some parameters are allocated uninitialized; give every one a value
    # from a generator seeded by its name. An engine at another TP sums in
    # another order, so near-ties must not decide anything: the MoE router
    # and the LM head are scaled up so each token's experts, and the greedy
    # token, win by a margin well above bf16 rounding.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                param.fill_(1.0)
            elif name.endswith("bias"):
                param.zero_()
            else:
                if name.endswith("block_sparse_moe.gate.weight"):
                    bound = 1.0
                elif "lm_head" in name:
                    bound = 0.5
                else:
                    bound = 0.02
                generator = torch.Generator().manual_seed(zlib.crc32(name.encode()))
                values = torch.empty(param.shape).uniform_(
                    -bound, bound, generator=generator
                )
                param.copy_(values)
    config.auto_map = json.load(open(os.path.join(hub, "config.json")))["auto_map"]
    model.save_pretrained(root, safe_serialization=True)
    for name in os.listdir(hub):
        if name.endswith((".py", ".model")) or name in (
            "tokenizer_config.json",
            "added_tokens.json",
        ):
            shutil.copy(os.path.join(hub, name), root)
    hf_hub_download(
        PROCESSOR_SOURCE,
        "preprocessor_config.json",
        revision=PROCESSOR_SOURCE_REVISION,
        local_dir=root,
    )


def image_url() -> str:
    import base64
    import io

    from PIL import Image

    size = 224
    image = Image.new("RGB", (size, size))
    image.putdata(
        [
            ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
            for y in range(size)
            for x in range(size)
        ]
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def generate(base_url: str, prompt: list[int]) -> list[int]:
    response = requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": prompt,
            "sampling_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["output_ids"]


def chat_with_image(base_url: str, model: str) -> tuple[str, int]:
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url()}},
                        {"type": "text", "text": "Describe the image."},
                    ],
                }
            ],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
        },
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    image_tokens = body["usage"]["prompt_tokens_details"]["image_tokens"]
    return body["choices"][0]["message"]["content"], image_tokens


def run_requests(base_url: str, model: str) -> dict:
    # One request at a time: both servers then decode at batch size 1, so
    # greedy output is comparable token for token.
    return {
        "text": [generate(base_url, prompt) for prompt in PROMPTS],
        "image": chat_with_image(base_url, model),
    }


def token_ranks(base_url: str, prompt: list[int], tokens: list[int]) -> list[int]:
    """Rank of each of ``tokens`` among the server's top candidates when it is
    fed ``prompt + tokens`` (``TOP_K`` when it is not among them)."""
    response = requests.post(
        f"{base_url}/generate",
        json={
            "input_ids": prompt + tokens,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": len(prompt) - 1,
            "top_logprobs_num": TOP_K,
        },
        timeout=600,
    )
    response.raise_for_status()
    # Entry j scores the token at position logprob_start_len + j, so entry 0
    # is the prompt's last token and token k of ``tokens`` is entry k + 1.
    tops = response.json()["meta_info"]["input_top_logprobs"]
    ranks = []
    for k, token in enumerate(tokens):
        candidates = [entry[1] for entry in tops[k + 1]]
        ranks.append(candidates.index(token) if token in candidates else TOP_K)
    return ranks


def read_log(path: str) -> str:
    with open(path, errors="replace") as f:
        return f.read()


def launch(
    model: str, port: int, args: list[str], log_path: str, env: dict | None = None
):
    log = open(log_path, "w")
    # Bind every interface: the peer engine and the router reach these servers
    # by pod name (popen_launch_server takes --host from this URL). It also
    # copies the server's output to this process's stdout/stderr, the pod log.
    process = popen_launch_server(
        model,
        f"http://0.0.0.0:{port}",
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=SERVER_ARGS + args,
        env={"TORCH_LOGS": "recompiles", **(env or {})},
        return_stdout_stderr=(log, log),
        device="tpu",
    )
    return process, log


def assert_no_serving_recompile(test: CustomTestCase, log: str, name: str) -> None:
    # Unfrozen recompiles log "Recompiling function"; frozen ones raise
    # "Detected recompile" and fail the request.
    test.assertNotIn("Detected recompile", log, f"{name} recompiled while serving")
