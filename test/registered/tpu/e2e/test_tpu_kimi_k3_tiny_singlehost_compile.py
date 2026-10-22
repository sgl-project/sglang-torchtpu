"""Kimi-K3 TPU compile boundary on a tiny dummy-weight model.

Text prefill, decode and image requests run with no Dynamo recompile after
precompile, and greedy output matches an eager server.
"""

import base64
import io
import json
import tempfile
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_tpu_ci(
    est_time=150,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "kimi-k3", "compile", "multimodal"],
)

TINY_MODEL = "inference-optimization/Kimi-K3-0.40B"
TINY_MODEL_REVISION = "d853649387ffe8f48ce0198a29ac1a44205031f7"
# The tiny repo ships no image processor config; the full model's applies.
PROCESSOR_SOURCE = "moonshotai/Kimi-K3"
PROCESSOR_SOURCE_REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
# Dummy weights. The Pallas KDA kernel needs head_dim 128 (the checkpoint has
# 32); v_head_dim follows it as in the full model.
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
    "--load-format",
    "dummy",
    "--json-model-override-args",
    json.dumps(MODEL_OVERRIDE),
    "--device",
    "tpu",
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
    # One extend bucket and decode buckets {1, 4} cover every request below.
    "--torch-compile-max-bs",
    "4",
    "--tpu-compile-tokens",
    "512",
    "--tpu-compile-bs",
    "1",
    "4",
]
MAX_NEW_TOKENS = 16
# Every rank logs this before the server takes requests.
PRECOMPILE_DONE = "TpuGraphRunner precompile done"


def _download_model(root: str) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    # Config, tokenizer and remote code only: the weights are dummy.
    snapshot_download(
        TINY_MODEL,
        revision=TINY_MODEL_REVISION,
        local_dir=root,
        allow_patterns=["*.json", "*.py", "*.model"],
        ignore_patterns=["checkpoint-*"],
    )
    hf_hub_download(
        PROCESSOR_SOURCE,
        "preprocessor_config.json",
        revision=PROCESSOR_SOURCE_REVISION,
        local_dir=root,
    )


def _prompt(length: int, seed: int) -> list[int]:
    return [(i * 7919 + seed * 104729) % 50000 + 100 for i in range(length)]


# A single prefill, then two uneven prompts whose decode pads bs2 to bs4.
TEXT_BATCHES = [[_prompt(100, 1)], [_prompt(37, 2), _prompt(300, 3)]]


def _image_url(width: int = 224, height: int = 224) -> str:
    from PIL import Image

    image = Image.new("RGB", (width, height))
    image.putdata(
        [
            ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
            for y in range(height)
            for x in range(width)
        ]
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _generate(prompts: list[list[int]]) -> list[list[int]]:
    response = requests.post(
        f"{DEFAULT_URL_FOR_TEST}/generate",
        json={
            "input_ids": prompts,
            "sampling_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    response.raise_for_status()
    return [output["output_ids"] for output in response.json()]


def _chat_with_image(
    model: str, width: int = 224, height: int = 224
) -> tuple[str, int]:
    response = requests.post(
        f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_url(width, height)},
                        },
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


def _read(log) -> str:
    with open(log.name, errors="replace") as f:
        return f.read()


def _serve(model: str, *, compile: bool) -> tuple[dict, str]:
    """Run the requests on a fresh server; return the outputs and its log."""
    log = tempfile.NamedTemporaryFile(mode="w+", suffix=".log")
    process = popen_launch_server(
        model,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=SERVER_ARGS + (["--enable-torch-compile"] if compile else []),
        env={
            "TORCH_LOGS": "recompiles",
            # The vision buckets up to the startup warmup's 32x32 grid.
            "SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES": "1024",
        },
        return_stdout_stderr=(log, log),
        device="tpu",
    )
    try:
        if compile:
            # The startup warmup ran the first forward; the next idle freezes.
            deadline = time.monotonic() + 120
            while "froze torch.compile" not in _read(log):
                if time.monotonic() > deadline:
                    raise TimeoutError("torch.compile was never frozen")
                time.sleep(1)
        outputs = {
            "text": [_generate(batch) for batch in TEXT_BATCHES],
            "image": _chat_with_image(model),
            # 32x16 patches: padded into the 24x24 vision bucket.
            "wide_image": _chat_with_image(model, width=448, height=224),
            "text_after_image": _generate(TEXT_BATCHES[0]),
        }
        return outputs, _read(log)
    finally:
        kill_process_tree(process.pid, wait_timeout=60)
        log.close()


class TestKimiK3TinyCompileBoundary(CustomTestCase):
    def test_compiled_serving_matches_eager_without_recompiles(self):
        model_dir = tempfile.TemporaryDirectory(prefix="kimi_k3_tiny_")
        self.addCleanup(model_dir.cleanup)
        _download_model(model_dir.name)
        eager, _ = _serve(model_dir.name, compile=False)
        compiled, log = _serve(model_dir.name, compile=True)

        # From the last precompile on: the window holds the startup image
        # warmup and its decode steps, which run before the freeze.
        start = log.rfind(PRECOMPILE_DONE)
        self.assertGreaterEqual(start, 0, f"{PRECOMPILE_DONE!r} not logged")
        served_log = log[start:]
        # Unfrozen recompiles log "Recompiling function"; frozen ones raise
        # "Detected recompile" and fail the request.
        self.assertNotIn("Recompiling function", served_log)
        self.assertNotIn("Detected recompile", served_log)
        self.assertGreater(compiled["image"][1], 0)
        self.assertGreater(compiled["wide_image"][1], compiled["image"][1])
        self.assertEqual(compiled, eager)


if __name__ == "__main__":
    unittest.main()
