"""GLM-5.3-Flash MTP: the draft prompt pass takes two passes while serving,
compiles nothing new, and the draft keeps accepting.

Starts the shipped MTP recipe (recipe/configs/glm_5_3_flash/mtp.py: real
weights, one TPU7x host, TP8, NEXTN 3/1/4, index sharing) and sends greedy
prompts that pad to the token buckets the recipe compiles: 1024 (one pass),
2048, 4096 and 8192 (two passes), and a chunked prompt (8192 + 1024). The
prefix cache is flushed before every request, and each prompt runs twice.

Pass: the server logs the two-pass branch at 2048, 4096 and 8192 and never at
1024; nothing recompiles after the post-warmup freeze; every request accepts at
least MIN_ACCEPT_LENGTH tokens per verify. Except for each request's last row,
the draft reads the prompt KV pass 1 stored, so a wrong store would lower the
two-pass prompts' acceptance.

That two passes propose what one pass does is covered by the CPU tests
(test_experimental_kv_only_prompt.py) and the cache-only store's TPU tests
(test_dsa_layer_kv_store_op.py). The single pass is only reachable by a code
edit (KV_ONLY_PROMPT_ENABLED), so this test does not compare against it.

Server divergence from the recipe: SGLANG_PLATFORM=tpu (DSA models run in mode
A only, see sglang_tpu/README.md; this lane runs mode B) and
TORCH_LOGS=recompiles. Weights at /weights/GLM-5.3-Flash, the path the recipe
and its draft read.
"""

import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    # One boot: weight load, then the graph compiles (two boots took 53 min).
    est_time=2700,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "glm-5.3-flash", "spec"],
)

CONFIG_MODULE = "recipe/configs/glm_5_3_flash/mtp.py"
WEIGHTS = "/weights/GLM-5.3-Flash"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3600"))
# The server log lands here, e.g. a job's output directory.
LOG_DIR = os.environ.get("KV_ONLY_PROMPT_LOG_DIR")
MAX_NEW_TOKENS = 64
# Prompt tokens: buckets 1024, 2048, 4096, 8192 and a chunked 8192 + 1024.
PROMPT_LENS = (600, 1500, 3000, 6000, 9000)
ONE_PASS_BUCKET = 1024
TWO_PASS_BUCKETS = (2048, 4096, 8192)
TWO_PASS_LOG = "Draft prompt pass took two passes at token bucket {}."
FREEZE_LOG = "froze torch.compile"
# Every request must accept at least this many tokens per verify. Measured 4.0
# on every request, the maximum with 3 draft tokens; the margin is arbitrary.
MIN_ACCEPT_LENGTH = 3.0

# Repetitive prose: the MTP head drafts it well, so acceptance depends on the
# draft's prompt KV and a wrong store would show up as lower acceptance.
_PASSAGE = (
    "Section {n}. The pump station on the north bank moves water from the river "
    "into the treatment plant. Each pump runs for six hours, rests for two hours, "
    "and is inspected at the end of every week. The operators record the flow, "
    "the pressure and the temperature in the log book before each shift ends. "
)


def _prompts() -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS)
    text = "".join(_PASSAGE.format(n=n) for n in range(1, 400))
    ids = tokenizer(text)["input_ids"]
    assert len(ids) >= max(PROMPT_LENS), len(ids)
    return [ids[:n] for n in PROMPT_LENS]


def _flush_cache() -> None:
    """Drop the prefix cache, so every run of a prompt is a fresh prefill."""
    deadline = time.monotonic() + 120
    while requests.post(f"{BASE_URL}/flush_cache", timeout=60).status_code != 200:
        if time.monotonic() > deadline:
            raise TimeoutError("the server kept refusing /flush_cache")
        time.sleep(2)


def _generate(prompt: list[int]) -> dict:
    _flush_cache()
    start = time.monotonic()
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "input_ids": prompt,
            "sampling_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=1800,
    )
    response.raise_for_status()
    output = response.json()
    meta = output["meta_info"]
    return {
        "output_ids": output["output_ids"],
        "spec_verify_ct": meta["spec_verify_ct"],
        "completion_tokens": meta["completion_tokens"],
        "seconds": time.monotonic() - start,
    }


class TestGlm53FlashMtpKvOnlyPrompt(TPULauncherTestCase):
    config_module = CONFIG_MODULE
    base_url = BASE_URL
    health_timeout = HEALTH_TIMEOUT
    health_endpoint = "/health_generate"
    tag = "glm-5-3-flash-mtp-kv-only-prompt"
    runtime_envs: ClassVar[dict[str, str | None]] = {
        # Drop an inherited MODEL so the recipe's checkpoint path wins.
        "MODEL": None,
        "HOST": urlsplit(BASE_URL).hostname,
        "PORT": str(urlsplit(BASE_URL).port),
        "SGLANG_PLATFORM": "tpu",
        "TORCH_LOGS": "recompiles",
    }

    @classmethod
    def setUpClass(cls) -> None:
        if not Path(WEIGHTS, "config.json").is_file():
            raise FileNotFoundError(
                f"GLM-5.3-Flash weights are not mounted at {WEIGHTS}"
            )
        log_dir = LOG_DIR or tempfile.mkdtemp(prefix="kv_only_prompt_")
        cls.log_path = str(Path(log_dir) / "server.log")
        super().setUpClass()

    def _server_log(self) -> str:
        return Path(self.log_path).read_text(errors="replace")

    def test_two_passes_serve_and_the_draft_accepts(self):
        prompts = _prompts()
        # The startup warmup ran the first forwards; the next idle freezes.
        deadline = time.monotonic() + 600
        while FREEZE_LOG not in self._server_log():
            if time.monotonic() > deadline:
                raise TimeoutError("torch.compile was never frozen")
            time.sleep(5)
        runs = [[_generate(p) for p in prompts] for _ in range(2)]
        accepts = [
            [o["completion_tokens"] / o["spec_verify_ct"] for o in run] for run in runs
        ]
        for run, accept in zip(runs, accepts):
            print(
                "seconds per prompt:",
                [round(o["seconds"], 2) for o in run],
                "accept length:",
                [round(a, 2) for a in accept],
                flush=True,
            )

        log = self._server_log()
        served = log[log.find(FREEZE_LOG) :]
        # Unfrozen recompiles log "Recompiling function"; frozen ones raise
        # "Detected recompile" and fail the request.
        self.assertNotIn("Recompiling function", served)
        self.assertNotIn("Detected recompile", served)
        self.assertNotIn("[direct_compile] new specialization", served)
        for bucket in TWO_PASS_BUCKETS:
            self.assertIn(TWO_PASS_LOG.format(bucket), log)
        self.assertNotIn(TWO_PASS_LOG.format(ONE_PASS_BUCKET), log)
        for accept in accepts:
            for prompt_len, a in zip(PROMPT_LENS, accept):
                self.assertGreaterEqual(a, MIN_ACCEPT_LENGTH, f"{prompt_len} tokens")


if __name__ == "__main__":
    unittest.main()
