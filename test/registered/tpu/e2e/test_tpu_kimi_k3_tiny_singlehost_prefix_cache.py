"""Kimi-K3 radix prefix cache on a tiny dummy-weight model.

Repeated and branched prompts restore KDA checkpoints from the radix cache, the
compiled radix-on server's greedy output matches a compiled radix-off server's,
and nothing recompiles after precompile.
"""

import tempfile
import time
import unittest

import requests

# Shares the compile test's tiny model, server flags and helpers.
import test_tpu_kimi_k3_tiny_singlehost_compile as tiny_k3

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_tpu_ci(
    est_time=300,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "kimi-k3", "compile", "prefix-cache"],
)

SHARED = tiny_k3._prompt(900, 11)
BRANCH = SHARED[:700] + tiny_k3._prompt(200, 12)
# Run in order: a repeat and a branch of the first prompt, then one batch that
# mixes a full prefix hit, a partial one and a miss.
STEPS = (
    ("first", [SHARED]),
    ("repeat", [SHARED]),
    ("branch", [BRANCH]),
    ("mixed", [SHARED, BRANCH, tiny_k3._prompt(300, 13)]),
)


def _generate(prompts: list[list[int]]) -> list[tuple[list[int], int]]:
    """Greedy output ids and cached prompt tokens, per prompt."""
    response = requests.post(
        f"{DEFAULT_URL_FOR_TEST}/generate",
        json={
            "input_ids": prompts,
            "sampling_params": {
                "max_new_tokens": tiny_k3.MAX_NEW_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    response.raise_for_status()
    return [
        (output["output_ids"], output["meta_info"]["cached_tokens"])
        for output in response.json()
    ]


def _serve(model: str, *, radix: bool) -> tuple[dict, str]:
    """Run STEPS on a fresh compiled server; return their outputs and its log."""
    other_args = tiny_k3.SERVER_ARGS + ["--enable-torch-compile"]
    if radix:
        other_args = [arg for arg in other_args if arg != "--disable-radix-cache"]
    log = tempfile.NamedTemporaryFile(mode="w+", suffix=".log")
    process = popen_launch_server(
        model,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        env={"TORCH_LOGS": "recompiles"},
        return_stdout_stderr=(log, log),
        device="tpu",
    )
    try:
        # The startup warmup ran the first forward; the next idle freezes.
        deadline = time.monotonic() + 120
        while "froze torch.compile" not in tiny_k3._read(log):
            if time.monotonic() > deadline:
                raise TimeoutError("torch.compile was never frozen")
            time.sleep(1)
        outputs = {name: _generate(prompts) for name, prompts in STEPS}
        return outputs, tiny_k3._read(log)
    finally:
        kill_process_tree(process.pid, wait_timeout=60)
        log.close()


class TestKimiK3TinyPrefixCache(CustomTestCase):
    def test_radix_hits_match_radix_off_without_recompiles(self):
        model_dir = tempfile.TemporaryDirectory(prefix="kimi_k3_tiny_")
        self.addCleanup(model_dir.cleanup)
        tiny_k3._download_model(model_dir.name)
        radix_off, _ = _serve(model_dir.name, radix=False)
        radix_on, log = _serve(model_dir.name, radix=True)

        for name in ("repeat", "branch"):
            for _, cached_tokens in radix_on[name]:
                self.assertGreater(cached_tokens, 0, f"{name!r} reused no prefix")
        start = log.rfind(tiny_k3.PRECOMPILE_DONE)
        self.assertGreaterEqual(start, 0, f"{tiny_k3.PRECOMPILE_DONE!r} not logged")
        served_log = log[start:]
        # Unfrozen recompiles log "Recompiling function"; frozen ones raise
        # "Detected recompile" and fail the request.
        self.assertNotIn("Recompiling function", served_log)
        self.assertNotIn("Detected recompile", served_log)
        # A checkpoint taken at the wrong token, or restored into the wrong
        # slot, changes the greedy tokens that follow the prefix hit.
        for name, _ in STEPS:
            self.assertEqual(
                [ids for ids, _ in radix_on[name]],
                [ids for ids, _ in radix_off[name]],
                name,
            )


if __name__ == "__main__":
    unittest.main()
