"""Input (prompt) logprobs on the single-host TPU compiled backend.

The compiled forward used to reject these outright: the LogitsProcessor gathers
input logprobs over per-request slices of the prefill, which is data-dependent
and cannot be expressed with static shapes. They are now served by capturing
every prefill row and running that gather eagerly after the forward.

torch.compile is enabled deliberately -- without it the request never reaches
the compiled path this covers, and the test would pass on a broken build.
"""

import os
import unittest
from contextlib import ExitStack

import requests

from sglang.srt.environ import envs
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
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3-0.6b", "logprob"],
)
register_tpu_ci(est_time=300, suite="nightly-tpu-4-v7", nightly=True)

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is Paris, and the capital of Italy is"


def _skip_if_no_tpu():
    try:
        import importlib.util

        if importlib.util.find_spec("torch_tpu") is None:
            return True
    except Exception:
        return True
    return False


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuQwen306bSinglehostPromptLogprob(CustomTestCase):
    """Serve prompt logprobs through the compiled forward."""

    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_PLATFORM"] = "tpu"
        # Prompt logprobs are opt-in: serving them warms the extend graph with
        # FULL capture, which costs every prefill, so the default build rejects
        # the request. override() mutates the real environment, so the server
        # spawned below inherits it; the stack is closed in tearDownClass.
        cls._env_stack = ExitStack()
        cls._env_stack.enter_context(
            envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.override(True)
        )
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--dtype",
                "bfloat16",
                "--device",
                "tpu",
                "--grammar-backend",
                "none",
                "--enable-torch-compile",
                "--tp-size",
                "1",
                "--page-size",
                "64",
                "--chunked-prefill-size",
                "512",
                "--context-length",
                "1024",
                # One extend bucket and one decode shape cover every request
                # below, which keeps precompile off the critical path.
                "--torch-compile-max-bs",
                "1",
                "--tpu-compile-bs",
                "1",
                "--tpu-compile-tokens",
                "64",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls._env_stack.close()

    def _generate(self, **extra):
        payload = {
            "text": PROMPT,
            "sampling_params": {"max_new_tokens": 4, "temperature": 0},
            **extra,
        }
        resp = requests.post(f"{self.base_url}/generate", json=payload, timeout=300)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _prompt_logprobs(self):
        """meta_info for a request that asks to score the prompt itself."""
        return self._generate(return_logprob=True, logprob_start_len=0)["meta_info"]

    def test_prompt_logprobs_are_returned(self):
        info = self._prompt_logprobs()
        entries = info["input_token_logprobs"]
        # One entry per prompt token, and the first carries no score: nothing
        # precedes it to condition on. Both are the cross-backend contract --
        # see test_disaggregation_rust_server.py and test_srt_endpoint.py.
        self.assertEqual(len(entries), info["prompt_tokens"])
        self.assertIsNone(entries[0][0])
        for logprob, token_id, _text in entries[1:]:
            self.assertIsNotNone(logprob)
            self.assertLessEqual(logprob, 0.0)
            self.assertEqual(logprob, logprob)  # not NaN
            self.assertIsInstance(token_id, int)

    def test_prompt_logprobs_are_reproducible(self):
        # Prefill is deterministic, so the same prompt must score identically.
        # This is what makes prompt logprobs usable as a numerics regression
        # signal, and it fails loudly if the eager gather ever reads padded
        # rows (their contents vary run to run).
        first = [entry[0] for entry in self._prompt_logprobs()["input_token_logprobs"]]
        second = [entry[0] for entry in self._prompt_logprobs()["input_token_logprobs"]]
        self.assertEqual(first, second)

    def test_prompt_logprobs_cover_the_whole_prompt(self):
        # A gather that silently read padded rows would still return the right
        # count, so also require the scores to vary: padding is uniform.
        info = self._prompt_logprobs()
        scored = [entry[0] for entry in info["input_token_logprobs"][1:]]
        self.assertEqual(len(scored), info["prompt_tokens"] - 1)
        self.assertGreater(len(set(scored)), 1)

    def test_output_logprobs_still_work(self):
        # Output-only logprobs never entered the graph; guard the regression.
        info = self._generate(return_logprob=True)["meta_info"]
        self.assertGreater(len(info["output_token_logprobs"]), 0)

    def test_plain_request_still_works(self):
        # The pin must be invisible to a batch that asked for nothing.
        data = self._generate()
        self.assertGreater(len(data["text"]), 0)
        self.assertNotIn("input_token_logprobs", data["meta_info"])


if __name__ == "__main__":
    unittest.main()
