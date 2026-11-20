"""True on-policy mode: the engine's decode log-probs equal its own prefill re-score of the same tokens, bit for bit.

An RL trainer recomputes the rollout's log-probs in one packed prefill-shaped forward. With
SGLANG_TPU_ENABLE_TRUE_ON_POLICY the engine runs batch-invariant programs, so a token's log-prob is the same whether
it was decoded inside a batch of concurrent requests or re-scored later in a prefill. Without the mode they differ
from the first token on.
"""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang_tpu.envs import envs

register_tpu_ci(
    est_time=300,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3-0.6b", "accuracy", "logprob", "seq-along-lane"],
)

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "What is 17 * 23? Explain briefly.",
    "Write a haiku about the ocean.",
    "List three prime numbers greater than 100.",
    "Solve for x: 3x + 7 = 25.",
    "Describe a binary search algorithm.",
    "Why do leaves change color in autumn?",
]
RESPONSE_TOKENS = 300


def _skip_if_no_tpu():
    import importlib.util

    return importlib.util.find_spec("torch_tpu") is None


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuQwen306bSinglehostTrueOnPolicy(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SGLANG_PLATFORM", "tpu")
        cls._env_stack = ExitStack()
        for env in (
            envs.SGLANG_TPU_ENABLE_TRUE_ON_POLICY,
            envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV,
            envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB,
        ):
            cls._env_stack.enter_context(env.override(True))
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL,
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
                "--tpu-compile-tracer",
                "direct",
                "--attention-backend",
                "pallas_tpu_rpa_batched",
                "--kv-cache-dtype",
                "bfloat16",
                "--tp-size",
                "1",
                "--page-size",
                "256",
                "--chunked-prefill-size",
                "1024",
                "--context-length",
                "1024",
                "--max-running-requests",
                "8",
                "--torch-compile-max-bs",
                "8",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls._env_stack.close()

    def _generate(self, **payload):
        resp = requests.post(
            f"{self.base_url}/generate",
            json={"return_logprob": True, **payload},
            timeout=600,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _assert_decode_equals_prefill_rescore(self, temperature):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)
        prompts = [tok.encode(p) for p in PROMPTS]

        def sample(i):
            sampling = {
                "temperature": temperature,
                # Past one 256-token KV page, so decode and prefill walk more than one KV block.
                "max_new_tokens": RESPONSE_TOKENS,
                "ignore_eos": True,
                "sampling_seed": i,
            }
            return self._generate(input_ids=prompts[i], sampling_params=sampling)

        with ThreadPoolExecutor(len(prompts)) as pool:  # decode them as one batch
            decoded = list(pool.map(sample, range(len(prompts))))
        for prompt, d in zip(prompts, decoded):
            out = d["meta_info"]["output_token_logprobs"]
            self.assertEqual(len(out), RESPONSE_TOKENS)
            rescore = self._generate(
                input_ids=prompt + [t[1] for t in out],
                sampling_params={"temperature": 1.0, "max_new_tokens": 1},
                # the first returned entry has no log-prob: start one token before the response
                logprob_start_len=len(prompt) - 1,
            )
            got = rescore["meta_info"]["input_token_logprobs"][1:]
            self.assertEqual([t[1] for t in got], [t[1] for t in out])
            self.assertEqual([t[0] for t in got], [t[0] for t in out])

    def test_decode_log_probs_equal_the_prefill_rescore(self):
        self._assert_decode_equals_prefill_rescore(temperature=1.0)

    def test_greedy_log_probs_equal_the_prefill_rescore(self):
        self._assert_decode_equals_prefill_rescore(temperature=0.0)

    def test_top_p_sampling_is_refused(self):
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": PROMPTS[0],
                "return_logprob": True,
                "sampling_params": {
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "max_new_tokens": 4,
                },
            },
            timeout=600,
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("top_p", resp.text)


if __name__ == "__main__":
    unittest.main()
