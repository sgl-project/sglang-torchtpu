"""One TP1 engine in true on-policy mode, and the checks each model's registered test runs against it."""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import ClassVar

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

# Every file under test/registered carries a registration.
register_tpu_ci(
    est_time=0,
    suite="stage-b-test-tpu-4-v7",
    disabled="helper module, no tests",
)

PROMPTS = [
    "What is 17 * 23? Explain briefly.",
    "Write a haiku about the ocean.",
    "List three prime numbers greater than 100.",
    "Solve for x: 3x + 7 = 25.",
    "Describe a binary search algorithm.",
    "Why do leaves change color in autumn?",
]
# Past one 256-token KV page.
RESPONSE_TOKENS = 300
# Every 7th prefix of each response is prefilled from scratch.
PREFIX_STRIDE = 7


class TrueOnPolicyTestCase(CustomTestCase):
    """Subclasses set model_path, and model_args to serve a few layers of a large model."""

    model_path: ClassVar[str] = ""
    model_args: ClassVar[tuple[str, ...]] = ()
    # A vision-language checkpoint served as its language model (model_args set language_model_only).
    vision_language: ClassVar[bool] = False

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SGLANG_PLATFORM", "tpu")
        cls._env_stack = ExitStack()
        flags = [
            envs.SGLANG_TPU_ENABLE_TRUE_ON_POLICY,
            envs.SGLANG_TPU_USE_SEQ_ALONG_LANE_KV,
        ]
        if cls.vision_language:
            flags.append(envs.SGLANG_TPU_VL_TEXT_ONLY_EXTEND)
            # The warmup request carries an image; the prefix check needs prefills from scratch.
            mode_args = ["--skip-server-warmup", "--disable-radix-cache"]
        else:
            flags.append(envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB)
            mode_args = []
        for flag in flags:
            cls._env_stack.enter_context(flag.override(True))
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model_path,
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
                *mode_args,
                *cls.model_args,
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

    def _decode(self, temperature: float, concurrency: int = len(PROMPTS)):
        """(prompt ids, [(log-prob, token id), ...]) per prompt, `concurrency` prompts decoded at a time."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_path)
        prompts = [tok.encode(p) for p in PROMPTS]

        def sample(i):
            sampling = {
                "temperature": temperature,
                "max_new_tokens": RESPONSE_TOKENS,
                "ignore_eos": True,
                "sampling_seed": i,
            }
            out = self._generate(input_ids=prompts[i], sampling_params=sampling)
            return [(t[0], t[1]) for t in out["meta_info"]["output_token_logprobs"]]

        with ThreadPoolExecutor(concurrency) as pool:
            decoded = list(pool.map(sample, range(len(prompts))))
        for out in decoded:
            self.assertEqual(len(out), RESPONSE_TOKENS)
        return list(zip(prompts, decoded))

    def assert_decode_equals_prefill_rescore(self, temperature: float):
        """Decode log-probs equal the prompt log-probs of a prefill over the same tokens."""
        for prompt, out in self._decode(temperature):
            rescore = self._generate(
                input_ids=prompt + [token for _, token in out],
                sampling_params={"temperature": 1.0, "max_new_tokens": 1},
                # the first returned entry has no log-prob: start one token before the response
                logprob_start_len=len(prompt) - 1,
            )
            got = rescore["meta_info"]["input_token_logprobs"][1:]
            self.assertEqual([t[1] for t in got], [token for _, token in out])
            self.assertEqual([t[0] for t in got], [logprob for logprob, _ in out])

    def assert_decode_equals_prefix_prefill(self, concurrency: int = len(PROMPTS)):
        """A greedy decode step equals the first greedy token of a prefill of the same prefix, from scratch."""
        cases = [
            (prompt + [token for _, token in out[:k]], out[k])
            for prompt, out in self._decode(temperature=0.0, concurrency=concurrency)
            for k in range(0, RESPONSE_TOKENS, PREFIX_STRIDE)
        ]

        def prefill(case):
            prefix, _ = case
            sampling = {"temperature": 0.0, "max_new_tokens": 1}
            out = self._generate(input_ids=prefix, sampling_params=sampling)
            logprob, token = out["meta_info"]["output_token_logprobs"][0][:2]
            return logprob, token

        # Concurrent, so prefills of different lengths share a batch.
        with ThreadPoolExecutor(8) as pool:
            got = list(pool.map(prefill, cases))
        self.assertEqual([token for _, token in got], [c[1][1] for c in cases])
        self.assertEqual([logprob for logprob, _ in got], [c[1][0] for c in cases])

    def assert_top_p_sampling_is_refused(self):
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

    def assert_custom_logit_processor_is_refused(self):
        from sglang.srt.sampling.custom_logit_processor import (
            DisallowedTokensLogitsProcessor,
        )

        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": PROMPTS[0],
                "return_logprob": True,
                "custom_logit_processor": DisallowedTokensLogitsProcessor.to_str(),
                "sampling_params": {
                    "temperature": 1.0,
                    "max_new_tokens": 4,
                    "custom_params": {"token_ids": [0]},
                },
            },
            timeout=600,
        )
        self.assertEqual(resp.status_code, 400, resp.text)
        self.assertIn("refused: a custom logit processor", resp.text)


if __name__ == "__main__":
    # Nothing is collected here; the model files subclass TrueOnPolicyTestCase.
    unittest.main()
