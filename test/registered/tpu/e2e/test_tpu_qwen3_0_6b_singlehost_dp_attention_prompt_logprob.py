"""Input (prompt) logprobs under DP attention on the TPU compiled backend.

The compiled forward serves input logprobs by re-running the LogitsProcessor
eagerly after the forward. Its logits all-gather is a program over the whole
mesh, so under DP attention every DP rank has to run it -- including the ones
whose batch asked for nothing -- as many times, at one shape. The scheduler's
MLP sync carries the largest gather of the step for that, which every rank
chunks alike; without it a prompt-logprob request hung the worker group.

Served at TP4 with two attention-DP groups and the LM head on the attention-TP
group, with plain traffic keeping the other group busy. Each position's top-1
is checked against the same server's greedy next token after that prefix, and
against an off-by-one control that a misaligned gather would match instead;
and a request's logprobs must not change when the other group gathers in the
same step, which the server's debug log of each step's gather confirms.
"""

import json
import os
import re
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
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
    est_time=400,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3-0.6b", "logprob", "dpa"],
)

MODEL = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "The capital of France is Paris, and the capital of Italy is",
    "In 1969, the first humans to walk on the Moon were Neil Armstrong and",
    "To make a cup of tea, first boil the water, then",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
]
# About 600 tokens, so it prefills over several --chunked-prefill-size steps,
# each gathering many more rows than any of PROMPTS.
LONG_PROMPT = " ".join(["The quick brown fox jumps over the lazy dog."] * 60)
TOP_K = 5
# A stalled collective shows as a request that never returns.
TIMEOUT = 300
# A pair shares a step when the short prompt reaches its scheduler while
# LONG_PROMPT is still prefilling, and misses when it is scheduled first, so it
# is retried until the log shows one did. The bound is arbitrary.
PAIR_ATTEMPTS = 20
# Loaded through SGLANG_LOGGING_CONFIG_PATH: the shim's per-step gather line is
# a debug log, so turn that one logger up and leave the rest at INFO.
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"plain": {"format": "[%(asctime)s] %(message)s"}},
    "handlers": {
        "stderr": {
            "class": "logging.StreamHandler",
            "formatter": "plain",
            "stream": "ext://sys.stderr",
        }
    },
    "root": {"level": "INFO", "handlers": ["stderr"]},
    "loggers": {"sglang_tpu.compilation.compiled_forward_shim": {"level": "DEBUG"}},
}
# One rank's gathers in one step: the DP-wide and local rows, then how many
# gathers of how many rows. Local rows below the DP-wide count, but not zero,
# mean another DP rank gathered more rows in that same step.
GATHER_LINE = re.compile(
    r"input-logprob gather: DP-wide (\d+) rows, local (\d+) rows \((\d+) x (\d+) rows\)"
)


def _skip_if_no_tpu():
    try:
        import importlib.util

        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuQwen306bSinglehostDPAttentionPromptLogprob(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SGLANG_PLATFORM", "tpu")
        cls._env_stack = ExitStack()
        cls._env_stack.enter_context(
            envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.override(True)
        )
        # Above every one of PROMPTS, below a LONG_PROMPT step: that step's
        # rows take several gathers, which a short prompt sharing it tops its
        # one chunk up to with zero rows.
        cls._env_stack.enter_context(envs.SGLANG_LOGPROB_CHUNK_SIZE.override(32))
        log_dir = cls._env_stack.enter_context(tempfile.TemporaryDirectory())
        logging_config = os.path.join(log_dir, "logging.json")
        with open(logging_config, "w") as f:
            json.dump(LOGGING_CONFIG, f)
        cls.log = cls._env_stack.enter_context(
            open(os.path.join(log_dir, "server.log"), "w")
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={"SGLANG_LOGGING_CONFIG_PATH": logging_config},
            return_stdout_stderr=(cls.log, cls.log),
            other_args=[
                "--dtype",
                "bfloat16",
                "--device",
                "tpu",
                "--grammar-backend",
                "none",
                "--enable-torch-compile",
                "--tp-size",
                "4",
                "--enable-dp-attention",
                "--dp-size",
                "2",
                "--enable-dp-lm-head",
                "--page-size",
                "64",
                "--chunked-prefill-size",
                "256",
                "--context-length",
                "1024",
                "--max-running-requests",
                "8",
                "--torch-compile-max-bs",
                "4",
                "--tpu-compile-bs",
                "1",
                "4",
                "--tpu-compile-tokens",
                "256",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls._env_stack.close()

    def _post(self, payload):
        resp = requests.post(f"{self.base_url}/generate", json=payload, timeout=TIMEOUT)
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _score(self, prompt, rank=None):
        """meta_info of a request scoring its whole prompt, with top-k."""
        return self._post(
            {
                "text": prompt,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": TOP_K,
                "routed_dp_rank": rank,
            }
        )["meta_info"]

    def _greedy(self, input_ids, rank):
        return self._post(
            {
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "routed_dp_rank": rank,
            }
        )["output_ids"][0]

    def _score_pair(self, prompt, rank):
        """Score prompt on rank while the other DP rank scores LONG_PROMPT,
        both sent at once."""
        start = threading.Barrier(2)

        def score(args):
            start.wait()
            return self._score(*args)

        with ThreadPoolExecutor(2) as pool:
            together, _long = pool.map(score, [(prompt, rank), (LONG_PROMPT, 1 - rank)])
        return together

    def _shared_a_step(self, since):
        """Whether the server logged, past byte offset since, a DP rank
        gathering input logprobs in one step with another holding more rows,
        over several gathers: its own and the zero-row ones topping it up."""
        with open(self.log.name, "rb") as f:
            f.seek(since)
            log = f.read().decode(errors="replace")
        return any(
            0 < int(local) < int(dp_wide) and int(count) > 1
            for dp_wide, local, count, _rows in GATHER_LINE.findall(log)
        )

    def _with_background_traffic(self, fn):
        """Run fn while plain requests keep both DP groups busy."""
        stop = threading.Event()
        errors = []

        def background():
            try:
                while not stop.is_set():
                    self._post(
                        {
                            "text": PROMPTS[1],
                            "sampling_params": {
                                "max_new_tokens": 32,
                                "temperature": 0,
                            },
                        }
                    )
            except Exception as e:  # surfaced below
                errors.append(e)

        threads = [threading.Thread(target=background) for _ in range(3)]
        for thread in threads:
            thread.start()
        try:
            return fn()
        finally:
            stop.set()
            for thread in threads:
                thread.join()
            if errors:
                raise errors[0]

    def test_prompt_logprobs_are_served(self):
        info = self._with_background_traffic(lambda: self._score(PROMPTS[0]))
        entries = info["input_token_logprobs"]
        self.assertEqual(len(entries), info["prompt_tokens"])
        self.assertIsNone(entries[0][0])
        scored = [logprob for logprob, _id, _text in entries[1:]]
        for logprob in scored:
            self.assertLessEqual(logprob, 0.0)
            self.assertEqual(logprob, logprob)  # not NaN
        # Padded rows are uniform; real rows are not.
        self.assertGreater(len(set(scored)), 1)
        tops = info["input_top_logprobs"]
        self.assertEqual(len(tops), info["prompt_tokens"])
        for row in tops[1:]:
            values = [value for value, _id, _text in row]
            self.assertEqual(len(values), TOP_K)
            self.assertEqual(values, sorted(values, reverse=True))

    def test_positions_agree_with_greedy_decoding(self):
        """The top-1 at prompt position k is the server's greedy next token
        after the first k tokens; a gather misaligned by one position would
        match the next position's greedy token instead."""
        # Each prompt and its prefixes on one DP rank, both ranks covered:
        # the two DP groups' forwards are not bit-identical to each other.
        ranks = [i % 2 for i in range(len(PROMPTS))]
        infos = self._with_background_traffic(
            lambda: [self._score(p, rank) for p, rank in zip(PROMPTS, ranks)]
        )
        same = shifted = total = 0
        for info, rank in zip(infos, ranks):
            ids = [token_id for _lp, token_id, _text in info["input_token_logprobs"]]
            top1 = [None] + [row[0][1] for row in info["input_top_logprobs"][1:]]
            for cut in range(2, len(ids)):
                greedy = self._greedy(ids[:cut], rank)
                total += 1
                same += top1[cut] == greedy
                shifted += cut + 1 < len(ids) and top1[cut + 1] == greedy
        print(f"greedy agreement {same}/{total}, off-by-one {shifted}/{total}")
        self.assertGreaterEqual(same / total, 0.9)
        self.assertGreater(same / total, shifted / total + 0.3)

    def test_the_other_groups_gather_leaves_a_request_unchanged(self):
        """Both DP groups ask for input logprobs in one step, at different row
        counts, so each runs the same DP-wide gathers: LONG_PROMPT's rows in
        several chunks, the short prompt's one chunk topped up with zero rows.
        A request's logprobs are bit-identical to its own rank's run alone.
        The pair is retried until the server's log shows such a step."""
        for prompt in PROMPTS:
            for rank in (0, 1):
                alone = self._score(prompt, rank)
                since = os.path.getsize(self.log.name)
                for attempt in range(1, PAIR_ATTEMPTS + 1):
                    together = self._score_pair(prompt, rank)
                    self.assertEqual(
                        together["input_token_logprobs"],
                        alone["input_token_logprobs"],
                        f"rank {rank}: {prompt!r}",
                    )
                    self.assertEqual(
                        together["input_top_logprobs"], alone["input_top_logprobs"]
                    )
                    if self._shared_a_step(since):
                        break
                else:
                    self.fail(
                        f"rank {rank}: {prompt!r} never gathered in one step with "
                        f"the other DP rank in {PAIR_ATTEMPTS} attempts"
                    )
                print(f"rank {rank}: {prompt!r} shared a step on attempt {attempt}")


if __name__ == "__main__":
    unittest.main()
