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
TOP_K = 5
# Two arbitrary ids well inside Qwen3's 151936-token vocabulary. The point of
# token_ids_logprob is that a named id is scored wherever it lands in the
# distribution, so which ids these are does not matter.
NAMED_TOKEN_IDS = [100, 200]


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
        # setdefault: mode B runs these suites with SGLANG_PLATFORM=tpu_oot.
        os.environ.setdefault("SGLANG_PLATFORM", "tpu")
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

    def _prompt_top_logprobs(self):
        """meta_info for a prompt-scoring request that also asks for top-k."""
        return self._generate(
            return_logprob=True, logprob_start_len=0, top_logprobs_num=TOP_K
        )["meta_info"]

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

    def test_prompt_top_logprobs_are_returned(self):
        # The eager gather computes top-k alongside the chosen-token logprobs,
        # but the shim used to copy only the latter onto the output. The
        # scheduler then indexed input_top_logprobs_val per request and the
        # whole request died on a None index.
        info = self._prompt_top_logprobs()
        entries = info["input_top_logprobs"]
        # Same length and same leading null as input_token_logprobs: both index
        # by prompt position, and position 0 has nothing before it to score.
        self.assertEqual(len(entries), info["prompt_tokens"])
        self.assertIsNone(entries[0])
        for row in entries[1:]:
            self.assertEqual(len(row), TOP_K)
            for logprob, token_id, _text in row:
                self.assertLessEqual(logprob, 0.0)
                self.assertEqual(logprob, logprob)  # not NaN
                self.assertIsInstance(token_id, int)

    def test_top_logprobs_bound_the_chosen_token(self):
        # Both lists read the same vocabulary row at each position: the chosen
        # token's logprob is one entry of that row, the top-k are its largest
        # entries, so the best of the top-k cannot be smaller than the chosen
        # one. Shift either list by a position and this goes red -- a count-only
        # check would not, which is what makes this the case worth having.
        info = self._prompt_top_logprobs()
        chosen = info["input_token_logprobs"]
        top = info["input_top_logprobs"]
        self.assertEqual(len(chosen), len(top))
        for pos, (chosen_entry, top_row) in enumerate(
            zip(chosen[1:], top[1:]), start=1
        ):
            best = max(value for value, _token_id, _text in top_row)
            self.assertLessEqual(
                chosen_entry[0],
                best + 1e-4,  # equal when the chosen token is itself the top-1
                f"position {pos}: chosen logprob {chosen_entry[0]} is above the "
                f"largest top-k value {best} from the same vocabulary row",
            )

    def test_top_logprobs_are_ranked_best_first(self):
        # Callers read entry 0 as the model's pick, so descending order is part
        # of the contract, not an accident of the implementation. torch.topk
        # sorts by default but takes sorted=False, and dropping the sort is the
        # kind of thing a throughput change does; every other case here uses
        # max() and would stay green through it.
        for row in self._prompt_top_logprobs()["input_top_logprobs"][1:]:
            values = [value for value, _token_id, _text in row]
            self.assertEqual(values, sorted(values, reverse=True))

    def test_token_ids_logprobs_are_returned(self):
        # The third read of the same row: score ids the caller names, wherever
        # they rank. Top-k cannot stand in -- a named id outside the top k comes
        # back empty. It was dropped by the same line as the top-k fields.
        info = self._generate(
            return_logprob=True,
            logprob_start_len=0,
            token_ids_logprob=NAMED_TOKEN_IDS,
        )["meta_info"]
        entries = info["input_token_ids_logprobs"]
        self.assertEqual(len(entries), info["prompt_tokens"])
        self.assertIsNone(entries[0])
        for row in entries[1:]:
            # Every scored position reports the same named ids, in order.
            self.assertEqual([token_id for _v, token_id, _t in row], NAMED_TOKEN_IDS)

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
