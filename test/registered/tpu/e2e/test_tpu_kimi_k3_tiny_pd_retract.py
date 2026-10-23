"""Kimi-K3 Raiden P/D whose decode retracts requests and resumes them from
their host backup.

SGLANG_TEST_RETRACT makes the decode retract its longest request every third
step (cpu_tensor backup, overlap scheduling on), and the four prompts run at
once so there is a request to retract. A resumed request needs its MLA pages,
its KDA conv and recurrent state and its output tokens to cover the same
positions. A state one token ahead -- what an overlap retraction backed up
before the in-flight result was processed -- decoded 9 and 11 consecutive
tokens that the colocated reference, fed the same prefix, did not rank in its
top TOP_K.

Run at once, the requests' greedy output depends on the batches they share on
this random-weight model (a near tie flips some request's first token in some
runs), so exact equality is not the test. Each disaggregated token is ranked in
the reference's teacher-forced distribution instead, and no request may miss
the top TOP_K more than MAX_MISS_RUN times in a row. A correct resume misses at
isolated positions, where the decode and prefill kernels disagree: at most two
in a row over 20 request runs. The decode's metrics must show retractions.
"""

import concurrent.futures
import unittest

import requests
from kimi_k3_tiny_harness import PROMPTS, TOP_K, generate
from test_tpu_kimi_k3_tiny_pd import DECODE_PORT, TinyPDParity, _decode_hostname

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=900,
    suite="tpu-integration",
    nightly=True,
    labels=["pd", "singlehost", "kimi-k3", "retraction"],
)

# The most consecutive tokens of one request that may fall outside the
# reference's top TOP_K.
MAX_MISS_RUN = 4
RETRACTED_METRIC = "sglang:num_retracted_requests_total"


def _longest_miss_run(ranks: list[int]) -> int:
    longest = run = 0
    for rank in ranks:
        run = run + 1 if rank >= TOP_K else 0
        longest = max(longest, run)
    return longest


class TestKimiK3TinyPDRetract(TinyPDParity, CustomTestCase):
    decode_args = ["--enable-metrics"]
    decode_env = {"SGLANG_TEST_RETRACT": "1", "SGLANG_TEST_RETRACT_INTERVAL": "3"}

    def serve_requests(self, base_url: str, model: str) -> dict:
        # At once, so the decode holds several requests and has one to retract.
        with concurrent.futures.ThreadPoolExecutor(len(PROMPTS)) as pool:
            return {"text": list(pool.map(lambda p: generate(base_url, p), PROMPTS))}

    def check_outputs(self, disaggregated, expected, ranks) -> None:
        metrics = requests.get(
            f"http://{_decode_hostname()}:{DECODE_PORT}/metrics", timeout=60
        )
        metrics.raise_for_status()
        retracted = sum(
            float(line.split()[-1])
            for line in metrics.text.splitlines()
            if line.startswith(RETRACTED_METRIC)
        )
        print(f"the decode retracted {retracted:.0f} requests")
        self.assertGreater(retracted, 0)
        for i, rank in enumerate(ranks):
            with self.subTest(request=i):
                self.assertLessEqual(_longest_miss_run(rank), MAX_MISS_RUN, rank)


if __name__ == "__main__":
    unittest.main()
