"""Kimi-K3 Raiden P/D with a narrower decode: prefill TP4, decode TP2.

(Attention DP would be the other way to narrow the decode's attention TP, but
Kimi-K3's multimodal wrapper refuses DP attention under TPU compilation.)

Each decode rank holds twice the KDA heads of a prefill rank, so it assembles
its state row from two prefill ranks' rows, and takes the replicated MLA
latent pages whole through the reshard push. Colocated servers at different TP
already part ways within a few greedy tokens of this random model, so the
first token must match the colocated reference and each of the first five
decoded tokens must rank in its top four. How a decode resumed from wrong KDA
state or MLA pages ranks is not measured. The image request only checks that
it is served. Each run prints the checked ranks of every request, which shows
MAX_RANK's margin.
"""

import unittest

from test_tpu_kimi_k3_tiny_pd import TinyPDParity

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=900,
    suite="tpu-integration",
    nightly=True,
    labels=["pd", "singlehost", "kimi-k3", "multimodal", "reshard"],
)


# The first decoded tokens, and the worst rank each may take in the colocated
# reference's distribution.
CHECKED_TOKENS = 5
MAX_RANK = 3


class TestKimiK3TinyPDReshard(TinyPDParity, CustomTestCase):
    # Later flags override the shared TP4 / EP4.
    decode_args = ["--tp-size", "2", "--ep-size", "2"]

    def check_outputs(self, disaggregated, expected, ranks) -> None:
        for i, (got, want, rank) in enumerate(
            zip(disaggregated["text"], expected["text"], ranks)
        ):
            with self.subTest(request=i):
                # The first token comes from the prefill engine, which is the
                # colocated computation.
                self.assertEqual(got[0], want[0])
                self.assertLessEqual(max(rank[:CHECKED_TOKENS]), MAX_RANK, rank)
        # Only that it is served: a reply, and the prompt's image token count.
        image, image_tokens = disaggregated["image"]
        self.assertTrue(image)
        self.assertEqual(image_tokens, expected["image"][1])


if __name__ == "__main__":
    unittest.main()
