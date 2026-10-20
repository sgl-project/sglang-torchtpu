# SPDX-License-Identifier: Apache-2.0
"""Attach to a four-host TP32/EP32 Muse Spark NVFP4 server and check generation.

Launch with --quantization modelopt_fp4 and without --enable-multimodal.
Set SGLANG_TPU_USE_LIVE_SERVER=1 and BASE_URL to attach; only text inputs are covered.
"""

import concurrent.futures
import os
import re
import unittest

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.evals.eval_support import wait_for_health

register_tpu_ci(
    est_time=1800,
    suite="tpu-integration",
    nightly=True,
    disabled="manual: attaches to a live four-host Muse server; no CI lane provides one",
    labels=["multihost", "muse-spark-open-816b-a42b", "smoke", "tp", "compile"],
)


@unittest.skipUnless(
    os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
    "the multi-host JobSet must provide the live server",
)
class TestMuseSparkNvfp4Multihost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
        wait_for_health(
            cls.base_url, 3600, endpoint="/health_generate", tag="muse-spark-nvfp4"
        )

    def test_batched_text_generation(self):
        cases = [
            ("What is 17 plus 25?", "42"),
            ("What is 12 multiplied by 7?", "84"),
            (
                "A box has 9 red balls and 6 blue balls. How many balls are in the box?",
                "15",
            ),
            (
                "Sam has 30 dollars and spends 12 dollars. How many dollars remain?",
                "18",
            ),
        ]

        def query(case):
            question, expected = case
            response = requests.post(
                self.base_url + "/v1/chat/completions",
                json={
                    "model": "Muse-Spark-Open-816B-A42B",
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                            + " End with #### followed by the numeric answer.",
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 1024,
                    "chat_template_kwargs": {"reasoning_effort": "low"},
                },
                timeout=1800,
            )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            content = choice["message"].get("content") or ""
            self.assertEqual(choice["finish_reason"], "stop", content)
            self.assertRegex(content, r"####\s*" + re.escape(expected) + r"\b")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(query, cases))


if __name__ == "__main__":
    unittest.main()
