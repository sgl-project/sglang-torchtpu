"""Kimi-K3.1 native server recipe and its cookbook profile (CPU only, no server)."""

import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmark.tpu.recipe.configs.kimi_k3.server import (  # noqa: E402
    CONFIG as K3_CONFIG,
)
from benchmark.tpu.recipe.configs.kimi_k3_1.server import (  # noqa: E402
    CONFIG,
    with_libtpu_flags,
)
from benchmark.tpu.recipe.configs.profiles import (  # noqa: E402
    PROFILES,
    apply_profile,
)


class TestKimiK31Server(unittest.TestCase):
    def test_sizing_is_consistent(self):
        args = CONFIG["server_args"]
        # One prefill bucket, and it is the chunk the scheduler cuts.
        self.assertEqual(
            args["tpu_compile_tokens"],
            [args["chunked_prefill_size"]],
        )
        self.assertEqual(args["chunked_prefill_size"], args["max_prefill_tokens"])
        self.assertEqual(args["torch_compile_max_bs"], args["max_running_requests"])
        # The HTTP warmup would send an image request to the text-only server.
        self.assertTrue(args["skip_server_warmup"])
        self.assertNotIn("speculative_algorithm", args)

    def test_runtime_env_extends_the_k3_recipe(self):
        env = CONFIG["runtime_env"]
        for name, value in K3_CONFIG["runtime_env"].items():
            if name != "LIBTPU_INIT_ARGS":
                self.assertEqual(env[name], value, name)
        self.assertEqual(env["SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT"], "1")

    def test_with_libtpu_flags_replaces_and_moves_to_the_end(self):
        self.assertEqual(
            with_libtpu_flags("--a=1 --b=2 --c=3", b="9", d="4"),
            "--a=1 --c=3 --b=9 --d=4",
        )

    def test_cookbook_profile_restates_the_recipe_sizing(self):
        """The cookbook renders server.py with long_128k_c64; it must not resize it."""
        sized = apply_profile(CONFIG, PROFILES["long_128k_c64"])
        self.assertEqual(sized, CONFIG)


if __name__ == "__main__":
    unittest.main()
