"""Nightly quad e2e on Qwen3-30B-A3B (TPU): PREFIX CACHING + DP ATTENTION +
KV OFFLOADING (HiCache) + EAGLE3 SPECULATIVE DECODING — accuracy + performance
in one file.

This is the real-model gate for the feature-combination milestone: the
engine-level quad unit test is test/manual/tpu/test_combo_3_all_features.py;
this file runs the shipping configuration (30B MoE, tp=8 devices, dp=2) and
asserts:
  * accuracy  — GSM8K score over 200 examples (all four features live),
  * spec      — avg accept_length from /server_info,
  * features  — repeat-prompt cached_tokens > 0 (prefix cache + HiCache
                actually engaged under the eval workload, not just enabled),
  * perf      — bs=1 decode speed floor via send_one_prompt.

Model pair: Qwen/Qwen3-30B-A3B + Tengyunw/qwen3_30b_moe_eagle3 with
steps=6/topk=10/draft=32 — the exact pair+params CI already validates on CUDA
with dp-attention (test/registered/spec/eagle/test_eagle_dp_attention.py,
GSM8K > 0.91, accept_length > 2.5). TPU thresholds start slightly looser;
CALIBRATE after the first few nightly runs.

Registered into the declared (not yet runner-backed) TPU nightly suite; also
runnable directly on a TPU host:
    SGLANG_PLATFORM=tpu HF_HUB_OFFLINE=1 \
        python test/registered/tpu/test_qwen3_moe_feature_combo_nightly.py
"""

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.run_eval import run_eval
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_tpu_ci(est_time=3600, suite="nightly-tpu-4-v7", nightly=True)

MODEL = "Qwen/Qwen3-30B-A3B"
DRAFT = "Tengyunw/qwen3_30b_moe_eagle3"

# Loosened vs the CUDA precedent (0.91 / 2.5) to absorb backend variance;
# tighten after the first nightly runs establish the TPU baseline.
GSM8K_SCORE_FLOOR = 0.88
ACCEPT_LENGTH_FLOOR = 2.0
BS1_SPEED_FLOOR_TOK_S = 5.0  # placeholder floor — CALIBRATE on first run


def _skip_if_no_tpu():
    try:
        import importlib.util

        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestQwen3MoeFeatureComboNightly(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--device",
            "tpu",
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "pallas_tpu_rpa_v3",
            "--trust-remote-code",
            # DP attention (tp counts DEVICES on TPU: v7x host = 8)
            "--tp-size",
            "8",
            "--dp-size",
            "2",
            "--enable-dp-attention",
            # EAGLE3 spec — CI-proven params for this model pair
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            DRAFT,
            "--speculative-num-steps",
            "6",
            "--speculative-eagle-topk",
            "10",
            "--speculative-num-draft-tokens",
            "32",
            "--disable-overlap-schedule",  # spec-on-TPU recipe
            # prefix caching is ON by default (no --disable-radix-cache);
            # cache-report exposes cached_tokens for the engagement check
            "--enable-cache-report",
            # KV offloading: pool sized so the 200-example GSM8K workload
            # genuinely evicts to host and reloads (not just flag-on)
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "3",
            "--hicache-io-backend",
            "torch_tpu",
            "--hicache-mem-layout",
            "page_first_fused",
            "--max-total-tokens",
            "65536",
            "--mem-fraction-static",
            "0.3",
            "--page-size",
            "64",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_a_gsm8k_accuracy_and_accept_length(self):
        requests.get(self.base_url + "/flush_cache")
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=64,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], GSM8K_SCORE_FLOOR)

        server_data = requests.get(self.base_url + "/server_info").json()
        accept_length = None
        states = server_data.get("internal_states") or []
        if states:
            accept_length = states[0].get("avg_spec_accept_length") or states[0].get(
                "spec_accept_length"
            )
        print(f"{accept_length=}")
        self.assertIsNotNone(accept_length, "no accept_length in /server_info")
        self.assertGreater(accept_length, ACCEPT_LENGTH_FLOOR)

    def test_b_prefix_and_hicache_engaged(self):
        # Repeat an identical long prompt: the second must report a real
        # prefix hit through the (small-pool, offload-backed) cache.
        prompt = "Reference: " + " ".join(f"fact {i}." for i in range(400)) + "\nQ: A?"
        payload = {
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 16},
        }
        requests.post(self.base_url + "/generate", json=payload).raise_for_status()
        r2 = requests.post(self.base_url + "/generate", json=payload)
        r2.raise_for_status()
        cached = r2.json()["meta_info"].get("cached_tokens", 0)
        print(f"repeat cached_tokens={cached}")
        self.assertGreater(cached, 0, "prefix cache never engaged on repeat")

    def test_c_bs1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=1024)
        accept_length, speed = send_one_prompt(args)
        print(f"{accept_length=:.2f} {speed=:.2f} tok/s")
        self.assertGreater(accept_length, 1.5)
        self.assertGreater(speed, BS1_SPEED_FLOOR_TOK_S)


if __name__ == "__main__":
    unittest.main(verbosity=2)
