"""Muse Spark NVFP4 1P (PP8) + 1D (TP8) DFlash, prefix cache, and KV offload.

Single-host PP=8, EP=1 prefill feeds a single-host TP=8, EP=1 decode over Raiden
pipeline transfer. The prefill pod also hosts the mini-LB and runs four gates:
1. /server_info configuration verification (PP=8 prefill + HiCache, TP=8 decode
   + DFLASH + cpu_tensor retraction backup).
2. Raiden HiCache L1 eviction, L2 host reload, and DFlash speculative verify.
3. >2,048-token SWA-crossing passkey recall and >=2,560-token L1 radix reuse
   over PP=8 -> TP=8 Raiden pipeline transfer.
4. 200-example GSM8K accuracy (>= 0.88) and DFlash accept length (>= 4.0).
"""

import json
import os
import socket
import unittest
import urllib.request
from typing import Any, Mapping

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.ci.metrics import dump_metric
from sglang.test.tpu import MUSE_SPARK
from sglang.test.tpu.utils.kv_offload_probe import (
    generate_with_cache_report,
    kv_offload_runner,
)
from sglang.test.tpu.utils.pd_singlehost_base import PDSingleHostTestCase
from sglang_tpu.testing.pd_pp_spec_perf import engine_configuration_failures

register_tpu_ci(
    est_time=7200,
    suite="tpu-integration",
    nightly=True,
    labels=[
        "singlehost",
        "muse-spark-open-816b-a42b",
        "pd",
        "pp",
        "spec",
        "dflash",
        "prefix-cache",
        "kv-offload",
    ],
)

TEST_NAME = "muse-spark-pd-singlehost-pp8-dflash"
PREFILL_PORT = int(os.environ.get("PREFILL_PORT", "30000"))
DECODE_PORT = int(os.environ.get("DECODE_PORT", "30000"))
PASSKEY = "MARIGOLD-8842"
# Muse Spark's sliding window is 2,048 tokens; >=2,560 cached prefix tokens
# proves both Full-attention recall past the SWA window and L1 radix reuse.
MIN_SWA_CROSSING_CACHED_TOKENS = 2560


def self_hostname() -> str:
    if os.environ.get("PD_SELF_HOST"):
        return os.environ["PD_SELF_HOST"]
    hostnames = [h for h in os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def _get_json(url: str, timeout: int = 30) -> Mapping[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def swa_crossing_passkey_runner(*, test_name: str):
    """Verify >2,048-token SWA-crossing passkey recall and L1 cache hit over PP8 -> TP8."""
    filler = (
        "The autumn wind rustled softly through the quiet valley while "
        "astronomers recorded distant starlight measurements. "
    ) * 180  # ~3,400 tokens (> 2,048 SWA sliding window)
    prefix = (
        f"IMPORTANT RECORD: The secret passkey for vault 749 is {PASSKEY}. "
        "Remember this exact code.\n\n" + filler
    )
    cold_prompt = (
        prefix + "\nFirst verification: State the exact secret passkey for vault 749: "
        "The passkey is"
    )
    warm_prompt = (
        prefix + "\nSecond verification: Repeat the exact secret passkey for vault 749 "
        "once more: The passkey is"
    )

    def run(base_url: str) -> dict[str, Any]:
        flush = requests.post(base_url.rstrip("/") + "/flush_cache", timeout=60)
        flush.raise_for_status()

        cold = generate_with_cache_report(
            base_url=base_url, prompt=cold_prompt, max_new_tokens=24
        )
        warm = generate_with_cache_report(
            base_url=base_url, prompt=warm_prompt, max_new_tokens=24
        )
        print(
            f"[swa-passkey] {test_name}: cold=({cold.device_tokens},{cold.host_tokens}) "
            f"text={cold.text!r} warm=({warm.device_tokens},{warm.host_tokens}) "
            f"text={warm.text!r}",
            flush=True,
        )

        failures: list[str] = []
        if cold.device_tokens or cold.host_tokens:
            failures.append(
                f"cold SWA passkey unexpectedly hit cache "
                f"({cold.device_tokens} device, {cold.host_tokens} host)"
            )
        if PASSKEY not in cold.text:
            failures.append(
                f"cold SWA-crossing passkey missed {PASSKEY!r}: got {cold.text!r}"
            )
        if warm.device_tokens < MIN_SWA_CROSSING_CACHED_TOKENS:
            failures.append(
                f"warm SWA-crossing passkey cached only {warm.device_tokens} device "
                f"tokens, expected >= {MIN_SWA_CROSSING_CACHED_TOKENS} (> 2,048 window)"
            )
        if warm.host_tokens:
            failures.append(
                f"warm SWA-crossing passkey unexpectedly reloaded {warm.host_tokens} "
                "tokens from host"
            )
        if PASSKEY not in warm.text:
            failures.append(
                f"warm SWA-crossing passkey missed {PASSKEY!r}: got {warm.text!r}"
            )
        if cold.spec_verify_ct <= 0 or warm.spec_verify_ct <= 0:
            failures.append(
                f"speculative decoding did not run on SWA passkey "
                f"(cold={cold.spec_verify_ct}, warm={warm.spec_verify_ct})"
            )

        dump_metric(
            f"{test_name}.swa_passkey_cached_tokens",
            warm.device_tokens,
            labels={"model": test_name, "kind": "info", "probe": "swa_passkey"},
        )
        return {
            "score": float(not failures),
            "_passed": not failures,
            "_failures": failures,
        }

    return run


class TestMuseSparkPDSinglehostPP8Dflash(PDSingleHostTestCase):
    """Gate Muse Spark NVFP4 PP8 prefill + TP8 decode DFlash, HiCache, and GSM8K."""

    config_module = (
        "recipe/configs/muse_spark/pd_singlehost_pp8_prefix_cache_kv_offload_dflash.py"
    )
    log_path = "/tmp/pd_muse_spark_pp8_prefill.log"
    decode_log_path = "/tmp/pd_muse_spark_pp8_decode.log"
    decode_host_env = "PD_DECODE_HOST"
    self_hostname = staticmethod(self_hostname)
    tag = "pd-singlehost-muse-spark-pp8"

    @classmethod
    def check_server_configuration(cls) -> None:
        prefill_info = _get_json(f"http://127.0.0.1:{PREFILL_PORT}/server_info")
        decode_info = _get_json(
            f"http://{cls.decode_hostname()}:{DECODE_PORT}/server_info"
        )
        failures = engine_configuration_failures(
            prefill_info=prefill_info,
            decode_info=decode_info,
            algorithm="DFLASH",
        )
        if not prefill_info.get("enable_hierarchical_cache"):
            failures.append(
                "prefill reports enable_hierarchical_cache="
                f"{prefill_info.get('enable_hierarchical_cache')!r}, expected True"
            )
        if decode_info.get("disaggregation_decode_retraction_backup") != "cpu_tensor":
            failures.append(
                "decode reports disaggregation_decode_retraction_backup="
                f"{decode_info.get('disaggregation_decode_retraction_backup')!r}, "
                "expected 'cpu_tensor'"
            )
        if failures:
            raise RuntimeError("; ".join(failures))

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return {
            "model": MUSE_SPARK["served_model_name"],
            "test_name": TEST_NAME,
            "gates": [
                {
                    "dataset": "kv_offload",
                    "label": "kv_offload_dflash",
                    "runner": kv_offload_runner(
                        test_name=TEST_NAME,
                        device_pool_tokens=8192,
                        page_size=128,
                        probe_tokens=2048,
                        hicache_ratio=3,
                        spec_decoding=True,
                    ),
                },
                {
                    "dataset": "swa_passkey",
                    "label": "swa_passkey_pp8_pd",
                    "runner": swa_crossing_passkey_runner(test_name=TEST_NAME),
                },
                {
                    "dataset": "gsm8k",
                    "baseline": 0.98,
                    "atol": 0.10,
                    "num_examples": 200,
                    "num_threads": 8,
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    "reasoning_effort": "low",
                    "spec_accept_length_threshold": 4.0,
                },
            ],
        }

    def test_pp8_prefix_cache_kv_offload_dflash(self) -> None:
        self.run_evaluations()


if __name__ == "__main__":
    unittest.main()
