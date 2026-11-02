"""GPT-OSS-120B block-major KV layout gate on a Raiden P/D pair with KV offload.

Both engines run with SGLANG_TPU_USE_BLOCK_MAJOR_KV=1, one host each, and the
prefill engine adds the Raiden HiCache host tier. The layout is env-driven and
invisible from the API, so each pod reads the pool allocation lines back from
its own launcher log before anything else is scored. The prefill pod then
checks that the Raiden host tier came up, runs the device-hit / churn /
host-reload probe through the router, and evaluates GSM8K.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import unittest
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

import msgspec

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.evals.eval_support import wait_for_health
from sglang.test.tpu.utils.kv_offload_probe import (
    EvictionPlan,
    plan_eviction,
    probe_prompt,
    run_probe,
    validate_probe,
)
from sglang.test.tpu.utils.pd_singlehost_base import PDSingleHostTestCase

register_tpu_ci(
    est_time=3600,
    suite="tpu-integration",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "pd", "kv-offload", "block-major"],
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")

LAUNCH_CONFIG_MODULE = "recipe/configs/gpt_oss_120b/pd_singlehost_block_major.py"

E = os.environ.get

SERVED_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"
MODEL_ENV = "GPT_OSS_MODEL"

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
SELF_HOST_ENV = "PD_SELF_HOST"
DECODE_HOST_ENV = "PD_DECODE_HOST"

HOSTS_PER_ENGINE = 1
CHIPLETS_PER_HOST = 8
TP_SIZE = HOSTS_PER_ENGINE * CHIPLETS_PER_HOST
EP_SIZE = TP_SIZE

PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LOCAL_HOST = "127.0.0.1"

WAIT_SECS = int(E("WAIT_SECS", "3600"))
HEALTH_TIMEOUT = float(E("HEALTH_TIMEOUT", "3600"))
PYTHON = E("PYTHON", "python3")
RENDER_JOBSET = os.path.join(REPO, ".github", "tpu-ci", "render_jobset.py")

# Short enough for the generated Job and Pod names, without role words.
RUN_NAME = "pd-sh-gpt-120b-block-major"

PREFILL_LOG = "/tmp/pd_block_major_prefill.log"
DECODE_LOG = "/tmp/pd_block_major_decode.log"

PAGE_SIZE = 64
# The recipe's max_total_tokens on each engine. The running engine splits it into
# a full and an SWA device pool, so the probe reads the pools from /server_info.
MAX_TOTAL_TOKENS = 393216
# Words in the probe prompt and in each filler, about one token each.
PROBE_TOKENS = int(E("CACHE_PROBE_TOKENS", "2048"))
# /flush_cache waits this long for the prefill engine to go idle.
FLUSH_TIMEOUT_S = 300
DISAGG_TRANSFER_BACKEND = "raiden"
# Raiden handles P/D KV transfer, the prefill host tier and decode host-pool I/O.
HICACHE_IO_BACKEND = "raiden"

BLOCK_MAJOR_ENV = "SGLANG_TPU_USE_BLOCK_MAJOR_KV"
# Logged by sglang_tpu/kv_pool.py once per pool per rank.
BUNDLE_LINE = "Allocating block-major TPU KV bundle"
FUSED_LINE = "Allocating fused TPU KV buffers"
# Logged by sglang_tpu/raiden_transfer.py when the host tier is up.
RAIDEN_READY_MARKER = "Raiden KV transfer ready"

# Measured 0.97 accuracy on this topology; 0.80 serves as the CI passing floor.
GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B", E("GSM8K_THRESHOLD", "0.80")))
ATOL = float(E("GSM8K_ATOL", "0.03"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "100"))
NUM_THREADS = int(E("GSM8K_NUM_THREADS", "8"))
MAX_TOKENS = int(E("GSM8K_MAX_TOKENS", "1024"))


def resolve_model_path(env: Mapping[str, str] | None = None) -> str:
    """Resolve the checkpoint from the environment, else the mounted default."""
    values = os.environ if env is None else env
    return values.get(MODEL_ENV) or values.get("MODEL") or DEFAULT_MODEL_PATH


def validate_model_source(model_path: str) -> None:
    """Fail clearly when CI points at an unprovisioned local checkpoint."""
    if not os.path.isabs(model_path):
        return  # HF repo ids and bucket URIs resolve remotely
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"GPT-OSS-120B checkpoint incomplete or missing at {model_path}, "
            f"provision it on the weights volume or set {MODEL_ENV}"
        )


def self_hostname(env: Mapping[str, str] | None = None) -> str:
    """This pod's own DNS name."""
    values = os.environ if env is None else env
    if values.get(SELF_HOST_ENV):
        return values[SELF_HOST_ENV]
    hostnames = [h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _read_log(log_path: str) -> str:
    with open(log_path, "rb") as f:
        return f.read().decode(errors="replace")


class PoolAllocationCounts(msgspec.Struct, frozen=True, kw_only=True):
    bundles: int
    fused: int


def count_pool_allocations(log_text: str) -> PoolAllocationCounts:
    """Count the block-major and layer-major pool allocations a server logged."""
    return PoolAllocationCounts(
        bundles=log_text.count(BUNDLE_LINE), fused=log_text.count(FUSED_LINE)
    )


def layout_failures(*, role: str, counts: PoolAllocationCounts) -> list[str]:
    """Reject an engine that allocated any pool layer-major, or no bundle at all."""
    failures = []
    if counts.bundles == 0:
        failures.append(
            f"{role} engine allocated no block-major bundle: "
            f"{BLOCK_MAJOR_ENV} did not reach it"
        )
    if counts.fused:
        failures.append(
            f"{role} engine allocated {counts.fused} pool(s) layer-major "
            "beside the bundles"
        )
    return failures


def check_layout(*, role: str, log_path: str) -> list[str]:
    """Read one engine's launcher log and report its pool layout."""
    counts = count_pool_allocations(_read_log(log_path))
    failures = layout_failures(role=role, counts=counts)
    print(
        f"[pd-block-major] {role} layout: bundle allocations={counts.bundles} "
        f"fused allocations={counts.fused} ({log_path})",
        flush=True,
    )
    return failures


def block_major_layout_runner(*, log_path: str):
    """Gate on the prefill engine having allocated every pool as a bundle."""

    def run(_base_url: str) -> dict[str, Any]:
        failures = check_layout(role=PREFILL_ROLE, log_path=log_path)
        return {
            "score": 0.0 if failures else 1.0,
            "_passed": not failures,
            "_failures": failures,
        }

    return run


def engine_configuration_failures(
    *, prefill_info: Mapping[str, Any], decode_info: Mapping[str, Any]
) -> list[str]:
    """Reject a run whose engines are not a Raiden P/D pair with a prefill host tier."""
    failures = []
    for label, info, expected_mode in (
        (PREFILL_ROLE, prefill_info, PREFILL_ROLE),
        (DECODE_ROLE, decode_info, DECODE_ROLE),
    ):
        mode = info.get("disaggregation_mode")
        if mode != expected_mode:
            failures.append(
                f"{label} engine reports disaggregation_mode={mode!r}, expected "
                f"{expected_mode!r}: it is not serving a P/D role"
            )
        backend = info.get("disaggregation_transfer_backend")
        if backend != DISAGG_TRANSFER_BACKEND:
            failures.append(
                f"{label} engine reports disaggregation_transfer_backend="
                f"{backend!r}, expected {DISAGG_TRANSFER_BACKEND!r}"
            )
    # The host-reload leg of the probe is void without the prefill host tier.
    if prefill_info.get("enable_hierarchical_cache") is not True:
        failures.append(
            "prefill engine reports enable_hierarchical_cache="
            f"{prefill_info.get('enable_hierarchical_cache')!r}, expected True"
        )
    io_backend = prefill_info.get("hicache_io_backend")
    if io_backend != HICACHE_IO_BACKEND:
        failures.append(
            f"prefill engine reports hicache_io_backend={io_backend!r}, "
            f"expected {HICACHE_IO_BACKEND!r}"
        )
    return failures


def raiden_engaged_runner(*, log_path: str):
    """Gate on the Raiden host tier having initialized on the prefill engine."""

    def run(_base_url: str) -> dict[str, Any]:
        engaged = RAIDEN_READY_MARKER in _read_log(log_path)
        print(
            f"[pd-block-major] raiden host tier ready={engaged} ({log_path})",
            flush=True,
        )
        failures = (
            []
            if engaged
            else [
                f"{RAIDEN_READY_MARKER!r} never appeared in {log_path}: the Raiden "
                "host tier did not initialize on the prefill engine"
            ]
        )
        return {
            "score": 1.0 if engaged else 0.0,
            "_passed": engaged,
            "_failures": failures,
        }

    return run


def flush_prefill_cache(prefill_url: str) -> None:
    """Empty the prefill's radix tree and host pool once the engine is idle.

    This goes to the prefill engine, not the router: the router's /flush_cache
    reports 200 even when an engine refused. The timeout makes the engine wait
    for idle; a flush without one skips that check and can free KV that a P/D
    transfer is still reading.
    """
    request = urllib.request.Request(
        f"{prefill_url.rstrip('/')}/flush_cache?timeout={FLUSH_TIMEOUT_S}",
        data=b"",
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=FLUSH_TIMEOUT_S + 60) as r:
            r.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"prefill /flush_cache failed with HTTP {exc.code}: {body}"
        ) from exc


def tokenize(prefill_url: str, text: str) -> list[int]:
    """Token ids of ``text`` without special tokens, from the prefill engine."""
    response = _post_json(
        f"{prefill_url.rstrip('/')}/tokenize",
        {"model": SERVED_MODEL_NAME, "prompt": text, "add_special_tokens": False},
        timeout=600,
    )
    return response["tokens"]


def eviction_plan(info: Mapping[str, Any], *, probe_tokens: int) -> EvictionPlan:
    """Size the churn from the prefill engine's /server_info.

    GPT-OSS is a hybrid SWA model, so the plan takes the SWA share: the host
    pool behind the smaller SWA device pool is the one the fillers must not
    overflow, or the reload reads host=0 and looks like a broken transfer.
    """
    return plan_eviction(
        device_pool_tokens=int(info["max_total_num_tokens"]),
        probe_tokens=probe_tokens,
        hicache_ratio=float(info["hicache_ratio"]),
        swa_full_tokens_ratio=float(info["swa_full_tokens_ratio"]),
    )


def cache_probe_runner(*, prefill_url: str, probe_tokens: int):
    """Score the device-hit, churn, host-reload round trip through the router."""

    def run(base_url: str) -> dict[str, Any]:
        flush_prefill_cache(prefill_url)
        info = _get_json(f"{prefill_url}/server_info")
        probe_ids = tokenize(
            prefill_url, probe_prompt(index=0, approx_tokens=probe_tokens)
        )
        plan = eviction_plan(info, probe_tokens=len(probe_ids))
        print(
            f"[pd-block-major] cache probe: {len(probe_ids)}-token probe, "
            f"{plan.num_fillers} fillers against a "
            f"{info['max_total_num_tokens']}-token full device pool and a "
            f"{plan.host_pool_tokens}-token SWA host pool",
            flush=True,
        )
        probe = run_probe(
            base_url=base_url,
            test_name=RUN_NAME,
            probe_tokens=probe_tokens,
            plan=plan,
            flush_cache=False,
        )
        failures = validate_probe(probe=probe, page_size=PAGE_SIZE, plan=plan)
        print(
            json.dumps(
                {
                    "cache_probe": "pd_block_major",
                    "probe_tokens": len(probe_ids),
                    "num_fillers": plan.num_fillers,
                    "host_pool_tokens": plan.host_pool_tokens,
                    "cold": msgspec.to_builtins(probe.cold),
                    "device_hit": msgspec.to_builtins(probe.warm),
                    "host_reload": msgspec.to_builtins(probe.reload),
                    "failures": failures,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return {
            "score": float(probe.reload.host_tokens),
            "_passed": not failures,
            "_failures": failures,
        }

    return run


def build_model_config() -> dict[str, Any]:
    """Assemble the gate list the prefill pod scores through the router."""
    prefill_url = f"http://{LOCAL_HOST}:{PREFILL_PORT}"
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": "gpt-oss-120b-pd-singlehost-block-major",
        "variant": "PD-1P1D-TP8+EP8+Raiden+block-major+offload-prefill",
        # Gates run in list order; the layout goes first because every later
        # gate would otherwise validate the layer-major pool under this name.
        "gates": [
            {
                "dataset": "block_major_layout",
                "label": "block-major-layout",
                "runner": block_major_layout_runner(log_path=PREFILL_LOG),
            },
            {
                "dataset": "raiden_host_tier",
                "label": "raiden-host-tier",
                "runner": raiden_engaged_runner(log_path=PREFILL_LOG),
            },
            {
                "dataset": "prefix_cache_kv_offload",
                "label": "prefix-cache-kv-offload",
                "runner": cache_probe_runner(
                    prefill_url=prefill_url, probe_tokens=PROBE_TOKENS
                ),
            },
            {
                "dataset": "gsm8k",
                "label": "gsm8k",
                "baseline": GSM8K_BASELINE,
                "atol": ATOL,
                "num_examples": NUM_EXAMPLES,
                "num_threads": NUM_THREADS,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
            },
        ],
    }


class TestPdSingleHostBlockMajorGptOss120bIntegration(PDSingleHostTestCase):
    """Configure the shared P/D fixture for this test's recipe and gates."""

    config_module = LAUNCH_CONFIG_MODULE
    repo_root = REPO
    cwd = BENCH
    log_path = PREFILL_LOG
    decode_log_path = DECODE_LOG
    self_hostname = staticmethod(self_hostname)
    decode_host_env = DECODE_HOST_ENV
    tag = "pd-singlehost-block-major"
    health_timeout = 2 * WAIT_SECS + 300
    health_generate_timeout = HEALTH_TIMEOUT

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return build_model_config()

    @classmethod
    def check_server_configuration(cls) -> None:
        failures = engine_configuration_failures(
            prefill_info=_get_json(f"http://{LOCAL_HOST}:{PREFILL_PORT}/server_info"),
            decode_info=_get_json(
                f"http://{cls.decode_hostname()}:{DECODE_PORT}/server_info"
            ),
        )
        if failures:
            raise RuntimeError("; ".join(failures))

    @classmethod
    def setUpClass(cls) -> None:
        validate_model_source(resolve_model_path())
        super().setUpClass()

    def run_evaluations(self) -> None:
        if self.role == DECODE_ROLE:
            # The decode log exists on this pod only; a layer-major decode pool
            # fails this replicated job, and the JobSet's failurePolicy ends
            # the run.
            wait_for_health(
                f"http://{LOCAL_HOST}:{DECODE_PORT}",
                self.health_timeout,
                process=self._launcher.process,
                endpoint="/health",
                tag=self.tag,
            )
            failures = check_layout(role=DECODE_ROLE, log_path=DECODE_LOG)
            self.assertEqual(failures, [], "; ".join(failures))
        super().run_evaluations()

    def test_accuracy(self) -> None:
        self.run_evaluations()


def render_jobset_yaml(
    image: str = "us-central1-docker.pkg.dev/sgl-project/tpu-ci/sglang:latest",
    lane: str = "nightly",
    weights_pvc: str = "model-weights",
    run_name: str = RUN_NAME,
    test_path: str = __file__,
) -> str:
    """Render one full prefill host plus one full decode host."""
    rel_path = os.path.relpath(test_path, REPO)
    cmd = [
        PYTHON,
        RENDER_JOBSET,
        "--pd-prefill-hosts",
        str(HOSTS_PER_ENGINE),
        "--pd-decode-hosts",
        str(HOSTS_PER_ENGINE),
        "--lane",
        lane,
        "--image",
        image,
        "--weights-pvc",
        weights_pvc,
        "--name",
        run_name,
        "--files",
        rel_path,
        "--exec",
        f"python3 {rel_path} integration",
    ]
    return subprocess.check_output(cmd, text=True)


class TestPdSingleHostBlockMajorGptOss120bConfiguration(unittest.TestCase):
    """Launch-contract and gate tests that do not initialize TPU hardware."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.plans = {}
        for role in (PREFILL_ROLE, DECODE_ROLE):
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "benchmark.tpu.launch_server",
                    "--config",
                    LAUNCH_CONFIG_MODULE + ":" + role.upper() + "_CONFIG",
                    "--runtime-envs",
                    json.dumps(
                        {"FORCE_SINGLE_HOST": "1", "SGLANG_HOST_IP": "127.0.0.1"}
                    ),
                    "--dry-run",
                ],
                cwd=REPO,
                text=True,
            )
            cls.plans[role] = json.loads(output)
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from benchmark.tpu.recipe.configs.gpt_oss_120b import pd_singlehost_block_major

        cls.recipe = pd_singlehost_block_major

    def test_both_engines_switch_on_the_block_major_layout(self) -> None:
        # A one-sided switch validates a layer-major pool under this test's name.
        for role, plan in self.plans.items():
            self.assertIn(BLOCK_MAJOR_ENV, plan["runtime_env_keys"], role)
            self.assertEqual(
                plan["server_args"]["attention_backend"], "pallas_tpu_rpa_v3"
            )
            self.assertEqual(plan["server_args"]["page_size"], PAGE_SIZE)
            self.assertEqual(plan["server_args"]["max_total_tokens"], MAX_TOTAL_TOKENS)
        for config in (self.recipe.PREFILL_CONFIG, self.recipe.DECODE_CONFIG):
            self.assertEqual(config["runtime_envs"][BLOCK_MAJOR_ENV], "1")

    def test_prefill_carries_the_raiden_host_tier_and_decode_does_not(self) -> None:
        prefill = self.plans[PREFILL_ROLE]["server_args"]
        self.assertTrue(prefill["enable_hierarchical_cache"])
        self.assertEqual(prefill["hicache_io_backend"], HICACHE_IO_BACKEND)
        self.assertEqual(prefill["hicache_mem_layout"], "page_first_fused")
        self.assertEqual(prefill["hicache_write_policy"], "write_through")
        self.assertTrue(prefill["enable_cache_report"])
        decode = self.plans[DECODE_ROLE]["server_args"]
        self.assertFalse(decode.get("enable_hierarchical_cache", False))

    def test_configuration_gate_requires_the_prefill_host_tier(self) -> None:
        prefill = {
            "disaggregation_mode": PREFILL_ROLE,
            "disaggregation_transfer_backend": DISAGG_TRANSFER_BACKEND,
            "enable_hierarchical_cache": True,
            "hicache_io_backend": HICACHE_IO_BACKEND,
        }
        decode = {
            "disaggregation_mode": DECODE_ROLE,
            "disaggregation_transfer_backend": DISAGG_TRANSFER_BACKEND,
        }
        self.assertEqual(
            engine_configuration_failures(prefill_info=prefill, decode_info=decode),
            [],
        )
        for broken in (
            {"enable_hierarchical_cache": False},
            {"hicache_io_backend": "kernel"},
            {"disaggregation_transfer_backend": "mooncake"},
        ):
            with self.subTest(broken=broken):
                self.assertTrue(
                    engine_configuration_failures(
                        prefill_info={**prefill, **broken}, decode_info=decode
                    )
                )

    def test_layout_gate_reads_pool_allocations(self) -> None:
        bundles = "\n".join(
            (
                "[TP0] Allocating block-major TPU KV bundle: shape=(6144, 36, 64, "
                "1, 4, 128), dtype=torch.float8_e4m3fn",
                "[TP0] Allocating block-major TPU KV bundle: shape=(6144, 36, 64, "
                "1, 4, 128), dtype=torch.float8_e4m3fn",
            )
        )
        fused = (
            "[TP0] Allocating fused TPU KV buffers: layers=36, shape per layer="
            "(6144, 64, 1, 4, 128), dtype=torch.float8_e4m3fn"
        )
        counts = count_pool_allocations(bundles)
        self.assertEqual((counts.bundles, counts.fused), (2, 0))
        self.assertEqual(layout_failures(role=PREFILL_ROLE, counts=counts), [])

        # One pool left layer-major beside the bundles must still fail.
        mixed = layout_failures(
            role=DECODE_ROLE, counts=count_pool_allocations(bundles + "\n" + fused)
        )
        self.assertEqual(len(mixed), 1)
        self.assertIn("layer-major", mixed[0])

        # A log with neither line is a server the switch never reached.
        none = layout_failures(
            role=PREFILL_ROLE, counts=count_pool_allocations("server ready")
        )
        self.assertEqual(len(none), 1)
        self.assertIn("no block-major bundle", none[0])

    def test_eviction_plan_takes_the_swa_host_pool(self) -> None:
        # max_total_tokens is the full pool; the SWA pool and its host pool are
        # swa_full_tokens_ratio times smaller, and a page is backed up to both.
        info = {
            "max_total_num_tokens": 220096,
            "swa_full_tokens_ratio": 0.8,
            "hicache_ratio": 3.0,
        }
        plan = eviction_plan(info, probe_tokens=2060)
        self.assertEqual(plan.host_pool_tokens, 528230)
        self.assertLessEqual(plan.backed_up_tokens, plan.host_pool_tokens)
        self.assertEqual(
            plan.num_fillers,
            plan_eviction(
                device_pool_tokens=220096, probe_tokens=2060, hicache_ratio=3.0
            ).num_fillers,
        )
        with self.assertRaisesRegex(ValueError, "evicted from host"):
            eviction_plan(dict(info, hicache_ratio=1.2), probe_tokens=2060)

    # Skip in runtime images, which do not contain the renderer.
    @unittest.skipUnless(
        os.path.exists(RENDER_JOBSET), "renderer is not shipped in the runtime image"
    )
    def test_dry_run_renders_one_host_per_role(self) -> None:
        yaml_content = render_jobset_yaml()
        self.assertIn(f"generateName: tpu-ci-{RUN_NAME}-", yaml_content)
        self.assertIn("name: prefill", yaml_content)
        self.assertIn("name: decode", yaml_content)
        self.assertNotIn("name: router", yaml_content)
        self.assertIn("CI_PD_ROLE", yaml_content)
        self.assertIn(
            "python3 test/registered/tpu/e2e/"
            "test_tpu_gpt_oss_120b_pd_singlehost_block_major.py integration",
            yaml_content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-host-per-engine GPT-OSS P/D block-major KV layout gate"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="unittest",
        choices=["unittest", "integration", "integration_dry_run"],
    )
    args, extra_args = parser.parse_known_args()

    if args.mode == "integration_dry_run":
        print(render_jobset_yaml())
        sys.exit(0)
    test_case = (
        TestPdSingleHostBlockMajorGptOss120bIntegration
        if args.mode == "integration"
        else TestPdSingleHostBlockMajorGptOss120bConfiguration
    )
    unittest.main(
        defaultTest=test_case.__name__,
        argv=[sys.argv[0], *extra_args],
        verbosity=2,
    )
