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
import time
import unittest
import urllib.request
from collections.abc import Mapping
from typing import Any

import msgspec

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.evals.eval_support import wait_for_health
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
# The recipe's max_total_tokens on each engine; a hybrid SWA engine splits it
# between its full and SWA device pools.
MAX_TOTAL_TOKENS = 393216
DISAGG_TRANSFER_BACKEND = "raiden"
# Raiden handles P/D KV transfer, the prefill host tier and decode host-pool I/O.
HICACHE_IO_BACKEND = "raiden"

BLOCK_MAJOR_ENV = "SGLANG_TPU_USE_BLOCK_MAJOR_KV"
# Logged by sglang_tpu/kv_pool.py once per pool per rank.
BUNDLE_LINE = "Allocating block-major TPU KV bundle"
FUSED_LINE = "Allocating fused TPU KV buffers"
# Logged by sglang_tpu/raiden_transfer.py when the host tier is up.
RAIDEN_READY_MARKER = "Raiden KV transfer ready"

GENERATE_TIMEOUT = 900

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


class CacheProbeConfig(msgspec.Struct, frozen=True, kw_only=True):
    page_size: int
    # None reads the full device pool from the prefill engine's /server_info.
    device_pool_tokens: int | None
    target_words: int
    filler_words: int
    # None sizes the churn from the first filler's measured token count.
    num_fillers: int | None
    max_new_tokens: int

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "CacheProbeConfig":
        values = os.environ if env is None else env
        fillers_raw = values.get("CACHE_PROBE_NUM_FILLERS")
        device_raw = values.get("CACHE_PROBE_DEVICE_TOKENS")
        return cls(
            page_size=PAGE_SIZE,
            device_pool_tokens=None if device_raw is None else int(device_raw),
            target_words=int(values.get("CACHE_PROBE_TARGET_WORDS", "2048")),
            filler_words=int(values.get("CACHE_PROBE_FILLER_WORDS", "2048")),
            num_fillers=None if fillers_raw is None else int(fillers_raw),
            # Room for all four copied item lines plus any preamble.
            max_new_tokens=int(values.get("CACHE_PROBE_MAX_NEW_TOKENS", "64")),
        )


class ProbePools(msgspec.Struct, frozen=True, kw_only=True):
    device_tokens: int
    host_tokens: int


def probe_pools(
    info: Mapping[str, Any], *, device_override: int | None = None
) -> ProbePools:
    """Read the prefill engine's full device pool and its smaller host pool.

    HiCache sizes each host pool as hicache_ratio times its device pool, and the
    SWA device pool is swa_full_tokens_ratio times the full one.
    """
    device = device_override or int(info["max_total_num_tokens"])
    swa_share = min(1.0, float(info["swa_full_tokens_ratio"]))
    host = int(device * float(info["hicache_ratio"]) * swa_share)
    return ProbePools(device_tokens=device, host_tokens=host)


def host_overflow_failure(
    *, num_fillers: int, filler_tokens: int, target_tokens: int, host_pool_tokens: int
) -> str | None:
    """Refuse a churn that would push the target out of the host tier as well.

    Write-through backs up the target and every filler, and the host tier evicts
    LRU, so past its capacity the reload becomes a recompute that reports host=0.
    """
    backed_up = num_fillers * filler_tokens + target_tokens
    if backed_up <= host_pool_tokens:
        return None
    return (
        f"churn backs up {backed_up} tokens but the host tier holds "
        f"{host_pool_tokens}, so the target would be evicted from host too; "
        "raise HICACHE_RATIO or set CACHE_PROBE_NUM_FILLERS lower"
    )


def churn_fillers_needed(
    *, measured_filler_tokens: int, device_pool_tokens: int
) -> int:
    """Fillers that push 2x the device pool through the cache.

    Sized from a measured filler's token count; a churn that runs short of the
    pool never evicts, and the host-reload leg then tests nothing.
    """
    if measured_filler_tokens <= 0:
        raise AssertionError(
            f"measured filler prompt_tokens={measured_filler_tokens}; cannot "
            "size the churn"
        )
    return (2 * device_pool_tokens) // measured_filler_tokens + 1


_VOCAB = (
    "the of and to in a is that for it as was with on be by at this from but not "
    "are or his they she had which were we when your can said there use each "
    "their time will how other about many then them these so some her would make "
    "like him into has two more very after our just name good through where help "
    "much before line right too mean old"
).split()


def make_prompt(index: int, nwords: int) -> str:
    seed = (index + 1) * 2654435761 & 0xFFFFFFFF
    words = []
    for _ in range(nwords):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        words.append(_VOCAB[seed % len(_VOCAB)])
    return f"Document {index} header alpha{index} beta{index * 7}. " + " ".join(words)


_ITEM_NOUNS = (
    "lantern kettle walnut anchor violet magnet saddle donkey harbor pencil "
    "quartz turnip magpie helmet donkey copper barrel falcon garnet hammer "
    "island jasper kitten locket mirror needle onward parrot quiver ribbon "
    "sombra tundra"
).split()


def make_target_prompt(nwords: int) -> tuple[str, tuple[int, ...]]:
    """A numbered item list ending in a verbatim copy instruction.

    The greedy continuation is forced: the model repeats four distinctive lines
    read through full attention from the queried positions' KV, so a text
    difference between cached passes means the KV values themselves changed.
    """
    num_items = max(40, nwords // 6)
    lines = []
    for i in range(1, num_items + 1):
        noun = _ITEM_NOUNS[(i * 2654435761 & 0xFFFFFFFF) % len(_ITEM_NOUNS)]
        lines.append(f"item {i}: {noun}-{i * 7 % 97}")
    queried = (
        num_items // 8,
        num_items // 3,
        (2 * num_items) // 3,
        num_items - 5,
    )
    ask = ", ".join(str(q) for q in queried)
    prompt = (
        "Below is a list of items.\n"
        + "\n".join(lines)
        + f"\nOutput items {ask} from the list above exactly as written, "
        "one per line, and nothing else.\nAnswer:\nitem "
    )
    return prompt, queried


class GenerationRecord(msgspec.Struct, frozen=True, kw_only=True):
    text: str
    prompt_tokens: int
    completion_tokens: int
    device_cached_tokens: int
    host_cached_tokens: int
    latency_seconds: float


def _generate(base_url: str, prompt: str, max_new_tokens: int) -> GenerationRecord:
    started = time.perf_counter()
    payload = _post_json(
        f"{base_url}/generate",
        {
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
        },
        timeout=GENERATE_TIMEOUT,
    )
    meta = payload.get("meta_info") or {}
    details = meta.get("cached_tokens_details") or {}
    return GenerationRecord(
        text=payload["text"],
        prompt_tokens=int(meta.get("prompt_tokens") or 0),
        completion_tokens=int(meta.get("completion_tokens") or 0),
        device_cached_tokens=int(details.get("device") or 0),
        host_cached_tokens=int(details.get("host") or 0),
        latency_seconds=round(time.perf_counter() - started, 3),
    )


def _flush_cache(url: str) -> None:
    with urllib.request.urlopen(f"{url}/flush_cache", timeout=60) as r:
        r.read()


# Restored KV is judged on the first characters only (see RECIPES.md).
DIVERGENCE_CHARS = 8


def _diverges(a: str, b: str) -> bool:
    return a[:DIVERGENCE_CHARS] != b[:DIVERGENCE_CHARS]


def validate_cache_probe(
    *,
    cold: GenerationRecord,
    device_hit: GenerationRecord,
    host_reload: GenerationRecord,
    post_reload_hit: GenerationRecord,
    page_size: int,
) -> list[str]:
    """The device-hit, churn, host-reload oracle.

    The three cached passes run the same extend shape over the same stored
    prefix KV, so their greedy text must agree over the first DIVERGENCE_CHARS
    characters; the TPU stack is not run-to-run deterministic past that, so
    full equality is printed, not gated. The cold pass runs a different extend
    shape and is held to cache-state and liveness only.
    """
    if cold.prompt_tokens < 2 * page_size:
        return [f"target prompt too short: {cold.prompt_tokens} tokens"]
    if cold.device_cached_tokens or cold.host_cached_tokens:
        return [
            "cold request unexpectedly hit cache: "
            f"device={cold.device_cached_tokens} host={cold.host_cached_tokens}"
        ]
    if not cold.text.strip() or cold.completion_tokens <= 0:
        return ["cold request produced no decode output: P/D KV transfer broken"]

    failures = []
    expected = max(page_size, (cold.prompt_tokens // page_size - 1) * page_size)
    if device_hit.device_cached_tokens < expected:
        failures.append(
            "immediate resend missed the device prefix cache: "
            f"device={device_hit.device_cached_tokens} expected>={expected}"
        )
    if device_hit.host_cached_tokens:
        failures.append(
            f"immediate resend loaded from host: {device_hit.host_cached_tokens}"
        )
    if not device_hit.text.strip() or device_hit.completion_tokens <= 0:
        failures.append("device-hit request produced no decode output")

    if host_reload.host_cached_tokens <= 0:
        failures.append(
            "target was not reloaded from the host tier after churn: "
            "KV offload never engaged"
        )
    if host_reload.device_cached_tokens + host_reload.host_cached_tokens < expected:
        failures.append(
            "host reload did not recover the cached prefix: "
            f"device={host_reload.device_cached_tokens} "
            f"host={host_reload.host_cached_tokens} expected>={expected}"
        )
    # A reload that raced its own host->device DMA reads clean once the pages
    # settle; corrupt values stay wrong on the post-reload pass.
    if _diverges(host_reload.text, device_hit.text):
        if not _diverges(post_reload_hit.text, device_hit.text):
            failures.append(
                "host-reloaded copy output diverged from the device-hit pass but "
                "the post-reload resend matches: the reload raced its own "
                "host->device transfer"
            )
        else:
            failures.append(
                "host-reloaded copy output diverged from the device-hit pass: "
                "the offloaded KV pages did not survive the host round trip"
            )
    elif _diverges(post_reload_hit.text, device_hit.text):
        failures.append(
            "post-reload resend diverged from the device-hit pass: the reloaded "
            "device-resident pages hold different KV values"
        )
    return failures


def run_cache_probe(
    *, base_url: str, flush_url: str, config: CacheProbeConfig
) -> dict[str, Any]:
    """Run the four probe passes and the churn between them through the router."""
    _flush_cache(flush_url)
    pools = probe_pools(
        _get_json(f"{flush_url}/server_info"),
        device_override=config.device_pool_tokens,
    )
    target, queried = make_target_prompt(config.target_words)
    print(f"[pd-block-major] cache probe target queries items {queried}", flush=True)
    cold = _generate(base_url, target, config.max_new_tokens)
    device_hit = _generate(base_url, target, config.max_new_tokens)

    first_filler = _generate(base_url, make_prompt(1000, config.filler_words), 1)
    num_fillers = config.num_fillers
    if num_fillers is None:
        num_fillers = churn_fillers_needed(
            measured_filler_tokens=first_filler.prompt_tokens,
            device_pool_tokens=pools.device_tokens,
        )
    print(
        f"[pd-block-major] churn: {num_fillers} fillers x "
        f"{first_filler.prompt_tokens} tokens against a "
        f"{pools.device_tokens}-token device pool and a "
        f"{pools.host_tokens}-token host pool",
        flush=True,
    )
    overflow = host_overflow_failure(
        num_fillers=num_fillers,
        filler_tokens=first_filler.prompt_tokens,
        target_tokens=cold.prompt_tokens,
        host_pool_tokens=pools.host_tokens,
    )
    if overflow:
        return {"score": 0.0, "_passed": False, "_failures": [overflow]}
    for index in range(1, num_fillers):
        _generate(base_url, make_prompt(1000 + index, config.filler_words), 1)
    host_reload = _generate(base_url, target, config.max_new_tokens)
    post_reload_hit = _generate(base_url, target, config.max_new_tokens)

    # Printed before validation so a failed oracle still carries every pass.
    print(
        json.dumps(
            {
                "cache_probe": "pd_block_major",
                "cold": msgspec.to_builtins(cold),
                "device_hit": msgspec.to_builtins(device_hit),
                "host_reload": msgspec.to_builtins(host_reload),
                "post_reload_hit": msgspec.to_builtins(post_reload_hit),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    failures = validate_cache_probe(
        cold=cold,
        device_hit=device_hit,
        host_reload=host_reload,
        post_reload_hit=post_reload_hit,
        page_size=config.page_size,
    )
    print(
        f"[pd-block-major] cache probe device_hit={device_hit.device_cached_tokens} "
        f"host_reload={host_reload.host_cached_tokens} "
        f"identical_text={host_reload.text == device_hit.text == post_reload_hit.text} "
        f"failures={len(failures)}",
        flush=True,
    )
    return {
        "score": float(host_reload.host_cached_tokens),
        "_passed": not failures,
        "_failures": failures,
    }


def cache_probe_runner(*, flush_url: str, config: CacheProbeConfig):
    """Score the device-hit, churn, host-reload round trip through the router."""

    def run(base_url: str) -> dict[str, Any]:
        return run_cache_probe(base_url=base_url, flush_url=flush_url, config=config)

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
                    flush_url=prefill_url, config=CacheProbeConfig.from_env()
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

    def test_cache_probe_oracle(self) -> None:
        def record(**overrides) -> GenerationRecord:
            fields = {
                "text": "1: lantern-7\nitem 2",
                "prompt_tokens": 2048,
                "completion_tokens": 8,
                "device_cached_tokens": 0,
                "host_cached_tokens": 0,
                "latency_seconds": 1.0,
            }
            return GenerationRecord(**{**fields, **overrides})

        cold = record()
        hit = record(device_cached_tokens=1984)
        reload = record(host_cached_tokens=1984)
        self.assertEqual(
            validate_cache_probe(
                cold=cold,
                device_hit=hit,
                host_reload=reload,
                post_reload_hit=hit,
                page_size=PAGE_SIZE,
            ),
            [],
        )
        for name, kwargs in (
            ("never engaged", {"host_reload": record(device_cached_tokens=1984)}),
            ("missed the device", {"device_hit": record()}),
            (
                "raced its own",
                {"host_reload": record(host_cached_tokens=1984, text="1: kettle")},
            ),
            (
                "did not survive",
                {
                    "host_reload": record(host_cached_tokens=1984, text="1: kettle"),
                    "post_reload_hit": record(
                        device_cached_tokens=1984, text="1: kettle"
                    ),
                },
            ),
        ):
            with self.subTest(name=name):
                failures = validate_cache_probe(
                    **{
                        "cold": cold,
                        "device_hit": hit,
                        "host_reload": reload,
                        "post_reload_hit": hit,
                        **kwargs,
                    },
                    page_size=PAGE_SIZE,
                )
                self.assertTrue(any(name in f for f in failures), failures)
        self.assertEqual(
            churn_fillers_needed(measured_filler_tokens=2048, device_pool_tokens=4096),
            5,
        )

    def test_churn_stays_inside_the_host_tier(self) -> None:
        # max_total_tokens covers the full and SWA pools together; a churn sized
        # from it overflows the host tier, and the reload reads host=0.
        info = {
            "max_total_num_tokens": 220096,
            "swa_full_tokens_ratio": 0.8,
            "hicache_ratio": 3.0,
        }
        pools = probe_pools(info)
        self.assertEqual(pools.device_tokens, 220096)
        self.assertEqual(pools.host_tokens, 528230)

        def overflow(device_pool_tokens: int) -> str | None:
            return host_overflow_failure(
                num_fillers=churn_fillers_needed(
                    measured_filler_tokens=2060,
                    device_pool_tokens=device_pool_tokens,
                ),
                filler_tokens=2060,
                target_tokens=2865,
                host_pool_tokens=pools.host_tokens,
            )

        self.assertIsNone(overflow(pools.device_tokens))
        self.assertIn("evicted from host", overflow(MAX_TOTAL_TOKENS) or "")
        self.assertEqual(probe_pools(info, device_override=4096).device_tokens, 4096)

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
