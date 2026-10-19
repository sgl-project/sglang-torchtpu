"""GPT-OSS-120B P/D MXINT8 KV multi-turn long-context test (issue #1066).

Topology: one TP8/EP8 host per engine (1P+1D), transferring KV through Raiden.
Both engines store target KV as MXINT8 (draft KV stays bf16) and run EAGLE3
speculative decoding, because the P/D handoff carries the spec bootstrap state.
Prefix caching and hierarchical KV offload (HiCache, raiden io) run only on the
prefill engine. The shared PDSingleHostTestCase fixture launches the recipe
(benchmark/tpu/recipe/configs/gpt_oss_120b/pd_singlehost_mxint8_multiturn.py):
the decode pod serves, and the prefill pod starts its engine and the mini-LB,
checks both engines' resolved configuration and runs three gates:

1. GSM8K with a speculative accept-length floor. It runs first because the
   decode's reported accept length is cumulative since boot.
2. A ~126k-token KV offload round trip: a device hit, eviction to host, and a
   host reload that must decode exactly like the device hit.
3. Multi-turn long context: sessions grow by 8192 tokens per turn to ~123k
   tokens, served round-robin so that their prefixes overflow HBM into host
   memory. Every returning turn must hit the previous turn's whole prompt. The
   phase also gates an aggregate hit rate, accepted drafts, zero P/D transfer,
   bootstrap, retry and retraction errors, and each session recalling a secret
   planted in its first turn. TTFT and P->D transfer time are recorded per turn.

MXINT8 needs a backend extension: in-tree,
KVCacheConfigurator._validate_mxint8 rejects it, so with the default dtype both
engines fail at startup until the TPU image ships that extension. Setting
PD_KV_CACHE_DTYPE=bfloat16 runs the same harness on bf16 KV, which validates
everything except the MXINT8 numerics; the external nightly does that until the
extension ships.
"""

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

SGLANG_IMPORT_ERROR = None
try:
    from sglang.test.ci.ci_register import register_tpu_ci
    from sglang.test.test_utils import dump_metric
    from sglang.test.tpu.utils.kv_offload_probe import (
        mean_accept_length,
        plan_eviction,
        probe_prompt,
        run_probe,
        spec_failures,
        synthetic_prompt,
        validate_probe,
    )
    from sglang.test.tpu.utils.pd_singlehost_base import (
        PDSingleHostTestCase,
        peer_hostname,
    )
except ModuleNotFoundError as exc:
    SGLANG_IMPORT_ERROR = exc

    def register_tpu_ci(*_args, **_kwargs):
        return None

    dump_metric = None
    mean_accept_length = plan_eviction = probe_prompt = run_probe = None
    spec_failures = synthetic_prompt = validate_probe = peer_hostname = None
    PDSingleHostTestCase = unittest.TestCase


# The external nightly runs this with PD_KV_CACHE_DTYPE=bfloat16
# (.github/tpu-ci/external/external-topology.yaml) until the TPU image ships the
# MXINT8 backend extension; then drop that override.
register_tpu_ci(
    est_time=5400,
    suite="tpu-integration",
    nightly=True,
    labels=[
        "singlehost",
        "gpt-oss-120b",
        "pd",
        "spec",
        "prefix-cache",
        "kv-offload",
        "mxint8",
        "long-context",
        "multi-turn",
    ],
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")
LAUNCH_CONFIG_MODULE = "recipe/configs/gpt_oss_120b/pd_singlehost_mxint8_multiturn.py"

E = os.environ.get

PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
WAIT_SECS = int(E("WAIT_SECS", "5400"))
PYTHON = E("PYTHON", "python3")
RENDER_JOBSET = os.path.join(REPO, ".github", "tpu-ci", "render_jobset.py")

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
SELF_HOST_ENV = "PD_SELF_HOST"
DECODE_HOST_ENV = "PD_DECODE_HOST"
LOCAL_HOST = "127.0.0.1"

RUN_NAME = "pd-sh-gpt-120b-mxint8-mt"
PREFILL_LOG = "/tmp/pd_mxint8_multiturn_prefill.log"
DECODE_LOG = "/tmp/pd_mxint8_multiturn_decode.log"

TEST_NAME = "gpt-oss-120b-pd-singlehost-mxint8-multiturn"
SERVED_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"
HF_REPO_ID = "openai/gpt-oss-120b"
MODEL_ENV = "GPT_OSS_MODEL"
DRAFT_MODEL_ENV = "GPT_OSS_DRAFT_MODEL"
DEFAULT_DRAFT_MODEL_PATH = "lmsys/EAGLE3-gpt-oss-120b-bf16"

# One tpu7x host per engine: 4 chips x 2 TensorCores.
TP_SIZE = 8
EP_SIZE = 8
# The values below are set in the launch recipe; the test reads them back from
# each engine's /server_info and sizes its probes against them.
# MXINT8 KV needs a power-of-two page of at least 128 tokens, and P/D needs the
# same page size (and KV dtype) on both engines.
PAGE_SIZE = 128
# GPT-OSS-120B's native context window. Not env-tunable: the offload probe and
# the multi-turn plan are sized against it.
CONTEXT_LEN = 131072
KV_CACHE_DTYPE_ENV = "PD_KV_CACHE_DTYPE"
# mxint8 is the configuration under test (the default). bfloat16 runs the same
# harness on an image without the MXINT8 backend extension.
KV_CACHE_DTYPES = ("mxint8", "bfloat16")
# The draft KV stays bf16 on both engines. Left unset it follows the target
# dtype onto MXINT8, and the in-tree EAGLE draft attention handles only bf16
# and fp8 KV.
DRAFT_KV_CACHE_DTYPE = "bfloat16"
# About two full contexts of HBM KV per engine. The decode holds one ~123k
# request at a time. On the prefill, three ~123k sessions outgrow this pool, so
# their prefixes spill to host memory and must come back from it.
#
# Deliberately below 262144 tokens: under that, sglang serializes VMEM
# prefetches (env_override.apply_small_kv_pool_vmem_prefetch_override), the
# guard against issue #363's startup precompile deadlock on GPT-OSS-120B. Its
# token threshold was measured on bf16 KV, but the deadlock tracks the pool's
# size in bytes: GPT-OSS-120B on fp8 KV hung below ~605k tokens
# (pd_reshard_harness.py), and an MXINT8 pool is about as small per token. At
# 262144 tokens or more an MXINT8 pool would get no guard.
MAX_TOTAL_TOKENS = 245760
# A host pool of 3x the device pool (~737k tokens) holds every session's
# prefix (~369k tokens), and the offload probe plus its fillers (~504k tokens).
HICACHE_RATIO = 3
HICACHE_WRITE_POLICY = "write_through"
DISAGG_TRANSFER_BACKEND = "raiden"
RETRACTION_BACKUP = "cpu_tensor"

# The single-host GPT-OSS EAGLE3 gate measured about 0.785 over 200 examples.
# A 0.75 baseline with the default 0.03 tolerance enforces a 0.72 score floor.
# Both floors were calibrated on bf16 KV.
GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B_SPEC", "0.75"))
ATOL = float(E("GSM8K_ATOL", "0.03"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "200"))
# Match client concurrency to the recipe's max_running_requests and the largest
# compiled batch.
NUM_THREADS = int(E("GSM8K_NUM_THREADS", "16"))
MAX_TOKENS = int(E("GSM8K_MAX_TOKENS", "1024"))
SPEC_ACCEPT_LENGTH_THRESHOLD = float(E("SPEC_ACCEPT_LENGTH_THRESHOLD", "1.90"))
HEALTH_TIMEOUT = float(E("HEALTH_TIMEOUT", "5400"))

# Defaults of the env-tunable probe and multi-turn knobs. The CPU tests check
# these, not the knobs, so an operator's overrides cannot fail them.
DEFAULT_OFFLOAD_PROBE_TOKENS = 126000
DEFAULT_MT_SESSIONS = 3
DEFAULT_MT_TURNS = 15
DEFAULT_MT_MIN_HIT_RATE = 0.85
DEFAULT_MT_MIN_ACCEPT_LENGTH = 1.0

# The offload probe's size in words (~1 token each). ~126k is the largest round
# size whose probe and fillers still fit the context with decode headroom.
OFFLOAD_PROBE_TOKENS = int(E("OFFLOAD_PROBE_TOKENS", str(DEFAULT_OFFLOAD_PROBE_TOKENS)))
# Covers the probe's 16 decoded tokens plus EAGLE's reserved draft slots.
DECODE_HEADROOM_TOKENS = 64

# Each turn appends the previous reply and a new user block, so a session's
# prompt grows by exactly MT_TURN_TOKENS per turn (to 122816 at turn 15).
MT_TURN_TOKENS = 8192
MT_OUTPUT_TOKENS = 64
# The default plan's ideal hit rate is 0.874 (every turn hits the previous
# turn's page-aligned prompt).
MT_MIN_HIT_RATE = float(E("MT_MIN_HIT_RATE", str(DEFAULT_MT_MIN_HIT_RATE)))
# Tokens per verify forward. Exactly 1.0 means no draft was ever accepted.
MT_MIN_ACCEPT_LENGTH = float(
    E("MT_MIN_ACCEPT_LENGTH", str(DEFAULT_MT_MIN_ACCEPT_LENGTH))
)

FLUSH_TIMEOUT_S = 300
REQUEST_TIMEOUT_S = 1800

MXINT8_UNSUPPORTED_MARKER = "MXINT8 KV cache is unsupported"


def resolve_kv_cache_dtype(env: Optional[Mapping[str, str]] = None) -> str:
    """The target KV dtype for both engines: mxint8 unless overridden."""
    values = os.environ if env is None else env
    dtype = values.get(KV_CACHE_DTYPE_ENV) or KV_CACHE_DTYPES[0]
    if dtype not in KV_CACHE_DTYPES:
        raise ValueError(
            f"{KV_CACHE_DTYPE_ENV} must be one of {KV_CACHE_DTYPES}, got {dtype!r}"
        )
    return dtype


def resolve_model_path(env: Optional[Mapping[str, str]] = None) -> str:
    values = os.environ if env is None else env
    return values.get(MODEL_ENV) or values.get("MODEL") or DEFAULT_MODEL_PATH


def resolve_draft_model_path(env: Optional[Mapping[str, str]] = None) -> str:
    values = os.environ if env is None else env
    return (
        values.get(DRAFT_MODEL_ENV)
        or values.get("DRAFT_MODEL")
        or DEFAULT_DRAFT_MODEL_PATH
    )


def validate_model_source(model_path: str, *, label: str) -> None:
    """Fail clearly for a missing mounted checkpoint; HF ids resolve remotely."""
    if not os.path.isabs(model_path):
        return
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"{label} checkpoint incomplete or missing at {model_path}; provision "
            f"it or set {MODEL_ENV if label == 'target' else DRAFT_MODEL_ENV}"
        )


def self_hostname(env: Optional[Mapping[str, str]] = None) -> str:
    values = os.environ if env is None else env
    if values.get(SELF_HOST_ENV):
        return values[SELF_HOST_ENV]
    hostnames = [h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def startup_failure_hint(log_path: str, kv_cache_dtype: str) -> Optional[str]:
    """Explain the expected startup failure on an image without MXINT8 support."""
    if kv_cache_dtype != "mxint8":
        return None
    try:
        with open(log_path, "rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - (1 << 20)))
            tail = log.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    if MXINT8_UNSUPPORTED_MARKER not in tail:
        return None
    return (
        "this image has no MXINT8 KV backend extension, so the engines cannot "
        f"start with mxint8 KV. Set {KV_CACHE_DTYPE_ENV}=bfloat16 to validate the "
        "harness on bf16 KV."
    )


# --- HTTP helpers for the gates ------------------------------------------------


def _http_json(url: str, payload: Optional[Dict] = None, timeout: float = 60) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _http_text(url: str, timeout: float = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def flush_prefill_cache(prefill_url: str) -> None:
    """Empty the prefill's radix tree and host pool once the engine is idle.

    This goes to the prefill engine, not the mini-LB: the LB's /flush_cache
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


def tokenize(prefill_url: str, texts: Sequence[str]) -> List[List[int]]:
    """Token ids for each text, without special tokens, from the prefill."""
    payload = {
        "model": SERVED_MODEL_NAME,
        "prompt": list(texts),
        "add_special_tokens": False,
    }
    response = _http_json(f"{prefill_url.rstrip('/')}/tokenize", payload, timeout=600)
    return response["tokens"]


SSE_DONE = "[DONE]"


def parse_sse_line(raw: Any) -> Any:
    """Decode one line of a /generate event stream.

    Returns None for blank, comment and non-data lines, SSE_DONE for the
    terminator, and the decoded JSON chunk otherwise (an error chunk is a dict
    with an "error" key).
    """
    line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if data == SSE_DONE:
        return SSE_DONE
    return json.loads(data)


def stream_generate(
    base_url: str, input_ids: Sequence[int], max_new_tokens: int
) -> Dict[str, Any]:
    """Greedy streaming /generate; returns the final chunk and client timings.

    TTFT is the client-side time to the first chunk that carries output tokens,
    which covers the prefill, the P->D transfer and the decode's first step. A
    non-streaming response's first_token_latency cannot stand in for it, since
    non-streaming outputs are sent in batches of tokens. Streamed chunks carry
    the cumulative output, so the last one is the whole response.
    """
    payload = {
        "input_ids": list(input_ids),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            # Every turn decodes exactly max_new_tokens, so the prompt lengths
            # and cached-token expectations of later turns are fixed.
            "ignore_eos": True,
        },
        "stream": True,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    ttft = None
    final = None
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        for raw in response:
            chunk = parse_sse_line(raw)
            if chunk is None:
                continue
            if chunk == SSE_DONE:
                break
            if "error" in chunk:
                raise RuntimeError(f"/generate stream error: {chunk['error']}")
            if ttft is None and chunk.get("output_ids"):
                ttft = time.perf_counter() - start
            final = chunk
    e2e = time.perf_counter() - start
    if final is None:
        raise RuntimeError("/generate stream ended without a data chunk")
    return {"final": final, "ttft_s": ttft, "e2e_s": e2e}


# --- Prometheus /metrics -------------------------------------------------------

_SAMPLE_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)")
_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
_TRANSFER_STAGE = {"stage": "prefill_transfer_kv_cache"}
FAILURE_COUNTERS = (
    "sglang:num_transfer_failed_reqs_total",
    "sglang:num_bootstrap_failed_reqs_total",
    "sglang:num_prefill_retries_total",
    "sglang:num_retracted_requests_total",
)


def sum_samples(
    metrics_text: str, sample_name: str, labels: Optional[Mapping[str, str]] = None
) -> float:
    """Sum one Prometheus sample over every series that carries ``labels``.

    Summing across series folds the per-rank copies together. A sample the
    server never emitted (a counter that never fired) sums to 0.
    """
    want = labels or {}
    total = 0.0
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if not match or match.group(1) != sample_name:
            continue
        series = dict(_LABEL_RE.findall(match.group(2) or ""))
        if all(series.get(key) == value for key, value in want.items()):
            total += float(match.group(3))
    return total


def _histogram_delta(
    before: str, after: str, name: str, labels: Optional[Mapping[str, str]] = None
) -> Tuple[float, float]:
    """(count, sum) a histogram gained between two scrapes."""
    count = sum_samples(after, f"{name}_count", labels) - sum_samples(
        before, f"{name}_count", labels
    )
    total = sum_samples(after, f"{name}_sum", labels) - sum_samples(
        before, f"{name}_sum", labels
    )
    return count, total


def transfer_delta(before: str, after: str) -> Dict[str, Any]:
    """P->D transfer timing the prefill recorded between two /metrics scrapes.

    ``stage_s`` is the scheduler's prefill_transfer_kv_cache stage (on TPU it
    ends at raiden's transfer finish); ``latency_ms`` and ``mb`` come from the
    transfer backend's own histograms. Means are None when nothing was observed.
    """
    stage_count, stage_sum = _histogram_delta(
        before, after, "sglang:per_stage_req_latency_seconds", _TRANSFER_STAGE
    )
    latency_count, latency_sum = _histogram_delta(
        before, after, "sglang:kv_transfer_latency_ms"
    )
    mb_count, mb_sum = _histogram_delta(before, after, "sglang:kv_transfer_total_mb")
    return {
        "stage_s": stage_sum / stage_count if stage_count > 0 else None,
        "latency_ms": latency_sum / latency_count if latency_count > 0 else None,
        "mb": mb_sum if mb_count > 0 else None,
        # Transfers seen by either histogram; the backend may not emit its own.
        "observations": max(stage_count, latency_count),
    }


def counter_deltas(
    before: Mapping[str, str], after: Mapping[str, str]
) -> Dict[str, float]:
    """Increase of every P/D failure counter per engine ({role: metrics text})."""
    return {
        f"{role} {name}": sum_samples(after[role], name)
        - sum_samples(before[role], name)
        for role in sorted(before)
        for name in FAILURE_COUNTERS
    }


# --- Engine configuration check -------------------------------------------------


def _normalize_dtype(value: Any) -> str:
    text = str(value or "").lower()
    return "bfloat16" if text == "bf16" else text


def engine_configuration_failures(
    prefill_info: Mapping[str, Any],
    decode_info: Mapping[str, Any],
    kv_cache_dtype: str,
) -> List[str]:
    """Compare both engines' resolved /server_info with what the recipe sets.

    A silently ignored or overridden flag would leave a gate passing on a
    configuration it was not written for, so check the settings it relies on.
    """
    failures: List[str] = []
    shared = {
        "disaggregation_transfer_backend": DISAGG_TRANSFER_BACKEND,
        "tp_size": TP_SIZE,
        "ep_size": EP_SIZE,
        "page_size": PAGE_SIZE,
        "context_length": CONTEXT_LEN,
        "enable_metrics": True,
    }
    for role, info in ((PREFILL_ROLE, prefill_info), (DECODE_ROLE, decode_info)):
        for key, want in {**shared, "disaggregation_mode": role}.items():
            if info.get(key) != want:
                failures.append(f"{role}: {key}={info.get(key)!r}, expected {want!r}")
        kv_dtype = _normalize_dtype(info.get("kv_cache_dtype"))
        if kv_dtype != kv_cache_dtype:
            failures.append(
                f"{role}: kv_cache_dtype={kv_dtype!r}, expected {kv_cache_dtype!r}"
            )
        draft_dtype = _normalize_dtype(info.get("speculative_draft_kv_cache_dtype"))
        if draft_dtype != DRAFT_KV_CACHE_DTYPE:
            failures.append(
                f"{role}: speculative_draft_kv_cache_dtype={draft_dtype!r}, "
                f"expected {DRAFT_KV_CACHE_DTYPE!r}"
            )
        algorithm = str(info.get("speculative_algorithm") or "")
        if algorithm.upper() != "EAGLE3":
            failures.append(
                f"{role}: speculative_algorithm={algorithm!r}, expected EAGLE3"
            )
        pool = info.get("max_total_num_tokens")
        if not isinstance(pool, int) or pool < CONTEXT_LEN:
            failures.append(
                f"{role}: max_total_num_tokens={pool!r} cannot hold one "
                f"{CONTEXT_LEN}-token request"
            )

    prefill_only = {
        "enable_hierarchical_cache": True,
        "hicache_io_backend": "raiden",
        "hicache_mem_layout": "page_first_fused",
        "hicache_write_policy": HICACHE_WRITE_POLICY,
        "disable_radix_cache": False,
    }
    for key, want in prefill_only.items():
        if prefill_info.get(key) != want:
            failures.append(
                f"prefill: {key}={prefill_info.get(key)!r}, expected {want!r}"
            )
    # The decode keeps HiCache off, and its own radix cache off, which EAGLE3
    # does not support.
    for key in (
        "enable_hierarchical_cache",
        "disaggregation_decode_enable_radix_cache",
        # The mini-LB relays the decode's stream, and stream_generate reads the
        # whole reply off the final chunk, which incremental streaming would
        # cut down to the last delta.
        "incremental_streaming_output",
    ):
        if decode_info.get(key):
            failures.append(f"decode: {key}={decode_info.get(key)!r}, expected off")
    return failures


def check_engine_configuration(
    prefill_url: str, decode_url: str, kv_cache_dtype: str
) -> None:
    prefill_info = _http_json(f"{prefill_url}/server_info")
    decode_info = _http_json(f"{decode_url}/server_info")
    failures = engine_configuration_failures(prefill_info, decode_info, kv_cache_dtype)
    if failures:
        raise RuntimeError(
            "engine configuration differs from the launch recipe:\n  "
            + "\n  ".join(failures)
        )
    print(
        f"[{TEST_NAME}] engines verified: {kv_cache_dtype} target KV, "
        f"{DRAFT_KV_CACHE_DTYPE} draft KV, page {PAGE_SIZE}, context {CONTEXT_LEN}, "
        f"device pools {prefill_info['max_total_num_tokens']} (prefill) / "
        f"{decode_info['max_total_num_tokens']} (decode) tokens",
        flush=True,
    )


def _dump_info(metric: str, value: Any, *, probe: str, kv_cache_dtype: str) -> None:
    """Store a measurement with the run (kind=info: kept, not charted)."""
    if dump_metric is None or value is None:
        return
    dump_metric(
        f"{TEST_NAME}.{metric}",
        value,
        labels={
            "model": TEST_NAME,
            "kind": "info",
            "probe": probe,
            "kv_cache_dtype": kv_cache_dtype,
        },
    )


# --- Gate 2: ~126k-token KV offload round trip ---------------------------------


def kv_offload_round_trip_runner(prefill_url: str, kv_cache_dtype: str):
    """A long-context device -> host -> device round trip through the P/D cluster.

    Reuses the shared probe: the device hit and the host reload must decode the
    same text in the same verify forwards. The pool sizes come from the running
    prefill engine, and the probe and its fillers are tokenized first, so a
    prompt that would not fit the context fails here rather than on the server.
    """

    def run(base_url: str) -> Dict[str, Any]:
        flush_prefill_cache(prefill_url)
        info = _http_json(f"{prefill_url}/server_info")
        device_pool_tokens = int(info["max_total_num_tokens"])
        hicache_ratio = float(info["hicache_ratio"])
        probe_ids, filler_ids = tokenize(
            prefill_url,
            [
                probe_prompt(index=0, approx_tokens=OFFLOAD_PROBE_TOKENS),
                synthetic_prompt(index=100, approx_tokens=OFFLOAD_PROBE_TOKENS),
            ],
        )
        longest = max(len(probe_ids), len(filler_ids))
        if longest + DECODE_HEADROOM_TOKENS > CONTEXT_LEN:
            return {
                "score": 0,
                "_passed": False,
                "_failures": [
                    f"the {OFFLOAD_PROBE_TOKENS}-word probe tokenizes to {longest} "
                    f"tokens, over the {CONTEXT_LEN}-token context less "
                    f"{DECODE_HEADROOM_TOKENS} for decoding; lower OFFLOAD_PROBE_TOKENS"
                ],
            }
        plan = plan_eviction(
            device_pool_tokens=device_pool_tokens,
            probe_tokens=len(probe_ids),
            hicache_ratio=hicache_ratio,
        )
        probe = run_probe(
            base_url=base_url,
            test_name=TEST_NAME,
            probe_tokens=OFFLOAD_PROBE_TOKENS,
            plan=plan,
            flush_cache=False,
        )
        failures = validate_probe(probe=probe, page_size=PAGE_SIZE, plan=plan)
        failures += spec_failures(probe=probe)
        accept_length = mean_accept_length(probe=probe)
        print(
            json.dumps(
                {
                    "pd_mxint8_kv_offload_probe": {
                        "kv_cache_dtype": kv_cache_dtype,
                        "probe_tokens": len(probe_ids),
                        "device_pool_tokens": device_pool_tokens,
                        "host_pool_tokens": plan.host_pool_tokens,
                        "num_fillers": plan.num_fillers,
                        "warm": [probe.warm.device_tokens, probe.warm.host_tokens],
                        "reload": [
                            probe.reload.device_tokens,
                            probe.reload.host_tokens,
                        ],
                        "spec_verify_ct": [
                            probe.cold.spec_verify_ct,
                            probe.warm.spec_verify_ct,
                            probe.reload.spec_verify_ct,
                        ],
                        "mean_accept_length": accept_length,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for metric, value in (
            ("offload_device_cached_tokens", probe.warm.device_tokens),
            ("offload_host_cached_tokens", probe.reload.host_tokens),
            ("offload_spec_accept_length", accept_length),
        ):
            _dump_info(metric, value, probe="kv_offload", kv_cache_dtype=kv_cache_dtype)
        return {
            "score": probe.reload.host_tokens,
            "_passed": not failures,
            "_failures": failures,
        }

    return run


# --- Gate 3: multi-turn long context --------------------------------------------


class MultiTurnPlan(NamedTuple):
    """Round-robin sessions; each turn adds a reply and a new user block."""

    sessions: int
    turns: int
    user_tokens: int
    output_tokens: int
    page_size: int = PAGE_SIZE
    context_len: int = CONTEXT_LEN


def default_multi_turn_plan(env: Optional[Mapping[str, str]] = None) -> MultiTurnPlan:
    """MT_SESSIONS x MT_TURNS from the environment, else the defaults."""
    values = os.environ if env is None else env
    return MultiTurnPlan(
        sessions=int(values.get("MT_SESSIONS") or DEFAULT_MT_SESSIONS),
        turns=int(values.get("MT_TURNS") or DEFAULT_MT_TURNS),
        user_tokens=MT_TURN_TOKENS - MT_OUTPUT_TOKENS,
        output_tokens=MT_OUTPUT_TOKENS,
    )


def turn_prompt_tokens(plan: MultiTurnPlan, turn: int) -> int:
    """Prompt length of a 1-based turn: every user block so far, every earlier reply."""
    return turn * plan.user_tokens + (turn - 1) * plan.output_tokens


def expected_cached_tokens(previous_prompt_tokens: int, page_size: int) -> int:
    """The prefix a returning turn must hit: the previous prompt, page-aligned.

    The prefill engine caches only the prompt it computed (the reply's KV is
    made on the decode engine), and the radix tree keeps whole pages.
    """
    return previous_prompt_tokens // page_size * page_size


def multi_turn_plan_failures(
    plan: MultiTurnPlan, min_hit_rate: float = MT_MIN_HIT_RATE
) -> List[str]:
    failures = []
    if plan.sessions < 1:
        failures.append(f"need at least one session, got {plan.sessions}")
    if plan.turns < 2:
        failures.append(
            f"need at least two turns for a returning turn, got {plan.turns}"
        )
    if plan.user_tokens <= 0 or plan.output_tokens <= 0:
        failures.append(
            f"user_tokens={plan.user_tokens} and output_tokens={plan.output_tokens} "
            "must be positive"
        )
    final = turn_prompt_tokens(plan, plan.turns)
    if final + plan.output_tokens + DECODE_HEADROOM_TOKENS > plan.context_len:
        failures.append(
            f"turn {plan.turns} needs {final} prompt + {plan.output_tokens} output "
            f"tokens, over the {plan.context_len}-token context less "
            f"{DECODE_HEADROOM_TOKENS} headroom"
        )
    if not failures:
        # The first turn of every session is a full miss, so fewer turns lower
        # the best reachable hit rate (0.597 at four turns).
        ideal = theoretical_hit_rate(plan)
        if ideal < min_hit_rate:
            failures.append(
                f"a {plan.turns}-turn plan reaches at most a {ideal:.3f} hit rate, "
                f"below the {min_hit_rate} floor; lower MT_MIN_HIT_RATE with MT_TURNS"
            )
    return failures


def theoretical_hit_rate(plan: MultiTurnPlan) -> float:
    """Cached / prompt tokens if every turn hits exactly its expected prefix."""
    prompts = [turn_prompt_tokens(plan, turn) for turn in range(1, plan.turns + 1)]
    cached = [0] + [expected_cached_tokens(p, plan.page_size) for p in prompts[:-1]]
    return sum(cached) / sum(prompts)


def requires_host_hits(plan: MultiTurnPlan, device_pool_tokens: int) -> bool:
    """Whether the sessions outgrow the prefill's device pool before their last turn.

    After round r the prefill holds every session's prompt (~sessions * N_r
    tokens). Once that exceeds the device pool, LRU eviction leaves part of some
    session's prefix only on host, and that session's next turn must load it
    back. One turn's growth of slack keeps a borderline plan from being judged
    on rounding.
    """
    held = plan.sessions * turn_prompt_tokens(plan, plan.turns - 1)
    return held >= device_pool_tokens + plan.user_tokens + plan.output_tokens


_SESSION_CREWS = (
    "amber",
    "birch",
    "cedar",
    "dune",
    "ember",
    "fern",
    "granite",
    "harbor",
)
_PORTS = (
    "Lisbon",
    "Osaka",
    "Denver",
    "Nairobi",
    "Oslo",
    "Lima",
    "Perth",
    "Quebec",
    "Austin",
    "Hanoi",
    "Seville",
    "Tunis",
)
_GOODS = (
    "copper wire",
    "olive oil",
    "glass jars",
    "wool coats",
    "tea leaves",
    "steel bolts",
    "paper rolls",
    "rice sacks",
    "cotton bales",
    "clay tiles",
)
# A log entry is ~27 tokens; this lower bound sizes the generated log before it
# is tokenized and checked.
_MIN_TOKENS_PER_LOG_ENTRY = 20
_FIVE_DIGITS = re.compile(r"(?<!\d)\d{5}(?!\d)")


def session_secret(session: int) -> str:
    """A five-digit code per session.

    The log uses only 3- and 4-digit numbers, so the secret is the only
    five-digit number in a session's context.
    """
    return str(10000 + (31847 + 7919 * session) % 90000)


def _session_crew(session: int) -> str:
    return _SESSION_CREWS[session % len(_SESSION_CREWS)]


def session_header(session: int) -> str:
    return (
        f"Session {session} shipping log, kept by the {_session_crew(session)} crew. "
        f"The secret code for session {session} is {session_secret(session)}. "
        "Remember it.\n\n"
    )


def session_question(session: int) -> str:
    return (
        f"\n\nQuestion: What is the secret code for session {session}? "
        "Reply with only the five-digit code.\n"
        f"Answer: The secret code for session {session} is"
    )


def session_log_text(session: int, num_entries: int) -> str:
    """Deterministic filler log, distinct per session from its first entry."""
    crew = _session_crew(session)
    seed = (session + 1) * 2654435761 & 0x7FFFFFFF
    entries = []
    for index in range(num_entries):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        entries.append(
            f"Entry {1000 + index % 9000} for the {crew} crew: truck "
            f"{100 + seed % 900} moved {100 + (seed >> 7) % 9900} crates of "
            f"{_GOODS[(seed >> 3) % len(_GOODS)]} from "
            f"{_PORTS[(seed >> 11) % len(_PORTS)]} to "
            f"{_PORTS[(seed >> 17) % len(_PORTS)]} at dock {100 + (seed >> 13) % 900}."
        )
    return " ".join(entries)


def recalled_code(text: str) -> Optional[str]:
    """The first five-digit number in a reply, or None."""
    match = _FIVE_DIGITS.search(text or "")
    return match.group(0) if match else None


def split_session_blocks(
    header_ids: Sequence[int],
    body_ids: Sequence[int],
    question_ids: Sequence[int],
    plan: MultiTurnPlan,
) -> List[List[int]]:
    """One session's new-text blocks, exactly plan.user_tokens ids per turn.

    The first block opens with the header, which plants the secret, and the
    last block ends with the question that asks for it.
    """
    blocks = []
    offset = 0
    for turn in range(1, plan.turns + 1):
        head = list(header_ids) if turn == 1 else []
        tail = list(question_ids) if turn == plan.turns else []
        take = plan.user_tokens - len(head) - len(tail)
        if take < 0:
            raise ValueError(
                f"turn {turn}: header and question take {len(head) + len(tail)} "
                f"tokens, more than the {plan.user_tokens}-token block"
            )
        body = list(body_ids[offset : offset + take])
        if len(body) < take:
            raise ValueError(
                f"the filler log ran out at turn {turn}: needed {take} tokens, "
                f"{len(body)} left"
            )
        offset += take
        blocks.append(head + body + tail)
    return blocks


def build_session_blocks(
    prefill_url: str, plan: MultiTurnPlan
) -> List[List[List[int]]]:
    """Tokenize every session's header, log and question; cut them into blocks."""
    num_entries = plan.turns * plan.user_tokens // _MIN_TOKENS_PER_LOG_ENTRY + 1
    sessions = []
    for session in range(plan.sessions):
        header_ids, body_ids, question_ids = tokenize(
            prefill_url,
            [
                session_header(session),
                session_log_text(session, num_entries),
                session_question(session),
            ],
        )
        sessions.append(split_session_blocks(header_ids, body_ids, question_ids, plan))
    return sessions


def turn_record(
    *,
    session: int,
    turn: int,
    sent_tokens: int,
    expected_cached: int,
    response: Mapping[str, Any],
    transfer: Mapping[str, Any],
) -> Dict[str, Any]:
    final = response["final"]
    meta = final.get("meta_info") or {}
    # /generate omits the breakdown when no tier hit, which means zero.
    details = meta.get("cached_tokens_details") or {}
    finish = meta.get("finish_reason")
    return {
        "session": session,
        "turn": turn,
        "sent_tokens": sent_tokens,
        "prompt_tokens": int(meta.get("prompt_tokens") or 0),
        "expected_cached": expected_cached,
        "cached_tokens": int(meta.get("cached_tokens") or 0),
        "device_cached_tokens": int(details.get("device") or 0),
        "host_cached_tokens": int(details.get("host") or 0),
        "completion_tokens": int(meta.get("completion_tokens") or 0),
        "spec_verify_ct": int(meta.get("spec_verify_ct") or 0),
        "finish_reason": finish.get("type") if isinstance(finish, dict) else finish,
        "ttft_s": response["ttft_s"],
        "e2e_s": response["e2e_s"],
        "transfer_stage_s": transfer.get("stage_s"),
        "kv_transfer_latency_ms": transfer.get("latency_ms"),
        "kv_transfer_mb": transfer.get("mb"),
        "text": final.get("text") or "",
        "output_ids": list(final.get("output_ids") or []),
    }


def _printable(record: Mapping[str, Any]) -> Dict[str, Any]:
    shown = {k: v for k, v in record.items() if k not in ("output_ids", "text")}
    shown["text_head"] = record["text"][:120]
    return shown


def run_multi_turn_sessions(
    base_url: str,
    prefill_url: str,
    plan: MultiTurnPlan,
    blocks: Sequence[Sequence[Sequence[int]]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Serve every session round-robin, one request in flight.

    Turn k of a session sends its turn k-1 prompt, the turn k-1 reply and a new
    block, as token ids, so the cached prefix is exact. Stops at the first
    failed request and returns what completed with the error.
    """
    prompts: List[List[int]] = [[] for _ in range(plan.sessions)]
    replies: List[List[int]] = [[] for _ in range(plan.sessions)]
    records: List[Dict[str, Any]] = []
    metrics_before = _http_text(f"{prefill_url}/metrics")
    for turn in range(1, plan.turns + 1):
        for session in range(plan.sessions):
            previous_tokens = len(prompts[session])
            prompt = (
                prompts[session] + replies[session] + list(blocks[session][turn - 1])
            )
            try:
                response = stream_generate(base_url, prompt, plan.output_tokens)
                metrics_after = _http_text(f"{prefill_url}/metrics")
            except Exception as exc:  # report the turns that completed
                return records, (
                    f"session {session} turn {turn} ({len(prompt)} prompt tokens): "
                    f"{type(exc).__name__}: {exc}"
                )
            record = turn_record(
                session=session,
                turn=turn,
                sent_tokens=len(prompt),
                expected_cached=(
                    expected_cached_tokens(previous_tokens, plan.page_size)
                    if turn > 1
                    else 0
                ),
                response=response,
                transfer=transfer_delta(metrics_before, metrics_after),
            )
            metrics_before = metrics_after
            records.append(record)
            print(
                json.dumps(
                    {"pd_mxint8_multiturn_turn": _printable(record)}, sort_keys=True
                ),
                flush=True,
            )
            prompts[session] = prompt
            replies[session] = record["output_ids"]
    return records, None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _percentile(values: Iterable[Optional[float]], fraction: float) -> Optional[float]:
    """Nearest-rank percentile, ignoring missing values."""
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize_multi_turn(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    prompt = sum(r["prompt_tokens"] for r in records)
    cached = sum(r["cached_tokens"] for r in records)
    completion = sum(r["completion_tokens"] for r in records)
    verify = sum(r["spec_verify_ct"] for r in records)
    last_turn = max((r["turn"] for r in records), default=0)
    last = [r for r in records if r["turn"] == last_turn]
    ttft = [r["ttft_s"] for r in records]
    return {
        "turns": len(records),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "device_cached_tokens": sum(r["device_cached_tokens"] for r in records),
        "host_cached_tokens": sum(r["host_cached_tokens"] for r in records),
        "hit_rate": cached / prompt if prompt else 0.0,
        # Tokens per verify forward (accepted drafts plus the bonus token). Each
        # reply's first token is sampled by the prefill, not by a verify, so it
        # is left out: no accepted draft anywhere gives exactly 1.0.
        "accept_length": (completion - len(records)) / verify if verify else 0.0,
        "ttft_s_mean": _mean(ttft),
        "ttft_s_p50": _percentile(ttft, 0.5),
        "ttft_s_p90": _percentile(ttft, 0.9),
        "ttft_s_max": max((t for t in ttft if t is not None), default=None),
        "ttft_s_last_turn_mean": _mean(r["ttft_s"] for r in last),
        "e2e_s_mean": _mean(r["e2e_s"] for r in records),
        "transfer_stage_s_mean": _mean(r["transfer_stage_s"] for r in records),
        "transfer_stage_s_last_turn_mean": _mean(r["transfer_stage_s"] for r in last),
        "kv_transfer_latency_ms_mean": _mean(
            r["kv_transfer_latency_ms"] for r in records
        ),
        "kv_transfer_mb_total": sum(r["kv_transfer_mb"] or 0.0 for r in records),
    }


def multi_turn_failures(
    *,
    records: Sequence[Mapping[str, Any]],
    plan: MultiTurnPlan,
    require_host_hits: bool,
    counter_deltas: Mapping[str, float],
    transfer_observations: float,
    error: Optional[str] = None,
    min_hit_rate: float = MT_MIN_HIT_RATE,
    min_accept_length: float = MT_MIN_ACCEPT_LENGTH,
) -> List[str]:
    """Every multi-turn judgement. An empty list means the phase passed."""
    failures: List[str] = []
    if error:
        failures.append(f"the multi-turn phase stopped at {error}")
    want_turns = plan.sessions * plan.turns
    if len(records) != want_turns:
        failures.append(f"{len(records)} of {want_turns} turns completed")

    for r in records:
        label = f"session {r['session']} turn {r['turn']}"
        if r["completion_tokens"] != plan.output_tokens:
            failures.append(
                f"{label}: {r['completion_tokens']} completion tokens, expected "
                f"{plan.output_tokens}"
            )
        # The next turn's prompt is built from these ids.
        if len(r["output_ids"]) != r["completion_tokens"]:
            failures.append(
                f"{label}: the final stream chunk carried {len(r['output_ids'])} "
                f"output ids for {r['completion_tokens']} completion tokens"
            )
        if r["finish_reason"] != "length":
            failures.append(
                f"{label}: finish_reason={r['finish_reason']!r}, expected 'length'"
            )
        if r["prompt_tokens"] != r["sent_tokens"]:
            failures.append(
                f"{label}: the server counted {r['prompt_tokens']} prompt tokens "
                f"for {r['sent_tokens']} sent"
            )
        if r["spec_verify_ct"] <= 0:
            failures.append(
                f"{label}: spec_verify_ct=0, speculative decoding did not run"
            )
        cached = r["cached_tokens"]
        if r["turn"] == 1:
            if cached >= plan.page_size:
                failures.append(
                    f"{label}: a first turn hit {cached} cached tokens on a flushed cache"
                )
            continue
        floor = r["expected_cached"] - plan.page_size
        if cached < floor:
            failures.append(
                f"{label}: {cached} cached tokens ({r['device_cached_tokens']} device, "
                f"{r['host_cached_tokens']} host), expected at least {floor}: the "
                f"previous turn's {r['expected_cached']} page-aligned prompt tokens "
                "less one page"
            )
        if cached >= r["prompt_tokens"]:
            failures.append(
                f"{label}: {cached} cached of {r['prompt_tokens']} prompt tokens; "
                "the last prompt token must be computed"
            )

    if records:
        summary = summarize_multi_turn(records)
        if summary["hit_rate"] < min_hit_rate:
            failures.append(
                f"aggregate prefix hit rate {summary['hit_rate']:.3f} < {min_hit_rate}"
            )
        if require_host_hits and summary["host_cached_tokens"] <= 0:
            failures.append(
                "no turn reloaded its prefix from host although the sessions "
                "outgrew the device pool before their final turn: KV offload never "
                "engaged"
            )
        if summary["accept_length"] <= min_accept_length:
            failures.append(
                f"accept length {summary['accept_length']:.3f} <= {min_accept_length} "
                "tokens per verify forward"
            )
        if transfer_observations <= 0:
            failures.append(
                "the prefill /metrics recorded no P->D transfer timing during the phase"
            )

    finals = {r["session"]: r for r in records if r["turn"] == plan.turns}
    for session, r in sorted(finals.items()):
        secret = session_secret(session)
        answer = recalled_code(r["text"])
        if answer != secret:
            failures.append(
                f"session {session}: the final turn answered {answer!r}, expected the "
                f"secret {secret} from its first turn; reply={r['text'][:160]!r}"
            )

    for name, delta in sorted(counter_deltas.items()):
        if delta > 0:
            failures.append(f"{name} rose by {delta:g} during the multi-turn phase")
    return failures


def multi_turn_runner(
    prefill_url: str,
    decode_url: str,
    kv_cache_dtype: str,
    env: Optional[Mapping[str, str]] = None,
):
    """Interleaved long-context sessions with exact per-turn prefix-hit gates."""
    plan = default_multi_turn_plan(env)

    def run(base_url: str) -> Dict[str, Any]:
        failures = multi_turn_plan_failures(plan)
        if failures:
            return {"score": 0.0, "_passed": False, "_failures": failures}
        flush_prefill_cache(prefill_url)
        device_pool_tokens = int(
            _http_json(f"{prefill_url}/server_info")["max_total_num_tokens"]
        )
        require_host = requires_host_hits(plan, device_pool_tokens)
        print(
            f"[{TEST_NAME}] multi-turn plan {plan._asdict()}: final prompt "
            f"{turn_prompt_tokens(plan, plan.turns)} tokens, ideal hit rate "
            f"{theoretical_hit_rate(plan):.3f}, prefill device pool "
            f"{device_pool_tokens} tokens, host reloads required: {require_host}",
            flush=True,
        )
        blocks = build_session_blocks(prefill_url, plan)
        engines = {"prefill": prefill_url, "decode": decode_url}
        before = {role: _http_text(f"{url}/metrics") for role, url in engines.items()}
        records, error = run_multi_turn_sessions(base_url, prefill_url, plan, blocks)
        after = {role: _http_text(f"{url}/metrics") for role, url in engines.items()}
        transfer = transfer_delta(before["prefill"], after["prefill"])
        deltas = counter_deltas(before, after)
        summary = summarize_multi_turn(records)
        summary.update(
            kv_cache_dtype=kv_cache_dtype,
            device_pool_tokens=device_pool_tokens,
            require_host_hits=require_host,
            transfer_observations=transfer["observations"],
            failure_counter_deltas=deltas,
        )
        print(
            json.dumps({"pd_mxint8_multiturn_summary": summary}, sort_keys=True),
            flush=True,
        )
        for metric in (
            "hit_rate",
            "host_cached_tokens",
            "accept_length",
            "ttft_s_p50",
            "ttft_s_max",
            "ttft_s_last_turn_mean",
            "transfer_stage_s_mean",
            "transfer_stage_s_last_turn_mean",
            "kv_transfer_latency_ms_mean",
        ):
            _dump_info(
                f"multi_turn_{metric}",
                summary[metric],
                probe="multi_turn",
                kv_cache_dtype=kv_cache_dtype,
            )
        failures = multi_turn_failures(
            records=records,
            plan=plan,
            require_host_hits=require_host,
            counter_deltas=deltas,
            transfer_observations=transfer["observations"],
            error=error,
        )
        return {
            "score": summary["hit_rate"],
            "_passed": not failures,
            "_failures": failures,
        }

    return run


def build_model_config(
    prefill_url: str, decode_url: str, env: Optional[Mapping[str, str]] = None
) -> Dict[str, Any]:
    """The gates run in list order: GSM8K first keeps its accept length clean."""
    kv_cache_dtype = resolve_kv_cache_dtype(env)
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": TEST_NAME,
        "variant": (
            f"PD-1P1D-TP8+EP8-raiden+EAGLE3+prefixcache+kvoffload+{kv_cache_dtype}KV"
            "+128k-multiturn"
        ),
        "gates": [
            {
                "dataset": "gsm8k",
                "label": "gsm8k",
                "baseline": GSM8K_BASELINE,
                "atol": ATOL,
                "num_examples": NUM_EXAMPLES,
                "num_threads": NUM_THREADS,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "spec_accept_length_threshold": SPEC_ACCEPT_LENGTH_THRESHOLD,
            },
            {
                "dataset": "kv_offload_round_trip",
                "label": "kv-offload-round-trip",
                "runner": kv_offload_round_trip_runner(prefill_url, kv_cache_dtype),
            },
            {
                "dataset": "multi_turn_long_context",
                "label": "multi-turn-long-context",
                "runner": multi_turn_runner(
                    prefill_url, decode_url, kv_cache_dtype, env
                ),
            },
        ],
    }


class TestPdSingleHostMxint8MultiTurnGptOss120bIntegration(PDSingleHostTestCase):
    """Configure the shared P/D fixture for this test's recipe and gates."""

    config_module = LAUNCH_CONFIG_MODULE
    repo_root = REPO
    cwd = BENCH
    log_path = PREFILL_LOG
    decode_log_path = DECODE_LOG
    self_hostname = staticmethod(self_hostname)
    decode_host_env = DECODE_HOST_ENV
    tag = "pd-singlehost-mxint8-multiturn"
    health_timeout = 2 * WAIT_SECS + 300
    health_generate_timeout = HEALTH_TIMEOUT

    @classmethod
    def _urls(cls) -> Tuple[str, str]:
        return (
            f"http://{LOCAL_HOST}:{PREFILL_PORT}",
            f"http://{cls.decode_hostname()}:{DECODE_PORT}",
        )

    @classmethod
    def build_model_config(cls) -> Dict[str, Any]:
        return build_model_config(*cls._urls())

    @classmethod
    def check_server_configuration(cls) -> None:
        check_engine_configuration(*cls._urls(), resolve_kv_cache_dtype())

    @classmethod
    def setUpClass(cls) -> None:
        validate_model_source(resolve_model_path(), label="target")
        validate_model_source(resolve_draft_model_path(), label="draft")
        try:
            super().setUpClass()
        except BaseException:
            log = PREFILL_LOG if os.environ.get("CI_PD_ROLE") == PREFILL_ROLE else None
            hint = log and startup_failure_hint(log, resolve_kv_cache_dtype())
            if hint:
                print(f"[{TEST_NAME}] {hint}", flush=True)
            raise

    def test_accuracy(self) -> None:
        self.run_evaluations()


def render_jobset_yaml(
    image: str = "sglang-tpu:latest",
    lane: str = "nightly",
    weights_pvc: str = "model-weights",
    run_name: str = RUN_NAME,
) -> str:
    """Render one full prefill host plus one full decode host."""
    rel_path = os.path.relpath(__file__, REPO)
    cmd = [
        PYTHON,
        RENDER_JOBSET,
        "--pd-prefill-hosts",
        "1",
        "--pd-decode-hosts",
        "1",
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


def launch_plan(role: str, env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The launcher's dry-run plan for one engine of the recipe, without a TPU."""
    child_env = {
        k: v
        for k, v in os.environ.items()
        # Operator overrides the recipe reads; the CPU checks pin its defaults.
        if k
        not in (KV_CACHE_DTYPE_ENV, MODEL_ENV, "MODEL", DRAFT_MODEL_ENV, "DRAFT_MODEL")
    }
    child_env.update(env or {})
    output = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "benchmark.tpu.launch_server",
            "--config",
            f"{LAUNCH_CONFIG_MODULE}:{role.upper()}_CONFIG",
            "--runtime-envs",
            json.dumps({"FORCE_SINGLE_HOST": "1", "SGLANG_HOST_IP": LOCAL_HOST}),
            "--dry-run",
        ],
        cwd=REPO,
        env=child_env,
        text=True,
    )
    return json.loads(output)


class TestPdSingleHostMxint8MultiTurnGptOss120bConfiguration(unittest.TestCase):
    """Host-side configuration and gate-logic tests runnable without TPU hardware."""

    PLAN = MultiTurnPlan(sessions=2, turns=3, user_tokens=8128, output_tokens=64)

    @classmethod
    def setUpClass(cls) -> None:
        cls.plans = {role: launch_plan(role) for role in (PREFILL_ROLE, DECODE_ROLE)}

    def test_both_engines_share_the_launch_contract(self) -> None:
        for role, plan in self.plans.items():
            with self.subTest(role):
                args = plan["server_args"]
                self.assertEqual(args["disaggregation_mode"], role)
                self.assertEqual(args["disaggregation_transfer_backend"], "raiden")
                self.assertEqual(args["tp_size"], TP_SIZE)
                self.assertEqual(args["ep_size"], EP_SIZE)
                self.assertEqual(args["page_size"], PAGE_SIZE)
                self.assertEqual(args["context_length"], CONTEXT_LEN)
                self.assertEqual(args["max_total_tokens"], MAX_TOTAL_TOKENS)
                self.assertEqual(args["kv_cache_dtype"], "mxint8")
                self.assertIsNone(args["quantization"])
                self.assertTrue(args["enable_metrics"])
                self.assertTrue(args["skip_server_warmup"])
                self.assertEqual(args["reasoning_parser"], "gpt-oss")
                self.assertEqual(args["tpu_compile_bs"], [1, 2, 4, 16])

    def test_kv_cache_dtype_override(self) -> None:
        bf16 = launch_plan(PREFILL_ROLE, env={KV_CACHE_DTYPE_ENV: "bfloat16"})
        self.assertEqual(bf16["server_args"]["kv_cache_dtype"], "bfloat16")
        self.assertEqual(resolve_kv_cache_dtype({KV_CACHE_DTYPE_ENV: ""}), "mxint8")
        for rejected in ("fp8_e4m3", "bf16", "auto"):
            with self.subTest(rejected=rejected):
                with self.assertRaisesRegex(ValueError, KV_CACHE_DTYPE_ENV):
                    resolve_kv_cache_dtype({KV_CACHE_DTYPE_ENV: rejected})

    def test_both_engines_run_eagle3_with_bf16_draft_kv(self) -> None:
        # The P/D handoff carries spec bootstrap state, so BOTH engines run spec.
        for plan in self.plans.values():
            args = plan["server_args"]
            self.assertEqual(args["speculative_algorithm"], "EAGLE3")
            self.assertEqual(
                args["speculative_draft_model_path"], DEFAULT_DRAFT_MODEL_PATH
            )
            self.assertEqual(args["speculative_eagle_topk"], 1)
            self.assertEqual(args["speculative_draft_kv_cache_dtype"], "bfloat16")

    def test_hierarchical_cache_runs_only_on_prefill(self) -> None:
        prefill = self.plans[PREFILL_ROLE]["server_args"]
        self.assertFalse(prefill["disable_radix_cache"])
        self.assertTrue(prefill["enable_hierarchical_cache"])
        self.assertEqual(prefill["hicache_ratio"], HICACHE_RATIO)
        self.assertEqual(prefill["hicache_io_backend"], "raiden")
        self.assertEqual(prefill["hicache_mem_layout"], "page_first_fused")
        self.assertEqual(prefill["hicache_write_policy"], HICACHE_WRITE_POLICY)
        self.assertTrue(prefill["enable_cache_report"])
        decode = self.plans[DECODE_ROLE]["server_args"]
        self.assertFalse(any(key.startswith("hicache") for key in decode))
        self.assertFalse(decode.get("enable_hierarchical_cache"))
        self.assertFalse(decode.get("disaggregation_decode_enable_radix_cache"))
        self.assertEqual(
            decode["disaggregation_decode_retraction_backup"], RETRACTION_BACKUP
        )

    def test_peer_hostname_maps_prefill_to_decode(self) -> None:
        prefill = f"tpu-ci-{RUN_NAME}-prefill-0-0"
        self.assertEqual(
            peer_hostname(prefill, from_role=PREFILL_ROLE, to_role=DECODE_ROLE),
            f"tpu-ci-{RUN_NAME}-decode-0-0",
        )
        self.assertEqual(self_hostname({SELF_HOST_ENV: "p.svc"}), "p.svc")
        self.assertEqual(
            self_hostname({"TPU_WORKER_HOSTNAMES": "a-prefill-0-0,b"}), "a-prefill-0-0"
        )

    def test_local_model_path_validation(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, MODEL_ENV):
            validate_model_source(
                "/weights/definitely-missing-gpt-oss-120b", label="target"
            )
        with self.assertRaisesRegex(FileNotFoundError, DRAFT_MODEL_ENV):
            validate_model_source("/weights/definitely-missing-draft", label="draft")
        validate_model_source(HF_REPO_ID, label="target")

    @unittest.skipUnless(
        os.path.exists(RENDER_JOBSET), "renderer is not shipped in the runtime image"
    )
    def test_dry_run_renders_one_host_per_role(self) -> None:
        yaml_content = render_jobset_yaml()
        self.assertIn("name: prefill", yaml_content)
        self.assertIn("name: decode", yaml_content)
        self.assertIn(f"{os.path.basename(__file__)} integration", yaml_content)

    def test_startup_hint_names_the_missing_mxint8_extension(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as log:
            log.write(
                "ValueError: MXINT8 KV cache is unsupported without a compatible "
                "backend extension; configure support before constructing...\n"
            )
        self.addCleanup(os.unlink, log.name)
        self.assertIn(KV_CACHE_DTYPE_ENV, startup_failure_hint(log.name, "mxint8"))
        self.assertIsNone(startup_failure_hint(log.name, "bfloat16"))
        self.assertIsNone(startup_failure_hint("/nonexistent/pd.log", "mxint8"))

    def test_default_multi_turn_plan_fits_and_outgrows_the_device_pool(self) -> None:
        plan = default_multi_turn_plan(env={})
        self.assertEqual(
            multi_turn_plan_failures(plan, min_hit_rate=DEFAULT_MT_MIN_HIT_RATE), []
        )
        self.assertEqual(turn_prompt_tokens(plan, plan.turns), 122816)
        self.assertLessEqual(
            turn_prompt_tokens(plan, plan.turns) + plan.output_tokens, CONTEXT_LEN
        )
        self.assertTrue(requires_host_hits(plan, MAX_TOTAL_TOKENS))
        # Four short turns fit in HBM, so they must not demand host reloads.
        self.assertFalse(requires_host_hits(plan._replace(turns=4), MAX_TOTAL_TOKENS))

    def test_expected_cached_tokens_is_the_previous_prompt_page_aligned(self) -> None:
        plan = default_multi_turn_plan(env={})
        for turn in range(2, plan.turns + 1):
            previous = turn_prompt_tokens(plan, turn - 1)
            # Prompts end mid-page (8192k - 64), so the radix match does not
            # depend on whether the engine matches len or len - 1 tokens.
            self.assertEqual(previous % PAGE_SIZE, 64)
            self.assertEqual(
                expected_cached_tokens(previous, PAGE_SIZE), 8192 * (turn - 1) - 128
            )
            self.assertEqual(
                expected_cached_tokens(previous, PAGE_SIZE),
                expected_cached_tokens(previous - 1, PAGE_SIZE),
            )

    def test_hit_rate_floor_is_reachable_with_per_turn_slack(self) -> None:
        plan = default_multi_turn_plan(env={})
        self.assertAlmostEqual(theoretical_hit_rate(plan), 0.874, places=3)
        # The per-turn gate allows one page less than ideal on every returning
        # turn; even then the aggregate floor holds.
        prompts = sum(turn_prompt_tokens(plan, k) for k in range(1, plan.turns + 1))
        worst = theoretical_hit_rate(plan) - (plan.turns - 1) * PAGE_SIZE / prompts
        self.assertGreaterEqual(worst, DEFAULT_MT_MIN_HIT_RATE)

    def test_multi_turn_plan_rejects_overlong_and_degenerate_plans(self) -> None:
        plan = default_multi_turn_plan(env={})
        self.assertTrue(multi_turn_plan_failures(plan._replace(turns=16)))
        self.assertTrue(multi_turn_plan_failures(plan._replace(turns=1)))
        self.assertTrue(multi_turn_plan_failures(plan._replace(sessions=0)))
        # A shortened plan cannot reach the default floor; it must say so up
        # front rather than fail after serving every turn.
        short = plan._replace(turns=4)
        (failure,) = multi_turn_plan_failures(
            short, min_hit_rate=DEFAULT_MT_MIN_HIT_RATE
        )
        self.assertIn("MT_MIN_HIT_RATE", failure)
        self.assertEqual(multi_turn_plan_failures(short, min_hit_rate=0.5), [])

    def test_split_session_blocks(self) -> None:
        plan = MultiTurnPlan(sessions=1, turns=3, user_tokens=10, output_tokens=2)
        header, question = [-1, -2], [-3, -4, -5]
        blocks = split_session_blocks(header, list(range(100)), question, plan)
        self.assertEqual([len(b) for b in blocks], [10, 10, 10])
        self.assertEqual(blocks[0][:2], header)
        self.assertEqual(blocks[-1][-3:], question)
        body = [t for b in blocks for t in b if t >= 0]
        self.assertEqual(body, list(range(len(body))))
        with self.assertRaisesRegex(ValueError, "ran out"):
            split_session_blocks(header, list(range(20)), question, plan)

    def test_each_session_context_holds_exactly_its_own_secret(self) -> None:
        secrets = [session_secret(s) for s in range(12)]
        self.assertEqual(len(set(secrets)), len(secrets))
        for session in range(12):
            text = (
                session_header(session)
                + session_log_text(session, 2000)
                + session_question(session)
            )
            self.assertEqual(_FIVE_DIGITS.findall(text), [secrets[session]])
        self.assertNotEqual(session_log_text(0, 3), session_log_text(1, 3))

    def test_recalled_code(self) -> None:
        self.assertEqual(recalled_code(" 31847.\n\nEntry 1234"), "31847")
        self.assertEqual(recalled_code("code:31847"), "31847")
        self.assertIsNone(recalled_code("no digits, 1234 or 123456"))
        self.assertIsNone(recalled_code(""))

    def test_parse_sse_line(self) -> None:
        chunk = parse_sse_line(b'data: {"text": "a", "output_ids": [1]}\n')
        self.assertEqual(chunk, {"text": "a", "output_ids": [1]})
        self.assertEqual(parse_sse_line("data: [DONE]"), SSE_DONE)
        self.assertIsNone(parse_sse_line(b"\n"))
        self.assertIsNone(parse_sse_line(": keep-alive"))
        error = parse_sse_line('data: {"error": {"message": "boom"}}')
        self.assertEqual(error["error"]["message"], "boom")

    METRICS_BEFORE = "\n".join(
        [
            "# HELP sglang:num_transfer_failed_reqs_total failed",
            'sglang:num_transfer_failed_reqs_total{engine_type="prefill",tp_rank="0"} 0.0',
            'sglang:per_stage_req_latency_seconds_sum{stage="prefill_transfer_kv_cache",tp_rank="0"} 1.0',
            'sglang:per_stage_req_latency_seconds_count{stage="prefill_transfer_kv_cache",tp_rank="0"} 2.0',
            'sglang:per_stage_req_latency_seconds_sum{stage="prefill_forward",tp_rank="0"} 9.0',
            'sglang:per_stage_req_latency_seconds_count{stage="prefill_forward",tp_rank="0"} 9.0',
        ]
    )
    METRICS_AFTER = "\n".join(
        [
            'sglang:num_transfer_failed_reqs_total{engine_type="prefill",tp_rank="0"} 1.0',
            'sglang:num_transfer_failed_reqs_total{engine_type="prefill",tp_rank="1"} 1.0',
            'sglang:per_stage_req_latency_seconds_sum{stage="prefill_transfer_kv_cache",tp_rank="0"} 2.5',
            'sglang:per_stage_req_latency_seconds_count{stage="prefill_transfer_kv_cache",tp_rank="0"} 3.0',
            'sglang:per_stage_req_latency_seconds_sum{stage="prefill_forward",tp_rank="0"} 99.0',
            'sglang:per_stage_req_latency_seconds_count{stage="prefill_forward",tp_rank="0"} 10.0',
            'sglang:kv_transfer_latency_ms_sum{tp_rank="0"} 40.0',
            'sglang:kv_transfer_latency_ms_count{tp_rank="0"} 1.0',
            'sglang:kv_transfer_total_mb_sum{tp_rank="0"} 512.0',
            'sglang:kv_transfer_total_mb_count{tp_rank="0"} 1.0',
        ]
    )

    def test_sum_samples_filters_labels_and_sums_ranks(self) -> None:
        after = self.METRICS_AFTER
        self.assertEqual(
            sum_samples(after, "sglang:num_transfer_failed_reqs_total"), 2.0
        )
        self.assertEqual(
            sum_samples(
                after, "sglang:per_stage_req_latency_seconds_count", _TRANSFER_STAGE
            ),
            3.0,
        )
        self.assertEqual(sum_samples(after, "sglang:num_retracted_requests_total"), 0.0)

    def test_transfer_and_counter_deltas(self) -> None:
        transfer = transfer_delta(self.METRICS_BEFORE, self.METRICS_AFTER)
        self.assertAlmostEqual(transfer["stage_s"], 1.5)
        self.assertAlmostEqual(transfer["latency_ms"], 40.0)
        self.assertAlmostEqual(transfer["mb"], 512.0)
        # One transfer, seen by both histograms, counts once.
        self.assertEqual(transfer["observations"], 1.0)
        idle = transfer_delta(self.METRICS_AFTER, self.METRICS_AFTER)
        self.assertIsNone(idle["stage_s"])
        self.assertEqual(idle["observations"], 0.0)
        deltas = counter_deltas(
            {"prefill": self.METRICS_BEFORE, "decode": ""},
            {"prefill": self.METRICS_AFTER, "decode": ""},
        )
        self.assertEqual(deltas["prefill sglang:num_transfer_failed_reqs_total"], 2.0)
        self.assertEqual(deltas["decode sglang:num_retracted_requests_total"], 0.0)

    def _records(self, plan: MultiTurnPlan) -> List[Dict[str, Any]]:
        """Records of a healthy run: ideal hits, 3 tokens per verify, recall."""
        records = []
        for turn in range(1, plan.turns + 1):
            for session in range(plan.sessions):
                prompt = turn_prompt_tokens(plan, turn)
                cached = (
                    expected_cached_tokens(
                        turn_prompt_tokens(plan, turn - 1), PAGE_SIZE
                    )
                    if turn > 1
                    else 0
                )
                records.append(
                    {
                        "session": session,
                        "turn": turn,
                        "sent_tokens": prompt,
                        "prompt_tokens": prompt,
                        "expected_cached": cached,
                        "cached_tokens": cached,
                        "device_cached_tokens": cached
                        - (128 if turn == plan.turns else 0),
                        "host_cached_tokens": 128 if turn == plan.turns else 0,
                        "completion_tokens": plan.output_tokens,
                        "spec_verify_ct": 21,
                        "finish_reason": "length",
                        "ttft_s": 1.0 + turn,
                        "e2e_s": 2.0 + turn,
                        "transfer_stage_s": 0.5,
                        "kv_transfer_latency_ms": 400.0,
                        "kv_transfer_mb": 100.0,
                        "text": f" {session_secret(session)}."
                        if turn == plan.turns
                        else "x",
                        "output_ids": [1] * plan.output_tokens,
                    }
                )
        return records

    def _failures(self, records, **overrides) -> List[str]:
        kwargs = dict(
            records=records,
            plan=self.PLAN,
            require_host_hits=True,
            counter_deltas={"decode sglang:num_retracted_requests_total": 0.0},
            transfer_observations=6.0,
            error=None,
            # PLAN's three turns reach at most a 0.497 hit rate.
            min_hit_rate=0.45,
            min_accept_length=DEFAULT_MT_MIN_ACCEPT_LENGTH,
        )
        kwargs.update(overrides)
        return multi_turn_failures(**kwargs)

    def test_multi_turn_failures_pass_a_healthy_run(self) -> None:
        records = self._records(self.PLAN)
        self.assertEqual(self._failures(records), [])
        summary = summarize_multi_turn(records)
        self.assertAlmostEqual(summary["accept_length"], 3.0)
        self.assertEqual(summary["ttft_s_last_turn_mean"], 4.0)
        self.assertEqual(summary["ttft_s_p50"], 3.0)

    def test_multi_turn_failures_catch_each_failure_mode(self) -> None:
        def broken(index: int, **fields) -> List[Dict[str, Any]]:
            records = self._records(self.PLAN)
            records[index].update(fields)
            return records

        last = len(self._records(self.PLAN)) - 1
        cases = {
            "short reply": (broken(3, completion_tokens=63), {}),
            "truncated stream": (broken(3, output_ids=[1] * 8), {}),
            "stopped on eos": (broken(3, finish_reason="stop"), {}),
            "prompt mismatch": (broken(3, prompt_tokens=10), {}),
            "spec did not run": (broken(3, spec_verify_ct=0), {}),
            "first turn hit": (broken(0, cached_tokens=PAGE_SIZE), {}),
            "returning turn missed": (broken(3, cached_tokens=4096), {}),
            "wrong secret": (broken(last, text=" 12345."), {}),
            "no secret": (broken(last, text=" I do not know."), {}),
            "missing turn": (self._records(self.PLAN)[:-1], {}),
            "stopped early": (self._records(self.PLAN), {"error": "turn 2: boom"}),
            "no host reload": (
                [dict(r, host_cached_tokens=0) for r in self._records(self.PLAN)],
                {},
            ),
            "low hit rate": (self._records(self.PLAN), {"min_hit_rate": 0.99}),
            "no accepted drafts": (
                [dict(r, spec_verify_ct=63) for r in self._records(self.PLAN)],
                {},
            ),
            "retraction": (
                self._records(self.PLAN),
                {"counter_deltas": {"decode sglang:num_retracted_requests_total": 1.0}},
            ),
            "no transfer timing": (
                self._records(self.PLAN),
                {"transfer_observations": 0},
            ),
        }
        for name, (records, overrides) in cases.items():
            with self.subTest(name):
                self.assertTrue(self._failures(records, **overrides))
        # Host reloads are only required when the plan outgrows the device pool.
        no_host = [dict(r, host_cached_tokens=0) for r in self._records(self.PLAN)]
        self.assertEqual(self._failures(no_host, require_host_hits=False), [])

    def _server_info(self, role: str) -> Dict[str, Any]:
        info = {
            "disaggregation_mode": role,
            "disaggregation_transfer_backend": "raiden",
            "tp_size": TP_SIZE,
            "ep_size": EP_SIZE,
            "page_size": PAGE_SIZE,
            "context_length": CONTEXT_LEN,
            "enable_metrics": True,
            "kv_cache_dtype": "mxint8",
            "speculative_draft_kv_cache_dtype": "bf16",
            "speculative_algorithm": "EAGLE3",
            "max_total_num_tokens": MAX_TOTAL_TOKENS,
            "enable_hierarchical_cache": role == "prefill",
            "disaggregation_decode_enable_radix_cache": False,
        }
        if role == "prefill":
            info.update(
                hicache_io_backend="raiden",
                hicache_mem_layout="page_first_fused",
                hicache_write_policy="write_through",
                disable_radix_cache=False,
            )
        return info

    def test_engine_configuration_failures(self) -> None:
        prefill, decode = self._server_info("prefill"), self._server_info("decode")
        self.assertEqual(engine_configuration_failures(prefill, decode, "mxint8"), [])
        cases = {
            "target dtype": (dict(prefill, kv_cache_dtype="bfloat16"), decode),
            "draft dtype": (
                prefill,
                dict(decode, speculative_draft_kv_cache_dtype="mxint8"),
            ),
            "page size": (dict(prefill, page_size=64), decode),
            "two hosts": (dict(prefill, tp_size=16, ep_size=16), decode),
            "small pool": (prefill, dict(decode, max_total_num_tokens=65536)),
            "no metrics": (dict(prefill, enable_metrics=False), decode),
            "prefill hicache off": (
                dict(prefill, enable_hierarchical_cache=False),
                decode,
            ),
            "decode hicache on": (
                prefill,
                dict(decode, enable_hierarchical_cache=True),
            ),
            "decode radix on": (
                prefill,
                dict(decode, disaggregation_decode_enable_radix_cache=True),
            ),
            "incremental stream": (
                prefill,
                dict(decode, incremental_streaming_output=True),
            ),
            "swapped roles": (decode, prefill),
        }
        for name, (p, d) in cases.items():
            with self.subTest(name):
                self.assertTrue(engine_configuration_failures(p, d, "mxint8"))
        self.assertEqual(
            engine_configuration_failures(
                dict(prefill, kv_cache_dtype="bf16"),
                dict(decode, kv_cache_dtype="bfloat16"),
                "bfloat16",
            ),
            [],
        )

    @unittest.skipIf(SGLANG_IMPORT_ERROR is not None, "needs the sglang test package")
    def test_offload_probe_fits_the_context_and_the_host_pool(self) -> None:
        self.assertLessEqual(
            DEFAULT_OFFLOAD_PROBE_TOKENS + DECODE_HEADROOM_TOKENS, CONTEXT_LEN
        )
        plan = plan_eviction(
            device_pool_tokens=MAX_TOTAL_TOKENS,
            probe_tokens=DEFAULT_OFFLOAD_PROBE_TOKENS,
            hicache_ratio=float(HICACHE_RATIO),
        )
        self.assertEqual(plan.num_fillers, 3)
        self.assertLessEqual(plan.backed_up_tokens, plan.host_pool_tokens)

    @unittest.skipIf(SGLANG_IMPORT_ERROR is not None, "needs the sglang test package")
    def test_device_pool_engages_the_precompile_deadlock_guard(self) -> None:
        from unittest import mock

        from sglang_tpu.env_override import apply_small_kv_pool_vmem_prefetch_override

        # GPT-OSS-120B has 36 hidden layers. The guard must fire at this pool.
        with mock.patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--existing=1"}):
            apply_small_kv_pool_vmem_prefetch_override(
                max_total_tokens=MAX_TOTAL_TOKENS, num_hidden_layers=36
            )
            flags = os.environ["LIBTPU_INIT_ARGS"].split()
        self.assertIn("--existing=1", flags)
        self.assertIn("--xla_gf_vmem_max_outstanding_prefetches=1", flags)
        self.assertEqual(MAX_TOTAL_TOKENS % PAGE_SIZE, 0)

    @unittest.skipIf(SGLANG_IMPORT_ERROR is not None, "needs the sglang test package")
    def test_gates_build(self) -> None:
        from sglang.test.tpu.evals.model_config import build_gates

        config = build_model_config(
            "http://prefill:30000", "http://decode:30000", env={}
        )
        gates = build_gates(config)
        self.assertEqual(
            [g.name for g in gates],
            ["gsm8k", "kv-offload-round-trip", "multi-turn-long-context"],
        )
        self.assertEqual(
            gates[0].spec_accept_length_threshold, SPEC_ACCEPT_LENGTH_THRESHOLD
        )
        self.assertTrue(all(callable(g.runner) for g in gates[1:]))
        self.assertIn("mxint8KV", config["variant"])

    def test_operator_overrides_reach_the_run_but_not_the_cpu_checks(self) -> None:
        # The nightly runs the integration case under PD_KV_CACHE_DTYPE=bfloat16;
        # the run must see it, and the CPU checks (which pass env={}) must not.
        from unittest import mock

        overrides = {
            KV_CACHE_DTYPE_ENV: "bfloat16",
            "MT_SESSIONS": "2",
            "MT_TURNS": "4",
        }
        with mock.patch.dict(os.environ, overrides):
            self.assertEqual(resolve_kv_cache_dtype(), "bfloat16")
            self.assertEqual(default_multi_turn_plan()[:2], (2, 4))
            self.assertIn("bfloat16KV", build_model_config("p", "d")["variant"])
            self.assertEqual(resolve_kv_cache_dtype({}), "mxint8")
            self.assertEqual(
                default_multi_turn_plan(env={})[:2],
                (DEFAULT_MT_SESSIONS, DEFAULT_MT_TURNS),
            )
            self.assertIn("mxint8KV", build_model_config("p", "d", env={})["variant"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-host-per-engine GPT-OSS P/D MXINT8 KV multi-turn test"
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
    # Run only the selected case: the integration job must not also run the
    # CPU checks under its env overrides.
    test_case = (
        TestPdSingleHostMxint8MultiTurnGptOss120bIntegration
        if args.mode == "integration"
        else TestPdSingleHostMxint8MultiTurnGptOss120bConfiguration
    )
    unittest.main(
        defaultTest=test_case.__name__,
        argv=[sys.argv[0], *extra_args],
        verbosity=2,
    )
