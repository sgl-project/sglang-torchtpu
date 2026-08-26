"""P/D disaggregation x prefix caching x KV offload integration test (issue #376).

Topology: a multihost PREFILL engine and a multihost DECODE engine, each spanning
its own TPU slice, connected by the PD mini load balancer. KV offload runs on the
prefill engine through the raiden HiCache io backend. The CI JobSet
(.github/tpu-ci/jobset-tpu-pd.yaml) runs this file on every pod of both slices;
roles are dispatched from the CI_PD_ROLE env plus the slice-local TPU worker env.

What each verification proves:
  * quality:    gsm8k through the LB must clear the model's accuracy floor,
                and the cached fraction of its prefill tokens must dominate
                (>= 0.5), so the score gates the cached prefill path rather
                than cold traffic.
  * cache hit:  an immediate resend of a long prompt must hit the device radix
                cache (cached_tokens_details.device).
  * offload:    after churn evicts the device pool, a resend must reload the
                prefix from the host tier (cached_tokens_details.host > 0), and
                the prefill log must show the raiden transfer engine came up.
  * KV transfer: every probe goes prefill -> decode across slices; the three
                cached passes (device hit, host reload, post-reload hit) run
                the same extend shape and must produce identical greedy text
                on a forced verbatim copy task -- a difference means stale or
                corrupt KV in the host offload round trip. The cold pass runs
                a different extend shape and is held to liveness only.

Roles (rank = slice-local TPU_WORKER_ID):
  prefill rank 0   launches the prefill engine, waits for decode, starts the
                   mini-LB, runs the probes + gsm8k -- its verdict gates the run.
  prefill rank >0  worker shard of the prefill engine.
  decode any rank  launches/joins the decode engine, serves until the leader
                   tears down.

Run one model profile per invocation on hardware:

  MODEL_PROFILE=gpt_oss_120b CI_PD_ROLE=prefill python3 \
      test/registered/tpu/integration/test_pd_prefix_caching_kv_offload.py

Without CI_PD_ROLE the file runs only its host-side configuration tests.
Model paths, pool sizes, churn workload, and gsm8k floors are env-overridable
(see the *Config structs below).
"""

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

import msgspec

SGLANG_IMPORT_ERROR = None
try:
    from sglang.test.ci.ci_register import register_tpu_ci
    from sglang.test.test_utils import dump_metric
except ModuleNotFoundError as exc:
    # Keep the configuration tests runnable in a source-only checkout; the
    # integration cases need the full SGLang TPU environment anyway.
    SGLANG_IMPORT_ERROR = exc

    def register_tpu_ci(*_args, **_kwargs):
        return None

    def dump_metric(*_args, **_kwargs):
        return None


# One invocation runs ONE model profile: two engines (weight load + compile),
# the cache probe with device-pool churn, and gsm8k.
register_tpu_ci(est_time=10800, suite="tpu-integration")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")

E = os.environ.get

PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LB_PORT = int(E("LB_PORT", "8000"))
BOOTSTRAP_PORT = int(E("BOOTSTRAP_PORT", "8998"))
WAIT_SECS = int(E("WAIT_SECS", "3600"))
KEEP_UP = E("KEEP_UP", "0") == "1"
PYTHON = E("PYTHON", "python3")

ROLE_PREFILL = "prefill"
ROLE_DECODE = "decode"

# Emitted by hardware_backend/tpu/raiden_transfer.py when the raiden host pool
# and DMA engine finish initializing; its absence means offload silently never
# engaged even if the probe below were to pass on the device tier alone.
RAIDEN_READY_MARKER = "Raiden KV transfer ready"

PREFILL_LOG = "/tmp/pd_prefill.log"
DECODE_LOG = "/tmp/pd_decode.log"
LB_LOG = "/tmp/pd_lb.log"


class ModelProfile(msgspec.Struct, frozen=True, kw_only=True):
    key: str
    display_name: str
    served_name: str
    default_model_path: str
    model_env: str
    page_size: int
    mem_fraction_prefill: str
    mem_fraction_decode: str
    quantization: str
    kv_cache_dtype: str
    # None sizes the KV pool from mem_fraction instead of an explicit token
    # count; device_pool_tokens_hint then feeds the churn workload sizing.
    max_total_tokens: Optional[int]
    device_pool_tokens_hint: int
    gsm8k_threshold: float
    # Appended after the launcher's default so it wins (argparse keeps the
    # last occurrence). The native "tpu" backend requires the fused MHA pool
    # and rejects GPT-OSS's hybrid SWAKVPool; raiden transfers both.
    disagg_transfer_backend: str = "raiden"
    extra_server_args: Tuple[str, ...] = ()
    extra_launch_env: Dict[str, str] = {}
    reasoning_effort: Optional[str] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None


MODEL_PROFILES = {
    "gpt_oss_120b": ModelProfile(
        key="gpt_oss_120b",
        display_name="GPT-OSS-120B",
        served_name="gpt-oss-120b",
        default_model_path="/weights/gpt-oss-120b",
        model_env="GPT_OSS_MODEL",
        page_size=64,
        mem_fraction_prefill="0.35",
        mem_fraction_decode="0.35",
        # GPT-OSS ships MXFP4 quantization_config; forcing fp8 would re-quantize.
        quantization="",
        kv_cache_dtype="fp8_e4m3",
        # The pool size is pinched from both sides on this model: below ~605k
        # tokens the extend-graph precompile hangs (issue #363), and with no
        # explicit bound the mem_fraction-sized pool is materialized unaliased
        # in the compiled graph and its HLO temporaries exceed HBM. 786k is
        # validated clean on both counts.
        max_total_tokens=786_432,
        device_pool_tokens_hint=786_432,
        # Measured 0.98 (1p1d, 2026-08-15) and 0.97 (2p2d, run 32908499644),
        # both over 100 examples; binomial sigma there is ~1.5 points, so 0.90
        # sits several sigma below measurement while still failing on a real
        # quality regression rather than only on catastrophic KV corruption.
        gsm8k_threshold=0.90,
        extra_server_args=("--reasoning-parser", "gpt-oss"),
        # The SparseCore ragged-gather MoE path cannot be lowered for GPT-OSS
        # (dynamic token dim in the extend graph).
        extra_launch_env={"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"},
        reasoning_effort="low",
    ),
    "qwen3_coder_480b": ModelProfile(
        key="qwen3_coder_480b",
        display_name="Qwen3-Coder-480B",
        served_name="qwen3-coder-480b",
        default_model_path="/weights/Qwen3-Coder-480B-A35B-Instruct-FP8",
        model_env="QWEN3_CODER_MODEL",
        page_size=64,
        mem_fraction_prefill="0.6",
        mem_fraction_decode="0.6",
        quantization="fp8",
        kv_cache_dtype="fp8_e4m3",
        # Small device pool so a short churn pass forces host offload.
        max_total_tokens=32768,
        device_pool_tokens_hint=32768,
        gsm8k_threshold=0.80,
        extra_server_args=("--tpu-fp8-runtime", "channelwise"),
        # The SparseCore MoE path halts this model's 2-host tp16 bring-up
        # (E0200, SC sequencers halted before warmup); the resharding test's
        # qwen profile carries the same setting.
        extra_launch_env={"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"},
        chat_template_kwargs={"enable_thinking": False},
    ),
}


class CacheProbeConfig(msgspec.Struct, frozen=True, kw_only=True):
    page_size: int
    device_pool_tokens: int
    target_words: int
    filler_words: int
    # None sizes the churn at runtime from the first filler's measured token
    # count (churn_fillers_needed); an explicit value overrides that.
    num_fillers: Optional[int]
    max_new_tokens: int

    @classmethod
    def from_env(
        cls, profile: ModelProfile, env: Optional[Mapping[str, str]] = None
    ) -> "CacheProbeConfig":
        values = os.environ if env is None else env
        device_tokens = int(
            values.get(
                "CACHE_PROBE_DEVICE_TOKENS", str(profile.device_pool_tokens_hint)
            )
        )
        fillers_raw = values.get("CACHE_PROBE_NUM_FILLERS")
        return cls(
            page_size=profile.page_size,
            device_pool_tokens=device_tokens,
            target_words=int(values.get("CACHE_PROBE_TARGET_WORDS", "2048")),
            filler_words=int(values.get("CACHE_PROBE_FILLER_WORDS", "2048")),
            num_fillers=None if fillers_raw is None else int(fillers_raw),
            # Room for all four copied item lines plus any preamble.
            max_new_tokens=int(values.get("CACHE_PROBE_MAX_NEW_TOKENS", "64")),
        )


def churn_fillers_needed(
    *, measured_filler_tokens: int, device_pool_tokens: int
) -> int:
    """Fillers that push ~2x the device pool through the cache.

    Sized from a measured filler's token count, not a tokens-per-word guess: a
    guess that runs short leaves the churn below pool capacity, no eviction
    ever fires, and the host-reload leg silently tests nothing (the 0.8x
    undershoot observed in run 32927137148). The 2x margin also absorbs slack
    between the configured pool bound and the allocator's effective capacity.
    """
    if measured_filler_tokens <= 0:
        raise AssertionError(
            f"measured filler prompt_tokens={measured_filler_tokens}; cannot "
            "size the churn"
        )
    return (2 * device_pool_tokens) // measured_filler_tokens + 1


class Gsm8kConfig(msgspec.Struct, frozen=True, kw_only=True):
    num_examples: int
    num_threads: int
    max_tokens: int
    threshold: float

    @classmethod
    def from_env(
        cls, profile: ModelProfile, env: Optional[Mapping[str, str]] = None
    ) -> "Gsm8kConfig":
        values = os.environ if env is None else env
        suffix = profile.key.upper()
        threshold_raw = values.get(
            f"GSM8K_THRESHOLD_{suffix}",
            values.get("GSM8K_THRESHOLD", str(profile.gsm8k_threshold)),
        )
        threshold = float(threshold_raw)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"GSM8K threshold must be in [0, 1], got {threshold}")
        return cls(
            num_examples=int(values.get("GSM8K_NUM_EXAMPLES", "200")),
            num_threads=int(values.get("GSM8K_NUM_THREADS", "8")),
            max_tokens=int(values.get("GSM8K_MAX_TOKENS", "2048")),
            threshold=threshold,
        )


class PodTopology(msgspec.Struct, frozen=True, kw_only=True):
    role: str
    group_rank: int
    group_hosts: int
    group_hostnames: Tuple[str, ...]

    @property
    def is_leader(self) -> bool:
        return self.role == ROLE_PREFILL and self.group_rank == 0

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "PodTopology":
        values = os.environ if env is None else env
        role = values.get("CI_PD_ROLE", "")
        if role not in (ROLE_PREFILL, ROLE_DECODE):
            raise ValueError(
                f"CI_PD_ROLE must be {ROLE_PREFILL!r} or {ROLE_DECODE!r}, got {role!r}"
            )
        hostnames = tuple(
            h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h
        )
        if not hostnames:
            # GKE injects TPU_WORKER_* only on multihost slices; a single-host
            # engine group is just this pod.
            hostnames = (socket.gethostname(),)
        return cls(
            role=role,
            group_rank=int(values.get("TPU_WORKER_ID", "0")),
            group_hosts=len(hostnames),
            group_hostnames=hostnames,
        )


def peer_group_hostname(hostname: str, from_role: str, to_role: str) -> str:
    """Map a pod hostname of one replicated job onto its peer job's pod.

    JobSet pod hostnames are ``<jobset>-<replicatedJob>-<jobIdx>-<podIdx>`` (the
    DNS suffix, when present, is the jobset name). The replicated jobs in
    jobset-tpu-pd.yaml are named ``prefill``/``decode`` and the jobset name never
    contains ``-prefill-``/``-decode-`` (render_jobset.py rejects such run
    names), so the role segment is the unique match.
    """
    needle, replacement = f"-{from_role}-", f"-{to_role}-"
    if needle not in hostname:
        raise ValueError(f"hostname {hostname!r} has no {needle!r} segment")
    return hostname.replace(needle, replacement)


def resolve_model_path(
    profile: ModelProfile, env: Optional[Mapping[str, str]] = None
) -> str:
    values = os.environ if env is None else env
    return (
        values.get(profile.model_env)
        or values.get("MODEL")
        or profile.default_model_path
    )


def validate_model_source(profile: ModelProfile, model_path: str) -> None:
    """Fail clearly when CI points at an unprovisioned local checkpoint."""
    if not os.path.isabs(model_path):
        return  # HF repo ids resolve remotely
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"{profile.display_name} checkpoint incomplete or missing at "
            f"{model_path}; provision it on the weights volume or set "
            f"{profile.model_env}"
        )


def hicache_server_args(
    profile: ModelProfile, env: Optional[Mapping[str, str]] = None
) -> str:
    values = os.environ if env is None else env
    args = [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        values.get("HICACHE_RATIO", "2"),
        "--hicache-io-backend",
        "raiden",
        "--hicache-mem-layout",
        "page_first_fused",
        "--enable-cache-report",
    ]
    if profile.max_total_tokens is not None:
        args += ["--max-total-tokens", str(profile.max_total_tokens)]
    return " ".join(args)


def _common_launch_env(
    profile: ModelProfile, group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    values = os.environ if env is None else env
    tp_size = int(values.get("TP_SIZE", str(8 * group_hosts)))
    return {
        "MODEL": resolve_model_path(profile, values),
        "TP_SIZE": str(tp_size),
        "EP_SIZE": values.get("EP_SIZE", str(tp_size)),
        "PAGE_SIZE": str(profile.page_size),
        "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
        "QUANTIZATION": profile.quantization,
        "KV_CACHE_DTYPE": profile.kv_cache_dtype,
        "DISABLE_RADIX_CACHE": "0",
        "ENABLE_TORCH_COMPILE": values.get("ENABLE_TORCH_COMPILE", "1"),
        "PYTHON": PYTHON,
        # The raiden connector's per-rank control ports start at this base;
        # its default (30000) collides with rank 0's HTTP server port.
        "SGLANG_RAIDEN_KV_PORT": values.get("SGLANG_RAIDEN_KV_PORT", "31000"),
        **profile.extra_launch_env,
    }


def build_prefill_env(
    profile: ModelProfile, group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    values = os.environ if env is None else env
    served = ("--served-model-name", profile.served_name)
    return {
        **_common_launch_env(profile, group_hosts, values),
        "MEM_FRACTION_STATIC": values.get(
            "MEM_FRACTION_STATIC", profile.mem_fraction_prefill
        ),
        "PORT": str(PREFILL_PORT),
        "BOOTSTRAP_PORT": str(BOOTSTRAP_PORT),
        "EXTRA_ARGS": " ".join(
            (
                hicache_server_args(profile, values),
                *served,
                *profile.extra_server_args,
                "--disaggregation-transfer-backend",
                profile.disagg_transfer_backend,
                # A disagg engine's warmup generation cannot complete without
                # its peer + the LB pairing them, so warmup just times out.
                "--skip-server-warmup",
                # The default TPU sampling backend's distributed top-k crashes
                # under torch.compile at tp>1 (dynamo cannot construct its
                # msgspec metadata struct in-graph); pin the torch sampler
                # until that is fixed upstream.
                "--sampling-backend",
                "pytorch",
                values.get("EXTRA_PREFILL_ARGS", ""),
            )
        ).strip(),
    }


def build_decode_env(
    profile: ModelProfile, group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    values = os.environ if env is None else env
    decode_args = ["--served-model-name", profile.served_name]
    if profile.max_total_tokens is not None:
        # The decode pool must be bounded too, or its auto-derived size OOMs
        # at precompile (pool materialized unaliased as compile args+outputs).
        decode_args += ["--max-total-tokens", str(profile.max_total_tokens)]
    return {
        **_common_launch_env(profile, group_hosts, values),
        "MEM_FRACTION_STATIC": values.get(
            "MEM_FRACTION_STATIC", profile.mem_fraction_decode
        ),
        "PORT": str(DECODE_PORT),
        "EXTRA_ARGS": " ".join(
            (
                *decode_args,
                *profile.extra_server_args,
                "--disaggregation-transfer-backend",
                profile.disagg_transfer_backend,
                "--skip-server-warmup",
                "--sampling-backend",
                "pytorch",
                values.get("EXTRA_DECODE_ARGS", ""),
            )
        ).strip(),
    }


def select_profile(raw: str) -> ModelProfile:
    key = raw.strip()
    if key not in MODEL_PROFILES:
        allowed = ", ".join(MODEL_PROFILES)
        raise ValueError(
            f"MODEL_PROFILE must name exactly one of: {allowed} (got {key!r}); "
            "P/D runs launch one engine pair per invocation"
        )
    return MODEL_PROFILES[key]


# ------------------------------ probe workload -------------------------------

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


def make_target_prompt(nwords: int) -> Tuple[str, Tuple[int, ...]]:
    """A numbered item list ending in a verbatim copy instruction.

    Returns (prompt, queried item numbers). The greedy continuation is forced:
    the model must repeat four distinctive lines drawn from across the list,
    read through full attention from the queried positions' KV. Kernel-level
    epsilon differences between cache paths cannot flip a forced copy, so any
    output difference between passes means the KV values themselves changed.
    (The repetitive filler prompts have near-tie logits at every step, where
    any epsilon flips the text -- useless as a correctness oracle.)
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
    req = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=json.dumps(
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                },
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        payload = json.loads(resp.read())
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
    with urllib.request.urlopen(url.rstrip("/") + "/flush_cache", timeout=60) as r:
        r.read()


def validate_cache_probe(
    cold: GenerationRecord,
    device_hit: GenerationRecord,
    host_reload: GenerationRecord,
    post_reload_hit: GenerationRecord,
    page_size: int,
) -> int:
    """The device-hit -> churn -> host-reload oracle; returns expected tokens.

    The three cached passes run the same extend shape over the same stored
    prefix KV, so their greedy text must be identical: device-hit vs
    host-reload catches a stale or corrupt host round trip, and host-reload
    vs post-reload-hit separates the two (a reload that raced its own
    host->device DMA reads clean once the pages settle; corrupt values stay
    wrong). Every pass crosses the P/D transfer.

    The cold pass is held to cache-state and liveness checks only: its
    prefill runs a different extend shape and consumes the prefix at compute
    dtype rather than KV-cache dtype, so its logits differ at epsilon scale
    and even the forced-copy prompt's preamble can break ties differently.
    """
    if cold.prompt_tokens < 2 * page_size:
        raise AssertionError(f"target prompt too short: {cold.prompt_tokens} tokens")
    if cold.device_cached_tokens or cold.host_cached_tokens:
        raise AssertionError(
            "cold request unexpectedly hit cache: "
            f"device={cold.device_cached_tokens} host={cold.host_cached_tokens}"
        )
    if not cold.text.strip() or cold.completion_tokens <= 0:
        raise AssertionError(
            "cold request produced no decode output -- P/D KV transfer broken"
        )

    expected = max(page_size, (cold.prompt_tokens // page_size - 1) * page_size)
    if device_hit.device_cached_tokens < expected:
        raise AssertionError(
            "immediate resend missed the device prefix cache: "
            f"device={device_hit.device_cached_tokens} expected>={expected}"
        )
    if device_hit.host_cached_tokens:
        raise AssertionError(
            f"immediate resend loaded from host: {device_hit.host_cached_tokens}"
        )
    if not device_hit.text.strip() or device_hit.completion_tokens <= 0:
        raise AssertionError("device-hit request produced no decode output")

    if host_reload.host_cached_tokens <= 0:
        raise AssertionError(
            "target was not reloaded from the host tier after churn -- "
            "KV offload never engaged"
        )
    if host_reload.device_cached_tokens + host_reload.host_cached_tokens < expected:
        raise AssertionError(
            "host reload did not recover the cached prefix: "
            f"device={host_reload.device_cached_tokens} "
            f"host={host_reload.host_cached_tokens} expected>={expected}"
        )
    if host_reload.text != device_hit.text:
        if post_reload_hit.text == device_hit.text:
            raise AssertionError(
                "host-reloaded copy output diverged from the device-hit pass "
                "but the post-reload resend matches -- the reload raced its "
                "own host->device transfer (no-stale gate not honored)"
            )
        raise AssertionError(
            "host-reloaded copy output diverged from the device-hit pass -- "
            "the offloaded KV pages did not survive the host round trip"
        )
    if post_reload_hit.text != device_hit.text:
        raise AssertionError(
            "post-reload resend diverged from the device-hit pass -- the "
            "reloaded device-resident pages hold different KV values"
        )
    return expected


def run_cache_probe(
    base_url: str, flush_url: str, profile: ModelProfile, config: CacheProbeConfig
) -> None:
    _flush_cache(flush_url)
    target, queried = make_target_prompt(config.target_words)
    print(f"cache probe target queries items {queried}", flush=True)
    cold = _generate(base_url, target, config.max_new_tokens)
    device_hit = _generate(base_url, target, config.max_new_tokens)

    first_filler = _generate(base_url, make_prompt(1000, config.filler_words), 1)
    num_fillers = config.num_fillers
    if num_fillers is None:
        num_fillers = churn_fillers_needed(
            measured_filler_tokens=first_filler.prompt_tokens,
            device_pool_tokens=config.device_pool_tokens,
        )
    print(
        f"churn: {num_fillers} fillers x {first_filler.prompt_tokens} tokens "
        f"against a {config.device_pool_tokens}-token device pool",
        flush=True,
    )
    for index in range(1, num_fillers):
        _generate(base_url, make_prompt(1000 + index, config.filler_words), 1)
    host_reload = _generate(base_url, target, config.max_new_tokens)
    # The reload leaves the prefix device-resident again; this pass rereads
    # the reloaded pages after any in-flight host->device transfers have
    # settled, separating a racing reload from corrupt reloaded values.
    post_reload_hit = _generate(base_url, target, config.max_new_tokens)

    # Emitted before validation so a failed oracle still carries every
    # generation for diagnosis.
    print(
        json.dumps(
            {
                "cache_probe": profile.key,
                "cold": msgspec.to_builtins(cold),
                "device_hit": msgspec.to_builtins(device_hit),
                "host_reload": msgspec.to_builtins(host_reload),
                "post_reload_hit": msgspec.to_builtins(post_reload_hit),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    expected = validate_cache_probe(
        cold=cold,
        device_hit=device_hit,
        host_reload=host_reload,
        post_reload_hit=post_reload_hit,
        page_size=config.page_size,
    )
    labels = {"model": profile.served_name, "test": "pd_prefix_caching_kv_offload"}
    dump_metric(
        "prefix_cache_device_hit_tokens", device_hit.device_cached_tokens, labels=labels
    )
    dump_metric(
        "kv_offload_host_reload_tokens", host_reload.host_cached_tokens, labels=labels
    )
    print(f"cache probe passed: expected_cached_tokens={expected}", flush=True)


def run_gsm8k(base_url: str, profile: ModelProfile, config: Gsm8kConfig) -> float:
    from sglang.test.run_eval import run_eval

    metrics = run_eval(
        SimpleNamespace(
            base_url=base_url.rstrip("/"),
            model=profile.served_name,
            eval_name="gsm8k",
            api="chat",
            num_examples=config.num_examples,
            num_threads=config.num_threads,
            num_shots=5,
            max_tokens=config.max_tokens,
            temperature=0.0,
            top_p=1.0,
            reasoning_effort=profile.reasoning_effort,
            chat_template_kwargs=profile.chat_template_kwargs,
            gsm8k_data_path=None,
        )
    )
    return float(metrics["score"])


# --------------------------- process/launch plumbing -------------------------


def _tee_command(argv: Tuple[str, ...], log_path: str) -> Tuple[str, ...]:
    """Wrap a command so its output reaches BOTH pod stdout and a log file.

    The CI log stream is the only view into the pods, so burying engine output
    in a file makes launch failures undiagnosable. pipefail keeps the wrapped
    command's exit code visible through the tee.
    """
    quoted = " ".join(f"'{a}'" for a in argv)
    return ("bash", "-c", f"set -o pipefail; {quoted} 2>&1 | tee -a '{log_path}'")


def _launch(script: str, extra_env: Dict[str, str], log_path: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(extra_env)
    return subprocess.Popen(
        _tee_command(("bash", os.path.join(BENCH, script)), log_path=log_path),
        env=env,
        preexec_fn=os.setsid,  # own pgid so teardown can killpg
        cwd=BENCH,
    )


def _kill(proc: Optional[subprocess.Popen]) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _start_host_watch() -> None:
    """[probe] Stream kernel kill evidence and host-memory headroom to stdout.

    A rank that dies without a Python traceback leaves nothing in the pod log;
    the kernel ring buffer (OOM kills, segfaults) and the memory slope are the
    only way to tell an OOM kill from a native abort. Privileged CI pods can
    read the host ring buffer.
    """
    script = (
        "(dmesg -w 2>/dev/null || echo '[hostwatch] dmesg unavailable') | "
        "grep --line-buffered -iE "
        "'out of memory|oom|killed process|segfault|hostwatch' | "
        "sed -u 's/^/[hostwatch:dmesg] /' & "
        "while :; do "
        "echo \"[hostwatch:mem] $(grep -E 'MemTotal|MemAvailable' /proc/meminfo"
        " | tr -s ' \\n' ' ')\"; "
        "for d in /proc/[0-9]*; do r=$(awk '{print $2}' $d/statm 2>/dev/null); "
        '[ -n "$r" ] && echo "$((r*4/1024))MB ${d##*/} $(cat $d/comm 2>/dev/null)"; '
        "done | sort -rn | head -3 | sed 's/^/[hostwatch:rss] /'; "
        "sleep 20; done"
    )
    subprocess.Popen(["bash", "-c", script], start_new_session=True)


def _http_ok(url: str, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:  # noqa: BLE001 - any transport error means "not healthy"
        return False


def _wait_http_200(url: str, secs: int, procs: Tuple = ()) -> None:
    deadline = time.time() + secs
    while time.time() < deadline:
        for p in procs:
            if p is not None and p.poll() is not None:
                raise RuntimeError(f"process for {url} exited rc={p.returncode}")
        if _http_ok(url):
            return
        time.sleep(10)
    raise TimeoutError(f"{url} not ready after {secs}s")


PREFILL_BATCH_RE = re.compile(
    r"Prefill batch.*?#new-token: (\d+), #cached-token: (\d+)"
)


def prefill_cache_coverage(log_text: str) -> Tuple[int, int, int]:
    """Sum (batches, new tokens, cached tokens) over scheduler prefill lines.

    The scheduler's per-batch accounting is the only per-request cache-hit
    signal that reaches the pod, so a traffic window's cache coverage is read
    from the engine log rather than from HTTP responses.
    """
    batches, new_tokens, cached_tokens = 0, 0, 0
    for match in PREFILL_BATCH_RE.finditer(log_text):
        batches += 1
        new_tokens += int(match.group(1))
        cached_tokens += int(match.group(2))
    return batches, new_tokens, cached_tokens


def _assert_raiden_engaged(log_path: str) -> None:
    with open(log_path, "rb") as f:
        log = f.read().decode(errors="replace")
    if RAIDEN_READY_MARKER not in log:
        raise AssertionError(
            f"{RAIDEN_READY_MARKER!r} never appeared in {log_path} -- the raiden "
            "offload engine did not initialize on the prefill engine"
        )


# ------------------------------- leader case ---------------------------------


class TestPdPrefixCachingKvOffload(unittest.TestCase):
    """Runs on the prefill leader only; its verdict gates the whole JobSet."""

    procs = []
    topo: PodTopology
    profile: ModelProfile

    @classmethod
    def setUpClass(cls) -> None:
        cls.topo = PodTopology.from_env()
        cls.profile = select_profile(E("MODEL_PROFILE", ""))
        model_path = resolve_model_path(cls.profile)
        validate_model_source(cls.profile, model_path)
        cls.cache_config = CacheProbeConfig.from_env(cls.profile)
        cls.gsm8k_config = Gsm8kConfig.from_env(cls.profile)
        cls.procs = []

        self_host = cls.topo.group_hostnames[0]
        decode_host = peer_group_hostname(
            self_host, from_role=ROLE_PREFILL, to_role=ROLE_DECODE
        )

        cls.procs.append(
            _launch(
                "launch_prefill.sh",
                build_prefill_env(cls.profile, cls.topo.group_hosts),
                PREFILL_LOG,
            )
        )
        _wait_http_200(
            f"http://127.0.0.1:{PREFILL_PORT}/health", WAIT_SECS, tuple(cls.procs)
        )
        _assert_raiden_engaged(PREFILL_LOG)
        _wait_http_200(f"http://{decode_host}:{DECODE_PORT}/health", WAIT_SECS)

        # The mini-LB injects the --prefill hostname into requests as the KV
        # bootstrap address the decode pods dial -- it must be this pod's
        # cluster-reachable name, never 127.0.0.1.
        cls.procs.append(
            subprocess.Popen(
                _tee_command(
                    (
                        PYTHON,
                        "-m",
                        "sglang_router.launch_router",
                        "--pd-disaggregation",
                        "--mini-lb",
                        "--prefill",
                        f"http://{self_host}:{PREFILL_PORT}",
                        "--decode",
                        f"http://{decode_host}:{DECODE_PORT}",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(LB_PORT),
                    ),
                    log_path=LB_LOG,
                ),
                preexec_fn=os.setsid,
            )
        )
        _wait_http_200(f"http://127.0.0.1:{LB_PORT}/health", 300, tuple(cls.procs))

    @classmethod
    def tearDownClass(cls) -> None:
        if KEEP_UP:
            print("KEEP_UP=1: leaving prefill + LB running")
            return
        for p in cls.procs:
            _kill(p)

    def test_01_cache_hit_and_host_offload(self) -> None:
        run_cache_probe(
            base_url=f"http://127.0.0.1:{LB_PORT}",
            flush_url=f"http://127.0.0.1:{PREFILL_PORT}",
            profile=self.profile,
            config=self.cache_config,
        )

    def test_02_gsm8k_quality(self) -> None:
        eval_log_offset = os.path.getsize(PREFILL_LOG)
        score = run_gsm8k(
            base_url=f"http://127.0.0.1:{LB_PORT}",
            profile=self.profile,
            config=self.gsm8k_config,
        )
        labels = {
            "model": self.profile.served_name,
            "test": "pd_prefix_caching_kv_offload",
        }
        dump_metric("gsm8k_score", score, labels=labels)
        print(f"gsm8k score={score:.4f} floor={self.gsm8k_config.threshold:.4f}")

        # The quality gate certifies the CACHED prefill path only if the eval
        # traffic actually hits the prefix cache: every request past the first
        # shares the few-shot prefix, so the cached fraction of its prefill
        # tokens must dominate. A regression that silently disables caching
        # would otherwise leave the score gating cold traffic only.
        with open(PREFILL_LOG, "rb") as f:
            f.seek(eval_log_offset)
            eval_log = f.read().decode(errors="replace")
        batches, new_tokens, cached_tokens = prefill_cache_coverage(eval_log)
        total = new_tokens + cached_tokens
        if total < 100 * self.gsm8k_config.num_examples:
            raise AssertionError(
                "gsm8k prefill accounting looks wrong: "
                f"{batches} batches / {total} prompt tokens for "
                f"{self.gsm8k_config.num_examples} examples -- the eval did "
                "not route through this prefill engine, or the scheduler's "
                "batch log format changed"
            )
        coverage = cached_tokens / total
        dump_metric("gsm8k_prefill_cached_token_fraction", coverage, labels=labels)
        print(
            f"gsm8k prefill cache coverage={coverage:.3f} "
            f"({cached_tokens}/{total} tokens over {batches} batches)"
        )
        self.assertGreaterEqual(
            coverage,
            0.5,
            "gsm8k traffic barely hit the prefix cache; its score does not "
            "gate the cached prefill path",
        )
        self.assertGreaterEqual(score, self.gsm8k_config.threshold)


# ------------------------------ worker serving -------------------------------


def _serve_role(topo: PodTopology, profile: ModelProfile) -> int:
    """Non-leader pods: serve this pod's engine role until the leader is done.

    No cross-slice shutdown channel exists, so workers watch the leader's
    prefill server: healthy once and then gone means the verdict is in. A pod's
    own engine dying early still gates -- the leader fails independently when a
    peer it needs is broken, and unambiguous local deaths return nonzero.
    """
    if topo.role == ROLE_DECODE:
        script, env, log = (
            "launch_decode.sh",
            build_decode_env(profile, topo.group_hosts),
            DECODE_LOG,
        )
        leader_host = peer_group_hostname(
            topo.group_hostnames[0], from_role=ROLE_DECODE, to_role=ROLE_PREFILL
        )
    else:
        script, env, log = (
            "launch_prefill.sh",
            build_prefill_env(profile, topo.group_hosts),
            PREFILL_LOG,
        )
        leader_host = topo.group_hostnames[0]

    proc = _launch(script, env, log)
    leader_url = f"http://{leader_host}:{PREFILL_PORT}/health"
    leader_seen, misses, waited = False, 0, 0
    while True:
        rc = proc.poll()
        if rc is not None:
            if leader_seen and not _http_ok(leader_url):
                print(f"[{topo.role}:{topo.group_rank}] exited during teardown")
                return 0
            return rc
        if _http_ok(leader_url):
            leader_seen, misses = True, 0
        elif leader_seen:
            misses += 1
            if misses >= 6:
                print(f"[{topo.role}:{topo.group_rank}] leader gone; stopping")
                _kill(proc)
                return 0
        elif waited > WAIT_SECS + 600:
            print(f"[{topo.role}:{topo.group_rank}] leader never healthy; stopping")
            _kill(proc)
            return 0
        time.sleep(10)
        waited += 10


# --------------------------- configuration tests -----------------------------


class TestPdIntegrationConfiguration(unittest.TestCase):
    """Host-side contract tests; runnable without TPU hardware."""

    def test_profile_selection_requires_exactly_one(self) -> None:
        self.assertIs(select_profile("gpt_oss_120b"), MODEL_PROFILES["gpt_oss_120b"])
        for bad in ("", "all", "gpt_oss_120b,qwen3_coder_480b", "unknown"):
            with self.assertRaisesRegex(ValueError, "MODEL_PROFILE"):
                select_profile(bad)

    def test_prefill_enables_prefix_cache_and_raiden_offload(self) -> None:
        for profile in MODEL_PROFILES.values():
            env = build_prefill_env(profile, 2, env={})
            self.assertEqual(env["DISABLE_RADIX_CACHE"], "0")
            self.assertEqual(env["TP_SIZE"], "16")
            self.assertIn("--enable-hierarchical-cache", env["EXTRA_ARGS"])
            self.assertIn("--hicache-io-backend raiden", env["EXTRA_ARGS"])
            self.assertIn("--hicache-mem-layout page_first_fused", env["EXTRA_ARGS"])
            self.assertIn("--enable-cache-report", env["EXTRA_ARGS"])

    def test_both_engines_use_the_raiden_transfer_backend(self) -> None:
        # The launchers hardcode the native "tpu" transfer backend, which
        # rejects GPT-OSS's hybrid SWA pool at init; the profiles must append
        # the raiden override on BOTH engines (a one-sided override would
        # desync the transfer handshake).
        for profile in MODEL_PROFILES.values():
            for env in (
                build_prefill_env(profile, 2, env={}),
                build_decode_env(profile, 2, env={}),
            ):
                self.assertIn(
                    "--disaggregation-transfer-backend raiden", env["EXTRA_ARGS"]
                )
                # Warmup deadlocks a lone disagg engine (no peer yet), and the
                # raiden control-port base must stay clear of the HTTP ports.
                self.assertIn("--skip-server-warmup", env["EXTRA_ARGS"])
                self.assertIn("--sampling-backend pytorch", env["EXTRA_ARGS"])
                raiden_base = int(env["SGLANG_RAIDEN_KV_PORT"])
                for port in (PREFILL_PORT, DECODE_PORT, LB_PORT, BOOTSTRAP_PORT):
                    self.assertNotIn(port, range(raiden_base, raiden_base + 4 * 16))

    def test_decode_has_no_hicache(self) -> None:
        for profile in MODEL_PROFILES.values():
            env = build_decode_env(profile, 2, env={})
            self.assertNotIn("hicache", env["EXTRA_ARGS"])

    def test_gpt_oss_pool_is_bounded_and_above_precompile_floor(self) -> None:
        # Regression guard, both directions observed on hardware: a pool below
        # ~605k tokens hangs the GPT-OSS extend-graph precompile (issue #363),
        # and an UNbounded pool at tp=16 blows HBM with compile temporaries
        # (150G > 94.75G). Both engines must carry the same explicit bound.
        profile = MODEL_PROFILES["gpt_oss_120b"]
        self.assertIsNotNone(profile.max_total_tokens)
        self.assertGreaterEqual(profile.max_total_tokens, 605_000)
        bound = f"--max-total-tokens {profile.max_total_tokens}"
        self.assertIn(bound, build_prefill_env(profile, 2, env={})["EXTRA_ARGS"])
        self.assertIn(bound, build_decode_env(profile, 2, env={})["EXTRA_ARGS"])

    def test_gpt_oss_does_not_force_quantization(self) -> None:
        # MXFP4 ships its own quantization_config; QUANTIZATION must be
        # explicitly empty or launch_server.sh applies its fp8 default.
        env = build_prefill_env(MODEL_PROFILES["gpt_oss_120b"], 2, env={})
        self.assertEqual(env["QUANTIZATION"], "")

    def test_churn_overflows_the_device_pool(self) -> None:
        # The churn is sized from a measured filler length; whatever that
        # measurement comes back as, the computed filler count must clear the
        # pool with margin. The 1766-token case replays the observed filler
        # length whose word-count estimate undershot the pool (run 32927137148).
        for profile in MODEL_PROFILES.values():
            config = CacheProbeConfig.from_env(profile, env={})
            self.assertIsNone(config.num_fillers)
            for measured in (1766, 2048, 2662):
                fillers = churn_fillers_needed(
                    measured_filler_tokens=measured,
                    device_pool_tokens=config.device_pool_tokens,
                )
                self.assertGreaterEqual(
                    fillers * measured,
                    2 * config.device_pool_tokens,
                    f"{profile.key}: churn of {fillers} x {measured}-token "
                    f"fillers cannot evict a {config.device_pool_tokens}-token "
                    "device pool",
                )

    def test_target_prompt_queries_resolve_to_unique_lines(self) -> None:
        # The copy oracle only forces the continuation if every queried item
        # exists exactly once, and only spans the prefix if the queries spread
        # across it.
        prompt, queried = make_target_prompt(2048)
        lines = prompt.split("\n")
        positions = []
        for number in queried:
            matches = [
                i for i, l in enumerate(lines) if l.startswith(f"item {number}: ")
            ]
            self.assertEqual(len(matches), 1, f"item {number}")
            positions.append(matches[0])
        self.assertLess(positions[0], len(lines) // 4)
        self.assertGreater(positions[-1], (3 * len(lines)) // 4)

    def test_explicit_filler_count_overrides_calibration(self) -> None:
        config = CacheProbeConfig.from_env(
            MODEL_PROFILES["gpt_oss_120b"], env={"CACHE_PROBE_NUM_FILLERS": "7"}
        )
        self.assertEqual(config.num_fillers, 7)

    def test_peer_hostname_maps_only_the_role_segment(self) -> None:
        self.assertEqual(
            peer_group_hostname(
                "tpu-ci-2p2d-kvoff-x7f2-prefill-0-1",
                from_role=ROLE_PREFILL,
                to_role=ROLE_DECODE,
            ),
            "tpu-ci-2p2d-kvoff-x7f2-decode-0-1",
        )
        with self.assertRaisesRegex(ValueError, "no '-prefill-' segment"):
            peer_group_hostname(
                "tpu-ci-2p2d-kvoff-x7f2-decode-0-0",
                from_role=ROLE_PREFILL,
                to_role=ROLE_DECODE,
            )

    def test_launcher_directory_resolves_from_this_file(self) -> None:
        # Regression guard: the repo root is derived by walking up from this
        # file's directory, and a miscounted level made the leader launch a
        # nonexistent benchmark path on its first hardware run. Any checkout
        # this file lives in has the launchers, so their absence means the
        # walk is wrong.
        self.assertTrue(
            os.path.isfile(os.path.join(BENCH, "launch_prefill.sh")),
            f"BENCH does not point at benchmark/tpu: {BENCH}",
        )

    def test_leader_is_prefill_rank_zero(self) -> None:
        env = {
            "CI_PD_ROLE": ROLE_PREFILL,
            "TPU_WORKER_ID": "0",
            "TPU_WORKER_HOSTNAMES": "a-prefill-0-0,a-prefill-0-1",
        }
        self.assertTrue(PodTopology.from_env(env).is_leader)
        self.assertFalse(PodTopology.from_env({**env, "TPU_WORKER_ID": "1"}).is_leader)
        self.assertFalse(
            PodTopology.from_env(
                {
                    "CI_PD_ROLE": ROLE_DECODE,
                    "TPU_WORKER_ID": "0",
                    "TPU_WORKER_HOSTNAMES": "a-decode-0-0,a-decode-0-1",
                }
            ).is_leader
        )

    def test_prefill_cache_coverage_reads_scheduler_batch_lines(self) -> None:
        # The regex mirrors the scheduler's batch-accounting line; decode
        # batches carry no cache counters and must not match.
        log = "\n".join(
            (
                "[2026-08-25 23:21:41 TP0 EP0] Prefill batch, #new-seq: 1, "
                "#new-token: 2112, #cached-token: 0, full token usage: 0.00,",
                "[2026-08-25 23:21:42 TP0 EP0] Prefill batch, #new-seq: 1, "
                "#new-token: 64, #cached-token: 2048, full token usage: 0.00,",
                "[2026-08-25 23:21:43 TP0 EP0] Decode batch, #running-req: 1,",
            )
        )
        self.assertEqual(prefill_cache_coverage(log), (2, 2176, 2048))

    def test_model_specific_gsm8k_threshold_wins(self) -> None:
        config = Gsm8kConfig.from_env(
            MODEL_PROFILES["qwen3_coder_480b"],
            env={"GSM8K_THRESHOLD": "0.50", "GSM8K_THRESHOLD_QWEN3_CODER_480B": "0.91"},
        )
        self.assertEqual(config.threshold, 0.91)

    def test_local_model_path_is_validated(self) -> None:
        profile = MODEL_PROFILES["qwen3_coder_480b"]
        with self.assertRaisesRegex(FileNotFoundError, profile.model_env):
            validate_model_source(profile, "/weights/definitely-missing-480b")
        validate_model_source(profile, "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8")


if __name__ == "__main__":
    if not E("CI_PD_ROLE"):
        sys.argv = [sys.argv[0], "TestPdIntegrationConfiguration"]
        unittest.main(verbosity=2)
    topology = PodTopology.from_env()
    _start_host_watch()
    if not topology.is_leader:
        sys.exit(_serve_role(topology, select_profile(E("MODEL_PROFILE", ""))))
    sys.argv = [sys.argv[0], "TestPdPrefixCachingKvOffload"]
    unittest.main(verbosity=2)
