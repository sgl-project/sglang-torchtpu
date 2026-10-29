"""Shared harness for the multihost P/D TP-resharding tests (issue #378).

Topology: a multihost PREFILL engine (TP16EP16) and a multihost DECODE engine
(DP2TP8EP16), each spanning its own TPU slice, connected by the PD mini load
balancer. Prefill uses attention TP=16; decode uses DP=2 with attention TP=8 per
replica, so the KV transfer reshards 16 prefill shards onto 8 decode shards per
replica via FAN_IN (2:1).

Everything both models share lives here -- profiles, engine launch, role
dispatch, probes and gsm8k. The dashboard contract is one job to one test file,
so each model gets a thin sibling that carries the registration and calls
main():

  test_tpu_gpt_oss_120b_pd_multihost_kv_offload_resharding.py     -> main("gpt_oss_120b")
  test_tpu_qwen3_coder_480b_pd_multihost_kv_offload_resharding.py -> main("qwen3_coder_480b")

Keeping both profiles here rather than in those files keeps the model matrix in
one place; a file selects one by key. This module registers itself as a disabled
helper, since run_suite requires a registration from every file under
test/registered but this one owns no row.

What each verification proves:
  * quality:       gsm8k through the LB must clear the model's accuracy floor
                   under TP resharding.
  * KV resharding: generation requests dispatched to prefill -> decode across
                   both DP replicas produce valid, deterministic, bit-equal
                   greedy text, certifying the KV cache resharded and
                   transferred intact.

Roles (rank = slice-local TPU_WORKER_ID):
  prefill rank 0   launches the prefill engine, waits for decode, starts the
                   mini-LB, runs the probes + gsm8k -- its verdict gates the run.
  prefill rank >0  worker shard of the prefill engine.
  decode any rank  launches/joins the decode engine, serves until teardown.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple, Union

try:
    import msgspec
except ModuleNotFoundError:

    class _MsgspecShim:
        class Struct:
            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__()

                def _init(self, **kw):
                    for base in reversed(cls.__mro__):
                        for k, v in base.__dict__.items():
                            if (
                                not k.startswith("_")
                                and not callable(v)
                                and not isinstance(
                                    v, (property, classmethod, staticmethod)
                                )
                            ):
                                setattr(self, k, v)
                    for k, v in kw.items():
                        setattr(self, k, v)

                cls.__init__ = _init

        @staticmethod
        def to_builtins(obj):
            if hasattr(obj, "__dict__"):
                return {
                    k: _MsgspecShim.to_builtins(v)
                    for k, v in obj.__dict__.items()
                    if not k.startswith("_")
                }
            return obj

    msgspec = _MsgspecShim()

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


# run_suite requires every file under test/registered to carry a registration;
# same convention as bench_matrix_base.py. The per-model files register the real
# rows.
register_tpu_ci(
    est_time=0,
    suite="tpu-integration",
    disabled="helper module, no tests",
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")

E = os.environ.get

PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LB_PORT = int(E("LB_PORT", "8000"))
BOOTSTRAP_PORT = int(E("BOOTSTRAP_PORT", "8998"))
DIST_PORT = int(E("DIST_PORT", "10011"))
WAIT_SECS = int(E("WAIT_SECS", "3600"))
KEEP_UP = E("KEEP_UP", "0") == "1"
PYTHON = E("PYTHON", "python3")

ROLE_PREFILL = "prefill"
ROLE_DECODE = "decode"

# Emitted by hardware_backend/tpu/raiden_transfer.py when the raiden host pool
# and DMA engine finish initializing; its absence means offload silently never
# engaged even if the probe below were to pass on the device tier alone.
RAIDEN_READY_MARKER = "Raiden KV transfer ready"

# Emitted by RaidenKVManager once per room when a hybrid model's sliding-window
# pool is actually pushed across the reshard. A hybrid model whose window never
# transferred still answers -- the full-attention layers carry most of the
# signal -- so accuracy alone does not catch it; gsm8k cleared its floor by 27
# points on the first green run, which is ample room for a silently missing
# window to hide in. This marker is the direct evidence.
RAIDEN_SWA_RESHARD_MARKER = "raiden reshard SWA pushed"

PREFILL_LOG = "/tmp/pd_prefill.log"
DECODE_LOG = "/tmp/pd_decode.log"
LB_LOG = "/tmp/pd_lb.log"


class ModelProfile(msgspec.Struct, frozen=True, kw_only=True):
    key: str
    display_name: str
    served_name: str
    default_model_path: str
    hf_repo_id: str
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
    # last occurrence).
    disagg_transfer_backend: str = "raiden"
    # Two KV pools with distinct page-id spaces, so raiden runs a second engine
    # over the sliding-window one and the reshard has to move both. Gates the
    # window-transfer assertion below; a single-pool model has nothing to check.
    hybrid_swa: bool = False
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
        hf_repo_id="openai/gpt-oss-120b",
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
        gsm8k_threshold=0.72,
        hybrid_swa=True,
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
        hf_repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        model_env="QWEN3_CODER_MODEL",
        page_size=64,
        mem_fraction_prefill="0.6",
        mem_fraction_decode="0.6",
        quantization="fp8",
        kv_cache_dtype="fp8_e4m3",
        # 262144 is what both green runs used; 32768 with the prefill cap below
        # has never been exercised. An earlier theory that the smaller pool
        # raced in-flight KV transfers was disproved (34282621754 failed with
        # 262144 and no cap), so this is here to match a verified configuration,
        # not because the pool was ever the fault.
        max_total_tokens=262144,
        device_pool_tokens_hint=262144,
        gsm8k_threshold=0.80,
        extra_server_args=("--tpu-fp8-runtime", "channelwise"),
        extra_launch_env={
            "SGLANG_TPU_FUSED_MOE_SC_GATHER": "0",
            # Keep every prefill forward inside the 256-token compile graph.
            #
            # Extends compiled at the 1024-token bucket corrupt this model's
            # output; extends at 64 or 256 never do. Three runs pin it: with
            # buckets "64 256 1024" cold prompts were clean to 216 tokens and
            # 3/3 degenerate from 271 up (34317645114); moving the middle bucket
            # to 128 moved the edge to 156 with prompt length and page count
            # unchanged (34344878767); and compiling a single 1024 bucket
            # corrupted even a 62-token prompt (34360046804). Warm prompts of
            # any length were always clean because their extend is a couple of
            # dozen new tokens and lands in the smallest graph.
            #
            # At 256 a ~900-token prompt is four extends instead of one padded
            # to 1024, so the broken graph is never reached. gsm8k went from a
            # flat 0.0 to 0.95 and 0.99 on repeat runs (34373221584, 34379061369).
            #
            # This is a workaround. The 1024 graph is still wrong, and it is
            # wrong only for this model -- gpt-oss runs the same bucket set and
            # scores 0.98 at 885 cold tokens, so fp8 weights are the suspect,
            # gpt-oss being MXFP4-native. Remove this once that is fixed.
            "CHUNKED_PREFILL_SIZE": "256",
        },
        chat_template_kwargs={"enable_thinking": False},
    ),
}


class CacheProbeConfig(msgspec.Struct, frozen=True, kw_only=True):
    page_size: int
    device_pool_tokens: int
    target_words: int
    filler_words: int
    num_fillers: int
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
        filler_words = int(values.get("CACHE_PROBE_FILLER_WORDS", "2048"))
        # ~1.3 tokens/word; sized to push ~1.2x the pool through it so the
        # churn evicts the warm target. Both are hints, not guarantees -- the
        # host-reload assertion is what actually proves eviction happened.
        filler_tokens_est = filler_words * 13 // 10
        default_fillers = (12 * device_tokens) // (10 * filler_tokens_est) + 2
        return cls(
            page_size=profile.page_size,
            device_pool_tokens=device_tokens,
            target_words=int(values.get("CACHE_PROBE_TARGET_WORDS", "2048")),
            filler_words=filler_words,
            num_fillers=int(
                values.get("CACHE_PROBE_NUM_FILLERS", str(default_fillers))
            ),
            max_new_tokens=int(values.get("CACHE_PROBE_MAX_NEW_TOKENS", "16")),
        )


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
            num_examples=int(values.get("GSM8K_NUM_EXAMPLES", "100")),
            num_threads=int(values.get("GSM8K_NUM_THREADS", "1")),
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
            raise ValueError(
                "TPU_WORKER_HOSTNAMES is empty -- this test needs the "
                "slice-local GKE TPU env of a multihost node pool"
            )
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
    explicit = values.get(profile.model_env) or values.get("MODEL")
    if explicit:
        return explicit
    if os.path.isdir(profile.default_model_path):
        return profile.default_model_path
    return profile.hf_repo_id


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
    """Prefill-side KV offload, matching the standalone kv-offload tests.

    Offload runs on the prefill engine only: it is where the pages are produced
    and where eviction pressure is, and the decode engine receives KV through
    the reshard rather than reloading it from host. build_decode_env therefore
    carries no hicache flag, and a test below pins that.
    """
    values = os.environ if env is None else env
    return " ".join(
        (
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            values.get("HICACHE_RATIO", "2"),
            "--hicache-io-backend",
            "raiden",
            "--hicache-mem-layout",
            "page_first_fused",
            "--enable-cache-report",
        )
    )


def _common_launch_env(
    profile: ModelProfile, group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    values = os.environ if env is None else env
    tp_size = int(values.get("TP_SIZE", str(8 * group_hosts)))
    default_topology = (
        "2x2x2" if group_hosts == 2 else "1x2x4" if group_hosts == 1 else ""
    )
    return {
        "MODEL": resolve_model_path(profile, values),
        "TP_SIZE": str(tp_size),
        "EP_SIZE": values.get("EP_SIZE", str(tp_size)),
        "TPU_TOPOLOGY": values.get("TPU_TOPOLOGY", default_topology),
        "PAGE_SIZE": str(profile.page_size),
        "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
        "QUANTIZATION": profile.quantization,
        "KV_CACHE_DTYPE": profile.kv_cache_dtype,
        "DISABLE_RADIX_CACHE": "0",
        "ENABLE_TORCH_COMPILE": values.get("ENABLE_TORCH_COMPILE", "1"),
        "TORCH_COMPILE_MAX_BS": values.get("TORCH_COMPILE_MAX_BS", "8"),
        "TPU_ENABLE_DIAGNOSTIC_COORDINATOR": "0",
        "TPU_DISABLE_DIAGNOSTIC_COORDINATOR": "1",
        "TPU_ENABLE_TELEMETRY_HARVESTING": "0",
        "TPU_ENABLE_TELEMETRY": "0",
        "ENABLE_TPU_TELEMETRY": "0",
        "TPU_TELEMETRY_INTERVAL_SECONDS": "86400",
        "TPU_TELEMETRY_INTERVAL_SEC": "86400",
        "TPU_DIAGNOSTIC_HARVESTER_INTERVAL_SEC": "86400",
        "PYTHON": PYTHON,
        # Raiden's reshard push needs a host buffer per layer.
        "SGLANG_TPU_RAIDEN_STAGING_LAYERS": "0",
        **profile.extra_launch_env,
    }


def build_prefill_env(
    profile: ModelProfile, group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    values = os.environ if env is None else env
    prefill_args = [
        "--served-model-name",
        profile.served_name,
        "--disaggregation-transfer-backend",
        profile.disagg_transfer_backend,
        "--watchdog-timeout",
        "3600",
        "--soft-watchdog-timeout",
        "3600",
        # A disagg engine's warmup generation cannot complete without
        # its peer + the LB pairing them, so warmup just times out.
        "--skip-server-warmup",
        # The default TPU sampling backend's distributed top-k crashes
        # under torch.compile at tp>1 (dynamo cannot construct its
        # msgspec metadata struct in-graph); pin the torch sampler
        # until that is fixed upstream.
        "--sampling-backend",
        "pytorch",
    ]
    if profile.max_total_tokens is not None:
        prefill_args += ["--max-total-tokens", str(profile.max_total_tokens)]
    tp_size = int(values.get("TP_SIZE", str(8 * group_hosts)))
    return {
        **_common_launch_env(profile, group_hosts, values),
        "TP_SIZE": str(tp_size),
        "EP_SIZE": values.get("EP_SIZE", str(tp_size)),
        "DP_SIZE": "1",
        "ENABLE_DP_ATTENTION": "0",
        "MAX_RUNNING": values.get("MAX_RUNNING", "8"),
        "TPU_COMPILE_BS": values.get("TPU_COMPILE_BS", "1"),
        "TPU_COMPILE_TOKENS": values.get("TPU_COMPILE_TOKENS", "64 256 1024"),
        "MEM_FRACTION_STATIC": values.get(
            "MEM_FRACTION_STATIC", profile.mem_fraction_prefill
        ),
        "PORT": str(PREFILL_PORT),
        "BOOTSTRAP_PORT": str(BOOTSTRAP_PORT),
        "EXTRA_ARGS": " ".join(
            (
                *prefill_args,
                hicache_server_args(profile, values),
                *profile.extra_server_args,
                values.get("EXTRA_PREFILL_ARGS", ""),
            )
        ).strip(),
    }


def build_decode_env(
    profile: ModelProfile, group_hosts: int, env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    values = os.environ if env is None else env
    tp_size = int(values.get("TP_SIZE", str(8 * group_hosts)))
    dp_size = int(values.get("DP_SIZE", "2"))
    decode_args = [
        "--served-model-name",
        profile.served_name,
        "--dp-size",
        str(dp_size),
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--disaggregation-transfer-backend",
        profile.disagg_transfer_backend,
        "--watchdog-timeout",
        "3600",
        "--soft-watchdog-timeout",
        "3600",
        "--skip-server-warmup",
        "--sampling-backend",
        "pytorch",
    ]
    if profile.max_total_tokens is not None:
        decode_args += ["--max-total-tokens", str(profile.max_total_tokens)]
    return {
        **_common_launch_env(profile, group_hosts, values),
        "TP_SIZE": str(tp_size),
        "EP_SIZE": values.get("EP_SIZE", str(tp_size)),
        "DP_SIZE": str(dp_size),
        "ENABLE_DP_ATTENTION": "1",
        "MAX_RUNNING": values.get("MAX_RUNNING", "8"),
        "TPU_COMPILE_BS": values.get("TPU_COMPILE_BS", "1 2 4 8"),
        "TPU_COMPILE_TOKENS": values.get("TPU_COMPILE_TOKENS", "64 256 1024"),
        "MEM_FRACTION_STATIC": values.get(
            "MEM_FRACTION_STATIC", profile.mem_fraction_decode
        ),
        "PORT": str(DECODE_PORT),
        "EXTRA_ARGS": " ".join(
            (
                *decode_args,
                *profile.extra_server_args,
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

_PROMPTS = (
    "Explain the theory of general relativity in simple terms.",
    "Write a Python function to solve the Traveling Salesperson Problem using dynamic programming.",
    "Summarize the key events that led to the Renaissance in Europe.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of cellular respiration step by step.",
    "How does quantum key distribution provide unconditional security in communications?",
    "Write a SQL query to find the top 3 highest spending customers in each department.",
    "Explain the concept of attention mechanism in transformer neural networks.",
)


class GenerationRecord(msgspec.Struct, frozen=True, kw_only=True):
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float


def _generate(base_url: str, prompt: str, max_new_tokens: int = 32) -> GenerationRecord:
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
    return GenerationRecord(
        text=payload["text"],
        prompt_tokens=int(meta.get("prompt_tokens") or 0),
        completion_tokens=int(meta.get("completion_tokens") or 0),
        latency_seconds=round(time.perf_counter() - started, 3),
    )


_SWEEP_PARAGRAPHS_TEXT = (
    "The harbour master keeps a ledger of every vessel that clears the outer "
    "channel, noting draught, cargo class and the pilot who signed the release. ",
    "Cartographers working the northern survey preferred aneroid barometers to "
    "mercury ones, since a broken column meant a week waiting on resupply. ",
    "A cooper's apprenticeship ran seven years: two spent shaping staves, three "
    "on hooping and fitting heads, and two repairing casks from the brewers. ",
    "The observatory's transit telescope stood on a pier sunk to bedrock so that "
    "footsteps on the gallery would not disturb a sighting. ",
    "Millwrights judged a waterwheel by the sound of its buckets emptying rather "
    "than by any gauge, and reset a wheel running dry at the crown. ",
    "Copperplate engravers bevelled a finished plate so the press blanket would "
    "not tear, retiring it for resurfacing after some hundreds of sheets. ",
    "In the salt marshes the reed cutters worked from flat punts, taking only "
    "second-year growth, because first-year reed shrank in the thatch. ",
    "Lighthouse keepers logged the hour the lamp was lit and doused, the oil "
    "drawn, and any vessel that passed close enough to hail. ",
    "Bell founders buried the mould in packed sand so the metal cooled evenly; a "
    "bell drawn too early rang false and went back to the furnace. ",
)
# One entry per KV page from 1 to 15, at roughly 31 tokens a paragraph, so the
# sweep brackets the boundary from well below it to well above.
_COLD_SWEEP_PARAGRAPHS = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 29)

# The sweep buries "what is 17 plus 25?" behind a wall of filler, so the answer
# is the evidence that the model read its whole context. Judge on that rather
# than on whether the continuation stays tidy: gpt-oss answers correctly and
# then rambles, which a spew detector reads as corruption on a model scoring
# 0.98 (34383772574). Corrupt output never contains the answer.
_SWEEP_ANSWER = "42"
_SWEEP_ANSWER_WINDOW = 32


def sweep_question(paragraphs: int, rotate: int = 0, tag: str = "") -> str:
    """Filler of a given length, then the question, then a unique opening.

    `rotate` and `tag` exist so two prompts of different lengths share no
    prefix. Without them a length sweep is nested -- each prompt is the previous
    one plus a paragraph -- and the prefill cache holds all but the last few
    tokens, so it measures an incremental extend rather than the prompt.
    """
    filler = "".join(
        _SWEEP_PARAGRAPHS_TEXT[(i + rotate) % len(_SWEEP_PARAGRAPHS_TEXT)]
        for i in range(paragraphs)
    )
    return (
        f"{tag}{filler}\n\nIgnore the notes above. Question: what is 17 plus 25? "
        "Reply with only the number and nothing else."
    )


def sweep_answer_missing(text: str) -> bool:
    """Whether the buried question went unanswered in the opening of a reply."""
    return _SWEEP_ANSWER not in re.sub(r"\s+", "", text)[:_SWEEP_ANSWER_WINDOW]


def run_cold_length_sweep(base_url: str, profile: ModelProfile) -> None:
    """Prompt length against correctness, with every prompt cold.

    gsm8k already fails when the context is corrupt, but it reports a flat 0.0
    an hour later and says nothing about where the boundary is. This costs about
    a minute and names the first bad length, which is what makes the failure
    diagnosable -- the boundary is what identified the compile bucket as the
    trigger rather than prompt length or KV page count.

    Every length gets its own paragraph rotation and leading tag, so no prompt
    shares a prefix with another and length is the only variable.
    """
    attempts = int(E("COLD_SWEEP_ATTEMPTS", "3"))
    results = []
    for step, paragraphs in enumerate(_COLD_SWEEP_PARAGRAPHS):
        prompt = sweep_question(
            paragraphs, rotate=step + 1, tag=f"Memo {step:02d}-{step * 7 + 3}. "
        )
        recs = [_generate(base_url, prompt, 64) for _ in range(attempts)]
        flags = [sweep_answer_missing(r.text) for r in recs]
        bad = [r for r, missing in zip(recs, flags) if missing]
        results.append(
            {
                "prompt_tokens": recs[0].prompt_tokens,
                "pages": -(-recs[0].prompt_tokens // profile.page_size),
                "rate": f"{len(bad)}/{attempts}",
                "sample_output": (bad[0] if bad else recs[0]).text[:100],
            }
        )
    print(
        json.dumps(
            {"cold_length_sweep": profile.key, "results": results}, sort_keys=True
        ),
        flush=True,
    )
    lost = [r for r in results if not r["rate"].startswith("0/")]
    if lost:
        first = lost[0]
        raise AssertionError(
            f"Cold prompts stopped answering at {first['prompt_tokens']} tokens "
            f"({first['pages']} KV pages, {first['rate']}); shorter ones answer. "
            f"Full sweep: {json.dumps(results, sort_keys=True)[:700]}"
        )


def run_tp_resharding_probe(base_url: str, profile: ModelProfile) -> None:
    records = []
    for i, prompt in enumerate(_PROMPTS):
        rec = _generate(base_url, prompt, 32)
        if not rec.text.strip() or rec.completion_tokens <= 0:
            raise AssertionError(
                f"Request {i} produced empty decode output -- TP resharding P/D transfer failed"
            )
        records.append(rec)

    resend = _generate(base_url, _PROMPTS[0], 32)
    if resend.text != records[0].text:
        raise AssertionError(
            "Greedy generation output diverged on resend under TP resharding"
        )

    print(
        json.dumps(
            {
                "tp_resharding_probe": profile.key,
                "num_requests": len(records),
                "sample_output": records[0].text[:100],
                "latencies": [r.latency_seconds for r in records],
            },
            sort_keys=True,
        ),
        flush=True,
    )


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
        preexec_fn=os.setsid,
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


def _http_ok(url: str, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def _wait_http_200(url: str, secs: int, procs: Tuple = ()) -> None:
    deadline = time.time() + secs
    waited = 0
    while time.time() < deadline:
        for p in procs:
            if p is not None and p.poll() is not None:
                raise RuntimeError(f"process for {url} exited rc={p.returncode}")
        if _http_ok(url):
            return
        if waited > 0 and waited % 60 == 0:
            print(f"[leader] waiting for {url} ({waited}s / {secs}s)...", flush=True)
        time.sleep(10)
        waited += 10
    raise TimeoutError(f"{url} not ready after {secs}s")


def _wait_for_log_pattern(
    log_path: str,
    patterns: Union[str, Tuple[str, ...]],
    secs: int,
    procs: Tuple = (),
) -> None:
    if isinstance(patterns, str):
        patterns = (patterns,)
    deadline = time.time() + secs
    waited = 0
    while time.time() < deadline:
        for p in procs:
            if p is not None and p.poll() is not None:
                raise RuntimeError(f"process exited rc={p.returncode}")
        if os.path.isfile(log_path):
            with open(log_path, "r", errors="replace") as f:
                content = f.read()
                for pat in patterns:
                    if pat in content:
                        print(
                            f"[leader] found {pat!r} in {log_path} after {waited}s",
                            flush=True,
                        )
                        return
        if waited > 0 and waited % 60 == 0:
            print(
                f"[leader] waiting for {patterns!r} in {log_path} ({waited}s / {secs}s)...",
                flush=True,
            )
        time.sleep(10)
        waited += 10
    raise TimeoutError(f"{patterns!r} not found in {log_path} after {secs}s")


# ------------------------------- leader case ---------------------------------


class TestPdTpResharding(unittest.TestCase):
    """Prefill leader (role=prefill rank=0): orchestrates and asserts the run."""

    procs: list[subprocess.Popen] = []
    topo: PodTopology
    profile: ModelProfile

    @classmethod
    def setUpClass(cls) -> None:
        if SGLANG_IMPORT_ERROR is not None:
            raise unittest.SkipTest(
                f"SGLang runtime unavailable: {SGLANG_IMPORT_ERROR}"
            )
        cls.topo = PodTopology.from_env()
        cls.profile = select_profile(E("MODEL_PROFILE", ""))
        model_path = resolve_model_path(cls.profile)
        validate_model_source(cls.profile, model_path)
        cls.gsm8k_config = Gsm8kConfig.from_env(cls.profile)
        cls.procs = []

        try:
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
            _wait_http_200(f"http://{decode_host}:{DECODE_PORT}/health", WAIT_SECS)

            # Wait for prefill server to finish background torch.compile warmup before starting tests
            _wait_for_log_pattern(
                PREFILL_LOG,
                ("precompile done", "Warmup complete"),
                WAIT_SECS,
                tuple(cls.procs),
            )

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
                            "--request-timeout-secs",
                            "3600",
                        ),
                        log_path=LB_LOG,
                    ),
                    preexec_fn=os.setsid,
                )
            )
            _wait_http_200(f"http://127.0.0.1:{LB_PORT}/health", 300, tuple(cls.procs))
        except Exception:
            for log_file, name in [
                (PREFILL_LOG, "prefill"),
                (DECODE_LOG, "decode"),
                (LB_LOG, "lb"),
            ]:
                if os.path.isfile(log_file):
                    with open(log_file, "rb") as f:
                        lines = f.readlines()
                        print(
                            f"=== Last {min(250, len(lines))} lines of {name} ({log_file}) ===",
                            flush=True,
                        )
                        for line in lines[-250:]:
                            sys.stdout.buffer.write(line)
                        sys.stdout.flush()
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        if KEEP_UP:
            print("[KEEP_UP=1] leaving servers running for debug")
            return
        for p in cls.procs:
            _kill(p)

    def test_01_tp_resharding_generation(self) -> None:
        run_tp_resharding_probe(
            base_url=f"http://127.0.0.1:{LB_PORT}",
            profile=self.profile,
        )

    def test_01a_cold_length_sweep(self) -> None:
        run_cold_length_sweep(
            base_url=f"http://127.0.0.1:{LB_PORT}",
            profile=self.profile,
        )

    def test_02_gsm8k_quality(self) -> None:
        score = run_gsm8k(
            base_url=f"http://127.0.0.1:{LB_PORT}",
            profile=self.profile,
            config=self.gsm8k_config,
        )
        dump_metric(
            "gsm8k_score",
            score,
            labels={
                "model": self.profile.served_name,
                "test": "pd_tp_resharding",
            },
        )
        print(f"gsm8k score={score:.4f} floor={self.gsm8k_config.threshold:.4f}")
        self.assertGreaterEqual(score, self.gsm8k_config.threshold)

    def test_02b_kv_offload_engaged(self) -> None:
        """The prefill engine's host offload tier actually initialized.

        Enabling --enable-hierarchical-cache is not evidence that offload ran.
        If the raiden HiCache backend fails to come up, the engine serves
        happily from the device pool alone and every other assertion here still
        passes -- the combo silently degrades to the one it was meant to extend.
        This marker is emitted by hardware_backend/tpu/raiden_transfer.py once
        the host pool and DMA engine are up, so its absence is the direct
        evidence of that silent skip.
        """
        with open(PREFILL_LOG, "r", errors="replace") as f:
            log = f.read()
        self.assertIn(
            RAIDEN_READY_MARKER,
            log,
            "KV offload never engaged on the prefill engine: no "
            f"{RAIDEN_READY_MARKER!r} in {PREFILL_LOG}. The run would still "
            "pass every other check while serving entirely from the device "
            "pool, so the offload half of this combo was not exercised.",
        )

    def test_03_sliding_window_crossed_the_reshard(self) -> None:
        """A hybrid model's window must actually be resharded, not just its
        full-attention pool.

        The two pools are separate engines with separate page-id spaces, and the
        window one can be skipped without any request failing: the
        full-attention layers still carry most of the signal, so the model keeps
        answering. Accuracy is a weak detector of that -- the first green run
        cleared its floor by 27 points. The push path now fails a room it cannot
        send the window for, so a silent skip is no longer possible; what stays
        undetected is the window path never being reached at all. Assert the
        transfer directly. Runs after the probe and gsm8k, so the leader has
        driven real traffic through prefill -> decode by now.
        """
        if not self.profile.hybrid_swa:
            self.skipTest(f"{self.profile.key} has a single KV pool")
        with open(PREFILL_LOG, "r", errors="replace") as f:
            log = f.read()
        self.assertIn(
            RAIDEN_SWA_RESHARD_MARKER,
            log,
            f"{self.profile.display_name} is hybrid, but no sliding-window "
            f"reshard push was logged in {PREFILL_LOG} -- the window did not "
            "cross, so the decode attended over whatever its window pool "
            "already held. Grep the log for 'Raiden reshard SWA push' for a "
            "room that failed on the way there.",
        )
        print(
            f"[leader] sliding-window reshard pushes={log.count(RAIDEN_SWA_RESHARD_MARKER)}"
        )


# ------------------------------ worker serving -------------------------------


def _serve_role(topo: PodTopology, profile: ModelProfile) -> int:
    """Non-leader pods: serve this pod's engine role until the leader is done.

    No cross-slice shutdown channel exists, so workers watch the leader's
    prefill server: healthy once and then gone means the verdict is in. A pod's
    own engine dying early still gates -- the leader fails independently when a
    peer it needs is broken, and unambiguous local deaths return nonzero.
    """
    model_path = resolve_model_path(profile)
    validate_model_source(profile, model_path)
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
            print(f"[{topo.role}:{topo.group_rank}] server exited unexpectedly rc={rc}")
            if os.path.isfile(log):
                with open(log, "rb") as f:
                    lines = f.readlines()
                    print(f"=== Last {min(250, len(lines))} lines of {log} ===")
                    for line in lines[-250:]:
                        sys.stdout.buffer.write(line)
                    sys.stdout.flush()
            return rc
        if _http_ok(leader_url):
            leader_seen, misses = True, 0
        elif leader_seen:
            misses += 1
            if misses >= 6:
                print(f"[{topo.role}:{topo.group_rank}] leader gone; stopping")
                _kill(proc)
                return 0
        elif waited > 0 and waited % 60 == 0 and not leader_seen:
            print(
                f"[{topo.role}:{topo.group_rank}] waiting for leader {leader_url} ({waited}s / {WAIT_SECS + 600}s)...",
                flush=True,
            )
        elif waited > WAIT_SECS + 600:
            print(f"[{topo.role}:{topo.group_rank}] leader never healthy; stopping")
            _kill(proc)
            return 0
        time.sleep(10)
        waited += 10


class TestPdTpReshardingConfiguration(unittest.TestCase):
    def test_profile_selection_requires_exactly_one(self) -> None:
        self.assertIs(select_profile("gpt_oss_120b"), MODEL_PROFILES["gpt_oss_120b"])
        self.assertIs(
            select_profile("qwen3_coder_480b"), MODEL_PROFILES["qwen3_coder_480b"]
        )
        for bad in ("", "all", "gpt_oss_120b,qwen3_coder_480b", "unknown"):
            with self.assertRaisesRegex(ValueError, "MODEL_PROFILE"):
                select_profile(bad)

    def test_prefill_and_decode_parallel_topologies(self) -> None:
        for profile in MODEL_PROFILES.values():
            p_env = build_prefill_env(profile, 2, env={})
            self.assertEqual(p_env["TP_SIZE"], "16")
            self.assertEqual(p_env["EP_SIZE"], "16")
            self.assertEqual(p_env["DP_SIZE"], "1")
            self.assertEqual(p_env["ENABLE_DP_ATTENTION"], "0")
            self.assertEqual(p_env["DISABLE_RADIX_CACHE"], "0")
            self.assertIn(
                f"--disaggregation-transfer-backend {profile.disagg_transfer_backend}",
                p_env["EXTRA_ARGS"],
            )

            d_env = build_decode_env(profile, 2, env={})
            self.assertEqual(d_env["TP_SIZE"], "16")
            self.assertEqual(d_env["EP_SIZE"], "16")
            self.assertEqual(d_env["DP_SIZE"], "2")
            self.assertEqual(d_env["ENABLE_DP_ATTENTION"], "1")
            self.assertIn(
                f"--disaggregation-transfer-backend {profile.disagg_transfer_backend}",
                d_env["EXTRA_ARGS"],
            )
            self.assertIn("--dp-size 2", d_env["EXTRA_ARGS"])
            self.assertIn("--enable-dp-attention", d_env["EXTRA_ARGS"])
            self.assertIn("--enable-dp-lm-head", d_env["EXTRA_ARGS"])
            self.assertNotIn("hicache", d_env["EXTRA_ARGS"])
            # Offload is prefill-side; decode receives KV through the reshard.
            self.assertIn("--enable-hierarchical-cache", p_env["EXTRA_ARGS"])
            self.assertIn("--hicache-io-backend raiden", p_env["EXTRA_ARGS"])
            self.assertIn("--hicache-mem-layout page_first_fused", p_env["EXTRA_ARGS"])

    def test_heterogeneous_attention_tp_resharding_ratio(self) -> None:
        for profile in MODEL_PROFILES.values():
            p_env = build_prefill_env(profile, 2, env={})
            d_env = build_decode_env(profile, 2, env={})
            prefill_tp = int(p_env["TP_SIZE"])
            decode_tp = int(d_env["TP_SIZE"])
            decode_dp = int(d_env["DP_SIZE"])
            prefill_attn_tp = prefill_tp
            decode_attn_tp = decode_tp // decode_dp
            self.assertEqual(prefill_attn_tp, 16)
            self.assertEqual(decode_attn_tp, 8)
            self.assertEqual(prefill_attn_tp // decode_attn_tp, 2)

    def test_gpt_oss_pool_is_bounded_and_above_precompile_floor(self) -> None:
        profile = MODEL_PROFILES["gpt_oss_120b"]
        self.assertIsNotNone(profile.max_total_tokens)
        self.assertGreaterEqual(profile.max_total_tokens, 605_000)
        bound = f"--max-total-tokens {profile.max_total_tokens}"
        self.assertIn(bound, build_prefill_env(profile, 2, env={})["EXTRA_ARGS"])
        self.assertIn(bound, build_decode_env(profile, 2, env={})["EXTRA_ARGS"])

    def test_gpt_oss_does_not_force_quantization(self) -> None:
        env = build_prefill_env(MODEL_PROFILES["gpt_oss_120b"], 2, env={})
        self.assertEqual(env["QUANTIZATION"], "")

    def test_peer_hostname_maps_only_the_role_segment(self) -> None:
        self.assertEqual(
            peer_group_hostname(
                "tpu-ci-pd-tpreshard-x7f2-prefill-0-1",
                from_role=ROLE_PREFILL,
                to_role=ROLE_DECODE,
            ),
            "tpu-ci-pd-tpreshard-x7f2-decode-0-1",
        )
        with self.assertRaisesRegex(ValueError, "no '-prefill-' segment"):
            peer_group_hostname(
                "tpu-ci-pd-tpreshard-x7f2-decode-0-0",
                from_role=ROLE_PREFILL,
                to_role=ROLE_DECODE,
            )

    def test_launcher_directory_resolves_from_this_file(self) -> None:
        self.assertTrue(
            os.path.isfile(os.path.join(BENCH, "launch_prefill.sh")),
            f"BENCH does not point at benchmark/tpu: {BENCH}",
        )
        self.assertTrue(
            os.path.isfile(os.path.join(BENCH, "launch_decode.sh")),
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

    def test_sweep_answer_check_reads_real_replies(self) -> None:
        """The sweep verdict, against replies both models actually produced.

        gpt-oss answers correctly and then rambles, including into runs of
        "... ... ...", and a spew-shaped check failed it on a bring-up scoring
        0.98 (34383772574). qwen's corrupted replies never contain the answer.
        Both directions matter: a check that flags the first costs a multi-hour
        run to disprove, and one that misses the second is why the corruption
        took so long to localize.
        """
        for label, text in (
            ("bare answer", " 42"),
            ("answer then ellipsis", '\n\n42\n\nThe user says: "... ... ... ..."'),
            ("answer then note", "\n\n42\n\n(Note: the answer is 42.)"),
            ("answer mid sentence", " The answer is 42."),
        ):
            with self.subTest(label):
                self.assertFalse(sweep_answer_missing(text))

        for label, text in (
            ("backslash run", "\\" * 40),
            ("multi-char cycle", " 4: 4: 4: 4: 4: 4: 4: 4:"),
            ("newline-broken run", " bufsize\n___\n___\n___\n___"),
            ("non-ascii repeat", "\u6a21\u6837" + "\uff1f" * 20),
            ("empty", "   "),
            ("answer only after the window", "x" * 40 + " 42"),
        ):
            with self.subTest(label):
                self.assertTrue(sweep_answer_missing(text))

    def test_qwen_prefill_is_capped_below_the_broken_bucket(self) -> None:
        """qwen must not compile a prefill extend at the 1024-token bucket.

        Removing the cap puts ~900-token prompts back in that graph and gsm8k
        returns to 0.0. Reported as a plain 0.0 an hour into the run, that is
        expensive to diagnose, so pin the cap and its relation to the buckets.
        """
        profile = MODEL_PROFILES["qwen3_coder_480b"]
        env = build_prefill_env(profile, 2, env={})
        cap = int(env["CHUNKED_PREFILL_SIZE"])
        buckets = [int(b) for b in env["TPU_COMPILE_TOKENS"].split()]
        self.assertIn(cap, buckets, "the cap must itself be a compiled bucket")
        self.assertLess(cap, max(buckets), "the cap must stay below the top bucket")

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


def _run_case(case: type) -> None:
    """Run one TestCase and exit with its verdict."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(case)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def main(profile_key: str) -> None:
    """Entry point for a per-model test file.

    `profile_key` is the model that file exists to cover. It is a default rather
    than a pin, so an explicit MODEL_PROFILE still wins and the combo dispatch
    keeps working; in the nightly the two agree by construction, since each job
    names one file and passes the matching profile.

    Without CI_PD_ROLE only the host-side configuration cases run, so the file
    is useful on a laptop. With it, non-leader ranks serve their engine and the
    prefill leader runs the live suite and returns the verdict.
    """
    os.environ.setdefault("MODEL_PROFILE", profile_key)
    if not E("CI_PD_ROLE"):
        _run_case(TestPdTpReshardingConfiguration)
    topology = PodTopology.from_env()
    if not topology.is_leader:
        sys.exit(_serve_role(topology, select_profile(E("MODEL_PROFILE", ""))))
    _run_case(TestPdTpResharding)


if __name__ == "__main__":
    # Not a dashboard row -- the registration above is disabled and CI never
    # invokes this file. Running it directly is for local work: it exercises the
    # host-side configuration cases, which need no TPU and no live server.
    sys.argv = [sys.argv[0], "TestPdTpReshardingConfiguration"]
    unittest.main(verbosity=2)
