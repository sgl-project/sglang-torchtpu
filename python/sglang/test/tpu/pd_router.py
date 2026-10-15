"""Build the P/D disaggregation router command line for TPU e2e tests.

All TPU P/D e2e tests launch a router in front of one or more prefill and
decode engines. Historically each test open-coded::

    python -m sglang_router.launch_router --pd-disaggregation --mini-lb ...

Centralizing the argv construction behind ``SGLANG_TPU_PD_ROUTER`` lets the
same test run the standalone ``experimental/sgl-router`` Rust binary
(``sgl-router``, default ``smg``) with ``cache_aware`` prefill routing and
``power_of_two`` decode routing, or fall back to the legacy in-process
``mini_lb`` loop by flipping one environment variable.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

ENV_ROUTER_MODE = "SGLANG_TPU_PD_ROUTER"
MINI_LB = "mini_lb"
SMG = "smg"
SGL_ROUTER = SMG
ROUTER_MODES = (MINI_LB, SMG)
VALID_MODES = ROUTER_MODES

ENV_ROUTER_BIN = "SGL_ROUTER_BIN"
DEFAULT_ROUTER_BIN = "sgl-router"
DEFAULT_MODEL_ID = "gpt-oss-120b"

# Valid --policy values in experimental/sgl-router (src/config/cli.rs).
PREFILL_POLICY_CHOICES = (
    "round_robin",
    "random",
    "power_of_two",
    "load_based",
    "fused_score",
    "score_policy",
    "session_aware",
    "cache_aware",
    "sticky",
)
# Valid --decode-policy values in experimental/sgl-router (src/config/cli.rs).
DECODE_POLICY_CHOICES = (
    "power_of_two",
    "legacy_host_affinity",
)

ENV_PREFILL_POLICY = "TPU_PD_PREFILL_POLICY"
ENV_DECODE_POLICY = "TPU_PD_DECODE_POLICY"
DEFAULT_PREFILL_POLICY = "cache_aware"
DEFAULT_DECODE_POLICY = "power_of_two"

ENV_CACHE_MIN_TOKENS = "TPU_PD_CACHE_MIN_TOKENS"
ENV_CACHE_THRESHOLD = "TPU_PD_CACHE_THRESHOLD"
ENV_BALANCE_ABS = "TPU_PD_BALANCE_ABS"
ENV_BALANCE_REL = "TPU_PD_BALANCE_REL"

ENV_KV_EVENTS_PORT = "TPU_PD_KV_EVENTS_PORT"
ENV_KV_EVENTS_ENABLED = "TPU_PD_KV_EVENTS_ENABLED"
DEFAULT_KV_EVENTS_PORT = 5557

PrefillEntry = Union[str, Tuple[str, Optional[int]]]


def _env(values: Optional[Dict[str, str]]) -> Dict[str, str]:
    return dict(os.environ) if values is None else values


def resolve_mode(
    mode: Optional[str] = None,
    *,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve the router backend from ``mode`` or ``$SGLANG_TPU_PD_ROUTER``."""
    source = _env(env)
    raw = mode if mode is not None else source.get(ENV_ROUTER_MODE, SMG)
    normalized = raw.strip().lower().replace("-", "_")
    if normalized == "sgl_router":
        normalized = SMG
    if normalized not in ROUTER_MODES:
        raise ValueError(
            f"Unsupported {ENV_ROUTER_MODE}={raw!r}; expected one of {ROUTER_MODES}"
        )
    return normalized


def _resolve_policy(
    *,
    name: str,
    explicit: Optional[str],
    env_key: str,
    default: str,
    choices: Tuple[str, ...],
    env: Dict[str, str],
) -> str:
    policy = explicit or env.get(env_key) or default
    if policy not in choices:
        raise ValueError(f"{name} policy {policy!r} is not one of {choices}.")
    return policy


def _normalize_prefills(
    prefill_url: Optional[str],
    bootstrap_port: Optional[int],
    prefills: Optional[Sequence[PrefillEntry]],
) -> List[Tuple[str, Optional[int]]]:
    if prefills is not None and prefill_url is not None:
        raise ValueError("Pass either prefill_url or prefills, not both")
    if prefills is not None:
        out: List[Tuple[str, Optional[int]]] = []
        for item in prefills:
            if isinstance(item, tuple):
                url, bp = item
                out.append((url, bp))
            else:
                out.append((item, bootstrap_port))
        if not out:
            raise ValueError("prefills must not be empty")
        return out
    if prefill_url is None:
        raise ValueError("One of prefill_url or prefills is required")
    return [(prefill_url, bootstrap_port)]


def _normalize_decodes(
    decode_url: Optional[str],
    decodes: Optional[Sequence[str]],
) -> List[str]:
    if decodes is not None and decode_url is not None:
        raise ValueError("Pass either decode_url or decodes, not both")
    if decodes is not None:
        out = list(decodes)
        if not out:
            raise ValueError("decodes must not be empty")
        return out
    if decode_url is None:
        raise ValueError("One of decode_url or decodes is required")
    return [decode_url]


def router_ready_url(
    *,
    host: str,
    port: int,
    mode: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """Return the readiness probe URL for the active router backend.

    ``sgl-router`` exposes ``GET /readyz`` (returning 200 once at least one
    prefill and one decode worker are healthy in P/D mode), whereas legacy
    ``mini_lb`` exposes ``GET /health``.
    """
    resolved = resolve_mode(mode, env=env)
    endpoint = "/readyz" if resolved == SMG else "/health"
    return f"http://{host}:{port}{endpoint}"


def router_metrics_url(
    *,
    host: str,
    port: int = 8000,
    mode: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Return the Prometheus metrics URL when running in ``smg`` mode, else ``None``.

    ``experimental/sgl-router`` serves ``GET /metrics`` on its main HTTP listener
    port alongside ``/readyz`` and ``/v1/chat/completions``.
    """
    if resolve_mode(mode, env=env) != SMG:
        return None
    return f"http://{host}:{port}/metrics"


def prefill_kv_events_args(
    *,
    port: Optional[int] = None,
    mode: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[str, ...]:
    """Return ``--kv-events-config`` CLI flags for prefill engines in ``smg`` mode.

    ``experimental/sgl-router`` discovers each prefill worker's ZMQ KV-events
    endpoint from ``/server_info`` and subscribes to ``BlockStored`` /
    ``BlockRemoved`` / ``AllBlocksCleared`` events to maintain its live
    ``cache_aware`` radix tree and ``BlockSizeOracle``. In ``mini_lb`` mode, no
    extra flags are needed.
    """
    source = _env(env)
    if resolve_mode(mode, env=source) != SMG:
        return ()
    enabled_raw = source.get(ENV_KV_EVENTS_ENABLED, "1").strip().lower()
    if enabled_raw in ("0", "false", "no", "off"):
        return ()
    resolved_port = (
        port
        if port is not None
        else int(source.get(ENV_KV_EVENTS_PORT, str(DEFAULT_KV_EVENTS_PORT)))
    )
    config_json = f'{{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:{resolved_port}"}}'
    return ("--kv-events-config", config_json)


def build_router_cmd(
    *,
    python: str,
    host: str,
    port: int,
    prefill_url: Optional[str] = None,
    decode_url: Optional[str] = None,
    bootstrap_port: Optional[int] = None,
    prefills: Optional[Sequence[PrefillEntry]] = None,
    decodes: Optional[Sequence[str]] = None,
    mode: Optional[str] = None,
    router_bin: Optional[str] = None,
    model_id: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    prefill_policy: Optional[str] = None,
    decode_policy: Optional[str] = None,
    cache_affinity_min_matched_tokens: Optional[int] = None,
    cache_affinity_min_match_ratio: Optional[float] = None,
    cache_threshold: Optional[float] = None,
    pressure_abs_threshold_tokens: Optional[int] = None,
    balance_abs_threshold: Optional[int] = None,
    pressure_rel_threshold: Optional[float] = None,
    balance_rel_threshold: Optional[float] = None,
    request_timeout_secs: Optional[int] = None,
    extra: Sequence[str] = (),
    extra_args: Optional[Iterable[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[str, ...]:
    """Build the router subprocess argv for either ``sgl-router`` or ``mini_lb``.

    In ``mode="smg"`` (default), this launches ``experimental/sgl-router``
    (``sgl-router``), which discovers worker roles and bootstrap ports directly
    from each worker's ``/server_info``. With ``mode="mini_lb"`` and
    ``bootstrap_port=None`` this reproduces the legacy ``sglang_router.launch_router``
    ``--mini-lb`` argv.
    """
    source = _env(env)
    resolved_mode = resolve_mode(mode, env=source)
    prefill_list = _normalize_prefills(prefill_url, bootstrap_port, prefills)
    decode_list = _normalize_decodes(decode_url, decodes)

    if resolved_mode == MINI_LB:
        cmd: List[str] = [
            python,
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
        ]
        for url, bp in prefill_list:
            cmd.extend(["--prefill", url])
            if bp is not None:
                cmd.append(str(bp))
        for url in decode_list:
            cmd.extend(["--decode", url])
        cmd.extend(["--host", host, "--port", str(port)])
        if request_timeout_secs is not None:
            cmd.extend(["--request-timeout-secs", str(request_timeout_secs)])
    else:
        binary = router_bin or source.get(ENV_ROUTER_BIN, DEFAULT_ROUTER_BIN)
        resolved_model_id = (
            model_id or source.get("SERVED_MODEL_NAME") or DEFAULT_MODEL_ID
        )
        resolved_tokenizer_path = (
            tokenizer_path
            or source.get("TOKENIZER_PATH")
            or source.get("GPT_OSS_MODEL")
            or source.get("MODEL")
        )
        if resolved_tokenizer_path and os.path.isdir(resolved_tokenizer_path):
            for candidate in ("tokenizer.json", "tiktoken.model"):
                candidate_path = os.path.join(resolved_tokenizer_path, candidate)
                if os.path.isfile(candidate_path):
                    resolved_tokenizer_path = candidate_path
                    break
        p_policy = _resolve_policy(
            name="prefill",
            explicit=prefill_policy,
            env_key=ENV_PREFILL_POLICY,
            default=DEFAULT_PREFILL_POLICY,
            choices=PREFILL_POLICY_CHOICES,
            env=source,
        )
        d_policy = _resolve_policy(
            name="decode",
            explicit=decode_policy,
            env_key=ENV_DECODE_POLICY,
            default=DEFAULT_DECODE_POLICY,
            choices=DECODE_POLICY_CHOICES,
            env=source,
        )
        worker_urls = [url for url, _ in prefill_list] + list(decode_list)
        cmd = [
            binary,
            "--model-id",
            resolved_model_id,
        ]
        if resolved_tokenizer_path:
            cmd.extend(["--tokenizer-path", resolved_tokenizer_path])
        cmd.extend(["--worker-urls", *worker_urls])
        cmd.extend(
            [
                "--host",
                host,
                "--port",
                str(port),
                "--policy",
                p_policy,
                "--decode-policy",
                d_policy,
            ]
        )
        if p_policy == "sticky":
            rkey_header = source.get("TPU_PD_ROUTING_KEY_HEADER", "x-smg-routing-key")
            cmd.extend(["--routing-key-header", rkey_header])

        min_tokens = (
            str(cache_affinity_min_matched_tokens)
            if cache_affinity_min_matched_tokens is not None
            else source.get(ENV_CACHE_MIN_TOKENS)
        )
        if min_tokens:
            cmd.extend(["--cache-affinity-min-matched-tokens", min_tokens])

        explicit_ratio = (
            cache_affinity_min_match_ratio
            if cache_affinity_min_match_ratio is not None
            else cache_threshold
        )
        match_ratio = (
            str(explicit_ratio)
            if explicit_ratio is not None
            else source.get(ENV_CACHE_THRESHOLD)
        )
        if match_ratio:
            cmd.extend(["--cache-affinity-min-match-ratio", match_ratio])

        explicit_abs = (
            pressure_abs_threshold_tokens
            if pressure_abs_threshold_tokens is not None
            else balance_abs_threshold
        )
        balance_abs = (
            str(explicit_abs)
            if explicit_abs is not None
            else source.get(ENV_BALANCE_ABS)
        )
        if balance_abs:
            cmd.extend(["--pressure-abs-threshold-tokens", balance_abs])

        explicit_rel = (
            pressure_rel_threshold
            if pressure_rel_threshold is not None
            else balance_rel_threshold
        )
        balance_rel = (
            str(explicit_rel)
            if explicit_rel is not None
            else source.get(ENV_BALANCE_REL)
        )
        if balance_rel:
            cmd.extend(["--pressure-rel-threshold", balance_rel])

        if request_timeout_secs is not None:
            cmd.extend(["--request-timeout-secs", str(request_timeout_secs)])

    if extra:
        cmd.extend(extra)
    if extra_args:
        cmd.extend(extra_args)

    return tuple(cmd)
