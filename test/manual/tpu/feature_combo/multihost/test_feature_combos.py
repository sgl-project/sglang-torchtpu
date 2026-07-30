"""Multihost feature-combination tests (July-30 milestone) — 2-host TPU dev box.

Row 1 (``pd``):   prefix caching + KV offloading (HiCache) + P/D disaggregation
Row 2 (``spec``): prefix caching + EAGLE3 speculative decoding + KV offloading

Both tests drive a phased probe (populate -> warm hit -> shared-prefix ->
churn -> host-reload -> concurrent wave -> flush/repopulate); greedy outputs
must stay bit-equal to the populate baseline throughout — for pd that
certifies the cross-host KV transfer. See ``run_probe`` for the exact oracle.

Topology & roles (auto-detected from the GKE TPU env, override via NODE_RANK):
  pd:    host 0 = prefill engine (tp=8) + mini-LB + probe/asserts
         host 1 = decode engine (tp=8); KV crosses hosts over DCN
  spec:  ONE server spans both hosts (tp = NNODES * CHIPS_PER_HOST = 16);
         host 0 also runs the probe, host >0 is a worker shard.

Run the SAME command on BOTH pods, e.g. from the workstation:
  ./multihost-scripts/k8s_launch.sh -j tpu-dev-pod-gcs-d -n grok-2-tpuv7 \
    'cd /cache/work/sglang-jax-workdir/sglang-torchtpu && \
     PYTHON=/cache/venv/bin/python HF_HUB_OFFLINE=1 \
     /cache/venv/bin/python test/manual/tpu/feature_combo/multihost/test_feature_combos.py pd'

Engines are launched through the canonical benchmark/tpu launchers
(launch_prefill.sh / launch_decode.sh / launch_server.sh) so the TPU env
(LIBTPU_INIT_ARGS, compile cache, attention backend, ...) has one source of
truth. Combo-relevant launcher defaults are pinned here: DISABLE_RADIX_CACHE=0
(prefix cache IS the feature under test; the launcher's InferenceX default
disables it) and empty QUANTIZATION / KV_CACHE_DTYPE (bf16 checkpoints).

Hard-won operational notes (each cost a debugging round):
* The mini-LB injects the --prefill URL's hostname into requests as the KV
  bootstrap address that the DECODE pod dials — it must be this pod's
  cluster-reachable DNS name, never 127.0.0.1.
* The decode engine caches the prefill's per-rank (ephemeral) ZMQ ports on
  first contact — ALWAYS restart decode when the prefill restarts.
* The decode engine needs --max-total-tokens bounded too, or its auto-derived
  pool OOMs at precompile (pool materialized as compile args+outputs,
  unaliased => 2x footprint).

Env knobs (defaults shown; every knob of the underlying launchers also works):
  MODEL              pd: Qwen/Qwen3-30B-A3B   spec: meta-llama/Llama-3.1-8B-Instruct
  DRAFT_MODEL        spec: lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge
  TP_SIZE=8 (per pd engine)  EP_SIZE=8  PAGE_SIZE=64
  MAX_TOTAL_TOKENS=16384     HICACHE_RATIO=3   MEM_FRACTION_STATIC=0.30
  PREFILL_PORT=30000 DECODE_PORT=30000 LB_PORT=8000 BOOTSTRAP_PORT=8998 PORT=30000
  SPEC_STEPS=3 SPEC_TOPK=1 SPEC_DRAFT_TOKENS=4 CONTEXT_LEN=4096 (spec)
  NUM_PROMPTS=4  PREFIX_REPEATS=96  MAX_NEW_TOKENS=32
  MIN_CACHED_TOKENS=64  MIN_ACCEPT_LENGTH=1.0  ACCEPT_LENGTH_REF (mean gate)
  WAIT_SECS=2400     KEEP_UP=0 (1 => leave servers running after the verdict)
  PYTHON=python3     (interpreter passed through to the launchers)

Requires TPU hardware + torch_tpu on both pods; sglang_router on host 0 (pd).
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 5))  # test/manual/tpu/feature_combo/multihost
BENCH = os.path.join(REPO, "benchmark", "tpu")

# ---- knobs (mirror the launcher contract; every one overridable via env) ----
E = os.environ.get
PYTHON = E("PYTHON", "python3")
TP_SIZE = E("TP_SIZE", "8")
EP_SIZE = E("EP_SIZE", "8")
PAGE_SIZE = E("PAGE_SIZE", "64")
MAX_TOTAL_TOKENS = E("MAX_TOTAL_TOKENS", "16384")
HICACHE_RATIO = E("HICACHE_RATIO", "3")
MEM_FRACTION_STATIC = E("MEM_FRACTION_STATIC", "0.30")
PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LB_PORT = int(E("LB_PORT", "8000"))
BOOTSTRAP_PORT = int(E("BOOTSTRAP_PORT", "8998"))
PORT = int(E("PORT", "30000"))
WAIT_SECS = int(E("WAIT_SECS", "2400"))
NUM_PROMPTS = int(E("NUM_PROMPTS", "4"))
PREFIX_REPEATS = int(E("PREFIX_REPEATS", "96"))
MAX_NEW_TOKENS = int(E("MAX_NEW_TOKENS", "32"))
MIN_CACHED_TOKENS = int(E("MIN_CACHED_TOKENS", "64"))
MIN_ACCEPT_LENGTH = float(E("MIN_ACCEPT_LENGTH", "1.0"))
# v7x, tp=16 Llama-3.1-8B-Instruct + SpecForge EAGLE3, greedy, 2026-07-30.
# Set ACCEPT_LENGTH_REF="" to fall back to calibration (report-only).
ACCEPT_LENGTH_REF = E("ACCEPT_LENGTH_REF", "2.810")
KEEP_UP = E("KEEP_UP", "0") == "1"

RANK = int(E("NODE_RANK", E("TPU_WORKER_ID", "0")))
HOSTNAMES = [h for h in E("TPU_WORKER_HOSTNAMES", "").split(",") if h]
SELF_HOST = E("SELF_HOST", HOSTNAMES[0] if HOSTNAMES else "127.0.0.1")
PEER_DECODE_HOST = E("PEER_DECODE_HOST", HOSTNAMES[1] if len(HOSTNAMES) > 1 else "")

# Pinned launcher defaults for the combo (see module docstring).
PINNED = {
    "DISABLE_RADIX_CACHE": E("DISABLE_RADIX_CACHE", "0"),
    "QUANTIZATION": E("QUANTIZATION", ""),
    "KV_CACHE_DTYPE": E("KV_CACHE_DTYPE", ""),
    "PYTHON": PYTHON,
}
HICACHE_ARGS = (
    f"--enable-hierarchical-cache --hicache-ratio {HICACHE_RATIO} "
    f"--hicache-io-backend torch_tpu --hicache-mem-layout page_first_fused "
    f"--max-total-tokens {MAX_TOTAL_TOKENS} --enable-cache-report"
)


# --------------------------- probe (2-pass client) ---------------------------

_PREFIX_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the curious crow "
    "counts clouds over the quiet canyon. "
)


def _prompts():
    prefix = _PREFIX_SENTENCE * PREFIX_REPEATS
    return [
        f"{prefix}\nQuestion {i}: in one sentence, what animal is mentioned "
        f"first in passage {i}?\nAnswer:"
        for i in range(NUM_PROMPTS)
    ]


def _shared_prefix_prompts():
    """Fresh questions on the SAME long prefix: partial-hit scenario."""
    prefix = _PREFIX_SENTENCE * PREFIX_REPEATS
    return [
        f"{prefix}\nQuestion {q}: in one sentence, {what}?\nAnswer:"
        for q, what in enumerate(
            ("what bird appears", "what does the crow count", "where is the canyon")
        )
    ]


def _churn_prompts():
    """Distinct docs sized to overflow the device pool, evicting warm prefixes to host."""
    doc_tokens = 18 * PREFIX_REPEATS  # rough tokens per doc
    n = max(4, 3 * int(MAX_TOTAL_TOKENS) // (2 * doc_tokens) + 1)
    return [
        f"Churn document {i}. "
        + (f"Unrelated filler sentence number {i} about weather patterns. ")
        * PREFIX_REPEATS
        for i in range(n)
    ]


def _generate(base_url, text, timeout=600):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=json.dumps(
            {
                "text": text,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    body["_elapsed_s"] = round(time.perf_counter() - t0, 3)
    return body


def _flush_cache(url):
    with urllib.request.urlopen(url.rstrip("/") + "/flush_cache", timeout=60) as r:
        r.read()


def run_probe(base_url, check_spec=False, flush_url=None):
    """Phased probe (see module docstring); returns (records, failures).

    Greedy outputs must stay bit-equal to the populate baseline: for pd that
    certifies the KV crossed hosts intact.
    """
    records, failures = [], []

    def spec_checks(meta, tag, rec):
        ct = meta.get("spec_verify_ct") or 0
        if ct <= 0:
            failures.append(
                f"{tag}: spec_verify_ct={ct} — spec decoding did not run "
                f"(meta keys: {sorted(meta)})"
            )
            return
        accept_length = (meta.get("completion_tokens") or 0) / ct
        rec["accept_length"] = round(accept_length, 3)
        if accept_length <= MIN_ACCEPT_LENGTH:
            failures.append(
                f"{tag}: accept_length={accept_length:.2f} <= "
                f"{MIN_ACCEPT_LENGTH} — drafts never accepted"
            )

    def probe(pass_name, i, prompt, check_cache=False, expect_text=None):
        tag = f"{pass_name}[{i}]"
        body = _generate(base_url, prompt)  # transport errors propagate
        meta = body.get("meta_info", {})
        text = body.get("text") or ""
        details = meta.get("cached_tokens_details") or {}
        rec = {
            "pass": pass_name,
            "i": i,
            "elapsed_s": body.get("_elapsed_s"),
            "cached_tokens": meta.get("cached_tokens"),
            "cached_host": details.get("host"),
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "spec_verify_ct": meta.get("spec_verify_ct"),
            "text_head": text[:80],
        }
        records.append(rec)
        if not text.strip():
            failures.append(f"{tag}: empty completion text")
        if expect_text is not None and text != expect_text:
            failures.append(f"{tag}: greedy output diverged from populate baseline")
        if check_cache:
            if meta.get("cached_tokens") is None:
                failures.append(
                    f"{tag}: no cached_tokens in meta_info — server missing "
                    f"--enable-cache-report (meta keys: {sorted(meta)})"
                )
                return meta, text
            if meta["cached_tokens"] < MIN_CACHED_TOKENS:
                failures.append(
                    f"{tag}: cached_tokens={meta['cached_tokens']} < "
                    f"{MIN_CACHED_TOKENS} — prefix cache MISS"
                )
            if check_spec:
                spec_checks(meta, tag, rec)
        return meta, text

    warm = _prompts()

    baseline = {}
    for i, p in enumerate(warm):  # 1. populate: greedy baseline
        _, baseline[i] = probe("populate", i, p)

    for i, p in enumerate(warm):  # 2. warm hit
        probe("warm", i, p, check_cache=True, expect_text=baseline[i])

    for i, p in enumerate(_shared_prefix_prompts()):  # 3. partial hit
        probe("shared-prefix", i, p, check_cache=True)

    for i, p in enumerate(_churn_prompts()):  # 4. evict warm set out of device
        probe("churn", i, p)

    # 5. reload from host: plain radix would MISS here; zero host tokens
    # means offload never engaged.
    host_total, details_missing = 0, True
    for i, p in enumerate(warm):
        meta, _ = probe(
            "host-reload", i, p, check_cache=True, expect_text=baseline[i]
        )
        d = meta.get("cached_tokens_details") or {}
        if d:
            details_missing = False
            host_total += d.get("host") or 0
    if details_missing:
        failures.append(
            "host-reload: no cached_tokens_details in meta_info — cannot "
            "prove the host tier engaged"
        )
    elif host_total <= 0:
        failures.append(
            "host-reload: cached_tokens_details.host == 0 across the pass — "
            "offload/reload never engaged despite eviction"
        )

    # 6. concurrent wave: reloads race fresh prefills; warm hits must match baseline.
    wave = (_churn_prompts()[:3] + [warm[0]]) * 2
    outs = [None] * len(wave)

    def wave_worker(k, p):
        try:
            outs[k] = _generate(base_url, p)
        except Exception as exc:  # noqa: BLE001 - recorded as a failure below
            outs[k] = exc

    threads = [
        threading.Thread(target=wave_worker, args=(k, p)) for k, p in enumerate(wave)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for k, out in enumerate(outs):
        if isinstance(out, Exception):
            failures.append(f"wave[{k}]: {out!r}")
        elif wave[k] == warm[0] and (out.get("text") or "") != baseline[0]:
            failures.append(f"wave[{k}]: warm hit under load diverged from baseline")

    # 7. flush + repopulate (flush hits the cache owner; the LB doesn't route it).
    if flush_url:
        _flush_cache(flush_url)
        meta, _ = probe("post-flush", 0, warm[0])
        if (meta.get("cached_tokens") or 0) != 0:
            failures.append(
                f"post-flush: cached_tokens={meta.get('cached_tokens')} != 0 "
                f"— flush did not clear the cache"
            )
        probe("repopulate", 0, warm[0], check_cache=True, expect_text=baseline[0])

    # 8. accept-length on par (spec rows): gate the MEAN at 90% of the pinned
    # reference; per-request floors above only catch total draft rejection.
    if check_spec:
        lengths = [r["accept_length"] for r in records if "accept_length" in r]
        if not lengths:
            failures.append("accept-length: no spec-checked probes recorded")
        else:
            mean_al = sum(lengths) / len(lengths)
            print(f"mean accept_length={mean_al:.3f} over {len(lengths)} probes")
            if ACCEPT_LENGTH_REF and mean_al < 0.9 * float(ACCEPT_LENGTH_REF):
                failures.append(
                    f"mean accept_length {mean_al:.3f} < 90% of "
                    f"reference {ACCEPT_LENGTH_REF}"
                )

    return records, failures


# ------------------------- process/launch plumbing ---------------------------


def _launch(script, extra_env, log_path):
    """Launch a benchmark/tpu shell launcher in its own process group."""
    env = dict(os.environ)
    env.update(PINNED)
    env.update(extra_env)
    log = open(log_path, "ab")
    return subprocess.Popen(
        ["bash", os.path.join(BENCH, script)],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # own pgid so teardown can killpg
        cwd=BENCH,
    )


def _kill(proc):
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


def _wait_http_200(url, secs, procs=()):
    deadline = time.time() + secs
    while time.time() < deadline:
        for p in procs:
            if p is not None and p.poll() is not None:
                raise RuntimeError(
                    f"process for {url} exited rc={p.returncode} during startup"
                )
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(10)
    raise TimeoutError(f"{url} not ready after {secs}s")


def _no_multihost_env():
    # P/D engines are single-host: hide the GKE multihost env so
    # launch_server.sh does not auto-fan tp across both pods.
    return {"TPU_WORKER_ID": "", "TPU_WORKER_HOSTNAMES": ""}


# ------------------------------- test cases ----------------------------------


class _ComboBase(unittest.TestCase):
    """Shared verdict plumbing: subclasses set records/failures."""

    maxDiff = None

    def _assert_probe(self, records, failures):
        for rec in records:
            extra = (
                f" accept_length={rec['accept_length']}"
                if "accept_length" in rec
                else ""
            )
            print(
                f"  {rec['pass']}[{rec['i']}] cached={rec['cached_tokens']} "
                f"prompt={rec['prompt_tokens']} "
                f"completion={rec['completion_tokens']} {rec['elapsed_s']}s{extra}"
            )
        self.assertEqual(
            failures, [], "\n".join(["combo assertions failed:"] + failures)
        )


class TestPDPrefixKVOffload(_ComboBase):
    """Row 1: prefix caching + KV offload + P/D across 2 hosts (rank 0 only)."""

    procs = []

    @classmethod
    def setUpClass(cls):
        model = E("MODEL", "Qwen/Qwen3-30B-A3B")
        assert PEER_DECODE_HOST, "PEER_DECODE_HOST unset (need TPU_WORKER_HOSTNAMES)"
        cls.procs = []

        prefill_env = {
            "MODEL": model,
            "TP_SIZE": TP_SIZE,
            "EP_SIZE": EP_SIZE,
            "PAGE_SIZE": PAGE_SIZE,
            "MEM_FRACTION_STATIC": MEM_FRACTION_STATIC,
            "PORT": str(PREFILL_PORT),
            "BOOTSTRAP_PORT": str(BOOTSTRAP_PORT),
            "EXTRA_ARGS": f"{HICACHE_ARGS} {E('EXTRA_PREFILL_ARGS', '')}",
            **_no_multihost_env(),
        }
        cls.procs.append(
            _launch("launch_prefill.sh", prefill_env, "/tmp/combo_prefill.log")
        )
        _wait_http_200(f"http://127.0.0.1:{PREFILL_PORT}/health", WAIT_SECS, cls.procs)
        _wait_http_200(f"http://{PEER_DECODE_HOST}:{DECODE_PORT}/health", WAIT_SECS)

        # LB: advertise prefill by cluster-reachable name (decode dials it).
        lb_log = open("/tmp/combo_lb.log", "ab")
        cls.procs.append(
            subprocess.Popen(
                [
                    PYTHON,
                    "-m",
                    "sglang_router.launch_router",
                    "--pd-disaggregation",
                    "--mini-lb",
                    "--prefill",
                    f"http://{SELF_HOST}:{PREFILL_PORT}",
                    "--decode",
                    f"http://{PEER_DECODE_HOST}:{DECODE_PORT}",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(LB_PORT),
                ],
                stdout=lb_log,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
        )
        _wait_http_200(f"http://127.0.0.1:{LB_PORT}/health", 300, cls.procs)

    @classmethod
    def tearDownClass(cls):
        if KEEP_UP:
            print("KEEP_UP=1: leaving prefill+LB running")
            return
        for p in getattr(cls, "procs", []):
            _kill(p)

    def test_prefix_cache_survives_pd_with_kv_offload(self):
        records, failures = run_probe(
            f"http://127.0.0.1:{LB_PORT}",
            flush_url=f"http://127.0.0.1:{PREFILL_PORT}",
        )
        self._assert_probe(records, failures)


class TestSpecPrefixKVOffload(_ComboBase):
    """Row 2: prefix caching + EAGLE3 + KV offload, one tp=16 server (rank 0)."""

    procs = []

    @classmethod
    def _combo_env(cls):
        model = E("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        draft = E("DRAFT_MODEL", "lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge")
        spec_args = (
            f"--speculative-algorithm EAGLE3 "
            f"--speculative-draft-model-path {draft} "
            f"--speculative-num-steps {E('SPEC_STEPS', '3')} "
            f"--speculative-eagle-topk {E('SPEC_TOPK', '1')} "
            f"--speculative-num-draft-tokens {E('SPEC_DRAFT_TOKENS', '4')}"
        )
        return {
            "MODEL": model,
            "PAGE_SIZE": PAGE_SIZE,
            "PORT": str(PORT),
            "CONTEXT_LEN": E("CONTEXT_LEN", "4096"),
            "MAX_TOTAL_TOKENS": MAX_TOTAL_TOKENS,
            "MEM_FRACTION_STATIC": MEM_FRACTION_STATIC,
            # spec-on-TPU recipe (matches spec/bench_eagle_e2e.sh defaults)
            "DISABLE_OVERLAP_SCHEDULE": E("DISABLE_OVERLAP_SCHEDULE", "1"),
            "ENABLE_TORCH_COMPILE": E("ENABLE_TORCH_COMPILE", "0"),
            "EXTRA_ARGS": f"{HICACHE_ARGS} {spec_args} {E('EXTRA_ARGS', '')}",
        }

    @classmethod
    def setUpClass(cls):
        cls.procs = [
            _launch("launch_server.sh", cls._combo_env(), "/tmp/combo_spec_server.log")
        ]
        _wait_http_200(f"http://127.0.0.1:{PORT}/health", WAIT_SECS, cls.procs)

    @classmethod
    def tearDownClass(cls):
        if KEEP_UP:
            print("KEEP_UP=1: leaving the server running")
            return
        for p in getattr(cls, "procs", []):
            _kill(p)

    def test_prefix_cache_and_spec_with_kv_offload(self):
        records, failures = run_probe(
            f"http://127.0.0.1:{PORT}",
            check_spec=True,
            flush_url=f"http://127.0.0.1:{PORT}",
        )
        self._assert_probe(records, failures)


# ------------------------------ role dispatch --------------------------------


def _http_ok(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:  # noqa: BLE001 - any transport error means "not healthy"
        return False


def _serve_worker(combo):
    """Rank >0: run this host's serving role until the leader finishes.

    No cross-host shutdown channel exists, so the worker watches the leader's
    server: healthy once and then gone => rank-0's verdict is in; tear down and
    exit 0. Real failures still gate: an engine that dies on its own returns
    its rc, and rank 0 fails independently when a peer it needs is broken.
    """
    leader = HOSTNAMES[0] if HOSTNAMES else "127.0.0.1"
    if combo == "pd":
        model = E("MODEL", "Qwen/Qwen3-30B-A3B")
        env = {
            "MODEL": model,
            "TP_SIZE": TP_SIZE,
            "EP_SIZE": EP_SIZE,
            "PAGE_SIZE": PAGE_SIZE,
            "MEM_FRACTION_STATIC": MEM_FRACTION_STATIC,
            "PORT": str(DECODE_PORT),
            # decode pool must be bounded too (precompile OOM otherwise)
            "EXTRA_ARGS": f"--max-total-tokens {MAX_TOTAL_TOKENS} "
            f"--enable-cache-report {E('EXTRA_DECODE_ARGS', '')}",
            **_no_multihost_env(),
        }
        proc = _launch("launch_decode.sh", env, "/tmp/combo_decode.log")
        leader_url = f"http://{leader}:{PREFILL_PORT}/health"
        leader_seen, misses, waited = False, 0, 0
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc  # decode died on its own: its rc is the verdict
            if _http_ok(leader_url):
                leader_seen, misses = True, 0
            elif leader_seen:
                misses += 1
                if misses >= 6:
                    print("[worker] leader served and is gone; stopping decode")
                    _kill(proc)
                    return 0
            elif waited > WAIT_SECS + 600:
                print("[worker] leader never became healthy; deferring to rank 0")
                _kill(proc)
                return 0
            time.sleep(10)
            waited += 10

    # spec: worker shard of the leader's ONE tp group; it dies with the
    # leader's engine. A nonzero rc with the leader gone is normal teardown.
    proc = _launch(
        "launch_server.sh",
        TestSpecPrefixKVOffload._combo_env(),
        "/tmp/combo_spec_worker.log",
    )
    rc = proc.wait()
    if rc != 0 and not _http_ok(f"http://{leader}:{PORT}/health"):
        print(f"[worker] shard exited rc={rc} after leader teardown; success")
        return 0
    return rc


if __name__ == "__main__":
    combo = (sys.argv[1] if len(sys.argv) > 1 else E("COMBO", "pd")).lower()
    assert combo in ("pd", "spec"), f"usage: {sys.argv[0]} pd|spec (got {combo!r})"
    if RANK != 0:
        sys.exit(_serve_worker(combo))
    sys.argv = [
        sys.argv[0],
        "TestPDPrefixKVOffload" if combo == "pd" else "TestSpecPrefixKVOffload",
    ]
    unittest.main(verbosity=2)
