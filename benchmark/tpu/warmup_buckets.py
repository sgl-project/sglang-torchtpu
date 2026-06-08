#!/usr/bin/env python3
"""Pre-warm sglang torch_tpu's compile grid via synthetic HTTP requests.

After ``kernel_warmup`` covers decode-mode model.forward for the
``{16, 32, 64}`` token buckets, the extend-mode prefill path is still
cold for the larger token buckets ``{128, 256, 512, 1024, 2048, 4096}``,
and the LP K-bucket grid is mostly untouched. Sending one synthetic
prompt at each input length forces dynamo + XLA to compile each
``(token_bucket, EXTEND)`` cell ONCE before the real bench begins.

bench_qwen3_32b.sh runs this automatically after /health (BENCH_WARMUP_BUCKETS=1);
you can also run it by hand:

  python3 warmup_buckets.py http://127.0.0.1:30000 Qwen/Qwen3-32B
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Token buckets used by hardware_backend/tpu/compilation compile-shape buckets.
# kernel_warmup covers 16/32/64 in DECODE mode only — and those entries have a
# different dispatch key set (no AutogradPrivateUse1) so they don't satisfy the
# bench's dynamo guards anyway. Cover the full bucket set in EXTEND mode here.
PREFILL_BUCKETS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# LP extend-mode sequence-count axis (K) after the K-pad wrapper. Capped at 64
# to match the launch script's default MAX_RUNNING (no more in-flight sequences
# are possible than the server's --max-running-requests).
K_BUCKETS = [8, 16, 32, 64]


_CALL_COUNTER = 0
_COUNTER_LOCK = __import__("threading").Lock()

# Conservative: Qwen3 has ~151k vocab; stay well under to be safe across
# Qwen/Llama-class models. Any unique-per-call prefix in this range works.
_SAFE_VOCAB = 32000


def _unique_input_ids(n: int) -> list:
    """Generate input_ids of length n with a per-call unique prefix so
    sglang's radix prefix-cache does NOT short-circuit the prefill.

    Without uniqueness: isl=4096 sees the isl=2048 call's [1..2048] as a
    cached prefix and only prefills 2048 *new* tokens — model.forward at
    input_ids.shape[0]=4096 is never compiled, defeating the warmup.
    """
    import random

    global _CALL_COUNTER
    with _COUNTER_LOCK:
        _CALL_COUNTER += 1
        seed = _CALL_COUNTER
    rng = random.Random(seed)
    # IDs in [1, _SAFE_VOCAB) — avoid 0 (often padding) and stay below any
    # special-token region.
    return [rng.randint(1, _SAFE_VOCAB - 1) for _ in range(n)]


def _send_prompt(
    base_url: str,
    model: str,
    n_input_tokens: int,
    n_output_tokens: int = 2,
    timeout: int = 300,
):
    """Send one synthetic generate request with the requested input length.

    Uses input_ids directly so we don't depend on a particular tokenizer or
    string length producing exactly n_input_tokens. Each call uses fresh
    random IDs to avoid radix prefix-cache reuse (see _unique_input_ids).
    """
    input_ids = _unique_input_ids(n_input_tokens)
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": n_output_tokens,
            "temperature": 0.0,
        },
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=timeout)
    dt = time.perf_counter() - t0
    return resp.status_code, dt


def _warm_serial(base_url: str, model: str, buckets):
    """Phase 1: single prompt at each isl — exercises extend-mode model body
    for each token bucket; LP fires with K=1 → padded to K_BUCKETS[0]=8."""
    print(f"[phase 1/2] serial extend-mode warmup over buckets={buckets}", flush=True)
    for isl in buckets:
        try:
            code, dt = _send_prompt(base_url, model, isl, n_output_tokens=2)
            print(f"  isl={isl:>5}: HTTP {code}  ({dt:.1f} s)", flush=True)
        except Exception as exc:
            print(f"  isl={isl:>5}: ERROR  {exc}", flush=True)


def _warm_concurrent(base_url: str, model: str, k_values, isl: int = 64, osl: int = 2):
    """Phase 2: send K concurrent short prompts — exercises the LP K-bucket
    grid at each value of K. Picks isl=64 (smallest non-pre-warmed prefill)
    so each call is cheap; osl=2 keeps the decode phase tiny."""
    print(
        f"[phase 2/2] concurrent LP K-bucket warmup k={k_values}, isl={isl}",
        flush=True,
    )
    for k in k_values:
        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=k) as ex:
                futures = [
                    ex.submit(_send_prompt, base_url, model, isl, osl) for _ in range(k)
                ]
                ok = 0
                for fut in as_completed(futures):
                    code, _ = fut.result()
                    if code == 200:
                        ok += 1
            print(
                f"  k={k:>3}: {ok}/{k} OK  ({time.perf_counter() - t0:.1f} s)",
                flush=True,
            )
        except Exception as exc:
            print(f"  k={k:>3}: ERROR  {exc}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base_url")
    p.add_argument("model")
    p.add_argument("--skip-phase-1", action="store_true")
    p.add_argument("--skip-phase-2", action="store_true")
    args = p.parse_args()

    t0 = time.perf_counter()
    if not args.skip_phase_1:
        _warm_serial(args.base_url, args.model, PREFILL_BUCKETS)
    if not args.skip_phase_2:
        _warm_concurrent(args.base_url, args.model, K_BUCKETS, isl=64, osl=2)
    print(f"TPU warmup done in {time.perf_counter()-t0:.1f} s.", flush=True)


if __name__ == "__main__":
    main()
