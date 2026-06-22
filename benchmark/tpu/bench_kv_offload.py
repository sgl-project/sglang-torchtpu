#!/usr/bin/env python3
"""Reload-heavy benchmark for the TPU HiCache KV-offload path (--hicache-io-backend
pallas_tpu). Drives an already-running sglang server over /generate.

It measures three operating points, all at the same concurrency, so you can see
what the host<->HBM reload actually costs end to end:

  device-hit   prompts whose KV still fits the device pool   -> best case (no reload)
  host-reload  prompts evicted to the host pool, then re-hit -> exercises load_to_device_layerwise
  recompute    fresh prompts, no cache hit (offload OFF)     -> worst case (full prefill)

Working-set sizing is the whole game: the reload set must exceed the device KV
pool (max-total-tokens) but fit the host pool (hicache-ratio * device). The script
prints cached_tokens_details so you can confirm host_tok>0 (a real reload fired).

Example — Qwen3-30B-A3B on 8 chiplets with offload forced:
  # launch (note: TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false must be
  # exported so the TP all-reduce stays in the SPMD graph at TP>1 under compile):
  TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false \
  python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --device tpu \
    --tp-size 8 --ep-size 8 --enable-torch-compile --page-size 128 \
    --attention-backend pallas_tpu_rpa_v3 --mem-fraction-static 0.30 \
    --enable-hierarchical-cache --hicache-io-backend pallas_tpu --hicache-ratio 3 \
    --max-total-tokens 16384 --enable-cache-report --port 30033
  # bench (device KV=16384 -> reload set 24x1300=31200 > device, < host 49152):
  python benchmark/tpu/bench_kv_offload.py --port 30033 --reload-prompts 24 \
    --plen 1300 --device-prompts 10 --concurrency 12

A/B the reload path (optional, dev only): the producer's reload path has a
benchmark toggle — when the flag file /cache/kvoffload_serial exists,
load_to_device_layerwise uses the old per-layer blocking copy->scatter->sync
instead of the pipelined (dispatch-all, single-barrier) path. Pass --ab to
alternate async vs serial and report the overlap win. (Remove that toggle before
merging the PR.)
"""

import argparse
import json
import os
import statistics
import threading
import time
import urllib.request

VOCAB = (
    "the of and to in a is that for it as was with on be by at this from but not are or "
    "his they she had which were we when your can said there use each their time will how "
    "other about many then them these so some her would make like him into has two more "
    "very after our just name good through where help much before line right too mean old "
    "any same tell boy follow came want show also around form three small set put end does "
    "another well large must big even such because turn here why ask went men read need land "
    "different home us move try kind hand picture again change off play spell air away animal "
    "house point page letter mother answer found study still learn should America world high"
).split()


def make_prompt(i: int, nwords: int) -> str:
    """Deterministic, distinct prompt: unique header + a pseudo-random stream of
    common (≈1-token) words, so nwords ≈ token count and prompts share no prefix."""
    seed = (i + 1) * 2654435761 & 0xFFFFFFFF
    words = []
    for _ in range(nwords):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        words.append(VOCAB[seed % len(VOCAB)])
    return (
        f"Document number {i} unique header alpha{i} beta{i*7} gamma{i*13}. "
        + " ".join(words)
    )


def make_client(base: str, max_new: int):
    def gen(prompt: str):
        body = json.dumps(
            {
                "text": prompt,
                "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0},
            }
        ).encode()
        req = urllib.request.Request(
            base + "/generate", data=body, headers={"Content-Type": "application/json"}
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        dt = time.perf_counter() - t0
        ctd = out.get("meta_info", {}).get("cached_tokens_details") or {}
        return dt, ctd.get("host", 0), ctd.get("device", 0)

    return gen


def run_round(gen, prompts, concurrency, tag=None):
    results, lock, idx = [], threading.Lock(), [0]
    work = list(prompts)

    def worker():
        while True:
            with lock:
                if idx[0] >= len(work):
                    return
                p = work[idx[0]]
                idx[0] += 1
            try:
                r = gen(p)
            except Exception as e:  # keep the run alive; report and move on
                with lock:
                    print(f"   req failed: {type(e).__name__}: {e}")
                continue
            with lock:
                results.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    if tag and results:
        lats = sorted(r[0] for r in results)
        host = sum(r[1] for r in results)
        dev = sum(r[2] for r in results)
        print(
            f"  {tag:18} reqs={len(results):3d} wall={wall:6.2f}s "
            f"thrpt={len(results)/wall:5.2f} req/s "
            f"lat_mean={statistics.mean(lats)*1000:5.0f}ms "
            f"p50={lats[len(lats)//2]*1000:5.0f}ms "
            f"host_tok={host:6d} dev_tok={dev:6d}"
        )
    return wall


def set_serial(on: bool):
    flag = "/cache/kvoffload_serial"
    if on:
        open(flag, "w").close()
    elif os.path.exists(flag):
        os.remove(flag)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", type=int, default=30033)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--reload-prompts",
        type=int,
        default=24,
        help="reload set size (must exceed device pool)",
    )
    ap.add_argument(
        "--device-prompts",
        type=int,
        default=10,
        help="device-hit set size (must fit device pool)",
    )
    ap.add_argument(
        "--plen", type=int, default=1300, help="approx prompt length in words (~tokens)"
    )
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument(
        "--samples", type=int, default=3, help="warm samples per operating point"
    )
    ap.add_argument(
        "--ab",
        action="store_true",
        help="A/B the async vs serial reload path (needs the dev toggle)",
    )
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    gen = make_client(base, args.max_new)
    reload_set = [make_prompt(i, args.plen) for i in range(args.reload_prompts)]
    device_set = [make_prompt(500 + i, args.plen) for i in range(args.device_prompts)]

    print(f"== warmup (compile shapes) ==")
    set_serial(False)
    run_round(gen, reload_set, args.concurrency)
    time.sleep(1.0)
    run_round(gen, reload_set, args.concurrency)

    print("\n== device-hit (no reload, best case) ==")
    run_round(gen, device_set, 2)  # populate device
    time.sleep(1.0)
    for k in range(args.samples):
        run_round(gen, device_set, args.concurrency, f"device-hit #{k+1}")

    print("\n== host reload (exercises load_to_device_layerwise) ==")
    modes = [("ASYNC", False), ("SERIAL", True)] if args.ab else [("ASYNC", False)]
    for k in range(args.samples):
        for name, serial in modes:
            set_serial(serial)
            run_round(gen, reload_set, 2)  # repopulate -> evict to host
            time.sleep(0.4)
            run_round(gen, reload_set, args.concurrency, f"reload-{name} #{k+1}")
    set_serial(False)

    print("\n== recompute (fresh prompts, no cache = offload OFF, worst case) ==")
    for k in range(args.samples):
        fresh = [
            make_prompt(9000 + 100 * k + i, args.plen)
            for i in range(args.reload_prompts)
        ]
        run_round(gen, fresh, args.concurrency, f"recompute #{k+1}")


if __name__ == "__main__":
    main()
