# SPDX-License-Identifier: Apache-2.0
"""Bounded same-score selector and complete-indexer measurements on one TPU.

Use the candidate's absolute interpreter, JAX_PLATFORMS=tpu, PYTHONPATH=.:python.
The existing DSA device-test fixture supplies discontiguous paged FP8 caches.
No serving or process groups are started. --source-v2 can point to the unchanged
JAX source file for a same-input comparison under this exact runtime/compiler.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
fixture_spec = importlib.util.spec_from_file_location(
    "glm52_dsa_fixture", ROOT / "test/registered/tpu/unit/test_dsa_index_topk.py"
)
fixture_module = importlib.util.module_from_spec(fixture_spec)
sys.modules[fixture_spec.name] = fixture_module
fixture_spec.loader.exec_module(fixture_module)
fixture = vars(fixture_module)

import jax
import jax.numpy as jnp
import numpy as np

from sglang_tpu.attention.pallas.radix_topk import select_radix_topk
from sglang_tpu.dsv4.sparsecore_topk import sparsecore_topk


def emit(**record):
    print(json.dumps(record), flush=True)


def latency(fn, *args, repeats):
    start = time.monotonic()
    result = fn(*args)
    jax.block_until_ready(result)
    compile_and_first_s = time.monotonic() - start
    for _ in range(3):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter_ns() - start) / 1e3)
    return result, {
        "compile_and_first_s": compile_and_first_s,
        "median_us": statistics.median(times),
        "min_us": min(times),
        "max_us": max(times),
        "samples_us": times,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-v2")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--skip-indexer", action="store_true")
    args = parser.parse_args()
    emit(
        kind="runtime",
        executable=sys.executable,
        versions={
            k: importlib.metadata.version(k)
            for k in ("jax", "jaxlib", "libtpu", "torch", "torch_tpu")
        },
        imports={k: str(__import__(k).__file__) for k in ("jax", "jaxlib")},
        devices=[str(d) for d in jax.devices()],
        flags={
            k: v
            for k, v in os.environ.items()
            if k
            in ("XLA_FLAGS", "LIBTPU_INIT_ARGS", "JAX_PLATFORMS", "TPU_VISIBLE_CHIPS")
        },
        mapped_libtpu=[
            line.strip()
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libtpu" in line
        ],
    )
    original = None
    if args.source_v2:
        spec = importlib.util.spec_from_file_location(
            "original_radix_v2", args.source_v2
        )
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
    rng = np.random.default_rng(20261116)
    for width in (53248, 92160):
        for rows in (6, 12, 18, 24, 64, 2048):
            scores_np = rng.normal(size=(rows, width)).astype(np.float32)
            scores = jnp.asarray(scores_np)
            lengths = jnp.full((rows,), width, jnp.int32)
            functions = {
                "sparsecore": jax.jit(lambda s, l: sparsecore_topk(s, 2048, l)),
                "radix_v2": jax.jit(lambda s, l: select_radix_topk(s, 2048, l)),
            }
            if original is not None:
                from sglang_tpu.attention.pallas.radix_topk.tuned_configs import (
                    get_score_width_config,
                )

                cfg = get_score_width_config(width)
                functions["jax_source_v2"] = jax.jit(
                    lambda s, l, cfg=cfg: original.radix_topk_v2(
                        s,
                        k=2048,
                        valid_lengths=l,
                        reverse_scan=True,
                        indices_only=True,
                        block_size=cfg.block_size,
                        reconstruct_threshold=cfg.reconstruct_threshold,
                    )
                )
            for name, fn in functions.items():
                result, timing = latency(fn, scores, lengths, repeats=args.repeats)
                picks = np.asarray(result)
                cutoff = np.partition(scores_np, -2048, axis=1)[:, -2048]
                picked_scores = np.take_along_axis(scores_np, picks, axis=1)
                assert np.all(picked_scores >= cutoff[:, None]), (name, rows, width)
                assert np.all(np.diff(np.sort(picks, axis=1), axis=1) > 0)
                emit(
                    kind="selector",
                    backend=name,
                    rows=rows,
                    width=width,
                    score_buffer_bytes=4 * rows * width,
                    exact=True,
                    **timing,
                )
    if args.skip_indexer:
        return
    for context in (53248, 92160):
        for kind, build, query_size, segments in (
            (
                "verify",
                {
                    "lens": [(6, context)] * 4,
                    "token_bucket": 64,
                    "padded_rows": 40,
                    "is_decode": True,
                },
                6,
                "decode",
            ),
            (
                "prefill",
                {"lens": [(2048, context)], "token_bucket": 2048, "is_decode": False},
                1,
                "mixed",
            ),
        ):
            step = fixture["_build_step"](**build, reference_scores=False)
            values = tuple(
                jnp.asarray(getattr(step, field))
                for field in (
                    "query",
                    "key",
                    "head_weights",
                    "cache",
                    "kv_lens",
                    "page_indices",
                    "cu_q_lens",
                    "distribution",
                )
            )
            for selector in ("sparsecore", "radix_v2"):
                fn = jax.jit(
                    lambda *a, query_size=query_size, segments=segments, selector=selector: fixture[
                        "dsa_indexer_step"
                    ](
                        *a,
                        weight_scale=fixture["_WEIGHT_SCALE"],
                        topk=2048,
                        decode_query_size=query_size,
                        segments=segments,
                        selector=selector,
                    )
                )
                (result, cache), timing = latency(fn, *values, repeats=args.repeats)
                np.testing.assert_array_equal(np.asarray(cache), step.expected_cache)
                output = np.asarray(result)
                np.testing.assert_array_equal(output[step.num_active_tokens :], -1)
                emit(
                    kind="indexer",
                    backend=selector,
                    workload=kind,
                    rows=build["token_bucket"],
                    context=context,
                    cache_bytes=step.cache.nbytes,
                    writer_and_score_included=True,
                    output_rows=list(output.shape),
                    **timing,
                )


if __name__ == "__main__":
    main()
