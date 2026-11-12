"""Bounded native JAX old/new selector A/B; prints reproducible JSONL receipts.

Run from the repository root with JAX TPU installed::

    PYTHONPATH=. python test/manual/kernels/bench_block_maximum_topk.py

No Torch/SGLang runtime imports. Timings measure selectors only, not scoring or
serving. A small independent NumPy oracle below checks exact output membership.
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.kernels.deepseek_v41.index_selection import block_topk as new
from sglang_tpu.kernels.deepseek_v41.index_selection import streaming_topk as old


def exact_topk(scores, lengths, k):
    """Independent full sort: descending FP32 keys (+0 > -0), then low IDs."""
    result = np.full((len(scores), k), -1, np.int32)
    for row, length in enumerate(np.clip(lengths, 0, scores.shape[1])):
        values = scores[row, :length]
        ids = np.flatnonzero(values > -np.inf)
        # Numeric ordering handles infinities; the extra key distinguishes zeros.
        order = np.lexsort((ids, np.signbit(values[ids]), -values[ids]))[:k]
        selected = np.sort(ids[order])
        result[row, : len(selected)] = selected
    return result


def published_blocks(scores, lengths, count, block_size):
    """Sort block maxima independently, reserving a slot for the newest block."""
    result = np.full((len(scores), count), -1, np.int32)
    for row, length in enumerate(np.clip(lengths, 0, scores.shape[1])):
        if not length or not count:
            continue
        visible = scores[row, :length]
        maxima = []
        for start in range(0, int(length), block_size):
            values = visible[start : start + block_size]
            maxima.append(max(values, key=lambda x: (float(x), not np.signbit(x))))
        newest = len(maxima) - 1
        maxima[newest] = -np.inf
        others = exact_topk(np.asarray([maxima], np.float32), [len(maxima)], count - 1)[
            0
        ]
        selected = np.sort(np.append(others[others >= 0], newest))
        result[row, : len(selected)] = selected
    return result


def benchmark_inputs(rows, width, k):
    rng = np.random.default_rng(20261112)
    scores = rng.normal(size=(rows, width)).astype(np.float32)
    # Empty/short rows, clipped lengths, ties, signed zeros and infinity.
    lengths = np.resize(
        np.array([-7, 0, 1, k - 1, k, k + 1, width, width + 7], np.int32), rows
    )
    scores[6::8] = 1
    scores[7::8] = -0.0
    scores[7::8, 1::2] = 0.0
    scores[6::8, ::7] = np.inf
    scores[np.arange(width)[None, :] >= np.clip(lengths, 0, width)[:, None]] = -np.inf
    return scores, lengths


def emit(**record):
    print(json.dumps(record, sort_keys=True), flush=True)


def benchmark(fn, inputs, repeats):
    start = time.perf_counter()
    compiled = fn.lower(*inputs).compile()
    compile_s = time.perf_counter() - start
    start = time.perf_counter()
    result = jax.block_until_ready(compiled(*inputs))
    warmup_s = time.perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = jax.block_until_ready(compiled(*inputs))
        samples.append((time.perf_counter() - start) * 1e6)
    return result, dict(
        compile_s=compile_s,
        first_execution_s=warmup_s,
        median_us=float(np.median(samples)),
        p95_us=float(np.percentile(samples, 95)),
        samples_us=samples,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--widths", type=int, nargs="+", default=[16384, 131072])
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--publish", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.rows <= 128 or not 1 <= args.repeats <= 1000:
        parser.error("rows must be 1..128 and repeats 1..1000")
    if any(not 1 <= width <= 1048576 for width in args.widths):
        parser.error("widths must be 1..1048576")
    if jax.default_backend() != "tpu":
        parser.error("requires native TPU JAX; CPU is not a numerical substitute")
    root = Path(__file__).resolve().parents[3]
    receipt_path = root / "block-topk-source-receipt.json"
    if receipt_path.exists():
        source_receipt = json.loads(receipt_path.read_text())
        import hashlib

        for name, expected_hash in source_receipt["files"].items():
            assert (
                hashlib.sha256((root / name).read_bytes()).hexdigest() == expected_hash
            ), name
        source = source_receipt["base_sha"]
    else:
        source = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    emit(
        kind="environment",
        source_sha=source,
        jax=jax.__version__,
        python=platform.python_version(),
        devices=[str(d) for d in jax.devices()],
        arguments=vars(args),
    )
    info = pltpu.get_tpu_info()
    for width in args.widths:
        shape = dict(rows=args.rows, width=width, k=args.k, block_size=args.block_size)
        if not new.is_supported(**shape, publish=args.publish):
            emit(kind="unsupported", **shape, publish=args.publish)
            continue
        scores, lengths = benchmark_inputs(args.rows, width, args.k)
        inputs = (jnp.asarray(scores), jnp.asarray(lengths))
        jax.block_until_ready(inputs)
        expected = exact_topk(scores, lengths, args.k)
        blocks = new.default_blocks(
            args.k, info.sparse_core.num_lanes, bool(args.publish)
        )
        bw = (
            (width // args.block_size + info.num_lanes - 1)
            // info.num_lanes
            * info.num_lanes
        )
        resident, buffers = new._plan(
            args.rows, width, bw, args.k, blocks, args.block_size, args.publish
        )
        selected, metrics = benchmark(
            jax.jit(
                lambda s, n: new.block_topk(
                    s, n, k=args.k, block_size=args.block_size, publish=args.publish
                )
            ),
            inputs,
            args.repeats,
        )
        if args.publish:
            selected, published = selected
            np.testing.assert_array_equal(
                np.asarray(published),
                published_blocks(scores, lengths, args.publish, args.block_size),
            )
        np.testing.assert_array_equal(np.asarray(selected), expected)
        emit(
            kind="measurement",
            implementation="block_topk",
            resident=resident,
            buffers=buffers,
            publish=args.publish,
            exact=True,
            **shape,
            **metrics,
        )
        if args.publish:
            # Publication is extra work: no misleading selector-only speedup ratio.
            emit(
                kind="comparison_note",
                message="new includes publication; old timing is token-only",
            )
        selected, metrics = benchmark(
            jax.jit(
                lambda s, n: old.streaming_topk(
                    s,
                    valid_lengths=n,
                    k=args.k,
                    exclude_neg_inf=True,
                    reconstruct_threshold=min(8192, 4 * args.k),
                )
            ),
            inputs,
            args.repeats,
        )
        np.testing.assert_array_equal(np.asarray(selected), expected)
        emit(
            kind="measurement",
            implementation="streaming_topk",
            publish=0,
            exact=True,
            **shape,
            **metrics,
        )


if __name__ == "__main__":
    main()
