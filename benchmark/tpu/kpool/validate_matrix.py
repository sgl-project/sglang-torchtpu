# SPDX-License-Identifier: Apache-2.0
"""Exact-byte gate for every measured Q/B pair, run after device timing."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import reference
import shape_matrix

from benchmark import load_candidate


def validation_cases(section="all"):
    cases = [{**case, "edge": "uniform"} for case in shape_matrix.cases(section)]

    def add(mode, batch, q, edge):
        if section in ("all", mode):
            cases.append(
                {
                    "mode": mode,
                    "batch": batch,
                    "query_length": q,
                    "tokens": batch * q
                    if mode == "decode"
                    else max(64, (batch * q + 63) // 64 * 64),
                    "edge": edge,
                }
            )

    for batch in shape_matrix.BATCHES:
        for mode, queries in (("decode", (1, 6)), ("prefill", (4, 128))):
            for q in queries:
                for edge in ("ragged", "padding", "inactive_tail"):
                    add(mode, batch, q, edge)
        add("prefill", batch, 128, "boundary")
    for batch in (2, 3, 8, 32, 64):
        for mode, q in (("decode", 1), ("decode", 6), ("prefill", 4)):
            add(mode, batch, q, "shared")
    add("decode", 32, 1, "shared_nonadjacent")
    for mode, q in (("decode", 6), ("prefill", 16)):
        for edge in ("zero", "tiny", "nonfinite"):
            add(mode, 4, q, edge)
    return cases


def make_inputs(case, page_size=128):
    batch, q, edge = case["batch"], case["query_length"], case["edge"]
    ps = page_size
    name = f"{case['mode']}_q{q}_b{batch}_{edge}"
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    # Page edge, one past it, the second page's end and the pooled-page edge 4P-1.
    starts = np.resize(
        np.array([3, 7, 11, 15, ps - 1, ps + 3, 2 * ps - 1, 4 * ps - 1], np.int32),
        batch,
    )
    if case["mode"] == "prefill":
        starts[:] = 0
    lengths = np.full(batch, q, np.int32)
    if edge == "ragged":
        lengths -= np.arange(batch, dtype=np.int32) % min(q + 1, 4)
    elif edge == "padding":
        lengths[:] = 0
    elif edge == "boundary":
        starts = np.resize(
            np.array([ps - 4, ps - 1, 4 * ps - 4, 4 * ps - 1], np.int32), batch
        )
    elif edge == "shared":
        pair = (3, 7) if q == 1 else (1, 8) if q == 6 else (0, 4)
        starts = np.resize(np.array(pair, np.int32), batch)
    elif edge == "shared_nonadjacent":
        starts[0], starts[-1] = 3, 7
    pages_per_request = max(8, (int(np.max(starts + lengths)) + ps - 1) // ps + 1)
    pages = batch * pages_per_request + 1
    mapping = rng.permutation(np.arange(1, pages)).astype(np.int32)
    if edge == "shared":
        for i in range(1, batch, 2):
            mapping[i * pages_per_request] = mapping[(i - 1) * pages_per_request]
    elif edge == "shared_nonadjacent":
        mapping[(batch - 1) * pages_per_request] = mapping[0]
    raw = rng.normal(size=(pages, ps, 256)).astype(np.float32)
    if edge == "zero":
        raw[..., :128] = 0
    elif edge == "tiny":
        raw[..., :128] *= 1e-7
    elif edge == "nonfinite":
        for i, value in enumerate((np.nan, np.inf, -np.inf)):
            end = int(starts[i]) // 4 * 4 + 3
            raw[mapping[i * pages_per_request + end // ps], end % ps - 3, 0] = value
    old = rng.integers(0, 256, (pages, ps // 4, 4, 256), dtype=np.uint8)
    cu = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    active = batch // 2 if edge == "inactive_tail" else batch
    distribution = (
        [active, active, active] if case["mode"] == "decode" else [0, 0, active]
    )
    args = (
        jnp.asarray(raw, jnp.bfloat16),
        jnp.asarray(rng.normal(size=(4, 128)), jnp.float32),
        jnp.asarray(starts + lengths),
        jnp.asarray(mapping),
        jnp.asarray(cu),
        jnp.asarray(distribution, jnp.int32),
    )
    return name, args, old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--section", choices=("all", "decode", "prefill"), default="all"
    )
    parser.add_argument("--queries", nargs="+", type=int)
    parser.add_argument("--batches", nargs="+", type=int)
    parser.add_argument(
        "--page-size",
        type=int,
        default=128,
        help="Raw page size P (a multiple of 16); the pooled cache is [pages, P//4, 4, 256]",
    )
    parser.add_argument(
        "--conservative-bound",
        action="store_true",
        help="Exercise the API fallback that uses total padded tokens as the bound",
    )
    opts = parser.parse_args()
    if opts.page_size < 16 or opts.page_size % 16:
        parser.error("Require a page size that is a positive multiple of 16")
    opts.output.mkdir(parents=True, exist_ok=True)
    candidate, digest = load_candidate(Path(__file__).resolve().parents[3])
    baseline = jax.jit(reference.computation, static_argnums=(7, 8))
    cases = [
        case
        for case in validation_cases(opts.section)
        if (opts.queries is None or case["query_length"] in opts.queries)
        and (opts.batches is None or case["batch"] in opts.batches)
    ]
    if not cases:
        parser.error("Empty validation selection")
    result = {
        "candidate_sha256": digest,
        "reference_sha256": hashlib.sha256(
            Path(reference.__file__).read_bytes()
        ).hexdigest(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "jax": jax.__version__,
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "libtpu", "numpy")
        },
        "expected_cases": len(cases),
        "page_size": opts.page_size,
        "selection": {
            "section": opts.section,
            "page_size": opts.page_size,
            "queries": opts.queries,
            "batches": opts.batches,
            "conservative_bound": opts.conservative_bound,
        },
        "rule": "exact entire cache byte equality",
        "cases": [],
        "complete": False,
        "passed": False,
    }
    print("MATRIX_ACCURACY_IDENTITY", json.dumps(result), flush=True)
    for case in cases:
        name, args, old = make_inputs(case, opts.page_size)
        print("MATRIX_ACCURACY_START", name, flush=True)
        fn = (
            candidate
            if opts.conservative_bound
            else partial(candidate, max_query_length=case["query_length"])
        )
        fused = jax.jit(fn, static_argnums=(7, 8), donate_argnums=(1,))
        row = {**case, "name": name, "page_size": opts.page_size}
        try:
            expected = np.asarray(
                baseline(args[0], jnp.asarray(old.copy()), *args[1:], case["tokens"], 4)
            )
            actual = np.asarray(
                fused(args[0], jnp.asarray(old.copy()), *args[1:], case["tokens"], 4)
            )
            mismatch = np.argwhere(actual != expected)
            row.update(
                passed=not len(mismatch),
                mismatched_bytes=len(mismatch),
                first=[
                    {
                        "index": idx.tolist(),
                        "expected": int(expected[tuple(idx)]),
                        "actual": int(actual[tuple(idx)]),
                    }
                    for idx in mismatch[:8]
                ],
            )
        except Exception as exc:  # noqa: BLE001 -- persist failures before stopping a halted runtime
            row.update(passed=False, error=str(exc))
        result["cases"].append(row)
        result["complete"] = len(result["cases"]) == len(cases)
        result["passed"] = result["complete"] and all(
            r["passed"] for r in result["cases"]
        )
        (opts.output / "accuracy.json").write_text(json.dumps(result, indent=2) + "\n")
        print("MATRIX_ACCURACY_RESULT", json.dumps(row), flush=True)
        if "error" in row:
            raise RuntimeError(f"{name}: runtime error; results preserved")
        del expected, actual, old, args, fused
        jax.clear_caches()
    print(
        "MATRIX_ACCURACY_SUMMARY",
        json.dumps(
            {
                "complete": result["complete"],
                "passed": result["passed"],
                "cases": len(result["cases"]),
                "failed_cases": sum(not row["passed"] for row in result["cases"]),
            }
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
