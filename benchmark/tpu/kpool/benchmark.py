# SPDX-License-Identifier: Apache-2.0
"""Capture same-device XProf events; numerical validation is a later gate."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import subprocess
import sys
import traceback
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import reference
import shape_matrix


def load_candidate(root, source=None):
    """Load the standalone kernel without importing the serving plugin."""
    path = source or root / "sglang_tpu/attention/pallas/kpool_compress.py"
    sources = [path, *sorted((path.parent / "kpool").glob("*.py"))]
    digest = hashlib.sha256()
    for file in sources:
        if len(sources) > 1:
            digest.update(str(file.relative_to(path.parent)).encode() + b"\0")
        digest.update(file.read_bytes())
    name = f"kpool_candidate_{digest.hexdigest()}"
    # A private package permits relative imports without loading the serving
    # plugin, and keeps a frozen baseline's helpers separate from the candidate.
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.compress_and_store, digest.hexdigest()


def load_revision(root, revision, output):
    """Restore the kernel and its helper modules as one immutable snapshot."""
    prefix = "sglang_tpu/attention/pallas/"
    files = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            prefix + "kpool_compress.py",
            prefix + "kpool",
        ],
        text=True,
    ).splitlines()
    for file in files:
        target = output / file.removeprefix(prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            subprocess.check_output(
                ["git", "-C", str(root), "show", f"{revision}:{file}"]
            )
        )
    return load_candidate(root, source=output / "kpool_compress.py")


def inputs(pages, tokens, decode, batch=1, query_len=1, raw=None):
    """Fragmented request pages and an already updated raw cache."""
    if pages < max(17, (tokens + 127) // 128 + 1):
        raise ValueError("Insufficient pages for the requested token bucket")
    if raw is None:
        raw = jax.jit(
            lambda key: jax.random.normal(key, (pages, 128, 256), jnp.bfloat16)
        )(jax.random.key(42))
    page_count = (pages - 1) // batch * batch
    page_map = np.random.default_rng(42).permutation(np.arange(1, page_count + 1))
    starts = (
        np.full(batch, 131, np.int32)
        if query_len == 1
        else np.resize(np.array([131, 127, 511, 130], np.int32), batch)
    )
    tail = (
        jnp.zeros((4, 128), jnp.float32),
        jnp.asarray(starts + query_len, jnp.int32)
        if decode
        else jnp.full((batch,), query_len, jnp.int32),
        jnp.asarray(page_map, jnp.int32),
        jnp.arange(batch + 1, dtype=jnp.int32) * query_len,
        jnp.asarray([batch, batch, batch] if decode else [0, 0, batch], jnp.int32),
    )
    jax.block_until_ready((raw, tail))
    return raw, tail


def benchmark_case(
    candidate,
    pages,
    tokens,
    decode,
    opts,
    output,
    batch=1,
    query_len=1,
    raw=None,
    baseline=None,
    comparisons=None,
):
    name = f"{'decode' if decode else 'prefill'}_p{pages}_t{tokens}"
    if batch != 1:
        name += f"_b{batch}"
    if query_len != 1 or not decode:
        name += f"_q{query_len}"
    print("CASE_START", name, flush=True)
    raw, tail = inputs(pages, tokens, decode, batch, query_len, raw)
    variants = {"reference": reference.computation}
    compared = dict(comparisons or {})
    if baseline is not None:
        compared = {"baseline": baseline, **compared}
    for label, function in compared.items():
        variants[label] = (
            partial(function, max_query_length=query_len)
            if "max_query_length" in inspect.signature(function).parameters
            else partial(function, decode=True, decode_max_q=query_len)
            if decode
            else function
        )
    variants["fused"] = partial(candidate, max_query_length=query_len)
    states = {}
    try:
        for label, function in variants.items():

            def entry(*args, _fn=function):
                return _fn(*args)

            entry.__name__ = f"kpool_{name}_{label}"
            cache = jnp.zeros((pages, 32, 4, 256), jnp.uint8)
            cache.block_until_ready()
            compiled = (
                jax.jit(entry, static_argnums=(7, 8), donate_argnums=(1,))
                .lower(raw, cache, *tail, tokens, 4)
                .compile()
            )
            hlo_error = None
            if opts.hlo_text:
                try:
                    (output / f"{name}-{label}.hlo.txt").write_text(compiled.as_text())
                except jax.errors.JaxRuntimeError:
                    hlo_error = traceback.format_exc()
            memory = compiled.memory_analysis()
            stats = {
                field: getattr(memory, field, None)
                for field in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "alias_size_in_bytes",
                    "temp_size_in_bytes",
                )
            }
            pointer = cache.unsafe_buffer_pointer()
            for _ in range(opts.warmup):
                cache = compiled(raw, cache, *tail)
                cache.block_until_ready()
            aliases = cache.unsafe_buffer_pointer() == pointer
            if not aliases:
                raise RuntimeError(f"{label}: donated pooled cache was not reused")
            states[label] = {
                "call": compiled,
                "cache": cache,
                "pointer": pointer,
                "module_name": f"jit_{entry.__name__}",
                "memory": stats,
                "hlo_export_error": hlo_error,
            }
            print("COMPILED", name, label, json.dumps(stats), flush=True)
        rows = {
            label: {
                "module_name": state["module_name"],
                "memory": state["memory"],
                "pooled_cache_buffer_reused": True,
                "hlo_export_error": state["hlo_export_error"],
            }
            for label, state in states.items()
        }
        result = {
            "name": name,
            "pages": pages,
            "token_bucket": tokens,
            "batch": batch,
            "query_length": query_len,
            "actual_new_tokens": batch * query_len,
            "closed_pools": int(
                np.sum(
                    np.asarray(tail[1]) // 4 - (np.asarray(tail[1]) - query_len) // 4
                )
            ),
            "variants": rows,
            "device_profile": str(output / name / "xprof"),
            "profile_status": "requested",
            "expected_samples_per_variant": opts.repeats,
            "timing_source": "XProf device module events",
        }
        # Persist identity before profiler startup, which can fail in native code.
        manifest_path = output / f"{name}-capture.json"
        manifest_path.write_text(json.dumps(result, indent=2))
        with jax.profiler.trace(str(output / name / "xprof")):
            for iteration in range(opts.repeats):
                order = list(states) if iteration % 2 == 0 else list(reversed(states))
                for label in order:
                    state = states[label]
                    # This host marker identifies a repetition; its duration is
                    # never used as device time. The device module has a unique
                    # compiled name and all variants run on the same device.
                    with jax.profiler.TraceAnnotation(
                        f"sample/{name}/{label}", repetition=iteration
                    ):
                        cache = state["call"](raw, state["cache"], *tail)
                        cache.block_until_ready()
                    if cache.unsafe_buffer_pointer() != state["pointer"]:
                        raise RuntimeError(f"{label}: donated allocation changed")
                    state["cache"] = cache
        traces = list((output / name / "xprof").rglob("*.xplane.pb"))
        if not traces or any(path.stat().st_size == 0 for path in traces):
            raise RuntimeError(f"{name}: no nonempty XPlane capture")
        result["profile_status"] = "captured_pending_device_event_verification"
        result["trace_files"] = [str(path) for path in traces]
        manifest_path.write_text(json.dumps(result, indent=2))
        print("CASE_RESULT", json.dumps(result), flush=True)
        return result
    finally:
        states.clear()
        del raw, tail
        jax.clear_caches()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--hlo-text", action="store_true")
    parser.add_argument(
        "--matrix-section", choices=("all", "decode", "prefill"), default="all"
    )
    parser.add_argument("--matrix-start", type=int, default=0)
    parser.add_argument("--matrix-stop", type=int)
    parser.add_argument(
        "--baseline-revision",
        help="Compare the previous kernel API from a Git revision",
    )
    parser.add_argument(
        "--comparison-revision",
        action="append",
        default=[],
        metavar="LABEL=REVISION",
        help="Add a named revision to the same alternating capture",
    )
    parser.add_argument(
        "--queries", nargs="+", type=int, help="Filter the declared matrix by Q"
    )
    parser.add_argument(
        "--batches", nargs="+", type=int, help="Filter the declared matrix by batch"
    )
    opts = parser.parse_args()
    if opts.matrix_start < 0 or (
        opts.matrix_stop is not None and opts.matrix_stop <= opts.matrix_start
    ):
        parser.error("Require 0 <= matrix_start < matrix_stop")
    if opts.warmup < 1 or opts.repeats < 2:
        parser.error("Require warmup >= 1 and repeats >= 2")
    root = Path(__file__).resolve().parents[3]
    opts.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running",
        "accuracy": "not_checked",
        "scope": "XProf device module spans and Pallas events, one device",
        "timing": "compile/input/warmup excluded; warm cache; donated output; no host timing",
        "warmup": opts.warmup,
        "repeats": opts.repeats,
        "cases": [],
    }
    try:
        if jax.default_backend() != "tpu":
            raise RuntimeError("This benchmark requires a TPU")
        candidate, digest = load_candidate(root)
        baseline = None
        baseline_digest = None
        if opts.baseline_revision:
            baseline, baseline_digest = load_revision(
                root, opts.baseline_revision, opts.output / "baseline"
            )
        comparisons = {}
        comparison_identities = {}
        for item in opts.comparison_revision:
            label, separator, revision = item.partition("=")
            if (
                not separator
                or not label.isidentifier()
                or not revision
                or label in ("reference", "baseline", "fused")
                or label in comparisons
            ):
                raise ValueError("Require a unique LABEL=REVISION comparison")
            function, source_hash = load_revision(root, revision, opts.output / label)
            comparisons[label] = function
            comparison_identities[label] = {
                "revision": revision,
                "source_sha256": source_hash,
            }
        result.update(
            source_revision=subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            candidate_sha256=digest,
            baseline_revision=opts.baseline_revision,
            baseline_sha256=baseline_digest,
            comparison_revisions=comparison_identities,
            reference_sha256=hashlib.sha256(
                Path(reference.__file__).read_bytes()
            ).hexdigest(),
            devices=[str(d) for d in jax.devices()],
            device_kind=jax.devices()[0].device_kind,
            runtime={
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "libtpu", "numpy")
            },
        )
        matrix = shape_matrix.cases(opts.matrix_section)[
            opts.matrix_start : opts.matrix_stop
        ]
        matrix = [
            case
            for case in matrix
            if (opts.queries is None or case["query_length"] in opts.queries)
            and (opts.batches is None or case["batch"] in opts.batches)
        ]
        if not matrix:
            raise ValueError("Empty matrix selection")
        result["matrix"] = matrix
        with jax.default_device(jax.devices()[0]):
            shared_raw, _ = inputs(14849, 32, True)
            for case in matrix:
                result["cases"].append(
                    benchmark_case(
                        candidate,
                        14849,
                        case["tokens"],
                        case["mode"] == "decode",
                        opts,
                        opts.output,
                        case["batch"],
                        case["query_length"],
                        shared_raw,
                        baseline=baseline,
                        comparisons=comparisons,
                    )
                )
                (opts.output / "metrics.json").write_text(json.dumps(result, indent=2))
        result["status"] = "completed_pending_accuracy"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        (opts.output / "metrics.json").write_text(json.dumps(result, indent=2))
        print("BENCHMARK_RESULT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
