# SPDX-License-Identifier: Apache-2.0
"""One bounded sweep over actual GLM-5.2 score widths and indexer shapes.

Run in a raw-JAX TPU process separate from the Torch serving-op smoke.
Eight explicit (input block, reconstruction threshold, reverse scan) choices
are timed with unchanged writer, paged FP8 scores, masks and output ABI.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

helper_spec = importlib.util.spec_from_file_location(
    "glm52_radix_benchmark", Path(__file__).with_name("benchmark_glm52_radix_topk.py")
)
helper_module = importlib.util.module_from_spec(helper_spec)
sys.modules[helper_spec.name] = helper_module
helper_spec.loader.exec_module(helper_module)
helpers = vars(helper_module)

import jax
import jax.numpy as jnp
import numpy as np

from sglang_tpu.attention.pallas.radix_topk import select_radix_topk

CONFIGS = (
    (4096, 2048, True),
    (8192, 512, True),
    (8192, 8192, True),
    (16384, 2048, True),
    (16384, 8192, True),
    (8192, 2048, True),
    (8192, 2048, False),
    (16384, 8192, False),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    emit, latency = helpers["emit"], helpers["latency"]
    fixture = helpers["fixture"]
    rng = np.random.default_rng(20261116)
    for width in (53248, 92160):
        # Same-score timing at actual/active verify shapes isolates selection.
        for rows in (24, 64):
            scores_np = rng.normal(size=(rows, width)).astype(np.float32)
            scores, lengths = jnp.asarray(scores_np), jnp.full(
                (rows,), width, jnp.int32
            )
            for config in CONFIGS:
                fn = jax.jit(
                    lambda s, l, config=config: select_radix_topk(
                        s, 2048, l, config=config
                    )
                )
                output, timing = latency(fn, scores, lengths, repeats=args.repeats)
                ids = np.asarray(output)
                cutoff = np.partition(scores_np, -2048, axis=1)[:, -2048]
                assert np.all(
                    np.take_along_axis(scores_np, ids, axis=1) >= cutoff[:, None]
                )
                emit(
                    kind="selector_tuning",
                    width=width,
                    rows=rows,
                    config=config,
                    **timing,
                )
        for workload, build, query_size, segments in (
            (
                "verify",
                {
                    "lens": [(6, width)] * 4,
                    "token_bucket": 64,
                    "padded_rows": 40,
                    "is_decode": True,
                },
                6,
                "decode",
            ),
            (
                "prefill",
                {"lens": [(2048, width)], "token_bucket": 2048, "is_decode": False},
                1,
                "mixed",
            ),
        ):
            step = fixture["_build_step"](**build, reference_scores=False)
            inputs = tuple(
                jnp.asarray(getattr(step, name))
                for name in (
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
            for config in (None, *CONFIGS):
                backend = "sparsecore" if config is None else "radix_v2"
                fn = jax.jit(
                    lambda *a, config=config, backend=backend, query_size=query_size, segments=segments: fixture[
                        "dsa_indexer_step"
                    ](
                        *a,
                        weight_scale=fixture["_WEIGHT_SCALE"],
                        topk=2048,
                        decode_query_size=query_size,
                        segments=segments,
                        selector=backend,
                        radix_config=config,
                    )
                )
                (output, cache), timing = latency(fn, *inputs, repeats=args.repeats)
                np.testing.assert_array_equal(np.asarray(cache), step.expected_cache)
                np.testing.assert_array_equal(
                    np.asarray(output)[step.num_active_tokens :], -1
                )
                emit(
                    kind="indexer_tuning",
                    width=width,
                    rows=build["token_bucket"],
                    workload=workload,
                    backend=backend,
                    config=config,
                    key_store="xla",
                    writer_and_score_included=True,
                    **timing,
                )
    print("TOPK_TUNING_PASS", flush=True)


if __name__ == "__main__":
    main()
