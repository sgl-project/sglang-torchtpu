#!/usr/bin/env python3
"""Single-chip raw-JAX isolation of GLM52's per-query SC gather.

Run each case in a fresh process on the pinned TPU runtime. No checkpoint or
server required. This does NOT test the TorchTPU bridge, attention, or decode.
Example: python -m benchmark.tpu.glm52_epic.repro_sparse_gather --queries 1
Repeat with --queries 64 for one default prefill gather chunk.
"""
import argparse
import importlib.metadata
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=int, choices=(1, 4, 64), default=1)
    parser.add_argument('--page-size', type=int, choices=(64, 128), default=128)
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--live-queries', type=int, default=None,
                        help='Live prefix of the padded query bucket; default all.')
    args = parser.parse_args()
    live_queries = args.queries if args.live_queries is None else args.live_queries
    if not 0 <= live_queries <= args.queries:
        parser.error('--live-queries must be between zero and --queries')
    if 'jax' in sys.modules:
        raise RuntimeError('run in a fresh process before JAX initialization')
    pins = {'jax': '0.11.2', 'jaxlib': '0.11.2', 'libtpu': '0.0.50'}
    versions = {name: importlib.metadata.version(name) for name in pins}
    print('RUNTIME ' + json.dumps(versions, sort_keys=True), flush=True)
    if versions != pins:
        raise RuntimeError(f'pinned runtime required: {pins}; found {versions}')
    os.environ['JAX_PLATFORMS'] = 'tpu'
    import jax
    import jax.numpy as jnp
    import numpy as np
    from sglang_tpu.attention.pallas.mla_sparse.dsa_gather import dsa_gather

    device = jax.devices()[0]
    if device.platform != 'tpu':
        raise RuntimeError('TPU required; CPU fallback is not a reproduction')
    from jax.experimental.pallas import tpu as pltpu
    if pltpu.get_tpu_info().sparse_core is None:
        raise RuntimeError('SparseCore-capable TPU required')
    topk, pages = 2048, 32
    n = args.queries * topk
    live = live_queries * topk
    rng = np.random.default_rng(42)
    nope = rng.integers(0, 256, (pages * args.page_size, 512), dtype=np.uint8)
    rope = rng.integers(0, 256, (pages * args.page_size, 128), dtype=np.uint8)
    indices = rng.integers(0, len(nope), n, dtype=np.int32)
    with jax.default_device(device):
        inputs = (jnp.asarray(nope.reshape(pages, args.page_size, 4, 128)),
                  jnp.asarray(rope.reshape(pages, args.page_size // 4, 4, 128)),
                  jnp.asarray(indices), jnp.asarray(live, dtype=jnp.int32))
        jax.block_until_ready(inputs)
        # Scalar valid count matches the production kernel's bounded gather.
        fn = jax.jit(lambda a, b, c, valid: dsa_gather(a, b, c, num_valid_indices=valid))
        print(f'STAGE lower queries={args.queries} topk={topk} page={args.page_size} device={device}', flush=True)
        start = time.monotonic()
        lowered = fn.lower(*inputs)
        print('STAGE compile', flush=True)
        compiled = lowered.compile()
        print(f'COMPILE_PASS elapsed_s={time.monotonic()-start:.3f}', flush=True)
        if args.compile_only:
            return
        print('STAGE execute', flush=True)
        out_nope, out_rope = jax.block_until_ready(compiled(*inputs))
        # Rows after the valid prefix are unspecified, as in production.
        assert out_nope.shape == (n, 4, 128), out_nope.shape
        assert out_rope.shape == (n, 128), out_rope.shape
        np.testing.assert_array_equal(np.asarray(out_nope).reshape(n, 512)[:live], nope[indices[:live]])
        np.testing.assert_array_equal(np.asarray(out_rope)[:live], rope[indices[:live]])
        print('RAW_JAX_GATHER_BYTE_EXACT_PASS (not a serving baseline)', flush=True)


if __name__ == '__main__':
    main()
