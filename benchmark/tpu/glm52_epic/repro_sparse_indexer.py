#!/usr/bin/env python3
"""Production DSA indexer bridge isolation, not a serving/score-accuracy test.

All causal keys fit within topk2048: check selection membership, not ordering.
Run in a fresh pinned-runtime process. No checkpoint or distributed init.
"""
import argparse
import importlib.metadata
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=int, choices=(1, 4, 64), default=4)
    parser.add_argument('--key-store', choices=('xla', 'pallas'), default='xla',
                        help='Explicit writer; serving can auto-select Pallas for short batches.')
    parser.add_argument('--segments', choices=('all', 'serving'), default='all',
                        help='All passes, or ordinary MLA serving specialization by forward mode.')
    args = parser.parse_args()
    if 'jax' in sys.modules:
        raise RuntimeError('fresh process required')
    pins = {'jax': '0.11.2', 'jaxlib': '0.11.2', 'libtpu': '0.0.50',
            'torch_tpu': '0.2.1.dev20261028201649',
            'tpu-sync-torch': '0.0.1.dev20261031081414'}
    found = {k: importlib.metadata.version(k) for k in pins}
    print('RUNTIME', found, flush=True)
    if found != pins:
        raise RuntimeError(f'pinned runtime required: {pins}')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import numpy as np
    import torch
    import torch_tpu  # noqa: F401
    if torch.__version__ not in ('2.11.0', '2.11.0+cpu'):
        raise RuntimeError('torch 2.11.0 CPU ABI required')
    from sglang_tpu.dsa_indexer_tpu import build_dsa_index_topk_op
    from sglang_tpu.attention.pallas.dsa_index_topk import index_key_cache_shape

    t, h, d, p, topk = args.queries, 32, 128, 128, 2048
    print('STAGE allocate', flush=True)
    def upload(a):
        return torch.from_numpy(a).to('tpu')

    rng = np.random.default_rng(42)
    q = upload(rng.normal(size=(t, h, d)).astype(np.float32)).to(torch.bfloat16)
    key = upload(rng.normal(size=(t, d)).astype(np.float32)).to(torch.bfloat16)
    weights = torch.ones((t, h), dtype=torch.float32, device='tpu')
    cache = torch.zeros(index_key_cache_shape(2, p, d), dtype=torch.uint8, device='tpu')
    metadata = [upload(np.array(a, np.int32)) for a in (
        [t], [1], [0, t], [1, 1, 1] if t == 1 else [0, 0, 1])]
    op = build_dsa_index_topk_op()
    segments = ('decode' if t == 1 else 'mixed') if args.segments == 'serving' else 'all'
    print(f'STAGE indexer_compile_execute segments={segments} key_store={args.key_store}', flush=True)
    with torch.no_grad():
        result = op(q, key, weights, cache, *metadata,
                    float(d ** -0.5 * h ** -0.5), topk, 1, segments, args.key_store)
        selected = result.cpu().numpy()
    assert selected.shape == (t, topk), selected.shape
    assert selected.dtype == np.int32, selected.dtype
    for row in range(t):
        np.testing.assert_array_equal(np.sort(selected[row, :row + 1]), np.arange(row + 1))
        np.testing.assert_array_equal(selected[row, row + 1:], -1)
    print(f'PRODUCTION_INDEXER_MEMBERSHIP_PASS queries={t} topk={topk}', flush=True)
    # No quantization/cache-byte oracle: selection membership is the scope.


if __name__ == '__main__':
    main()
