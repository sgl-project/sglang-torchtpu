#!/usr/bin/env python3
"""Checkpoint-free production sparse MLA op with analytical output oracles.

Use a fresh pinned-runtime process. Not a model/decode baseline, full-topk
accuracy test, or performance benchmark. Cache writes and reuse are checked.
"""
import argparse
import importlib.metadata
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=int, choices=(1, 4, 64), default=1)
    parser.add_argument('--selection', choices=('self', 'causal-mean'), default='self',
                        help='Single key, or causal keys with zero queries and uniform weights.')
    parser.add_argument('--entry', choices=('per-query', 'prefill-fallback'), default='per-query',
                        help='Token-major op or head-major prefill wrapper below q-block threshold.')
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
    from sglang_tpu.attention_pallas_mla_sparse import _build_pallas_sparse_mla_kernel_op

    t, h, p, k = args.queries, 64, 128, 2048
    print('STAGE allocate', flush=True)
    def upload(a):
        return torch.from_numpy(a).to('tpu')

    query_value = 1 if args.selection == 'self' else 0
    q = torch.full((t, h, 512), query_value, dtype=torch.bfloat16, device='tpu')
    qr = torch.full((t, h, 64), query_value, dtype=torch.bfloat16, device='tpu')
    # Whole allocations, not views: the production op donates these buffers.
    cn = torch.zeros((2, p, 4, 128), dtype=torch.uint8, device='tpu')
    cr = torch.zeros((2, p // 4, 4, 128), dtype=torch.uint8, device='tpu')
    topk = np.full((t, k), -1, np.int32)
    if args.selection == 'self':
        topk[:, 0] = np.arange(t)
    else:
        for row in range(t):
            topk[row, :row + 1] = np.arange(row + 1)
    head_major = args.entry == 'prefill-fallback'
    distribution = [1, 1, 1] if t == 1 and not head_major else [0, 0, 1]
    metadata = [upload(np.array(a, np.int32)) for a in (
        [t], [1], [0, t], [0, p], distribution)]
    topk_tensor = upload(topk)
    if head_major:
        q = q.transpose(0, 1).contiguous()
        qr = qr.transpose(0, 1).contiguous()
    op = _build_pallas_sparse_mla_kernel_op(qblock_kpool=1 if head_major else 0)
    for iteration, sign in enumerate((1, -1)):
        # Exactly representable FP8 dyadics. Self selection returns that value;
        # zero-query causal selection returns the prefix's uniform mean.
        rows = np.arange(t)[:, None]
        values = sign * ((rows % 8 + 1) / 8) * np.where(np.arange(512) % 2, -1, 1)
        rope = np.broadcast_to(sign * ((rows % 4 + 1) / 8), (t, 64)).copy()
        vn_cpu = torch.from_numpy(values.astype(np.float32)).to(torch.float8_e4m3fn)
        vr_cpu = torch.from_numpy(rope.astype(np.float32)).to(torch.float8_e4m3fn)
        vn, vr = vn_cpu.to('tpu'), vr_cpu.to('tpu')
        print(f'STAGE production_op iteration={iteration}', flush=True)
        with torch.no_grad():
            out = op(q, qr, vn, vr, cn, cr, *metadata,
                     1 / 16, 1.0, 1.0, topk_tensor)
            if head_major:
                out = out.transpose(0, 1)
            actual = out.cpu().float().numpy()
            actual_cn = cn.cpu().numpy()
            actual_cr = cr.cpu().numpy()
        expected_rows = values
        if args.selection == 'causal-mean':
            expected_rows = values.cumsum(axis=0) / np.arange(1, t + 1)[:, None]
        expected = np.broadcast_to(expected_rows[:, None, :], (t, h, 512))
        assert actual.shape == (t, h, 512), actual.shape
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1 / 128)
        expected_cn = np.zeros((2, p, 512), np.uint8)
        expected_cr = np.zeros((2, p, 128), np.uint8)
        expected_cn[1, :t] = vn_cpu.view(torch.uint8).numpy()
        expected_cr[1, :t, :64] = vr_cpu.view(torch.uint8).numpy()
        np.testing.assert_array_equal(actual_cn.reshape(2, p, 512), expected_cn)
        np.testing.assert_array_equal(actual_cr.reshape(2, p, 128), expected_cr)
        print(f'PRODUCTION_SPARSE_STEP_PASS iteration={iteration} queries={t} selection={args.selection} entry={args.entry}', flush=True)


if __name__ == '__main__':
    main()
