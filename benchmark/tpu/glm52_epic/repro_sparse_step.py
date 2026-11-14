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
    parser.add_argument('--queries', type=int, choices=(1, 4, 64, 2048), default=1)
    parser.add_argument('--selection', choices=('self', 'causal-mean'), default='self',
                        help='Single key, or causal keys with zero queries and uniform weights.')
    parser.add_argument('--entry', choices=('per-query', 'prefill-fallback', 'prefill-qblock'), default='per-query',
                        help='Token-major op, small head-major fallback, or2048-token fused qblock.')
    parser.add_argument('--dynamo', action='store_true',
                        help='Compile the production op fullgraph with backend=tpu, static shapes.')
    args = parser.parse_args()
    if (args.entry == 'prefill-qblock') != (args.queries == 2048):
        parser.error('prefill-qblock requires queries2048; other entries retain small probes')
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
    num_pages = 65 if args.entry == 'prefill-qblock' else 2
    cn = torch.zeros((num_pages, p, 4, 128), dtype=torch.uint8, device='tpu')
    cr = torch.zeros((num_pages, p // 4, 4, 128), dtype=torch.uint8, device='tpu')
    topk = np.full((t, k), -1, np.int32)
    if args.selection == 'self':
        topk[:, 0] = np.arange(t)
    else:
        for row in range(t):
            topk[row, :row + 1] = np.arange(row + 1)
    head_major = args.entry in ('prefill-fallback', 'prefill-qblock')
    distribution = [1, 1, 1] if t == 1 and not head_major else [0, 0, 1]
    pages = list(range(1, 34)) if args.entry == 'prefill-qblock' else [1]
    metadata = [upload(np.array(a, np.int32)) for a in (
        [t], pages, [0, t], [0, t if args.entry == 'prefill-qblock' else p], distribution)]
    topk_tensor = upload(topk)
    if head_major:
        q = q.transpose(0, 1).contiguous()
        qr = qr.transpose(0, 1).contiguous()
    op = _build_pallas_sparse_mla_kernel_op(qblock_kpool=1 if head_major else 0)
    if args.dynamo:
        # Same backend registration import as the plugin's startup path.
        import torch_tpu._internal.compile  # noqa: F401
        op = torch.compile(op, backend='tpu', fullgraph=True, dynamic=False)
        print('PRODUCTION_SPARSE_DYNAMO_ENABLED fullgraph=True dynamic=False', flush=True)
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
        expected_cn = np.zeros((num_pages * p, 512), np.uint8)
        expected_cr = np.zeros((num_pages * p, 128), np.uint8)
        expected_cn[p:p + t] = vn_cpu.view(torch.uint8).numpy()
        expected_cr[p:p + t, :64] = vr_cpu.view(torch.uint8).numpy()
        np.testing.assert_array_equal(actual_cn.reshape(num_pages * p, 512), expected_cn)
        np.testing.assert_array_equal(actual_cr.reshape(num_pages * p, 128), expected_cr)
        print(f'PRODUCTION_SPARSE_STEP_PASS iteration={iteration} queries={t} selection={args.selection} entry={args.entry}', flush=True)


if __name__ == '__main__':
    main()
