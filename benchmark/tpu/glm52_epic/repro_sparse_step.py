#!/usr/bin/env python3
"""Checkpoint-free production sparse MLA op, single-key analytical oracle.

Use a fresh pinned-runtime process. Not a model/decode baseline, full-topk
accuracy test, or performance benchmark. Cache writes and reuse are checked.
"""
import argparse
import importlib.metadata
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=int, choices=(1, 64), default=1)
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

    q = torch.ones((t, h, 512), dtype=torch.bfloat16, device='tpu')
    qr = torch.ones((t, h, 64), dtype=torch.bfloat16, device='tpu')
    # Whole allocations, not views: the production op donates these buffers.
    cn = torch.zeros((2, p, 4, 128), dtype=torch.uint8, device='tpu')
    cr = torch.zeros((2, p // 4, 4, 128), dtype=torch.uint8, device='tpu')
    topk = np.full((t, k), -1, np.int32)
    topk[:, 0] = np.arange(t)
    metadata = [upload(np.array(a, np.int32)) for a in (
        [t], [1], [0, t], [0, p], [1, 1, 1] if t == 1 else [0, 0, 1])]
    topk_tensor = upload(topk)
    op = _build_pallas_sparse_mla_kernel_op(qblock_kpool=0)
    for iteration, sign in enumerate((1, -1)):
        # Exactly representable FP8 dyadics. Each query selects only itself:
        # softmax has one element, so its output must be the selected value.
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
            actual = out.cpu().float().numpy()
            actual_cn = cn.cpu().numpy()
            actual_cr = cr.cpu().numpy()
        expected = np.broadcast_to(values[:, None, :], (t, h, 512))
        assert actual.shape == (t, h, 512), actual.shape
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1 / 128)
        expected_cn = np.zeros((2, p, 512), np.uint8)
        expected_cr = np.zeros((2, p, 128), np.uint8)
        expected_cn[1, :t] = vn_cpu.view(torch.uint8).numpy()
        expected_cr[1, :t, :64] = vr_cpu.view(torch.uint8).numpy()
        np.testing.assert_array_equal(actual_cn.reshape(2, p, 512), expected_cn)
        np.testing.assert_array_equal(actual_cr.reshape(2, p, 128), expected_cr)
        print(f'PRODUCTION_SPARSE_STEP_PASS iteration={iteration} queries={t}', flush=True)


if __name__ == '__main__':
    main()
