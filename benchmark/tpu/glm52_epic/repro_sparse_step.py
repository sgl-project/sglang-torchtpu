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
    parser.add_argument('--selection', choices=('self', 'causal-mean', 'rope-pair', 'prefix-pair'), default='self',
                        help='Single key, or causal keys with zero queries and uniform weights.')
    parser.add_argument('--entry', choices=('per-query', 'prefill-fallback', 'prefill-qblock', 'prefill-long-table'), default='per-query',
                        help='Token-major, small prefill fallback,2048-token qblock, or481-page static-table fallback.')
    parser.add_argument('--decode-capacity', action='store_true',
                        help='Per-query self probe only: page64 and2112-page static request table (context135168). Not full-model decode.')
    parser.add_argument('--dynamo', action='store_true',
                        help='Compile the production op fullgraph with backend=tpu, static shapes.')
    parser.add_argument('--prefix-tokens', type=int, choices=(0, 49152), default=0,
                        help='Long-table probe only: shift live writes/selected keys to high logical pages; prefix cache remains unselected zero.')
    parser.add_argument('--requests', type=int, choices=(1, 2), default=1,
                        help='Two disjoint long-table prefix-pair requests,1024 queries each.')
    args = parser.parse_args()
    if args.decode_capacity and (args.entry != 'per-query' or args.queries != 1
                                 or args.selection != 'self' or args.requests != 1
                                 or args.prefix_tokens != 0):
        parser.error('decode-capacity requires per-query queries1 self requests1 prefix0')
    if args.requests == 2 and args.selection != 'prefix-pair':
        parser.error('two-request probe requires prefix-pair')
    if args.prefix_tokens and args.entry != 'prefill-long-table':
        parser.error('nonzero prefix requires prefill-long-table')
    if args.selection == 'prefix-pair' and args.prefix_tokens != 49152:
        parser.error('prefix-pair requires long-table prefix49152')
    large_prefill = args.entry in ('prefill-qblock', 'prefill-long-table')
    if large_prefill != (args.queries == 2048):
        parser.error('qblock/long-table entries require queries2048; other entries retain small probes')
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

    t, h, p, k = args.queries, 64, 64 if args.decode_capacity else 128, 2048
    print('STAGE allocate', flush=True)
    def upload(a):
        return torch.from_numpy(a).to('tpu')

    query_value = 1 if args.selection == 'self' else 0
    q = torch.full((t, h, 512), query_value, dtype=torch.bfloat16, device='tpu')
    qr = torch.full((t, h, 64), 1 if args.selection == 'rope-pair' else query_value,
                    dtype=torch.bfloat16, device='tpu')
    # Whole allocations, not views: the production op donates these buffers.
    # Long-table isolates static context capacity. Only16 pages contain new
    # nonzero tokens; optional prefix shifts them to logical pages384..399.
    # Prefix cache remains zero/unselected: this is not realistic long attention.
    table_pages = 481 if args.entry == 'prefill-long-table' else 33
    num_pages = (1 + 481 * args.requests) if args.entry == 'prefill-long-table' else (65 if large_prefill else 2)
    if args.decode_capacity:
        table_pages = 135168 // p
        num_pages = table_pages + 1
        print(f'DECODE_CAPACITY page_size={p} pages_per_seq={table_pages} live_kv=1', flush=True)
    per_request = t // args.requests
    token_request = np.arange(t) // per_request
    local_row = np.arange(t) % per_request
    cn = torch.zeros((num_pages, p, 4, 128), dtype=torch.uint8, device='tpu')
    cr = torch.zeros((num_pages, p // 4, 4, 128), dtype=torch.uint8, device='tpu')
    topk = np.full((t, k), -1, np.int32)
    if args.selection == 'self':
        topk[:, 0] = np.arange(t)
    elif args.selection == 'rope-pair':
        # Unique causal keys, including row0 only once (no bitmap duplicates).
        topk[:, 0] = 0
        topk[1:, 1] = np.arange(1, t)
    else:
        for row in range(t):
            topk[row, :row + 1] = np.arange(row + 1)
    # Preserve padding -1 while moving selected keys across page-gather chunk
    # boundaries (prefix49152/page128 = logical page384, fourth128-page chunk).
    topk = np.where(topk >= 0, topk + args.prefix_tokens, topk)
    if args.selection == 'prefix-pair':
        topk[:] = -1
        topk[:, 0] = 0
        topk[:, 1] = args.prefix_tokens + local_row
    live_kv = args.prefix_tokens + per_request
    head_major = args.entry == 'prefill-fallback' or large_prefill
    distribution = [1, 1, 1] if t == 1 and not head_major else [0, 0, args.requests]
    pages = list(range(1, table_pages * args.requests + 1)) if large_prefill or args.decode_capacity else [1]
    print(f'SPARSE_REQUESTS count={args.requests} queries_per_request={per_request}', flush=True)
    print(f'SPARSE_METADATA pages_per_seq={len(pages) // args.requests} live_kv={live_kv} prefix={args.prefix_tokens} '
          f'expected_prefill_path={"per-query" if len(pages) > 256 else "shape-dependent"}',
          flush=True)
    metadata = [upload(np.array(a, np.int32)) for a in (
        [live_kv] * args.requests, pages,
        list(range(0, t + 1, per_request)),
        [i * (live_kv if large_prefill else p) for i in range(args.requests + 1)],
        distribution)]
    topk_tensor = upload(topk)
    prefix_nope = prefix_rope = None
    if args.selection == 'prefix-pair':
        # Seed whole donated allocations once. Logical position0 maps to page1.
        # Fixed prefix values deliberately differ from both signed new writes.
        prefix_nope = torch.full((512,), 0.25).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        prefix_rope = torch.full((64,), -0.5).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        seed_n = np.zeros((num_pages * p, 512), np.uint8)
        seed_r = np.zeros((num_pages * p, 128), np.uint8)
        for request in range(args.requests):
            base = p * (1 + request * table_pages)
            # Request1 uses negative prefix to detect request0 cache leakage.
            n = prefix_nope if request == 0 else torch.full((512,), -0.25).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
            seed_n[base] = n
            seed_r[base, :64] = prefix_rope
        cn = upload(seed_n.reshape(num_pages, p, 4, 128))
        cr = upload(seed_r.reshape(num_pages, p // 4, 4, 128))
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
        if args.selection == 'prefix-pair':
            prefix_values = np.where(token_request[:, None] == 0, 0.25, -0.25)
            expected_rows = (values + prefix_values) / 2
        elif args.selection == 'causal-mean':
            expected_rows = values.cumsum(axis=0) / np.arange(1, t + 1)[:, None]
        elif args.selection == 'rope-pair':
            # NoPE query=0, RoPE query=1 over64 lanes, scale1/16. All
            # stored FP8 values are exact dyadics, so logits are4*rope_scalar.
            delta = 4 * (rope[:, :1] - rope[0, :1])
            w_self = 1 / (1 + np.exp(-delta))
            expected_rows = (1 - w_self) * values[0] + w_self * values
            expected_rows[0] = values[0]
        expected = np.broadcast_to(expected_rows[:, None, :], (t, h, 512))
        assert actual.shape == (t, h, 512), actual.shape
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1 / 128)
        expected_cn = np.zeros((num_pages * p, 512), np.uint8)
        expected_cr = np.zeros((num_pages * p, 128), np.uint8)
        for request in range(args.requests):
            base = p * (1 + request * table_pages) if large_prefill else p
            if prefix_nope is not None:
                expected_cn[base] = seed_n[base]
                expected_cr[base, :64] = prefix_rope
            write_start = base + args.prefix_tokens
            rows = slice(request * per_request, (request + 1) * per_request)
            expected_cn[write_start:write_start + per_request] = vn_cpu.view(torch.uint8).numpy()[rows]
            expected_cr[write_start:write_start + per_request, :64] = vr_cpu.view(torch.uint8).numpy()[rows]
        np.testing.assert_array_equal(actual_cn.reshape(num_pages * p, 512), expected_cn)
        np.testing.assert_array_equal(actual_cr.reshape(num_pages * p, 128), expected_cr)
        print(f'PRODUCTION_SPARSE_STEP_PASS iteration={iteration} queries={t} selection={args.selection} entry={args.entry}', flush=True)


if __name__ == '__main__':
    main()
