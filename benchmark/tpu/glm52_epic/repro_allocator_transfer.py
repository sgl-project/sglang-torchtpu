#!/usr/bin/env python3
"""CPU production page expansion -> int32 H2D -> blocking D2H isolation.

This covers the producer/transfer boundary seen in host request-table stacks,
not allocator ownership/eviction or concurrent model/collective work.
"""
import os
import sys


def main():
    if 'jax' in sys.modules:
        raise RuntimeError('fresh process required')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import execution_mode
    from sglang_tpu.allocator_tpu import alloc_extend_vectorized_cpu

    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_NEVER
    print(f'EAGER_MODE {execution_mode.eager_mode}', flush=True)
    for prefix, count in ((0, 4), (0, 256), (3, 4), (127, 4)):
        # Existing prefix resides on physical page1; free pages start at2.
        last = 128 + prefix - 1 if prefix else -1
        need = (prefix + count + 127) // 128 - (prefix + 127) // 128
        pages = list(range(2, 2 + need))
        oracle = []
        for pos in range(prefix, prefix + count):
            if prefix and pos // 128 == (prefix - 1) // 128:
                oracle.append(128 + pos % 128)
            else:
                first_new_logical_page = (prefix + 127) // 128
                oracle.append(pages[pos // 128 - first_new_logical_page] * 128 + pos % 128)
        for iteration in range(2):
            out = torch.empty(count, dtype=torch.int32)
            alloc_extend_vectorized_cpu(
                torch.tensor([prefix], dtype=torch.int64),
                torch.tensor([prefix + count], dtype=torch.int64),
                torch.tensor([last], dtype=torch.int64),
                torch.tensor(pages, dtype=torch.int32), out, 128)
            torch.testing.assert_close(out, torch.tensor(oracle, dtype=torch.int32), rtol=0, atol=0)
            print(f'STAGE transfer prefix={prefix} count={count} iteration={iteration}', flush=True)
            device_out = out.to('tpu', non_blocking=True)
            actual = device_out.cpu()
            torch.testing.assert_close(actual, out, rtol=0, atol=0)
            print(f'ALLOCATOR_TRANSFER_PASS prefix={prefix} count={count} iteration={iteration}', flush=True)


if __name__ == '__main__':
    main()
