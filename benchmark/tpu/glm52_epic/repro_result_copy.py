#!/usr/bin/env python3
"""Narrow production async_d2h/Event isolation; no model or distributed groups."""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('copy', 'event', 'copy-event'), required=True)
    parser.add_argument('--elements', type=int, choices=(1, 64, 512), default=1)
    parser.add_argument('--producer', choices=('upload', 'argmax'), default='upload')
    args = parser.parse_args()
    if 'jax' in sys.modules:
        raise RuntimeError('fresh process required')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import torch
    import torch_tpu  # noqa: F401
    from sglang.srt.utils.common import async_d2h
    from torch_tpu._internal import execution_mode

    print(f'EAGER_MODE initial={execution_mode.eager_mode}', flush=True)
    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_NEVER
    print(f'EAGER_MODE effective={execution_mode.eager_mode}', flush=True)

    for iteration in range(2):
        expected = torch.arange(args.elements, dtype=torch.int64) + iteration
        print(f'STAGE upload iteration={iteration} case={args.case}', flush=True)
        if args.producer == 'argmax':
            # Real device computation yielding int64 token IDs; no data reads
            # or synchronization between dispatch and result-copy boundary.
            logits = torch.full((args.elements, args.elements + 2), -1.0)
            logits.scatter_(1, expected[:, None], 1.0)
            value = logits.to('tpu').argmax(dim=-1)
        else:
            value = expected.to('tpu')
        # Record actual predicate rather than assuming CUDA pinning semantics.
        print(f'BRANCH device={value.device} is_cuda={value.is_cuda}', flush=True)
        event = torch.tpu.Event() if args.case != 'copy' else None
        print('STAGE copy_or_record', flush=True)
        host = async_d2h(value) if args.case != 'event' else None
        if event is not None:
            event.record()
            event.synchronize()
        else:
            torch.tpu.synchronize()
        if host is not None:
            torch.testing.assert_close(host, expected, rtol=0, atol=0)
        else:
            # Only after the event fence: keep this read outside tested copy path.
            torch.testing.assert_close(value.cpu(), expected, rtol=0, atol=0)
        print(f'RESULT_COPY_PASS case={args.case} producer={args.producer} elements={args.elements} iteration={iteration}',
              flush=True)


if __name__ == '__main__':
    main()
