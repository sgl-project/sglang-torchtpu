#!/usr/bin/env python3
"""Local multi-device transfer/collective boundary, not multi-host serving.

Launch via torch.tpu.distributed discovery and torch.distributed.run, as the
registered TPU unit tests do. All ranks enter the same device collectives;
asymmetric mode varies only local work between Gloo coordination barriers.
"""
import argparse
import os
import sys
from datetime import timedelta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('roundtrip', 'all-reduce', 'asymmetric'), required=True)
    args = parser.parse_args()
    if 'jax' in sys.modules:
        raise RuntimeError('fresh process required')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import torch
    import torch_tpu  # noqa: F401
    import torch.distributed as dist
    from torch_tpu._internal import execution_mode

    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_NEVER
    dist.init_process_group(backend='tpu_dist', timeout=timedelta(seconds=120))
    cpu_group = None
    try:
        cpu_group = dist.new_group(backend='gloo', timeout=timedelta(seconds=120))
        rank, world = dist.get_rank(), dist.get_world_size()
        print(f'BOUNDARY rank={rank} world={world} mode={execution_mode.eager_mode} case={args.case}', flush=True)
        for iteration in range(2):
            dist.barrier(group=cpu_group)
            active = args.case != 'asymmetric' or rank % 2 == iteration
            if active:
                expected = torch.arange(256, dtype=torch.int32) + rank + iteration
                value = expected.to('tpu', non_blocking=True)
                if args.case == 'all-reduce':
                    # Every rank takes this branch. Keep integer dtype separate
                    # from model numerics; expected is the exact rank-sum formula.
                    dist.all_reduce(value)
                    expected = (torch.arange(256, dtype=torch.int32) + iteration) * world + world * (world - 1) // 2
                print(f'STAGE readback rank={rank} iteration={iteration}', flush=True)
                actual = value.cpu()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                event = torch.tpu.Event()
                event.record()
                event.synchronize()
            dist.barrier(group=cpu_group)
            print(f'DISTRIBUTED_BOUNDARY_PASS rank={rank} iteration={iteration} active={active}', flush=True)
    finally:
        # No failure-path barrier: peers may already have failed. Outer timeout
        # still bounds native destroy calls that fail to return.
        try:
            if cpu_group is not None:
                dist.destroy_process_group(cpu_group)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == '__main__':
    # Match the registered multi-process TPU tests: native StopSession at exit
    # can wedge subsequent TPU initialization. This probe does not test teardown.
    import traceback
    status = 1
    try:
        main()
        status = 0
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
