"""Model-free compile-only reproduction of exp-9320106a77's VMEM failure.

Run only in the pinned TPU runtime, one process/device. Default reproduces
H64/T2048/page128, request capacity16, cache pages65 and topk2048. Candidate
--query-block 32 --pages-per-step 2 changes the query tile only. Compilation
success is NOT numerical validation or permission to change production tiles.
No model checkpoint, Torch distributed group, or Flash path is involved.
"""
import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-block', type=int, choices=(32, 64, 128), default=128)
    parser.add_argument('--pages-per-step', type=int, choices=(1, 2, 4, 8), default=2)
    args = parser.parse_args()
    import jax
    import jax.numpy as jnp
    from sglang_tpu.attention.pallas.mla_sparse_qblock_tokens import (
        sparse_mla_prefill_qblock_tokens,
    )

    devices = jax.devices()
    if not devices or devices[0].platform != 'tpu':
        raise RuntimeError('TPU backend required; CPU lowering is not VMEM evidence')
    spec = jax.ShapeDtypeStruct
    # Operand shapes copied from the failing HLO allocation diagnostic. RoPE's
    # external width is64; the wrapper pads it to128 before pallas_call.
    inputs = (
        spec((64, 2048, 512), jnp.bfloat16),
        spec((64, 2048, 64), jnp.bfloat16),
        spec((65, 128, 4, 128), jnp.uint8),
        spec((65, 32, 4, 128), jnp.uint8),
        spec((2048, 2048), jnp.int32),
        spec((16,), jnp.int32),
        spec((16 * 32,), jnp.int32),
        spec((17,), jnp.int32),
        spec((3,), jnp.int32),
    )

    def call(*values):
        return sparse_mla_prefill_qblock_tokens(
            *values, sm_scale=576 ** -0.5, k_scale=1.0,
            query_block=args.query_block, pages_per_step=args.pages_per_step,
        )

    started = time.monotonic()
    print('QBLOCK_COMPILE_BEGIN ' + json.dumps({
        'query_block': args.query_block, 'pages_per_step': args.pages_per_step,
        'jax_version': jax.__version__, 'device': str(devices[0]),
        'scope': 'compile-only; no runtime or numerical acceptance',
    }), flush=True)
    executable = jax.jit(call).lower(*inputs).compile()
    print('QBLOCK_COMPILE_PASS ' + json.dumps({
        'query_block': args.query_block, 'pages_per_step': args.pages_per_step,
        'elapsed_seconds': time.monotonic() - started,
        'compiled': executable is not None,
    }), flush=True)


if __name__ == '__main__':
    main()
