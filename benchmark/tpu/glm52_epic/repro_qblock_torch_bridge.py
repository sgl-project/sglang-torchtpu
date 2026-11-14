"""Model-free qblock tile probe via the production TorchTPU export bridge.

JAX stays CPU-only; TorchTPU owns PJRT. A zero-cache oracle checks execution
and padding, not general softmax accuracy. Use a fresh process per tile.
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-block', type=int, choices=(32, 128), required=True)
    args = parser.parse_args()
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import torch
    import torch_tpu  # noqa: F401
    import jax
    from torch_tpu._internal import pallas
    from torch_tpu._internal import execution_mode
    from sglang_tpu.attention.pallas.mla_sparse_qblock_tokens import (
        sparse_mla_prefill_qblock_tokens,
    )
    from sglang_tpu.attention_pallas_mla_sparse import tpu_info_for_cpu_export

    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_NEVER

    def wrapped(q: jax.Array, qr: jax.Array, cn: jax.Array, cr: jax.Array,
                topk: jax.Array, lens: jax.Array, pages: jax.Array,
                cu: jax.Array, distribution: jax.Array) -> jax.Array:
        return sparse_mla_prefill_qblock_tokens(
            q, qr, cn, cr, topk, lens, pages, cu, distribution,
            sm_scale=576 ** -0.5, k_scale=1.0,
            query_block=args.query_block, pages_per_step=2,
        )

    kernel = pallas.custom_jax_kernel(
        wrapped, name=f'glm52_qblock_probe_qb{args.query_block}',
        warn_deprecated=False,
    )
    print(f'QBLOCK_BRIDGE_BEGIN query_block={args.query_block}', flush=True)
    # Whole cache allocations, no mutation/donation in this attention-only probe.
    q = torch.zeros((64, 2048, 512), dtype=torch.bfloat16, device='tpu')
    qr = torch.zeros((64, 2048, 64), dtype=torch.bfloat16, device='tpu')
    cn = torch.zeros((65, 128, 4, 128), dtype=torch.uint8, device='tpu')
    cr = torch.zeros((65, 32, 4, 128), dtype=torch.uint8, device='tpu')
    topk = torch.full((2048, 2048), -1, dtype=torch.int32)
    topk[:, 0] = torch.arange(2048, dtype=torch.int32)
    values = [x.to('tpu') for x in (
        topk, torch.tensor([2048], dtype=torch.int32),
        torch.arange(1, 34, dtype=torch.int32),
        torch.tensor([0, 2048], dtype=torch.int32),
        torch.tensor([0, 0, 1], dtype=torch.int32),
    )]
    with torch.no_grad(), tpu_info_for_cpu_export():
        out = kernel(q, qr, cn, cr, *values).cpu()
    assert out.shape == (64, 2048, 512)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out) == 0
    print(f'QBLOCK_BRIDGE_ZERO_PASS query_block={args.query_block}', flush=True)


if __name__ == '__main__':
    main()
