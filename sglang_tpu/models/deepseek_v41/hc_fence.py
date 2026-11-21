# SPDX-License-Identifier: Apache-2.0
"""An XLA optimization barrier on the hyper-connection state before its mixing
statistics (DeepSeek-V4 mHC, deepseek_v4_mhc._mix_stats_impl).

Why: the mixing GEMM reads the flattened state [T, hc_mult * hidden]; when its
     producer (the previous sublayer's post-combine) is still unmaterialized, XLA
     folds it into the GEMM operand in a transposed layout ({0,1}) that the TPU
     fusion emitter cannot window ("Cannot find a valid window for ...
     bf16[512,20480]{0,1}", fusion_emitter.cc), aborting the compile.
How: the state passes through jax.lax.optimization_barrier first, so it is
     materialized in its own layout. Values are unchanged.
"""

# PEP 563 must stay off here: cpu_export_jax_op reads the wrapper's jax.Array
# annotations (see sglang_tpu/kimi_k3/oproj_row.py).

import torch

_OP = None


def hc_fence(x: torch.Tensor) -> torch.Tensor:
    global _OP
    if _OP is None:
        import jax

        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        def _fence(x: jax.Array) -> jax.Array:
            return jax.lax.optimization_barrier(x)

        op = cpu_export_jax_op("sglang_tpu_pallas::dsv4_hc_mix_fence", _fence)
        op.register_fake(lambda x: torch.empty_like(x))
        _OP = op
    return _OP(x)
