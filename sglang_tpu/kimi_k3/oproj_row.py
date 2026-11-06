# SPDX-License-Identifier: Apache-2.0
"""Keep the single-token KDA / DSA o_proj GEMV in row form (opt-in).

Why: in the compiled decode graph XLA turns the [1, 384] o_proj input into a
     [384] vector and emits W @ v (operand [384, 1]{0,1}); that dot runs at
     1.88 us per KDA layer on 16 chips, while the same dot in row form
     (x[1, 384] @ W.T) measures 0.86 us standalone (1.30 us as column /
     vector), bit-identical.
How: SGLANG_TPU_K3_OPROJ_ROW. o_proj's quant method passes a single-token
     input through an XLA optimization barrier before the dot, so the [1, K]
     row cannot be folded into the producer as a vector. Values unchanged.
"""

# PEP 563 must stay off here: cpu_export_jax_op reads the wrapper's jax.Array
# annotations (see host_split_op.py).

import logging

import torch

logger = logging.getLogger(__name__)
_OP = None


def _row_op():
    global _OP
    if _OP is None:
        import jax

        from sglang_tpu.pallas_jax_op import cpu_export_jax_op

        def _row(x: jax.Array) -> jax.Array:
            return jax.lax.optimization_barrier(x)

        op = cpu_export_jax_op("sglang_tpu_pallas::k3_row_fence", _row)
        op.register_fake(lambda x: torch.empty_like(x))
        _OP = op
    return _OP


class _RowLinear:
    def __init__(self, orig, op):
        self.orig, self.op = orig, op

    def __getattr__(self, name):
        return getattr(self.orig, name)

    def apply(self, layer, x, bias=None):
        if x.dim() == 2 and x.shape[0] == 1:
            x = self.op(x)
        return self.orig.apply(layer, x, bias)


def bind_oproj_row(model) -> int:
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_K3_OPROJ_ROW.get() or getattr(
        model, "_oproj_row_bound", False
    ):
        return 0
    model._oproj_row_bound = True
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.layers.utils import PPMissingLayer

    n = 0
    for layer in model.model.layers:
        if isinstance(layer, PPMissingLayer):
            continue
        o = getattr(getattr(layer, "self_attn", None), "o_proj", None)
        qm = getattr(o, "quant_method", None) if o is not None else None
        if type(qm) is not UnquantizedLinearMethod:
            continue
        o.quant_method = _RowLinear(o.quant_method, _row_op())
        n += 1
    logger.info("Kimi-K3 single-token o_proj kept in row form: %d layers", n)
    return n
