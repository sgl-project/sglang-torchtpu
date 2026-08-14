"""Tests for the TPU Pallas GDN linear-attention custom-op bridge.

Parallel to :mod:`test.manual.tpu.test_pallas_moe_op_registration` and
:mod:`test.manual.tpu.test_gdn_attention_v3` (the raw-kernel correctness
test). This file pins the torch<->JAX bridge in
:mod:`sglang.srt.hardware_backend.tpu.attention_gdn_pallas`:

  * a torch.library custom op ``sglang_tpu_pallas::gdn_attention`` is
    registered and reachable via ``torch.ops.sglang_tpu_pallas``;
  * ``build_gdn_pallas_op`` is idempotent (the ``_PALLAS_GDN_OP`` singleton
    short-circuits — torch.library rejects double registration);
  * the op's output matches the underlying JAX kernel, and it mutates
    ``conv_state`` / ``recurrent_state`` in place (the state write-back that
    the hybrid backend relies on).

Requires torch_tpu + JAX with a TPU backend visible to ``jax.devices()``
because building the op does a JAX export at registration time.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

import unittest

import numpy as np
import torch

from sglang.test.test_utils import CustomTestCase

try:
    import jax  # noqa: F401
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:  # pragma: no cover
    _HAS_JAX = False


def _has_tpu_via_jax() -> bool:
    if not _HAS_JAX:
        return False
    try:
        return any(getattr(d, "platform", "") == "tpu" for d in jax.devices())
    except Exception:
        return False


_SKIP_REASON = "Requires torch_tpu + JAX with a TPU backend visible to jax.devices()"


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestGDNAttentionOp(CustomTestCase):

    N_KQ = 2
    N_V = 8
    D_K = 128
    D_V = 128
    K = 4

    def test_op_registered_and_idempotent(self):
        from sglang.srt.hardware_backend.tpu.attention_gdn_pallas import (
            build_gdn_pallas_op,
        )

        op1 = build_gdn_pallas_op()
        op2 = build_gdn_pallas_op()
        self.assertIs(op1, op2, "builder must be idempotent (singleton)")
        self.assertTrue(hasattr(torch.ops, "sglang_tpu_pallas"))
        self.assertTrue(hasattr(torch.ops.sglang_tpu_pallas, "gdn_attention"))

    def test_op_matches_kernel_and_mutates_state(self):
        from sglang.srt.hardware_backend.tpu.attention_gdn_pallas import (
            build_gdn_pallas_op,
        )
        from sglang.srt.layers.attention.pallas.gdn import fused_conv1d_gdn

        n_kq, n_v, d_k, d_v, K = self.N_KQ, self.N_V, self.D_K, self.D_V, self.K
        # Mixed batch: 2 decode + 1 prefill.
        lengths = [1, 1, 64]
        q_loc = [0, 1, 2, 66]
        distribution = [2, 3, 3]
        max_reqs = 3
        num_tokens = sum(lengths)
        num_blocks = max_reqs + 1
        conv_dim = n_kq * d_k * 2 + n_v * d_v

        rng = np.random.default_rng(0)

        def rn(*shape):
            return rng.standard_normal(shape).astype(np.float32)

        qkv = rn(num_tokens, conv_dim)
        b = rn(num_tokens, n_v)
        a = rn(num_tokens, n_v)
        conv_state = np.zeros((num_blocks, K - 1, conv_dim), np.float32)
        recurrent_state = np.zeros((num_blocks, n_v, d_k, d_v), np.float32)
        conv_weight = rn(conv_dim, 1, K)
        conv_bias = rn(conv_dim)
        a_log = rn(n_v)
        dt_bias = rn(n_v)
        query_start_loc = np.array(q_loc, np.int32)
        state_indices = np.arange(1, max_reqs + 1, dtype=np.int32)
        distribution_np = np.array(distribution, np.int32)
        seq_lens = np.array(lengths, np.int32)  # fresh slots (context_len 0)

        # Reference: run the JAX kernel directly.
        (conv_ref, rec_ref), out_ref = fused_conv1d_gdn(
            jnp.asarray(qkv),
            jnp.asarray(b),
            jnp.asarray(a),
            jnp.asarray(conv_state),
            jnp.asarray(recurrent_state),
            jnp.asarray(conv_weight),
            jnp.asarray(conv_bias),
            jnp.asarray(a_log),
            jnp.asarray(dt_bias),
            jnp.asarray(query_start_loc),
            jnp.asarray(state_indices),
            jnp.asarray(distribution_np),
            jnp.asarray(seq_lens),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=K,
        )

        # Bridge: run the torch custom op on TPU tensors.
        op = build_gdn_pallas_op()
        dev = torch.device("tpu")

        def tt(x):
            return torch.from_numpy(np.ascontiguousarray(x)).to(dev)

        conv_t = tt(conv_state)
        rec_t = tt(recurrent_state)
        out_t = op(
            tt(qkv),
            tt(b),
            tt(a),
            conv_t,
            rec_t,
            tt(conv_weight),
            tt(conv_bias),
            tt(a_log),
            tt(dt_bias),
            torch.from_numpy(query_start_loc).to(dev),
            torch.from_numpy(state_indices).to(dev),
            torch.from_numpy(distribution_np).to(dev),
            torch.from_numpy(seq_lens).to(dev),
            n_kq,
            n_v,
            d_k,
            d_v,
            K,
        )

        np.testing.assert_allclose(
            out_t.cpu().numpy(), np.asarray(out_ref), rtol=2e-2, atol=2e-2
        )
        # conv_state / recurrent_state were donated + aliased => mutated.
        np.testing.assert_allclose(
            conv_t.cpu().numpy(), np.asarray(conv_ref), rtol=2e-2, atol=2e-2
        )
        np.testing.assert_allclose(
            rec_t.cpu().numpy(), np.asarray(rec_ref), rtol=2e-2, atol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
