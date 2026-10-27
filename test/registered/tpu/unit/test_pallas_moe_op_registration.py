"""Regression tests for the TPU Pallas fused-MoE custom-op registration.

Parallel to :mod:`test.manual.tpu.test_pallas_rpa_v3_dispatch` for the
attention backend -- this file pins the TPU-specific dispatch glue
that turns ``UnquantizedFusedMoEMethod.forward_tpu`` into a usable
torch op.

The behaviour being guarded here:

  * A torch.library custom op named ``sglang_tpu_pallas::fused_moe``
    is registered and reachable via ``torch.ops.sglang_tpu_pallas``.
  * ``_build_fused_moe_op`` is idempotent — torch.library rejects
    double-registration of the same op with a confusing error, so the
    module-level ``_PALLAS_MOE_OP`` singleton must short-circuit on
    subsequent calls.
These tests require torch_tpu + JAX with a TPU backend visible to
``jax.devices()`` because ``_build_fused_moe_op`` does JAX export at
registration time.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.test.test_utils import CustomTestCase

try:
    import jax  # noqa: F401

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
class TestPallasMoeOpRegistration(CustomTestCase):
    """The fused-MoE custom op must be registered under the expected
    qualified name so ``UnquantizedFusedMoEMethod.forward_tpu``'s call to
    ``fused_moe_call`` reaches the right kernel. A typo or namespace
    drift here would silently produce a separate, unregistered op."""

    def test_op_registered_under_expected_qualified_name(self):
        from sglang_tpu.moe.pallas.fused_moe_op import _build_fused_moe_op

        _build_fused_moe_op()
        # Reachable via the torch.ops namespace -- this is what dynamo
        # and the production model code path resolve at trace time.
        self.assertTrue(
            hasattr(torch.ops, "sglang_tpu_pallas"),
            "sglang_tpu_pallas namespace missing from torch.ops",
        )
        self.assertTrue(
            hasattr(torch.ops.sglang_tpu_pallas, "fused_moe"),
            "fused_moe op missing from sglang_tpu_pallas namespace",
        )


@unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON)
class TestPallasMoeOpIdempotency(CustomTestCase):
    """``_build_fused_moe_op`` is called during ``FusedMoE`` layer
    construction and again from ``fused_moe_call``. torch.library rejects
    double-registration of the same op, so subsequent calls must return
    the cached singleton instead of re-registering."""

    def test_build_returns_same_singleton(self):
        from sglang_tpu.moe.pallas.fused_moe_op import _build_fused_moe_op

        op_a = _build_fused_moe_op()
        op_b = _build_fused_moe_op()
        self.assertIs(op_a, op_b)


class TestPallasMoeCallContract(CustomTestCase):
    """``fused_moe_call`` must forward its arguments to the registered
    op in the op's exact positional order — no other test covers this
    boundary. Runs without a TPU: the op builder is mocked out.
    """

    def test_forwards_ep_and_sparse_core_options_to_registered_op(self):
        import sglang_tpu.moe.pallas.fused_moe_op as fused_moe_op

        hidden_states = torch.empty(2, 8, dtype=torch.bfloat16)
        w1 = torch.empty(2, 8, 16, dtype=torch.bfloat16)
        w2 = torch.empty(2, 8, 8, dtype=torch.bfloat16)
        topk_weights = torch.empty(2, 2, dtype=torch.bfloat16)
        topk_ids = torch.empty(2, 2, dtype=torch.int32)

        expected = torch.empty_like(hidden_states)
        op = Mock(return_value=expected)

        with (
            patch.object(fused_moe_op, "_build_fused_moe_op", return_value=op),
            patch.object(
                fused_moe_op.envs.SGLANG_TPU_FUSED_MOE_SC_GATHER,
                "get",
                return_value=True,
            ),
            patch.object(
                fused_moe_op.envs.SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD,
                "get",
                return_value=128,
            ),
        ):
            actual = fused_moe_op.fused_moe_call(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                topk=2,
                activation="silu",
                global_num_experts=8,
                expert_group_offset=4,
            )

        self.assertIs(actual, expected)
        # The tensor arguments must be the SAME objects passed above —
        # fresh equal-valued tensors would break assert_called_once_with.
        op.assert_called_once_with(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            None,  # w1_scale
            None,  # w2_scale
            None,  # w1_bias
            None,  # w2_bias
            2,  # topk
            "silu",  # activation
            False,  # no_combine
            8,  # global_num_experts
            4,  # expert_group_offset
            1,  # use_sc_gather (int(True))
            0,  # w_quant_bits (0 => not weight-only quantized)
            128,  # permute_dense_threshold
            None,  # w1_channel_scale (vmem-fp8 only)
            None,  # w2_channel_scale (vmem-fp8 only)
            None,  # expert_group_offset_tensor (lifted when trace_rank is active)
        )

    def test_lifts_expert_group_offset_when_trace_rank_is_active(self):
        import sglang_tpu.moe.pallas.fused_moe_op as fused_moe_op
        from sglang_tpu.rank_local import trace_rank

        hidden_states = torch.empty(2, 8, dtype=torch.bfloat16)
        w1 = torch.empty(2, 8, 16, dtype=torch.bfloat16)
        w2 = torch.empty(2, 8, 8, dtype=torch.bfloat16)
        topk_weights = torch.empty(2, 2, dtype=torch.bfloat16)
        topk_ids = torch.empty(2, 2, dtype=torch.int32)
        op = Mock(return_value=hidden_states)
        rank_tensor = torch.tensor([3], dtype=torch.int32)

        with (
            patch.object(fused_moe_op, "_build_fused_moe_op", return_value=op),
            patch("sglang_tpu.rank_local.dist.get_rank", return_value=3),
            trace_rank(rank_tensor),
        ):
            fused_moe_op.fused_moe_call(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                topk=2,
                activation="silu",
                global_num_experts=8,
                expert_group_offset=6,
            )

        self.assertEqual(op.call_args.args[13], 0)
        offset_tensor = op.call_args.args[19]
        self.assertIsInstance(offset_tensor, torch.Tensor)
        torch.testing.assert_close(offset_tensor, torch.tensor([6], dtype=torch.int32))

    def test_situ_params_are_encoded_in_the_activation(self):
        """SiTU (beta, linear_beta) reach the op in the activation string."""
        import sglang_tpu.moe.pallas.fused_moe_op as fused_moe_op

        t = torch.empty(2, 2)
        op = Mock(return_value=t)
        call = dict(hidden_states=t, w1=t, w2=t, topk_weights=t, topk_ids=t, topk=2)
        with patch.object(fused_moe_op, "_build_fused_moe_op", return_value=op):
            fused_moe_op.fused_moe_call(
                **call, activation="situ", gemm1_alpha=4.0, gemm1_clamp_limit=25.0
            )
            self.assertEqual(op.call_args.args[10], "situ:4.0:25.0")
            fused_moe_op.fused_moe_call(**call, activation="situ", gemm1_alpha=4.0)
            self.assertEqual(op.call_args.args[10], "situ:4.0:none")
            # A missing beta must fail, not silently default.
            with self.assertRaises(ValueError):
                fused_moe_op.fused_moe_call(**call, activation="situ")


if __name__ == "__main__":
    unittest.main()
