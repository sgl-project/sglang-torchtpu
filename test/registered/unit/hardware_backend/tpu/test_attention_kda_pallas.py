"""The TPU KDA backend hands the fused op KDA mode, the layer's gate bound and
the kernel's per-channel layouts."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_kda_pallas import TpuKDAAttnBackend

_N_V, _D_K, _D_V, _K, _T = 2, 4, 4, 4, 3
_DIM = 2 * _N_V * _D_K + _N_V * _D_V


def _run(lower_bound, conv_weights=None, is_decode=True, **layer_attrs):
    conv_state, rec_state = (
        torch.zeros(4, _K - 1, 1, _DIM),
        torch.zeros(4, _N_V, _D_K, _D_V),
    )
    cache = SimpleNamespace(conv=[conv_state], temporal=rec_state)
    layer = SimpleNamespace(
        layer_id=0,
        lower_bound=lower_bound,
        num_k_heads=_N_V,
        num_v_heads=_N_V,
        head_k_dim=_D_K,
        head_v_dim=_D_V,
        conv_weights=torch.zeros(_DIM, _K) if conv_weights is None else conv_weights,
        bias=None,
        A_log=torch.zeros(_N_V),
        dt_bias=torch.zeros(_N_V * _D_K),
        **layer_attrs,
    )
    backend = TpuKDAAttnBackend.__new__(TpuKDAAttnBackend)
    backend.req_to_token_pool = SimpleNamespace(mamba2_layer_cache=lambda _: cache)
    backend._query_start_loc = backend._state_indices = backend._seq_lens = None
    backend._distribution = None
    backend._is_decode = is_decode
    backend._op = Mock(return_value=torch.zeros(_T, _N_V, _D_V))
    out = backend._run(
        layer,
        mixed_qkv=torch.zeros(_T, _DIM),
        # The model passes a [1, T, n_v, d_k] gate and a [1, T, n_v] beta.
        a=torch.zeros(1, _T, _N_V, _D_K),
        b=torch.zeros(1, _T, _N_V),
    )
    call = backend._op.call_args
    return call.args, out, (conv_state, rec_state), call.kwargs


class TestTpuKDAAttnBackend(CustomTestCase):
    def test_kda_mode_and_gate_bound_are_forwarded(self):
        for bound in (None, -5.0):
            with self.subTest(bound=bound):
                args, _, _, _ = _run(bound)
                # (..., n_kq, n_v, d_k, d_v, kernel_size, kda, gate_lower_bound)
                self.assertEqual(args[13:], (_N_V, _N_V, _D_K, _D_V, _K, True, bound))

    def test_kernel_layouts(self):
        args, out, (conv_state, rec_state), _ = _run(-5.0)
        self.assertEqual(tuple(args[1].shape), (_T, _N_V))
        self.assertEqual(tuple(args[2].shape), (_T, _N_V * _D_K))
        self.assertIs(args[3], conv_state)
        self.assertIs(args[4], rec_state)
        self.assertEqual(tuple(args[5].shape), (_DIM, 1, _K))
        self.assertEqual(tuple(args[8].shape), (_N_V, _D_K))
        # The model squeezes dim 0 after o_norm; a one-token decode must keep T.
        self.assertEqual(tuple(out.shape), (1, _T, _N_V, _D_V))

    def test_depthwise_conv_weight_passes_through(self):
        # Kimi-K3 binds its depthwise [dim, 1, K] parameter: passed as is.
        weight = torch.zeros(_DIM, 1, _K)
        args, _, _, _ = _run(-5.0, conv_weights=weight)
        self.assertIs(args[5], weight)
        self.assertEqual(args[17], _K)

    def test_single_pass_variant_follows_forward_mode(self):
        for is_decode in (True, False):
            with self.subTest(is_decode=is_decode):
                kwargs = _run(-5.0, is_decode=is_decode)[3]
                self.assertEqual(kwargs["batched_only"], is_decode)
                self.assertEqual(kwargs["prefill_only"], not is_decode)

    def test_packed_weights_are_optional(self):
        # Kimi-Linear / Bailing layers never set packed_kernel_weights.
        self.assertIsNone(_run(-5.0)[3]["packed_weights"])
        packed = torch.zeros(8, 128)
        kwargs = _run(-5.0, packed_kernel_weights=packed)[3]
        self.assertIs(kwargs["packed_weights"], packed)

    def test_fused_qkvg_is_opt_in(self):
        # Only a layer that declares the fused [q, k, v, g] input (Kimi-K3)
        # lets the op skip trailing qkv columns.
        self.assertFalse(_run(-5.0)[3]["fused_qkvg"])
        self.assertTrue(_run(-5.0, fused_qkvg=True)[3]["fused_qkvg"])


if __name__ == "__main__":
    unittest.main()
