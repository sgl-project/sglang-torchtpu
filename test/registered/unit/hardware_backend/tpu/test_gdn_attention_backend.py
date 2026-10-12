"""Unit tests for the TPU GDN attention backend's metadata (CPU-runnable).

These cover the pure-Python/torch part of
``sglang_tpu.attention_gdn_pallas.TpuGDNAttnBackend`` that
needs no JAX / torch_tpu / TPU device: the ``distribution`` dispatch vector the
fused kernel keys on.

``distribution = [decode_end, prefill_end, mixed_end]`` decides how the kernel
splits a ragged batch into decode / prefill regions. A wrong vector silently
routes tokens through the wrong phase (plausible-but-wrong output, not a crash),
so it is worth pinning directly: ``[bs, bs, bs]`` for a pure-decode batch,
``[0, bs, bs]`` for extend/prefill (matching the RPA v3 backend's convention),
int32, and cached per ``(is_decode, bs, device)``.

Kernel/numerics (including mixed prefill+decode batches) are gated on TPU by
``test/registered/tpu/unit/test_gdn_attention_v3.py`` (kernel vs. eager reference)
and ``test_gdn_attention_op.py`` (bridge op + state write-back). This file only
pins the CPU-side metadata contract.

The backend's ``__init__`` builds the JAX op eagerly and needs a ``ModelRunner``;
we instantiate a thin subclass that skips both — ``_get_distribution`` only
touches ``self._distribution_cache``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_gdn_pallas import TpuGDNAttnBackend


class _BackendNoInit(TpuGDNAttnBackend):
    """TpuGDNAttnBackend without the JAX-op build / ModelRunner dependency.

    ``_get_distribution`` reads only ``self._distribution_cache``, so a bare
    instance with that dict is enough to exercise the metadata logic on CPU.
    """

    def __init__(self):  # noqa: D107  (deliberately skips super().__init__)
        self._distribution_cache = {}


class TestGDNDistribution(CustomTestCase):
    def setUp(self):
        self.backend = _BackendNoInit()
        self.device = torch.device("cpu")

    def test_decode_distribution(self):
        # Pure-decode batch: every sequence is one step -> [bs, bs, bs].
        for bs in (1, 4, 37):
            dist = self.backend._get_distribution(True, bs, self.device)
            self.assertEqual(dist.tolist(), [bs, bs, bs])
            self.assertEqual(dist.dtype, torch.int32)
            self.assertEqual(dist.device.type, "cpu")

    def test_extend_distribution(self):
        # Extend/prefill batch: no decode seqs -> [0, 0, bs]. The middle element
        # is inert (the GDN wrapper reads only distribution[0] and [-1]); it is
        # written as num_decode to match tpu-inference's tpu_runner.py.
        for bs in (1, 4, 37):
            dist = self.backend._get_distribution(False, bs, self.device)
            self.assertEqual(dist.tolist(), [0, 0, bs])
            self.assertEqual(dist.dtype, torch.int32)

    def test_distribution_last_element_is_the_real_seq_count(self):
        # What the kernel actually depends on: seqs[distribution[0]:distribution[-1]]
        # bounds the non-decode range, so the last element must be the request
        # count in both modes.
        for is_decode in (True, False):
            for bs in (1, 8):
                dist = self.backend._get_distribution(is_decode, bs, self.device)
                self.assertEqual(dist.tolist()[-1], bs)

    def test_decode_and_extend_differ(self):
        bs = 8
        decode = self.backend._get_distribution(True, bs, self.device)
        extend = self.backend._get_distribution(False, bs, self.device)
        self.assertNotEqual(decode.tolist(), extend.tolist())

    def test_cached_identity_per_key(self):
        # Same (is_decode, bs, device) returns the cached tensor (not rebuilt);
        # a different key builds a distinct one.
        a = self.backend._get_distribution(True, 4, self.device)
        b = self.backend._get_distribution(True, 4, self.device)
        self.assertIs(a, b)
        c = self.backend._get_distribution(True, 5, self.device)
        self.assertIsNot(a, c)
        d = self.backend._get_distribution(False, 4, self.device)
        self.assertIsNot(a, d)


class TestGDNSinglePassFlags(CustomTestCase):
    """init_forward_metadata derives the kernel's single-pass flags from the
    same forward mode as the distribution: decode runs batched_only over
    [bs, bs, bs], extend runs prefill_only over [0, 0, bs]. A mismatch would run
    one pass over an empty range: zero output, no state update, no error."""

    def _run(self, forward_mode, bs=3):
        n_kq, n_v, d, k = 1, 2, 4, 4
        dim = 2 * n_kq * d + n_v * d
        backend = _BackendNoInit()
        backend._op = Mock(return_value=torch.zeros(bs, n_v, d))
        cache = SimpleNamespace(
            conv=[torch.zeros(bs + 1, k - 1, 1, dim)],
            temporal=torch.zeros(bs + 1, n_v, d, d),
        )
        backend.req_to_token_pool = SimpleNamespace(mamba2_layer_cache=lambda _: cache)

        def base_init(self, forward_batch):
            self.forward_metadata = SimpleNamespace(
                mamba_cache_indices=torch.arange(1, bs + 1),
                query_start_loc=torch.arange(bs + 1),
            )

        batch = SimpleNamespace(
            seq_lens=torch.full((bs,), 5), forward_mode=forward_mode
        )
        with patch.object(MambaAttnBackendBase, "init_forward_metadata", base_init):
            backend.init_forward_metadata(batch)
        layer = SimpleNamespace(
            layer_id=0,
            conv_weights=torch.zeros(dim, k),
            bias=None,
            A_log=torch.zeros(n_v),
            dt_bias=torch.zeros(n_v),
            num_k_heads=n_kq,
            num_v_heads=n_v,
            head_k_dim=d,
            head_v_dim=d,
        )
        backend._run(
            layer,
            mixed_qkv=torch.zeros(bs, dim),
            a=torch.zeros(bs, n_v),
            b=torch.zeros(bs, n_v),
        )
        kwargs = backend._op.call_args.kwargs
        distribution = backend._op.call_args.args[11].tolist()
        return distribution, kwargs["batched_only"], kwargs["prefill_only"]

    def test_flags_follow_the_distribution(self):
        bs = 3
        for mode, decode in (
            (ForwardMode.DECODE, True),
            (ForwardMode.IDLE, True),
            (ForwardMode.EXTEND, False),
            (ForwardMode.MIXED, False),
        ):
            with self.subTest(mode=mode.name):
                distribution, batched_only, prefill_only = self._run(mode, bs)
                self.assertEqual(distribution, [bs, bs, bs] if decode else [0, 0, bs])
                self.assertEqual(batched_only, decode)
                self.assertEqual(prefill_only, not decode)


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX")
class TestGDNRequestTilePadding(CustomTestCase):
    def test_partial_verify_tiles_have_inert_metadata_rows(self):
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.gdn.metadata import (
            compute_batched_seq_metadata,
        )

        for requests in (1, 3, 5):
            with self.subTest(requests=requests):
                md = compute_batched_seq_metadata(
                    cfg=SimpleNamespace(seq_tile_size=4, tile_size=4),
                    seq_lens=jnp.full((requests,), 19, dtype=jnp.int32),
                    query_start_loc=jnp.arange(requests + 1, dtype=jnp.int32) * 3,
                    state_indices=1 + jnp.arange(requests, dtype=jnp.int32) * 3,
                    read_offsets=jnp.zeros(requests, dtype=jnp.int32),
                    end_seq=jnp.asarray(requests, dtype=jnp.int32),
                )
                padded = ((requests + 3) // 4) * 4
                self.assertEqual(
                    md.p_id_to_r_size.data.tolist(),
                    [3] * requests + [0] * (padded - requests),
                )
                self.assertEqual(
                    md.p_id_is_first_tile.data.tolist(),
                    [True] * requests + [False] * (padded - requests),
                )
                self.assertEqual(
                    md.p_id_is_last_tile.data.tolist(),
                    [True] * requests + [False] * (padded - requests),
                )
                self.assertEqual(
                    md.p_id_to_s_idx.data.tolist(),
                    list(range(requests)) + [0] * (padded - requests),
                )


if __name__ == "__main__":
    unittest.main()
