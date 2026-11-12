# SPDX-License-Identifier: Apache-2.0
"""ShortConv request-to-state mapping and graph metadata lifecycle on CPU."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    ShortConvHybridAttnBackend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_short_conv import TpuShortConvAttnBackend, short_conv
from sglang_tpu.compilation.hybrid_linear_direct_compile import (
    bind_traced_attention_metadata,
    traced_attention_metadata,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _backend():
    # Keep the actual backend methods; only the runner and pool allocation are
    # omitted. Request rows -> virtual slots -> physical state rows are distinct.
    backend = TpuShortConvAttnBackend.__new__(TpuShortConvAttnBackend)
    request_to_virtual = torch.tensor([0, 2, 1, 3])
    virtual_to_physical = torch.tensor([4, 3, 1, 2])
    layers = [
        SimpleNamespace(conv=[torch.arange(30).reshape(5, 2, 1, 3).float()]),
        SimpleNamespace(conv=[torch.full((5, 2, 1, 3), 100.0)]),
    ]
    backend.req_to_token_pool = SimpleNamespace(
        get_mamba_indices=lambda rows: request_to_virtual[rows],
        translate_mamba_indices=lambda slots: virtual_to_physical[slots],
        mamba_map={4: 1, 9: 0},
        mamba_pool=SimpleNamespace(mamba2_layer_cache=lambda layer: layers[layer]),
    )
    backend._metadata = None
    backend._decode_query_start_loc = {}
    return backend


class TestShortConvBackend(CustomTestCase):
    def test_padded_prefill_maps_tokens_and_physical_slots(self):
        backend = _backend()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.tensor([2, 3, 1, -1, 0]),
            input_ids=torch.zeros(8, dtype=torch.int64),
            extend_seq_lens=torch.tensor([2, 0, 1, 0, 0], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([0, 0, 7, 0, 0]),
            extend_start_loc=torch.tensor([0, 2, 2, 0, 0]),
        )
        backend.init_forward_metadata(batch)
        metadata = backend.conv_state_metadata(9, batch)
        self.assertEqual(metadata.cache_indices.tolist(), [3, 2, 1, 0, 0])
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 2, 2, 3, 3, 3])
        self.assertEqual(
            metadata.has_initial_state.tolist(), [False, False, True, False, False]
        )
        self.assertEqual(
            metadata.token_request_indices.tolist(), [0, 0, 2, 0, 0, 0, 0, 0]
        )
        self.assertEqual(
            metadata.valid_tokens.tolist(),
            [True, True, True, False, False, False, False, False],
        )

    def test_decode_refreshes_slots_within_the_same_batch_bucket(self):
        backend = _backend()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.tensor([2, -1, 1, 0]),
        )
        backend.init_forward_metadata(batch)
        first = backend.traced_metadata()
        batch.req_pool_indices = torch.tensor([1, 2, -1, 0])
        backend.init_forward_metadata_out_graph(batch)
        current = backend.traced_metadata()
        self.assertEqual(first.cache_indices.tolist(), [3, 0, 1, 0])
        self.assertEqual(current.cache_indices.tolist(), [1, 3, 0, 0])
        self.assertEqual(current.query_start_loc.tolist(), [0, 1, 2, 3, 4])

        state = backend.conv_state_metadata(9, batch).layer_cache.conv[0]
        previous = state.clone()
        actual = short_conv(
            torch.ones(4, 3),
            torch.ones(3, 3),
            None,
            backend.conv_state_metadata(9, batch),
            ForwardMode.DECODE,
        )
        expected = torch.stack(
            (
                previous[1, :, 0].sum(0) + 1,
                previous[3, :, 0].sum(0) + 1,
                torch.zeros(3),
                torch.zeros(3),
            )
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(state[[0, 2, 4]], previous[[0, 2, 4]])

    def test_traced_metadata_restores_after_error_and_keeps_layers_separate(self):
        backend = _backend()
        full_backend = AttentionBackend()
        full_backend.req_to_token_pool = backend.req_to_token_pool
        full_backend.token_to_kv_pool = None
        hybrid = ShortConvHybridAttnBackend(full_backend, backend, [0])
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.tensor([1]),
        )
        backend.init_forward_metadata(batch)
        full_backend.forward_metadata = object()
        captured = traced_attention_metadata(hybrid)
        batch.req_pool_indices = torch.tensor([2])
        backend.init_forward_metadata(batch)
        full_backend.forward_metadata = object()
        current_full = full_backend.forward_metadata
        current = backend.traced_metadata()
        other_layer = backend.conv_state_metadata(4, batch).layer_cache.conv[0]
        other_before = other_layer.clone()
        state = backend.conv_state_metadata(9, batch).layer_cache.conv[0]
        previous = state.clone()

        with self.assertRaisesRegex(RuntimeError, "trace failed"):
            with bind_traced_attention_metadata(hybrid, captured, ForwardMode.DECODE):
                self.assertIs(full_backend.forward_metadata, captured[0])
                actual = short_conv(
                    torch.ones(1, 3),
                    torch.ones(3, 3),
                    None,
                    hybrid.conv_state_metadata(9, batch),
                    ForwardMode.DECODE,
                )
                torch.testing.assert_close(actual[0], previous[1, :, 0].sum(0) + 1)
                raise RuntimeError("trace failed")

        self.assertIs(backend.traced_metadata(), current)
        self.assertIs(full_backend.forward_metadata, current_full)
        torch.testing.assert_close(state[[0, 2, 3, 4]], previous[[0, 2, 3, 4]])
        torch.testing.assert_close(other_layer, other_before)

    def test_speculative_modes_fail_before_mutating_state(self):
        backend = _backend()
        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(NotImplementedError, "speculation"),
            ):
                backend.init_forward_metadata(SimpleNamespace(forward_mode=mode))


if __name__ == "__main__":
    unittest.main()
