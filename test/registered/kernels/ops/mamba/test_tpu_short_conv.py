# SPDX-License-Identifier: Apache-2.0
"""ShortConv outputs and persistent history against a grouped-conv1d reference.

CPU cases check the math; the TPU registration also exercises device lowering
and in-place state writes. No model weights or server are needed.
"""

import importlib.util
import itertools
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_short_conv import TpuShortConvMetadata, short_conv

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_tpu_ci(
    est_time=120,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "lfm2", "short-conv"],
)


def _reference(x, weight, bias, state, slots, lengths, has_prefix):
    """Run one independent depthwise convolution per real request on CPU."""
    output = torch.zeros_like(x)
    next_state = state.clone()
    offset = 0
    for slot, length, prefix in zip(slots, lengths, has_prefix):
        if slot > 0 and length > 0:
            history = (
                state[slot, :, 0] if prefix else torch.zeros_like(state[slot, :, 0])
            )
            sequence = torch.cat((history, x[offset : offset + length].float()))
            result = F.conv1d(
                sequence.T.unsqueeze(0),
                weight.float().unsqueeze(1),
                bias=None if bias is None else bias.float(),
                groups=x.shape[1],
            )
            output[offset : offset + length] = result.squeeze(0).T.to(x.dtype)
            next_state[slot, :, 0] = sequence[-(weight.shape[1] - 1) :]
        offset += length
    return output, next_state


class TestShortConvCPU(CustomTestCase):
    device = "cpu"

    def setUp(self):
        self.generator = torch.Generator().manual_seed(42)

    def _randn(self, *shape, dtype=torch.float32):
        return torch.randn(*shape, generator=self.generator).to(dtype)

    def _prefill(self, x, weight, bias, state, slots, lengths, has_prefix):
        boundaries = [0, *itertools.accumulate(lengths)]
        requests = [row for row, length in enumerate(lengths) for _ in range(length)]
        padding = x.shape[0] - len(requests)
        metadata = TpuShortConvMetadata(
            layer_cache=SimpleNamespace(conv=[state]),
            cache_indices=torch.tensor(slots, device=self.device),
            query_start_loc=torch.tensor(boundaries, device=self.device),
            has_initial_state=torch.tensor(has_prefix, device=self.device),
            token_request_indices=torch.tensor(
                requests + [0] * padding, dtype=torch.int64, device=self.device
            ),
            valid_tokens=torch.tensor(
                [True] * len(requests) + [False] * padding, device=self.device
            ),
        )
        return short_conv(x, weight, bias, metadata, ForwardMode.EXTEND)

    def _assert_result(self, actual, state, expected, expected_state):
        tolerance = 1e-2 if actual.dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(
            actual.cpu(), expected, rtol=tolerance, atol=tolerance
        )
        torch.testing.assert_close(state.cpu(), expected_state, rtol=0, atol=0)

    def test_packed_prefill_preserves_request_boundaries_and_inactive_slots(self):
        # Includes a zero-length real request, repeated null slots, dirty history
        # on fresh requests, and token padding after the last real sequence.
        slots = [3, 2, 1, 4, 0, 0]
        lengths = [1, 0, 4, 2, 0, 0]
        has_prefix = [False, True, True, False, False, False]
        for width, dtype, use_bias in itertools.product(
            (2, 3, 4), (torch.float32, torch.bfloat16), (False, True)
        ):
            with self.subTest(width=width, dtype=dtype, bias=use_bias):
                x = self._randn(10, 5, dtype=dtype)
                weight = self._randn(5, width, dtype=dtype)
                bias = self._randn(5, dtype=dtype) if use_bias else None
                initial = self._randn(6, width - 1, 1, 5)
                expected, expected_state = _reference(
                    x, weight, bias, initial, slots, lengths, has_prefix
                )
                state = initial.clone().to(self.device)
                actual = self._prefill(
                    x.to(self.device),
                    weight.to(self.device),
                    None if bias is None else bias.to(self.device),
                    state,
                    slots,
                    lengths,
                    has_prefix,
                )
                self._assert_result(actual, state, expected, expected_state)

    def test_decode_reorders_requests_and_keeps_null_slot_unchanged(self):
        weight = self._randn(5, 3)
        expected_state = self._randn(5, 2, 1, 5)
        state = expected_state.clone().to(self.device)
        for slots in ([3, 0, 1, 0], [1, 3, 0, 0]):
            x = self._randn(4, 5)
            expected, expected_state = _reference(
                x, weight, None, expected_state, slots, [1] * 4, [True] * 4
            )
            metadata = TpuShortConvMetadata(
                SimpleNamespace(conv=[state]),
                torch.tensor(slots, device=self.device),
                torch.arange(5, device=self.device),
                None,
                None,
                None,
            )
            actual = short_conv(
                x.to(self.device),
                weight.to(self.device),
                None,
                metadata,
                ForwardMode.DECODE,
            )
            self._assert_result(actual, state, expected, expected_state)

    def test_chunked_prefill_then_decode_matches_full_sequence(self):
        x = self._randn(8, 5, dtype=torch.bfloat16)
        weight = self._randn(5, 3, dtype=torch.bfloat16)
        initial = self._randn(3, 2, 1, 5)
        expected, expected_state = _reference(
            x, weight, None, initial, [2], [8], [False]
        )
        state = initial.clone().to(self.device)
        pieces = []
        for start, end in ((0, 1), (1, 5), (5, 7)):
            pieces.append(
                self._prefill(
                    x[start:end].to(self.device),
                    weight.to(self.device),
                    None,
                    state,
                    [2],
                    [end - start],
                    [start > 0],
                )
            )
        metadata = TpuShortConvMetadata(
            SimpleNamespace(conv=[state]),
            torch.tensor([2], device=self.device),
            torch.tensor([0, 1], device=self.device),
            None,
            None,
            None,
        )
        pieces.append(
            short_conv(
                x[-1:].to(self.device),
                weight.to(self.device),
                None,
                metadata,
                ForwardMode.DECODE,
            )
        )
        self._assert_result(torch.cat(pieces), state, expected, expected_state)

    def test_all_padding_preserves_dirty_cache(self):
        initial = self._randn(3, 2, 1, 5)
        state = initial.clone().to(self.device)
        actual = self._prefill(
            self._randn(4, 5).to(self.device),
            self._randn(5, 3).to(self.device),
            self._randn(5).to(self.device),
            state,
            [0, 0, 0],
            [0, 0, 0],
            [False, False, False],
        )
        self._assert_result(actual, state, torch.zeros(4, 5), initial)


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX interpreter")
class TestShortConvPallasInterpreter(CustomTestCase):
    def test_ragged_dma_halos_and_state_alias_against_conv1d(self):
        """A short prefix, partial last tile and padding must not share history."""
        import jax.numpy as jnp
        from jax.experimental.pallas import tpu as pltpu

        from sglang_tpu.kernels.mamba.short_conv_pallas import short_conv_pallas

        generator = torch.Generator().manual_seed(37)
        lengths, slots = [1, 0, 65, 137, 0], [3, 2, 1, 4, 0]
        prefix = [False, True, True, False, False]
        for channels, width, dtype in (
            (5, 4, torch.float32),
            (1536, 3, torch.bfloat16),
        ):
            with self.subTest(channels=channels, width=width, dtype=dtype):

                def randn(*shape):
                    return torch.randn(*shape, generator=generator)

                x = randn(256, channels).to(dtype)
                weight = randn(channels, width).to(dtype)
                bias = randn(channels).to(dtype)
                state = randn(7, width - 1, 1, channels)
                expected, expected_state = _reference(
                    x, weight, bias, state, slots, lengths, prefix
                )
                jdtype = jnp.bfloat16 if dtype == torch.bfloat16 else jnp.float32
                actual_state, actual = short_conv_pallas(
                    jnp.asarray(x.float().numpy(), dtype=jdtype),
                    jnp.asarray(weight.float().numpy(), dtype=jdtype),
                    jnp.asarray(bias.float().numpy(), dtype=jdtype),
                    jnp.asarray(state.numpy()),
                    jnp.asarray(slots),
                    jnp.asarray([0, *itertools.accumulate(lengths)]),
                    jnp.asarray(prefix),
                    decode=False,
                    interpret=pltpu.InterpretParams(detect_races=True),
                )
                torch.testing.assert_close(
                    torch.from_numpy(np.array(actual, dtype=np.float32)).to(dtype),
                    expected,
                    rtol=1e-2 if dtype == torch.bfloat16 else 1e-5,
                    atol=1e-2 if dtype == torch.bfloat16 else 1e-5,
                )
                torch.testing.assert_close(
                    torch.from_numpy(np.array(actual_state)),
                    expected_state,
                    rtol=0,
                    atol=0,
                )
                # Decode a partial vector tile with reordered slots, consuming
                # exactly the state just written by packed prefill.
                decode_slots = [4, 0, 3, 1, 2, 0, 5, 6, 0, 0]
                x = randn(10, channels).to(dtype)
                expected, expected_state = _reference(
                    x, weight, None, expected_state, decode_slots, [1] * 10, [True] * 10
                )
                actual_state, actual = short_conv_pallas(
                    jnp.asarray(x.float().numpy(), dtype=jdtype),
                    jnp.asarray(weight.float().numpy(), dtype=jdtype),
                    None,
                    actual_state,
                    jnp.asarray(decode_slots),
                    None,
                    None,
                    decode=True,
                    interpret=pltpu.InterpretParams(detect_races=True),
                )
                torch.testing.assert_close(
                    torch.from_numpy(np.array(actual, dtype=np.float32)).to(dtype),
                    expected,
                    rtol=1e-2 if dtype == torch.bfloat16 else 1e-5,
                    atol=1e-2 if dtype == torch.bfloat16 else 1e-5,
                )
                torch.testing.assert_close(
                    torch.from_numpy(np.array(actual_state)),
                    expected_state,
                    rtol=0,
                    atol=0,
                )


@unittest.skipUnless(importlib.util.find_spec("torch_tpu"), "requires torch_tpu")
class TestShortConvTPU(TestShortConvCPU):
    device = "tpu"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch_tpu  # noqa: F401

    def test_compiled_prefill_replays_lengths_slots_and_prefix_flags(self):
        """Tile halos and donated state stay correct across one graph bucket."""
        from sglang_tpu.kernels.mamba.short_conv import build_short_conv_op

        op = torch.compile(
            build_short_conv_op(), backend="tpu", fullgraph=True, dynamic=False
        )
        channels = 1536
        weight = self._randn(channels, 3, dtype=torch.bfloat16)
        bias = self._randn(channels, dtype=torch.bfloat16)
        expected_state = self._randn(7, 2, 1, channels)
        state = expected_state.clone().to(self.device)
        weight_t, bias_t = weight.to(self.device), bias.to(self.device)
        for lengths, slots, prefix in (
            ([1, 0, 65, 129, 0], [3, 2, 1, 4, 0], [False, True, True, False, False]),
            ([63, 2, 1, 129, 0], [4, 3, 2, 1, 0], [True, True, False, True, False]),
            ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [False] * 5),
        ):
            x = self._randn(256, channels, dtype=torch.bfloat16)
            expected, expected_state = _reference(
                x, weight, bias, expected_state, slots, lengths, prefix
            )
            actual = op(
                x.to(self.device),
                weight_t,
                bias_t,
                state,
                torch.tensor(slots, device=self.device),
                torch.tensor([0, *itertools.accumulate(lengths)], device=self.device),
                torch.tensor(prefix, device=self.device),
                False,
            )
            self._assert_result(actual, state, expected, expected_state)

    def test_compiled_decode_partial_vector_tile_keeps_state_across_replays(self):
        from sglang_tpu.kernels.mamba.short_conv import build_short_conv_op

        op = torch.compile(
            build_short_conv_op(), backend="tpu", fullgraph=True, dynamic=False
        )
        channels = 2048
        weight = self._randn(channels, 4)
        expected_state = self._randn(12, 3, 1, channels)
        state = expected_state.clone().to(self.device)
        weight_t = weight.to(self.device)
        for slots in (
            [3, 0, 1, 4, 5, 6, 7, 8, 2, 0],
            [2, 8, 0, 7, 6, 5, 4, 1, 0, 3],
        ):
            x = self._randn(10, channels)
            expected, expected_state = _reference(
                x, weight, None, expected_state, slots, [1] * 10, [True] * 10
            )
            actual = op(
                x.to(self.device),
                weight_t,
                None,
                state,
                torch.tensor(slots, device=self.device),
                None,
                None,
                True,
            )
            self._assert_result(actual, state, expected, expected_state)


if __name__ == "__main__":
    unittest.main()
