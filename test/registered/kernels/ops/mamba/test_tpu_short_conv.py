# SPDX-License-Identifier: Apache-2.0
"""ShortConv outputs and persistent history against a grouped-conv1d reference.

CPU cases check the math; the TPU registration also exercises device lowering
and in-place state writes. No model weights or server are needed.
"""

import importlib.util
import itertools
import unittest
from types import SimpleNamespace

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


@unittest.skipUnless(importlib.util.find_spec("torch_tpu"), "requires torch_tpu")
class TestShortConvTPU(TestShortConvCPU):
    device = "tpu"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch_tpu  # noqa: F401


if __name__ == "__main__":
    unittest.main()
