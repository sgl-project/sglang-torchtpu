"""Hermetic unit tests for gpt-oss Eagle3 aux capture across pipeline stages.

Eagle3 captures three target layers that PP spreads over different stages
(`[1, 17, 32]` +1 for gpt-oss-120b-Eagle3, so layers 2/18/33 of 36), while the
draft runs on the last stage only. `_relay_aux_hidden_states` is what carries
the earlier stages' captures forward, so these pin the slot bookkeeping.

Pure Python (no GPU, no model weights): the model body is a duck-typed stand-in
holding only the fields the two methods read.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.gpt_oss import GptOssModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_LAYERS, HIDDEN, TOKENS = 36, 4, 3
CAPTURE_LAYERS = [2, 18, 33]


def _stage(start_layer: int, end_layer: int) -> SimpleNamespace:
    return SimpleNamespace(
        layers=[SimpleNamespace() for _ in range(NUM_LAYERS)],
        start_layer=start_layer,
        end_layer=end_layer,
        layers_to_capture=[],
        local_capture_slots=[],
    )


def _capture(value: float) -> torch.Tensor:
    return torch.full((TOKENS, HIDDEN), value)


class TestGptOssPpAuxRelay(CustomTestCase):
    def test_each_stage_owns_the_slots_of_the_layers_it_holds(self):
        # pp8 over 36 layers: stage 0 [0,4), stage 4 [16,21), stage 7 [31,36).
        for start, end, expected in (
            (0, 4, [0]),
            (4, 8, []),
            (16, 21, [1]),
            (31, 36, [2]),
        ):
            with self.subTest(stage=(start, end)):
                stage = _stage(start, end)
                GptOssModel._set_layers_to_capture(stage, CAPTURE_LAYERS)
                self.assertEqual(stage.local_capture_slots, expected)

    def test_last_stage_owns_the_final_norm_capture(self):
        stage = _stage(31, NUM_LAYERS)
        GptOssModel._set_layers_to_capture(stage, [2, 18, NUM_LAYERS])
        self.assertEqual(stage.local_capture_slots, [2])

    def test_relay_accumulates_every_stage_into_its_own_slot(self):
        reference = torch.zeros((TOKENS, HIDDEN))
        buffer = None
        stages = ((0, 4, 1.0), (4, 8, None), (16, 21, 2.0), (31, 36, 3.0))
        for start, end, value in stages:
            stage = _stage(start, end)
            GptOssModel._set_layers_to_capture(stage, CAPTURE_LAYERS)
            captured = [] if value is None else [_capture(value)]
            buffer = GptOssModel._relay_aux_hidden_states(
                stage, buffer, captured, reference
            )
            self.assertEqual(buffer.shape, (TOKENS, HIDDEN * len(CAPTURE_LAYERS)))

        slots = torch.split(buffer, HIDDEN, dim=-1)
        self.assertEqual([float(slot[0, 0]) for slot in slots], [1.0, 2.0, 3.0])

    def test_a_stage_that_captures_nothing_passes_the_buffer_through(self):
        reference = torch.zeros((TOKENS, HIDDEN))
        stage = _stage(4, 8)
        GptOssModel._set_layers_to_capture(stage, CAPTURE_LAYERS)
        incoming = torch.arange(
            TOKENS * HIDDEN * len(CAPTURE_LAYERS), dtype=torch.float32
        ).reshape(TOKENS, -1)
        relayed = GptOssModel._relay_aux_hidden_states(stage, incoming, [], reference)
        torch.testing.assert_close(relayed, incoming)

    def test_unfilled_slots_stay_zero(self):
        reference = torch.zeros((TOKENS, HIDDEN))
        stage = _stage(0, 4)
        GptOssModel._set_layers_to_capture(stage, CAPTURE_LAYERS)
        buffer = GptOssModel._relay_aux_hidden_states(
            stage, None, [_capture(1.0)], reference
        )
        slots = torch.split(buffer, HIDDEN, dim=-1)
        self.assertEqual([float(slot.abs().max()) for slot in slots], [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
