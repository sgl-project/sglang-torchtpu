import contextlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.models.gpt_oss import GptOssModel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeLayer(torch.nn.Module):
    def __init__(self, increment, keep_residual):
        super().__init__()
        self.increment = increment
        self.keep_residual = keep_residual

    def forward(self, positions, hidden, batch, residual, captured_last_layer_outputs):
        current = hidden if residual is None else hidden + residual
        if captured_last_layer_outputs is not None:
            captured_last_layer_outputs.append(current)
        if self.keep_residual:
            return torch.full_like(current, self.increment), current
        return current + self.increment, None


class FakeNorm(torch.nn.Module):
    def forward(self, hidden, residual=None):
        if residual is None:
            return hidden * 2
        return (hidden + residual) * 2, hidden + residual


def test_final_boundary_capture_is_before_norm_with_both_residual_forms():
    for keep_residual in (False, True):
        model = object.__new__(GptOssModel)
        torch.nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.start_layer, model.end_layer = 0, 3
        model.layers = torch.nn.ModuleList(
            [FakeLayer(i + 1, keep_residual) for i in range(3)]
        )
        model.norm = FakeNorm()
        model.set_eagle3_layers_to_capture([1, 3])
        recorder = SimpleNamespace(
            with_current_layer=lambda _: contextlib.nullcontext()
        )
        with patch(
            "sglang.srt.models.gpt_oss.get_global_expert_distribution_recorder",
            return_value=recorder,
        ):
            output, auxiliary = model.forward(
                torch.tensor([1, 2]),
                torch.tensor([0, 1]),
                SimpleNamespace(),
                input_embeds=torch.zeros(2, 4),
            )
        torch.testing.assert_close(output, torch.full((2, 4), 12.0))
        assert len(auxiliary) == 2
        torch.testing.assert_close(auxiliary[0], torch.ones(2, 4))
        torch.testing.assert_close(auxiliary[1], torch.full((2, 4), 6.0))


def test_interior_only_capture_keeps_existing_behavior():
    model = object.__new__(GptOssModel)
    torch.nn.Module.__init__(model)
    model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
    model.set_eagle3_layers_to_capture([1, 2])
    assert not model.capture_final_eagle_hidden_state
    assert model.layers[1]._is_layer_to_capture and model.layers[2]._is_layer_to_capture


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
