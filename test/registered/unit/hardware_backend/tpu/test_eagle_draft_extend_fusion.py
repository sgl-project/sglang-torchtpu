"""Selected-row projection must preserve full-window draft KV updates."""

import unittest
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.eagle_draft_extend import (
    install_fused_draft_extend,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ProjectionMetadata(NamedTuple):
    forward_mode: ForwardMode
    draft_extend_select_index: torch.Tensor


def _fixture(compile_graph=False):
    model = LlamaForCausalLMEagle3.__new__(LlamaForCausalLMEagle3)
    torch.nn.Module.__init__(model)
    pruner = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(pruner)
    hidden = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8) / 100
    weight = torch.arange(8 * 11, dtype=torch.float32).reshape(8, 11) / 100
    kv = torch.full_like(hidden, -1)

    def forward(input_ids, positions, batch):
        # The backbone updates all four positions even when only one is accepted.
        kv.copy_(hidden)
        metadata = _ProjectionMetadata(
            forward_mode=batch.forward_mode,
            draft_extend_select_index=batch.spec_info.select_index,
        )
        selected = pruner._get_pruned_states(hidden, None, None, metadata)[0]
        torch._assert(selected.shape[0] == 4, "project only one row per request")
        return LogitsProcessorOutput(
            next_token_logits=selected @ weight, hidden_states=hidden + 1
        )

    model.forward = forward
    original = Mock(return_value="original")
    policy = SimpleNamespace(
        owner=model,
        draft_token_num=4,
        compiled_forward_shim=SimpleNamespace(fn=original),
        _direct_compile_fused_draft_steps=(
            lambda fn: (
                torch.compile(fn, backend="aot_eager", fullgraph=True)
                if compile_graph
                else fn
            )
        ),
    )
    batch = SimpleNamespace(
        input_ids=torch.arange(16),
        positions=torch.arange(16),
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        spec_info=EagleDraftExtendInput(
            num_accept_tokens=torch.tensor([1, 2, 4, 0], dtype=torch.int32),
            num_tokens_per_req=4,
            select_index=None,
        ),
    )
    return policy, batch, hidden, weight, kv, original


class TestEagleDraftExtendFusion(CustomTestCase):
    def test_direct_trace_needs_only_hidden_states_in_spec_pytree(self):
        policy, batch, hidden, weight, _, _ = _fixture()

        def compile_with_spec_projection(fn):
            def call(forward_batch, *args, **kwargs):
                original = forward_batch.spec_info
                # direct_compile's registered spec pytree drops all other fields.
                forward_batch.spec_info = EagleDraftExtendInput(
                    hidden_states=original.hidden_states
                )
                try:
                    return fn(forward_batch, *args, **kwargs)
                finally:
                    forward_batch.spec_info = original

            return call

        policy._direct_compile_fused_draft_steps = compile_with_spec_projection
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(False):
            install_fused_draft_extend(policy)
        out = policy.compiled_forward_shim.fn(batch.input_ids, batch.positions, batch)
        selected = torch.tensor([0, 5, 11, 12])
        torch.testing.assert_close(
            out.tpu_draft_extend_outputs[0], (hidden + 1)[selected]
        )
        torch.testing.assert_close(
            out.tpu_draft_extend_outputs[2],
            (hidden @ weight)[selected].argmax(-1, keepdim=True),
        )

    def test_selected_projection_preserves_state_and_full_kv_window(self):
        for compile_graph in (False, True):
            with self.subTest(compile_graph=compile_graph):
                policy, batch, hidden, weight, kv, _ = _fixture(compile_graph)
                with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(False):
                    install_fused_draft_extend(policy)
                for counts in ([1, 2, 4, 0], [4, 1, 3, 2]):
                    batch.spec_info.num_accept_tokens.copy_(torch.tensor(counts))
                    out = policy.compiled_forward_shim.fn(
                        batch.input_ids, batch.positions, batch
                    )
                    selected = torch.tensor(
                        [
                            row * 4 + max(count, 1) - 1
                            for row, count in enumerate(counts)
                        ]
                    )
                    full_logits = hidden @ weight
                    expected = (
                        (hidden + 1)[selected],
                        torch.ones((4, 1)),
                        full_logits[selected].argmax(-1, keepdim=True),
                    )
                    for actual, reference in zip(
                        out.tpu_draft_extend_outputs, expected
                    ):
                        torch.testing.assert_close(actual, reference)
                    torch.testing.assert_close(kv, hidden)
                    self.assertIsNone(batch.spec_info.select_index)
                    self.assertIsNone(out.next_token_logits)

    def test_non_extension_and_diagnostic_paths_keep_original_forward(self):
        policy, batch, _, _, _, original = _fixture()
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True):
            install_fused_draft_extend(policy)
        self.assertIs(policy.compiled_forward_shim.fn, original)
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(False):
            install_fused_draft_extend(policy)
        batch.forward_mode = ForwardMode.EXTEND
        self.assertEqual(
            policy.compiled_forward_shim.fn(batch.input_ids, batch.positions, batch),
            "original",
        )
        original.assert_called_once()

    def test_selector_is_restored_after_forward_error(self):
        policy, batch, _, _, _, _ = _fixture()
        previous = torch.tensor([0, 4, 8, 12])
        batch.spec_info.select_index = previous
        policy.owner.forward = Mock(side_effect=RuntimeError("forward failed"))
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(False):
            install_fused_draft_extend(policy)
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            policy.compiled_forward_shim.fn(batch.input_ids, batch.positions, batch)
        self.assertIs(batch.spec_info.select_index, previous)


if __name__ == "__main__":
    unittest.main()
