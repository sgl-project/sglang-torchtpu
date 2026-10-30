import importlib.util
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

from sglang.srt.compilation.compile import (
    _infer_dynamic_arg_dims_from_annotations,
    _mark_dynamic_forward_batch,
    _runtime_dynamic_dim_for_argument,
)
from sglang.srt.compilation.support_torch_compile import NoopCompilePolicy
from sglang.test.test_utils import CustomTestCase


class TestEagerDraftWarmup(CustomTestCase):
    def test_noop_accepts_optional_fused_draft_callback(self):
        """Eager EAGLE/MTP startup must accept the worker warmup call.

        The worker passes a third argument even without compilation; the
        default policy must accept it without executing a draft forward.
        """

        def unexpected_forward(*args):
            self.fail("An eager warmup must not execute the fused draft callback")

        policy = NoopCompilePolicy()
        policy.precompile_draft(None, None)
        policy.precompile_draft(None, None, None)
        policy.precompile_draft(None, None, unexpected_forward)


class _MropeModel:
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
    ):
        return input_ids, positions, forward_batch


class _StringAnnotatedMropeModel:
    def forward(
        self,
        input_ids: "torch.Tensor",
        positions: "torch.Tensor",
        forward_batch,
    ):
        return input_ids, positions, forward_batch


def test_compiled_op_imports_and_runs_without_jax():
    module_path = importlib.util.find_spec("sglang.srt.compilation.compiled_op").origin
    with patch.dict(sys.modules, {"jax": None, "sglang_tpu": None}):
        module = runpy.run_path(module_path)

        @module["compiled_op"]()
        def helper(x):
            return x + 1

        x = torch.tensor([1, 2])
        self_contained_warmup = module["warmup_compiled_op"]
        assert self_contained_warmup(helper, x) is None
        torch.testing.assert_close(helper(x), x + 1)


def test_compiled_op_accepts_custom_ops():
    module_path = importlib.util.find_spec("sglang.srt.compilation.compiled_op").origin
    module = runpy.run_path(module_path)

    @torch.library.custom_op("compiled_op_test::scale", mutates_args=())
    def scale(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    x = torch.tensor([1, 2])
    op = module["compiled_op"](warmup=lambda _: (x,))(scale)
    assert module["_compiled_ops"]["compiled_op_test::scale"].op is op
    assert module["warmup_compiled_op"](op, x) is None
    assert torch.equal(op(x), x * 2)


def test_positions_marks_the_token_axis_dynamic_for_mrope_and_1d_rope():
    dynamic_dims = _infer_dynamic_arg_dims_from_annotations(_MropeModel.forward)
    string_dynamic_dims = _infer_dynamic_arg_dims_from_annotations(
        _StringAnnotatedMropeModel.forward
    )

    assert dynamic_dims["input_ids"] == 0
    assert dynamic_dims["positions"] == -1
    assert string_dynamic_dims["positions"] == -1


def test_runtime_dynamic_dim_uses_the_token_axis_for_mrope_metadata():
    assert _runtime_dynamic_dim_for_argument("positions") == -1
    assert _runtime_dynamic_dim_for_argument("position_ids") == -1
    assert _runtime_dynamic_dim_for_argument("mrope_positions") == -1
    assert _runtime_dynamic_dim_for_argument("input_ids") == 0


def test_forward_batch_marks_token_and_batch_metadata_dynamic():
    batch = SimpleNamespace(
        input_embeds=torch.empty(8, 16),
        seq_lens=torch.empty(2, dtype=torch.int64),
        mrope_positions=torch.empty(3, 8, dtype=torch.int64),
        scalar=torch.tensor(1),
    )
    marked = []

    def record_mark_dynamic(value, dims):
        marked.append((id(value), tuple(dims)))

    with patch("torch._dynamo.maybe_mark_dynamic", side_effect=record_mark_dynamic):
        _mark_dynamic_forward_batch(batch)

    assert (id(batch.input_embeds), (0,)) in marked
    assert (id(batch.seq_lens), (0,)) in marked
    assert (id(batch.mrope_positions), (1,)) in marked
    assert all(value_id != id(batch.scalar) for value_id, _ in marked)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
