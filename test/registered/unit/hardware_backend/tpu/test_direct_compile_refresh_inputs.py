"""direct_compile: re-resolve a compiled artifact's model inputs after rebinding.

A caller may rebind model parameters after compiling (load the real weights
while the compile runs). The executable reads the tensors captured at compile
time, so refresh_model_inputs must look each input up again by the name or
attribute it came from, keep fixed inputs, and reject a changed shape or dtype.
Runs on the host.
"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation.direct_compile import CompiledFunction, _CompiledArtifact

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)


def _artifact(inputs, sources):
    return _CompiledArtifact(
        executable=None,
        model_inputs=tuple(inputs),
        kv_pool=None,
        tensor_inputs_key_paths=(),
        tensor_inputs_traced_shapes=(),
        output_treespec=None,
        output_count=0,
        model_input_sources=tuple(sources),
    )


def _compiled(model, cache):
    fn = CompiledFunction.__new__(CompiledFunction)
    fn._model = model
    fn.cache = cache
    return fn


class TestRefreshModelInputs(CustomTestCase):
    def setUp(self):
        self.model = nn.Module()
        self.model.lin = nn.Linear(4, 4, bias=False)
        self.model.register_buffer("scale", torch.ones(4))
        self.helper = SimpleNamespace(table=torch.zeros(3))
        self.rank = torch.tensor([0], dtype=torch.int32)
        self.artifact = _artifact(
            [self.model.lin.weight, self.model.scale, self.helper.table, self.rank],
            [
                ("module", "lin.weight"),
                ("module", "scale"),
                ("attr", self.helper, "table"),
                ("keep",),
            ],
        )
        self.fn = _compiled(self.model, {"k": self.artifact, "pending": None})

    def test_rebound_inputs_are_picked_up(self):
        self.model.lin.weight = nn.Parameter(torch.full((4, 4), 2.0))
        self.model.scale = torch.full((4,), 3.0)
        self.helper.table = torch.ones(3)
        self.assertEqual(self.fn.refresh_model_inputs(), 3)
        weight, scale, table, rank = self.artifact.model_inputs
        self.assertIs(weight, self.model.lin.weight)
        self.assertIs(scale, self.model.scale)
        self.assertIs(table, self.helper.table)
        self.assertIs(rank, self.rank)

    def test_unchanged_inputs_count_zero(self):
        self.assertEqual(self.fn.refresh_model_inputs(), 0)

    def test_shape_change_is_rejected(self):
        self.model.lin.weight = nn.Parameter(torch.zeros(4, 8))
        with self.assertRaises(ValueError):
            self.fn.refresh_model_inputs()

    def test_dtype_change_is_rejected(self):
        self.model.scale = torch.ones(4, dtype=torch.float16)
        with self.assertRaises(ValueError):
            self.fn.refresh_model_inputs()

    def test_release_drops_references_until_refresh(self):
        self.fn.release_model_inputs()
        weight, scale, table, rank = self.artifact.model_inputs
        self.assertEqual((weight, scale, table), (None, None, None))
        self.assertIs(rank, self.rank)
        # Rebuild the parameters (the old ones can now be freed), then refresh.
        self.model.lin.weight = nn.Parameter(torch.full((4, 4), 5.0))
        self.model.scale = torch.full((4,), 6.0)
        self.helper.table = torch.ones(3)
        self.assertEqual(self.fn.refresh_model_inputs(), 3)
        self.assertIs(self.artifact.model_inputs[0], self.model.lin.weight)
        self.assertIsNone(self.artifact.released_input_specs)

    def test_unbound_spmd_lists_are_released_and_refilled_in_place(self):
        # An SPMD consumer before bind(): its bind closure reads these lists.
        full = [self.model.lin.weight, self.model.scale, self.helper.table, self.rank]
        kv = torch.zeros(2)
        flat = [*full, kv]
        self.artifact.unbound_inputs = (full, flat)
        self.fn.release_model_inputs()
        self.assertEqual(full[:3], [None, None, None])
        self.assertEqual(flat[:3], [None, None, None])
        self.assertIs(full[3], self.rank)
        self.assertIs(flat[4], kv)
        self.model.lin.weight = nn.Parameter(torch.full((4, 4), 7.0))
        self.model.scale = torch.full((4,), 8.0)
        self.helper.table = torch.ones(3)
        self.fn.refresh_model_inputs()
        self.assertIs(full[0], self.model.lin.weight)
        self.assertIs(flat[1], self.model.scale)
        self.assertIs(flat[2], self.helper.table)
        self.assertIs(flat[4], kv)

    def test_refresh_after_release_checks_the_traced_spec(self):
        self.fn.release_model_inputs()
        self.model.lin.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.float16))
        with self.assertRaises(ValueError):
            self.fn.refresh_model_inputs()


if __name__ == "__main__":
    unittest.main()
