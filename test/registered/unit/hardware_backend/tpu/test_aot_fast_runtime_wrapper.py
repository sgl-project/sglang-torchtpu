"""Lean AOTAutograd runtime wrapper (sglang_tpu/compilation/aot_fast_runtime_wrapper.py):
which graphs it takes, and that it copies mutated inputs back like torch's wrapper."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation import aot_fast_runtime_wrapper as aot


def _meta(*, data=True, metadata=False, storage=False):
    return SimpleNamespace(
        mutates_data=data, mutates_metadata=metadata, mutates_storage_metadata=storage
    )


def _runtime_metadata(input_info, mutated, **overrides):
    md = SimpleNamespace(
        num_outputs_aliased=0,
        num_intermediate_bases=0,
        dynamic_outputs=False,
        grad_enabled_mutation=None,
        tokens=(),
        mutated_inp_stream_indices=None,
        mutated_inp_runtime_indices=mutated,
        num_mutated_inp_runtime_indices=len(mutated),
        input_info=input_info,
        num_outputs=1,
    )
    for k, v in overrides.items():
        setattr(md, k, v)
    return md


class TestLeanRuntimeWrapper(CustomTestCase):
    def test_plain_data_mutations_are_eligible(self):
        md = _runtime_metadata([_meta(), _meta(data=False), _meta()], [0, 2])
        self.assertEqual(aot._eligible(md, False, False), (2, ((0, 0), (1, 2))))

    def test_everything_else_keeps_torch_wrapper(self):
        info = [_meta()]
        for md, joint, keep in (
            (_runtime_metadata(info, [0]), True, False),
            (_runtime_metadata(info, [0]), False, True),
            (_runtime_metadata(info, [0], num_outputs_aliased=1), False, False),
            (_runtime_metadata(info, [0], tokens=(object(),)), False, False),
            (_runtime_metadata([_meta(metadata=True)], [0]), False, False),
            (_runtime_metadata([_meta(storage=True)], [0]), False, False),
        ):
            with self.subTest(md=md, joint=joint, keep=keep):
                self.assertIsNone(aot._eligible(md, joint, keep))

    def test_wrapper_copies_back_and_returns_user_outputs(self):
        state = torch.zeros(3)
        calls = []

        def compiled(args):
            calls.append("compiled")
            return [args[0] + 1, args[1] * 2]

        compiled._boxed_call = True  # takes the args list, like AOTAutograd's

        def fallback(args):
            calls.append("fallback")
            return ["torch"]

        wrapper = aot._make_fast_wrapper(
            compiled, fallback, 1, ((0, 0),), 2, False, use_foreach=True
        )
        self.assertEqual(wrapper([state, torch.ones(2)]), ["torch"])  # first call
        out = wrapper([state, torch.ones(2)])
        torch.testing.assert_close(state, torch.ones(3))
        torch.testing.assert_close(out[0], torch.full((2,), 2.0))
        self.assertEqual(calls, ["fallback", "compiled"])


if __name__ == "__main__":
    unittest.main()
