# SPDX-License-Identifier: Apache-2.0
"""CPU tests for SGLANG_TPU_OPT_H3_MEGAKERNEL_ADALN_PRECOMPUTE (h3_megakernel):
a request's AdaLN projections of every step in one GEMM a block, its tables
then built from those rows, kept only if they are the per-step bits."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
# The CPU gate installs python/ only; an installed sglang_tpu still wins.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from sglang_tpu.envs import envs  # noqa: E402
from sglang_tpu.multimodal_gen.h3_megakernel import hook  # noqa: E402


class _Proj(nn.Module):
    """An AdaLN projection (3 modalities x 6 tables x 8 wide) whose GEMM runs
    row by row, so a batch gives each row the per-step bits, unless ``skew``
    perturbs batches of more than one step (a GEMM that tiles them apart)."""

    modality_num, expand_ratio, hidden_size = 3, 6, 8
    split_output = dit.MiniMaxH3AdalnProj.split_output

    def __init__(self, seed: int, skew: float = 0.0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(16, 144, generator=generator).bfloat16()
        self.skew, self.calls = skew, 0

    def project_local(self, x):
        self.calls += 1
        out = torch.stack([row @ self.weight for row in x])
        return out + self.skew if x.shape[0] > 2 else out

    def forward(self, x):
        return self.split_output(self.project_local(x))


def _stack(projs):
    stack = hook.H3MegakernelStack.__new__(hook.H3MegakernelStack)
    nn.Module.__init__(stack)
    stack.adaln, stack.adaln_shard, stack.head_dim = nn.ModuleList(projs), None, 8
    stack._vouching, stack.vouch_margin = None, None
    stack._adaln_batch, stack._adaln_step = None, None
    return stack


_MODEL = SimpleNamespace(
    _time_embedding=lambda t: torch.sin(t[:, None] * torch.arange(1, 17))
)
# H3's first step has one unique timestep (t = 0 for video and audio).
_PLAN = [torch.tensor([0.0])] + [
    torch.tensor([1.0 - s, 0.9 - 0.8 * s]) for s in torch.linspace(0.1, 0.9, 6).tolist()
]


def _tables(stack, step):
    adaln_input = hook._adaln_input(_MODEL, _PLAN[step])
    args = (
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, 8),
        torch.tensor([0, 4, 4]),
    )
    return stack.operands(adaln_input, *args)[3]


class TestAdalnPrecompute(CustomTestCase):
    def setUp(self):
        patcher = patch.dict(hook._ADALN_BATCH_EXACT, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_each_step_takes_its_batched_rows_bit_for_bit(self):
        stack = _stack([_Proj(seed) for seed in range(3)])
        want = [_tables(stack, step) for step in range(len(_PLAN))]
        stack.precompute_adaln(_MODEL, _PLAN)
        self.assertEqual(hook._ADALN_BATCH_EXACT, {(1, 2, 2, 2, 2, 2, 2): True})
        calls = [proj.calls for proj in stack.adaln]
        for step in reversed(range(len(_PLAN))):  # by its tensor, not its turn
            stack.adaln_step(_PLAN[step])
            self.assertTrue(torch.equal(_tables(stack, step), want[step]), step)
        self.assertEqual([proj.calls for proj in stack.adaln], calls)  # no GEMM
        # A forward of other timesteps, or one after the request: per step.
        stack.adaln_step(_PLAN[0].clone())
        self.assertTrue(torch.equal(_tables(stack, 0), want[0]))
        stack.end_request()
        stack.adaln_step(_PLAN[3])
        self.assertTrue(torch.equal(_tables(stack, 3), want[3]))
        self.assertEqual([proj.calls for proj in stack.adaln], [c + 2 for c in calls])

    def test_a_batch_with_other_bits_is_turned_off(self):
        stack = _stack([_Proj(0), _Proj(1, skew=2.0**-6)])
        want = [_tables(stack, step) for step in range(len(_PLAN))]
        with self.assertLogs(hook.logger, "WARNING") as logs:
            stack.precompute_adaln(_MODEL, _PLAN)
        self.assertIn("differ from the per-step projections", logs.output[0])
        self.assertIsNone(stack._adaln_batch)
        stack.adaln_step(_PLAN[1])
        self.assertTrue(torch.equal(_tables(stack, 1), want[1]))
        # Checked once a pattern: the next request projects nothing up front.
        calls = stack.adaln[0].calls
        stack.precompute_adaln(_MODEL, _PLAN)
        self.assertEqual(stack.adaln[0].calls, calls)
        self.assertIsNone(stack._adaln_batch)

    def test_the_hook_precomputes_when_the_loop_stages_the_plan(self):
        cls = dit.MiniMaxH3DiTModel
        names = ("post_load_weights", "forward", "prepare_adaln_plans")
        with (
            patch.multiple(cls, **{name: getattr(cls, name) for name in names}),
            envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ADALN_PRECOMPUTE.override(True),
        ):
            hook.install()
            self.addCleanup(delattr, cls, "_tpu_h3_megakernel_installed")
            stack = _stack([_Proj(0)])
            model = SimpleNamespace(
                adaln_cache=None, _time_embedding=_MODEL._time_embedding
            )
            model.__dict__[hook._STATE] = stack
            self.assertIsNone(cls.prepare_adaln_plans(model, _PLAN))
        self.assertEqual(stack._adaln_batch[1][id(_PLAN[4])], (7, 2))  # 1 + 3 x 2


if __name__ == "__main__":
    unittest.main()
