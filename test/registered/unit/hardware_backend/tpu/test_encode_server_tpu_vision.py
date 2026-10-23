import pkgutil
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessResult
from sglang.srt.managers.schedule_batch import Modality
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu import hooks
from sglang_tpu.compilation.encoder_compile_policy import (
    install_vision_compile_policies,
)
from sglang_tpu.hooks import encoder as encoder_hooks
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)


def _encoder():
    return SimpleNamespace(
        preprocessor=SimpleNamespace(
            get_num_patches=lambda grid, modality: int(grid[0])
        ),
        _build_model_mm_items=lambda feature, result, group, modality: list(group),
    )


def _result(patch_counts: list[int]) -> EncoderPreprocessResult:
    return EncoderPreprocessResult(
        mm_inputs={},
        grid_thw=torch.tensor([[count, 1, 1] for count in patch_counts]),
        token_counts=list(patch_counts),
    )


class TestEncoderPatchBudget(unittest.TestCase):
    def test_split_keeps_order_and_isolates_oversized_items(self):
        groups = encoder_hooks.split_indices_by_patch_budget(
            patch_counts=[100, 200, 400, 50, 60],
            indices=[3, 1, 0, 2, 4],
            budget=300,
        )
        self.assertEqual(groups, [[3, 1], [0], [2, 4]])

    def test_encode_runs_one_call_per_group_under_no_grad(self):
        patch_counts = [100, 200, 300]
        calls = []

        def get_feature_fn(group):
            calls.append(
                (group, torch.is_grad_enabled(), torch.is_inference_mode_enabled())
            )
            return torch.full((len(group), 1, 2), float(group[0]))

        with patch.object(
            encoder_hooks, "precompiled_max_patch_count", return_value=300
        ):
            output = encoder_hooks.encode_indices_within_patch_budget(
                None,
                _encoder(),
                None,
                _result(patch_counts),
                [0, 1, 2],
                Modality.IMAGE,
                get_feature_fn,
            )

        self.assertEqual(calls, [([0, 1], False, False), ([2], False, False)])
        torch.testing.assert_close(
            output, torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]])
        )

    def test_audio_bypasses_the_split(self):
        sentinel = torch.zeros(1, 1)

        def original(*args):
            return sentinel

        output = encoder_hooks.encode_indices_within_patch_budget(
            original, None, None, None, [0], Modality.AUDIO, None
        )
        self.assertIs(output, sentinel)


class TestEncoderCompileInstall(unittest.TestCase):
    def test_raises_when_no_vision_policy_installs(self):
        with self.assertRaisesRegex(RuntimeError, "no TPU vision compile policy"):
            install_vision_compile_policies(nn.Linear(2, 2))

    def test_encoder_hook_targets_resolve(self):
        for target, _, _ in hooks.HOOKS:
            if ".encoder." not in target:
                continue
            obj_path, attr = target.rsplit(".", 1)
            self.assertTrue(hasattr(pkgutil.resolve_name(obj_path), attr), target)


class TestTpuSetDevice(unittest.TestCase):
    def test_gpu_id_does_not_select_the_device(self):
        """Encoder rank 3 asks for device 3 but addresses only its own chiplet."""
        with patch.object(torch.accelerator, "set_device_index") as set_index:
            TpuSRTPlatform.set_device(None, torch.device("tpu", 3))
        set_index.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
