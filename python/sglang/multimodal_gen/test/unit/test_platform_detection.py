# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime import platforms


class NVMLUnavailableError(Exception):
    pass


CUDA_QUALNAME = "sglang.multimodal_gen.runtime.platforms.cuda.CudaPlatform"
TPU_QUALNAME = "sglang.multimodal_gen.runtime.platforms.tpu.TpuPlatform"


class TestPlatformPriority(unittest.TestCase):
    def test_cuda_wins_when_torch_tpu_is_also_installed(self):
        with (
            patch.dict(os.environ, {"SGLANG_DIFFUSION_PLATFORM_OVERRIDE": ""}),
            patch.object(platforms, "mps_platform_plugin", return_value=None),
            patch.object(platforms, "xpu_platform_plugin", return_value=None),
            patch.object(platforms, "rocm_platform_plugin", return_value=None),
            patch.object(platforms, "cuda_platform_plugin", return_value=CUDA_QUALNAME),
            patch.object(platforms, "tpu_platform_plugin", return_value=TPU_QUALNAME),
        ):
            self.assertEqual(
                platforms.resolve_current_platform_cls_qualname(), CUDA_QUALNAME
            )

    def test_installed_torch_tpu_wins_over_cpu(self):
        with (
            patch.dict(os.environ, {"SGLANG_DIFFUSION_PLATFORM_OVERRIDE": ""}),
            patch.object(platforms, "mps_platform_plugin", return_value=None),
            patch.object(platforms, "xpu_platform_plugin", return_value=None),
            patch.object(platforms, "rocm_platform_plugin", return_value=None),
            patch.object(platforms, "cuda_platform_plugin", return_value=None),
            patch.object(platforms, "npu_platform_plugin", return_value=None),
            patch.object(platforms, "musa_platform_plugin", return_value=None),
            patch.object(platforms, "tpu_platform_plugin", return_value=TPU_QUALNAME),
        ):
            self.assertEqual(
                platforms.resolve_current_platform_cls_qualname(), TPU_QUALNAME
            )

    def test_cpu_when_torch_tpu_is_absent(self):
        with (
            patch.dict(os.environ, {"SGLANG_DIFFUSION_PLATFORM_OVERRIDE": ""}),
            patch.object(platforms, "mps_platform_plugin", return_value=None),
            patch.object(platforms, "xpu_platform_plugin", return_value=None),
            patch.object(platforms, "rocm_platform_plugin", return_value=None),
            patch.object(platforms, "cuda_platform_plugin", return_value=None),
            patch.object(platforms, "npu_platform_plugin", return_value=None),
            patch.object(platforms, "musa_platform_plugin", return_value=None),
            patch.object(platforms, "tpu_platform_plugin", return_value=None),
        ):
            self.assertIn(
                "CpuPlatform", platforms.resolve_current_platform_cls_qualname()
            )


class TestCudaPlatformDetection(unittest.TestCase):
    def test_torch_fallback_excludes_hip(self):
        cases = (
            ("6.0", None),
            (
                None,
                "sglang.multimodal_gen.runtime.platforms.cuda.CudaPlatform",
            ),
        )

        for hip_version, expected in cases:
            with (
                self.subTest(hip_version=hip_version),
                patch(
                    "sglang.multimodal_gen.utils.import_pynvml",
                    side_effect=NVMLUnavailableError,
                ),
                patch.object(platforms.os.path, "isfile", return_value=False),
                patch.object(platforms.os.path, "exists", return_value=False),
                patch.object(torch.version, "hip", hip_version, create=True),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "device_count", return_value=1),
            ):
                self.assertEqual(platforms.cuda_platform_plugin(), expected)


if __name__ == "__main__":
    unittest.main()
