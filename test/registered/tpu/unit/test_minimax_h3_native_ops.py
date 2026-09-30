# SPDX-License-Identifier: Apache-2.0
"""Single-device lowering checks for H3's existing native tensor operations.

These do not load model weights or establish end-to-end H3 TPU support.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import torch

import sglang
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")


class TestMiniMaxH3NativeOpsTpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch_tpu") is None:
            if os.environ.get("CI"):
                raise RuntimeError("TPU CI requires torch_tpu")
            raise unittest.SkipTest("torch_tpu is not installed")

        from sglang.srt.hardware_backend.tpu.env_override import apply_tpu_env_overrides

        apply_tpu_env_overrides()
        import torch_tpu  # noqa: F401

        # Import just the operations under test, without bootstrapping the
        # diffusion config/quantization registry or pretending TPU admission is enabled.
        path = (
            Path(sglang.__file__).parent
            / "multimodal_gen/runtime/models/dits/minimax_h3_ops.py"
        )
        spec = importlib.util.spec_from_file_location("minimax_h3_ops_tpu_test", path)
        cls.ops = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.ops
        spec.loader.exec_module(cls.ops)
        # Device/initialization/lowering failures must fail, not silently skip.
        torch.ones(1, device="tpu").cpu()

    def _check(self, compiled):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype, compiled=compiled), torch.no_grad():
                x = torch.linspace(-1, 1, 8 * 32).reshape(8, 32).to(dtype)
                table = torch.linspace(-0.5, 0.5, 3 * 32).reshape(3, 32).to(dtype)
                indices = torch.tensor([2, 0, 2, 1, 0, 1, 2, 0])
                norm_cpu = torch.nn.RMSNorm(32, eps=1e-6, dtype=dtype)
                norm_tpu = torch.nn.RMSNorm(32, eps=1e-6, dtype=dtype).to("tpu")

                # CPU reference explicitly spells out the pre-existing math.
                modulated = x * (1 + table[indices]) + table[indices]
                gated = x + table[indices] * modulated
                activated = torch.nn.functional.silu(x) * gated
                expected = (
                    modulated,
                    gated,
                    activated,
                    norm_cpu(activated),
                    norm_cpu(gated),
                )

                def forward(x, table, indices):
                    modulated = self.ops._modulate_scale_shift(
                        x, table, table, indices, dtype=dtype
                    )
                    gated = self.ops._modulate_gate(
                        x, table, modulated, indices, dtype=dtype
                    )
                    activated = self.ops._silu_mul(
                        torch.cat((x, gated), dim=-1), reuse_input=True
                    )
                    q, k = self.ops._apply_qk_norm(
                        activated, gated, norm_tpu, norm_tpu, 32
                    )
                    return modulated, gated, activated, q, k

                run = (
                    torch.compile(forward, backend="tpu", fullgraph=True)
                    if compiled
                    else forward
                )
                actual = run(x.to("tpu"), table.to("tpu"), indices.to("tpu"))
                tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
                for result, reference in zip(actual, expected):
                    self.assertEqual(result.device.type, "tpu")
                    torch.testing.assert_close(
                        result.cpu(), reference, rtol=tolerance, atol=tolerance
                    )

    def test_eager_native_operations(self):
        self._check(compiled=False)

    def test_compiled_native_operations(self):
        self._check(compiled=True)


if __name__ == "__main__":
    unittest.main()
