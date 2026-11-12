# SPDX-License-Identifier: Apache-2.0
"""Native TorchTPU MoE CPU export must target physical TPU metadata.

NVFP4 block-512, hidden/intermediate, activation, top-k and SparseCore gather
match the captured DSv4.1 MoE. Two local experts bound allocation; the full
48-expert shape is separately exportable without allocation.
"""

import importlib.util
import os
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"
from sglang_tpu.env_override import apply_tpu_env_overrides

apply_tpu_env_overrides()
import torch
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=240,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)


@unittest.skipUnless(importlib.util.find_spec("torch_tpu"), "requires TorchTPU")
class TestFusedMoECpuExport(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch_tpu  # noqa: F401

        from sglang_tpu.patches import _register_tpu_compile_backend

        _register_tpu_compile_backend()
        torch.set_float32_matmul_precision("medium")

    def test_native_nvfp4_fullgraph(self):
        import jax

        from sglang_tpu.moe.pallas.fused_moe_op import _build_fused_moe_op

        self.assertEqual(jax.default_backend(), "cpu")
        op = _build_fused_moe_op()
        hidden, intermediate = 5120, 2560
        # FP4 code 2 means +1; A means -1. K-adjacent nibbles are packed.
        w1 = torch.full((2, hidden // 2, 2 * intermediate), 0x22, dtype=torch.uint8)
        w1[1, :, :intermediate] = 0xAA
        w2 = torch.full((2, intermediate // 2, hidden), 0x22, dtype=torch.uint8)
        s1 = torch.full((2, hidden // 512, 1, 2 * intermediate), 1 / 64)
        s2 = torch.full((2, intermediate // 512, 1, hidden), 1 / 1024)
        device_weights = tuple(t.to("tpu") for t in (w1, w2, s1, s2))
        for tokens in (128, 1024):
            with self.subTest(tokens=tokens):
                torch._dynamo.reset()

                def forward(x, ids, weights):
                    return op(
                        x,
                        device_weights[0],
                        device_weights[1],
                        weights,
                        ids,
                        device_weights[2],
                        device_weights[3],
                        None,
                        None,
                        6,
                        "silu_and_mul_clamp:10.0",
                        False,
                        384,
                        288,
                        1,
                    )

                compiled = torch.compile(forward, backend="tpu", fullgraph=True)
                for replay in range(2):
                    scalar = torch.tensor([0.125, -0.125, 0.25, -0.25]).roll(replay)
                    x = (
                        scalar.repeat(tokens // 4)[:, None]
                        .expand(-1, hidden)
                        .contiguous()
                        .to(torch.bfloat16)
                    )
                    ids = torch.tensor(
                        [288, 289, 0, 1, 382, -1], dtype=torch.int32
                    ).repeat(tokens, 1)
                    weights = torch.tensor(
                        [0.5, 0.25, 0.125, 0.125, 0.0, 0.0], dtype=torch.bfloat16
                    ).repeat(tokens, 1)
                    # Independent analytic dequantized reference for constant-K
                    # FP4 matrices: dot(x, +/-1/64) = +/-sum(x)/64.
                    total = x.float().sum(1) / 64
                    expected = torch.zeros(tokens)
                    for expert, coefficient in ((0, 0.5), (1, 0.25)):
                        gate = (total if expert == 0 else -total).clamp(max=10)
                        up = total.clamp(-10, 10)
                        activated = (
                            (torch.nn.functional.silu(gate) * up)
                            .to(torch.bfloat16)
                            .float()
                        )
                        down = (
                            (activated * (intermediate / 1024))
                            .to(torch.bfloat16)
                            .float()
                        )
                        expected += coefficient * down
                    expected = (
                        expected.to(torch.bfloat16).float()[:, None].expand(-1, hidden)
                    )
                    result = (
                        compiled(x.to("tpu"), ids.to("tpu"), weights.to("tpu"))
                        .cpu()
                        .float()
                    )
                    self.assertEqual(result.shape, expected.shape)
                    self.assertTrue(torch.isfinite(result).all())
                    self.assertGreater(expected.abs().mean().item(), 1.0)
                    # Existing NVFP4 block-512 admission: FP8 LHS quantization
                    # differs from the independently dequantized BF16 reference.
                    relative_mean = (
                        result - expected
                    ).abs().mean() / expected.abs().mean()
                    # FP32 cosine on 5M entries reports 0.993626 even for
                    # identical vectors. Accumulate this CPU-only metric in FP64;
                    # kernel precision and admission thresholds stay unchanged.
                    cosine = torch.nn.functional.cosine_similarity(
                        result.double().flatten(), expected.double().flatten(), dim=0
                    )
                    self.assertLess(relative_mean.item(), 0.08)
                    self.assertGreater(cosine.item(), 0.995)
        self.assertEqual(jax.default_backend(), "cpu")


if __name__ == "__main__":
    unittest.main()
