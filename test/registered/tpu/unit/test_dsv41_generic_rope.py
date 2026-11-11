# SPDX-License-Identifier: Apache-2.0
"""Generic Q/KV/index RoPE must register before fullgraph tracing.

The real model reached this shared path only after fused QA/QB preparation;
first-use registration in ratio-2/index-key publication created a custom JAX
kernel inside Dynamo. Exercise the same backend startup and first compiled
call in a fresh test process, then replay with changed positions.
"""

import importlib.util
import os
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"

import torch
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang_tpu.models.deepseek_v41.attention_backend import (
    DSV41TPUAttnBackend,
    apply_rope,
)

register_tpu_ci(
    est_time=180,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)


def _complex_reference(x, positions, table):
    cos, sin = table[positions.long()].chunk(2, -1)
    broadcast = (x.shape[0],) + (1,) * (x.ndim - 2) + (32,)
    phase = torch.complex(cos.reshape(broadcast), sin.reshape(broadcast))
    tail = torch.view_as_complex(
        x[..., -64:].float().reshape(*x.shape[:-1], 32, 2).contiguous()
    )
    rotated = torch.view_as_real(tail * phase).flatten(-2).to(x.dtype)
    return torch.cat((x[..., :-64], rotated), -1)


@unittest.skipUnless(importlib.util.find_spec("torch_tpu"), "requires TorchTPU")
class TestGenericRopeStartup(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch_tpu  # noqa: F401

        from sglang_tpu.patches import _register_tpu_compile_backend

        _register_tpu_compile_backend()
        DSV41TPUAttnBackend._register_ops()

    def test_first_fullgraph_call_and_position_replay(self):
        def forward(q, latent, index_key, positions, table):
            return tuple(
                apply_rope(x, positions, table) for x in (q, latent, index_key)
            )

        for tokens in (1, 128, 256):
            with self.subTest(tokens=tokens):
                torch.manual_seed(902 + tokens)
                values = [
                    torch.randn(tokens, 64, 512).bfloat16(),
                    torch.randn(tokens, 512).bfloat16(),
                    torch.randn(tokens, 128).bfloat16(),
                ]
                theta = torch.randn(2048, 32)
                table = torch.cat((theta.cos(), theta.sin()), -1)
                run = torch.compile(
                    forward, backend="tpu", fullgraph=True, dynamic=False
                )
                device_values = [x.to("tpu") for x in values]
                device_table = table.to("tpu")
                for replay in range(2):
                    positions = (
                        torch.arange(tokens, dtype=torch.int32) * 2 + replay * 7
                    ) % 2048
                    actual = run(*device_values, positions.to("tpu"), device_table)
                    for got, value in zip(actual, values):
                        torch.testing.assert_close(
                            got.cpu(),
                            _complex_reference(value, positions, table),
                            rtol=1e-2,
                            atol=1e-2,
                        )


if __name__ == "__main__":
    unittest.main()
