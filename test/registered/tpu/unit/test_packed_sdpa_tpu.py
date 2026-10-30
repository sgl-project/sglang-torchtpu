# SPDX-License-Identifier: Apache-2.0
"""Real single-device TPU checks for the shared packed SDPA implementation."""

import importlib.util
import os
import unittest

import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.packed_sdpa_test_utils import (
    load_sdpa_module,
    packed_inputs,
    packed_reference,
)

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit")


class TestPackedSDPATpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch_tpu") is None:
            if os.environ.get("CI"):
                raise RuntimeError("TPU CI requires torch_tpu")
            raise unittest.SkipTest("torch_tpu is not installed")
        from sglang.srt.hardware_backend.tpu.env_override import apply_tpu_env_overrides

        apply_tpu_env_overrides()
        import torch_tpu  # noqa: F401

        cls.sdpa = load_sdpa_module()
        # Do not silently skip initialization, compilation, or numerical failures.
        torch.ones(1, device="tpu").cpu()

    def _check(self, *, compiled):
        bounds = (0, 0, 1, 6, 6, 17, 17)
        for heads, kv_heads, dim, value_dim in (
            (1, 1, 32, 32),
            (4, 2, 128, 128),
            (4, 1, 64, 16),
        ):
            for dtype in (torch.float32, torch.bfloat16):
                for causal in (False, True):
                    with (
                        self.subTest(
                            heads=heads,
                            kv_heads=kv_heads,
                            dim=dim,
                            value_dim=value_dim,
                            dtype=dtype,
                            causal=causal,
                        ),
                        torch.no_grad(),
                    ):
                        host = packed_inputs(
                            bounds, heads, kv_heads, dim, value_dim, dtype
                        )
                        expected = packed_reference(
                            *host, bounds, scale=0.37, causal=causal
                        ).to(dtype)
                        device_inputs = tuple(tensor.to("tpu") for tensor in host)
                        before = tuple(tensor.clone() for tensor in host)
                        cu = torch.tensor(bounds, dtype=torch.int32, device="tpu")
                        impl = self.sdpa.SDPAImpl(
                            heads, dim, causal, 0.37, num_kv_heads=kv_heads
                        )

                        def forward(q, k, v):
                            return impl.forward_varlen(
                                q,
                                k,
                                v,
                                cu_seqlens=cu,
                                cu_seqlens_host=bounds,
                                max_seqlen=11,
                            )

                        if compiled:
                            # Independent shape/dtype cases must not exhaust
                            # Dynamo's per-code-object recompilation limit.
                            torch._dynamo.reset()
                            run = torch.compile(forward, backend="tpu", fullgraph=True)
                        else:
                            run = forward
                        if dtype == torch.bfloat16:
                            tolerance = 2e-2
                        elif dim >= 128 or value_dim < dim:
                            # TPU MXU SDPA uses BF16 math for these FP32 shape
                            # classes in both eager and compiled execution. The
                            # 128-wide eager case measured 3.1e-2 max error;
                            # the narrower-value padded specialization is
                            # similarly bounded. CPU reference checks stay
                            # strict in test/registered/unit/utils.
                            tolerance = 4e-2
                        else:
                            # Lower-width TPU FP32 shapes measured below 8e-3
                            # max error against the FP64 oracle.
                            tolerance = 1e-2
                        for _ in range(2):
                            actual = run(*device_inputs)
                            self.assertEqual(actual.device.type, "tpu")
                            torch.testing.assert_close(
                                actual.cpu(), expected, rtol=tolerance, atol=tolerance
                            )
                        for tensor, saved in zip(device_inputs, before):
                            torch.testing.assert_close(
                                tensor.cpu(), saved, rtol=0, atol=0
                            )

    def test_eager_packed_sdpa(self):
        self._check(compiled=False)

    def test_compiled_packed_sdpa(self):
        self._check(compiled=True)

    def test_empty_packed_sdpa(self):
        q = torch.empty(0, 4, 32, device="tpu")
        k = torch.empty(0, 1, 32, device="tpu")
        v = torch.empty(0, 1, 16, device="tpu")
        impl = self.sdpa.SDPAImpl(4, 32, False, 0.37, num_kv_heads=1)
        actual = impl.forward_varlen(
            q,
            k,
            v,
            cu_seqlens=torch.tensor([0, 0, 0], dtype=torch.int32, device="tpu"),
            cu_seqlens_host=(0, 0, 0),
            max_seqlen=0,
        )
        self.assertEqual(actual.device.type, "tpu")
        self.assertEqual(actual.cpu().shape, (0, 4, 16))


if __name__ == "__main__":
    unittest.main()
