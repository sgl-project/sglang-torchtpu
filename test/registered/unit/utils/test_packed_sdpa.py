# SPDX-License-Identifier: Apache-2.0
"""Model-independent packed SDPA semantics and host-metadata contracts."""

import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import (
    register_cpu_ci,
    register_cuda_ci,
    register_mlx_ci,
)
from sglang.test.packed_sdpa_test_utils import (
    load_sdpa_module,
    packed_inputs,
    packed_reference,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cuda_ci(est_time=20, stage="base-a", runner_config="1-gpu-small")
register_mlx_ci(est_time=30, suite="stage-a-unit-test-mlx")

sdpa = load_sdpa_module()


class TestPackedSDPA(unittest.TestCase):
    def _check(
        self,
        bounds,
        *,
        heads=2,
        kv_heads=2,
        dim=32,
        value_dim=32,
        dtype=torch.float32,
        causal=False,
        strided=False,
        host=True,
    ):
        q, k, v = packed_inputs(
            bounds, heads, kv_heads, dim, value_dim, dtype, strided=strided
        )
        before = [tensor.clone() for tensor in (q, k, v)]
        scale = 0.37  # Deliberately not 1/sqrt(head_dim).
        impl = sdpa.SDPAImpl(heads, dim, causal, scale, num_kv_heads=kv_heads)
        cu = torch.tensor(bounds, dtype=torch.int32)
        actual = impl.forward_varlen(
            q,
            k,
            v,
            cu_seqlens=cu,
            cu_seqlens_host=bounds if host else None,
            max_seqlen=max(b - a for a, b in zip(bounds[:-1], bounds[1:])),
        )
        expected = packed_reference(q, k, v, bounds, scale=scale, causal=causal).to(
            dtype
        )
        self.assertEqual(actual.shape, (bounds[-1], heads, value_dim))
        tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        for tensor, saved in zip((q, k, v), before):
            torch.testing.assert_close(tensor, saved, rtol=0, atol=0)

    def test_packed_matrix(self):
        for dtype in (torch.float32, torch.bfloat16):
            for heads, kv_heads, dim in (
                (1, 1, 32),
                (4, 4, 64),
                (4, 2, 128),
                (4, 1, 32),
            ):
                for causal in (False, True):
                    for host in (False, True):
                        with self.subTest(
                            dtype=dtype,
                            heads=heads,
                            kv_heads=kv_heads,
                            dim=dim,
                            causal=causal,
                            host=host,
                        ):
                            self._check(
                                (0, 0, 1, 6, 6, 15, 15),
                                heads=heads,
                                kv_heads=kv_heads,
                                dim=dim,
                                value_dim=dim,
                                dtype=dtype,
                                causal=causal,
                                host=host,
                                strided=True,
                            )

    def test_value_width_can_differ_from_query_width(self):
        for value_dim in (1, 16, 64):
            for causal in (False, True):
                with self.subTest(value_dim=value_dim, causal=causal):
                    self._check(
                        (0, 3, 3, 8),
                        heads=4,
                        kv_heads=2,
                        value_dim=value_dim,
                        causal=causal,
                    )

    def test_all_empty_segments(self):
        self._check((0, 0, 0), heads=4, kv_heads=1, value_dim=16)

    def test_segments_do_not_leak(self):
        bounds = (0, 3, 8)
        q, k, v = packed_inputs(bounds, 2, 2, 32, 32, torch.float32)
        impl = sdpa.SDPAImpl(2, 32, False, 0.37)
        metadata = dict(
            cu_seqlens=torch.tensor(bounds, dtype=torch.int32),
            cu_seqlens_host=bounds,
            max_seqlen=5,
        )
        before = impl.forward_varlen(q, k, v, **metadata)
        k[3:] = 1000
        v[3:] = -1000
        after = impl.forward_varlen(q, k, v, **metadata)
        torch.testing.assert_close(after[:3], before[:3], rtol=0, atol=0)
        self.assertFalse(torch.equal(after[3:], before[3:]))

    def test_inherited_varlen_uses_forward_override(self):
        class AlternateBackend(sdpa.SDPAImpl):
            def forward(self, q, k, v, metadata):
                self.lengths.append(q.shape[1])
                return v + 7

        bounds = (0, 0, 3, 3, 8, 8)
        q, k, v = packed_inputs(bounds, 2, 2, 32, 16, torch.float32)
        impl = AlternateBackend(2, 32, False, 0.37)
        impl.lengths = []
        actual = impl.forward_varlen(
            q,
            k,
            v,
            cu_seqlens=torch.tensor(bounds, dtype=torch.int32),
            cu_seqlens_host=bounds,
            max_seqlen=5,
        )
        self.assertEqual(impl.lengths, [3, 5])
        torch.testing.assert_close(actual, v + 7, rtol=0, atol=0)

    def test_host_metadata_needs_no_tensor_read(self):
        bounds = (0, 3, 3, 8)
        q, k, v = packed_inputs(bounds, 2, 2, 32, 32, torch.float32)
        impl = sdpa.SDPAImpl(2, 32, False, 0.37)
        # A meta tensor cannot be copied to the host or converted to a list.
        cu = torch.empty(len(bounds), device="meta", dtype=torch.int32)
        with (
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("device read")
            ),
            patch.object(
                torch.Tensor, "item", side_effect=AssertionError("device read")
            ),
        ):
            actual = impl.forward_varlen(
                q, k, v, cu_seqlens=cu, cu_seqlens_host=bounds, max_seqlen=5
            )
        expected = packed_reference(q, k, v, bounds, scale=0.37).float()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_fullgraph_capture_with_host_metadata(self):
        bounds = (0, 1, 1, 8)
        q, k, v = packed_inputs(bounds, 4, 2, 32, 32, torch.float32)
        impl = sdpa.SDPAImpl(4, 32, True, 0.37, num_kv_heads=2)
        cu = torch.empty(len(bounds), device="meta", dtype=torch.int32)

        def forward(q, k, v):
            return impl.forward_varlen(
                q, k, v, cu_seqlens=cu, cu_seqlens_host=bounds, max_seqlen=7
            )

        # Tests Dynamo capture only, not device lowering.
        actual = torch.compile(forward, backend="eager", fullgraph=True)(q, k, v)
        expected = packed_reference(q, k, v, bounds, scale=0.37, causal=True).float()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


class _MpsBranchTensor(torch.Tensor):
    @property
    def device(self):
        return torch.device("mps")


class TestPackedSDPAMpsChunking(unittest.TestCase):
    def test_chunked_causal_and_noncausal_semantics(self):
        # Exercise the real MPS chunking control flow, using CPU attention for
        # each chunk. This is not MPS hardware/lowering evidence.
        class CpuChunks(sdpa.SDPAImpl):
            def forward(self, q, k, v, metadata):
                return super().forward(
                    *(x.as_subclass(torch.Tensor) for x in (q, k, v)), metadata
                )

        for length in (127, 128, 129, 257):
            for causal in (False, True):
                with self.subTest(length=length, causal=causal):
                    bounds = (0, 3, 3, 3 + length)
                    q, k, v = packed_inputs(bounds, 2, 1, 32, 32, torch.float32)
                    impl = CpuChunks(2, 32, causal, 0.37, num_kv_heads=1)
                    with (
                        patch.object(torch.mps, "synchronize"),
                        patch.object(torch.mps, "empty_cache"),
                    ):
                        actual = impl.forward_varlen(
                            q.as_subclass(_MpsBranchTensor),
                            k,
                            v,
                            cu_seqlens=torch.tensor(bounds, dtype=torch.int32),
                            cu_seqlens_host=bounds,
                            max_seqlen=length,
                        )
                    expected = packed_reference(
                        q, k, v, bounds, scale=0.37, causal=causal
                    ).float()
                    torch.testing.assert_close(
                        actual.as_subclass(torch.Tensor), expected, rtol=2e-5, atol=2e-5
                    )


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS hardware is required")
class TestPackedSDPAMpsHardware(unittest.TestCase):
    def test_causal_and_noncausal_chunks(self):
        for length in (129, 257):
            for causal in (False, True):
                with self.subTest(length=length, causal=causal):
                    bounds = (0, 3, 3, 3 + length)
                    inputs = packed_inputs(bounds, 2, 2, 32, 32, torch.float32)
                    impl = sdpa.SDPAImpl(2, 32, causal, 0.37)
                    actual = impl.forward_varlen(
                        *(tensor.to("mps") for tensor in inputs),
                        cu_seqlens=torch.tensor(
                            bounds, dtype=torch.int32, device="mps"
                        ),
                        cu_seqlens_host=bounds,
                        max_seqlen=length,
                    )
                    expected = packed_reference(
                        *inputs, bounds, scale=0.37, causal=causal
                    ).float()
                    torch.testing.assert_close(
                        actual.cpu(), expected, rtol=2e-5, atol=2e-5
                    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware is required")
class TestPackedSDPACudaHardware(unittest.TestCase):
    def test_packed_shapes_and_values(self):
        bounds = (0, 0, 1, 6, 6, 15, 15)
        for dtype in (torch.float32, torch.bfloat16):
            for value_dim in (1, 16, 32):
                for causal in (False, True):
                    with self.subTest(dtype=dtype, value_dim=value_dim, causal=causal):
                        inputs = packed_inputs(bounds, 4, 2, 32, value_dim, dtype)
                        impl = sdpa.SDPAImpl(4, 32, causal, 0.37, num_kv_heads=2)
                        actual = impl.forward_varlen(
                            *(tensor.to("cuda") for tensor in inputs),
                            cu_seqlens=torch.tensor(
                                bounds, dtype=torch.int32, device="cuda"
                            ),
                            cu_seqlens_host=bounds,
                            max_seqlen=9,
                        )
                        expected = packed_reference(
                            *inputs, bounds, scale=0.37, causal=causal
                        ).to(dtype)
                        tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
                        torch.testing.assert_close(
                            actual.cpu(), expected, rtol=tolerance, atol=tolerance
                        )


if __name__ == "__main__":
    unittest.main()
