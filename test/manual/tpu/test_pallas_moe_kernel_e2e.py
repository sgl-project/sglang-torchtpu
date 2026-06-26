"""End-to-end tests for the Pallas fused-MoE kernel.

Parallel to ``test_pallas_kernel_e2e.py`` (the ragged-paged-attention v3
kernel test). These tests require a TPU device and torch_tpu installed.
They:

* Build small per-expert MoE weights and a randomized routing topology.
* Run ``fused_moe_call`` both eagerly and through
  ``torch.compile(backend="tpu")``.
* Compare against a CPU-fp32 reference that does the same permute /
  GMM1 / gated-activation / GMM2 / un-permute + weighted reduce pipeline.
* Specifically exercise the M-axis pad-to-32 path used to keep
  ``gmm_v2.zero_out_start`` from crashing on warmup shapes whose row
  count (``num_tokens * topk``) is not a multiple of the sublane tiling.

Run as:
    SGLANG_PLATFORM=tpu python test/manual/tpu/test_pallas_moe_kernel_e2e.py
"""

from __future__ import annotations

import unittest

import torch

try:
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import compile as _torch_tpu_compile  # noqa: F401

    _HAS_TPU = True
except Exception:  # pragma: no cover
    _HAS_TPU = False


def _has_tpu_device() -> bool:
    if not _HAS_TPU:
        return False
    try:
        torch.empty(1, device="tpu")
        return True
    except Exception:
        return False


_SKIP_REASON = "Requires torch_tpu + a TPU device"


def _ref_moe_torch(
    hidden_states: torch.Tensor,  # [m, k] bf16
    w1: torch.Tensor,  # [g, k, 2*n] bf16  -- gmm_v2 (g, k, n) layout, gate||up
    w2: torch.Tensor,  # [g, n, k]   bf16
    topk_weights: torch.Tensor,  # [m, topk]   bf16
    topk_ids: torch.Tensor,  # [m, topk]   int32
    activation: str,
) -> torch.Tensor:
    """Plain MoE reference, one (token, expert) at a time.

    Computes everything in float32 on CPU to avoid TPU-specific numerics
    leaking into the ground truth. Output is cast back to
    ``hidden_states.dtype``.

    Layout invariant: ``w1`` is in gmm_v2's LHS-RHS ``(g, k, n)`` form
    with ``[:, :, :n]`` = gate and ``[:, :, n:]`` = up — same convention
    ``fused_moe_op._activation`` splits on.
    """
    m, k = hidden_states.shape
    g, _k, two_n = w1.shape
    assert _k == k
    assert two_n % 2 == 0
    n = two_n // 2
    topk = int(topk_ids.shape[-1])

    out = torch.zeros(m, k, dtype=torch.float32)
    for i in range(m):
        x = hidden_states[i].float()
        for j in range(topk):
            e = int(topk_ids[i, j].item())
            g1 = x @ w1[e].float()  # [2n]
            gate, up = g1[:n], g1[n:]
            if activation == "silu":
                act = torch.nn.functional.silu(gate) * up
            elif activation == "gelu":
                act = torch.nn.functional.gelu(gate) * up
            else:
                raise AssertionError(
                    f"unsupported activation in reference: {activation}"
                )
            o = act @ w2[e].float()  # [k]
            out[i] += float(topk_weights[i, j].item()) * o
    return out.to(hidden_states.dtype)


def _make_inputs(
    m: int,
    k: int,
    n: int,
    g: int,
    topk: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
):
    """Build a randomized (hidden, w1, w2, topk_weights, topk_ids) tuple on ``device``.

    Weights are small-magnitude so activations don't saturate the
    swigluoai clamp limits (this test only exercises silu, but keeping
    the magnitudes modest makes bf16 round-trip noise tractable too).
    """
    torch.manual_seed(seed)
    hidden = (torch.randn(m, k, dtype=dtype) * 0.5).to(device)
    w1 = (torch.randn(g, k, 2 * n, dtype=dtype) * 0.05).to(device)
    w2 = (torch.randn(g, n, k, dtype=dtype) * 0.05).to(device)
    topk_ids = torch.randint(0, g, (m, topk), dtype=torch.int32, device=device)
    topk_weights = torch.softmax(
        torch.randn(m, topk, device=device).float(), dim=-1
    ).to(dtype)
    return hidden, w1, w2, topk_weights, topk_ids


def _maybe_init_tpu_runtime() -> None:
    """Force-initialize the torch_tpu PjRt client so ``.to('tpu')`` works.

    ``_has_tpu_device``'s ``torch.empty(1, device='tpu')`` is enough to
    enter the test body, but copying a real CPU tensor onto the device
    later fails with ``PjRt client not initialized in TpuMallocAndMemcpyHtoD``
    unless ``apply_tpu_patches`` has run. ``apply_tpu_patches`` is the
    canonical entry point that the server uses at startup; call it here so
    the test file is self-contained.
    """
    try:
        from sglang.srt.hardware_backend.tpu.patches import apply_tpu_patches

        apply_tpu_patches()
    except Exception:  # pragma: no cover
        pass


@unittest.skipUnless(_has_tpu_device(), _SKIP_REASON)
class TestPallasFusedMoEEager(unittest.TestCase):
    """Run the Pallas fused-MoE op in eager mode and validate output."""

    @classmethod
    def setUpClass(cls):
        _maybe_init_tpu_runtime()
        from sglang.srt.layers.moe.pallas.fused_moe_op import (
            _build_fused_moe_op,
        )

        # Eager registration of the singleton custom op (mirrors how
        # PallasAttnBackend.__init__ pre-warms the RPA op).
        cls.op = _build_fused_moe_op()
        cls.device = "tpu"
        cls.dtype = torch.bfloat16

    def _run(self, hidden, w1, w2, topk_weights, topk_ids, activation):
        from sglang.srt.layers.moe.pallas.fused_moe_op import fused_moe_call

        return fused_moe_call(
            hidden_states=hidden,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            topk=int(topk_ids.shape[-1]),
            activation=activation,
        )

    def test_silu_aligned_shape_matches_reference(self):
        """m*topk already a multiple of 32 — no padding code path hit."""
        # m=32, topk=2 -> m*topk = 64.
        m, k, n, g, topk = 32, 64, 64, 4, 2
        hidden, w1, w2, w_tk, ids = _make_inputs(
            m, k, n, g, topk, self.device, self.dtype, seed=0
        )

        out = self._run(hidden, w1, w2, w_tk, ids, "silu").cpu().float()
        torch.tpu.synchronize()

        ref = _ref_moe_torch(
            hidden.cpu(), w1.cpu(), w2.cpu(), w_tk.cpu(), ids.cpu(), "silu"
        ).float()

        self.assertEqual(out.shape, ref.shape)
        max_diff = (out - ref).abs().max().item()
        # bf16 matmul accumulation tolerance.
        self.assertLess(
            max_diff, 5e-2, f"Pallas MoE vs reference max diff = {max_diff}"
        )
        self.assertTrue(torch.isfinite(out).all().item())

    def test_silu_unaligned_shape_exercises_m_padding(self):
        """m*topk = 24 — gmm_v2 sublane reshape would crash without pad-to-32.

        Regression for the bug where ``gmm_v2.zero_out_start`` does
        ``zero_ref.reshape(-1, size_lhs_sublane, num_lanes)`` and
        24 % 16 != 0 raises a Pallas ``ValueError`` at JAX trace time.
        ``_fused_moe_jax`` pads to 32 internally; this test fails
        loudly if that padding is ever removed.
        """
        # m=3, topk=8 -> m*topk = 24.
        m, k, n, g, topk = 3, 64, 64, 4, 8
        hidden, w1, w2, w_tk, ids = _make_inputs(
            m, k, n, g, topk, self.device, self.dtype, seed=1
        )

        out = self._run(hidden, w1, w2, w_tk, ids, "silu").cpu().float()
        torch.tpu.synchronize()

        ref = _ref_moe_torch(
            hidden.cpu(), w1.cpu(), w2.cpu(), w_tk.cpu(), ids.cpu(), "silu"
        ).float()

        self.assertEqual(out.shape, ref.shape)
        self.assertTrue(torch.isfinite(out).all().item())
        max_diff = (out - ref).abs().max().item()
        self.assertLess(
            max_diff, 5e-2, f"unaligned-m vs reference max diff = {max_diff}"
        )

    def test_gelu_activation_matches_reference(self):
        """Gated GeLU is the other supported in-tree activation (for mxfp4 path parity)."""
        m, k, n, g, topk = 16, 64, 64, 4, 2
        hidden, w1, w2, w_tk, ids = _make_inputs(
            m, k, n, g, topk, self.device, self.dtype, seed=2
        )

        out = self._run(hidden, w1, w2, w_tk, ids, "gelu").cpu().float()
        torch.tpu.synchronize()

        ref = _ref_moe_torch(
            hidden.cpu(), w1.cpu(), w2.cpu(), w_tk.cpu(), ids.cpu(), "gelu"
        ).float()

        max_diff = (out - ref).abs().max().item()
        self.assertLess(max_diff, 5e-2, f"gelu vs reference max diff = {max_diff}")

    def test_silu_through_process_weights_after_loading(self):
        """Pin the full ``process_weights_after_loading`` -> ``fused_moe_call`` contract.

        The other tests in this file hand kernel-layout ``(g, k, n)``
        weights straight to ``fused_moe_call`` and so bypass the
        ``UnquantizedFusedMoEMethod.process_weights_after_loading`` TPU
        branch entirely. That leaves two silent regression classes
        uncovered:

          * A broken transpose in ``process_weights_after_loading``
            (wrong axes, missing ``.contiguous()``, accidentally a no-op).
            Nothing downstream asserts the kernel-expected layout, so a
            wrong swap just produces garbage.
          * A drift between the layout ``process_weights_after_loading``
            emits and the layout ``gmm_v2`` actually consumes -- i.e. the
            integration contract between the loader hook and the kernel.

        Build w1/w2 in SGLang's storage layout (the transpose of gmm_v2's
        form -- what the model weight loader produces), run them through
        ``process_weights_after_loading``, then through ``fused_moe_call``,
        and compare to the same CPU reference the other tests use. A single
        numeric mismatch catches either drift.
        """
        from sglang.srt.layers.quantization import unquant
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedFusedMoEMethod,
        )

        m, k, n, g, topk = 32, 64, 64, 4, 2

        torch.manual_seed(7)
        hidden = (torch.randn(m, k, dtype=self.dtype) * 0.5).to(self.device)
        # Ground-truth weights in gmm_v2 (g, k, n) layout.
        w1_gmm = torch.randn(g, k, 2 * n, dtype=self.dtype) * 0.05
        w2_gmm = torch.randn(g, n, k, dtype=self.dtype) * 0.05
        # SGLang storage layout = gmm_v2 layout transposed on (1, 2).
        # This is what the weight loader hands to the MoE method.
        w13_sglang = w1_gmm.transpose(1, 2).contiguous()  # (g, 2n, k)
        w2_sglang = w2_gmm.transpose(1, 2).contiguous()  # (g, k, n)
        topk_ids = torch.randint(0, g, (m, topk), dtype=torch.int32, device=self.device)
        topk_weights = torch.softmax(
            torch.randn(m, topk, device=self.device).float(), dim=-1
        ).to(self.dtype)

        # _is_tpu is True under the TPU runtime that gates this whole
        # class. Assert the precondition so a host where it drifts to
        # False fails loudly instead of silently exercising a non-TPU
        # branch.
        self.assertTrue(
            unquant._is_tpu,
            "unquant._is_tpu must be True under the TPU runtime",
        )

        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(w13_sglang, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_sglang, requires_grad=False)
        UnquantizedFusedMoEMethod().process_weights_after_loading(layer)

        # Cheap CPU pin: the hook must recover gmm_v2 layout exactly.
        torch.testing.assert_close(layer.w13_weight.data, w1_gmm)
        torch.testing.assert_close(layer.w2_weight.data, w2_gmm)
        self.assertFalse(layer.w13_weight.requires_grad)
        self.assertFalse(layer.w2_weight.requires_grad)

        # Integration pin: post-process weights through the kernel must
        # match the CPU reference within bf16 matmul tolerance.
        out = (
            self._run(
                hidden,
                layer.w13_weight.data.to(self.device),
                layer.w2_weight.data.to(self.device),
                topk_weights,
                topk_ids,
                "silu",
            )
            .cpu()
            .float()
        )
        torch.tpu.synchronize()

        ref = _ref_moe_torch(
            hidden.cpu(),
            w1_gmm,
            w2_gmm,
            topk_weights.cpu(),
            topk_ids.cpu(),
            "silu",
        ).float()

        self.assertEqual(out.shape, ref.shape)
        self.assertTrue(torch.isfinite(out).all().item())
        max_diff = (out - ref).abs().max().item()
        self.assertLess(
            max_diff,
            5e-2,
            "process_weights_after_loading -> fused_moe_call vs reference "
            f"max diff = {max_diff}",
        )


@unittest.skipUnless(_has_tpu_device(), _SKIP_REASON)
class TestPallasFusedMoETorchCompile(unittest.TestCase):
    """Run ``fused_moe_call`` through ``torch.compile(backend='tpu')``.

    Verifies that dynamo can trace the singleton custom op as an opaque
    tensor-in/tensor-out call, and that the compiled output matches
    both the eager call and the CPU reference.
    """

    @classmethod
    def setUpClass(cls):
        _maybe_init_tpu_runtime()
        from sglang.srt.layers.moe.pallas.fused_moe_op import (
            _build_fused_moe_op,
        )

        cls.op = _build_fused_moe_op()
        cls.device = "tpu"
        cls.dtype = torch.bfloat16

    def _compiled_callable(self, activation: str):
        from sglang.srt.layers.moe.pallas.fused_moe_op import fused_moe_call

        def _run(hidden, w1, w2, topk_weights, topk_ids):
            return fused_moe_call(
                hidden_states=hidden,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                topk=int(topk_ids.shape[-1]),
                activation=activation,
            )

        return torch.compile(_run, backend="tpu", fullgraph=True, dynamic=False)

    def test_compile_silu_runs_and_matches_eager(self):
        """Compiled path is bit-stable vs eager and matches the CPU reference."""
        from sglang.srt.layers.moe.pallas.fused_moe_op import fused_moe_call

        m, k, n, g, topk = 32, 64, 64, 4, 2
        hidden, w1, w2, w_tk, ids = _make_inputs(
            m, k, n, g, topk, self.device, self.dtype, seed=3
        )

        out_eager = fused_moe_call(
            hidden_states=hidden,
            w1=w1,
            w2=w2,
            topk_weights=w_tk,
            topk_ids=ids,
            topk=topk,
            activation="silu",
        )
        torch.tpu.synchronize()

        compiled = self._compiled_callable("silu")
        out_compiled = compiled(hidden, w1, w2, w_tk, ids)
        torch.tpu.synchronize()

        e = out_eager.cpu().float()
        c = out_compiled.cpu().float()
        # Eager vs compiled can drift by XLA fusion choices, but the
        # mathematical result should be identical to within bf16 noise.
        self.assertLess(
            (e - c).abs().max().item(),
            1e-3,
            "eager vs compiled differ by more than bf16 tolerance",
        )

        ref = _ref_moe_torch(
            hidden.cpu(), w1.cpu(), w2.cpu(), w_tk.cpu(), ids.cpu(), "silu"
        ).float()
        self.assertLess(
            (c - ref).abs().max().item(),
            5e-2,
            "compiled vs reference exceeds bf16 matmul tolerance",
        )


if __name__ == "__main__":
    unittest.main()
