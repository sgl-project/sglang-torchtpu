"""Tests for the TPU Pallas GDN linear-attention custom-op bridge.

Parallel to :mod:`test.manual.tpu.test_pallas_moe_op_registration` and
:mod:`test.manual.tpu.test_gdn_attention_v3` (the raw-kernel correctness
test). This file pins the torch<->JAX bridge in
:mod:`sglang_tpu.attention_gdn_pallas`:

  * a torch.library custom op ``sglang_tpu_pallas::gdn_attention`` is
    registered and reachable via ``torch.ops.sglang_tpu_pallas``;
  * ``build_gdn_pallas_op`` is idempotent (the ``_PALLAS_GDN_OP`` singleton
    short-circuits — torch.library rejects double registration);
  * the op's output matches the underlying JAX kernel, and it mutates
    ``conv_state`` / ``recurrent_state`` in place (the state write-back that
    the hybrid backend relies on).

Requires torch_tpu. Note the split gating below: importing sglang pins
``JAX_PLATFORMS=cpu`` process-wide for the Pallas export path, so
``jax.devices()`` never reports a TPU in this process. Only the leg that needs
a *TPU-backed JAX reference* may gate on that; gating the whole class on it
(as this file used to) made every case skip on a real TPU host, which is how
the op's missing CPU-export guard reached production.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=25, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import numpy as np
import torch

from sglang.test.test_utils import CustomTestCase

try:
    import jax  # noqa: F401
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:  # pragma: no cover
    _HAS_JAX = False


def _has_tpu_via_jax() -> bool:
    if not _HAS_JAX:
        return False
    try:
        return any(getattr(d, "platform", "") == "tpu" for d in jax.devices())
    except Exception:
        return False


_SKIP_REASON = "Requires torch_tpu"
_SKIP_REASON_JAX_TPU = (
    "Requires a TPU backend visible to jax.devices(); sglang pins "
    "JAX_PLATFORMS=cpu at import, so this leg only runs when JAX is bound to "
    "the TPU before sglang is imported."
)


def _has_torch_tpu() -> bool:
    try:
        return bool(torch.accelerator.is_available())
    except Exception:  # pragma: no cover
        return False


def _poison_freed_device_memory(dev, num_elems, copies=64):
    """Fill device buffers with NaN, materialize and free them, so an output
    buffer the kernel never writes reads NaN instead of fresh zeros."""
    bufs = [
        torch.full((num_elems,), float("nan"), dtype=torch.bfloat16, device=dev)
        for _ in range(copies)
    ]
    for x in bufs:
        x[:1].float().cpu()
    del bufs


@unittest.skipUnless(_has_torch_tpu(), _SKIP_REASON)
class TestGDNAttentionOp(CustomTestCase):
    N_KQ = 2
    N_V = 8
    D_K = 128
    D_V = 128
    K = 4

    def test_speculative_checkpoints_and_resume(self):
        """Verify checkpoints and rejection rollback match sequential decoding.

        This runs through torch_tpu with JAX pinned to CPU, just like serving.
        Both conv and recurrent state must resume from the selected checkpoint.
        """
        from sglang_tpu.attention_gdn_pallas import (
            build_gdn_pallas_op,
        )

        op = build_gdn_pallas_op()
        torch.manual_seed(17)
        n_kq, n_v, d_k, d_v, kernel_size = 1, 2, 128, 128, 4
        width = 3
        base = width
        dim = 2 * n_kq * d_k + n_v * d_v
        device = "tpu"

        def tensor(shape):
            return (torch.randn(shape) * 0.1).to(device)

        weights = [tensor((dim, 1, kernel_size)), None, tensor((n_v,)), tensor((n_v,))]
        qkv, b, a = tensor((width, dim)), tensor((width, n_v)), tensor((width, n_v))
        conv = tensor((2 * width, kernel_size - 1, dim))
        rec = tensor((2 * width, n_v, d_k, d_v))
        # Keep reference snapshots on the host: donated TPU buffers must not
        # share lazy clone storage across independent executions.
        conv_initial, rec_initial = conv.cpu(), rec.cpu()
        # Plain runs address the state row directly; verify runs pass the base
        # of the checkpoint rows [width, 2 * width) and a per-sequence offset.
        indices = torch.tensor([base], dtype=torch.int32, device=device)
        window = torch.tensor([width], dtype=torch.int32, device=device)
        distribution = torch.ones(3, dtype=torch.int32, device=device)

        def table(offset):
            return torch.tensor([offset], dtype=torch.int32, device=device)

        def run(q, bb, aa, c, r, prefix, offset=None, num_spec_tokens=0):
            count = q.shape[0]
            spec = num_spec_tokens > 0
            return op(
                q,
                bb,
                aa,
                c,
                r,
                *weights,
                torch.tensor([0, count], dtype=torch.int32, device=device),
                window if spec else indices,
                distribution,
                torch.tensor([prefix + count], dtype=torch.int32, device=device),
                n_kq,
                n_v,
                d_k,
                d_v,
                kernel_size,
                read_offsets=offset,
                num_spec_tokens=num_spec_tokens,
            )

        output = run(qkv, b, a, conv, rec, 16, table(0), width - 1)
        compiled_verify = torch.compile(
            lambda q, bb, aa, c, r, offsets: run(
                q, bb, aa, c, r, 16, offsets, width - 1
            ),
            backend="tpu",
            fullgraph=True,
        )
        output_cpu = output.cpu()
        compiled_conv = conv_initial.to(device, copy=True)
        compiled_rec = rec_initial.to(device, copy=True)
        compiled_output = compiled_verify(
            qkv, b, a, compiled_conv, compiled_rec, table(0)
        )
        torch.testing.assert_close(
            compiled_output.cpu(), output_cpu, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(
            compiled_conv.cpu(), conv.cpu(), rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(compiled_rec.cpu(), rec.cpu(), rtol=2e-2, atol=2e-2)
        ref_conv = conv_initial.to(device, copy=True)
        ref_rec = rec_initial.to(device, copy=True)
        outputs, checkpoints = [], []
        for step in range(width):
            outputs.append(
                run(
                    qkv[step : step + 1],
                    b[step : step + 1],
                    a[step : step + 1],
                    ref_conv,
                    ref_rec,
                    16 + step,
                )
            )
            torch.testing.assert_close(
                conv[base + step].cpu(), ref_conv[base].cpu(), rtol=2e-2, atol=2e-2
            )
            torch.testing.assert_close(
                rec[base + step].cpu(), ref_rec[base].cpu(), rtol=2e-2, atol=2e-2
            )
            checkpoints.append((ref_conv.cpu(), ref_rec.cpu()))
        torch.testing.assert_close(
            output.cpu(), torch.cat(outputs).cpu(), rtol=2e-2, atol=2e-2
        )
        # The null group's state must remain untouched.
        torch.testing.assert_close(conv[:base].cpu(), conv_initial[:base].cpu())
        torch.testing.assert_close(rec[:base].cpu(), rec_initial[:base].cpu())

        next_q, next_b, next_a = tensor((1, dim)), tensor((1, n_v)), tensor((1, n_v))
        for offset in range(width):
            with self.subTest(resume_checkpoint=offset):
                resume_conv = conv.cpu().to(device, copy=True)
                resume_rec = rec.cpu().to(device, copy=True)
                expected_conv = checkpoints[offset][0].to(device, copy=True)
                expected_rec = checkpoints[offset][1].to(device, copy=True)
                actual = run(
                    next_q,
                    next_b,
                    next_a,
                    resume_conv,
                    resume_rec,
                    17 + offset,
                    table(offset),
                    width - 1,
                )
                expected = run(
                    next_q, next_b, next_a, expected_conv, expected_rec, 17 + offset
                )
                torch.testing.assert_close(
                    actual.cpu(), expected.cpu(), rtol=2e-2, atol=2e-2
                )
                torch.testing.assert_close(
                    resume_conv[base].cpu(),
                    expected_conv[base].cpu(),
                    rtol=2e-2,
                    atol=2e-2,
                )
                torch.testing.assert_close(
                    resume_rec[base].cpu(),
                    expected_rec[base].cpu(),
                    rtol=2e-2,
                    atol=2e-2,
                )

    def test_op_registered_and_idempotent(self):
        from sglang_tpu.attention_gdn_pallas import (
            build_gdn_pallas_op,
        )

        op1 = build_gdn_pallas_op()
        op2 = build_gdn_pallas_op()
        self.assertIs(op1, op2, "builder must be idempotent (singleton)")
        self.assertTrue(hasattr(torch.ops, "sglang_tpu_pallas"))
        self.assertTrue(hasattr(torch.ops.sglang_tpu_pallas, "gdn_attention"))

    def test_op_lowers_under_cpu_export_pin(self):
        """The op must lower on a host where JAX is pinned to CPU.

        This is the configuration the server actually runs in: sglang sets
        ``JAX_PLATFORMS=cpu`` so Pallas kernels export against CPU
        placeholders, and torch_tpu executes the registered kernel on the TPU.
        The GDN kernel picks its block sizes from ``pltpu.get_tpu_info()``, so
        a lowering that is not wrapped in ``tpu_info_for_cpu_export`` resolves
        the *CPU* device kind and dies with "Unsupported TPU device kind: cpu"
        -- on the first request, after startup has already reported success.

        ``custom_jax_kernel`` lowers lazily, so registering the op is not
        enough to trigger it; the case has to call the op. It deliberately
        asserts only shape and state mutation, with no JAX reference, so that
        it does not need the TPU-backed ``jax.devices()`` the numerics case
        below requires -- that gate is what kept this path untested.
        """
        from sglang_tpu.attention_gdn_pallas import (
            build_gdn_pallas_op,
        )

        n_kq, n_v, d_k, d_v, K = self.N_KQ, self.N_V, self.D_K, self.D_V, self.K
        lengths = [1, 1, 64]  # mixed batch: 2 decode + 1 prefill
        max_reqs = 3
        num_tokens = sum(lengths)
        num_blocks = max_reqs + 1
        conv_dim = n_kq * d_k * 2 + n_v * d_v

        rng = np.random.default_rng(0)

        def rn(*shape):
            return rng.standard_normal(shape).astype(np.float32)

        dev = torch.device("tpu")

        def tt(x):
            return torch.from_numpy(np.ascontiguousarray(x)).to(dev)

        conv_t = tt(np.zeros((num_blocks, K - 1, conv_dim), np.float32))
        rec_t = tt(np.zeros((num_blocks, n_v, d_k, d_v), np.float32))

        out = build_gdn_pallas_op()(
            tt(rn(num_tokens, conv_dim)),
            tt(rn(num_tokens, n_v)),
            tt(rn(num_tokens, n_v)),
            conv_t,
            rec_t,
            tt(rn(conv_dim, 1, K)),
            tt(rn(conv_dim)),
            tt(rn(n_v)),
            tt(rn(n_v)),
            tt(np.array([0, 1, 2, 66], np.int32)),
            tt(np.arange(1, max_reqs + 1, dtype=np.int32)),
            tt(np.array([2, 3, 3], np.int32)),
            tt(np.array(lengths, np.int32)),
            n_kq,
            n_v,
            d_k,
            d_v,
            K,
        )

        out_np = out.cpu().numpy()
        self.assertEqual(out_np.shape, (num_tokens, n_v, d_v))
        self.assertTrue(np.isfinite(out_np).all(), "output has NaN/Inf")
        # conv_state / recurrent_state are donated and aliased back, so a
        # lowering that silently dropped the aliasing would leave them zero.
        self.assertTrue((conv_t.cpu().numpy() != 0).any(), "conv_state not written")
        self.assertTrue((rec_t.cpu().numpy() != 0).any(), "recurrent_state not written")

    def test_op_lowers_each_batch_cap_under_cpu_export_pin(self):
        """MIXED batches share a token bucket across batch caps: two calls with
        66 tokens, 3 and then 2 sequences, must each lower under the export
        pin. An export guard keyed by the token count alone runs the second
        lowering unguarded, which dies on device_kind="cpu"."""
        from sglang_tpu.attention_gdn_pallas import (
            build_gdn_pallas_op,
        )

        n_kq, n_v, d_k, d_v, K = self.N_KQ, self.N_V, self.D_K, self.D_V, self.K
        conv_dim = n_kq * d_k * 2 + n_v * d_v
        rng = np.random.default_rng(1)

        def rn(*shape):
            return rng.standard_normal(shape).astype(np.float32)

        dev = torch.device("tpu")

        def tt(x):
            return torch.from_numpy(np.ascontiguousarray(x)).to(dev)

        # The same 66 tokens as 2 decode + 1 prefill, then 1 decode + 1 prefill.
        for lengths in ([1, 1, 64], [1, 65]):
            with self.subTest(num_seqs=len(lengths)):
                num_seqs = len(lengths)
                num_tokens = sum(lengths)
                out = build_gdn_pallas_op()(
                    tt(rn(num_tokens, conv_dim)),
                    tt(rn(num_tokens, n_v)),
                    tt(rn(num_tokens, n_v)),
                    tt(np.zeros((num_seqs + 1, K - 1, conv_dim), np.float32)),
                    tt(np.zeros((num_seqs + 1, n_v, d_k, d_v), np.float32)),
                    tt(rn(conv_dim, 1, K)),
                    tt(rn(conv_dim)),
                    tt(rn(n_v)),
                    tt(rn(n_v)),
                    tt(np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)),
                    tt(np.arange(1, num_seqs + 1, dtype=np.int32)),
                    tt(np.array([num_seqs - 1, num_seqs, num_seqs], np.int32)),
                    tt(np.array(lengths, np.int32)),
                    n_kq,
                    n_v,
                    d_k,
                    d_v,
                    K,
                )
                out_np = out.cpu().numpy()
                self.assertEqual(out_np.shape, (num_tokens, n_v, d_v))
                self.assertTrue(np.isfinite(out_np).all(), "output has NaN/Inf")

    @unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON_JAX_TPU)
    def test_op_matches_kernel_and_mutates_state(self):
        from sglang_tpu.attention.pallas.gdn import fused_conv1d_gdn
        from sglang_tpu.attention_gdn_pallas import (
            build_gdn_pallas_op,
        )

        n_kq, n_v, d_k, d_v, K = self.N_KQ, self.N_V, self.D_K, self.D_V, self.K
        # Mixed batch: 2 decode + 1 prefill.
        lengths = [1, 1, 64]
        q_loc = [0, 1, 2, 66]
        distribution = [2, 3, 3]
        max_reqs = 3
        num_tokens = sum(lengths)
        num_blocks = max_reqs + 1
        conv_dim = n_kq * d_k * 2 + n_v * d_v

        rng = np.random.default_rng(0)

        def rn(*shape):
            return rng.standard_normal(shape).astype(np.float32)

        qkv = rn(num_tokens, conv_dim)
        b = rn(num_tokens, n_v)
        a = rn(num_tokens, n_v)
        conv_state = np.zeros((num_blocks, K - 1, conv_dim), np.float32)
        recurrent_state = np.zeros((num_blocks, n_v, d_k, d_v), np.float32)
        conv_weight = rn(conv_dim, 1, K)
        conv_bias = rn(conv_dim)
        a_log = rn(n_v)
        dt_bias = rn(n_v)
        query_start_loc = np.array(q_loc, np.int32)
        state_indices = np.arange(1, max_reqs + 1, dtype=np.int32)
        distribution_np = np.array(distribution, np.int32)
        seq_lens = np.array(lengths, np.int32)  # fresh slots (context_len 0)

        # Reference: run the JAX kernel directly.
        (conv_ref, rec_ref), out_ref = fused_conv1d_gdn(
            jnp.asarray(qkv),
            jnp.asarray(b),
            jnp.asarray(a),
            jnp.asarray(conv_state),
            jnp.asarray(recurrent_state),
            jnp.asarray(conv_weight),
            jnp.asarray(conv_bias),
            jnp.asarray(a_log),
            jnp.asarray(dt_bias),
            jnp.asarray(query_start_loc),
            jnp.asarray(state_indices),
            jnp.asarray(distribution_np),
            jnp.asarray(seq_lens),
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=K,
        )

        # Bridge: run the torch custom op on TPU tensors.
        op = build_gdn_pallas_op()
        dev = torch.device("tpu")

        def tt(x):
            return torch.from_numpy(np.ascontiguousarray(x)).to(dev)

        conv_t = tt(conv_state)
        rec_t = tt(recurrent_state)
        out_t = op(
            tt(qkv),
            tt(b),
            tt(a),
            conv_t,
            rec_t,
            tt(conv_weight),
            tt(conv_bias),
            tt(a_log),
            tt(dt_bias),
            torch.from_numpy(query_start_loc).to(dev),
            torch.from_numpy(state_indices).to(dev),
            torch.from_numpy(distribution_np).to(dev),
            torch.from_numpy(seq_lens).to(dev),
            n_kq,
            n_v,
            d_k,
            d_v,
            K,
        )

        np.testing.assert_allclose(
            out_t.cpu().numpy(),
            np.asarray(out_ref).reshape(out_t.shape),
            rtol=2e-2,
            atol=2e-2,
        )
        # conv_state / recurrent_state were donated + aliased => mutated.
        np.testing.assert_allclose(
            conv_t.cpu().numpy(), np.asarray(conv_ref), rtol=2e-2, atol=2e-2
        )
        np.testing.assert_allclose(
            rec_t.cpu().numpy(), np.asarray(rec_ref), rtol=2e-2, atol=2e-2
        )

    def _kda_case(self):
        # KDA runs without GQA: n_kq == n_v, per-channel gate and dt_bias.
        # Conv state in the rank-4 [slots, K-1, 1, dim] layout the TPU pool
        # allocates.
        n_v, d_k, d_v, K = 4, self.D_K, self.D_V, self.K
        lengths = [1, 1, 64]
        rng = np.random.default_rng(1)

        def rn(*shape):
            return rng.standard_normal(shape).astype(np.float32)

        num_tokens, num_blocks = sum(lengths), len(lengths) + 1
        conv_dim = 2 * n_v * d_k + n_v * d_v
        args = (
            rn(num_tokens, conv_dim),
            rn(num_tokens, n_v),
            rn(num_tokens, n_v * d_k),
            np.zeros((num_blocks, K - 1, 1, conv_dim), np.float32),
            np.zeros((num_blocks, n_v, d_k, d_v), np.float32),
            rn(conv_dim, 1, K),
            rn(conv_dim),
            rn(n_v),
            rn(n_v, d_k),
            np.array([0, 1, 2, 66], np.int32),
            np.arange(1, len(lengths) + 1, dtype=np.int32),
            np.array([2, 3, 3], np.int32),
            np.array(lengths, np.int32),
        )
        return args, (n_v, n_v, d_k, d_v, K)

    def test_kda_op_lowers_under_cpu_export_pin(self):
        """KDA mode through the op, in the server's CPU-pinned export config."""
        from sglang_tpu.attention_gdn_pallas import (
            build_gdn_pallas_op,
        )

        args, dims = self._kda_case()
        dev = torch.device("tpu")
        t = [torch.from_numpy(np.ascontiguousarray(x)).to(dev) for x in args]
        out = build_gdn_pallas_op()(*t, *dims, True, -5.0)

        n_v, d_v = dims[1], dims[3]
        out_np = out.cpu().numpy()
        self.assertEqual(out_np.shape, (args[0].shape[0], n_v, d_v))
        self.assertTrue(np.isfinite(out_np).all(), "output has NaN/Inf")
        self.assertEqual(tuple(t[3].shape), args[3].shape)
        self.assertTrue((t[3].cpu().numpy() != 0).any(), "conv_state not written")
        self.assertTrue((t[4].cpu().numpy() != 0).any(), "recurrent_state not written")

    @unittest.skipUnless(_has_tpu_via_jax(), _SKIP_REASON_JAX_TPU)
    def test_kda_op_matches_kernel(self):
        """The op forwards KDA mode and the gate bound to the kernel unchanged."""
        from sglang_tpu.attention.pallas.gdn import fused_conv1d_gdn
        from sglang_tpu.attention.pallas.gdn.config import AttentionMode
        from sglang_tpu.attention_gdn_pallas import (
            build_gdn_pallas_op,
        )

        args, (n_kq, n_v, d_k, d_v, K) = self._kda_case()
        (conv_ref, rec_ref), out_ref = fused_conv1d_gdn(
            *[jnp.asarray(x) for x in args],
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            kernel_size=K,
            attention_mode=AttentionMode.KDA,
            gate_lower_bound=-5.0,
        )

        dev = torch.device("tpu")
        t = [torch.from_numpy(np.ascontiguousarray(x)).to(dev) for x in args]
        out_t = build_gdn_pallas_op()(*t, n_kq, n_v, d_k, d_v, K, True, -5.0)

        np.testing.assert_allclose(
            out_t.cpu().numpy(),
            np.asarray(out_ref).reshape(out_t.shape),
            rtol=2e-2,
            atol=5e-3,
        )
        np.testing.assert_allclose(
            t[3].cpu().numpy(), np.asarray(conv_ref), rtol=2e-2, atol=2e-2
        )
        np.testing.assert_allclose(
            t[4].cpu().numpy(), np.asarray(rec_ref), rtol=2e-2, atol=2e-2
        )

    def _k3_decode(self, num_seqs, num_real):
        """Kimi-K3's TPU decode op call (fused [q, k, v, g] input, packed
        weights, batched_only) or the plain call it replaces, on `num_seqs`
        rows of which the first `num_real` are real; the rest are bucket
        padding pointing at the null slot. Returns run(fast) -> numpy
        (out, conv_state, recurrent_state) from fresh state copies."""
        from sglang_tpu.attention_gdn_pallas import (
            build_gdn_pallas_op,
        )
        from sglang_tpu.kda_layout import pack_kda_weights

        n_v, d_k, d_v, K = 4, self.D_K, self.D_V, self.K
        conv_dim = 2 * n_v * d_k + n_v * d_v
        rng = np.random.default_rng(2)

        def rn(*shape, scale=1.0):
            return (scale * rng.standard_normal(shape)).astype(np.float32)

        qkvg = rn(num_seqs, conv_dim + n_v * d_v)
        b, a = rn(num_seqs, n_v), rn(num_seqs, n_v * d_k)
        conv_w, dt_bias, a_log = rn(conv_dim, 1, K), rn(n_v, d_k), rn(n_v)
        packed = pack_kda_weights(
            *(torch.from_numpy(x) for x in (conv_w, dt_bias, a_log)), n_v, d_k
        ).numpy()
        conv0 = rn(num_seqs + 1, K - 1, 1, conv_dim)
        rec0 = rn(num_seqs + 1, n_v, d_k, d_v, scale=0.1)
        rows = np.arange(num_seqs, dtype=np.int32)

        dev = torch.device("tpu")

        def tt(x, dtype=None):
            t = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
            return t if dtype is None else t.to(dtype)

        # Device inputs are built once; run() only allocates the donated states
        # and the output, so poisoned freed memory reaches the output buffer.
        ba = (tt(b, torch.bfloat16), tt(a, torch.bfloat16))
        weights = (tt(conv_w), None, tt(a_log), tt(dt_bias))
        metadata = (
            tt(np.arange(num_seqs + 1, dtype=np.int32)),
            tt(np.where(rows < num_real, rows + 1, 0).astype(np.int32)),
            tt(np.array([num_real] * 3, np.int32)),
            tt(np.full(num_seqs, 6, np.int32)),
        )
        qkv_fast = tt(qkvg, torch.bfloat16)
        qkv_plain = tt(qkvg[:, :conv_dim], torch.bfloat16)
        fast_kwargs = dict(
            batched_only=True, fused_qkvg=True, packed_weights=tt(packed)
        )

        def run(fast):
            conv_t, rec_t = tt(conv0), tt(rec0)
            out = build_gdn_pallas_op()(
                qkv_fast if fast else qkv_plain,
                *ba,
                conv_t,
                rec_t,
                *weights,
                *metadata,
                n_v,
                n_v,
                d_k,
                d_v,
                K,
                True,
                -5.0,
                **(fast_kwargs if fast else {}),
            )
            return [x.float().cpu().numpy() for x in (out, conv_t, rec_t)]

        return run

    def _gdn_decode_padding(self):
        """TpuGDNAttnBackend's decode call (batched_only) on a one-row bucket
        whose row is padding. Returns run() -> numpy output."""
        from sglang_tpu.attention_gdn_pallas import (
            build_gdn_pallas_op,
        )

        n_kq, n_v, d_k, d_v, K = self.N_KQ, self.N_V, self.D_K, self.D_V, self.K
        dim = 2 * n_kq * d_k + n_v * d_v
        rng = np.random.default_rng(3)
        dev = torch.device("tpu")

        def tt(x, dtype=None):
            t = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
            return t if dtype is None else t.to(dtype)

        def rn(*shape):
            return rng.standard_normal(shape).astype(np.float32)

        inputs = (
            tt(rn(1, dim), torch.bfloat16),
            tt(rn(1, n_v), torch.bfloat16),
            tt(rn(1, n_v), torch.bfloat16),
        )
        weights = (tt(rn(dim, 1, K)), tt(rn(dim)), tt(rn(n_v)), tt(rn(n_v)))
        metadata = (
            tt(np.arange(2, dtype=np.int32)),
            tt(np.zeros(1, np.int32)),
            tt(np.zeros(3, np.int32)),
            tt(np.zeros(1, np.int32)),
        )

        def run():
            conv_t = torch.zeros(2, K - 1, 1, dim, device=dev)
            rec_t = torch.zeros(2, n_v, d_k, d_v, device=dev)
            out = build_gdn_pallas_op()(
                *inputs,
                conv_t,
                rec_t,
                *weights,
                *metadata,
                n_kq,
                n_v,
                d_k,
                d_v,
                K,
                batched_only=True,
            )
            return out.float().cpu().numpy()

        return run

    def test_kda_op_decode_fast_path_matches_default(self):
        """Kimi-K3's decode call gives the same bits as the plain op call, for
        a one-row decode and a multi-row one."""
        for num_seqs in (1, 3):
            with self.subTest(num_seqs=num_seqs):
                run = self._k3_decode(num_seqs, num_seqs)
                for got, want in zip(run(fast=True), run(fast=False)):
                    np.testing.assert_array_equal(got, want)

    def test_decode_padding_rows_are_zero(self):
        """A one-row decode whose row is padding (an idle DP-attention rank run
        as decode: distribution [0, 0, 0]) returns zeros. The kernel never
        writes that row, and the row feeds residual / MoE / collectives
        downstream, so the output must not hold stale memory. Freed device
        memory is filled with NaN first so an unwritten output shows up."""
        dev = torch.device("tpu")
        run_k3, run_gdn = self._k3_decode(1, 0), self._gdn_decode_padding()
        for name, run in (
            ("kda_k3", lambda: run_k3(fast=True)[0]),
            ("gdn", run_gdn),
        ):
            run()  # compile before poisoning
            for num_elems in (1 << 10, 1 << 11, 1 << 19):
                with self.subTest(op=name, poison=num_elems):
                    _poison_freed_device_memory(dev, num_elems)
                    out = run()
                    np.testing.assert_array_equal(out, np.zeros_like(out))

    def test_wide_qkv_requires_fused_qkvg(self):
        """Only an explicit fused_qkvg call may pass a qkv wider than the conv
        dim; any other width fails the op's width check."""
        from sglang_tpu.attention_gdn_pallas import (
            build_gdn_pallas_op,
        )

        n_v, d_k, d_v, K = 4, self.D_K, self.D_V, self.K
        conv_dim = 2 * n_v * d_k + n_v * d_v
        dev = torch.device("tpu")
        with self.assertRaises(Exception) as raised:
            build_gdn_pallas_op()(
                torch.zeros(1, conv_dim + n_v * d_v, device=dev),
                torch.zeros(1, n_v, device=dev),
                torch.zeros(1, n_v * d_k, device=dev),
                torch.zeros(2, K - 1, 1, conv_dim, device=dev),
                torch.zeros(2, n_v, d_k, d_v, device=dev),
                torch.zeros(conv_dim, 1, K, device=dev),
                None,
                torch.zeros(n_v, device=dev),
                torch.zeros(n_v, d_k, device=dev),
                torch.arange(2, dtype=torch.int32, device=dev),
                torch.ones(1, dtype=torch.int32, device=dev),
                torch.ones(3, dtype=torch.int32, device=dev),
                torch.full((1,), 6, dtype=torch.int32, device=dev),
                n_v,
                n_v,
                d_k,
                d_v,
                K,
                True,
                -5.0,
                batched_only=True,
            ).cpu()
        # Whatever type torch_tpu surfaces, the op's own check must be the cause
        # (the kernel's conv-weight assert carries no message).
        chain, exc = [], raised.exception
        while exc is not None and len(chain) < 16:
            chain.append(exc)
            exc = exc.__cause__ or exc.__context__
        self.assertTrue(any("fused_qkvg=True" in str(e) for e in chain), chain)


if __name__ == "__main__":
    unittest.main()
