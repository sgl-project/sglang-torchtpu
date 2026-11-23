"""Batch invariance of the true on-policy kernels (sglang_tpu/rl/true_on_policy/)."""

import importlib.util
import unittest

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=240, suite="stage-a-test-tpu-unit", chips=1)

import torch  # noqa: E402

from sglang.test.test_utils import CustomTestCase  # noqa: E402


def _tpu_available() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        import torch_tpu  # noqa: F401

        torch.zeros(1).to("tpu")
        return True
    except Exception:
        return False


_TPU = _tpu_available()
ROW_COUNTS = (1, 7, 100, 512)


def _rnd(*shape, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16).to("tpu")


# A small gated-delta-net layer: 4 key heads, 8 value heads, head size 128, conv kernel 4.
_GDN_NKQ, _GDN_NV, _GDN_D, _GDN_KERNEL = 4, 8, 128, 4
_GDN_DIM = 2 * _GDN_NKQ * _GDN_D + _GDN_NV * _GDN_D


def _gdn_inputs(tokens):
    shapes = ((tokens, _GDN_DIM), (tokens, _GDN_NV), (tokens, _GDN_NV))  # qkv, b, a
    return tuple(_rnd(*shape, seed=seed) for seed, shape in enumerate(shapes))


def _gdn_weights():
    return (
        _rnd(_GDN_DIM, 1, _GDN_KERNEL, scale=0.3, seed=3),  # conv weight
        _rnd(_GDN_NV, scale=0.5, seed=4),  # A_log
        _rnd(_GDN_NV, scale=0.5, seed=5),  # dt_bias
    )


def _packed_gdn(kernels, qkv, b, a, cu_seqlens):
    return kernels.packed_gated_delta_net(
        qkv, b, a, *_gdn_weights(), cu_seqlens, _GDN_NKQ, _GDN_NV, _GDN_D, _GDN_D
    )


def _gdn_decode_step_op():
    """The engine kernel on its decode path: one token of one request, state slot 1."""
    import jax
    import jax.numpy as jnp
    from torch_tpu._internal import pallas

    from sglang_tpu.attention.pallas.gdn import fused_conv1d_gdn
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    def decode_step(
        qkv: jax.Array,
        b: jax.Array,
        a: jax.Array,
        conv_state: jax.Array,
        recurrent_state: jax.Array,
        conv_weight: jax.Array,
        a_log: jax.Array,
        dt_bias: jax.Array,
        seq_len: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        with tpu_info_for_cpu_export():
            (conv_state, recurrent_state), out = fused_conv1d_gdn(
                qkv,
                b,
                a,
                conv_state,
                recurrent_state,
                conv_weight,
                None,
                a_log,
                dt_bias,
                jnp.array([0, 1], jnp.int32),
                jnp.array([1], jnp.int32),
                jnp.array([1, 1, 1], jnp.int32),  # one decode sequence
                seq_len,
                n_kq=_GDN_NKQ,
                n_v=_GDN_NV,
                d_k=_GDN_D,
                d_v=_GDN_D,
                kernel_size=_GDN_KERNEL,
                batched_only=True,
            )
        return conv_state, recurrent_state, out

    return pallas.jax_op("sglang_tpu_true_on_policy_test::gdn_decode_step", decode_step)


@unittest.skipIf(not _TPU, "needs a TPU")
class TestTrueOnPolicyKernels(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang_tpu.rl import true_on_policy

        cls.k = true_on_policy

    def _assert_grads_match(self, ours, reference, inputs):
        grads = [torch.autograd.grad(f(), inputs) for f in (ours, reference)]
        for g, ref in zip(*grads):
            cos = torch.nn.functional.cosine_similarity(
                g.float().flatten(), ref.float().flatten(), dim=0
            ).item()
            self.assertGreater(cos, 0.99)

    def _assert_rows_invariant(self, fn, args, rows_max=1024):
        full = fn(*args).cpu()
        for m in ROW_COUNTS:
            part = fn(*(a[:m] if a.shape[0] == rows_max else a for a in args)).cpu()
            self.assertTrue(torch.equal(part, full[:m]), f"rows differ at M={m}")

    def test_matmul_rows_do_not_depend_on_row_count(self):
        x, w = _rnd(1024, 1536), _rnd(640, 1536, scale=1536**-0.5, seed=1)
        self._assert_rows_invariant(self.k.matmul, (x, w))
        ref = x.float() @ w.float().T
        self.assertLess((self.k.matmul(x, w).float() - ref).abs().max().item(), 0.05)

    def test_matmul_refuses_a_partial_k_step(self):
        with self.assertRaisesRegex(Exception, "K % 512"):
            self.k.matmul(_rnd(8, 1000), _rnd(16, 1000)).cpu()

    def test_logprob_matches_log_softmax_and_is_row_invariant(self):
        vocab = 5000  # not a multiple of the column tile
        logits = _rnd(1024, vocab, scale=3.0)
        temperature = torch.full((1024,), 0.7, device="tpu")
        target = torch.randint(0, vocab, (1024,)).to(torch.int32).to("tpu")
        self._assert_rows_invariant(self.k.logprob, (logits, temperature, target))
        ref = torch.log_softmax(logits.cpu().double() / 0.7, dim=-1).gather(
            1, target.cpu().long()[:, None]
        )[:, 0]
        self.assertLess(
            (self.k.logprob(logits, temperature, target).cpu().double() - ref)
            .abs()
            .max()
            .item(),
            1e-5,
        )

    def test_logprob_is_finite_when_the_first_column_tile_is_masked(self):
        vocab = 5000
        g = torch.Generator().manual_seed(0)
        logits = (torch.randn(64, vocab, generator=g) * 3.0).to(torch.bfloat16)
        logits[:, :2048] = float("-inf")  # the whole first column tile
        target = torch.randint(2048, vocab, (64,), generator=g).to(torch.int32)
        out = self.k.logprob(
            logits.to("tpu"), torch.ones(64, device="tpu"), target.to("tpu")
        ).cpu()
        ref = torch.log_softmax(logits.double(), dim=-1).gather(
            1, target.long()[:, None]
        )[:, 0]
        self.assertLess((out.double() - ref).abs().max().item(), 1e-5)

    def test_rms_norm_rounds_where_forward_native_rounds(self):
        x, r, w = _rnd(1024, 1024), _rnd(1024, 1024, seed=1), _rnd(1024, seed=2)
        self._assert_rows_invariant(
            lambda a, b: self.k.rms_norm(a, b, w, 1e-6)[0], (x, r)
        )
        self._assert_rows_invariant(lambda a: self.k.rms_norm(a, None, w, 1e-6), (x,))
        out, new_residual = self.k.rms_norm(x, r, w, 1e-6)
        s = x.cpu().float() + r.cpu().float()
        self.assertTrue(torch.equal(new_residual.cpu(), s.to(torch.bfloat16)))
        ref = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6) * w.cpu().float()
        self.assertLess((out.cpu().float() - ref).abs().max().item(), 0.05)

    def test_gemma_rms_norm_rounds_where_forward_native_rounds(self):
        x, r, w = _rnd(1024, 1024), _rnd(1024, 1024, seed=1), _rnd(1024, seed=2)

        def norm(a, b=None):
            return self.k.rms_norm(a, b, w, 1e-6, gemma=True)

        self._assert_rows_invariant(lambda a, b: norm(a, b)[0], (x, r))
        self._assert_rows_invariant(norm, (x,))
        out, new_residual = norm(x, r)
        # GemmaRMSNorm adds in bf16 and normalizes that rounded sum, with weight 1 + w.
        s = x.cpu() + r.cpu()
        self.assertTrue(torch.equal(new_residual.cpu(), s))
        s = s.float()
        ref = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = ref * (1.0 + w.cpu().float())
        self.assertLess((out.cpu().float() - ref).abs().max().item(), 0.05)

    def test_gated_rms_norm_matches_the_torch_module_and_is_row_invariant(self):
        # [tokens x value heads, value head dim], as the gated-delta-net layer calls it
        x, z, w = _rnd(1024, 128), _rnd(1024, 128, seed=1), _rnd(128, seed=2)
        self._assert_rows_invariant(
            lambda a, b: self.k.gated_rms_norm(a, b, w, 1e-6), (x, z)
        )
        xf, zf = x.cpu().float(), z.cpu().float()
        ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = ref * w.cpu().float() * torch.nn.functional.silu(zf)
        out = self.k.gated_rms_norm(x, z, w, 1e-6)
        self.assertLess((out.cpu().float() - ref).abs().max().item(), 0.05)

    def test_packed_attention_does_not_depend_on_packing(self):
        lens = (300, 700, 41)
        q, k, v = (
            _rnd(1041, 16, 128, scale=2.0),
            _rnd(1041, 8, 128, seed=1),
            _rnd(1041, 8, 128, seed=2),
        )

        def attend(*args):
            return self.k.packed_causal_attention(*args, 128**-0.5).cpu()

        cu = torch.tensor([0, 300, 1000, 1041], dtype=torch.int32, device="tpu")
        packed = attend(q, k, v, cu)
        # each sequence alone, padded to another length, with empty sequences after it
        start = 0
        for n in lens:
            sl = slice(start, start + n)
            pad = 1024 - n
            qs, ks, vs = (
                torch.nn.functional.pad(t[sl], (0, 0, 0, 0, 0, pad)) for t in (q, k, v)
            )
            cu_alone = torch.tensor([0, n, n, n], dtype=torch.int32, device="tpu")
            alone = attend(qs, ks, vs, cu_alone)[:n]
            self.assertTrue(torch.equal(alone, packed[sl]), f"sequence of {n} differs")
            start += n

    def test_attention_refuses_flags_that_change_rounding(self):
        from sglang_tpu.envs import envs

        with envs.SGLANG_TPU_RPA_EXP2_SOFTMAX.override(True):
            with self.assertRaisesRegex(ValueError, "SGLANG_TPU_RPA_EXP2_SOFTMAX"):
                self.k.refuse_rpa_rounding_flags()

    def test_packed_gated_delta_net_does_not_depend_on_packing(self):
        lens = (150, 300, 41)
        qkv, b, a = _gdn_inputs(491)

        def gdn(qkv, b, a, starts):
            cu = torch.tensor(starts, dtype=torch.int32, device="tpu")
            return _packed_gdn(self.k, qkv, b, a, cu).cpu()

        packed = gdn(qkv, b, a, [0, 150, 450, 491])
        # each sequence alone, padded to another length, with empty sequences after it
        start = 0
        for n in lens:
            sl = slice(start, start + n)
            pad = (0, 0, 0, 512 - n)
            rows = (torch.nn.functional.pad(t[sl], pad) for t in (qkv, b, a))
            alone = gdn(*rows, [0, n, n, n])[:n]
            self.assertTrue(torch.equal(alone, packed[sl]), f"sequence of {n} differs")
            start += n

    def test_packed_gated_delta_net_equals_the_engine_decode_steps(self):
        tokens = 24
        qkv, b, a = _gdn_inputs(tokens)
        cu = torch.tensor([0, tokens], dtype=torch.int32, device="tpu")
        packed = _packed_gdn(self.k, qkv, b, a, cu).cpu()
        self.assertEqual(packed.shape, (tokens, _GDN_NV, _GDN_D))
        # the engine decodes one token per call, carrying the conv and recurrent state in its pool
        decode_step = _gdn_decode_step_op()
        conv = torch.zeros(2, _GDN_KERNEL - 1, 1, _GDN_DIM, device="tpu")
        recurrent = torch.zeros(2, _GDN_NV, _GDN_D, _GDN_D, device="tpu")
        outs = []
        for t in range(tokens):
            seq_len = torch.tensor([t + 1], dtype=torch.int32, device="tpu")
            row = (x[t : t + 1].clone() for x in (qkv, b, a))
            conv, recurrent, out = decode_step(
                *row, conv, recurrent, *_gdn_weights(), seq_len
            )
            outs.append(out.cpu())
        decoded = torch.cat(outs).reshape(packed.shape)
        self.assertTrue(torch.equal(decoded, packed))

    def test_packed_gated_delta_net_reads_a_log_and_dt_bias_in_bf16(self):
        # The engine serves its fp32 A_log rounded to bf16; a trainer passing the fp32 values gets the same bytes.
        tokens = 64
        qkv, b, a = _gdn_inputs(tokens)
        cu = torch.tensor([0, tokens], dtype=torch.int32, device="tpu")
        conv, a_log, dt_bias = _gdn_weights()
        fp32 = [
            w.float() * (1 + 1e-3 * _rnd(_GDN_NV, seed=6 + i).float())
            for i, w in enumerate((a_log, dt_bias))
        ]

        def gdn(a_log, dt_bias):
            args = (
                qkv,
                b,
                a,
                conv,
                a_log,
                dt_bias,
                cu,
                _GDN_NKQ,
                _GDN_NV,
                _GDN_D,
                _GDN_D,
            )
            return self.k.packed_gated_delta_net(*args).cpu()

        self.assertTrue(
            torch.equal(gdn(*fp32), gdn(*(w.to(torch.bfloat16) for w in fp32)))
        )

    def test_moe_experts_rows_do_not_depend_on_row_count(self):
        hidden, inter, experts, topk = 512, 256, 16, 4
        x = _rnd(1024, hidden)
        w1 = _rnd(experts, hidden, 2 * inter, scale=hidden**-0.5, seed=1)
        w2 = _rnd(experts, inter, hidden, scale=inter**-0.5, seed=2)
        g = torch.Generator().manual_seed(3)
        ids = torch.stack(
            [torch.randperm(experts, generator=g)[:topk] for _ in range(1024)]
        )
        weights = torch.softmax(torch.randn(1024, topk, generator=g), dim=-1)
        weights = weights.to(torch.bfloat16)

        def moe(rows, row_weights, row_ids):
            return self.k.moe_experts(rows, w1, w2, row_weights, row_ids)

        args = (x, weights.to("tpu"), ids.to(torch.int32).to("tpu"))
        self._assert_rows_invariant(moe, args)
        # the first rows against the dense math: a slot is (silu(gate) * up) @ w2, summed with its weight
        n = 32
        xf, w1f, w2f = x[:n].cpu().float(), w1.cpu().float(), w2.cpu().float()
        hidden_out = torch.einsum("rh,rkhn->rkn", xf, w1f[ids[:n]])
        gate, up = hidden_out[..., :inter], hidden_out[..., inter:]
        slots = torch.einsum(
            "rki,rkih->rkh", torch.nn.functional.silu(gate) * up, w2f[ids[:n]]
        )
        ref = (weights[:n].float()[..., None] * slots).sum(dim=1)
        out = moe(*(a[:n].clone() for a in args)).cpu().float()
        self.assertLess((out - ref).abs().max().item(), 0.05)

    def test_moe_refuses_flags_that_change_rounding(self):
        from sglang_tpu.envs import envs

        with envs.SGLANG_TPU_PALLAS_VMEM_RIGHTSIZE.override(True):
            with self.assertRaisesRegex(ValueError, "SGLANG_TPU_PALLAS_VMEM_RIGHTSIZE"):
                self.k.refuse_moe_rounding_flags()

    def test_backward_is_the_gradient_of_the_same_math(self):
        x = _rnd(64, 512).requires_grad_()
        w = _rnd(256, 512, scale=512**-0.5, seed=1).requires_grad_()
        nw = _rnd(256, seed=2).requires_grad_()
        r = _rnd(64, 256, seed=3).requires_grad_()
        target = torch.randint(0, 256, (64,)).to(torch.int32).to("tpu")
        t = torch.full((64,), 0.9, device="tpu")

        # matmul -> plain norm -> residual norm (both of its outputs feed the logits) -> log-prob
        def ours():
            h = self.k.rms_norm(self.k.matmul(x, w), None, nw, 1e-6)
            h, residual = self.k.rms_norm(h, r, nw, 1e-6)
            return self.k.logprob(h + residual, t, target).sum()

        def norm(s):
            return s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6) * nw.float()

        def reference():
            s = norm(x.float() @ w.float().T) + r.float()
            return (
                torch.log_softmax((norm(s) + s) / 0.9, dim=-1)
                .gather(1, target.long()[:, None])
                .sum()
            )

        self._assert_grads_match(ours, reference, (x, w, nw, r))

    def test_backward_of_the_gemma_and_gated_norms(self):
        x, r, z = (_rnd(64, 128, seed=s).requires_grad_() for s in range(3))
        w = _rnd(128, scale=0.1, seed=3).requires_grad_()
        gw = _rnd(128, seed=4).requires_grad_()
        c = _rnd(64, 128, seed=5).float()  # a fixed projection of the output

        def ours():
            h, residual = self.k.rms_norm(x, r, w, 1e-6, gemma=True)
            return (self.k.gated_rms_norm(h + residual, z, gw, 1e-6).float() * c).sum()

        def normed(s):
            return s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6)

        def reference():
            s = x.float() + r.float()
            t = normed(s) * (1.0 + w.float()) + s
            gate = torch.nn.functional.silu(z.float())
            return (normed(t) * gw.float() * gate * c).sum()

        self._assert_grads_match(ours, reference, (x, r, w, z, gw))


if __name__ == "__main__":
    unittest.main()
