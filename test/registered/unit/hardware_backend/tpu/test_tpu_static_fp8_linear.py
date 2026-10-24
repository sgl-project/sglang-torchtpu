"""Static-input-scale fp8 GEMV for RMSNorm-fed weights (TPU decode).

`quantize_rmsnorm_fed_weight` fixes the activation scale at
sqrt(K) * max|g| / fp8_max instead of reducing an amax every step. That is
only safe because an RMSNorm output is bounded: y = x / rms(x) * g gives
|y_i| <= sqrt(K) * |g_i|. These tests pin the bound (the fp8 cast does not
saturate, so an over-range value becomes NaN), the accuracy against a bf16
reference, and that every bound tensor is a base tensor (torch_tpu
re-materializes a view graph input on every call).

They run on CPU and cover the arithmetic; TPU lowering is covered by the
Kimi-K3 serving run.
"""

import unittest
from unittest import mock

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.fp8_linear import (
    apply_tpu_static_fp8_linear,
    quantize_rmsnorm_fed_weight,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
K, N = 512, 256


def _rmsnorm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * g.float()).to(
        torch.bfloat16
    )


def _inputs(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    w = (torch.randn(K, N, generator=gen) * 0.02).to(torch.bfloat16)
    g = (torch.rand(K, generator=gen) * 0.5 + 0.05).to(torch.bfloat16)
    return w, g, gen


class TestStaticFp8Bound(CustomTestCase):
    def test_worst_case_rmsnorm_output_stays_in_range(self):
        """All energy in the element with the largest |g|: |y| hits sqrt(K)*g_max."""
        w, g, _ = _inputs()
        packed = quantize_rmsnorm_fed_weight(w, g)
        x = torch.zeros(1, K)
        x[0, int(g.float().abs().argmax())] = 1.0
        y = _rmsnorm(x, g)
        self.assertLessEqual(
            float((y.float() * packed.inv_x_scale).abs().max()), FP8_MAX
        )
        out = apply_tpu_static_fp8_linear(y, packed)
        self.assertTrue(torch.isfinite(out).all())

    def test_heavy_tailed_inputs_are_finite_and_accurate(self):
        w, g, gen = _inputs(1)
        packed = quantize_rmsnorm_fed_weight(w, g)
        x = torch.distributions.StudentT(3.0).sample((8, K))
        x[:, :4] *= 50  # a few massive channels, as in a residual stream
        y = _rmsnorm(x, g)
        ref = y.float() @ w.float()
        out = apply_tpu_static_fp8_linear(y, packed).float()
        self.assertTrue(torch.isfinite(out).all())
        rel = float((out - ref).norm() / ref.norm())
        self.assertLess(rel, 0.06)


class TestStaticFp8Apply(CustomTestCase):
    def test_matches_bf16_reference(self):
        w, g, gen = _inputs(2)
        packed = quantize_rmsnorm_fed_weight(w, g)
        y = _rmsnorm(torch.randn(4, K, generator=gen), g)
        ref = y.float() @ w.float()
        out = apply_tpu_static_fp8_linear(y, packed)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(tuple(out.shape), (4, N))
        rel = float((out.float() - ref).norm() / ref.norm())
        self.assertLess(rel, 0.06)

    def test_row_padding_does_not_change_real_rows(self):
        """M is padded to 16 for _scaled_mm; each row is independent of it."""
        w, g, gen = _inputs(3)
        packed = quantize_rmsnorm_fed_weight(w, g)
        y = _rmsnorm(torch.randn(17, K, generator=gen), g)
        full = apply_tpu_static_fp8_linear(y, packed)
        self.assertEqual(tuple(full.shape), (17, N))
        for m in (1, 5, 16):
            torch.testing.assert_close(
                apply_tpu_static_fp8_linear(y[:m], packed), full[:m], rtol=0, atol=0
            )

    def test_bound_weight_is_callable(self):
        """Models call the bound weight directly instead of importing the helper."""
        w, g, gen = _inputs(5)
        packed = quantize_rmsnorm_fed_weight(w, g)
        y = _rmsnorm(torch.randn(3, K, generator=gen), g)
        torch.testing.assert_close(
            packed(y), apply_tpu_static_fp8_linear(y, packed), rtol=0, atol=0
        )


class TestStaticFp8Pack(CustomTestCase):
    def test_bound_tensors_are_contiguous_base_tensors(self):
        """A transposed [N, K] parameter must not leave views in the pack."""
        w, g, _ = _inputs(4)
        w_nk = w.t().contiguous()  # how nn.Linear stores it
        packed = quantize_rmsnorm_fed_weight(w_nk.t(), g)
        for name in ("weight", "out_scale", "one"):
            t = getattr(packed, name)
            with self.subTest(name=name):
                self.assertIsNone(t._base)
                self.assertTrue(t.is_contiguous())
        self.assertEqual(packed.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(packed.out_scale.shape), (N,))

    def test_rejects_mismatched_norm_width(self):
        w, g, _ = _inputs()
        with self.assertRaises(ValueError):
            quantize_rmsnorm_fed_weight(w, g[:-1])

    def test_rejects_n_not_multiple_of_16(self):
        w, g, _ = _inputs()
        with self.assertRaises(ValueError):
            quantize_rmsnorm_fed_weight(w[:, :250], g)


class TestStaticFp8PallasGemvHelpers(CustomTestCase):
    """CPU-side pieces of the Pallas GEMV path: N padding and the K block choice.

    The kernel itself is covered on TPU by test_static_fp8_gemv_pallas.py.
    """

    def test_padded_n_matches_unpadded(self):
        """N=2112 (Kimi-K3 MLA q_a) padded to 2176 must not change the output."""
        from sglang_tpu.fp8_linear import pad_static_fp8_n as pad_n

        gen = torch.Generator().manual_seed(6)
        w = (torch.randn(K, 2112, generator=gen) * 0.02).to(torch.bfloat16)
        g = (torch.rand(K, generator=gen) * 0.5 + 0.05).to(torch.bfloat16)
        packed = quantize_rmsnorm_fed_weight(w, g)
        padded = pad_n(packed, force=True)
        self.assertEqual(tuple(padded.weight.shape), (K, 2176))
        self.assertEqual(padded.n_out, 2112)
        self.assertIsNone(padded.weight._base)
        self.assertIsNone(padded.out_scale._base)
        y = _rmsnorm(torch.randn(3, K, generator=gen), g)
        torch.testing.assert_close(
            apply_tpu_static_fp8_linear(y, padded),
            apply_tpu_static_fp8_linear(y, packed),
            rtol=0,
            atol=0,
        )

    def test_aligned_n_is_left_alone(self):
        from sglang_tpu.fp8_linear import pad_static_fp8_n as pad_n

        w, g, _ = _inputs(7)
        packed = quantize_rmsnorm_fed_weight(w, g)
        self.assertIs(pad_n(packed, force=True), packed)
        self.assertIs(pad_n(packed), packed)  # kernel not registered on CPU

    def test_inv_x_tensor_matches_scalar(self):
        w, g, _ = _inputs(8)
        packed = quantize_rmsnorm_fed_weight(w, g)
        # f32 tensor for the kernel, Python float for the XLA path: equal
        # up to the f32 rounding.
        self.assertEqual(
            float(packed.inv_x),
            float(torch.tensor(packed.inv_x_scale, dtype=torch.float32)),
        )

    def test_block_k_divides_k_and_fits_vmem(self):
        from sglang_tpu.kernels.static_fp8_gemv import block_k

        # Kimi-K3 TP32 shapes: KDA qkvg, MoE front, latent up.
        for k, n in ((7168, 1536), (7168, 3968), (3584, 7168)):
            with self.subTest(k=k, n=n):
                bk = block_k(k, n)
                self.assertGreater(bk, 0)
                self.assertEqual(k % bk, 0)
                self.assertEqual(bk % 128, 0)
                self.assertLessEqual(bk, 1792)
        # A weight too wide for any K block falls back to the XLA path.
        self.assertEqual(block_k(7168, 1 << 20), 0)


class TestKimiK3StaticFp8Selector(CustomTestCase):
    def test_parts_and_all(self):
        from sglang_tpu.kimi_k3.static_fp8 import PARTS, selected_parts

        with tpu_envs.SGLANG_TPU_K3_STATIC_FP8.override("moe_front,kda_qkvg"):
            self.assertEqual(selected_parts(), {"moe_front", "kda_qkvg"})
        with tpu_envs.SGLANG_TPU_K3_STATIC_FP8.override("all"):
            # mla_qkv_a is not validated, so "all" leaves it out.
            self.assertEqual(selected_parts(), PARTS - {"mla_qkv_a"})
        with tpu_envs.SGLANG_TPU_K3_STATIC_FP8.override("all,mla_qkv_a"):
            self.assertEqual(selected_parts(), PARTS)
        self.assertEqual(selected_parts(), frozenset())  # unset: everything bf16

    def test_unknown_part_raises(self):
        from sglang_tpu.kimi_k3.static_fp8 import selected_parts

        with tpu_envs.SGLANG_TPU_K3_STATIC_FP8.override("moe_frnt"):
            with self.assertRaises(ValueError):
                selected_parts()

    def test_unknown_part_fails_at_startup(self):
        """apply_server_args_defaults validates the selection before any load."""
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        with (
            tpu_envs.SGLANG_TPU_K3_STATIC_FP8.override("moe_frnt"),
            mock.patch(
                "sglang.srt.arg_groups.overrides.resolving_view",
                side_effect=AssertionError("validation must come first"),
            ),
        ):
            with self.assertRaises(ValueError):
                TpuSRTPlatform().apply_server_args_defaults(mock.Mock())


class TestKimiK3StaticFp8SpecBatch(CustomTestCase):
    def test_latent_up_needs_one_request_under_speculation(self):
        from sglang_tpu.kimi_k3 import check_static_fp8_spec_batch as check

        latent_up = frozenset({"moe_front", "latent_up"})
        check(latent_up, speculative_algorithm="DSPARK", max_running_requests=1)
        check(latent_up, speculative_algorithm=None, max_running_requests=4)
        check(
            frozenset({"moe_front"}),
            speculative_algorithm="DSPARK",
            max_running_requests=4,
        )
        for n in (4, None):
            with self.subTest(max_running_requests=n):
                with self.assertRaisesRegex(ValueError, "max-running-requests 1"):
                    check(
                        latent_up,
                        speculative_algorithm="DSPARK",
                        max_running_requests=n,
                    )


class TestStaticFp8PallasGemvGate(CustomTestCase):
    def _w(self, n):
        return mock.Mock(weight=torch.zeros(1024, n))

    def test_batches_up_to_the_padded_tile(self):
        from sglang_tpu import fp8_linear as f

        x = lambda m: torch.zeros(m, 1024, dtype=torch.bfloat16)
        with mock.patch.object(f, "_STATIC_FP8_GEMV_READY", True):
            self.assertTrue(f._static_fp8_gemv_eligible(x(32), self._w(1536)))
            self.assertFalse(f._static_fp8_gemv_eligible(x(33), self._w(1536)))
            # An unpadded N keeps the XLA dot.
            self.assertFalse(f._static_fp8_gemv_eligible(x(1), self._w(1500)))

    def test_switch_off_leaves_the_op_unregistered(self):
        from sglang_tpu import fp8_linear as f

        with (
            tpu_envs.SGLANG_TPU_ENABLE_K3_STATIC_FP8_PALLAS_GEMV.override(False),
            mock.patch.object(f, "_STATIC_FP8_GEMV_READY", False),
            mock.patch.object(f, "_STATIC_FP8_GEMV_OP", None),
        ):
            f.register_static_fp8_gemv_op()
            self.assertFalse(f.static_fp8_gemv_enabled())
            self.assertIsNone(f._STATIC_FP8_GEMV_OP)


H, LATENT, SHARED, QKVG, QKV_A = 256, 128, 128, 192, 96


class _Norm(nn.Module):
    def __init__(self, width, variance_size_override=None):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(width) + 0.5)
        self.variance_size_override = variance_size_override


def _linear(n, k, unquantized=True):
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    lin = nn.Module()
    lin.quant_method = UnquantizedLinearMethod() if unquantized else object()
    lin.input_size = k
    lin.weight = nn.Parameter(torch.randn(n, k) * 0.02)  # [out, in]
    return lin


class _FakeMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_latent_moe = True
        self.moe_hidden_size = LATENT
        self._shared_experts_tp_comm = False
        self._sbo_shared_overlap = False
        self._eligible_for_fused_front = False
        self.routed_expert_down_proj = _linear(LATENT, H)
        self.routed_expert_up_proj = _linear(H, LATENT)
        self.routed_expert_norm = _Norm(LATENT)
        self.shared_experts = nn.Module()
        self.shared_experts.gate_up_proj = _linear(SHARED, H)
        self._static_fp8_front = None
        self._static_fp8_latent_up = None


class _FakeKDA(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_full_rank_gate = True
        self.fused_qkvg_proj = _linear(QKVG, H)
        self._static_fp8_qkvg = None


class _FakeMLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_qkv_a_proj_with_mqa = _linear(QKV_A, H)
        self._static_fp8_qkv_a = None


def _layer(attn):
    layer = nn.Module()
    layer.self_attn = attn
    layer.mlp = _FakeMoE()
    layer.input_layernorm = _Norm(H)
    layer.post_attention_layernorm = _Norm(H)
    return layer


def _model(*layers):
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList(layers)
    return model


class TestKimiK3StaticFp8Binder(CustomTestCase):
    """bind_static_fp8 on stand-in Kimi-K3 layers (the real classes are
    patched to these fakes, so isinstance dispatch is the binder's own)."""

    def setUp(self):
        from sglang.srt.models import kimi_k3
        from sglang_tpu.kimi_k3 import static_fp8

        patches = [
            mock.patch.object(kimi_k3, "KimiK3MoE", _FakeMoE),
            mock.patch.object(kimi_k3, "KimiK3DeltaAttention", _FakeKDA),
            mock.patch.object(kimi_k3, "KimiK3MLAAttention", _FakeMLA),
            # No JAX on CPU: the binder would register the Pallas op here.
            mock.patch.object(static_fp8, "register_static_fp8_gemv_op"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _bind(self, model, parts):
        from sglang_tpu.kimi_k3.static_fp8 import bind_static_fp8

        with tpu_envs.SGLANG_TPU_K3_STATIC_FP8.override(parts):
            bind_static_fp8(model)

    def test_all_binds_validated_parts_and_releases_weights(self):
        kda, mla = _layer(_FakeKDA()), _layer(_FakeMLA())
        model = _model(kda, mla)
        self._bind(model, "all")
        for layer in (kda, mla):
            moe = layer.mlp
            self.assertIsNotNone(moe._static_fp8_front)
            self.assertEqual(moe._static_fp8_front_split, [LATENT, SHARED])
            self.assertIsNotNone(moe._static_fp8_latent_up)
            self.assertEqual(moe.routed_expert_down_proj.weight.numel(), 0)
            self.assertEqual(moe.shared_experts.gate_up_proj.weight.numel(), 0)
            self.assertEqual(moe.routed_expert_up_proj.weight.numel(), 0)
        self.assertIsNotNone(kda.self_attn._static_fp8_qkvg)
        self.assertEqual(kda.self_attn.fused_qkvg_proj.weight.numel(), 0)
        # "all" leaves mla_qkv_a in bf16.
        self.assertIsNone(mla.self_attn._static_fp8_qkv_a)
        self.assertEqual(
            tuple(mla.self_attn.fused_qkv_a_proj_with_mqa.weight.shape), (QKV_A, H)
        )
        self.assertIn("Online weight updates", model._derived_weight_cache_error)

    def test_bound_front_matches_the_bf16_gemv(self):
        layer = _layer(_FakeKDA())
        moe = layer.mlp
        ref_w = torch.cat(
            [moe.routed_expert_down_proj.weight, moe.shared_experts.gate_up_proj.weight]
        ).detach()
        self._bind(_model(layer), "moe_front")
        x = torch.randn(3, H)
        y = (
            x
            * torch.rsqrt(x.pow(2).mean(-1, keepdim=True))
            * layer.post_attention_layernorm.weight
        )
        y = y.detach().to(torch.bfloat16)
        ref = y.float() @ ref_w.t().float()
        got = moe._static_fp8_front(y).float()
        rel = (got - ref).norm() / ref.norm()
        self.assertLess(float(rel), 0.06)

    def test_logs_the_input_scale_headroom(self):
        layer = _layer(_FakeKDA())
        with self.assertLogs("sglang_tpu.kimi_k3.static_fp8", "INFO") as logs:
            self._bind(_model(layer), "moe_front,kda_qkvg")
        text = "\n".join(logs.output)
        self.assertIn("max|g| / median|g|", text)
        self.assertIn("moe_front", text)

    def test_second_bind_is_a_no_op(self):
        layer = _layer(_FakeKDA())
        model = _model(layer)
        self._bind(model, "all")
        bound = layer.mlp._static_fp8_front
        self._bind(model, "all")
        self.assertIs(layer.mlp._static_fp8_front, bound)

    def test_skips(self):
        from sglang.srt.layers.utils import PPMissingLayer

        partial = _layer(_FakeKDA())
        partial.post_attention_layernorm.variance_size_override = 64
        partial.input_layernorm.variance_size_override = 64
        quantized = _layer(_FakeKDA())
        quantized.self_attn.fused_qkvg_proj = _linear(QKVG, H, unquantized=False)
        transposed = _layer(_FakeKDA())
        # [in, out] storage does not read a K-wide input as [out, in].
        transposed.self_attn.fused_qkvg_proj.weight = nn.Parameter(torch.randn(H, QKVG))
        model = _model(PPMissingLayer(), partial, quantized, transposed)
        self._bind(model, "moe_front,kda_qkvg")
        # A partial-width norm breaks the static bound: nothing it feeds binds.
        self.assertIsNone(partial.mlp._static_fp8_front)
        self.assertIsNone(partial.self_attn._static_fp8_qkvg)
        self.assertIsNone(quantized.self_attn._static_fp8_qkvg)
        self.assertIsNone(transposed.self_attn._static_fp8_qkvg)
        # The MoE front of the other layers is unaffected.
        self.assertIsNotNone(quantized.mlp._static_fp8_front)

    def test_selection_with_no_qualifying_layer_warns(self):
        layer = _layer(_FakeKDA())
        layer.self_attn.use_full_rank_gate = False
        with self.assertLogs("sglang_tpu.kimi_k3.static_fp8", "WARNING") as logs:
            self._bind(_model(layer), "kda_qkvg")
        self.assertIn("kda_qkvg", "\n".join(logs.output))


class TestKimiK3HostSplitSlices(CustomTestCase):
    """Every rank of a host must bind the same kind of tensor: a per-rank view
    (storage offset) or a zeros constant made torch_tpu compile a different
    decode program per rank, and the ranks then hung in a collective."""

    def test_every_rank_gets_a_fresh_base_tensor_of_one_shape(self):
        from sglang_tpu.kimi_k3 import static_fp8 as sf

        w_kn = torch.randn(3584, 64).t()  # [64, 3584] view, like _dense_weight_kn
        used = sf._split_ranks(w_kn.shape[1], 8)
        self.assertEqual(used, 7)  # 3584 = 7 x 512
        slices = [sf._column_slice(w_kn, used, index) for index in range(8)]
        for index, t in enumerate(slices):
            self.assertIsNone(t._base, f"rank {index}")
            self.assertTrue(t.is_contiguous())
            self.assertEqual(t.shape, slices[0].shape)
        torch.testing.assert_close(slices[3], w_kn[:, 1536:2048], rtol=0, atol=0)
        # Past the used ranks: a copy of slice 0 (dropped by the exchange), not zeros.
        torch.testing.assert_close(slices[7], slices[0], rtol=0, atol=0)

    def test_host_split_needs_moe_front_and_latent_up(self):
        from sglang_tpu.kimi_k3 import static_fp8_parts

        with tpu_envs.SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV.override(True):
            with tpu_envs.SGLANG_TPU_K3_STATIC_FP8.override("moe_front,kda_qkvg"):
                with self.assertRaises(ValueError):
                    static_fp8_parts()
            with tpu_envs.SGLANG_TPU_K3_STATIC_FP8.override("all"):
                self.assertIn("latent_up", static_fp8_parts())

    def _layout(self, world, nnodes, rank=13, world_size=None, compile_bs=None):
        from sglang_tpu.kimi_k3 import static_fp8 as sf

        with (
            tpu_envs.SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV.override(True),
            mock.patch(
                "torch.distributed.get_world_size",
                return_value=world if world_size is None else world_size,
            ),
            mock.patch(
                "sglang.srt.distributed.parallel_state."
                "get_tensor_model_parallel_world_size",
                return_value=world,
            ),
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_tensor_model_parallel_rank",
                return_value=rank,
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_server_args",
                return_value=mock.Mock(nnodes=nnodes, tpu_compile_bs=compile_bs),
            ),
        ):
            return sf._host_split_layout()

    def test_layout_from_hosts(self):
        self.assertEqual(self._layout(32, 4), (32, 8, 5))  # tpu7x 2x2x4, TP32

    def test_layout_off_for_other_groups(self):
        from sglang_tpu.kimi_k3 import static_fp8 as sf

        # Not host-major spanning the world, or not a measured topology (one
        # host, or 8 hosts x 8 ranks).
        cases = ((32, 3, 32), (32, 32, 32), (16, 2, 32), (8, 1, 8), (64, 8, 64))
        for world, nnodes, world_size in cases:
            with self.subTest(world=world, nnodes=nnodes, world_size=world_size):
                with self.assertLogs(sf.logger, "WARNING"):
                    self.assertIsNone(
                        self._layout(world, nnodes, world_size=world_size)
                    )

    def test_host_split_op_module_keeps_real_annotations(self):
        """cpu_export_jax_op reads the wrappers' jax.Array annotations; PEP 563
        would turn them into strings and the op build fails at bind time."""
        import ast

        import sglang_tpu.kimi_k3.host_split_op as mod

        tree = ast.parse(open(mod.__file__).read())
        future = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "__future__"
            for alias in node.names
        ]
        self.assertNotIn("annotations", future)

    def test_warns_without_compile_bs_1(self):
        from sglang_tpu.kimi_k3 import static_fp8 as sf

        with self.assertLogs(sf.logger, "WARNING") as logs:
            self.assertEqual(self._layout(32, 4, compile_bs=[2, 4]), (32, 8, 5))
        self.assertIn("no batch size 1", "\n".join(logs.output))

    def test_ranks_must_agree_on_the_switch(self):
        from sglang_tpu.kimi_k3 import static_fp8 as sf

        def gather(seen, flag, group=None):
            seen[:] = [flag, flag, not flag, flag]

        world_group = mock.Mock(world_size=4, cpu_group=None)
        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_world_group",
                return_value=world_group,
            ),
            mock.patch("torch.distributed.all_gather_object", side_effect=gather),
        ):
            with self.assertRaisesRegex(RuntimeError, r"ranks \[0, 1, 3\]"):
                sf._check_ranks_agree(True)

    def test_split_ranks_needs_128_multiples(self):
        from sglang_tpu.kimi_k3 import static_fp8 as sf

        self.assertEqual(sf._split_ranks(7168, 8), 8)
        self.assertEqual(sf._split_ranks(896, 8), 7)
        self.assertEqual(sf._split_ranks(100, 8), 0)


if __name__ == "__main__":
    unittest.main()
