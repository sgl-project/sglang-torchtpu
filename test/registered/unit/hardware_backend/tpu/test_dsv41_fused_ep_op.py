# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the DeepSeek-V4.1 fused EP wiring (SGLANG_TPU_DSV41_FUSED_EP).

sglang_tpu/models/deepseek_v41/fused_ep_op.py converts the FP4 experts in the
gmm layout (see fused_ep.py) into fused_ep_moe_v2 operands at load and sets
each prepared layer's ``mlp.dp_local_forward``, through which the non-CUDA
DeepSeek V4 decoder layer replaces its DP gather -> gmm_v2 -> all-reduce ->
scatter by gate on local rows -> one fused EP op -> + TP1 shared expert.
Nothing here needs a TPU: torch_tpu's mesh and op export are replaced by a CPU
JAX mesh and an eager runner that hands the op body the global arrays
torch_tpu would, and the kernel by dsv41_fused_ep_reference.reference_moe
(fused_ep_op.kernel_override).

    PYTHONPATH=<tree>/python:<tree> python -m pytest -q <this file>
"""

import ast
import inspect
import os
import re
import sys
import types
import zlib
from types import SimpleNamespace

# CPU JAX (bit-exact checks as CPU XLA computes them; the chip is torch_tpu's)
# with four virtual devices for the EP meshes; no effect once JAX is initialized.
if "jax" not in sys.modules:
    os.environ["JAX_PLATFORMS"] = "cpu"
    if "device_count" not in os.environ.get("XLA_FLAGS", ""):
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + " --xla_force_host_platform_device_count=4"
        ).strip()

import numpy as np
import pytest

try:
    from sglang.test.ci.ci_register import register_tpu_ci
except ImportError:  # minimal env without SGLang's runtime dependencies

    def register_tpu_ci(*_args, **_kwargs):
        return None


# The TPU unit lane has jax and ml_dtypes; the CPU lane does not.
register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

jax = pytest.importorskip("jax")
pytest.importorskip("ml_dtypes")
torch = pytest.importorskip("torch")

import dsv41_fused_ep_reference as ref  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from sglang_tpu.models.deepseek_v41 import fused_ep as fe  # noqa: E402
from sglang_tpu.models.deepseek_v41 import fused_ep_op as op_mod  # noqa: E402
from sglang_tpu.models.deepseek_v41 import hooks as dsv41_hooks  # noqa: E402

FP4_X2 = torch.float4_e2m1fn_x2
IMAGE_TOKEN_ID = 7
# Tiny served structure: H and the padded intermediate are one 512 block each,
# the logical intermediate is padded inside each gate/up half like V4.1's.
E, H, I, TOPK, SCALE = 4, 512, 384, 2, 1.5
IP = 512
SHARED_I = 64


# --------------------------------------------------------------------------- #
# gmm-layout experts and an independent decode of the gmm layout.
# --------------------------------------------------------------------------- #
def _checkpoint_experts(num_experts, hidden, inter, seed):
    """Block-32 MXFP4 experts as SGLang's loader leaves them (as in
    test_dsv41_fused_ep_weights.py)."""
    rng = np.random.default_rng(seed)
    w13 = rng.integers(0, 256, (num_experts, 2 * inter, hidden // 2), dtype=np.uint8)
    w2 = rng.integers(0, 256, (num_experts, hidden, inter // 2), dtype=np.uint8)
    e13 = rng.integers(117, 122, (num_experts, 2 * inter, hidden // 32), dtype=np.uint8)
    e2 = rng.integers(117, 122, (num_experts, hidden, inter // 32), dtype=np.uint8)
    return (
        torch.from_numpy(w13),
        ref.e8m0_to_float32(e13),
        torch.from_numpy(w2),
        ref.e8m0_to_float32(e2),
    )


_GMM_CACHE = {}


def _gmm_prepared(num_experts, hidden, inter):
    """The real gmm preparation on random checkpoint experts -> numpy tensors."""
    key = (num_experts, hidden, inter)
    if key not in _GMM_CACHE:
        ckpt = _checkpoint_experts(*key, seed=zlib.crc32(repr(key).encode()) % 1000)
        out = ref.run_gmm_prep(*(t.clone() for t in ckpt))
        _GMM_CACHE[key] = {k: v.numpy() for k, v in out.items()}
    return _GMM_CACHE[key]


def _gmm_layout_dequant(packed_kn, scale):
    """gmm-layout (E, K/2, N) bytes + (E, K/512, 1, N) scale -> f32 [E, K, N]
    with the preparation's own code table (independent of the code under test)."""
    lut = torch.tensor(
        ref.load_gmm_mxfp4_prep().module._FP4_E2M1_LUT, dtype=torch.float32
    )
    w_nk = torch.from_numpy(np.ascontiguousarray(packed_kn)).permute(0, 2, 1)
    low, high = (w_nk & 0x0F).long(), ((w_nk >> 4) & 0x0F).long()
    w32 = torch.stack((lut[low], lut[high]), dim=-1).flatten(-2)  # [E, N, K]
    s = torch.from_numpy(np.ascontiguousarray(scale)).squeeze(2).permute(0, 2, 1)
    deq = (w32.unflatten(-1, (-1, 512)) * s.unsqueeze(-1)).flatten(-2)
    return deq.permute(0, 2, 1).contiguous().numpy()


def _bits(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def _x2_to_float4(t):
    """What torch_tpu hands JAX for a float4_e2m1fn_x2 [..., N/2] tensor:
    float4_e2m1fn [..., N], low nibble = even column (fp4_moe's convention)."""
    b = t.contiguous().view(torch.uint8).numpy()
    codes = np.stack((b & 0x0F, b >> 4), axis=-1).reshape(*b.shape[:-1], -1)
    return ref.codes_to_float4(np.ascontiguousarray(codes))


# --------------------------------------------------------------------------- #
# A V4.1 DeepseekV2MoE stand-in (duck-typed like fused_ep_op expects).
# --------------------------------------------------------------------------- #
class FakeGate(torch.nn.Module):
    def __init__(self, weight, bias, bias_vl):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.e_score_correction_bias = torch.nn.Parameter(bias, requires_grad=False)
        self.e_score_correction_bias_vl = torch.nn.Parameter(
            bias_vl, requires_grad=False
        )

    def forward(self, x, gemm_output_zero_allocator=None, forward_batch=None):
        # MoEGate.forward off CUDA with the weight in the model dtype.
        return F.linear(x, self.weight, None)


class FakeShared(torch.nn.Module):
    """A replicated (TP1) clamped-SwiGLU MLP."""

    def __init__(self, gate_up, down):
        super().__init__()
        self.gate_up = torch.nn.Parameter(gate_up, requires_grad=False)
        self.down = torch.nn.Parameter(down, requires_grad=False)

    def forward(self, x):
        gate, up = F.linear(x, self.gate_up).float().chunk(2, dim=-1)
        act = F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)
        return F.linear(act.to(x.dtype), self.down)


class FakeExperts(torch.nn.Module):
    def __init__(self, gmm, first, count):
        super().__init__()
        for name, value in gmm.items():
            local = torch.from_numpy(np.ascontiguousarray(value[first : first + count]))
            self.register_parameter(
                name, torch.nn.Parameter(local.clone(), requires_grad=False)
            )
        self.num_local_experts = count
        self.quant_method = SimpleNamespace(is_fp4_expert=True)


def _config(**overrides):
    values = dict(
        model_type="deepseek_v41",
        n_routed_experts=E,
        num_experts_per_tok=TOPK,
        norm_topk_prob=True,
        routed_scaling_factor=SCALE,
        scoring_func="sqrtsoftplus",
        swiglu_limit=10.0,
        moe_intermediate_size=I,
        hidden_size=H,
        image_token_id=IMAGE_TOKEN_ID,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeMoE(torch.nn.Module):
    is_deepseek_v4 = True

    def __init__(self, config, gate, experts, shared, *, layer_id=3, tp1=True):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.gate = gate
        self.experts = experts
        self.shared_experts = shared
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_fused_shared_experts = 0
        self._shared_expert_tp1 = tp1
        self.is_hash = False
        self.is_nextn = False
        self.dp_local_forward = None
        from sglang.srt.layers.moe.topk import TopKConfig

        self.topk = SimpleNamespace(
            topk_config=TopKConfig(
                top_k=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                routed_scaling_factor=config.routed_scaling_factor,
                apply_routed_scaling_factor_on_output=False,
                scoring_func=config.scoring_func,
                layer_id=layer_id,
            ),
            # No padding bound and no expert-location dispatch, as resolved
            # outside a forward with trivial expert placement.
            routing_inputs=lambda: (None, None),
        )

    def _forward_shared_experts(self, hidden_states):
        if hidden_states.shape[0] == 0:
            return None
        return self.shared_experts(hidden_states)


def _parts(dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    gate = FakeGate(
        (torch.randn(E, H, generator=g) * 0.08).to(dtype),
        torch.randn(E, generator=g) * 0.1,
        torch.randn(E, generator=g) * 0.1,
    )
    shared = FakeShared(
        (torch.randn(2 * SHARED_I, H, generator=g) * 0.05).to(dtype),
        (torch.randn(H, SHARED_I, generator=g) * 0.05).to(dtype),
    )
    return gate, shared


def _moe(gmm, gate, shared, *, first=0, count=E, **kwargs):
    return FakeMoE(_config(), gate, FakeExperts(gmm, first, count), shared, **kwargs)


# --------------------------------------------------------------------------- #
# The current (switch off) semantics, as the TPU path computes them.
# --------------------------------------------------------------------------- #
def _current_path(moe, x, input_ids, gmm):
    """gate -> vision_topk (TPU torch fallback) -> gmm_v2 experts on the
    gmm-layout weights (dense f32 here) -> x routed_scaling_factor
    (deepseek_v2.py forward_normal, after the experts, in the activation dtype)
    -> + the TP1 shared expert, on the gathered rows."""
    from sglang.srt.multimodal.dsv41.vl_routing import vision_topk

    logits = moe.gate(x)
    topk = vision_topk(moe, logits, input_ids)
    weights, ids = topk.topk_weights, topk.topk_ids
    w1 = torch.from_numpy(
        _gmm_layout_dequant(gmm["w13_weight"], gmm["w13_weight_scale"])
    )
    w2 = torch.from_numpy(_gmm_layout_dequant(gmm["w2_weight"], gmm["w2_weight_scale"]))
    x32 = x.float()
    routed = torch.zeros_like(x32)
    for e in range(w1.shape[0]):
        coef = (weights * (ids == e)).sum(-1, keepdim=True)
        h = x32 @ w1[e]
        act = F.silu(h[:, :I].clamp(max=10.0)) * h[:, IP : IP + I].clamp(-10.0, 10.0)
        routed += coef * (act @ w2[e, :I])
    routed = routed.to(x.dtype)
    routed *= moe.routed_scaling_factor
    return routed + moe._forward_shared_experts(x)


# --------------------------------------------------------------------------- #
# torch_tpu stand-ins: CPU mesh, global-array runner, reference kernel.
# --------------------------------------------------------------------------- #
def _to_jax(t):
    if t.dtype == FP4_X2:
        return jnp.asarray(_x2_to_float4(t))
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _to_torch(a):
    if a.dtype == jnp.bfloat16:
        return torch.from_numpy(np.array(a.astype(jnp.float32))).to(torch.bfloat16)
    return torch.from_numpy(np.array(a))


def _global_args(per_rank_args, specs, mesh):
    """The global arrays torch_tpu builds from every rank's local tensors: a
    sharded operand is the concatenation of the ranks' shards in mesh order,
    a replicated one is any rank's copy (checked equal)."""
    out = []
    for i, spec in enumerate(specs):
        locals_ = [_to_jax(args[i]) for args in per_rank_args]
        if spec == P():
            for other in locals_[1:]:
                np.testing.assert_array_equal(
                    np.asarray(other, np.float32), np.asarray(locals_[0], np.float32)
                )
            value = locals_[0]
        else:
            value = jnp.concatenate(locals_, axis=0)
        out.append(jax.device_put(value, NamedSharding(mesh, spec)))
    return out


class _Exported:
    """Stands in for cpu_export_jax_op's op: one rank calls it eagerly."""

    def __init__(self, name, fn, mesh, in_specs, out_specs):
        self.name, self.fn, self.mesh = name, fn, mesh
        self.in_specs, self.out_specs = in_specs, out_specs
        self.fake = None
        self.calls = []

    def register_fake(self, fn):
        self.fake = fn

    def __call__(self, *args):
        self.calls.append(args)
        if self.mesh.shape["ep"] != 1:
            # Several emulated ranks: collect, run the body once afterwards.
            return torch.zeros_like(args[0])
        (out,) = [self.fn(*_global_args([args], self.in_specs, self.mesh))]
        return _to_torch(out)


@pytest.fixture
def cpu_op(monkeypatch):
    """Patch torch_tpu's mesh and export; returns the exports made."""
    try:
        ref.import_fused_moe_v2(isolated=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"fused_moe_v2 does not import here: {exc!r}")
    exports = []
    meshes = {}

    def get_mesh(world_size, axis_name="ep"):
        if len(jax.devices()) < world_size:
            pytest.skip(f"needs {world_size} CPU devices")
        key = (world_size, axis_name)
        if key not in meshes:
            meshes[key] = Mesh(np.asarray(jax.devices()[:world_size]), (axis_name,))
        return meshes[key]

    def export(name, fn, *, mesh, input_partition_specs, output_partition_specs):
        exported = _Exported(
            name, fn, mesh, input_partition_specs, output_partition_specs
        )
        exports.append(exported)
        return exported

    monkeypatch.setattr(op_mod, "_get_mesh", get_mesh)
    monkeypatch.setattr(op_mod, "_export_op", export)
    return exports


class ReferenceKernel:
    """fused_ep_moe_v2's contract on global arrays, computed by reference_moe."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, w1, w2, w1_scale, w2_scale, gating, **kw):
        self.calls.append(dict(kw, w1=w1, w2=w2, w1_scale=w1_scale, w2_scale=w2_scale))
        out = ref.reference_moe(
            x,
            gating,
            ref.dequantize_kernel_layout(w1, w1_scale),
            ref.dequantize_kernel_layout(w2, w2_scale),
            kw["score_bias"],
            topk=kw["topk"],
            renormalize=kw["renormalize"],
            routed_scaling_factor=kw["routed_scaling_factor"],
            logical_intermediate_size=kw["logical_intermediate_size"],
        )
        return out.astype(x.dtype)


class MaxLen:
    def __init__(self, max_len=True):
        self.max_len = max_len

    def is_max_len(self):
        return self.max_len


def _rel_rows(got, want):
    got, want = got.float(), want.float()
    return ((got - want).norm(dim=-1) / want.norm(dim=-1)).max().item()


# --------------------------------------------------------------------------- #
# Switch off: nothing registered, nothing touched.
# --------------------------------------------------------------------------- #
_LOADER = "sglang.srt.model_loader.loader.DefaultModelLoader.postprocess_weights"


def test_switch_off_registers_no_hook(monkeypatch):
    from sglang.srt.plugins.hook_registry import HookType

    monkeypatch.delenv("SGLANG_TPU_DSV41_FUSED_EP", raising=False)
    assert dsv41_hooks.fused_ep_hook_specs() == []
    off = dsv41_hooks.hook_specs()
    monkeypatch.setenv("SGLANG_TPU_DSV41_FUSED_EP", "0")
    assert dsv41_hooks.hook_specs() == off

    monkeypatch.setenv("SGLANG_TPU_DSV41_FUSED_EP", "1")
    monkeypatch.setenv("SGLANG_SHARED_EXPERT_TP1", "1")
    on = dsv41_hooks.hook_specs()
    assert on[: len(off)] == off, "the default hooks keep their order"
    assert [(t, h, ht) for t, h, ht in on[len(off) :]] == [
        (_LOADER, dsv41_hooks.prepare_tpu_dsv41_fused_ep, HookType.AFTER),
    ]


def test_missing_shared_expert_tp1_fails_at_startup(monkeypatch):
    """The switch without SGLANG_SHARED_EXPERT_TP1=1 fails while the server args
    resolve, not after the weights load; building the hook specs must not raise,
    since load_plugins would only log it and drop every TPU hook."""
    from sglang_tpu.tpu_platform import TpuSRTPlatform

    # Its DSV4 resolution-hook registration imports every TPU hook module.
    dsv4_hooks = types.ModuleType("sglang_tpu.hooks.dsv4")
    dsv4_hooks.register_tpu_dsv4_resolution_hooks = lambda: None
    monkeypatch.setitem(sys.modules, dsv4_hooks.__name__, dsv4_hooks)
    monkeypatch.setenv("SGLANG_TPU_DSV41_FUSED_EP", "1")
    monkeypatch.delenv("SGLANG_SHARED_EXPERT_TP1", raising=False)
    assert dsv41_hooks.fused_ep_hook_specs()
    with pytest.raises(ValueError, match="needs SGLANG_SHARED_EXPERT_TP1=1"):
        TpuSRTPlatform().apply_server_args_defaults(SimpleNamespace())
    monkeypatch.setenv("SGLANG_SHARED_EXPERT_TP1", "1")
    op_mod.check_dsv41_fused_ep_env()
    monkeypatch.delenv("SGLANG_TPU_DSV41_FUSED_EP")
    monkeypatch.delenv("SGLANG_SHARED_EXPERT_TP1")
    op_mod.check_dsv41_fused_ep_env()


def test_hook_target_and_seam_exist():
    from sglang.srt.model_loader.loader import DefaultModelLoader

    assert isinstance(DefaultModelLoader.__dict__["postprocess_weights"], staticmethod)
    params = list(inspect.signature(DefaultModelLoader.postprocess_weights).parameters)
    assert params == ["model", "target_device"]
    # deepseek_v4 does not import in the minimal env; read its source instead.
    # TPU runs DeepseekV4PlatformDecoderLayer, whose MoE step is this mixin's.
    path = os.path.join(
        os.path.dirname(fe.__file__),
        *[".."] * 3,
        "python/sglang/srt/models/deepseek_v4_platform_stage.py",
    )
    with open(path) as f:
        tree = ast.parse(f.read())
    (layer,) = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "NonCudaDecoderMixin"
    ]
    (method,) = [
        n
        for n in layer.body
        if isinstance(n, ast.FunctionDef) and n.name == "_run_moe_ffn_dp_sync"
    ]
    body = ast.unparse(method)
    assert "dp_local_forward = self.mlp.dp_local_forward" in body
    assert "dp_local_forward(hidden_states, forward_batch, input_ids=input_ids)" in body


def test_hook_applies_like_the_registry_does(monkeypatch):
    """AFTER on a staticmethod, through HookRegistry itself, on a stand-in."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    mod = types.ModuleType("_fake_dsv41_targets")

    class Loader:
        @staticmethod
        def postprocess_weights(model, target_device):
            model.append("postprocessed")

    mod.Loader = Loader
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    prepared = []
    monkeypatch.setattr(
        op_mod, "prepare_model_for_dsv41_fused_ep", lambda m: prepared.append(m)
    )
    saved_hooks = {k: list(v) for k, v in HookRegistry._hooks.items()}
    saved_patched = set(HookRegistry._patched)
    try:
        HookRegistry.reset()
        HookRegistry.register(
            f"{mod.__name__}.Loader.postprocess_weights",
            dsv41_hooks.prepare_tpu_dsv41_fused_ep,
            HookType.AFTER,
        )
        HookRegistry.apply_hooks()
    finally:
        HookRegistry.reset()
        HookRegistry._hooks.update(saved_hooks)
        HookRegistry._patched.update(saved_patched)

    assert isinstance(Loader.__dict__["postprocess_weights"], staticmethod)
    model = []
    Loader.postprocess_weights(model, torch.device("cpu"))
    assert model == ["postprocessed"] and prepared == []
    Loader.postprocess_weights(model, SimpleNamespace(type="tpu"))
    assert prepared == [model]

    # A refusal (unmet requirement) leaves the hooked loader step, so it fails
    # model loading and with it server start; the AFTER wrapper swallows nothing.
    def refuse(m):
        raise ValueError("SGLANG_TPU_DSV41_FUSED_EP=1 needs X")

    monkeypatch.setattr(op_mod, "prepare_model_for_dsv41_fused_ep", refuse)
    with pytest.raises(ValueError, match="SGLANG_TPU_DSV41_FUSED_EP=1 needs X"):
        Loader.postprocess_weights(model, SimpleNamespace(type="tpu"))


def test_switch_off_prepares_nothing(monkeypatch):
    monkeypatch.delenv("SGLANG_TPU_DSV41_FUSED_EP", raising=False)
    gate, shared = _parts(torch.bfloat16)
    moe = _moe(_gmm_prepared(E, H, I), gate, shared)
    before = {
        n: (p.data_ptr(), p.shape, p.dtype) for n, p in moe.experts.named_parameters()
    }
    model = torch.nn.ModuleDict({"mlp": moe})
    assert op_mod.prepare_model_for_dsv41_fused_ep(model) == 0
    assert not getattr(moe, op_mod.READY_ATTR, False)
    for name, param in moe.experts.named_parameters():
        assert (param.data_ptr(), param.shape, param.dtype) == before[name]
    assert not dict(moe.experts.named_buffers())


# --------------------------------------------------------------------------- #
# Weights.
# --------------------------------------------------------------------------- #
def test_relayout_matches_numpy_bridge():
    rng = np.random.default_rng(5)
    packed = rng.integers(0, 256, (5, 24, 40), dtype=np.uint8)
    want = ref.gmm_to_torch_fp4_x2_bytes(packed)
    got = op_mod.kpacked_to_npacked_fp4_bytes(torch.from_numpy(packed))
    np.testing.assert_array_equal(got.numpy(), want)
    as_int8 = torch.from_numpy(packed).view(torch.int8)
    for chunk in (1, 2, 3, 8):
        np.testing.assert_array_equal(
            op_mod.relayout_gmm_weight(as_int8, chunk).numpy(), want
        )
    x2 = op_mod.materialize_float4_x2(got)
    assert x2.dtype == FP4_X2 and x2.shape == got.shape
    assert x2.untyped_storage().data_ptr() != got.untyped_storage().data_ptr()
    np.testing.assert_array_equal(
        _x2_to_float4(x2).view(np.uint8), ref.unpack_kpacked_fp4_codes(packed)
    )


@pytest.mark.parametrize(
    "geometry", [(4, 1024, 768), (2, 5120, 2304)], ids=["reduced", "real"]
)
def test_prepared_layer_dequantizes_like_gmm_layout(monkeypatch, geometry):
    num_experts, hidden, inter = geometry
    ip = -(-inter // 512) * 512
    gmm = _gmm_prepared(*geometry)
    built = []
    monkeypatch.setattr(op_mod, "_build_op", lambda *key: built.append(key) or "op")
    g = torch.Generator().manual_seed(1)
    gate = FakeGate(
        torch.zeros(num_experts * 2, hidden, dtype=torch.bfloat16),
        torch.randn(num_experts * 2, generator=g),
        torch.randn(num_experts * 2, generator=g),
    )
    config = _config(
        n_routed_experts=2 * num_experts,
        hidden_size=hidden,
        moe_intermediate_size=inter,
    )
    moe = FakeMoE(
        config,
        gate,
        FakeExperts(gmm, 0, num_experts),
        FakeShared(torch.zeros(2, hidden), torch.zeros(hidden, 1)),
    )
    experts = moe.experts

    # The old tensor is freed before the float4 copy of the new one is made.
    seen = []
    materialize = op_mod.materialize_float4_x2

    def spy(packed_n):
        seen.append(
            (experts.w13_weight.numel(), experts.w2_weight.numel(), packed_n.shape)
        )
        return materialize(packed_n)

    monkeypatch.setattr(op_mod, "materialize_float4_x2", spy)
    with op_mod.kernel_override(None):
        assert op_mod.prepare_moe_layer_for_fused_ep(
            moe, ep=2, ep_rank=1, multimodal_enabled=False
        )
        assert built == [(2, TOPK, True, SCALE, inter, "ep")]
        # Idempotent.
        assert op_mod.prepare_moe_layer_for_fused_ep(
            moe, ep=2, ep_rank=1, multimodal_enabled=False
        )
        assert len(built) == 1
    assert seen[0][:2] == (0, gmm["w2_weight"].size) and seen[1][:2] == (0, 0)

    for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        assert getattr(experts, name).numel() == 0, f"{name} must be freed"
    w1, w2 = getattr(experts, op_mod.W1), getattr(experts, op_mod.W2)
    s1, s2 = getattr(experts, op_mod.W1_SCALE), getattr(experts, op_mod.W2_SCALE)
    assert w1.dtype == w2.dtype == FP4_X2
    assert w1.shape == (num_experts, hidden, ip)  # float4 [E, H, 2*Ip]
    assert w2.shape == (num_experts, ip, hidden // 2)  # float4 [E, Ip, H]
    rows_h, rows_i = (
        fe.fp4_scale_rows(hidden, padded=True),
        fe.fp4_scale_rows(ip, padded=True),
    )
    assert s1.dtype == s2.dtype == torch.float32
    assert s1.shape == (num_experts * rows_h, 2 * ip)
    assert s2.shape == (num_experts * rows_i, hidden)
    rank = getattr(experts, op_mod.RANK)
    assert rank.dtype == torch.int32 and rank.tolist() == [[1]]
    assert set(dict(experts.named_buffers())) == {
        op_mod.W1,
        op_mod.W2,
        op_mod.W1_SCALE,
        op_mod.W2_SCALE,
        op_mod.RANK,
    }
    assert "_fused_ep_w1" not in experts.state_dict()

    # Same operands as fused_ep's host relayout, and bit-identical values.
    want = ref.gmm_to_fused_ep_weights(
        gmm["w13_weight"],
        gmm["w13_weight_scale"],
        gmm["w2_weight"],
        gmm["w2_weight_scale"],
    )
    f1, f2 = _x2_to_float4(w1), _x2_to_float4(w2)
    np.testing.assert_array_equal(f1.view(np.uint8), want.w1.view(np.uint8))
    np.testing.assert_array_equal(f2.view(np.uint8), want.w2.view(np.uint8))
    np.testing.assert_array_equal(_bits(s1.numpy()), _bits(want.w1_scale))
    np.testing.assert_array_equal(_bits(s2.numpy()), _bits(want.w2_scale))
    for w, s, packed, src in (
        (f1, s1.numpy(), gmm["w13_weight"], gmm["w13_weight_scale"]),
        (f2, s2.numpy(), gmm["w2_weight"], gmm["w2_weight_scale"]),
    ):
        np.testing.assert_array_equal(
            _bits(ref.dequantize_kernel_layout(w, s)),
            _bits(_gmm_layout_dequant(packed, src)),
        )


# --------------------------------------------------------------------------- #
# Eligibility.
# --------------------------------------------------------------------------- #
def _parallel(**overrides):
    values = dict(
        attn_tp_size=1,
        attn_dp_size=8,
        moe_ep_size=8,
        moe_ep_rank=3,
        pp_size=1,
        attn_cp_size=1,
        world_group=SimpleNamespace(world_size=8),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _exec(eplb=False, redundant=0, replay=False, **overrides):
    """The server-arg defaults (ExecMoe / ExecFeatures) with these changes."""
    import msgspec

    from sglang.srt.arg_groups.fields.exec_ import ExecFeatures, ExecMoe

    moe_keys = {"expert_distribution_recorder_mode", "init_expert_location"}
    moe = {k: v for k, v in overrides.items() if k in moe_keys}
    features = {k: v for k, v in overrides.items() if k not in moe_keys}
    return SimpleNamespace(
        moe=msgspec.structs.replace(
            ExecMoe(), enable_eplb=eplb, ep_num_redundant_experts=redundant, **moe
        ),
        features=msgspec.structs.replace(
            ExecFeatures(), enable_decoder_swa_bounded_replay=replay, **features
        ),
    )


A2A_NONE = SimpleNamespace(is_none=lambda: True)


def test_global_requirements():
    check = op_mod.unmet_global_requirements
    assert check(_parallel(), _exec(), A2A_NONE) == []
    assert check(_parallel(attn_tp_size=2, attn_dp_size=4), _exec(), A2A_NONE) == [
        "attention TP size 1",
        "attention DP size == EP size (4 vs 8)",
    ]
    assert check(
        _parallel(moe_ep_size=4, world_group=SimpleNamespace(world_size=8)),
        _exec(),
        A2A_NONE,
    )[0].startswith("MoE EP size == world size")
    assert check(_parallel(), _exec(eplb=True), A2A_NONE) == [
        "EPLB and redundant experts off"
    ]
    # Decoder SWA bounded replay is allowed: the tail's real rows are a
    # contiguous prefix and its num_token_non_padded is swapped on entry.
    assert check(_parallel(), _exec(replay=True), A2A_NONE) == []
    # Fused layers skip TopK: no routed-experts capture, no recorder, and the
    # experts are laid out in logical order.
    assert check(_parallel(), _exec(enable_return_routed_experts=True), A2A_NONE) == [
        "--enable-return-routed-experts off (fused layers capture no routing)"
    ]
    for mode in ("stat", "per_token"):
        assert check(
            _parallel(), _exec(expert_distribution_recorder_mode=mode), A2A_NONE
        ) == [
            "--expert-distribution-recorder-mode unset (fused layers record no routing)"
        ]
    assert check(
        _parallel(), _exec(init_expert_location="/tmp/location.json"), A2A_NONE
    ) == ["--init-expert-location trivial (experts in logical order)"]
    a2a = SimpleNamespace(is_none=lambda: False)
    assert check(_parallel(pp_size=2), _exec(), a2a) == [
        "pipeline parallel size 1",
        "no MoE all-to-all backend",
    ]


def test_layer_requirements_and_skips(monkeypatch):
    gate, shared = _parts(torch.bfloat16)
    gmm = _gmm_prepared(E, H, I)
    check = op_mod.unmet_layer_requirements
    assert check(_moe(gmm, gate, shared), ep=1, multimodal_enabled=False) == []
    assert check(_moe(gmm, gate, shared, count=2), ep=2, multimodal_enabled=False) == []
    # Vision routing may run unless multimodal serving is explicitly off.
    for mm in (None, True):
        (unmet,) = check(_moe(gmm, gate, shared), ep=1, multimodal_enabled=mm)
        assert unmet.startswith("text-only serving")
    (unmet,) = check(_moe(gmm, gate, shared, tp1=False), ep=1, multimodal_enabled=False)
    assert "SGLANG_SHARED_EXPERT_TP1=1" in unmet
    moe = _moe(gmm, gate, shared)
    moe.experts.quant_method.is_fp4_expert = False
    (unmet,) = check(moe, ep=1, multimodal_enabled=False)
    assert "SGLANG_DSV4_FP4_EXPERTS=1" in unmet
    moe = _moe(gmm, gate, shared)
    moe.config.swiglu_limit = 7.0
    assert check(moe, ep=1, multimodal_enabled=False) == [
        "swiglu_limit 10 (fixed in the kernel's activation)"
    ]
    assert check(_moe(gmm, gate, shared), ep=3, multimodal_enabled=False) == [
        "routed experts split evenly over EP"
    ]

    # A hash-routed gate has no correction bias (MoEGate), so such a layer is
    # refused even if is_hash were not set: the op ignores input_ids.
    no_bias_gate, _ = _parts(torch.bfloat16)
    no_bias_gate.e_score_correction_bias = None
    assert check(_moe(gmm, no_bias_gate, shared), ep=1, multimodal_enabled=False) == [
        "fp32 correction bias over every routed expert"
    ]

    # Hash layers always keep gmm_v2; a refused layer raises.
    moe = _moe(gmm, gate, shared)
    moe.is_hash = True
    assert not op_mod.prepare_moe_layer_for_fused_ep(
        moe, ep=1, ep_rank=0, multimodal_enabled=False
    )
    assert moe.experts.w13_weight.numel() > 0
    assert moe.dp_local_forward is None
    with pytest.raises(ValueError, match="SGLANG_SHARED_EXPERT_TP1"):
        op_mod.prepare_moe_layer_for_fused_ep(
            _moe(gmm, gate, shared, tp1=False),
            ep=1,
            ep_rank=0,
            multimodal_enabled=False,
        )


def test_prepare_model_checks_and_counts(monkeypatch):
    import sglang.srt.layers.moe as moe_pkg
    import sglang.srt.runtime_context as rc

    monkeypatch.setenv("SGLANG_TPU_DSV41_FUSED_EP", "1")
    monkeypatch.setattr(op_mod, "_build_op", lambda *key: "op")
    monkeypatch.setattr(op_mod, "_multimodal_enabled", lambda: False)
    monkeypatch.setattr(rc, "get_exec", lambda: _exec())
    monkeypatch.setattr(moe_pkg, "get_moe_a2a_backend", lambda: A2A_NONE)
    gate, shared = _parts(torch.bfloat16)
    gmm = _gmm_prepared(E, H, I)
    hashed = _moe(gmm, gate, shared, count=2, layer_id=0)
    hashed.is_hash = True
    # DSpark draft stages and MTP layers build their MoE with is_nextn=True.
    draft = _moe(gmm, gate, shared, count=2, layer_id=3)
    draft.is_nextn = True
    layers = (
        [hashed] + [_moe(gmm, gate, shared, count=2, layer_id=i) for i in (1, 2)]
    ) + [draft]
    model = torch.nn.ModuleList(layers)

    monkeypatch.setattr(rc, "get_parallel", lambda: _parallel(attn_tp_size=2))
    with pytest.raises(ValueError, match="attention TP size 1"):
        op_mod.prepare_model_for_dsv41_fused_ep(model)
    assert not any(getattr(m, op_mod.READY_ATTR, False) for m in layers)

    parallel = _parallel(
        attn_dp_size=2,
        moe_ep_size=2,
        moe_ep_rank=1,
        world_group=SimpleNamespace(world_size=2),
    )
    monkeypatch.setattr(rc, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        rc, "get_exec", lambda: _exec(enable_return_routed_experts=True)
    )
    with pytest.raises(
        ValueError, match="SGLANG_TPU_DSV41_FUSED_EP=1 needs --enable-return-routed"
    ):
        op_mod.prepare_model_for_dsv41_fused_ep(model)
    assert not any(getattr(m, op_mod.READY_ATTR, False) for m in layers)

    monkeypatch.setattr(rc, "get_exec", lambda: _exec())
    with op_mod.kernel_override(None):
        assert op_mod.prepare_model_for_dsv41_fused_ep(model) == 3
    assert [getattr(m, op_mod.READY_ATTR, False) for m in layers] == [
        False,
        True,
        True,
        True,
    ]
    # Only prepared layers take the seam in _run_moe_ffn_dp_sync; hash-routed
    # layers keep their gmm_v2 experts while DSpark draft layers (is_nextn)
    # are prepared alongside target layers.
    assert hashed.experts.w13_weight.numel() > 0
    assert hashed.dp_local_forward is None
    for m in layers[1:]:
        assert m.dp_local_forward.func is op_mod.forward_fused_ep_moe
        assert m.dp_local_forward.args == (m,)
    # A model without V4.1 MoE layers (e.g. a draft) is left alone.
    assert op_mod.prepare_model_for_dsv41_fused_ep(torch.nn.Linear(2, 2)) == 0


# --------------------------------------------------------------------------- #
# Pad rows, equal shapes, op naming.
# --------------------------------------------------------------------------- #
def test_pad_rows_get_minus_inf_logits():
    gate, shared = _parts(torch.bfloat16)
    moe = _moe(_gmm_prepared(E, H, I), gate, shared)
    x = torch.randn(8, H).to(torch.bfloat16)
    full = gate(x).float()
    for n in (0, 5, 8):
        for dtype in (torch.int32, torch.int64):
            got = op_mod.local_gating_logits(moe, x, torch.tensor(n, dtype=dtype))
            assert got.dtype == torch.float32
            np.testing.assert_array_equal(
                _bits(got[:n].numpy()), _bits(full[:n].numpy())
            )
            assert torch.isneginf(got[n:]).all()
    np.testing.assert_array_equal(
        _bits(op_mod.local_gating_logits(moe, x, None).numpy()), _bits(full.numpy())
    )


def test_equal_rows_are_checked():
    check = op_mod.check_equal_dp_rows
    check(
        16, SimpleNamespace(dp_padding_mode=MaxLen(), global_num_tokens_cpu=[16] * 4), 4
    )
    check(16, SimpleNamespace(dp_padding_mode=None, global_num_tokens_cpu=None), 1)
    for fb, match in (
        (
            SimpleNamespace(
                dp_padding_mode=MaxLen(False), global_num_tokens_cpu=[16] * 4
            ),
            "MAX_LEN",
        ),
        (
            SimpleNamespace(
                dp_padding_mode=MaxLen(), global_num_tokens_cpu=[16, 16, 8, 16]
            ),
            "equal rows",
        ),
        (
            SimpleNamespace(dp_padding_mode=MaxLen(), global_num_tokens_cpu=[16] * 2),
            "4 per-rank",
        ),
        (
            SimpleNamespace(dp_padding_mode=MaxLen(), global_num_tokens_cpu=None),
            "global_num_tokens_cpu",
        ),
    ):
        with pytest.raises(RuntimeError, match=match):
            check(16, fb, 4)


def test_zero_rows_skip_the_op(monkeypatch):
    moe = SimpleNamespace(**{op_mod._KEY_ATTR: (2, TOPK, True, SCALE, I, "ep")})
    monkeypatch.setattr(op_mod, "_get_op", lambda *k: pytest.fail("op called"))
    fb = SimpleNamespace(dp_padding_mode=MaxLen(), global_num_tokens_cpu=[0, 0])
    out = op_mod.forward_fused_ep_moe(moe, torch.empty(0, H, dtype=torch.bfloat16), fb)
    assert out.shape == (0, H) and out.dtype == torch.bfloat16


def test_op_names_are_unique_and_cached(monkeypatch):
    keys = [
        (8, 6, True, 1.5, 2304),
        (4, 6, True, 1.5, 2304),
        (8, 8, True, 1.5, 2304),
        (8, 6, False, 1.5, 2304),
        (8, 6, True, 2.5, 2304),
        (8, 6, True, 1.5, 2048),
        (8, 6, True, 1.5, 2304, "model"),
    ]
    names = [op_mod._op_name(*k) for k in keys]
    assert len(set(names)) == len(names)
    assert (
        names[0]
        == "sglang_tpu_pallas::dsv41_fused_ep_ep8_k6_r1_s1_5_i2304_c128_q512_ep"
    )
    for name in names:
        assert re.fullmatch(r"sglang_tpu_pallas::[A-Za-z0-9_]+", name)

    built = []
    monkeypatch.setattr(op_mod, "_build_op", lambda *k: built.append(k) or object())
    with op_mod.kernel_override(None):
        a = op_mod._get_op(8, 6, True, 1.5, 2304)
        assert op_mod._get_op(8, 6, True, 1.5, 2304) is a
        assert op_mod._get_op(8.0, 6, 1, 1.5, 2304) is a  # normalized key
        assert op_mod._get_op(8, 6, True, 2.5, 2304) is not a
        assert built == [(8, 6, True, 1.5, 2304, "ep"), (8, 6, True, 2.5, 2304, "ep")]
    assert (8, 6, True, 1.5, 2304, "ep") not in op_mod._OPS  # override restored


def test_build_op_specs_and_signature(cpu_op):
    with op_mod.kernel_override(ReferenceKernel()):
        op = op_mod._get_op(1, TOPK, True, SCALE, I)
    (exported,) = cpu_op
    assert op is exported
    assert exported.name == op_mod._op_name(1, TOPK, True, SCALE, I)
    rows = P("ep")
    assert exported.in_specs == (rows,) * 6 + (P(), rows)
    assert exported.out_specs == (rows,)
    sig = inspect.signature(exported.fn)
    assert list(sig.parameters) == [
        "x",
        "gating",
        "w1",
        "w2",
        "w1_scale",
        "w2_scale",
        "score_bias",
        "rank",
    ]
    # cpu_export_jax_op treats a parameter not annotated jax.Array as static.
    assert all(p.annotation is jax.Array for p in sig.parameters.values())
    x = torch.empty(16, H, dtype=torch.bfloat16)
    fake = exported.fake(x, *[None] * 7)
    assert fake.shape == x.shape and fake.dtype == x.dtype


# --------------------------------------------------------------------------- #
# The fused forward against today's semantics.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_fused_forward_matches_current_path(cpu_op, dtype):
    gmm = _gmm_prepared(E, H, I)
    gate, shared = _parts(dtype, seed=2)
    moe = _moe(gmm, gate, shared)
    kernel = ReferenceKernel()
    g = torch.Generator().manual_seed(9)
    x = torch.randn(8, H, generator=g).to(dtype)
    ids = torch.randint(10, 100, (8,), generator=g)  # no image token
    fb = SimpleNamespace(
        num_token_non_padded=torch.tensor(6, dtype=torch.int32),
        global_num_tokens_cpu=None,
        dp_padding_mode=None,
    )
    with op_mod.kernel_override(kernel):
        assert op_mod.prepare_moe_layer_for_fused_ep(
            moe, ep=1, ep_rank=0, multimodal_enabled=False
        )
        # As NonCudaDecoderMixin._run_moe_ffn_dp_sync calls it.
        got = moe.dp_local_forward(x, fb, input_ids=ids)
    assert got.dtype == dtype and got.shape == x.shape

    want = _current_path(_moe(gmm, gate, shared), x, ids, gmm)
    tol = 2e-5 if dtype == torch.float32 else 1e-2
    assert _rel_rows(got[:6], want[:6]) < tol
    # Padding rows: routed part exactly 0, so only the shared expert remains.
    np.testing.assert_array_equal(
        # shared() over the same full batch: a CPU matmul over 2 rows may sum in
        # another order than over all 8 and differ in the last f32 bits.
        _bits(got[6:].float().numpy()),
        _bits(shared(x)[6:].float().numpy()),
    )

    (call,) = kernel.calls
    assert call["topk"] == TOPK and call["renormalize"] is True
    assert call["routed_scaling_factor"] == SCALE
    assert call["logical_intermediate_size"] == I
    assert call["mesh_ep_ranks"] == (0,) and call["dispatch"] == "all_gather"
    np.testing.assert_array_equal(np.asarray(call["rank"]), [[0]])
    np.testing.assert_array_equal(
        np.asarray(call["score_bias"]), gate.e_score_correction_bias.numpy()
    )
    (exported,) = cpu_op
    (args,) = exported.calls
    assert args[0] is x and args[6] is gate.e_score_correction_bias
    assert args[1].dtype == torch.float32 and torch.isneginf(args[1][6:]).all()


def test_two_ranks_against_gathered_current_path(cpu_op):
    """EP2 emulated in one process: each rank prepares its half of the experts
    and calls the op with its own rows (rank 1 idle, every row padding); the
    op body then runs once on the global arrays over a 2-device mesh."""
    dtype = torch.bfloat16
    gmm = _gmm_prepared(E, H, I)
    gate, shared = _parts(dtype, seed=4)
    kernel = ReferenceKernel()
    g = torch.Generator().manual_seed(11)
    t_local = 8
    xs = [torch.randn(t_local, H, generator=g).to(dtype) for _ in range(2)]
    real = [5, 0]
    outs, moes = [], []
    with op_mod.kernel_override(kernel):
        for rank in range(2):
            moe = _moe(gmm, gate, shared, first=2 * rank, count=2)
            assert op_mod.prepare_moe_layer_for_fused_ep(
                moe, ep=2, ep_rank=rank, multimodal_enabled=False
            )
            fb = SimpleNamespace(
                num_token_non_padded=torch.tensor(real[rank], dtype=torch.int32),
                global_num_tokens_cpu=[t_local, t_local],
                dp_padding_mode=MaxLen(),
            )
            outs.append(moe.dp_local_forward(xs[rank], fb, input_ids=None))
            moes.append(moe)
        (exported,) = cpu_op  # one op for both ranks (same key)
        assert len(exported.calls) == 2
        assert [c[7].tolist() for c in exported.calls] == [[[0]], [[1]]]
        routed = exported.fn(
            *_global_args(exported.calls, exported.in_specs, exported.mesh)
        )
    (call,) = kernel.calls
    np.testing.assert_array_equal(np.asarray(call["rank"]), [[0], [1]])
    assert call["w1"].shape == (E, H, 2 * IP) and call["w2"].shape == (E, IP, H)
    assert call["mesh"] is exported.mesh and call["mesh_ep_ranks"] == (0, 1)
    routed = _to_torch(routed)
    fused = [routed[r * t_local : (r + 1) * t_local] + outs[r] for r in range(2)]

    want = _current_path(
        _moe(gmm, gate, shared), torch.cat(xs), torch.full((2 * t_local,), 11), gmm
    )
    assert _rel_rows(fused[0][:5], want[:5]) < 1e-2
    assert not routed[5:t_local].any() and not routed[t_local:].any()
    for r in range(2):
        np.testing.assert_array_equal(
            _bits(outs[r].float().numpy()), _bits(shared(xs[r]).float().numpy())
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# --------------------------------------------------------------------------- #
# TARGET_VERIFY: only the live requests' verify rows are real.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_live_count_keeps_real_rows_bit_identical(cpu_op, dtype):
    """Masking the padding rows changes nothing on the live rows: the fused
    forward with count = live rows equals the one counting every row, bitwise,
    on the live rows; the padding rows then carry the shared expert only."""
    gmm = _gmm_prepared(E, H, I)
    gate, shared = _parts(dtype, seed=3)
    moe = _moe(gmm, gate, shared)
    g = torch.Generator().manual_seed(11)
    x = torch.randn(8, H, generator=g).to(dtype)
    ids = torch.randint(10, 100, (8,), generator=g)

    def run(count):
        fb = SimpleNamespace(
            num_token_non_padded=torch.tensor(count, dtype=torch.int32),
            global_num_tokens_cpu=None,
            dp_padding_mode=None,
        )
        return moe.dp_local_forward(x, fb, input_ids=ids)

    with op_mod.kernel_override(ReferenceKernel()):
        assert op_mod.prepare_moe_layer_for_fused_ep(
            moe, ep=1, ep_rank=0, multimodal_enabled=False
        )
        all_rows = run(8)
        live = run(6)
    np.testing.assert_array_equal(
        _bits(live[:6].float().numpy()), _bits(all_rows[:6].float().numpy())
    )
    np.testing.assert_array_equal(
        _bits(live[6:].float().numpy()), _bits(shared(x)[6:].float().numpy())
    )


def test_set_live_verify_rows(monkeypatch):
    fb = SimpleNamespace(
        num_token_non_padded=torch.tensor(48, dtype=torch.int32),
        global_num_token_non_padded=torch.tensor(48, dtype=torch.int32),
        global_num_token_non_padded_cpu=48,
    )
    monkeypatch.setattr(op_mod, "is_dsv41_fused_ep_enabled", lambda: False)
    op_mod.set_live_verify_rows(fb, 6)
    assert int(fb.num_token_non_padded) == 48
    monkeypatch.setattr(op_mod, "is_dsv41_fused_ep_enabled", lambda: True)
    op_mod.set_live_verify_rows(fb, 6)  # one live request x 6 verify rows
    assert int(fb.num_token_non_padded) == 6
    assert fb.num_token_non_padded.dtype == torch.int32
    assert fb.num_token_non_padded.shape == torch.Size([])
    assert int(fb.global_num_token_non_padded) == 6
    assert fb.global_num_token_non_padded_cpu == 6
    op_mod.set_live_verify_rows(fb, 0)  # idle DP rank: in lockstep, routes none
    assert int(fb.num_token_non_padded) == 0
    none = SimpleNamespace(num_token_non_padded=None)
    op_mod.set_live_verify_rows(none, 6)
    assert none.num_token_non_padded is None
