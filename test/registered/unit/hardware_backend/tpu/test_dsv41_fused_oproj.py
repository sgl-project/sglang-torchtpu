"""DeepSeek-V4.1 fused output projection (wo_a + wo_b): CPU checks.

The kernel itself needs a v7 TPU; here its arithmetic (``reference_output_projection``)
is checked against the default path, the weight layout round trip, the hook routing,
and that the kernel lowers for TPU.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang_tpu.models.deepseek_v41 import output_projection as oproj

try:
    from sglang.test.ci.ci_register import register_tpu_ci
except ImportError:  # minimal env without SGLang's runtime dependencies

    def register_tpu_ci(*_args, **_kwargs):
        return None


register_tpu_ci(est_time=45, suite="stage-a-test-tpu-unit", chips=1)

G, R, D, H, HD, N = 8, 1024, 4096, 64, 512, 5120


def _wo_b_fp8(seed=1):
    """A wo_b in the TPU FP8 linear's post-load layout: FP8 [K, N], FP32 [N] scale."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(G * R, N, generator=g) * 0.02
    scale = torch.clamp(w.abs().amax(dim=0) / 448.0, min=1e-12)
    return (w / scale).to(torch.float8_e4m3fn), scale


def _default_wo_b(y, wb, sb):
    """apply_tpu_fp8_linear's math: per-row FP32 amax/448 scale, FP8 dot, scales."""
    x_scale = torch.clamp(y.abs().amax(dim=1, keepdim=True).float() / 448.0, min=1e-12)
    x_q = (y.float() / x_scale).to(torch.float8_e4m3fn)
    out = (x_q.float() @ wb.float()) * x_scale * sb.reshape(1, -1).float()
    return out.to(y.dtype)


def _default_path(x, wo_a_bf16, wb, sb):
    """BF16 grouped wo_a einsum, then the default FP8 wo_b."""
    t = x.shape[0]
    y = torch.einsum("tgd,grd->tgr", x.reshape(t, G, D), wo_a_bf16.view(G, R, D))
    return _default_wo_b(y.flatten(1), wb, sb)


def _close(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    rel = (a - b).norm() / b.norm()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0)
    return rel.item(), cos.item()


def test_quantize_wo_a_layout_round_trip_is_exact():
    # Small integers with one 448 per output channel quantize to FP8 exactly
    # (scale 1), so any layout slip shows as a mismatch.
    g = torch.Generator().manual_seed(0)
    w = torch.randint(-8, 9, (G * R, D), generator=g).float()
    w[:, 0] = 448.0
    w = w.to(torch.bfloat16)
    wa, sa = oproj.quantize_wo_a(w)
    assert wa.shape == (D, G * R) and wa.dtype == torch.float8_e4m3fn
    assert sa.shape == (G * R,) and torch.all(sa == 1.0)
    back = oproj.dequantize_wo_a(wa, sa)
    assert back.shape == (G, R, D)
    assert torch.equal(back, w.view(G, R, D))
    # Group g's output channel r is column g * R + r.
    assert torch.equal(wa[:, 3 * R + 5].float(), w.view(G, R, D)[3, 5].float())


def test_reference_wo_b_matches_default_fp8_linear():
    g = torch.Generator().manual_seed(2)
    y = (torch.randn(24, G * R, generator=g) * 0.5).to(torch.bfloat16)
    wb, sb = _wo_b_fp8()
    # The reference's wo_b stage alone: identity wo_a is not expressible, so
    # compare the two FP8 row quantizations directly.
    s = y.float().abs().amax(dim=1, keepdim=True) / 448.0
    xs = torch.where(s == 0, torch.ones_like(s), s)
    qb = torch.clamp(y.float() / xs, -448.0, 448.0).to(torch.float8_e4m3fn)
    kernel_b = ((qb.float() @ wb.float()) * xs * sb.reshape(1, -1)).to(y.dtype)
    rel, cos = _close(kernel_b, _default_wo_b(y, wb, sb))
    assert rel < 1e-3 and cos > 0.99999


@pytest.mark.parametrize("tokens", [6, 48])
def test_reference_is_close_to_default_path(tokens):
    g = torch.Generator().manual_seed(3)
    x = (torch.randn(tokens, H, HD, generator=g)).to(torch.bfloat16)
    wo_a = (torch.randn(G * R, D, generator=g) * 0.02).to(torch.bfloat16)
    wb, sb = _wo_b_fp8()
    wa, sa = oproj.quantize_wo_a(wo_a)
    fused = oproj.reference_output_projection(x, wa, sa, wb, sb)
    assert fused.shape == (tokens, N) and fused.dtype == torch.bfloat16
    rel, cos = _close(fused, _default_path(x, wo_a, wb, sb))
    # FP8 activations + per-channel FP8 wo_a vs a BF16 einsum.
    assert rel < 0.06 and cos > 0.998, (rel, cos)


def test_token_envelope():
    # Every positive count runs the kernel, incl. the vendored-excluded 321..415
    # (e.g. the 384-row bs64 verify), as two tiles.
    for t in (1, 6, 48, 96, 192, 288, 320, 321, 384, 415, 416, 1024, 2048):
        assert oproj.token_count_supported(t), t
    assert not oproj.token_count_supported(0)
    from sglang_tpu.kernels import dsv41_output_projection as kernel

    assert kernel.token_tiles(384) == (2, 192, 0)


def test_layer_filter(monkeypatch):
    from sglang_tpu.envs import envs as tpu_envs

    monkeypatch.setattr(tpu_envs.SGLANG_TPU_DSV41_FUSED_OPROJ_LAYERS, "get", lambda: "")
    assert oproj.layer_filter() is None
    monkeypatch.setattr(
        tpu_envs.SGLANG_TPU_DSV41_FUSED_OPROJ_LAYERS, "get", lambda: "0-2, 7,10"
    )
    assert oproj.layer_filter() == {0, 1, 2, 7, 10}


class _FakeWoB(torch.nn.Module):
    def __init__(self, wb, sb):
        super().__init__()
        self.weight = torch.nn.Parameter(wb, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(sb.reshape(-1, 1), requires_grad=False)
        self.bias = None
        self.calls = 0

    def forward(self, y, skip_all_reduce=False):
        self.calls += 1
        return _default_wo_b(y, self.weight, self.weight_scale.reshape(-1)), None


def _fake_layer(layer_id=3, seed=4):
    g = torch.Generator().manual_seed(seed)
    attn = torch.nn.Module()
    attn.is_dsv41 = True
    attn.attn_tp_size = 1
    attn.n_local_heads, attn.n_local_groups, attn.o_lora_rank = H, G, R
    attn.layer_id = layer_id
    attn.wo_a = torch.nn.Module()
    attn.wo_a.weight = torch.nn.Parameter(
        (torch.randn(G * R, D, generator=g) * 0.02).to(torch.bfloat16),
        requires_grad=False,
    )
    attn.wo_b = _FakeWoB(*_wo_b_fp8())
    return attn


def test_unmet_requirements_and_prepare_layer():
    attn = _fake_layer()
    assert oproj.unmet_layer_requirements(attn) == []
    bad = _fake_layer()
    bad.attn_tp_size = 2
    assert any("attn_tp_size" in m for m in oproj.unmet_layer_requirements(bad))
    mega = _fake_layer()
    mega.use_mega_attention = True
    assert any("mega" in m for m in oproj.unmet_layer_requirements(mega))

    wo_a_bf16 = attn.wo_a.weight.data.clone()
    oproj.prepare_layer(attn)
    assert getattr(attn, oproj._ENABLED_ATTR)
    assert attn.wo_a.weight.numel() == 0  # BF16 copy released
    assert attn.dsv41_oproj_wa.shape == (D, G * R)
    assert attn.dsv41_oproj_sb.shape == (N,)
    wa, sa = oproj.quantize_wo_a(wo_a_bf16)
    assert torch.equal(attn.dsv41_oproj_wa.data.float(), wa.float())
    assert torch.equal(attn.dsv41_oproj_sa.data, sa)


@pytest.fixture
def routed(monkeypatch):
    """Kernel op -> the torch reference; backend inverse RoPE -> identity (CSA2)."""
    calls = []

    def fake_op(x, wa, sa, wb, sb):
        calls.append(x.shape[0])
        return oproj.reference_output_projection(x, wa, sa, wb, sb)

    monkeypatch.setattr(oproj, "register_op", lambda: fake_op)
    backend = SimpleNamespace(apply_mqa_inv_rope=lambda layer, o, positions: o)
    import sglang.srt.model_executor.forward_context as fc

    monkeypatch.setattr(fc, "get_attn_backend", lambda: backend)
    return calls


def _project(attn, x):
    def original_a(*args, **kwargs):
        raise AssertionError("prepared layer must not run the default wo_a")

    def original_b(*args, **kwargs):
        raise AssertionError("prepared layer must not run the default wo_b")

    pre = oproj.project_wo_a(
        original_a,
        attn,
        x,
        positions=None,
        forward_batch=None,
        inv_rope=None,
        wo_a_quant_freqs=None,
    )
    assert isinstance(pre, oproj.PreWoA)
    return oproj.project_wo_b(original_b, attn, pre, False)


def test_prepared_layer_runs_kernel(routed):
    attn = _fake_layer()
    oproj.prepare_layer(attn)
    x = torch.randn(48, H, HD).to(torch.bfloat16)
    out = _project(attn, x)
    assert routed == [48] and attn.wo_b.calls == 0
    ref = oproj.reference_output_projection(
        x,
        attn.dsv41_oproj_wa,
        attn.dsv41_oproj_sa,
        attn.wo_b.weight,
        attn.dsv41_oproj_sb,
    )
    assert torch.equal(out, ref)


def test_bs64_verify_rows_run_the_kernel(routed):
    """384 rows (bs64 x 6) used to fall back to a per-call FP8 dequantization."""
    attn = _fake_layer()
    oproj.prepare_layer(attn)
    x = torch.randn(384, H, HD).to(torch.bfloat16)
    out = _project(attn, x)
    assert routed == [384] and attn.wo_b.calls == 0 and out.shape == (384, N)


def test_non_bf16_input_is_cast(routed):
    attn = _fake_layer()
    oproj.prepare_layer(attn)
    out = _project(attn, torch.randn(12, H, HD, dtype=torch.float32))
    assert routed == [12] and out.dtype == torch.bfloat16


def test_empty_batch(routed):
    attn = _fake_layer()
    oproj.prepare_layer(attn)
    out = _project(attn, torch.empty(0, H, HD, dtype=torch.bfloat16))
    assert out.shape == (0, N) and routed == []


def test_unprepared_layer_uses_default():
    attn = _fake_layer()
    seen = []
    out = oproj.project_wo_a(
        lambda *a, **k: seen.append("a") or "A",
        attn,
        None,
        positions=None,
        forward_batch=None,
        inv_rope=None,
        wo_a_quant_freqs=None,
    )
    assert out == "A"
    assert oproj.project_wo_b(lambda *a: seen.append("b") or "B", attn, "y") == "B"
    assert seen == ["a", "b"]


def test_kernel_lowers_for_tpu(monkeypatch):
    jax = pytest.importorskip("jax")
    from jax.experimental.pallas import tpu as pltpu

    try:
        from jax._src import tpu_info  # noqa: F401
    except ImportError:
        tpu_info = None
    if not hasattr(pltpu, "matmul_push_rhs") or tpu_info is None:
        pytest.skip(f"jax {jax.__version__}: the kernel targets jax 0.11.2")
    import jax.numpy as jnp
    from jax._src import tpu_info as core_tpu_info  # noqa: F401  (JAX >= 0.11)

    import sglang_tpu.pallas_export as pallas_export
    from sglang_tpu.kernels import dsv41_output_projection as kernel
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    monkeypatch.setattr(kernel, "get_tpu_version", lambda: 7)
    # No torch_tpu on a CPU host: report the TPU7x device the kernel targets.
    monkeypatch.setattr(pallas_export, "get_torch_tpu_device_kind", lambda: "TPU7x")
    # 321 / 385 / 415 run a full tile then an aliased shorter one.
    for tokens in (6, 288, 321, 384, 385, 415, 416, 1024):
        args = (
            jax.ShapeDtypeStruct((tokens, H, HD), jnp.bfloat16),
            jax.ShapeDtypeStruct((D, G * R), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((G * R,), jnp.float32),
            jax.ShapeDtypeStruct((G * R, N), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((N,), jnp.float32),
        )
        with tpu_info_for_cpu_export():
            exported = jax.export.export(
                jax.jit(oproj.fused_any_tokens), platforms=("tpu",)
            )(*args)
        assert exported.out_avals[0].shape == (tokens, N)


def test_hook_specs_only_with_env(monkeypatch):
    from sglang.srt.plugins.hook_registry import HookType
    from sglang_tpu.models.deepseek_v41 import hooks as dsv41_hooks

    monkeypatch.delenv("SGLANG_TPU_DSV41_FUSED_OPROJ", raising=False)
    assert dsv41_hooks.fused_oproj_hook_specs() == []
    off = dsv41_hooks.hook_specs()
    monkeypatch.setenv("SGLANG_TPU_DSV41_FUSED_OPROJ", "1")
    on = dsv41_hooks.hook_specs()
    assert on[: len(off)] == off, "the default hooks keep their order"
    mqa = "sglang.srt.models.deepseek_v4.MQALayer"
    assert on[len(off) :] == [
        (
            "sglang.srt.model_loader.loader.DefaultModelLoader.postprocess_weights",
            dsv41_hooks.prepare_tpu_dsv41_fused_oproj,
            HookType.AFTER,
        ),
        (
            f"{mqa}._project_wo_a",
            dsv41_hooks.tpu_dsv41_fused_oproj_wo_a,
            HookType.AROUND,
        ),
        (
            f"{mqa}._project_wo_b",
            dsv41_hooks.tpu_dsv41_fused_oproj_wo_b,
            HookType.AROUND,
        ),
    ]


def test_seam_in_core_source():
    """The attention forward hands _project_wo_a's result straight to _project_wo_b,
    which the PreWoA pass-through relies on (read from source: deepseek_v4 does not
    import in the minimal env)."""
    import os
    import re

    path = os.path.join(
        os.path.dirname(oproj.__file__),
        *[".."] * 3,
        "python/sglang/srt/models/deepseek_v4.py",
    )
    src = open(path).read()
    assert "class MQALayer(" in src
    assert (
        "def _project_wo_a(self, o, *, positions, forward_batch, inv_rope, wo_a_quant_freqs):"
        in src
    )
    assert "def _project_wo_b(self, o, defer_all_reduce: bool = False):" in src
    assert re.search(
        r"o = self\._project_wo_a\(\s*o,\s*positions=positions,\s*forward_batch=forward_batch,"
        r"\s*inv_rope=inv_rope,\s*wo_a_quant_freqs=wo_a_quant_freqs,\s*\)\s*"
        r"return self\._project_wo_b\(o, defer_all_reduce\)",
        src,
    ), "the wo_a -> wo_b hand-off changed"
    # The other _project_wo_b caller is the mega-attention path, which
    # unmet_layer_requirements excludes.
    assert (
        src.count("self._project_wo_a(") == 1 and src.count("self._project_wo_b(") == 2
    )
    mega = re.search(
        r"if self\.use_mega_attention:\s*o = self\._forward_mega\([^)]*\)\s*return self\._project_wo_b\(o, defer_all_reduce\)",
        src,
    )
    assert mega, "the mega-attention caller of _project_wo_b changed"


def test_register_op_passes_annotation_objects(monkeypatch):
    """torch_tpu's _verify_signature rejects string annotations, which this module's
    ``from __future__ import annotations`` would otherwise produce (server boot failed
    with "Invalid signature for JAX function")."""
    import inspect
    import sys
    import types

    jax = pytest.importorskip("jax")
    captured = {}

    class _Op:
        def register_fake(self, fn):
            captured["fake"] = fn

    def fake_export(name, function, **kwargs):
        captured["name"], captured["fn"] = name, function
        return _Op()

    stub = types.ModuleType("sglang_tpu.pallas_jax_op")
    stub.cpu_export_jax_op = fake_export
    monkeypatch.setitem(sys.modules, "sglang_tpu.pallas_jax_op", stub)
    monkeypatch.setattr(oproj, "_OP", None)
    oproj.register_op()
    sig = inspect.signature(captured["fn"], follow_wrapped=False)
    assert list(sig.parameters) == ["x", "wa", "sa", "wb", "sb"]
    assert all(p.annotation is jax.Array for p in sig.parameters.values())
    assert sig.return_annotation is jax.Array
    x = torch.empty(12, H, HD, dtype=torch.bfloat16)
    wb = torch.empty(G * R, N, dtype=torch.float8_e4m3fn)
    assert captured["fake"](x, None, None, wb, None).shape == (12, N)


def test_min_tokens_gate(routed, monkeypatch):
    """SGLANG_TPU_DSV41_FUSED_OPROJ_MIN_TOKENS: smaller forwards keep the default
    path, so the BF16 wo_a stays; larger ones run the kernel."""
    from sglang_tpu.envs import envs as tpu_envs

    monkeypatch.setattr(
        tpu_envs.SGLANG_TPU_DSV41_FUSED_OPROJ_MIN_TOKENS, "get", lambda: 416
    )
    assert oproj.min_tokens() == 416
    attn = _fake_layer()
    oproj.prepare_layer(attn, keep_bf16=oproj.min_tokens() > 0)
    assert attn.wo_a.weight.numel() == G * R * D  # kept for the small forwards

    seen = []

    def original_a(attn_, o, **kwargs):
        seen.append("a")
        return o

    def original_b(attn_, o, defer_all_reduce=False):
        seen.append("b")
        return o

    small = torch.randn(48, H, HD).to(torch.bfloat16)
    pre = oproj.project_wo_a(
        original_a,
        attn,
        small,
        positions=None,
        forward_batch=None,
        inv_rope=None,
        wo_a_quant_freqs=None,
    )
    assert not isinstance(pre, oproj.PreWoA)
    oproj.project_wo_b(original_b, attn, pre, False)
    assert seen == ["a", "b"] and routed == []

    big = torch.randn(1024, H, HD).to(torch.bfloat16)
    out = _project(attn, big)
    assert routed == [1024] and out.shape == (1024, N)


def test_engagement_logged_once_per_bucket(routed, monkeypatch):
    seen = []
    monkeypatch.setattr(oproj, "_LOGGED", set())
    monkeypatch.setattr(
        oproj,
        "_log_engagement",
        lambda rows, used, path, _orig=oproj._log_engagement: (
            seen.append((rows, used, path)),
            _orig(rows, used, path),
        ),
    )
    attn = _fake_layer()
    oproj.prepare_layer(attn)
    for t in (48, 48, 384):
        _project(attn, torch.randn(t, H, HD).to(torch.bfloat16))
    assert (48, True, "kernel") in seen and (384, True, "kernel") in seen
    assert oproj._LOGGED == {(48, "kernel"), (384, "kernel")}


def test_kernel_parameters_own_their_storage():
    """No view-shaped Parameters reach the compiled program (each costs ~19 us
    per call on TPU): sb used to be a view of wo_b's [N, 1] weight_scale."""
    attn = _fake_layer()
    oproj.prepare_layer(attn)
    for name in ("dsv41_oproj_wa", "dsv41_oproj_sa", "dsv41_oproj_sb"):
        p = getattr(attn, name).data
        assert p._base is None and p.is_contiguous(), name
    assert (
        attn.dsv41_oproj_sb.data.untyped_storage().data_ptr()
        != attn.wo_b.weight_scale.data.untyped_storage().data_ptr()
    )


def test_wo_b_scale_reshaped_only_on_a_host_copy():
    """Nothing but a copy to the host touches wo_b's [N, 1] scale; the reshape runs
    on that copy. On TPU under DEFER_AND_FUSE a device-side reshape + clone stayed a
    deferred as_strided that re-ran as an eager program every step."""
    attn = _fake_layer()
    ptr = attn.wo_b.weight_scale.data.untyped_storage().data_ptr()
    seen = []

    class Record(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if any(
                isinstance(a, torch.Tensor) and a.untyped_storage().data_ptr() == ptr
                for a in args
            ):
                seen.append(getattr(func, "__name__", str(func)))
            return func(*args, **(kwargs or {}))

    with Record():
        oproj.prepare_layer(attn)
    assert not {"reshape", "view", "flatten", "clone", "contiguous"} & set(seen), seen
    assert torch.equal(
        attn.dsv41_oproj_sb.data, attn.wo_b.weight_scale.data.reshape(-1).float()
    )


def test_engagement_logging_survives_dynamo_fullgraph(routed, monkeypatch):
    """A fresh bucket logs from Python; under Dynamo the logger call would break a
    fullgraph capture, so it is skipped there."""
    monkeypatch.setattr(oproj, "_LOGGED", set())
    attn = _fake_layer()
    oproj.prepare_layer(attn)
    x = torch.randn(48, H, HD).to(torch.bfloat16)

    def run(x):
        return oproj.project_wo_b(lambda *a: None, attn, oproj.PreWoA(x), False)

    out = torch.compile(run, backend="eager", fullgraph=True)(x)
    assert out.shape == (48, N)


def test_prepare_model_selection(monkeypatch):
    from sglang_tpu.attention.pallas import util as pallas_util
    from sglang_tpu.envs import envs as tpu_envs

    class MQALayer(torch.nn.Module):
        _project_wo_b = None

    class DeepseekV4ForCausalLM(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = torch.nn.ModuleList(layers)

    def layer(layer_id, **over):
        src = _fake_layer(layer_id=layer_id)
        m = MQALayer()
        for k, v in vars(src).items():
            if not k.startswith("_"):
                setattr(m, k, v)
        m.wo_a, m.wo_b = src.wo_a, src.wo_b
        for k, v in over.items():
            setattr(m, k, v)
        return m

    monkeypatch.setattr(tpu_envs.SGLANG_TPU_DSV41_FUSED_OPROJ, "get", lambda: True)
    monkeypatch.setattr(
        tpu_envs.SGLANG_TPU_DSV41_FUSED_OPROJ_LAYERS, "get", lambda: "0-1"
    )
    monkeypatch.setattr(oproj, "register_op", lambda: None)

    monkeypatch.setattr(pallas_util, "get_tpu_version", lambda: 6)
    model = DeepseekV4ForCausalLM([layer(0)])
    assert oproj.prepare_model(model) == 0  # not v7: nothing touched
    assert model.layers[0].wo_a.weight.numel() == G * R * D

    monkeypatch.setattr(pallas_util, "get_tpu_version", lambda: 7)
    model = DeepseekV4ForCausalLM([layer(0), layer(1, attn_tp_size=2), layer(2)])
    assert oproj.prepare_model(model) == 1  # layer 1 ineligible, layer 2 filtered out
    l0, l1, l2 = model.layers
    assert (
        getattr(l0, oproj._ENABLED_ATTR)
        and "Online weight updates" in l0._derived_weight_cache_error
    )
    assert not hasattr(l1, "dsv41_oproj_wa") and not hasattr(l2, "dsv41_oproj_wa")
