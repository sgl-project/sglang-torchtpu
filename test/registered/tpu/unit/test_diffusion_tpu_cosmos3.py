# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's Cosmos3 patches (sglang_tpu.multimodal_gen.cosmos3)."""

import contextlib
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.cosmos3video import (
    Cosmos3VideoArchConfig,
    Cosmos3VideoConfig,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.cache.conditioning import ConditioningCache
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import cosmos3video as dit
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    cosmos3 as stage,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import cosmos3, worker

# Whole node, not chips=1: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS pin.
# Nightly: the stage-a unit suite is at its time budget.
register_tpu_ci(est_time=25, suite="nightly-tpu-4-v7", nightly=True)


@contextlib.contextmanager
def _installed():
    """The TPU paths on CPU, with every patched name restored afterwards."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (dit.Cosmos3OmniTransformer, "_compute_rope_position_ids"),
            (dit.Cosmos3OmniTransformer, "__call__"),
            (USPAttention, "forward_with_replicated_kv_prefix"),
            (stage.Cosmos3DenoisingStage, "_predict_noise_cfg_batched"),
            (stage.Cosmos3DenoisingStage, "_maybe_enable_torch_compile"),
            (dit.Cosmos3LanguageModel, "forward"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(cosmos3, "_DEVICE_TYPES", ("cpu",)))
        cosmos3.install()
        cosmos3.install()  # idempotent
        yield


def _model():
    torch.manual_seed(0)
    arch = Cosmos3VideoArchConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=96,
        latent_channel=4,
        out_channels=4,
        mrope_section=(2, 3, 3),
        vocab_size=100,
    )
    model = dit.Cosmos3OmniTransformer(
        config=Cosmos3VideoConfig(arch_config=arch), hf_config={}
    )
    for param in model.parameters():
        param.data.normal_(0, 0.1)
    return model.float().eval()


def _inputs(text_len, padded_len=None, frames=2):
    """One branch's transformer kwargs; ``padded_len`` pads the text as the
    tokenizer stage does for CFG (masked rows past the real length)."""
    gen = torch.Generator().manual_seed(text_len)
    total = padded_len or text_len
    text_ids = torch.randint(1, 100, (1, total), generator=gen)
    text_mask = (torch.arange(total) < text_len).long()[None]
    return dict(
        hidden_states=torch.randn(1, 4, frames, 4, 6, generator=gen),
        encoder_hidden_states=None,
        timestep=torch.full((1,), 500.0),
        text_ids=text_ids,
        text_mask=text_mask,
        fps=24.0,
        cache_key="cond",
        max_text_seq_len=text_len,
    )


def _run(model, kwargs):
    model.reset_cache()
    with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
        return model(**kwargs)


class _TestBase(CustomTestCase):
    def setUp(self):
        self._previous_args = server_args_module._global_server_args
        set_global_server_args(_make_unit_server_args())
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            with patch.object(worker, "install_runtime_patches"):
                maybe_init_distributed_environment_and_model_parallel(
                    tp_size=1, sp_size=1
                )

    def tearDown(self):
        set_global_server_args(self._previous_args)


class TestCosmos3Transformer(_TestBase):
    def test_bucketed_prompt_matches_stock(self):
        model = _model()
        for text_len, padded_len, frames in ((5, None, 2), (37, 41, 2), (9, None, 1)):
            kwargs = _inputs(text_len, padded_len, frames)
            expected = _run(model, kwargs)
            with _installed():
                actual = _run(model, kwargs)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)

    def test_stock_path_call_after_bucketed_call_matches_stock(self):
        model = _model()
        control = _inputs(40)
        control["control_latents"] = torch.randn(
            1, 4, 1, 4, 6, generator=torch.Generator().manual_seed(1)
        )
        expected = [_run(model, control), _run(model, _inputs(40))]
        with _installed():
            _run(model, _inputs(5))
            # A control call runs the stock path, then a bucketed call follows.
            actual = [_run(model, control), _run(model, _inputs(40))]
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e, atol=2e-5, rtol=1e-4)

    def test_moved_module_matches_stock(self):
        model = _model()
        expected = _run(model, _inputs(40))
        with _installed():
            _run(model, _inputs(5))
            # A cross-device ``Module.to`` copies each module's buffers.
            model._apply(lambda t: t.clone())
            actual = _run(model, _inputs(40))
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)

    def test_ring_and_kv_gather_run_stock(self):
        model = _model()
        kwargs = _inputs(40)
        expected = _run(model, kwargs)
        attns = [layer.cross_attention.attn for layer in model.gen_layers]
        for mode, ring in (("kv_gather", 1), ("ulysses", 2)):
            with (
                self.subTest(mode=mode, ring=ring),
                _installed(),
                patch.object(cosmos3, "get_sp_world_size", return_value=2),
                patch.object(
                    cosmos3, "get_ring_parallel_world_size", return_value=ring
                ),
                contextlib.ExitStack() as stack,
            ):
                for attn in attns:
                    stack.enter_context(patch.object(attn, "sp_attention_mode", mode))
                actual = _run(model, kwargs)
                for attn in attns:
                    self.assertIsNone(getattr(attn, cosmos3._TEXT_KEYS, None))
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)

    def test_fallback_attention_matches_float32(self):
        heads, kv_heads, head_dim, real = 4, 2, 16, 37
        attn = SimpleNamespace(
            attn_impl=SimpleNamespace(softmax_scale=head_dim**-0.5),
            skip_sequence_parallel=True,
            _tpu_text_keys=torch.tensor([0, real], dtype=torch.int32),
        )
        gen = torch.Generator().manual_seed(0)
        q = 4 * torch.randn(1, 300, heads, head_dim, generator=gen)
        k_text, v_text, k_vis, v_vis = (
            torch.randn(1, n, kv_heads, head_dim, generator=gen)
            for n in (512, 512, 300, 300)
        )
        q, k_text, v_text, k_vis, v_vis = (
            t.bfloat16() for t in (q, k_text, v_text, k_vis, v_vis)
        )
        with _installed():
            out = cosmos3.prefix_attention(attn, q, k_text, v_text, k_vis, v_vis)
        self.assertEqual(out.dtype, torch.bfloat16)
        keys = torch.cat((k_text[:, :real], k_vis), 1)
        values = torch.cat((v_text[:, :real], v_vis), 1)
        upcast = lambda t: t.float().transpose(1, 2)  # noqa: E731
        expected = F.scaled_dot_product_attention(
            upcast(q),
            upcast(keys),
            upcast(values),
            scale=head_dim**-0.5,
            enable_gqa=True,
        ).transpose(1, 2)
        # float32 attention, rounded once to bfloat16.
        torch.testing.assert_close(out.float(), expected, atol=1e-5, rtol=2**-8)

    def test_host_positions_match_stock(self):
        model = _model()
        mask = (torch.arange(20) < 13).long()[None]
        expected = model._compute_rope_position_ids(mask, 3, 2, 4, 24.0, "cpu")
        with _installed():
            actual = model._compute_rope_position_ids(mask, 3, 2, 4, 24.0, "cpu")
        for a, e in zip(actual, expected, strict=True):
            self.assertTrue(torch.equal(a, e))

    def test_gen_layers_compile_once_across_prompt_lengths(self):
        model = _model()
        graphs = []

        def counting(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        with _installed():
            # An eager call first registers the text-key count on the layers.
            _run(model, _inputs(5))
            for i, layer in enumerate(model.gen_layers):
                model.gen_layers[i] = torch.compile(
                    layer, backend=counting, dynamic=False
                )
            _run(model, _inputs(5))
            first = len(graphs)
            for text_len in (40, 300, 511):
                _run(model, _inputs(text_len))
        self.assertGreater(first, 0)
        self.assertEqual(len(graphs), first)


def _views(tensors):
    """The tensors laid out unlike a new tensor (storage offset, strides)."""
    return [
        (tuple(t.shape), t.storage_offset(), t.stride())
        for t in tensors
        if torch.is_tensor(t)
        and (t.storage_offset() != 0 or t.stride() != torch.empty(t.shape).stride())
    ]


class TestCosmos3CompiledInputs(_TestBase):
    """A compiled program is guarded on its inputs' storage offsets (torch_tpu)
    and strides (Dynamo)."""

    def _compile(self, model):
        """The stage's GEN-layer compile; returns each graph's view inputs."""
        views = []

        def recording(gm, example_inputs):
            views.append(_views(example_inputs))
            return gm.forward

        # An eager call first builds the shared RoPE module: built in a traced
        # layer, it splits the graph and exposes the layer's own views.
        _run(model, _inputs(5))
        torch._dynamo.reset()
        platform = SimpleNamespace(
            is_npu=lambda: False,
            is_out_of_tree=lambda: True,
            prepare_spmd_region_wrapper=lambda: None,
        )
        with (
            patch.object(stage, "current_platform", platform),
            patch.object(
                stage,
                "build_torch_compile_kwargs",
                lambda mode, module: {"backend": recording},
            ),
        ):
            stage.Cosmos3DenoisingStage._maybe_enable_torch_compile(
                None, model, SimpleNamespace(enable_torch_compile=True)
            )
        return views

    def test_ulysses_shards_of_control_stream_enter_fresh(self):
        control = _inputs(40)
        control["control_latents"] = torch.randn(
            1, 4, 1, 4, 6, generator=torch.Generator().manual_seed(1)
        )
        graphs = []
        # 2 Ulysses ranks: the first GEN layer gets rank r's 9 of the 18 control
        # and video tokens. The output gather is faked by repetition.
        for rank in (0, 1):
            model = _model()
            with (
                self.subTest(rank=rank),
                _installed(),
                patch.object(model, "sp_size", 2),
                patch.object(model, "sp_rank", rank),
                patch.object(
                    dit,
                    "sequence_model_parallel_all_gather",
                    lambda t, dim: torch.cat((t, t), dim),
                ),
            ):
                views = self._compile(model)
                for step in (900.0, 500.0):
                    control["timestep"] = torch.full((1,), step)
                    _run(model, control)
                graphs.append(len(views))
                self.assertGreater(len(views), 0)
                self.assertEqual([v for v in views if v], [])
        # Every rank compiles the same programs.
        self.assertEqual(len(set(graphs)), 1, graphs)

    def test_text_kv_enter_fresh_on_cache_miss_and_hit(self):
        model = _model()
        cache = ConditioningCache(1 << 30)
        with _installed():
            views = self._compile(model)
            # One request per call: the first misses the cache, the second hits.
            with cache.scope():
                _run(model, _inputs(40))
                # Copied once per prompt, not by every GEN layer call.
                for kv in model.cached_kv["cond"]:
                    self.assertEqual(_views(kv), [])
                _run(model, _inputs(40))
        self.assertEqual((cache.misses, cache.hits), (1, 1))
        self.assertGreater(len(views), 0)
        self.assertEqual([v for v in views if v], [])


class TestCosmos3Stage(CustomTestCase):
    def test_equal_length_text_cfg_keeps_batch_one(self):
        calls = []

        class Stage(stage.Cosmos3DenoisingStage):
            def __init__(self):
                pass

            def _run_transformer(self, *, latents, text_ids, **kwargs):
                calls.append((latents.shape[0], kwargs["max_text_seq_len"]))
                return latents * text_ids.float().mean(1).view(-1, 1, 1, 1, 1)

        gen = torch.Generator().manual_seed(0)
        cond_ids, uncond_ids = torch.randint(1, 100, (2, 1, 7), generator=gen)
        mask = torch.ones(1, 7, dtype=torch.long)
        kwargs = dict(
            latents=torch.randn(1, 4, 2, 4, 6, generator=gen),
            timestep=torch.full((1,), 500.0),
            cond_text_ids=cond_ids,
            cond_text_mask=mask,
            uncond_text_ids=uncond_ids,
            uncond_text_mask=mask,
            video_shape=(2, 4, 6),
            fps=24.0,
            guidance_scale=3.0,
            max_text_seq_len=7,
            current_timestep=0,
        )
        expected = Stage()._predict_noise_cfg_batched(**kwargs)
        self.assertEqual(calls, [(2, 7)])
        calls.clear()
        with _installed():
            actual = Stage()._predict_noise_cfg_batched(**kwargs)
        self.assertEqual(calls, [(1, 7), (1, 7)])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class TestFlowUniPCOneDtypeEinsum(CustomTestCase):
    def test_bf16_steps_match_with_an_einsum_that_takes_one_dtype(self):
        # Cosmos3 steps bfloat16 latents with float solver weights: CPU einsum
        # promotes them, torch_tpu's takes one dtype. The steps must not change.
        def run():
            scheduler = FlowUniPCMultistepScheduler(solver_order=2, predict_x0=True)
            scheduler.set_timesteps(35, device="cpu")
            gen = torch.Generator().manual_seed(0)
            x = torch.randn(1, 4, 2, 4, 6, generator=gen).bfloat16()
            for t in scheduler.timesteps[:4]:
                out = torch.randn(1, 4, 2, 4, 6, generator=gen).bfloat16()
                x = scheduler.step(out, t, x, return_dict=False)[0]
            return x

        expected = run()
        stock_einsum = torch.einsum

        def one_dtype(equation, *operands):
            self.assertEqual(len({operand.dtype for operand in operands}), 1)
            return stock_einsum(equation, *operands)

        with patch.object(torch, "einsum", one_dtype):
            actual = run()
        self.assertTrue(torch.equal(actual, expected))


def _has_tpu() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        return torch.tpu.device_count() > 0
    except Exception:
        return False


@unittest.skipUnless(_has_tpu(), "requires a TPU")
class TestCosmos3OnTpu(CustomTestCase):
    def test_prefix_kernel_steps_match_reference(self):
        from sglang_tpu.multimodal_gen import tpu_splash_attn

        heads, kv_heads, head_dim = 8, 1, 128
        impl = tpu_splash_attn.TpuSplashAttentionImpl(
            num_heads=heads,
            head_size=head_dim,
            causal=False,
            softmax_scale=head_dim**-0.5,
            num_kv_heads=kv_heads,
        )
        real = 140
        attn = SimpleNamespace(
            attn_impl=impl,
            skip_sequence_parallel=True,
            sp_attention_mode="ulysses",
            _tpu_text_keys=torch.tensor([0, real], dtype=torch.int32).to("tpu"),
        )
        gen = torch.Generator().manual_seed(0)
        q = torch.randn(1, 2048, heads, head_dim, generator=gen)
        k_text, v_text, k_vis, v_vis = (
            torch.randn(1, n, kv_heads, head_dim, generator=gen)
            for n in (512, 512, 2048, 2048)
        )
        to_tpu = lambda t: t.bfloat16().to("tpu")  # noqa: E731
        out = cosmos3.prefix_attention(
            attn, *map(to_tpu, (q, k_text, v_text, k_vis, v_vis))
        )
        keys = torch.cat((k_text[:, :real], k_vis), 1)
        values = torch.cat((v_text[:, :real], v_vis), 1)
        rounded = lambda t: t.bfloat16().float().transpose(1, 2)  # noqa: E731
        expected = F.scaled_dot_product_attention(
            rounded(q),
            rounded(keys),
            rounded(values),
            scale=head_dim**-0.5,
            enable_gqa=True,
        ).transpose(1, 2)
        torch.testing.assert_close(out.float().cpu(), expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
