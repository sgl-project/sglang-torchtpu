# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's Ideogram 4 patches (sglang_tpu.multimodal_gen.ideogram)."""

import contextlib
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.ideogram import (
    Ideogram4DiTArchConfig,
    Ideogram4DiTConfig,
)
from sglang.multimodal_gen.configs.models.encoders.ideogram import (
    Ideogram4TextEncoderConfig,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.quantization import weight_only_fp8 as fp8
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import ideogram as dit
from sglang.multimodal_gen.runtime.models.encoders import ideogram as encoder
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    ideogram as stage,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import ideogram, tpu_splash_attn, worker

# Whole node, not chips=1: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS pin.
# Nightly: the stage-a unit suite is at its time budget.
register_tpu_ci(est_time=40, suite="nightly-tpu-4-v7", nightly=True)

IMAGE_OFFSET = 65536


@contextlib.contextmanager
def _installed():
    """The TPU paths on CPU, with every patched name restored afterwards."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (dit.Ideogram4Attention, "forward"),
            (dit.Ideogram4Transformer2DModel, "forward"),
            (dit.Ideogram4Transformer2DModel, "__call__"),
            (encoder.IdeogramQwen3VLTextEncoder, "encode_ideogram_features"),
            (dit.Ideogram4TransformerBlock, "forward"),
            (stage.LogitNormalSchedule, "__call__"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(ideogram, "_DEVICE_TYPES", ("cpu",)))
        stack.enter_context(patch.dict(ideogram._CACHE, clear=True))
        ideogram.install()
        ideogram.install()  # idempotent
        yield


def _model():
    torch.manual_seed(0)
    arch = Ideogram4DiTArchConfig(
        adaln_dim=16,
        attention_head_dim=32,
        in_channels=8,
        intermediate_size=96,
        llm_features_dim=40,
        mrope_section=(6, 5, 5),
        num_attention_heads=3,
        num_layers=2,
    )
    model = dit.Ideogram4Transformer2DModel(
        config=Ideogram4DiTConfig(arch_config=arch), hf_config={}
    )
    for module in model.modules():
        if isinstance(module, fp8.WeightOnlyFP8Linear):
            weight = torch.randn(module.weight.shape) * 0.2
            module.weight.data = weight.to(fp8.FP8_WEIGHT_DTYPE)
            module.weight_scale.data = torch.rand(module.weight_scale.shape) + 0.5
            if module.bias is not None:
                module.bias.data = torch.randn(module.bias.shape) * 0.05
    for name, param in model.named_parameters():
        if param.dtype != fp8.FP8_WEIGHT_DTYPE and "weight_scale" not in name:
            if "linear" not in name and param.dim() == 1:
                param.data = 1.0 + 0.1 * torch.randn(param.shape)
    model.embed_image_indicator.weight.data.normal_(0, 0.1)
    return model.eval()


def _inputs(text_len, grid=(4, 6), seed=0):
    """One request's DiT kwargs, as Ideogram4TextEncodingStage builds them."""
    gen = torch.Generator().manual_seed(seed + text_len)
    gh, gw = grid
    image = gh * gw
    total = text_len + image
    h = torch.arange(gh).view(-1, 1).expand(gh, gw).reshape(-1)
    w = torch.arange(gw).view(1, -1).expand(gh, gw).reshape(-1)
    text_pos = torch.arange(text_len)[:, None].expand(text_len, 3)
    image_pos = torch.stack([torch.zeros_like(h), h, w], 1) + IMAGE_OFFSET
    indicator = torch.cat(
        (
            torch.full((text_len,), dit.LLM_TOKEN_INDICATOR),
            torch.full((image,), dit.OUTPUT_IMAGE_INDICATOR),
        )
    )
    llm = torch.zeros(1, total, 40)
    llm[0, :text_len] = torch.randn(text_len, 40, generator=gen)
    latents = torch.randn(1, image, 8, generator=gen)
    return dict(
        llm_features=llm,
        x=torch.cat((torch.zeros(1, text_len, 8), latents), 1),
        t=torch.full((1,), 0.4),
        position_ids=torch.cat((text_pos, image_pos))[None],
        segment_ids=torch.ones(1, total, dtype=torch.long),
        indicator=indicator[None],
    )


def _run(model, kwargs):
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


class TestIdeogramDiT(_TestBase):
    def test_padded_rows_match_stock(self):
        for text_len in (5, 37):
            model = _model()
            kwargs = _inputs(text_len)
            expected = _run(model, kwargs)
            with _installed():
                actual = _run(model, kwargs)
            self.assertEqual(actual.shape, expected.shape)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)

    def test_unconditional_branch_matches_stock(self):
        # The unconditional transformer sees only the image rows.
        model = _model()
        full = _inputs(9)
        kwargs = {k: v[:, 9:] if k != "t" else v for k, v in full.items()}
        kwargs["llm_features"] = torch.zeros_like(kwargs["llm_features"])
        expected = _run(model, kwargs)
        with _installed():
            actual = _run(model, kwargs)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)

    def test_fp8_weights_dequantize_once_bit_identically(self):
        model = _model()
        linear = model.layers[0].attention.qkv
        x = torch.randn(1, 7, linear.in_features, dtype=torch.bfloat16)
        before = linear(x)
        with patch.object(ideogram, "_DEVICE_TYPES", ("cpu",)):
            ideogram._dequantize_fp8_weights(model, torch.bfloat16)
        self.assertEqual(linear.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(linear(x), before))

    def test_whole_dit_compiles_once_across_prompt_lengths_and_steps(self):
        model = _model()
        graphs = []

        offsets = set()

        def counting(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        def guarded(gm, example_inputs):
            # torch_tpu's backend guards every tensor input's storage offset.
            offsets.update(
                t.storage_offset()
                for t in example_inputs
                if isinstance(t, torch.Tensor)
            )
            return counting(gm, example_inputs)

        with _installed():
            # An eager call first: the FP8 weights dequantize at the first call.
            _run(model, _inputs(5))
            model.compile(backend=guarded, dynamic=False)
            schedule = torch.linspace(1.0, 0.0, 8)
            for step, text_len in enumerate((5, 40, 300, 5)):
                kwargs = _inputs(text_len)
                # The stage's timestep: a view at the step's offset into the schedule.
                kwargs["t"] = schedule[step + 1].expand(1)
                _run(model, kwargs)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(offsets, {0})

    def test_local_rows_are_a_contiguous_copy_of_this_ranks_rows(self):
        for batch in (1, 2):
            t = torch.randn(batch, 700, 1, 5)
            padded = F.pad(t, (0, 0, 0, 0, 0, 324))
            for rank in range(2):
                local = ideogram._local(t, 1024, 2, rank)
                self.assertTrue(
                    torch.equal(local, padded[:, rank * 512 : (rank + 1) * 512])
                )
                storage = local.untyped_storage().nbytes()
                self.assertEqual(storage, local.numel() * local.element_size())
                # Rank 0's share is full: its strides must not depend on the
                # prompt's row count (compiled programs guard strides).
                self.assertEqual(local.stride(), torch.empty(local.shape).stride())

    def test_mixed_prompt_lengths_are_rejected(self):
        model = _model()
        kwargs = _inputs(5)
        kwargs["segment_ids"][0, 0] = -1
        with _installed(), self.assertRaises(NotImplementedError):
            _run(model, kwargs)


class TestIdeogramSchedule(CustomTestCase):
    def test_host_schedule_matches_stock(self):
        schedule = stage.get_schedule_for_resolution((1024, 1024), 0.0, 1.75)
        steps = stage.make_step_intervals(20)
        expected = schedule(steps)
        with _installed():
            actual = schedule(steps)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))


def _text_encoder():
    torch.manual_seed(0)
    config = Ideogram4TextEncoderConfig()
    config.arch_config.text_config = dict(
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=300,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_parameters=dict(
            rope_type="default",
            rope_theta=5000000,
            mrope_section=[4, 2, 2],
            mrope_interleaved=True,
        ),
    )
    config.arch_config.ideogram_fp8_weight_only = False
    model = encoder.IdeogramQwen3VLTextEncoder(config)
    for param in model.parameters():
        param.data.normal_(0, 0.1)
    model._activation_layers = (0, 1, 2)
    return model.float().eval()


class TestIdeogramTextEncoder(_TestBase):
    def test_padded_prompt_rows_match_stock(self):
        model = _text_encoder()
        for text_len in (6, 21):
            kwargs = _inputs(text_len)
            gen = torch.Generator().manual_seed(text_len)
            tokens = torch.zeros(1, kwargs["indicator"].shape[1], dtype=torch.long)
            tokens[0, :text_len] = torch.randint(1, 300, (text_len,), generator=gen)
            args = (
                tokens,
                kwargs["position_ids"],
                kwargs["indicator"],
                dit.LLM_TOKEN_INDICATOR,
            )
            with torch.no_grad():
                expected = model.encode_ideogram_features(*args)
                with _installed():
                    actual = model.encode_ideogram_features(*args)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _has_tpu() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        return torch.tpu.device_count() > 0
    except Exception:
        return False


def _identity_all_to_all(t, head_dim):
    return t


class TestIdeogramAttention(_TestBase):
    def test_short_rows_run_the_float32_reference(self):
        # 512 rows are below the kernel step's shapes: its float32 reference
        # runs over the real keys, as tpu_attention runs float32 SDPA.
        impl = tpu_splash_attn.TpuSplashAttentionImpl(
            num_heads=3, head_size=64, causal=False, softmax_scale=0.125
        )
        impl.use_splash = True
        attn = SimpleNamespace(attn_impl=impl, sp_attention_mode="ulysses")
        gen = torch.Generator().manual_seed(0)
        q, k, v = (torch.randn(1, 512, 3, 64, generator=gen) for _ in range(3))
        k[:, 300:], v[:, 300:] = float("nan"), float("nan")
        keys = torch.tensor([0, 300], dtype=torch.int32)
        with patch.object(ideogram, "_DEVICE_TYPES", ("cpu",)):
            out = ideogram._attention(attn, *(t.bfloat16() for t in (q, k, v)), keys)
        rounded = lambda t: t.bfloat16().float().transpose(1, 2)  # noqa: E731
        expected = F.scaled_dot_product_attention(
            rounded(q), rounded(k[:, :300]), rounded(v[:, :300]), scale=0.125
        ).transpose(1, 2)
        # Equal to the float32 result rounded once to bf16.
        torch.testing.assert_close(out.float(), expected, atol=1e-6, rtol=2**-8)

    def test_auto_kv_gather_mode_runs_ulysses(self):
        gen = torch.Generator().manual_seed(0)
        q, k, v = (torch.randn(1, 8, 3, 16, generator=gen) for _ in range(3))
        keys = torch.tensor([0, 6], dtype=torch.int32)
        impl = SimpleNamespace(softmax_scale=0.25)
        attn = SimpleNamespace(attn_impl=impl, sp_attention_mode="ulysses")
        expected = ideogram._attention(attn, q, k, v, keys)
        attn.sp_attention_mode = "kv_gather"
        with (
            patch.object(ideogram, "get_sp_world_size", return_value=2),
            patch.object(ideogram, "_usp_input_all_to_all", _identity_all_to_all),
            patch.object(ideogram, "_usp_output_all_to_all", _identity_all_to_all),
        ):
            attn.sp_attention_mode_is_auto = True
            actual = ideogram._attention(attn, q, k, v, keys)
            attn.sp_attention_mode_is_auto = False
            with self.assertRaises(NotImplementedError):
                ideogram._attention(attn, q, k, v, keys)
        torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(_has_tpu(), "requires a TPU")
    def test_padded_attention_matches_reference(self):
        heads, head_dim, rows, real = 18, 256, 4608, 4130
        impl = tpu_splash_attn.TpuSplashAttentionImpl(
            num_heads=heads,
            head_size=head_dim,
            causal=False,
            softmax_scale=head_dim**-0.5,
        )
        attn = SimpleNamespace(attn_impl=impl, sp_attention_mode="ulysses")
        gen = torch.Generator().manual_seed(0)
        q, k, v = (
            torch.randn(1, rows, heads, head_dim, generator=gen) for _ in range(3)
        )
        # Padded keys hold NaN: they must not be read.
        k[:, real:], v[:, real:] = float("nan"), float("nan")
        to_tpu = lambda t: t.bfloat16().to("tpu")  # noqa: E731
        keys = torch.tensor([0, real], dtype=torch.int32).to("tpu")
        out = ideogram._attention(attn, *map(to_tpu, (q, k, v)), keys).float().cpu()
        rounded = lambda t: t.bfloat16().float().transpose(1, 2)  # noqa: E731
        expected = F.scaled_dot_product_attention(
            rounded(q), rounded(k[:, :real]), rounded(v[:, :real]), scale=head_dim**-0.5
        ).transpose(1, 2)
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
