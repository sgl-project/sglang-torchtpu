# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's Kandinsky 6 patches (sglang_tpu.multimodal_gen.kandinsky6)."""

import contextlib
import inspect
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.kandinsky6 import (
    Kandinsky6VideoAudioConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs import kandinsky6 as config
from sglang.multimodal_gen.configs.pipeline_configs import qwen_image as qwen_config
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import kandinsky6 as dit
from sglang.multimodal_gen.runtime.models.encoders import qwen_vl
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6 import (
    denoising as stage,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import kandinsky6, worker

# Nightly: the stage-a unit suite is at its time budget.
register_tpu_ci(est_time=60, suite="nightly-tpu-4-v7", nightly=True)


@contextlib.contextmanager
def _installed(device_types=("cpu",)):
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (dit.Kandinsky6Transformer3DModel, "forward"),
            (dit.Kandinsky6Transformer3DModel, "_compile_regions_spmd_safe"),
            (dit.Kandinsky6TransformerEncoderBlock, "forward"),
            (dit.Kandinsky6FusedTransformerDecoderBlock, "forward"),
            (qwen_vl.QwenVLModelBase, "_get_text_rope_index"),
            (qwen_config, "_extract_masked_hidden"),
            (stage.Kandinsky6DenoisingStage, "_text_rope_pos"),
            (dit, "_apply_rotary"),
            (config.Kandinsky6TI2VAPipelineConfig, "tokenize_prompt"),
        ):
            # A static lookup keeps a staticmethod a staticmethod when restored.
            stack.enter_context(
                patch.object(owner, name, inspect.getattr_static(owner, name))
            )
        stack.enter_context(patch.object(kandinsky6, "_DEVICE_TYPES", device_types))
        kandinsky6.install()
        kandinsky6.install()  # idempotent
        yield


def _tiny_model(visual_token_types: int = 0) -> dit.Kandinsky6Transformer3DModel:
    config = Kandinsky6VideoAudioConfig()
    config.update_model_arch(
        dict(
            model_dim=256,
            time_dim=32,
            ff_dim=256,
            axes_dims=(32, 48, 48),
            in_visual_dim=4,
            out_visual_dim=4,
            in_text_dim=16,
            in_text_dim2=8,
            in_audio_dim=4,
            out_audio_dim=4,
            num_text_blocks=2,
            num_visual_blocks=2,
            patch_size=(1, 2, 2),
            visual_cond=False,
            visual_token_type_num_embeddings=visual_token_types,
            is_multimodal=True,
            model_dim_a=128,
            time_dim_a=32,
            ff_dim_a=128,
            axes_dims_a=(32, 48, 48),
            ca_rope=True,
            cross_gates=True,
            fix_modulation=True,
        )
    )
    model = dit.Kandinsky6Transformer3DModel(config, {}).eval()
    for module in model.modules():
        if isinstance(module, CustomOp):
            # The forwards TPU dispatches to, on any test host.
            module._forward_method = module.forward_native
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen) * 0.05)
    return model


def _inputs(num_text: int, frames: int = 2, seed: int = 1, token_types=False):
    gen = torch.Generator().manual_seed(seed)
    inputs = dict(
        hidden_states=torch.randn(1, frames, 4, 6, 4, generator=gen),
        hidden_states_audio=torch.randn(1, 5, 4, generator=gen),
        encoder_hidden_states=torch.randn(1, num_text, 16, generator=gen),
        pooled_projections=torch.randn(1, 8, generator=gen),
        timestep=torch.tensor([700.0]),
        visual_rope_pos=[torch.arange(n) for n in (frames, 2, 3)],
        text_rope_pos=torch.arange(num_text),
    )
    if token_types:
        types = torch.zeros(1, frames, dtype=torch.long)
        types[:, -1] = 1
        inputs["visual_token_type_ids"] = types
    return inputs


class _DistributedTestCase(CustomTestCase):
    def setUp(self):
        self._previous_args = server_args_module._global_server_args
        server_args_module.set_global_server_args(_make_unit_server_args())
        if not parallel_state.model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            with patch.object(worker, "install_runtime_patches"):
                parallel_state.maybe_init_distributed_environment_and_model_parallel(
                    tp_size=1, sp_size=1
                )

    def tearDown(self):
        server_args_module.set_global_server_args(self._previous_args)


class TestPaddedPrompt(_DistributedTestCase):
    def test_matches_stock_for_every_prompt_length(self):
        for token_types in (False, True):
            model = _tiny_model(visual_token_types=2 if token_types else 0)
            for num_text in (1, 37, 300, 512):
                inputs = _inputs(num_text, token_types=token_types)
                with torch.no_grad(), set_forward_context(0, None):
                    want = kandinsky6._STOCK_FORWARD(model, **inputs)
                    with _installed():
                        got = model(**inputs)
                for actual, expected in zip(got, want, strict=True):
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)

    def test_padded_rows_never_reach_real_rows(self):
        model = _tiny_model()
        inputs = _inputs(9)
        stock_pad = torch.nn.functional.pad
        with torch.no_grad(), set_forward_context(0, None), _installed():
            want = model(**inputs)

            def pad_with_noise(x, pad, *args, **kwargs):
                out = stock_pad(x, pad, *args, **kwargs)
                out[:, x.shape[1] :] = 1e4 * torch.randn_like(out[:, x.shape[1] :])
                return out

            with patch.object(kandinsky6.F, "pad", pad_with_noise):
                got = model(**inputs)
        for actual, expected in zip(got, want, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_blocks_take_one_shape_at_offset_zero(self):
        model = _tiny_model()
        seen = []

        def recorder(container):
            def record(module, args, kwargs):
                seen.append(
                    (
                        container,
                        tuple(
                            (tuple(t.shape), t.storage_offset(), t.is_contiguous())
                            for t in torch.utils._pytree.tree_leaves((args, kwargs))
                            if isinstance(t, torch.Tensor)
                        ),
                    )
                )

            return record

        for container in (
            "video_text_transformer_blocks",
            "audio_text_transformer_blocks",
            "visual_transformer_blocks",
        ):
            for block in getattr(model, container):
                block.register_forward_pre_hook(recorder(container), with_kwargs=True)

        stock_shard_seq, stock_shard_like = (
            kandinsky6.shard_seq,
            kandinsky6.shard_like,
        )

        def offset_view(x):
            # A rank's shard: a view into a larger buffer.
            return torch.cat([x[:, :1], x], dim=1)[:, 1:]

        def shard_seq(x, *args, **kwargs):
            out, shard = stock_shard_seq(x, *args, **kwargs)
            return offset_view(out), shard

        def shard_like(x, *args, **kwargs):
            return offset_view(stock_shard_like(x, *args, **kwargs))

        with (
            torch.no_grad(),
            set_forward_context(0, None),
            _installed(),
            patch.object(kandinsky6, "shard_seq", shard_seq),
            patch.object(kandinsky6, "shard_like", shard_like),
        ):
            for num_text in (3, 40, 200):
                model(**_inputs(num_text))
        by_container = {}
        for container, tensors in seen:
            by_container.setdefault(container, set()).add(tensors)
        self.assertEqual(len(by_container), 3)
        # One input signature per block stack, for every prompt length.
        for container, signatures in by_container.items():
            self.assertEqual(len(signatures), 1, container)
            for shape, offset, contiguous in next(iter(signatures)):
                self.assertEqual((offset, contiguous), (0, True), (container, shape))

    def test_other_devices_keep_stock(self):
        model = _tiny_model()
        with (
            torch.no_grad(),
            set_forward_context(0, None),
            _installed(device_types=("tpu",)),
            patch.object(
                kandinsky6, "_STOCK_FORWARD", wraps=kandinsky6._STOCK_FORWARD
            ) as stock,
        ):
            self.assertTrue(dit.Kandinsky6Transformer3DModel._compile_regions_spmd_safe)
            model(**_inputs(7))
        stock.assert_called_once()

    def test_text_conditioning_on_the_host(self):
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]])
        hidden = torch.randn(2, 6, 3)
        want_pos = kandinsky6._STOCK_TEXT_ROPE_INDEX(None, None, mask)
        want_rows = kandinsky6._STOCK_EXTRACT_MASKED_HIDDEN(hidden, mask)
        want_rope = kandinsky6._STOCK_TEXT_ROPE_POS(mask, "cpu")
        with _installed():
            stock_index = patch.object(
                kandinsky6,
                "_STOCK_TEXT_ROPE_INDEX",
                wraps=kandinsky6._STOCK_TEXT_ROPE_INDEX,
            )
            with stock_index as index:
                got_pos = qwen_vl.QwenVLModelBase._get_text_rope_index(
                    None, torch.zeros(2, 6, dtype=torch.long), mask
                )
            # The stock positions see only the (host) mask.
            self.assertIsNone(index.call_args.args[1])
            got_rows = qwen_config._extract_masked_hidden(hidden, mask)
            got_rope = stage.Kandinsky6DenoisingStage._text_rope_pos(mask, "cpu")
        for got, want in zip(got_pos, want_pos, strict=True):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        self.assertEqual(len(got_rows), len(want_rows))
        for got, want in zip(got_rows, want_rows, strict=True):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        torch.testing.assert_close(got_rope, want_rope, atol=0, rtol=0)

    def test_rope_pair_rotates_bitwise_as_stock(self):
        gen = torch.Generator().manual_seed(3)
        rope_1d = dit.Kandinsky6RoPE1D(128)
        rope_3d = dit.Kandinsky6RoPE3D((32, 48, 48))
        tables = (
            rope_1d(torch.arange(300)).unsqueeze(0),
            rope_3d((1, 3, 4, 5), [torch.arange(n) for n in (3, 4, 5)]).flatten(1, 3),
        )
        for rope in tables:
            x = torch.randn(1, rope.shape[1], 4, 128, generator=gen)
            for dtype in (torch.bfloat16, torch.float32):
                want = kandinsky6._STOCK_APPLY_ROTARY(x, rope, dtype)
                got = kandinsky6._apply_rotary(x, kandinsky6.rope_pair(rope), dtype)
                if dtype == torch.bfloat16:
                    torch.testing.assert_close(got, want, atol=0, rtol=0)
                else:
                    # float32 x @ R is not an exact product on every backend.
                    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)
        # Tables in the stock layout keep the stock rotation.
        x = torch.randn(1, 300, 4, 128, generator=gen)
        torch.testing.assert_close(
            kandinsky6._apply_rotary(x, tables[0], torch.bfloat16),
            kandinsky6._STOCK_APPLY_ROTARY(x, tables[0], torch.bfloat16),
            atol=0,
            rtol=0,
        )

    def test_reason1_tokens_pad_to_a_fixed_length(self):
        calls = []

        def tokenizer(text, padding, max_length, truncation=True, **kwargs):
            calls.append((padding, max_length))
            n = min(len(text[0].split()), max_length)
            n = max_length if padding == "max_length" else n
            real = min(len(text[0].split()), n)
            mask = torch.tensor([[1] * real + [0] * (n - real)])
            return {"input_ids": torch.arange(n)[None], "attention_mask": mask}

        cfg = config.Kandinsky6TI2VAPipelineConfig()
        reason1 = dict(max_length=641, padding=True, truncation=True)
        clip = dict(max_length=77, padding="max_length", truncation=True)
        with _installed():
            short = cfg.tokenize_prompt(["a " * 40], tokenizer, reason1)
            long = cfg.tokenize_prompt(["a " * 600], tokenizer, reason1)
            clip_out = cfg.tokenize_prompt(["a " * 10], tokenizer, clip)
        self.assertEqual(tuple(short["input_ids"].shape), (1, 512))
        self.assertEqual(int(short["attention_mask"].sum()), 40)
        # Longer prompts keep the stock padding (and the 641-token cap).
        self.assertEqual(tuple(long["input_ids"].shape), (1, 600))
        self.assertEqual(tuple(clip_out["input_ids"].shape), (1, 77))
        self.assertEqual(
            calls,
            [(True, 641), ("max_length", 512), (True, 641), ("max_length", 77)],
        )

    def test_text_rows(self):
        self.assertEqual(kandinsky6.text_rows(1), 512)
        self.assertEqual(kandinsky6.text_rows(512), 512)
        self.assertEqual(kandinsky6.text_rows(513), 1024)


if __name__ == "__main__":
    import unittest

    unittest.main()
