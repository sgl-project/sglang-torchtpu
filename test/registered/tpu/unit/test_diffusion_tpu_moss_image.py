# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's MOSS-Image patches (sglang_tpu.multimodal_gen.moss_image)."""

import contextlib
import os
import socket
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
import torch.multiprocessing as mp
from transformers import BatchEncoding, Qwen3_5TextConfig

from sglang.multimodal_gen.configs.models.dits.moss_image import (
    MOSSImageArchConfig,
    MOSSImageDiTConfig,
)
from sglang.multimodal_gen.configs.models.encoders.qwen3_5 import (
    Qwen3_5EncoderConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs import moss_image as config
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import moss_image as dit
from sglang.multimodal_gen.runtime.models.encoders.qwen3_5 import Qwen3_5Model
from sglang.multimodal_gen.runtime.pipelines import moss_image as pipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    moss_image as stages,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import moss_image, worker

# Whole node, not chips=1: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS pin.
register_tpu_ci(est_time=60, suite="nightly-tpu-4-v7", nightly=True)

_HEIGHT, _WIDTH = 3, 5  # 15 image rows
_TOL = dict(rtol=1e-5, atol=1e-5)


@contextlib.contextmanager
def _installed():
    """install() on CPU tensors with 32-row joint buckets (prompts of 5 to 17
    tokens share one) and 16-token prompt buckets; patches restored after."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (dit, "_attention"),
            (dit.MOSSImageTransformer, "prepare_conditioning"),
            (dit.MOSSImageTransformer, "forward"),
            (dit.MOSSImageTransformer, "_compile_conditions"),
            (dit.MOSSImageTransformer, "_compile_regions_spmd_safe"),
            (stages.MOSSImageDenoisingStage, "_before_denoising_loop"),
            (stages.MOSSImageBeforeDenoisingStage, "_encode_prompt"),
            (pipeline.MOSSImagePipeline, "create_pipeline_stages"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(moss_image, "_DEVICE_TYPES", ("tpu", "cpu")))
        stack.enter_context(patch.object(moss_image, "JOINT_ROW_ALIGNMENT", 32))
        stack.enter_context(patch.object(moss_image, "PROMPT_TOKEN_ALIGNMENT", 16))
        moss_image._rope_tables.cache_clear()
        moss_image.install()
        moss_image.install()  # idempotent
        yield


def _model():
    torch.manual_seed(0)
    arch = MOSSImageArchConfig(
        dim=256,
        ffn_dim=512,
        num_heads=2,
        num_layers=3,
        text_dim=64,
        freq_dim=32,
        latent_channels=16,
    )
    model = dit.MOSSImageTransformer(MOSSImageDiTConfig(arch_config=arch))
    for param in model.parameters():
        param.data.normal_(0, 0.05)
    return model.eval()


def _run(model, lengths, skip=(), batch_samples=True, step=None, tweak=None):
    """One DiT call on right-padded prompt states and their prefix mask."""
    gen = torch.Generator().manual_seed(sum(lengths))
    latents = torch.randn(len(lengths), 16, _HEIGHT, _WIDTH, generator=gen)
    mask = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
    states = torch.randn(len(lengths), max(lengths), 64, generator=gen)
    states = states * mask[..., None]
    timesteps = torch.tensor([1000.0, 630.0])
    with torch.no_grad(), set_forward_context(step or 0, None):
        prepared = model.prepare_conditioning(
            states, mask, _HEIGHT, _WIDTH, torch.float32, batch_samples=batch_samples
        )
        if step is not None and isinstance(prepared, moss_image.TpuConditioning):
            table = moss_image._time_embeddings(timesteps, 32, latents.device)
            prepared = msgspec.structs.replace(prepared, time_embeddings=table)
        if tweak is not None:
            tweak(prepared)
        return model(latents, timesteps[step or 0], states, mask, prepared, skip)


def _init_parallel_state(sp=1):
    set_global_server_args(_make_unit_server_args())
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        with patch.object(worker, "install_runtime_patches"):
            maybe_init_distributed_environment_and_model_parallel(
                tp_size=1, sp_size=sp, ulysses_degree=sp
            )


def _sp_worker(rank, port, out_dir, cases):
    """One Ulysses rank of an SP 2 run over gloo; rank 0 saves the outputs."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
    )
    _init_parallel_state(sp=2)
    model, layouts = _model(), []
    for layer in model.layers:
        layer.register_forward_pre_hook(
            lambda _, args: layouts.append(
                (args[0].shape, args[0].stride(), args[0].storage_offset())
            )
        )
    with _installed():
        for index, (lengths, skip) in enumerate(cases):
            out = _run(model, lengths, skip)
            if rank == 0:
                torch.save(out, os.path.join(out_dir, f"{index}.pt"))
    torch.save(layouts, os.path.join(out_dir, f"layouts.{rank}.pt"))


class _TestBase(CustomTestCase):
    def setUp(self):
        self._previous_args = server_args_module._global_server_args
        _init_parallel_state()

    def tearDown(self):
        set_global_server_args(self._previous_args)


class TestMossImageDenoise(_TestBase):
    def test_padded_joint_rows_match_stock(self):
        # 5, 9 and 17 prompt tokens share the 32-row bucket; 18 takes 64.
        # Stock groups (5, 9, 5, 7) by length; one bucket holds all four, each
        # sample attending its own real keys. Steps 0 and 1 take the table.
        model, rows = _model(), []
        model.layers[0].register_forward_pre_hook(
            lambda _, args: rows.append(tuple(args[0].shape[:2]))
        )
        for lengths, skip, batch_samples, step in (
            ((5,), (), True, None),
            ((9,), (1,), True, 1),
            ((17,), (1,), True, 0),
            ((18,), (), True, 1),
            ((5, 9, 5, 7), (), True, None),
            ((5, 9, 5, 7), (1,), False, None),
        ):
            want = _run(model, lengths, skip, batch_samples, step)
            rows.clear()
            with _installed():
                got = _run(model, lengths, skip, batch_samples, step)
            torch.testing.assert_close(got, want, **_TOL)
            bucket = 64 if lengths == (18,) else 32
            groups = [(len(lengths), bucket)] if batch_samples else [(1, bucket)] * 4
            self.assertEqual(rows, groups)

    def test_nan_in_padded_prompt_rows_stays_there(self):
        def poison(prepared):
            for group in prepared.groups:
                for row, length in enumerate(group.lengths):
                    group.text[row, length:] = float("nan")

        model = _model()
        with _installed():
            want = _run(model, (5, 9))
            got = _run(model, (5, 9), tweak=poison)
        self.assertTrue(torch.isfinite(got).all())
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_padded_text_refiner_and_host_tables_match_stock(self):
        model = _model()
        states = torch.randn(3, 17, 64)
        mask = torch.arange(17)[None] < torch.tensor([[5], [9], [17]])
        with torch.no_grad(), set_forward_context(0, None):
            want = model.prepare_conditioning(states, mask, 3, 5, torch.float32)
            rope = dit._prepare_rope(model.rope(3, 5, 9, torch.device("cpu")))
            # The model's frequency buffer stays on its device (meta here).
            model.rope.register_buffer("bands", model.rope.bands.to("meta"))
            with _installed():
                got = model.prepare_conditioning(states, mask, 3, 5, torch.float32)
        for g, w in zip(got.groups, want.groups, strict=True):
            (length,) = g.lengths
            self.assertEqual(g.text.shape[1], 17)
            torch.testing.assert_close(g.text[:, :length], w.text, **_TOL)
            self.assertFalse(g.text[:, length:].any())
        for table, stock in zip(got.groups[1].rope, rope, strict=True):
            torch.testing.assert_close(table[:, :24], stock, rtol=0, atol=0)
            self.assertFalse(table[:, 24:].any())
        # The timestep sinusoid of every step, as the stock float64 code.
        timesteps = torch.tensor([1000.0, 938.0, 24.0])
        table = moss_image._time_embeddings(timesteps, 32, torch.device("cpu"))
        freq = 10000 ** -(torch.arange(16, dtype=torch.float64) / 16)
        phase = timesteps.double()[:, None] * freq
        want = torch.cat((phase.cos(), phase.sin()), -1).float()
        torch.testing.assert_close(torch.cat(table), want, rtol=0, atol=0)

    def test_sequence_parallel_shards_match_one_rank(self):
        # SP 2 over gloo: 15 image rows plus 5, 18 or (5, 9) prompt tokens.
        cases = [((5,), ()), ((18,), (1,)), ((5, 9), ())]
        model = _model()
        wants = [_run(model, lengths, skip) for lengths, skip in cases]
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with (
            tempfile.TemporaryDirectory() as out_dir,
            patch.dict(os.environ, SGLANG_DIFFUSION_PLATFORM_OVERRIDE="cpu"),
        ):
            mp.spawn(_sp_worker, args=(port, out_dir, cases), nprocs=2)
            for index, want in enumerate(wants):
                got = torch.load(os.path.join(out_dir, f"{index}.pt"))
                torch.testing.assert_close(got, want, **_TOL)
            layouts = [
                torch.load(os.path.join(out_dir, f"layouts.{rank}.pt"))
                for rank in range(2)
            ]
        # torch_tpu bakes an input's strides and storage offset into the block
        # program: every rank must feed every block one layout at offset 0.
        self.assertEqual(layouts[0], layouts[1])
        self.assertEqual({offset for _, _, offset in layouts[0]}, {0})
        self.assertEqual({shape[1] for shape, _, _ in layouts[0]}, {16, 32})


class TestMossImageCompile(_TestBase):
    def setUp(self):
        super().setUp()
        # The TPU worker turns automatic dynamic shapes off.
        self.enterContext(torch._dynamo.config.patch(automatic_dynamic_shapes=False))
        self.addCleanup(torch._dynamo.reset)

    def test_joint_blocks_compile_once_per_bucket(self):
        model, graphs, names, counts = _model(), [], [], []
        torch._dynamo.reset()
        with _installed():
            for name, module in model.named_modules():
                if name and any(c(name, module) for c in model._compile_conditions):
                    module.compile(
                        backend=lambda gm, _: graphs.append(gm) or gm.forward,
                        dynamic=False,
                    )
                    names.append(name)
            for lengths in ((5,), (17,), (9,), (18,)):
                _run(model, lengths)
                counts.append(len(graphs))
        # Only the joint blocks: the text refiner runs once per request.
        self.assertEqual(names, ["layers.0", "layers.1", "layers.2"])
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts[2], counts[0])  # 17 and 9 tokens: 5's bucket
        self.assertGreater(counts[3], counts[2])  # 18 tokens: the next bucket


class _Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, prompt, **kwargs):
        ids = torch.tensor([[ord(c) % 60 + 1 for c in prompt][: kwargs["max_length"]]])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})


class TestMossImagePromptPadding(_TestBase):
    def test_prompt_tokens_pad_to_the_bucket_and_real_rows_match_stock(self):
        torch.manual_seed(0)
        encoder_config = Qwen3_5EncoderConfig()
        encoder_config.arch_config.text_config = Qwen3_5TextConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000.0,
                partial_rotary_factor=0.25,
                mrope_section=[3, 3, 2],
                mrope_interleaved=True,
            ),
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            vocab_size=64,
        )
        encoder_config.post_diffusers_config_update()
        encoder = Qwen3_5Model(encoder_config).eval()
        for param in encoder.parameters():
            param.data.normal_(0, 0.1)
        stage = stages.MOSSImageBeforeDenoisingStage(encoder, _Tokenizer(), None)
        seen = []
        encoder.register_forward_pre_hook(
            lambda _, args, kwargs: seen.append(kwargs["input_ids"].shape[1]),
            with_kwargs=True,
        )
        prompts = ["a red apple", "a much longer prompt about a lighthouse"]
        want, want_mask = stage._encode_prompt(prompts, torch.device("cpu"))
        with _installed():
            got, mask = stage._encode_prompt(prompts, torch.device("cpu"))
        self.assertEqual(seen, [11, 39, 16, 48])
        self.assertEqual(tuple(got.shape), (2, 48, 64))
        self.assertTrue(torch.equal(mask, torch.nn.functional.pad(want_mask, (0, 9))))
        # The encoder is causal: real rows never see the padding. CPU GEMMs
        # round by row count, so not bitwise.
        torch.testing.assert_close(got[:, :39], want, **_TOL)
        self.assertFalse(got[~mask].any())


class TestMossImageDecode(unittest.TestCase):
    def test_vae_decodes_one_image_at_a_time(self):
        vae = SimpleNamespace(use_slicing=False)
        modules = SimpleNamespace(get_module={"vae": vae}.get)
        with _installed(), patch.object(moss_image, "_STOCK_CREATE_STAGES") as stock:
            pipeline.MOSSImagePipeline.create_pipeline_stages(modules, None)
        stock.assert_called_once_with(modules, None)
        self.assertTrue(vae.use_slicing)


class TestMossImageServerArgs(unittest.TestCase):
    def test_regional_compile_on_tpu_only(self):
        for tpu, regional in ((True, True), (True, False), (False, True)):
            args = SimpleNamespace(
                tp_size=1,
                enable_torch_compile=True,
                regional_compile=regional,
                disable_autocast=True,
            )
            refused = (
                contextlib.nullcontext()
                if tpu and regional
                else self.assertRaisesRegex(ValueError, "torch.compile")
            )
            with patch.object(config.current_platform, "is_tpu", return_value=tpu):
                with refused:
                    config.MOSSImagePipelineConfig().validate_server_args(args)


if __name__ == "__main__":
    unittest.main()
