# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's JoyAI-Image-Edit patches (sglang_tpu.multimodal_gen.joy_image)."""

import contextlib
import inspect
import os
import socket
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.multiprocessing as mp
from transformers import BatchFeature

from sglang.kernels.ops.diffusion.common.fallback_torch import (
    fuse_scale_shift_kernel_native,
)
from sglang.multimodal_gen.configs.models.dits.joy_image import (
    JoyImageArchConfig,
    JoyImageDiTConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.joy_image import (
    joy_image_postprocess_text,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers import elementwise, layernorm
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import joy_image as dit
from sglang.multimodal_gen.runtime.pipelines import joy_image as pipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import joy_image, worker

# Whole node, not chips=1: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS pin.
register_tpu_ci(est_time=120, suite="nightly-tpu-4-v7", nightly=True)

_SIZES = [(1, 4, 4), (1, 4, 4)]  # the noisy image, then the condition image
# 15 image rows: under SP 2 the image pads one row, which the stock DiT attends.
_ODD_SIZES = [(1, 3, 3), (1, 2, 3)]
# With padded prompt rows the attention reads the same keys under a mask, in
# other blocks of the CPU kernel's sum: some bfloat16 outputs (|x| < 4 here)
# move by one rounding step (2**-6), as far from a float32 run as stock is.
# A prompt that fills its bucket matches bitwise.
_TOLERANCE = dict(rtol=0, atol=2**-5)


@contextlib.contextmanager
def _installed():
    """Run install() with every patched name restored afterwards."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (dit.JoyTransformer3DModel, "__call__"),
            (dit.JoyTransformer3DModel, "forward"),
            (dit.JoyTransformer3DModel, "_compile_regions_spmd_safe"),
            (dit.MMDoubleStreamBlock, "forward"),
            (pipeline.JoyImageEditPipeline, "create_pipeline_stages"),
            (ImageEncodingStage, "encoding_image_edit"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        joy_image.install()
        joy_image.install()  # idempotent
        yield


@contextlib.contextmanager
def _cpu_kernels():
    # The TPU kernel selection; the CPU one needs sgl_kernel's CPU ops.
    with contextlib.ExitStack() as stack:
        for module in (layernorm, elementwise):
            stack.enter_context(
                patch.object(
                    module, "fuse_scale_shift_kernel", fuse_scale_shift_kernel_native
                )
            )
        stack.enter_context(
            patch.object(CustomOp, "dispatch_forward", lambda op: op.forward_tpu)
        )
        yield


def _model():
    torch.manual_seed(0)
    arch = JoyImageArchConfig(
        num_attention_heads=2,
        attention_head_dim=128,
        mm_double_blocks_depth=2,
        text_states_dim=64,
        mlp_width_ratio=2.0,
    )
    model = dit.JoyTransformer3DModel(
        config=JoyImageDiTConfig(arch_config=arch), hf_config={}
    )
    for param in model.parameters():
        param.data.normal_(0, 0.05)
    # The stock blocks take bfloat16 or float16 only.
    return model.to(torch.bfloat16).eval()


def _inputs(text_len, dtype=torch.bfloat16, sizes=_SIZES):
    """The DiT kwargs of one request, as the denoising stage passes them."""
    gen = torch.Generator().manual_seed(text_len)
    image_len = sum(t * h * w for t, h, w in sizes)
    hidden = torch.randn(1, image_len, 16, 1, 2, 2, generator=gen)
    text = torch.randn(1, text_len, 64, generator=gen)
    return dict(
        hidden_states=hidden.bfloat16().to(dtype),
        encoder_hidden_states=[text.bfloat16().to(dtype)],
        timestep=torch.tensor([500.0]),
        encoder_hidden_states_mask=[torch.ones(1, text_len, dtype=torch.bool)],
    )


def _run(model, kwargs, sizes=_SIZES):
    batch = SimpleNamespace(enable_sequence_shard=True, vae_image_sizes=sizes)
    with (
        set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=batch
        ),
        torch.no_grad(),
    ):
        return model(**kwargs)


def _init_parallel_state(sp=1):
    set_global_server_args(_make_unit_server_args())
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        with patch.object(worker, "install_runtime_patches"):
            maybe_init_distributed_environment_and_model_parallel(
                tp_size=1, sp_size=sp, ulysses_degree=sp
            )


def _sp_worker(rank, port, out_dir, text_lens):
    """One Ulysses rank of an SP 2 run over gloo; rank 0 saves the outputs."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
    )
    _init_parallel_state(sp=2)
    with _cpu_kernels():
        model = _model()
        layouts = []
        for block in model.double_blocks:
            block.register_forward_pre_hook(
                lambda module, args: layouts.append(
                    [(t.stride(), t.storage_offset()) for t in args]
                )
            )
        with _installed():
            for text_len in text_lens:
                out = _run(model, _inputs(text_len, sizes=_ODD_SIZES), _ODD_SIZES)
                if rank == 0:
                    torch.save(out, os.path.join(out_dir, f"{text_len}.pt"))
        torch.save(layouts, os.path.join(out_dir, f"layouts.{rank}.pt"))


class _TestBase(CustomTestCase):
    def setUp(self):
        self.enterContext(_cpu_kernels())
        self._previous_args = server_args_module._global_server_args
        _init_parallel_state()

    def tearDown(self):
        set_global_server_args(self._previous_args)


class TestJoyImageRope(_TestBase):
    def test_host_table_and_pair_swap_match_the_stock_complex_rope(self):
        model = _model()
        sizes = tuple(_SIZES)
        table = joy_image._image_rope(model, 32, 0, sizes, torch.device("cpu"))
        cos, sin = model._compute_rope_for_local_shard(32, 0, sizes, table.device)
        torch.testing.assert_close(table, torch.cat((cos, sin), -1), rtol=0, atol=0)
        block = model.double_blocks[0]
        for dtype in (torch.bfloat16, torch.float32):
            norms = [
                norm.to(dtype)
                for norm in (block.img_attn_q_norm, block.img_attn_k_norm)
            ]
            q, k = (torch.randn(1, 32, 2, 128).to(dtype) for _ in range(2))
            want = dit._joy_image_qk_rope(
                q, k, *norms, table, dit._joy_complex_freqs(table)
            )
            got = joy_image._image_qk_norm_rope(q, k, *norms, table)
            for g, w in zip(got, want, strict=True):
                torch.testing.assert_close(g, w, rtol=0, atol=0)


class TestJoyImageDenoise(_TestBase):
    def test_bucketed_prompt_matches_stock(self):
        # 37 and 300 prompt rows share the 512-row bucket; 600 takes 768.
        model, float32_model = _model(), _model().float()
        rows = []
        model.condition_embedder.register_forward_pre_hook(
            lambda module, args: rows.append(args[1].shape[1])
        )
        for text_len in (37, 300, 600):
            stock = _run(model, _inputs(text_len))
            with _installed():
                got = _run(model, _inputs(text_len))
                # Float32 without padded rows: in bfloat16 this path is
                # bitwise stock (the test below).
                with (
                    patch.object(joy_image, "TEXT_ALIGNMENT", 1),
                    patch.object(joy_image, "MIN_TEXT_ROWS", 1),
                ):
                    exact = _run(float32_model, _inputs(text_len, torch.float32))
            torch.testing.assert_close(got, stock, **_TOLERANCE)
            # At most one bfloat16 rounding step (|x| < 4) further from the
            # float32 run than stock is (TPU kernels sum in other blocks too).
            self.assertLessEqual(
                (got - exact).abs().max(), (stock - exact).abs().max() + 2**-6
            )
        self.assertEqual(rows, [37, 512, 300, 512, 600, 768])

    def test_prompt_filling_its_bucket_matches_stock_bitwise(self):
        # No padded rows: the same keys in the same order, so the same sums.
        model = _model()
        for text_len in (512, 768):
            want = _run(model, _inputs(text_len))
            with _installed():
                got = _run(model, _inputs(text_len))
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_nan_in_padded_prompt_rows_stays_there(self):
        model = _model()

        def poisoned(self, hidden_states, encoder_hidden_states, *args, **kwargs):
            encoder_hidden_states = encoder_hidden_states.clone()
            encoder_hidden_states[:, 37:] = float("nan")
            return joy_image.forward(
                self, hidden_states, encoder_hidden_states, *args, **kwargs
            )

        with _installed():
            want = _run(model, _inputs(37))
            with patch.object(dit.JoyTransformer3DModel, "forward", poisoned):
                got = _run(model, _inputs(37))
        self.assertTrue(torch.isfinite(got).all())
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_sequence_parallel_prompt_shards_match_stock(self):
        # SP 2 over gloo: 37 rows leave rank 1 only padding, 300 fill rank 0
        # and part of rank 1, 600 take the next bucket.
        text_lens = (37, 300, 600)
        model = _model()
        wants = [
            _run(model, _inputs(n, sizes=_ODD_SIZES), _ODD_SIZES) for n in text_lens
        ]
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        # The ranks run on the CPU platform over gloo, on a TPU host too.
        with (
            tempfile.TemporaryDirectory() as out_dir,
            patch.dict(os.environ, SGLANG_DIFFUSION_PLATFORM_OVERRIDE="cpu"),
        ):
            mp.spawn(_sp_worker, args=(port, out_dir, text_lens), nprocs=2)
            for text_len, want in zip(text_lens, wants, strict=True):
                got = torch.load(os.path.join(out_dir, f"{text_len}.pt"))
                torch.testing.assert_close(got, want, **_TOLERANCE)
            layouts = [
                torch.load(os.path.join(out_dir, f"layouts.{rank}.pt"))
                for rank in range(2)
            ]
        # torch_tpu bakes an input's strides and storage offset into the block
        # program: every block on every rank must see fresh tensors, or the
        # ranks run different programs around their spmd-safe collectives.
        self.assertEqual(layouts[0], layouts[1])
        self.assertEqual(layouts[0][0::2], layouts[0][1::2])  # block 0 == block 1
        self.assertEqual({offset for call in layouts[0] for _, offset in call}, {0})


class _CompileCounter:
    """Regional compile of the DiT blocks, as --regional-compile on TPU."""

    def __init__(self, model):
        self.graphs = []
        torch._dynamo.reset()
        for block in model.double_blocks:
            block.compile(backend=self._backend, dynamic=False)

    def _backend(self, graph_module, example_inputs):
        self.graphs.append(graph_module)
        return graph_module.forward


class TestJoyImageCompile(_TestBase):
    def setUp(self):
        super().setUp()
        # The TPU worker turns automatic dynamic shapes off.
        self.enterContext(torch._dynamo.config.patch(automatic_dynamic_shapes=False))
        self.addCleanup(torch._dynamo.reset)

    def test_blocks_compile_once_per_prompt_bucket(self):
        model = _model()
        counter, counts = _CompileCounter(model), []
        with _installed():
            for text_len in (37, 300, 600):
                _run(model, _inputs(text_len))
                counts.append(len(counter.graphs))
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts[1], counts[0])  # 300 rows: 37's bucket
        self.assertGreater(counts[2], counts[1])  # 600 rows: the next bucket

    def test_positive_and_negative_prompts_share_one_program(self):
        # A JoyAI edit's prompts: ~1024 vision tokens plus each prompt's own.
        model = _model()
        counter = _CompileCounter(model)
        with _installed():
            _run(model, _inputs(1056))
            positive = len(counter.graphs)
            _run(model, _inputs(1033))
        self.assertGreater(positive, 0)
        self.assertEqual(len(counter.graphs), positive)


def _causal_hidden(input_ids):
    """Per-token rows that see only earlier tokens, like the text encoder."""
    return SimpleNamespace(hidden_states=(input_ids[..., None].float().cumsum(1),))


class TestJoyImagePromptPadding(_TestBase):
    def test_prompt_tokens_pad_to_256_and_only_real_rows_reach_the_dit(self):
        ids = torch.arange(1, 81)[None]

        def process(images=None, text=None, videos=None, **kwargs):
            return BatchFeature(
                {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            )

        processor = Mock(side_effect=process)
        processor.tokenizer.pad_token_id = 0
        stage = ImageEncodingStage(processor, text_encoder=SimpleNamespace())
        config = SimpleNamespace(postprocess_text_funcs=(joy_image_postprocess_text,))
        with _installed(), patch.object(joy_image, "_STOCK_CREATE_STAGES") as stock:
            pipeline.JoyImageEditPipeline.create_pipeline_stages(
                SimpleNamespace(stages=[stage]), None
            )
            stock.assert_called_once()
            self.assertEqual(
                list(inspect.signature(stage.image_processor.__call__).parameters),
                ["images", "text", "videos", "kwargs"],
            )
            padded = stage.image_processor(images=[None], text=["p"])
            got = stage.encoding_image_edit(
                _causal_hidden(padded["input_ids"]), padded, config
            )
        self.assertEqual(processor.call_args.kwargs["text"], ["p"])
        self.assertEqual(tuple(padded["input_ids"].shape), (1, 256))
        self.assertEqual(padded["attention_mask"].sum().item(), 80)
        want = joy_image_postprocess_text(_causal_hidden(ids), process())
        self.assertTrue(torch.equal(got, want))


if __name__ == "__main__":
    import unittest

    unittest.main()
