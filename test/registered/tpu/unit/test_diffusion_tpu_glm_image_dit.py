# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's GLM-Image DiT glyph bucket (sglang_tpu.multimodal_gen.glm_image_dit)."""

import contextlib
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

# CPU tensors throughout: on the CPU platform, neither the distributed
# bootstrap nor the spawned ranks open a TPU chip.
os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = "cpu"

import torch
import torch.multiprocessing as mp

from sglang.kernels.ops.diffusion.common import fallback_torch
from sglang.multimodal_gen.configs.models.dits.glmimage import (
    GlmImageArchConfig,
    GlmImageDitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.glm_image import (
    GlmImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.layers import layernorm
from sglang.multimodal_gen.runtime.managers.forward_context import (
    set_forward_context,
)
from sglang.multimodal_gen.runtime.models.dits import glm_image as dit
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import glm_image_dit

register_tpu_ci(est_time=40, suite="stage-a-test-tpu-unit", chips=1)

_GLOBAL_ARGS = (
    "sglang.multimodal_gen.runtime.server_args.server_args._global_server_args"
)
TEXT_DIM, CODEBOOK = 24, 50


@contextlib.contextmanager
def _native_norms():
    # The CPU fused norms need sgl_kernel's CPU build; the native norm runs
    # uncompiled (the CPU platform compiles it with inductor).
    norm = layernorm._ScaleResidualNormScaleShift
    with (
        patch.object(norm, "forward_cpu", norm.forward_native.__wrapped__),
        patch.object(
            layernorm,
            "fuse_scale_shift_kernel",
            fallback_torch.fuse_scale_shift_kernel_native,
        ),
    ):
        yield


@contextlib.contextmanager
def _installed():
    with contextlib.ExitStack() as stack:
        for cls, name in (
            (dit.GlmImageTransformer2DModel, "__call__"),
            (dit.GlmImageAttention, "forward"),
        ):
            stack.enter_context(patch.object(cls, name, getattr(cls, name)))
        stack.enter_context(patch.object(glm_image_dit, "_DEVICE_TYPES", ("cpu",)))
        stack.enter_context(_native_norms())
        glm_image_dit.install()
        glm_image_dit.install()  # idempotent
        yield


def _model():
    with patch.object(PipelineConfig, "from_kwargs", return_value=PipelineConfig()):
        server_args = ServerArgs.from_kwargs(model_path="/fake")
    arch = GlmImageArchConfig(
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=4,
        in_channels=4,
        out_channels=4,
        text_embed_dim=TEXT_DIM,
        time_embed_dim=32,
        condition_dim=16,
        prior_vq_quantizer_codebook_size=CODEBOOK,
    )
    with patch(_GLOBAL_ARGS, server_args), _native_norms():
        model = dit.GlmImageTransformer2DModel(
            GlmImageDitConfig(arch_config=arch), hf_config={}
        )
    gen = torch.Generator().manual_seed(0)
    for param in model.parameters():
        param.data.normal_(0, 0.1, generator=gen)
    for block in model.transformer_blocks:
        # The joint attention runs the Pallas backend, as on a v7x worker.
        block.attn1.attn.attn_impl.use_splash = True
    return model.eval()


def _inputs(text_len, height=8, width=8):
    gen = torch.Generator().manual_seed(text_len)
    tokens = (height // 2) * (width // 2)
    return dict(
        hidden_states=torch.randn(1, 4, height, width, generator=gen),
        encoder_hidden_states=torch.randn(1, text_len, TEXT_DIM, generator=gen),
        prior_token_id=torch.randint(0, CODEBOOK, (1, tokens), generator=gen),
        # The unconditional pass: one glyph row, every prior token dropped.
        prior_token_drop=torch.full((1, tokens), text_len == 1),
        timestep=torch.tensor([500.0]),
        target_size=torch.tensor([[64.0, 64.0]]),
        crop_coords=torch.zeros(1, 2),
    )


def _padded_and_stock(model, inputs):
    """The DiT through the patched ``__call__``, and the stock forward."""
    with set_forward_context(current_timestep=0, attn_metadata=None):
        return model(**inputs), model.forward(**inputs)


def _sp_worker(rank, world_size, init_file):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    parallel_state.maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=world_size,
        ulysses_degree=world_size,
        distributed_init_method=f"file://{init_file}",
    )
    model = _model()
    config = GlmImagePipelineConfig()
    offsets = set()

    def recording(gm, example_inputs):
        def run(*args):
            offsets.update(a.storage_offset() for a in args if torch.is_tensor(a))
            return gm.forward(*args)

        return run

    # The denoising stage compiles the whole DiT; torch_tpu specializes it on
    # its inputs' storage offsets.
    model.compile(backend=recording, dynamic=False)
    with _installed(), torch.no_grad():
        # The conditional and unconditional passes of a step.
        for text_len in (5, 1):
            inputs = _inputs(text_len)
            latents = inputs["hidden_states"]
            side = latents.shape[-1] * config.vae_scale_factor
            # This rank's latent rows and their RoPE rows, as the pipeline shards.
            inputs.update(
                hidden_states=config.shard_latents_for_sp(None, latents)[0],
                freqs_cis=config.get_freqs_cis(
                    SimpleNamespace(height=side, width=side),
                    latents.device,
                    model.rotary_emb,
                    latents.dtype,
                ),
            )
            padded, stock = _padded_and_stock(model, inputs)
            torch.testing.assert_close(padded, stock, rtol=1e-5, atol=1e-5)
    # Every rank runs one program: its inputs start their storage.
    assert offsets == {0}, offsets
    parallel_state.cleanup_dist_env_and_memory()


class TestGlmImageGlyphBucket(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._rendezvous = None
        if not parallel_state.model_parallel_is_initialized():
            cls._rendezvous = tempfile.TemporaryDirectory()
            parallel_state.maybe_init_distributed_environment_and_model_parallel(
                tp_size=1,
                sp_size=1,
                distributed_init_method=f"file://{cls._rendezvous.name}/store",
            )

    @classmethod
    def tearDownClass(cls):
        if cls._rendezvous is not None:
            parallel_state.cleanup_dist_env_and_memory()
            cls._rendezvous.cleanup()

    def test_padded_text_matches_stock(self):
        # One glyph row (the unconditional pass), a few, and a second bucket.
        model = _model()
        with _installed(), torch.no_grad():
            for text_len in (1, 5, 300):
                padded, stock = _padded_and_stock(model, _inputs(text_len))
                torch.testing.assert_close(padded, stock, rtol=1e-5, atol=1e-5)

    def test_other_backends_and_no_qk_norm_run_stock(self):
        model = _model()
        impls = [block.attn1.attn.attn_impl for block in model.transformer_blocks]
        attns = [block.attn1 for block in model.transformer_blocks]
        for targets, name in ((impls, "use_splash"), (attns, "norm_q")):
            with contextlib.ExitStack() as stack, torch.no_grad():
                stack.enter_context(_installed())
                for target in targets:
                    stack.enter_context(patch.object(target, name, None))
                self.assertFalse(glm_image_dit._pads_text(model))
                padded, stock = _padded_and_stock(model, _inputs(5))
                torch.testing.assert_close(padded, stock, rtol=1e-5, atol=1e-5)

    def test_text_buckets_are_powers_of_two_from_16(self):
        buckets = [glm_image_dit.text_bucket(n) for n in (1, 5, 16, 17, 64, 65, 300)]
        self.assertEqual(buckets, [16, 16, 16, 32, 64, 128, 512])

    def test_compiled_dit_compiles_once_across_glyph_lengths(self):
        model = _model()
        graphs = []

        def counting(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        torch._dynamo.reset()
        model.compile(backend=counting, dynamic=False)
        with _installed(), torch.no_grad():
            # One bucket: the unconditional pass's row and short glyph texts.
            for step, text_len in enumerate((5, 7, 1, 16)):
                padded, stock = _padded_and_stock(model, _inputs(text_len))
                torch.testing.assert_close(padded, stock, rtol=1e-5, atol=1e-5)
                if step == 0:
                    first_call_graphs = len(graphs)
        torch._dynamo.reset()
        self.assertGreater(first_call_graphs, 0)
        self.assertEqual(len(graphs), first_call_graphs)

    def test_ulysses_matches_stock(self):
        with (
            tempfile.TemporaryDirectory() as rendezvous,
            # Ranks without torch_tpu.
            patch.dict(os.environ, {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}),
        ):
            mp.spawn(_sp_worker, args=(2, f"{rendezvous}/store"), nprocs=2)


if __name__ == "__main__":
    import unittest

    unittest.main()
