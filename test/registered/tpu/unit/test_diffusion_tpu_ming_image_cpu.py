# SPDX-License-Identifier: Apache-2.0
"""CPU tests for sglang_tpu's Ming-Image patches (sglang_tpu.multimodal_gen.ming_image)."""

import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# CPU tensors throughout: on the CPU platform the distributed bootstrap opens no
# TPU chip.
os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = "cpu"

import torch
from test_diffusion_tpu_ming_image import (
    IMAGE_TOKEN,
    _encoder,
    _encoder_inputs,
    _gmm_reference,
    _installed,
    _run,
    _stock_reference,
    _TestBase,
)

from sglang.multimodal_gen.configs.models.dits.ming_image import (
    MingImageArchConfig,
    MingImageDitConfig,
)
from sglang.multimodal_gen.configs.models.encoders.ming_image import (
    MingImageEncoderConfig,
)
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.loader.component_loaders import (
    ming_image as ming_loader,
)
from sglang.multimodal_gen.runtime.models.dits import ming_image as dit
from sglang.multimodal_gen.runtime.models.encoders import base as encoder_base
from sglang.multimodal_gen.runtime.models.encoders import ming_image as encoder
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.multimodal_gen import ming_image

register_tpu_ci(est_time=40, suite="stage-a-test-tpu-unit", chips=1)


@contextlib.contextmanager
def _on_cpu(counting=None):
    """The TPU paths on CPU: the Pallas MoE is the reference above, and each
    compiled block is traced whole, then run eagerly."""
    from sglang_tpu.moe.pallas import fused_moe_op

    with (
        _installed(counting or "eager"),
        patch.object(ming_image, "_DEVICE_TYPES", ("cpu",)),
        patch.object(ming_image, "_tpu_metadata", contextlib.nullcontext),
        patch.object(fused_moe_op, "fused_moe_call", _gmm_reference),
        patch.object(fused_moe_op, "_build_fused_moe_op", lambda: None),
    ):
        yield


def _counting(graphs):
    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    return backend


def _reference_inputs():
    """A prompt with a reference image: 16 patches, 4 tokens after the merge."""
    gen = torch.Generator().manual_seed(1)
    ids = [5, 6] + [401] + [IMAGE_TOKEN] * 4 + [402] + [7, 8, 9]
    ids += [401] + [IMAGE_TOKEN] * 4 + [402]
    positions = encoder.ming_position_ids(ids, [(1, 4, 4), (1, 2, 8)], IMAGE_TOKEN)
    return dict(
        input_ids=torch.tensor([ids]),
        position_ids=positions,
        pixel_values=torch.randn(16, 3 * 2 * 14 * 14, generator=gen),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
    )


def _dit():
    gen = torch.Generator().manual_seed(0)
    arch = MingImageArchConfig(
        dim=64,
        num_layers=2,
        n_refiner_layers=1,
        num_attention_heads=2,
        n_kv_heads=2,
        cap_feat_dim=48,
        in_channels=16,
        axes_dims=(8, 12, 12),
        alignment_padding_mode="zero_masked",
        multi_frame_output=False,
    )
    with patch.object(CustomOp, "dispatch_forward", lambda op: op.forward_tpu):
        model = dit.MingImageTransformer2DModel(
            config=MingImageDitConfig(arch_config=arch), hf_config={}
        )
    for param in model.parameters():
        param.data.normal_(0, 0.05, generator=gen)
    return model.float().eval()


def _dit_inputs(prompt_rows, size=8):
    gen = torch.Generator().manual_seed(prompt_rows)
    return dict(
        hidden_states=torch.randn(1, 16, 1, size, size, generator=gen),
        encoder_hidden_states=torch.randn(1, 4, 48, generator=gen),
        timestep=torch.full((1,), 500.0),
        direct_embeddings=torch.randn(1, prompt_rows, 64, generator=gen),
    )


class TestMingImageEncoder(_TestBase):
    def test_padded_compiled_encoder_matches_stock(self):
        # One and two 256-token buckets, and a reference image.
        for inputs in (_encoder_inputs(9), _encoder_inputs(300), _reference_inputs()):
            with patch.object(encoder, "fused_experts", _stock_reference):
                expected = _run(_encoder(), inputs)
            model = _encoder()
            with _on_cpu():
                actual = _run(model, inputs)
                # The experts moved to the TPU layout once and stay there.
                _run(model, inputs)
            for a, e in (
                (actual.last_hidden_state, expected.last_hidden_state),
                (actual.hidden_states[0], expected.hidden_states[0]),
            ):
                torch.testing.assert_close(a, e, atol=1e-5, rtol=1e-4)

    def test_encoder_compiles_one_graph_per_block_kind(self):
        model = _encoder(layers=6, connector_layers=3)
        graphs = []
        with _on_cpu(_counting(graphs)):
            for prompt_length in (9, 60):  # one 256-token bucket
                _run(model, _encoder_inputs(prompt_length))
        # The dense layer, the five MoE layers and the connector layers.
        self.assertEqual(len(graphs), 3)

    def test_encoder_folds_only_when_the_connector_heads_divide(self):
        for policy, ranks, heads, expected in (
            ("fold", 8, 12, None),  # 12 connector heads over 8 ranks
            ("fold", 4, 12, "world"),
            ("fold", 8, 16, "world"),
            ("replicate", 4, 12, None),
        ):
            config = MingImageEncoderConfig()
            config.arch_config.connector_config = {"num_attention_heads": heads}
            config.parallel_folding_mode = "world"
            group = SimpleNamespace(world_size=ranks)
            with (
                patch.object(encoder_base, "get_folding_tp_group", return_value=group),
                patch.object(ming_loader, "get_folding_tp_group", return_value=group),
            ):
                ming_loader.resolve_ming_encoder_folding(config, policy)
            self.assertEqual(config.parallel_folding_mode, expected)


class TestMingImageDiT(_TestBase):
    def test_bucketed_caption_matches_stock(self):
        model = _dit()
        for prompt_rows in (5, 40):
            kwargs = _dit_inputs(prompt_rows)
            expected = _run(model, kwargs)
            with _on_cpu():
                actual = _run(model, kwargs)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)

    def test_whole_dit_compiles_once_across_caption_lengths(self):
        model = _dit()
        graphs = []
        with _on_cpu():
            # An eager call first: fused_op resolves its backend lazily, and
            # that global would add one recompile unrelated to the caption.
            _run(model, _dit_inputs(5))
            model.compile(backend=_counting(graphs), dynamic=False)
            for prompt_rows in (5, 40, 300):  # all in the 512-row caption bucket
                _run(model, _dit_inputs(prompt_rows))
        self.assertEqual(len(graphs), 1)


if __name__ == "__main__":
    unittest.main()
