# SPDX-License-Identifier: Apache-2.0
"""Reference geometry and checkpoint contracts for Muse's TPU vision path."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from PIL import Image

from sglang.srt.configs.muse_spark import MuseSparkVisionConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.models.muse_spark_vision import (
    _patch_grid_positions,
    prepare_vision_input,
    vision_metadata,
)
from sglang_tpu.multimodal.muse_spark_processor import (
    PreprocessConfig,
    preprocess_image,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMuseVision(unittest.TestCase):
    def test_dp_exchange_padding_is_initialized_for_empty_and_partial_ranks(self):
        from sglang_tpu.models.muse_spark_omni import _pad_dp_embeddings

        for count in (0, 100, 196):
            local = torch.full((count, 8), 1.5, dtype=torch.bfloat16)
            padded = _pad_dp_embeddings(local, 196)
            torch.testing.assert_close(padded[:count], local)
            torch.testing.assert_close(padded[count:], torch.zeros_like(padded[count:]))
            self.assertTrue(torch.isfinite(padded).all())

    def test_rejects_attention_backends_that_ignore_segment_ids(self):
        """A backend that drops segment_ids lets padding and other windows leak in."""
        from sglang.srt.layers.attention.vision import (
            VisionSdpaAttention,
            VisionTpuAttention,
        )
        from sglang_tpu.models import muse_spark_vision

        def attention(backend_cls, name):
            backend = backend_cls.__new__(backend_cls)
            return lambda **_: SimpleNamespace(
                qkv_backend=backend, qkv_backend_name=name
            )

        with patch.object(
            muse_spark_vision, "VisionAttention", attention(VisionSdpaAttention, "sdpa")
        ):
            with self.assertRaisesRegex(ValueError, "'sdpa' ignores them"):
                muse_spark_vision.NaViTBlock(32, 64, 2, True)
        with patch.object(
            muse_spark_vision, "VisionAttention", attention(VisionTpuAttention, "tpu")
        ):
            muse_spark_vision.NaViTBlock(32, 64, 2, True)

    def test_vision_features_preserve_device_buffer_through_host_placement(self):
        from sglang_tpu.mm_embed_tpu import _place_mm_rows
        from sglang_tpu.mm_embedding import PaddedEmbedding, split_embedding
        from sglang_tpu.models.muse_spark_omni import (
            MuseSparkForConditionalGeneration as Omni,
        )

        model = Omni.__new__(Omni)
        torch.nn.Module.__init__(model)
        model.text_config = SimpleNamespace(hidden_size=8)
        model.model = torch.nn.Module()
        model.model.vision_tower = torch.nn.Linear(1, 1)
        model.model.vision_tower.config = SimpleNamespace(
            patch_temporal=2, patch_size=14, merge_size=2
        )
        items = [SimpleNamespace(feature=torch.zeros(3, 28, 28))] * 2
        buffer = torch.arange(16).reshape(2, 8).float()
        with patch(
            "sglang_tpu.models.muse_spark_omni._run_dp_sharded_encode",
            return_value=buffer,
        ):
            embedding = model.get_image_feature(items)
        self.assertIsInstance(embedding, PaddedEmbedding)
        self.assertIs(embedding.spans[0].buffer, buffer)
        # Cache splitting and chunk slicing retain the device buffer; the host
        # placement boundary, rather than dtype conversion, performs the copy.
        second = split_embedding(embedding, [1, 1])[1].slice(0, 1)
        mask = torch.tensor([[False], [True], [False]])
        with patch.object(PaddedEmbedding, "to_host", autospec=True) as to_host:
            to_host.return_value = buffer[1:]
            placed, union = _place_mm_rows(
                masks=[mask],
                embeddings=[second],
                rows=3,
                hidden=8,
                dtype=torch.float32,
            )
        to_host.assert_called_once_with(second)
        torch.testing.assert_close(placed[1], buffer[1])
        torch.testing.assert_close(placed[[0, 2]], torch.zeros(2, 8))
        torch.testing.assert_close(union, mask)

    def test_encoder_owns_host_feature_transfer_before_dp_assignment(self):
        from sglang_tpu import mm_schedule_tpu

        class HostEncoder:
            keeps_mm_features_on_host = True

            def get_image_feature(self, items):
                return items

        encoder = HostEncoder()
        feature = torch.ones(3, 28, 28)
        item = SimpleNamespace(feature=feature)
        with patch.object(mm_schedule_tpu, "_default_move_items_to_device") as default:
            mm_schedule_tpu._move_items_to_device(
                [item], torch.device("meta"), encoder.get_image_feature
            )
            default.assert_not_called()
            self.assertIs(item.feature, feature)
            encoder.keeps_mm_features_on_host = False
            mm_schedule_tpu._move_items_to_device(
                [item], torch.device("meta"), encoder.get_image_feature
            )
            default.assert_called_once_with(
                [item], torch.device("meta"), encoder.get_image_feature
            )

    def test_multimodal_compile_policy_interface_and_decode_dispatch(self):
        from sglang_tpu.models.muse_spark_omni import (
            MuseSparkForConditionalGeneration as Omni,
        )
        from sglang_tpu.models.muse_spark_omni import (
            _NormalizedEmbedding,
        )

        embedding = torch.nn.Embedding(16, 8)
        normalized = _NormalizedEmbedding(embedding, 1e-5)
        self.assertEqual(normalized.embedding_dim, 8)
        self.assertEqual(normalized.num_embeddings, 16)
        self.assertIs(normalized.weight, embedding.weight)
        self.assertEqual(list(normalized.parameters()), [])
        self.assertTrue(Omni.decode_embeds_in_language_model)
        model = Omni.__new__(Omni)
        torch.nn.Module.__init__(model)
        model.vision_enabled = True
        model.language_model = Mock(return_value="decoded")
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True))
        ids = torch.tensor([1])
        positions = torch.tensor([0])
        with patch(
            "sglang_tpu.models.muse_spark_omni.mm_utils.general_mm_embed_routine"
        ) as merge:
            self.assertEqual(model(ids, positions, batch), "decoded")
            merge.assert_not_called()
        model.language_model.assert_called_once_with(ids, positions, batch)

    def test_decode_reuses_warmed_graphs_for_multiple_batch_sizes(self):
        from sglang_tpu.models.muse_spark_omni import (
            MuseSparkForConditionalGeneration as Omni,
        )

        class Decoder(torch.nn.Module):
            def forward(self, input_ids, positions, forward_batch, input_embeds=None):
                return input_ids + positions

        model = Omni.__new__(Omni)
        torch.nn.Module.__init__(model)
        model.vision_enabled = True
        model.language_model = Decoder()
        model.language_model.forward = torch.compile(
            model.language_model.forward, backend="eager", fullgraph=True, dynamic=False
        )
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True))
        # The decoder policy warms exactly three positional arguments. Serving
        # must reuse these graphs after fail_on_recompile has been installed.
        with torch.no_grad():
            for size in (1, 2, 4, 8):
                ids = torch.arange(size)
                model.language_model(ids, ids, batch)
            with torch.compiler.set_stance("fail_on_recompile"):
                for size in (1, 2, 4, 8):
                    ids = torch.arange(size)
                    torch.testing.assert_close(model(ids, ids, batch), ids * 2)

    def test_positional_interpolation_matches_grid_sample_on_rectangles(self):
        table = torch.randn(8 * 6, 12)
        for h, w in ((4, 10), (10, 4), (8, 6), (2, 2)):
            meta = vision_metadata(h, w, 8, 8, 6, 10000, 2, 1024)
            order, idx, weights = meta[:3]
            actual = (table[idx] * weights[..., None]).sum(1)[: h * w]
            grid = _patch_grid_positions(h, w, torch.device("cpu"))
            expected = F.grid_sample(
                table.reshape(8, 6, 12).permute(2, 0, 1)[None],
                grid[None, None],
                align_corners=False,
            )[0, :, 0].T
            torch.testing.assert_close(actual, expected[order], atol=1e-6, rtol=1e-5)
            # Both full and window attention isolate every padding token.
            self.assertFalse(torch.isin(meta[4][0, : h * w], meta[4][0, h * w :]).any())
            self.assertFalse(torch.isin(meta[5][0, : h * w], meta[5][0, h * w :]).any())
            # Restore raster order before the checkpoint's 2x2 pixel shuffle.
            raster = torch.arange(h * w)
            restored = F.pad(raster[order], (0, 1024 - h * w))[meta[6]][: h * w]
            expected_shuffle = (
                raster.reshape(h // 2, 2, w // 2, 2).permute(0, 2, 1, 3).flatten()
            )
            torch.testing.assert_close(restored, expected_shuffle)

    def test_preprocessing_patch_count_and_temporal_duplication(self):
        image = Image.new("RGB", (84, 56), (255, 127, 0))
        feature, count = preprocess_image(
            image, PreprocessConfig(use_rust_resize=False)
        )
        self.assertEqual(count, 6)
        cfg = MuseSparkVisionConfig(hidden_size=16, num_attention_heads=2)
        inputs, actual = prepare_vision_input(feature, cfg)
        self.assertEqual(actual, count)
        self.assertEqual(inputs[0].shape, (1024, 3 * 14 * 14 * 2))
        torch.testing.assert_close(inputs[0][:24, :588], inputs[0][:24, 588:])
        with self.assertRaisesRegex(ValueError, "align"):
            prepare_vision_input(torch.zeros(3, 55, 84), cfg)

    def test_vision_qkv_bias_permutation_and_completeness(self):
        from sglang_tpu.models.muse_spark_omni import _VisionWeightLoader

        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.vision_tower = torch.nn.Module()
        model.model.vision_tower.config = SimpleNamespace(num_attention_heads=2)
        model.model.vision_tower.attn = torch.nn.Module()
        model.model.vision_tower.attn.qkv_proj = torch.nn.Linear(4, 24, bias=True)
        layer = model.model.vision_tower.attn.qkv_proj

        def loader(param, weight, shard):
            param.data["qkv".index(shard) * 8 : ("qkv".index(shard) + 1) * 8].copy_(
                weight
            )

        layer.weight.weight_loader = loader
        layer.bias.weight_loader = loader
        tensors = [("model.language_model.norm.weight", torch.ones(4))]
        for shard in "qkv":
            tensors.extend(
                [
                    (
                        f"model.vision_tower.attn.{shard}_proj.weight",
                        torch.arange(32).reshape(8, 4).float(),
                    ),
                    (
                        f"model.vision_tower.attn.{shard}_proj.bias",
                        torch.arange(8).float(),
                    ),
                ]
            )
        incomplete = _VisionWeightLoader(model)
        list(incomplete.filter_text(tensors[:-1]))
        with self.assertRaisesRegex(ValueError, "qkv_proj.bias"):
            incomplete.finish()
        vision = _VisionWeightLoader(model)
        # Decoder tensors pass through untouched for the text loader.
        self.assertEqual(
            [name for name, _ in vision.filter_text(tensors)],
            ["model.language_model.norm.weight"],
        )
        vision.finish()
        perm = torch.tensor([0, 2, 1, 3, 4, 6, 5, 7])
        torch.testing.assert_close(
            layer.bias,
            torch.cat(
                [torch.arange(8)[perm], torch.arange(8)[perm], torch.arange(8)]
            ).float(),
        )


if __name__ == "__main__":
    unittest.main()
