"""Qwen-Image conditioning resolves positions and kept rows from host tensors."""

import unittest
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
from transformers import BatchEncoding, BatchFeature

from sglang.multimodal_gen.configs.pipeline_configs import qwen_image as config
from sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl import Qwen2_5_VLModel
from sglang.multimodal_gen.runtime.pipelines_core.stages import image_encoding
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.test.unit.test_text_encoding_cache import (
    _GLOBAL_ARGS_PATCH,
    FullHiddenStateEncoder,
    make_server_args,
    make_text_config,
)

_MASK = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])


def _host(cls):
    """``cls`` that records its upload: positions must come from host tensors."""

    class Host(cls):
        uploaded = False

        def to(self, *args, **kwargs):
            self.uploaded = True
            return super().to(*args, **kwargs)

    return Host


class _PositionEncoder(FullHiddenStateEncoder):
    def forward(self, input_ids, position_ids=None, **kwargs):
        self.position_ids = position_ids
        return super().forward(input_ids, **kwargs)


class _VisionLanguageEncoder(torch.nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.inputs = inputs
        self.positions = torch.arange(70).view(1, 1, -1).expand(3, 1, -1)
        self.model = SimpleNamespace(get_rope_index=self._rope_index)

    def _rope_index(self, input_ids, image_grid_thw, video_grid_thw, attention_mask):
        assert input_ids.device.type == "cpu" and not self.inputs.uploaded
        return self.positions, None

    def forward(self, input_ids, position_ids=None, **kwargs):
        self.position_ids = position_ids
        return SimpleNamespace(hidden_states=(input_ids[..., None].float(),))


class TestQwenImageHostConditioning(unittest.TestCase):
    def test_extract_masked_hidden_matches_boolean_indexing(self):
        hidden = torch.randn(2, 5, 3)
        got = config._extract_masked_hidden(hidden, _MASK)
        want = torch.split(hidden[_MASK.bool()], [3, 5])
        self.assertTrue(all(torch.equal(g, w) for g, w in zip(got, want, strict=True)))

    def test_text_position_hook_matches_get_rope_index(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(spatial_merge_size=2),
                image_token_id=7,
                video_token_id=8,
                vision_start_token_id=6,
            )
        )
        model._get_text_rope_index = MethodType(
            Qwen2_5_VLModel._get_text_rope_index, model
        )
        inputs = {"input_ids": torch.arange(10).view(2, 5), "attention_mask": _MASK}
        want, _ = Qwen2_5_VLModel.get_rope_index(
            model, inputs["input_ids"], attention_mask=_MASK
        )
        hook = config.QwenImagePipelineConfig().get_text_encoder_position_ids
        self.assertTrue(torch.equal(hook(inputs, 0, None).expand(3, -1, -1), want))

    def test_text_stage_passes_host_position_ids(self):
        text_config = make_text_config()
        tokenize = text_config.tokenize_prompt
        text_config.tokenize_prompt = lambda *args: _host(BatchEncoding)(
            tokenize(*args)
        )

        def position_ids(inputs, index, encoder):
            assert not inputs.uploaded
            return config.QwenImagePipelineConfig.get_text_encoder_position_ids(
                text_config, inputs, index, encoder
            )

        text_config.get_text_encoder_position_ids = position_ids
        encoder = _PositionEncoder().eval()
        with patch(_GLOBAL_ARGS_PATCH, return_value=MagicMock()):
            stage = TextEncodingStage(text_encoders=[encoder], tokenizers=[object()])
        stage._begin_text_encoder_use = MagicMock()
        stage._text_encode_dp_group = MagicMock(return_value=None)
        args = make_server_args(pipeline_config=text_config)
        stage.encode_text("hi", args, device="cpu")
        self.assertEqual(encoder.position_ids.tolist(), [[0, 1]])

    def _edit_stage(self, pipeline_config):
        """The positions the edit stage passes, the encoder's own, the batch."""
        ids = torch.arange(70)[None]
        inputs = _host(BatchFeature)(
            {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "pixel_values": torch.zeros(4, 3),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        )
        encoder = _VisionLanguageEncoder(inputs)
        with patch(_GLOBAL_ARGS_PATCH, return_value=MagicMock()):
            stage = image_encoding.ImageEncodingStage(
                Mock(return_value=inputs), text_encoder=encoder
            )
        stage.use_declared_component = Mock(return_value=nullcontext(encoder))
        batch = SimpleNamespace(
            condition_image=[object()],
            prompt="p",
            do_classifier_free_guidance=False,
            prompt_embeds=[],
            prompt_embeds_mask=None,
            prompt_seq_lens=None,
        )
        with patch.object(
            image_encoding, "get_local_torch_device", return_value=torch.device("cpu")
        ):
            stage.forward(batch, SimpleNamespace(pipeline_config=pipeline_config))
        return encoder.position_ids, encoder.positions, batch

    def test_edit_stage_passes_host_position_ids(self):
        got, want, batch = self._edit_stage(config.QwenImageEditPlusPipelineConfig())
        self.assertTrue(torch.equal(got, want))
        # The template prefix dropped by host row indices.
        self.assertEqual(batch.prompt_embeds[0].shape[1], 70 - 64)
        # A pipeline whose config keeps the default hook leaves them to the encoder.
        other = config.QwenImageEditPlusPipelineConfig()
        other.get_text_encoder_position_ids = lambda *args: None
        self.assertIsNone(self._edit_stage(other)[0])


if __name__ == "__main__":
    unittest.main()
