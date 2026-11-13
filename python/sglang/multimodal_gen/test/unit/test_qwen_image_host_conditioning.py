"""Qwen-Image conditioning resolves positions and kept rows from host tensors."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch
from transformers import BatchEncoding, BatchFeature

from sglang.multimodal_gen.configs.pipeline_configs import qwen_image as config
from sglang.multimodal_gen.runtime.models.encoders.qwen2_5vl import Qwen2_5_VLModel
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
)
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


def test_extract_masked_hidden_matches_boolean_indexing():
    hidden = torch.randn(2, 5, 3)
    got = config._extract_masked_hidden(hidden, _MASK)
    want = torch.split(hidden[_MASK.bool()], [3, 5])
    assert all(torch.equal(g, w) for g, w in zip(got, want, strict=True))


def test_text_position_hook_matches_get_rope_index():
    vision = SimpleNamespace(spatial_merge_size=2)
    model = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=vision,
            image_token_id=7,
            video_token_id=8,
            vision_start_token_id=6,
        )
    )
    inputs = {"input_ids": torch.arange(10).view(2, 5), "attention_mask": _MASK}
    want, _ = Qwen2_5_VLModel.get_rope_index(
        model, inputs["input_ids"], attention_mask=_MASK
    )
    hook = config.QwenImagePipelineConfig().get_text_encoder_position_ids
    assert torch.equal(hook(inputs, 0).expand(3, -1, -1), want)


class _PositionEncoder(FullHiddenStateEncoder):
    def forward(self, input_ids, position_ids=None, **kwargs):
        self.position_ids = position_ids
        return super().forward(input_ids, **kwargs)


def test_text_stage_passes_host_position_ids():
    text_config = make_text_config()
    tokenize = text_config.tokenize_prompt
    text_config.tokenize_prompt = lambda *args: _host(BatchEncoding)(tokenize(*args))

    def position_ids(inputs, index):
        assert not inputs.uploaded
        return config.QwenImagePipelineConfig.get_text_encoder_position_ids(
            text_config, inputs, index
        )

    text_config.get_text_encoder_position_ids = position_ids
    encoder = _PositionEncoder().eval()
    with patch(_GLOBAL_ARGS_PATCH, return_value=MagicMock()):
        stage = TextEncodingStage(text_encoders=[encoder], tokenizers=[object()])
    stage._begin_text_encoder_use = MagicMock()
    stage._text_encode_dp_group = MagicMock(return_value=None)
    stage.encode_text("hi", make_server_args(pipeline_config=text_config), device="cpu")
    assert encoder.position_ids.tolist() == [[0, 1]]


class _VisionLanguageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.positions = torch.arange(70).view(1, 1, -1).expand(3, 1, -1)
        self.model = SimpleNamespace(get_rope_index=Mock(side_effect=self._rope_index))

    def _rope_index(self, input_ids, image_grid_thw, video_grid_thw, attention_mask):
        assert input_ids.device.type == "cpu" and not self.inputs.uploaded
        return self.positions, None

    def forward(self, input_ids, position_ids=None, **kwargs):
        self.position_ids = position_ids
        return SimpleNamespace(hidden_states=(input_ids[..., None].float(),))


def test_image_stage_passes_host_position_ids():
    encoder = _VisionLanguageEncoder()
    ids = torch.arange(70)[None]
    encoder.inputs = _host(BatchFeature)(
        {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.zeros(4, 3),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
    )
    with patch(_GLOBAL_ARGS_PATCH, return_value=MagicMock()):
        stage = ImageEncodingStage(
            Mock(return_value=encoder.inputs), text_encoder=encoder
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
    pipeline_config = config.QwenImageEditPlusPipelineConfig()
    stage.forward(batch, SimpleNamespace(pipeline_config=pipeline_config))
    assert torch.equal(encoder.position_ids, encoder.positions)
    assert batch.prompt_embeds[0].shape[1] == 70 - 64  # the template prefix dropped
