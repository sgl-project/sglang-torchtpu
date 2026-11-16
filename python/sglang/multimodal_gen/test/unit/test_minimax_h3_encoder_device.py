import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.models.encoders import minimax_h3_qwen3vl
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3Qwen3VLEncoder,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLModel


class TestMiniMaxH3EncoderDevice(unittest.TestCase):
    """`device` must name the compute side, not the parameter storage side.

    Component offload stores parameters on CPU between uses. Input construction
    must still target the accelerator selected for the forward rather than infer
    the compute device from an offloaded parameter.
    """

    def _encoder_with_param_on(self, device: torch.device) -> MiniMaxH3Qwen3VLEncoder:
        encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
        torch.nn.Module.__init__(encoder)
        encoder.register_parameter(
            "offloaded", torch.nn.Parameter(torch.zeros(1, device=device))
        )
        return encoder

    def test_device_ignores_cpu_offloaded_parameters(self):
        encoder = self._encoder_with_param_on(torch.device("cpu"))
        compute_device = torch.device("cuda", 3)

        with mock.patch.object(
            minimax_h3_qwen3vl, "get_local_torch_device", return_value=compute_device
        ):
            self.assertEqual(encoder.device, compute_device)

        # The parameter really is on CPU: the property is not just echoing it back.
        self.assertEqual(next(encoder.parameters()).device.type, "cpu")

    def test_device_follows_local_device_on_cpu_only_platforms(self):
        encoder = self._encoder_with_param_on(torch.device("cpu"))
        cpu = torch.device("cpu")

        with mock.patch.object(
            minimax_h3_qwen3vl, "get_local_torch_device", return_value=cpu
        ):
            self.assertEqual(encoder.device, cpu)


class _PositionsSeen(Exception):
    pass


class _LanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 8)

    def forward(self, *, position_ids, **kwargs):
        raise _PositionsSeen(position_ids)


class TestMiniMaxH3EncoderTextPositions(unittest.TestCase):
    def test_text_only_positions_match_the_model_fallback(self):
        """Text-only positions resolved on the host, before the transfer, equal
        the ones Qwen3VLModel.forward computes on device when given none."""
        model = Qwen3VLModel.__new__(Qwen3VLModel)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2),
            image_token_id=30,
            video_token_id=31,
            vision_start_token_id=29,
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
        )
        model.rope_deltas = None
        model.language_model = _LanguageModel()
        ids = torch.randint(0, 29, (1, 11))
        with self.assertRaises(_PositionsSeen) as fallback:
            model(input_ids=ids, attention_mask=torch.ones_like(ids))
        encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
        torch.nn.Module.__init__(encoder)
        encoder.model = model
        with (
            mock.patch.object(
                minimax_h3_qwen3vl, "get_local_torch_device", return_value=ids.device
            ),
            self.assertRaises(_PositionsSeen) as host,
        ):
            encoder.encode_ids(ids[0])
        torch.testing.assert_close(host.exception.args[0], fallback.exception.args[0])


if __name__ == "__main__":
    unittest.main()
