import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.models.encoders import minimax_h3_qwen3vl
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3Qwen3VLEncoder,
)


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

    def test_text_positions_resolve_on_host_without_changing_values(self):
        encoder = self._encoder_with_param_on(torch.device("cpu"))
        encoder.hidden_dim = 8
        encoder.conditioning_projection = None
        config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2),
            image_token_id=100,
            video_token_id=101,
            vision_start_token_id=102,
        )
        position_model = SimpleNamespace(config=config)
        resolve = mock.Mock(
            side_effect=lambda *a, **kw: minimax_h3_qwen3vl.Qwen3VLModel.get_rope_index(
                position_model, *a, **kw
            )
        )
        encoder.model = SimpleNamespace(get_rope_index=resolve)
        ids = torch.tensor([10, 13, 3, 20], dtype=torch.long)
        hidden = torch.arange(32).reshape(1, 4, 8).to(torch.bfloat16)
        with (
            mock.patch.object(
                minimax_h3_qwen3vl,
                "get_local_torch_device",
                return_value=torch.device("cpu"),
            ),
            mock.patch.object(
                MiniMaxH3Qwen3VLEncoder,
                "__call__",
                return_value=SimpleNamespace(last_hidden_state=hidden),
            ) as forward,
        ):
            output = encoder.encode_ids(ids)
        torch.testing.assert_close(output, hidden[0], rtol=0, atol=0)
        resolve.assert_called_once()
        self.assertEqual(resolve.call_args.args[0].device.type, "cpu")
        torch.testing.assert_close(forward.call_args.kwargs["input_ids"], ids[None])
        torch.testing.assert_close(
            forward.call_args.kwargs["position_ids"],
            torch.arange(4).view(1, 1, 4).expand(3, 1, 4),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
