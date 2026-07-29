import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.wan import t5_postprocess_text


def _make_inputs(batch: int = 1, seq: int = 16, dim: int = 8):
    hidden = torch.randn(batch, seq, dim)
    mask = torch.zeros(batch, seq, dtype=torch.long)
    mask[:, :5] = 1
    return hidden, mask


class TestT5PostprocessText(unittest.TestCase):
    def test_uses_encoder_output_mask(self):
        hidden, mask = _make_inputs()
        outputs = BaseEncoderOutput(last_hidden_state=hidden, attention_mask=mask)
        embeds = t5_postprocess_text(outputs, _text_inputs=None)
        self.assertEqual(embeds.shape, (1, 512, 8))
        torch.testing.assert_close(embeds[0, :5], hidden[0, :5])
        self.assertTrue((embeds[0, 5:] == 0).all())

    def test_falls_back_to_tokenizer_mask(self):
        # Plain transformers ModelOutput objects carry no attention_mask; the
        # tokenizer inputs hold the same mask.
        hidden, mask = _make_inputs()
        outputs = SimpleNamespace(last_hidden_state=hidden)
        text_inputs = SimpleNamespace(attention_mask=mask)
        embeds = t5_postprocess_text(outputs, text_inputs)
        self.assertEqual(embeds.shape, (1, 512, 8))
        torch.testing.assert_close(embeds[0, :5], hidden[0, :5])


if __name__ == "__main__":
    unittest.main()
