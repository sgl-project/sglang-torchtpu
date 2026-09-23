import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.models import kimi_k25
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _forward(*, decode_in_graph: bool, is_decode: bool):
    """Call the wrapper's forward on a stand-in self with a mocked LM."""
    language_model = MagicMock(return_value="lm-out")
    fake_self = SimpleNamespace(
        decode_embeds_in_language_model=decode_in_graph,
        language_model=language_model,
        get_image_feature=MagicMock(),
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: is_decode)
    )
    input_ids = torch.tensor([1, 2, 3])
    positions = torch.tensor([0, 1, 2])
    with patch.object(
        kimi_k25, "general_mm_embed_routine", return_value="routine-out"
    ) as routine:
        out = KimiK25ForConditionalGeneration.forward(
            fake_self, input_ids, positions, forward_batch
        )
    return out, language_model, routine, input_ids, positions, forward_batch


class TestKimiDecodeEmbedsInGraph(CustomTestCase):
    def test_decode_hands_raw_input_ids_to_the_language_model(self):
        out, lm, routine, ids, pos, fb = _forward(decode_in_graph=True, is_decode=True)
        self.assertEqual(out, "lm-out")
        routine.assert_not_called()
        lm.assert_called_once_with(
            input_ids=ids, positions=pos, forward_batch=fb, pp_proxy_tensors=None
        )

    def test_prefill_still_uses_the_embed_routine(self):
        out, lm, routine, *_ = _forward(decode_in_graph=True, is_decode=False)
        self.assertEqual(out, "routine-out")
        routine.assert_called_once()
        lm.assert_not_called()

    def test_opt_out_keeps_decode_on_the_routine(self):
        out, lm, routine, *_ = _forward(decode_in_graph=False, is_decode=True)
        self.assertEqual(out, "routine-out")
        routine.assert_called_once()
        lm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
