"""The pipeline hand-off carries the model's declared row width.

Qwen4-Exp hands its hc_count hyper-connection streams on side by side in one
hidden row, so a stage sends [T, hc_count * hidden_size]; the wave buffer used
to be sized at hidden_size.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch  # noqa: E402

from sglang_tpu.pp_wave import PPWave, pp_transfer_width  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 64
HC = 4


def _wave(hidden):
    return PPWave(
        rank=1,
        size=4,
        rows=16,
        hidden=hidden,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        launcher=lambda x: x,
    )


class TestPPTransferWidth(unittest.TestCase):
    def test_default_is_hidden_size(self):
        model = SimpleNamespace()
        config = SimpleNamespace(hidden_size=HIDDEN)
        self.assertEqual(pp_transfer_width(model, config), HIDDEN)

    def test_model_declares_a_wider_row(self):
        model = SimpleNamespace(pp_transfer_hidden_size=HC * HIDDEN)
        config = SimpleNamespace(hidden_size=HIDDEN)
        self.assertEqual(pp_transfer_width(model, config), HC * HIDDEN)

    def test_wave_pads_hc_wide_rows_at_the_declared_width(self):
        streams = torch.randn(5, HC * HIDDEN).to(torch.bfloat16)
        # The streams carry the residual themselves: hidden_states only.
        merged = _wave(HC * HIDDEN).merge({"hidden_states": streams})
        self.assertEqual(tuple(merged.shape), (16, HC * HIDDEN))
        self.assertTrue(torch.equal(merged[:5], streams))
        self.assertEqual(float(merged[5:].abs().sum()), 0.0)
        # A buffer sized at hidden_size cannot carry them.
        with self.assertRaises(RuntimeError):
            _wave(HIDDEN).merge({"hidden_states": streams})

    def test_qwen4_exp_declares_hc_wide_rows(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration

        model = SimpleNamespace(
            config=SimpleNamespace(
                text_config=SimpleNamespace(hc_count=HC, hidden_size=2560)
            )
        )
        width = Qwen4ExpForConditionalGeneration.pp_transfer_hidden_size.fget(model)
        self.assertEqual(width, HC * 2560)


if __name__ == "__main__":
    unittest.main()
