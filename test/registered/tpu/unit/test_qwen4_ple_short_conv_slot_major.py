"""The TPU PLE short conv over slot-major state matches core's math.

short_conv_slot_major reads and writes TpuShortConvPool's [slots, state_len, 1,
channels] state with index_select / index_copy_ instead of core's view writes.
Checked against core's general-path formula over [slots, channels, state_len].
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang_tpu.hooks.qwen4_exp import short_conv_slot_major
from sglang_tpu.ple_state_pool import TpuShortConvPool

C, K, NGRAM, SLOTS = 12, 4, 3, 6
L = (K - 1) * NGRAM  # state_len


def _reference(state, conv_w, x, batch):
    """Core's general _short_conv over a [slots, C, L] state; returns (out, state)."""
    state = state.clone()
    st = state.index_select(0, batch.state_indices)
    padded = x.new_zeros((batch.lengths.shape[0], batch.row_width, C))
    padded[batch.req_indices, batch.token_offsets] = x
    conv_input = torch.cat([st, padded.transpose(1, 2)], dim=-1)
    out = F.conv1d(conv_input, conv_w, dilation=NGRAM, groups=C).transpose(1, 2)
    cols = torch.arange(L)
    nxt = conv_input.gather(
        2, (batch.lengths.unsqueeze(1) + cols).unsqueeze(1).expand(-1, C, -1)
    )
    state[batch.state_indices] = nxt
    return F.silu(out[batch.req_indices, batch.token_offsets]), state


class TestShortConvSlotMajor(unittest.TestCase):
    def test_matches_core_math_and_writes_slot_rows(self):
        torch.manual_seed(0)
        pool = TpuShortConvPool(
            size=SLOTS - 1, state_shape=(C, L), layer_ids=[1], dtype=torch.float32, device="cpu"
        )
        logical = torch.randn(SLOTS, C, L)
        pool.layer_states[0].copy_(logical.transpose(1, 2).unsqueeze(2))
        lengths = torch.tensor([3, 1])
        req_indices = torch.tensor([0, 0, 0, 1])
        token_offsets = torch.tensor([0, 1, 2, 0])
        batch = SimpleNamespace(
            lengths=lengths,
            row_width=3,
            req_indices=req_indices,
            token_offsets=token_offsets,
            state_indices=torch.tensor([4, 2]),
            use_decode_fast_path=False,
            mode=SimpleNamespace(is_target_verify=lambda: False),
        )
        conv = torch.nn.Conv1d(C, C, K, groups=C, bias=False)
        ple = SimpleNamespace(
            layer_id=1,
            conv_channels=C,
            conv1d=conv,
            short_conv_dilation=NGRAM,
            short_conv_state_len=L,
        )
        x = torch.randn(4, C)
        want_out, want_state = _reference(logical, conv.weight.detach(), x, batch)
        fake_module = SimpleNamespace(
            get_req_to_token_pool=lambda: SimpleNamespace(short_conv_pool=pool),
            _ple_track_targets=lambda forward_batch, batch: None,
        )
        with patch.dict("sys.modules", {"sglang.srt.models.qwen4_exp": fake_module}), patch(
            "sglang.srt.models.qwen4_exp", fake_module, create=True
        ):
            got = short_conv_slot_major(None, ple, x, None, batch)
        torch.testing.assert_close(got, want_out)
        torch.testing.assert_close(
            pool.layer_states[0][:, :, 0, :].transpose(1, 2), want_state
        )


if __name__ == "__main__":
    unittest.main()
