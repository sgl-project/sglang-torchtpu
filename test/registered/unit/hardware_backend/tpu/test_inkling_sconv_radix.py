"""Inkling2 sconv radix snapshots on the TPU sconv backend's XLA path (CPU-runnable).

A radix hit copies a track slot into a new request's slot and runs only the
uncached suffix, so the snapshot must be exactly the conv state a request has
after the cached prefix; anything else silently changes the suffix's outputs.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_inkling_sconv_pallas import TpuInklingSconvBackend
from sglang_tpu.envs import envs

_K, _DIM, _SLOTS = 4, 8, 8


class _Backend(TpuInklingSconvBackend):
    """The sconv backend over one stream of one layer, without a ModelRunner."""

    def __init__(self, state: torch.Tensor):  # noqa: D107  (skips the op build)
        self.req_to_token_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(mamba_cache=SimpleNamespace(conv=[[state]])),
            mamba2_layer_index=lambda layer_id: 0,
        )
        self._num_spec_tokens = 0
        self._spec_window = 1
        self._read_offsets = None

    def run(self, x, weight, *, lens, seq_lens, slots, track=None, decode=False):
        """One forward of packed x; track is (track slot, end offset in the request)."""
        qsl = torch.tensor(
            [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32
        )
        self._query_start_loc = qsl
        self._seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        self._state_indices = torch.tensor(slots, dtype=torch.int32)
        self._is_decode = decode
        track = track or [(0, -1)] * len(lens)
        self._track_indices = torch.tensor([t[0] for t in track], dtype=torch.int32)
        self._track_end = torch.tensor(
            [
                -1 if end < 0 else qsl[i].item() + end
                for i, (_, end) in enumerate(track)
            ],
            dtype=torch.int32,
        )
        with envs.SGLANG_TPU_ENABLE_INKLING_SCONV_PALLAS.override(False):
            return self.run_sconv(x, layer_id=0, stream=0, weight=weight)


class TestInklingSconvRadixSnapshot(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.state = torch.zeros(_SLOTS, _K - 1, 1, _DIM)
        self.backend = _Backend(self.state)
        self.weight = torch.randn(_DIM, 1, _K)
        self.x = torch.randn(11, _DIM)

    def test_prefix_hit_resumes_from_the_snapshot(self):
        """Request A prefills 6 tokens, then extends 6..10 with a boundary at 8, so
        the window (tokens 5-7) starts in the prior state. B hits that prefix and
        must reproduce A's outputs and final state; C is a fresh 8-token prefill."""
        x, w, state = self.x, self.weight, self.state
        filler = torch.randn(4, _DIM)
        self.backend.run(
            torch.cat((x[:8], x[:6])), w, lens=[8, 6], seq_lens=[8, 6], slots=[2, 1]
        )
        y_a = self.backend.run(
            torch.cat((filler, x[6:])),
            w,
            lens=[4, 5],
            seq_lens=[4, 11],
            slots=[4, 1],
            track=[(0, -1), (5, 2)],
        )
        torch.testing.assert_close(state[5], state[2])
        state[3] = state[5]
        y_b = self.backend.run(x[8:], w, lens=[3], seq_lens=[11], slots=[3])
        torch.testing.assert_close(y_b, y_a[6:])
        torch.testing.assert_close(state[3], state[1])

    def test_decode_snapshot_is_the_post_token_state(self):
        """A tracked decode row snapshots the state after its token; a row without
        a snapshot leaves its track slot and the null slot unchanged."""
        x, w, state = self.x, self.weight, self.state
        self.backend.run(x[:8], w, lens=[8], seq_lens=[8], slots=[1])
        state[6].fill_(7.0)
        self.backend.run(
            x[8:10],
            w,
            lens=[1, 1],
            seq_lens=[9, 1],
            slots=[1, 2],
            track=[(5, 1), (6, -1)],
            decode=True,
        )
        torch.testing.assert_close(state[5], state[1])
        self.assertTrue(torch.all(state[0] == 0))
        self.assertTrue(torch.all(state[6] == 7.0))

    def test_verify_takes_no_extend_snapshot(self):
        """A verify's window end includes unaccepted drafts; its snapshot comes from
        the accepted checkpoint after acceptance, never from the forward itself."""
        self.backend._num_spec_tokens = 1
        self.backend._track_end = torch.tensor([2], dtype=torch.int32)
        self.assertIsNone(self.backend._track_snapshot(self.x[:2], self.state))


if __name__ == "__main__":
    unittest.main()
