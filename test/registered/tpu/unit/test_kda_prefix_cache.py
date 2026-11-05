"""KDA radix checkpoints: the fused op's snapshot equals the state a request
reaches when it restores that checkpoint and replays the remaining tokens.

KDA analogue of test_gdn_prefix_cache.py, on the Kimi-K3 call shape (fused
[q, k, v, g] projection, packed f32 kernel weights, bounded per-channel gate).
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit", chips=1)

# Kimi-K3 KDA per rank, fewer heads: d=128, short conv 4, gate lower bound -5.
NK = NV = 4
D, K, LOWER = 128, 4, -5.0
DIM = NK * D * 2 + NV * D
T = 193  # one 64-token tile past the 128-token checkpoint, plus a ragged tail


def _ids(x):
    return torch.tensor(x, dtype=torch.int32, device="tpu")


class TestKDAPrefixCache(CustomTestCase):
    def setUp(self):
        from sglang_tpu.attention_gdn_pallas import build_gdn_pallas_op
        from sglang_tpu.kda_layout import pack_kda_weights

        self.op = build_gdn_pallas_op()
        torch.manual_seed(7)
        conv_w = torch.randn(DIM, 1, K) * 0.3
        dt_bias = torch.randn(NV, D) * 0.5
        a_log = torch.randn(NV) * 0.5
        self.weights = [
            conv_w.to("tpu"),
            None,
            a_log.to("tpu"),
            dt_bias.to("tpu"),
        ]
        self.packed = pack_kda_weights(conv_w, dt_bias, a_log, NV, D).to("tpu")
        bf = torch.bfloat16
        # Fused [q, k, v, g]: the op reads only the leading conv-dim columns.
        self.qkvg = (torch.randn(T, DIM + NV * D) * 0.5).to(bf).to("tpu")
        self.beta = torch.randn(T, NV).to(bf).to("tpu")
        self.gate = torch.randn(T, NV * D).to(bf).to("tpu")

    def states(self, rec_dtype=torch.float32, slots=7):
        return (
            torch.zeros(slots, K - 1, 1, DIM, dtype=torch.bfloat16, device="tpu"),
            torch.zeros(slots, NV, D, D, dtype=rec_dtype, device="tpu"),
        )

    def run_op(self, states, start, end, slot, *, prefix=0, track=None, decode=False):
        return self.op(
            self.qkvg[start:end].contiguous(),
            self.beta[start:end].contiguous(),
            self.gate[start:end].contiguous(),
            *states,
            *self.weights,
            _ids([0, end - start]),
            _ids([slot]),
            _ids([1 if decode else 0, 1 if decode else 0, 1]),
            _ids([prefix + end - start]),
            NK,
            NV,
            D,
            D,
            K,
            True,
            LOWER,
            batched_only=decode,
            prefill_only=not decode,
            fused_qkvg=True,
            packed_weights=self.packed,
            track_indices=_ids([track[0]]) if track else None,
            track_end=_ids([track[1]]) if track else None,
        )

    def _copy_slot(self, states, src, dsts):
        from sglang_tpu.mamba_pool import TpuMambaPool, TpuPerLayerState

        pool = TpuMambaPool.__new__(TpuMambaPool)
        pool.device = "tpu"
        pool.checkpoint_width = 1
        pool.mamba_cache = TpuPerLayerState(conv=[[states[0]]], temporal=[states[1]])
        pool.copy_from(torch.tensor([src] * len(dsts)), torch.tensor(dsts))

    def test_prefill_snapshot_restore_matches_full_prefill(self):
        # Kimi-K3 runs packed weights; f32 and bf16 recurrent state.
        for rec_dtype in (torch.float32, torch.bfloat16):
            with self.subTest(rec_dtype=rec_dtype):
                full, reference = self.states(rec_dtype), self.states(rec_dtype)
                # Full prefill into slot 1, snapshot after 128 tokens into slot 2.
                output = self.run_op(full, 0, T, 1, track=(2, 128))
                untracked = self.run_op(reference, 0, T, 5)
                # Tracking must not perturb the output or the active state.
                torch.testing.assert_close(
                    output.cpu(), untracked.cpu(), atol=0, rtol=0
                )
                for a, b in zip(full, reference):
                    torch.testing.assert_close(a[1].cpu(), b[5].cpu(), atol=0, rtol=0)
                # The snapshot is the state after 128 tokens, not after T.
                self.run_op(reference, 0, 128, 1)
                for actual, expected in zip(full, reference):
                    torch.testing.assert_close(
                        actual[2].cpu(), expected[1].cpu(), atol=0, rtol=0
                    )
                # Radix hit: restore the checkpoint into a fresh slot and replay
                # the remaining tokens; output and final state equal the full
                # prefill's. A second branch decodes from the same snapshot.
                snapshot = [t[2].cpu().clone() for t in full]
                self._copy_slot(full, 2, [3, 4])
                resumed = self.run_op(full, 128, T, 3, prefix=128)
                torch.testing.assert_close(
                    resumed.cpu().float(),
                    output[128:].cpu().float(),
                    atol=2e-2,
                    rtol=2e-2,
                )
                self.run_op(full, 128, 129, 4, prefix=128, decode=True)
                for i, (state, saved) in enumerate(zip(full, snapshot)):
                    torch.testing.assert_close(state[2].cpu(), saved, atol=0, rtol=0)
                    # Chunk tiling restarts at the checkpoint, so the restored
                    # path agrees to kernel precision, not bitwise.
                    got, want = state[3].cpu().float(), state[1].cpu().float()
                    rel = (got - want).norm() / want.norm()
                    self.assertLess(rel.item(), 2e-2, f"state {i} rel {rel}")

    def test_final_decode_and_disabled_snapshots(self):
        for length, decode in [(64, False), (1, True)]:
            with self.subTest(length=length):
                states = self.states()
                self.run_op(states, 0, length, 1, track=(2, length), decode=decode)
                for state in states:
                    torch.testing.assert_close(
                        state[2].cpu(), state[1].cpu(), atol=0, rtol=0
                    )
                saved = [t[2].cpu().clone() for t in states]
                self.run_op(states, 0, length, 1, track=(2, -1), decode=decode)
                for state, expected in zip(states, saved):
                    torch.testing.assert_close(state[2].cpu(), expected, atol=0, rtol=0)

    def test_two_sequence_prefill_snapshots(self):
        # bs=2 extend: each row snapshots at its own boundary into its own slot.
        full, reference = self.states(), self.states()
        self.op(
            self.qkvg[:T].contiguous(),
            self.beta[:T].contiguous(),
            self.gate[:T].contiguous(),
            *full,
            *self.weights,
            _ids([0, 129, T]),
            _ids([1, 2]),
            _ids([0, 0, 2]),
            _ids([129, T - 129]),
            NK,
            NV,
            D,
            D,
            K,
            True,
            LOWER,
            prefill_only=True,
            fused_qkvg=True,
            packed_weights=self.packed,
            track_indices=_ids([3, 4]),
            track_end=_ids([128, -1]),
        )
        self.run_op(reference, 0, 128, 3)
        self.run_op(reference, 129, T, 2)
        for actual, expected in zip(full, reference):
            torch.testing.assert_close(
                actual[3].cpu(), expected[3].cpu(), atol=0, rtol=0
            )
            torch.testing.assert_close(
                actual[2].cpu(), expected[2].cpu(), atol=0, rtol=0
            )
            self.assertEqual(torch.count_nonzero(actual[4].cpu()).item(), 0)


if __name__ == "__main__":
    unittest.main()
