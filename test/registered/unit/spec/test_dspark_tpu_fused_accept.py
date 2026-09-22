"""The fused TPU accept graphs must match the eager accept chain.

The TPU path folds argmax + correct-draft count + cap + finalize + out-token
build into two compiled graphs, and in doing so collapses two derivations the
eager chain keeps apart: the bonus token (masked sum when uncapped, gathered at
the capped index otherwise) and the cap itself (skipped when there is no ragged
layout, fed a full-width sentinel here). Both collapses are only valid under
reasoning, so they are pinned against the eager reference.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    accept_greedy,
    finalize_accept_lens,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import build_out_tokens
from sglang.srt.hardware_backend.tpu.dspark_utils_tpu import (
    DsparkTpuAcceptor,
    dspark_accept_greedy_tpu,
    dspark_finalize_accept_tpu,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_VOCAB = 32
_WIDTH = 6
_FIELDS = (
    "correct_len",
    "bonus",
    "cap_trim_lens",
    "commit_lens",
    "new_seq_lens",
    "out_tokens",
)


def _sweep_inputs():
    """One row per reachable accept length, so every branch of the run is hit."""
    torch.manual_seed(0)
    bs = _WIDTH
    target_predict = torch.randint(0, _VOCAB, (bs, _WIDTH), dtype=torch.int64)
    candidates = torch.randint(0, _VOCAB, (bs, _WIDTH), dtype=torch.int64)
    for row in range(bs):
        accept_len = row  # 0 .. _WIDTH - 1, the last row accepts the whole block
        candidates[row, 1 : 1 + accept_len] = target_predict[row, :accept_len]
        if accept_len < _WIDTH - 1:
            candidates[row, 1 + accept_len] = (
                target_predict[row, accept_len] + 1
            ) % _VOCAB
    target_logits = torch.nn.functional.one_hot(
        target_predict.reshape(-1), num_classes=_VOCAB
    ).to(torch.float32)
    prefix_lens = torch.arange(100, 100 + bs, dtype=torch.int64)
    return candidates, target_logits, prefix_lens


def _reference(*, candidates, target_logits, verify_lens, prefix_lens):
    correct_len, bonus, cap_trim_lens = accept_greedy(
        candidates=candidates,
        target_logits=target_logits,
        verify_num_draft_tokens=_WIDTH,
        cutoff_verify_lens=verify_lens,
    )
    finalized = finalize_accept_lens(
        correct_len=correct_len, cap_trim_lens=cap_trim_lens, prefix_lens=prefix_lens
    )
    out_tokens = build_out_tokens(
        draft_tokens=candidates[:, 1:],
        correct_len=correct_len,
        bonus=bonus,
        verify_num_draft_tokens=_WIDTH,
        gamma=_WIDTH - 1,
    )
    return (
        correct_len,
        bonus,
        finalized.cap_trim_lens,
        finalized.commit_lens,
        finalized.new_seq_lens,
        out_tokens,
    )


class TestDsparkTpuFusedAccept(CustomTestCase):
    def _assert_same(self, fused, reference):
        for name, got, want in zip(_FIELDS, fused, reference):
            with self.subTest(field=name):
                self.assertEqual(got.dtype, want.dtype)
                torch.testing.assert_close(got, want)

    def test_graphs_match_eager_chain(self):
        candidates, target_logits, prefix_lens = _sweep_inputs()
        bs = candidates.shape[0]
        # Row r accepts r drafts, so a budget of r caps it one short.
        binding = torch.arange(bs, dtype=torch.int64).clamp(min=1)
        for label, verify_lens in (
            ("no layout", None),
            ("uniform budget", torch.full((bs,), _WIDTH, dtype=torch.int64)),
            ("binding cap", binding),
        ):
            with self.subTest(verify_lens=label):
                packed = dspark_accept_greedy_tpu(
                    candidates,
                    target_logits,
                    (
                        torch.full((bs,), _WIDTH, dtype=torch.int64)
                        if verify_lens is None
                        else verify_lens
                    ),
                )
                fused = dspark_finalize_accept_tpu(
                    packed, prefix_lens, candidates[:, 1:]
                )
                self._assert_same(
                    fused,
                    _reference(
                        candidates=candidates,
                        target_logits=target_logits,
                        verify_lens=verify_lens,
                        prefix_lens=prefix_lens,
                    ),
                )

    def test_binding_cap_actually_trims(self):
        """Without this the cap cases above would pass on a no-op cap."""
        candidates, target_logits, prefix_lens = _sweep_inputs()
        bs = candidates.shape[0]
        # Row r accepts r drafts, so a budget of r caps it one short.
        binding = torch.arange(bs, dtype=torch.int64).clamp(min=1)
        packed = dspark_accept_greedy_tpu(candidates, target_logits, binding)
        self.assertGreater(int(packed[2].sum()), 0)

    def test_padded_bucket_matches_unpadded(self):
        """Rows past the live batch must not leak into the trimmed outputs."""
        candidates, target_logits, prefix_lens = _sweep_inputs()
        bs = candidates.shape[0]
        pad_bs = 2 * bs
        padded_candidates = torch.randint(
            0, _VOCAB, (pad_bs, _WIDTH), dtype=torch.int64
        )
        padded_candidates[:bs] = candidates
        padded_logits = torch.randn(pad_bs * _WIDTH, _VOCAB)
        padded_logits[: bs * _WIDTH] = target_logits
        # Entry padding hands the acceptor bucket-shaped rows; the ghosts carry
        # junk rather than zeros so a leak would show up as a mismatch.
        padded_prefix_lens = torch.randint(0, 999, (pad_bs,), dtype=torch.int64)
        padded_prefix_lens[:bs] = prefix_lens
        padded_draft_tokens = torch.randint(
            0, _VOCAB, (pad_bs, _WIDTH - 1), dtype=torch.int64
        )
        padded_draft_tokens[:bs] = candidates[:, 1:]

        acceptor = DsparkTpuAcceptor(
            device=torch.device("cpu"),
            verify_num_draft_tokens=_WIDTH,
            tp_sync=SpecTpSync(SimpleNamespace(world_size=1, rank_in_group=0)),
        )
        fused = acceptor.accept(
            candidates=padded_candidates,
            target_logits=padded_logits,
            verify_lens=None,
            prefix_lens=padded_prefix_lens,
            draft_tokens=padded_draft_tokens,
        )
        self._assert_same(
            tuple(t[:bs] for t in fused),
            _reference(
                candidates=candidates,
                target_logits=target_logits,
                verify_lens=None,
                prefix_lens=prefix_lens,
            ),
        )


if __name__ == "__main__":
    unittest.main()
