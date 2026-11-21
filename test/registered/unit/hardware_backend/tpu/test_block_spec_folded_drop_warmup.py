"""The block spec worker warms the folded-row drop's gather feeding the draft KV
writer at every extend token bucket when encoder SWA replay folds. CPU only."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.experimental.speculative import block_spec_worker as bsw

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

HIDDEN = 6


def _worker(*, folds=True, ships_pd_hidden=False):
    policy = object.__new__(DecoderTpuCompilePolicy)
    policy.t_bucketer = SimpleNamespace(buckets=[128, 512])
    policy.live_cache_loc_dtype = lambda: torch.int32
    calls = []

    class Draft:
        def _compile_policy(self):
            return SimpleNamespace(dflash_target_hidden_size=HIDDEN)

        def draft_context(self):
            return contextlib.nullcontext()

        def write_draft_kv(self, **kwargs):
            calls.append(kwargs)

    worker = object.__new__(bsw.TpuBlockSpecWorker)
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(folds_encoder_swa_replay=folds),
            compile_policy=policy,
            dtype=torch.bfloat16,
        )
    )
    worker._block_draft_worker = Draft()
    worker.device = "cpu"
    worker._pd_hidden_pool = object() if ships_pd_hidden else None
    return worker, calls


def _exec(replay):
    return patch.object(
        bsw,
        "get_exec",
        lambda: SimpleNamespace(
            features=SimpleNamespace(enable_encoder_swa_bounded_replay=replay)
        ),
    )


class TestFoldedDropDraftKvWarmup(CustomTestCase):
    def test_writes_gathered_hidden_at_every_token_bucket(self):
        worker, calls = _worker()
        with (
            _exec(True),
            patch.object(
                bsw.TpuBlockSpecWorker, "ships_pd_hidden", property(lambda self: False)
            ),
        ):
            worker._warmup_folded_drop_draft_kv()
        self.assertEqual(
            [c["target_hidden"].shape for c in calls], [(128, 6), (512, 6)]
        )
        for call in calls:
            rows = call["target_hidden"].shape[0]
            self.assertEqual(call["target_hidden"].dtype, torch.bfloat16)
            self.assertEqual(call["positions"].shape, (rows,))
            self.assertEqual(call["cache_loc"].dtype, torch.int32)
            self.assertEqual(int(call["cache_loc"].abs().sum()), 0)

    def test_skipped_without_a_folding_replay(self):
        for replay, folds in ((False, True), (True, False)):
            worker, calls = _worker(folds=folds)
            with (
                self.subTest(replay=replay, folds=folds),
                _exec(replay),
                patch.object(
                    bsw.TpuBlockSpecWorker,
                    "ships_pd_hidden",
                    property(lambda self: False),
                ),
            ):
                worker._warmup_folded_drop_draft_kv()
                self.assertEqual(calls, [])

    def test_skipped_on_a_pd_prefill_shipping_hidden(self):
        worker, calls = _worker()
        with (
            _exec(True),
            patch.object(
                bsw.TpuBlockSpecWorker, "ships_pd_hidden", property(lambda self: True)
            ),
        ):
            worker._warmup_folded_drop_draft_kv()
        self.assertEqual(calls, [])


class TestEngramCommitIdsWarmup(CustomTestCase):
    def _worker(self, hasher):
        worker = object.__new__(bsw.TpuBlockSpecWorker)
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                ngram_embedding_manager=SimpleNamespace(engram_hasher=hasher)
            )
        )
        worker.num_verify_rows = 6
        worker._bucket_sizes = lambda: (8, 16)
        worker._warmup_accept = lambda pbs, width: SimpleNamespace(
            bonus_tokens=torch.zeros(pbs, dtype=torch.int64),
            accept_tokens=torch.full((pbs, width), -1, dtype=torch.int64),
            accept_lens=torch.zeros(pbs, dtype=torch.int64),
        )
        return worker

    def test_builds_the_commit_ids_at_every_decode_bucket(self):
        shapes = []
        real = bsw.committed_verify_ids

        def record(bonus, accept):
            out = real(bonus, accept)
            shapes.append(tuple(out.shape))
            return out

        with patch.object(bsw, "committed_verify_ids", record):
            self._worker(object())._warmup_engram_commit_ids()
        self.assertEqual(shapes, [(8, 7), (16, 7)])

    def test_skipped_without_engram(self):
        with patch.object(bsw, "committed_verify_ids", side_effect=AssertionError):
            self._worker(None)._warmup_engram_commit_ids()


if __name__ == "__main__":
    unittest.main()
