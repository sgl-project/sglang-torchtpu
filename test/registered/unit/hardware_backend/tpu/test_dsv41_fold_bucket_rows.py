"""DeepSeek-V4.1 TPU extend work stays at the token bucket: the folded-row drop
keeps the hidden's row count (the spec workers' draft KV write never runs at a live
extend length), the fold keeps its row maps and slots on the host, and an idle DP
rank builds its dummy extend on the host. CPU only."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor import encoder_swa_replay
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.experimental.sampling.sampling_batch_info import (
    HostPaddedSamplingInfo,
)
from sglang_tpu.models.deepseek_v41 import hooks
from sglang_tpu.models.deepseek_v41.compile_policy import (
    DeepseekV4DecoderTpuCompilePolicy,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MAX_CTX = 2048
HIDDEN = 8
# (req slot, cached prefix, new tokens): a miss, a full 128-row replay, and a
# 64-token prefix replayed from 0.
REQS = [(1, 0, 40), (2, 512, 30), (3, 64, 20)]
HITS = [(1, 384, 512), (2, 0, 64)]
BUCKET = 512


class _DeviceSlots:
    """Stands in for device req_pool_indices: reading a slot back fails."""

    def __init__(self, slots):
        self.slots = slots

    def __getitem__(self, key):
        raise AssertionError("the fold read a device slot back")


def _scheduled_batch():
    req_to_token = torch.arange(4 * MAX_CTX, dtype=torch.int32).view(4, MAX_CTX)
    rows = [(s, p, p + n) for s, p, n in REQS]
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        reqs=[
            SimpleNamespace(
                full_untruncated_fill_ids=[1000 * (s + 1) + q for q in range(p + n)]
            )
            for s, p, n in REQS
        ],
        req_pool_indices=_DeviceSlots([s for s, _, _ in REQS]),
        req_pool_indices_cpu=torch.tensor([s for s, _, _ in REQS]),
        prefix_lens=[p for _, p, _ in REQS],
        extend_lens=[n for _, _, n in REQS],
        extend_num_tokens=sum(n for _, _, n in REQS),
        input_ids=torch.tensor(
            [1000 * (s + 1) + q for s, a, b in rows for q in range(a, b)]
        ),
        out_cache_loc=torch.tensor(
            [int(req_to_token[s, q]) for s, a, b in rows for q in range(a, b)],
            dtype=torch.int32,
        ),
        prefill_input_ids_cpu=None,
        extend_logprob_start_lens=None,
        engram_history=None,
    )
    runner = SimpleNamespace(
        device="cpu",
        model=None,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool=None,
    )
    return batch, runner


def _fold():
    """The V4.1 TPU fold of the scheduled batch (the hook around _fold_batch)."""
    batch, runner = _scheduled_batch()
    rows = list(HITS)
    hook_runner = SimpleNamespace(**vars(runner))
    tpu_platform = SimpleNamespace(is_tpu=True)
    pool_cls = type("Pool", (), {})
    hook_runner.token_to_kv_pool = pool_cls()
    from unittest.mock import patch

    with (
        patch("sglang.srt.runtime_context.get_platform", return_value=tpu_platform),
        patch(
            "sglang_tpu.models.deepseek_v41.memory_pool.DSV41TPUTokenToKVPool",
            pool_cls,
        ),
    ):
        return hooks.fold_tpu_dsv41_encoder_replay(
            encoder_swa_replay._fold_batch, batch=batch, runner=hook_runner, rows=rows
        )


class TestFoldKeepsHostRowMaps(CustomTestCase):
    def test_row_maps_stay_on_the_host_and_out_cache_loc_keeps_its_dtype(self):
        folded = _fold()
        self.assertEqual(folded.keep_rows.device.type, "cpu")
        self.assertEqual(folded.row_floor.device.type, "cpu")
        self.assertEqual(folded.batch.out_cache_loc.dtype, torch.int32)
        self.assertEqual(folded.num_rows, 90 + 128 + 64)
        # Slots came from the host mirror; the folded batch keeps the device ones.
        self.assertIsInstance(folded.batch.req_pool_indices, _DeviceSlots)


class TestDropFoldedRowsAtBucket(CustomTestCase):
    def _hidden(self, rows):
        torch.manual_seed(0)
        return torch.randn(rows, HIDDEN)

    def test_bucket_hidden_keeps_its_rows_and_matches_the_live_drop(self):
        """The scheduled rows lead, bit-identical to the core drop; the tail
        repeats the last row and the row count stays the bucket."""
        folded = _fold()
        hidden = self._hidden(BUCKET)
        live = LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)
        encoder_swa_replay.drop_folded_rows(logits_output=live, folded=folded)
        padded = LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)
        hooks.drop_tpu_dsv41_folded_rows(
            encoder_swa_replay.drop_folded_rows, logits_output=padded, folded=folded
        )
        n = folded.keep_rows.shape[0]
        self.assertEqual(live.hidden_states.shape, (n, HIDDEN))
        self.assertEqual(padded.hidden_states.shape, (BUCKET, HIDDEN))
        self.assertTrue(torch.equal(padded.hidden_states[:n], live.hidden_states))
        self.assertTrue(
            torch.equal(
                padded.hidden_states[n:],
                hidden[BUCKET - 1].expand(BUCKET - n, HIDDEN),
            )
        )

    def test_hidden_at_the_folded_rows_keeps_that_count(self):
        folded = _fold()
        hidden = self._hidden(folded.num_rows)
        out = LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)
        hooks.drop_tpu_dsv41_folded_rows(
            encoder_swa_replay.drop_folded_rows, logits_output=out, folded=folded
        )
        n = folded.keep_rows.shape[0]
        self.assertEqual(out.hidden_states.shape[0], folded.num_rows)
        self.assertTrue(torch.equal(out.hidden_states[:n], hidden[folded.keep_rows]))

    def test_tail_indices_and_no_hidden_take_the_core_path(self):
        folded = _fold()
        calls = []

        def original(**kwargs):
            calls.append(kwargs)

        out = LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=self._hidden(BUCKET),
            hidden_states_token_indices=torch.arange(4),
        )
        hooks.drop_tpu_dsv41_folded_rows(original, logits_output=out, folded=folded)
        out = LogitsProcessorOutput(next_token_logits=None, hidden_states=None)
        hooks.drop_tpu_dsv41_folded_rows(original, logits_output=out, folded=folded)
        self.assertEqual(len(calls), 2)


class TestDropIndexBounds(CustomTestCase):
    """Every gather index stays inside the hidden, whatever the fold's row count:
    whole-admitted hits up to the bucket (769-896 new + 128 replay rows), a
    folded extend past the bucket, and multi-request batches."""

    def _check(self, reqs, hits, rows):
        batch, runner = _scheduled_batch()
        req_to_token = runner.req_to_token_pool.req_to_token
        loc_rows = [(s, p, p + n) for s, p, n in reqs]
        batch.reqs = [
            SimpleNamespace(full_untruncated_fill_ids=list(range(p + n)))
            for s, p, n in reqs
        ]
        batch.req_pool_indices = _DeviceSlots([s for s, _, _ in reqs])
        batch.req_pool_indices_cpu = torch.tensor([s for s, _, _ in reqs])
        batch.prefix_lens = [p for _, p, _ in reqs]
        batch.extend_lens = [n for _, _, n in reqs]
        batch.extend_num_tokens = sum(batch.extend_lens)
        batch.input_ids = torch.cat([torch.arange(a, b) for _, a, b in loc_rows])
        batch.out_cache_loc = torch.cat(
            [req_to_token[s, a:b] for s, a, b in loc_rows]
        ).to(torch.int32)
        hook_runner = SimpleNamespace(**vars(runner))
        pool_cls = type("Pool", (), {})
        hook_runner.token_to_kv_pool = pool_cls()
        from unittest.mock import patch

        with (
            patch(
                "sglang.srt.runtime_context.get_platform",
                return_value=SimpleNamespace(is_tpu=True),
            ),
            patch(
                "sglang_tpu.models.deepseek_v41.memory_pool.DSV41TPUTokenToKVPool",
                pool_cls,
            ),
        ):
            folded = hooks.fold_tpu_dsv41_encoder_replay(
                encoder_swa_replay._fold_batch,
                batch=batch,
                runner=hook_runner,
                rows=hits,
            )
        self.assertLess(int(folded.keep_rows.max()), folded.num_rows)
        hidden = torch.arange(rows, dtype=torch.float32)[:, None]
        out = LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)
        hooks.drop_tpu_dsv41_folded_rows(
            encoder_swa_replay.drop_folded_rows, logits_output=out, folded=folded
        )
        return folded, hidden, out.hidden_states

    def test_whole_admitted_hit_fills_the_bucket(self):
        # 896 new + 128 replay = 1024 folded rows in a 1024 bucket.
        folded, hidden, out = self._check([(2, 512, 896)], [(0, 384, 512)], 1024)
        self.assertEqual(folded.num_rows, 1024)
        self.assertEqual(out.shape[0], 1024)
        self.assertTrue(torch.equal(out[:896, 0], torch.arange(128, 1024).float()))

    def test_three_requests_with_two_hits(self):
        reqs = [(1, 0, 256), (2, 512, 300), (3, 64, 200)]
        folded, hidden, out = self._check(reqs, [(1, 384, 512), (2, 0, 64)], 1024)
        n = folded.keep_rows.shape[0]
        self.assertEqual(n, 756)
        self.assertTrue(torch.equal(out[:n], hidden[folded.keep_rows]))
        self.assertTrue(bool((out[n:, 0] == 1023).all()))

    def test_folded_rows_past_the_hidden_take_the_core_path(self):
        # 1000 new + 128 replay = 1128 rows; a 1024-row hidden is left alone,
        # as core does (no gather at all).
        folded, hidden, out = self._check([(2, 512, 1000)], [(0, 384, 512)], 1024)
        self.assertEqual(folded.num_rows, 1128)
        self.assertIs(out, hidden)


class _Batch(SimpleNamespace):
    pass


def _policy(*, is_draft_worker=False):
    policy = object.__new__(DeepseekV4DecoderTpuCompilePolicy)
    policy.model_runner = SimpleNamespace(is_draft_worker=is_draft_worker)
    return policy


def _host_padded():
    return object.__new__(HostPaddedSamplingInfo)


class TestIdleRankDummyOnHost(CustomTestCase):
    def _idle_batch(self, sampling_info):
        meta = torch.empty(0, dtype=torch.int64, device="meta")
        return _Batch(
            forward_mode=ForwardMode.EXTEND,
            sampling_info=sampling_info,
            seq_lens=meta,
            seq_lens_cpu=None,
            input_ids=meta,
            out_cache_loc=meta.to(torch.int32),
            req_pool_indices=meta,
            req_pool_indices_cpu=None,
        )

    def _materialize(self, sampling_info):
        policy = _policy()
        policy._dummy_out_cache_loc_dtype = lambda mode: torch.int32
        fb = self._idle_batch(sampling_info)
        self.assertTrue(policy.materialize_dp_idle_extend(fb, 877))
        return fb

    def test_host_padded_idle_rank_builds_its_dummy_on_the_host(self):
        """The busiest rank's live length reaches the device only through the
        bucket upload, never as an on-device arange / zeros."""
        fb = self._materialize(_host_padded())
        for name in (
            "input_ids",
            "positions",
            "out_cache_loc",
            "seq_lens",
            "req_pool_indices",
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_start_loc",
        ):
            with self.subTest(name=name):
                self.assertEqual(getattr(fb, name).device.type, "cpu")
        self.assertTrue(torch.equal(fb.positions, torch.arange(877)))
        self.assertEqual(fb.out_cache_loc.dtype, torch.int32)
        self.assertEqual(int(fb.out_cache_loc.abs().sum()), 0)

    def test_other_idle_ranks_keep_the_lengths_device(self):
        fb = self._materialize(None)
        self.assertEqual(fb.positions.device.type, "meta")

    def test_the_composed_dspark_policy_keeps_the_host_dummy(self):
        """The DSpark server composes the block-spec mixin over this policy."""
        from sglang_tpu.experimental.compilation.block_compile_policy import (
            TpuBlockSpecDecodeCompilePolicyMixin,
        )

        composed = type(
            "SpecPolicy",
            (TpuBlockSpecDecodeCompilePolicyMixin, DeepseekV4DecoderTpuCompilePolicy),
            {},
        )
        self.assertIs(
            composed._dp_idle_extend_device,
            DeepseekV4DecoderTpuCompilePolicy._dp_idle_extend_device,
        )


if __name__ == "__main__":
    unittest.main()
