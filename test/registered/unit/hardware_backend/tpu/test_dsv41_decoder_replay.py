# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for decoder SWA bounded replay (CED prefill) on the DeepSeek-V4.1
TPU backend: the row plan, its DP bucket, the decoder rows' metadata and the
padded gather of the decoder rows."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsv4.attn_metadata_kernels import (
    late_layer_tail_layout,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import attention_backend as backend
from sglang_tpu.models.deepseek_v41.decoder_replay import (
    DecoderTail,
    decoder_rows_bucket,
    plan_decoder_rows,
    plan_tail_rows,
    scheduled_decoder_plan,
    scheduled_decoder_rows,
)
from sglang_tpu.models.deepseek_v41.hooks import check_tpu_dsv41_decoder_replay

W = 128


def _kept_positions(*, prompt: int, chunks, rule: str) -> list:
    """Absolute positions that run the decoder over a chunked prompt."""
    kept, start = [], 0
    for n in chunks:
        plan = plan_decoder_rows(
            starts=[start],
            extend_lens=[n],
            fill_lens=[prompt],
            floors=[0 if start else 0],
            rule=rule,
        )
        kept.extend(range(start + n - plan.kept[0], start + n))
        start += n
    assert start == prompt
    return kept


class TestRowPlan(CustomTestCase):
    def test_prompt_rule_keeps_the_prompt_tail_whatever_the_chunking(self):
        prompt = 1000
        want = list(range(prompt - W, prompt))
        for chunks in ([1000], [256] * 3 + [232], [300, 300, 300, 100], [960, 40]):
            with self.subTest(chunks=chunks):
                self.assertEqual(
                    _kept_positions(prompt=prompt, chunks=chunks, rule="prompt"), want
                )

    def test_prompt_rule_skips_chunks_before_the_tail(self):
        plan = plan_decoder_rows(
            starts=[0, 256],
            extend_lens=[256, 256],
            fill_lens=[1000, 1000],
            floors=[0, 0],
            rule="prompt",
        )
        self.assertEqual(plan.kept, [0, 0])
        self.assertEqual(plan.replay_starts, [872, 872])

    def test_short_final_chunk_reads_the_previous_chunk(self):
        # The final chunk holds 40 rows; the previous one supplies 88 of the window.
        plan = plan_decoder_rows(
            starts=[0, 960],
            extend_lens=[960, 40],
            fill_lens=[1000, 1000],
            floors=[0, 0],
            rule="prompt",
        )
        self.assertEqual(plan.kept, [88, 40])
        self.assertEqual(plan.replay_starts, [872, 872])

    def test_chunk_rule_is_core_late_layer_tail_layout(self):
        starts, ext = [0, 300, 7], [40, 500, 128]
        seq = [s + n for s, n in zip(starts, ext)]
        plan = plan_decoder_rows(
            starts=starts,
            extend_lens=ext,
            fill_lens=seq,
            floors=[0, 0, 0],
            rule="chunk",
        )
        _, tail_lens, floor = late_layer_tail_layout(
            extend_lens_cpu=ext,
            seq_lens_cpu=seq,
            tail_len=W,
            device=torch.device("cpu"),
        )
        self.assertEqual(plan.kept, tail_lens)
        per_row = [r for r, n in zip(plan.replay_starts, tail_lens) for _ in range(n)]
        self.assertEqual(per_row, floor.tolist())

    def test_fold_floor_and_short_prompts(self):
        # A hit folded from 384 (prefix 512) with 30 new tokens: its window is the
        # prompt's last 128, all inside the folded rows. A 40-token prompt keeps all.
        plan = plan_decoder_rows(
            starts=[384, 0],
            extend_lens=[158, 40],
            fill_lens=[542, 40],
            floors=[384, 0],
            rule="prompt",
        )
        self.assertEqual(plan.kept, [128, 40])
        self.assertEqual(plan.replay_starts, [414, 0])

    def test_extend_past_the_fill_length_raises(self):
        with self.assertRaises(ValueError):
            plan_decoder_rows(
                starts=[0], extend_lens=[10], fill_lens=[5], floors=[0], rule="prompt"
            )


def _scheduled(prefix, extend, reset, fill):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        prefix_lens=prefix,
        extend_lens=extend,
        encoder_swa_reset=reset,
        decoder_fill_lens=fill,
        reqs=[None] * len(prefix),
    )


def _folded_rows(batch, *, bucket=None):
    """The backend's plan of the folded batch (as encoder_swa_replay._fold_batch
    lays it out: a reset hit starts at its replay start and floors there)."""
    starts, ext, floors = [], [], []
    for p, n, reset in zip(
        batch.prefix_lens, batch.extend_lens, batch.encoder_swa_reset
    ):
        r = min(p, 128) if (reset and p) else 0
        starts.append(p - r)
        ext.append(n + r)
        floors.append(p - r if r else 0)
    counts = torch.tensor(ext)
    qsl = torch.nn.functional.pad(torch.cumsum(counts, 0), (1, 0))
    positions = torch.cat(
        [torch.arange(s, s + n) for s, n in zip(starts, ext)]
        + [torch.zeros(5, dtype=torch.int64)]
    )
    valid = torch.arange(positions.shape[0]) < int(qsl[-1])
    return plan_tail_rows(
        fill_lens=batch.decoder_fill_lens,
        query_start_loc=qsl,
        positions=positions,
        valid=valid,
        floors=torch.tensor(floors),
        bucket=bucket,
    )


class TestSchedulerCountMatchesBackend(CustomTestCase):
    CASES = [
        # (prefix, extend, reset, fill)
        (
            [0, 512, 64, 300],
            [40, 30, 20, 256],
            [True, True, True, False],
            [40, 542, 84, 2000],
        ),
        ([0, 0], [1024, 512], [True, True], [5000, 512]),
        ([2048], [100], [False], [2148]),
        ([1024, 256], [1024, 64], [False, True], [None, None]),
    ]

    def test_mlp_sync_count_is_the_folded_backend_plan(self):
        for prefix, extend, reset, fill in self.CASES:
            fill = None if fill[0] is None else fill
            batch = _scheduled(prefix, extend, reset, fill)
            with self.subTest(prefix=prefix, extend=extend):
                self.assertEqual(
                    scheduled_decoder_rows(batch, folds=True),
                    _folded_rows(batch).num_real,
                )

    def test_unfolded_replay_starts_match_the_folded_ones_on_extend_rows(self):
        # The DSpark worker masks its draft ring rows with the unfolded plan.
        batch = _scheduled(
            [0, 512, 64], [40, 30, 20], [True, True, True], [40, 542, 84]
        )
        unfolded = scheduled_decoder_plan(batch, folds=False)
        folded = _folded_rows(batch)
        for p, n, a, b in zip(
            batch.prefix_lens,
            batch.extend_lens,
            unfolded.replay_starts,
            folded.replay_starts.tolist(),
        ):
            pos = torch.arange(p, p + n)
            self.assertTrue(torch.equal(pos >= a, pos >= b))


class TestScheduledCount(CustomTestCase):
    def test_decode_rank_counts_one_row_per_request(self):
        """A peer's extend turns a decode batch into a one-row-per-request extend
        (MAX_LEN padding); every such row is a decoder row."""
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, batch_size=lambda: 5)
        self.assertEqual(scheduled_decoder_rows(batch, folds=True), 5)
        self.assertEqual(scheduled_decoder_rows(None, folds=True), 0)
        idle = SimpleNamespace(forward_mode=ForwardMode.IDLE, batch_size=lambda: 0)
        self.assertEqual(scheduled_decoder_rows(idle, folds=True), 0)

    def test_row_rule_comes_from_the_tpu_env(self):
        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.models.deepseek_v41.decoder_replay import decoder_row_rule

        self.assertEqual(decoder_row_rule(), "chunk")
        with tpu_envs.SGLANG_TPU_DSV41_DECODER_REPLAY_ROWS.override("prompt"):
            self.assertEqual(decoder_row_rule(), "prompt")
        with tpu_envs.SGLANG_TPU_DSV41_DECODER_REPLAY_ROWS.override("bogus"):
            with self.assertRaises(ValueError):
                decoder_row_rule()


class TestBucketAndTailRows(CustomTestCase):
    def test_bucket_table(self):
        cases = {
            (0, 1024): 0,
            (1, 1024): 128,
            (128, 1024): 128,
            (129, 1024): 256,
            (512, 1024): 512,
            (513, 1024): 1024,
            (300, 512): 512,
            (100, 128): 128,
            (1024, 1024): 1024,
        }
        for (rows, t), want in cases.items():
            with self.subTest(rows=rows, t=t):
                self.assertEqual(decoder_rows_bucket(rows, t), want)
        with self.assertRaises(ValueError):
            decoder_rows_bucket(1025, 1024)

    def test_padding_rows_gather_one_row_the_decoder_does_not_keep(self):
        batch = _scheduled([0, 0], [200, 50], [True, True], [200, 50])
        plan = _folded_rows(batch, bucket=256)
        self.assertEqual(plan.num_real, 178)
        self.assertEqual(plan.gather[:128].tolist(), list(range(72, 200)))
        self.assertEqual(plan.gather[128:178].tolist(), list(range(200, 250)))
        # The last token-bucket padding row (T = 250 + 5): core's scatter puts
        # the padding rows back there, where nothing reads them.
        self.assertEqual(plan.gather[178:].unique().tolist(), [254])
        self.assertEqual(plan.counts.tolist(), [128, 50])
        self.assertEqual(plan.replay_starts.tolist(), [72, 0])

    def test_rows_over_the_shared_bucket_raise(self):
        batch = _scheduled([0, 0], [300, 300], [True, True], [300, 300])
        with self.assertRaises(RuntimeError):
            _folded_rows(batch, bucket=128)

    def test_dummy_extend_rank_has_only_padding_rows(self):
        # materialize_dp_idle_extend: one request, every out_cache_loc 0.
        qsl = torch.tensor([0, 512])
        plan = plan_tail_rows(
            fill_lens=None,
            query_start_loc=qsl,
            positions=torch.arange(512),
            valid=torch.zeros(512, dtype=torch.bool),
            floors=torch.zeros(1),
            bucket=256,
        )
        self.assertEqual(plan.num_real, 0)
        self.assertEqual(plan.counts.tolist(), [0])
        self.assertEqual(plan.gather.shape[0], 256)

    def test_warmup_compiles_padding_rows(self):
        plan = plan_tail_rows(
            fill_lens=None,
            query_start_loc=torch.tensor([0, 512, 1024]),
            positions=torch.arange(1024),
            valid=torch.ones(1024, dtype=torch.bool),
            floors=torch.zeros(2),
            bucket=128,
            warmup=True,
        )
        self.assertEqual((plan.num_real, plan.gather.shape[0]), (0, 128))
        self.assertEqual(plan.counts.tolist(), [0, 0])


class TestBucketPicker(CustomTestCase):
    def _pick(self, *, dp, rows, bucket=None, mode=ForwardMode.EXTEND):
        from unittest.mock import patch

        from sglang_tpu.models.deepseek_v41 import compile_policy

        policy = object.__new__(compile_policy.DeepseekV4DecoderTpuCompilePolicy)
        policy.model_runner = SimpleNamespace(
            attn_backend=SimpleNamespace(decoder_replay=True)
        )
        fb = SimpleNamespace(
            forward_mode=mode,
            decoder_rows_bucket=bucket,
            global_decoder_rows=rows,
            input_ids=torch.zeros(1024, dtype=torch.int64),
        )
        with patch.object(
            compile_policy,
            "get_parallel",
            return_value=SimpleNamespace(attn_dp_size=dp),
        ):
            policy._pick_decoder_rows_bucket(fb)
        return fb.decoder_rows_bucket

    def test_dp_max_of_the_mlp_sync_rows(self):
        self.assertEqual(self._pick(dp=8, rows=[0, 130, 5, 0, 0, 0, 0, 0]), 256)
        self.assertEqual(self._pick(dp=8, rows=[0] * 8), 0)

    def test_dummy_without_counts_takes_the_token_bucket_under_dp(self):
        self.assertEqual(self._pick(dp=8, rows=None), 1024)

    def test_preset_bucket_and_other_modes_are_left_alone(self):
        self.assertEqual(self._pick(dp=8, rows=[3] * 8, bucket=512), 512)
        self.assertIsNone(self._pick(dp=8, rows=[3] * 8, mode=ForwardMode.DECODE))


class TestDecoderTail(CustomTestCase):
    def test_rows_then_core_scatter_restores_kept_rows(self):
        from sglang.srt.models.deepseek_v4 import _scatter_tail_rows

        batch = _scheduled([0, 0], [200, 50], [True, True], [200, 50])
        plan = _folded_rows(batch, bucket=256)
        tail = DecoderTail(gather=plan.gather.int(), positions=None, num_tokens=255)
        x = torch.randn(255, 8)
        back = _scatter_tail_rows(tail=tail, rows=tail.rows(x), num_tokens=255)
        kept = torch.zeros(255, dtype=torch.bool)
        kept[plan.gather[: plan.num_real]] = True
        self.assertTrue(torch.equal(back[kept], x[kept]))
        self.assertEqual(tail.num_rows, 256)
        self.assertIsNone(tail.rows(None))

    def test_scatter_hook_zero_fills_the_full_layout(self):
        """SGLANG_TPU_DSV41_TAIL_GATHER_SCATTER=0 (no inverse map): a scatter."""
        from sglang.srt.models import deepseek_v4 as core_model
        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            scatter_decoder_rows,
        )

        tail = DecoderTail(
            gather=torch.tensor([2, 0, 3], dtype=torch.int32),
            positions=None,
            num_tokens=4,
        )
        rows = torch.tensor([[1.0], [2.0], [9.0]])
        out = scatter_decoder_rows(
            core_model._scatter_tail_rows, tail=tail, rows=rows, num_tokens=4
        )
        self.assertEqual(out[:, 0].tolist(), [2.0, 0.0, 1.0, 9.0])
        # Any other tail goes to core.
        calls = []
        other = SimpleNamespace()
        scatter_decoder_rows(
            lambda **kw: calls.append(kw), tail=other, rows=rows, num_tokens=4
        )
        self.assertIs(calls[0]["tail"], other)

    def test_gather_hook_zero_fills_the_full_layout(self):
        """The default path: the rows come back by a gather over the inverse
        map; padding rows (past num_real) go nowhere."""
        from sglang.srt.models import deepseek_v4 as core_model
        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            scatter_decoder_rows,
            tail_inverse_rows,
        )

        gather = torch.tensor([2, 0, 3], dtype=torch.int32)
        tail = DecoderTail(
            gather=gather,
            positions=None,
            num_tokens=4,
            inverse=tail_inverse_rows(gather, 2, 4),
        )
        self.assertEqual(tail.inverse.tolist(), [1, -1, 0, -1])
        rows = torch.tensor([[1.0], [2.0], [9.0]])
        out = scatter_decoder_rows(
            core_model._scatter_tail_rows, tail=tail, rows=rows, num_tokens=4
        )
        self.assertEqual(out[:, 0].tolist(), [2.0, 0.0, 1.0, 0.0])

    def test_gather_matches_scatter_except_on_padding_targets(self):
        """SGLANG_TPU_DSV41_TAIL_GATHER_SCATTER=0 (scatter) and the default
        (gather) agree on every row but the spare row padding scatters onto."""
        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            _scatter_zeroed,
            tail_inverse_rows,
        )

        batch = _scheduled([0, 0], [200, 50], [True, True], [200, 50])
        plan = _folded_rows(batch, bucket=256)
        self.assertLess(plan.num_real, 256)
        gather = plan.gather.int()
        scatter_tail = DecoderTail(gather=gather, positions=None, num_tokens=255)
        gather_tail = DecoderTail(
            gather=gather,
            positions=None,
            num_tokens=255,
            inverse=tail_inverse_rows(plan.gather, plan.num_real, 255),
        )
        rows = torch.randn(256, 3, 8)
        scattered = _scatter_zeroed(scatter_tail, rows, 255)
        gathered = _scatter_zeroed(gather_tail, rows, 255)
        padding_targets = torch.zeros(255, dtype=torch.bool)
        padding_targets[plan.gather[plan.num_real :]] = True
        self.assertTrue(bool(padding_targets.any()))
        self.assertTrue(
            torch.equal(scattered[~padding_targets], gathered[~padding_targets])
        )
        self.assertFalse(bool(gathered[padding_targets].any()))

    def test_default_path_issues_no_device_scatter(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            _scatter_zeroed,
            tail_inverse_rows,
        )

        class Ops(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.names = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.names.append(func.overloadpacket.__name__)
                return func(*args, **(kwargs or {}))

        def scatters(names):
            return [
                n
                for n in names
                if "index_put" in n or "scatter" in n or n.startswith("index_copy")
            ]

        batch = _scheduled([0, 0], [200, 50], [True, True], [200, 50])
        plan = _folded_rows(batch, bucket=256)
        gather = plan.gather.int()
        inverse = tail_inverse_rows(plan.gather, plan.num_real, 255)
        rows = torch.randn(256, 8)
        with Ops() as ops:
            _scatter_zeroed(
                DecoderTail(
                    gather=gather, positions=None, num_tokens=255, inverse=inverse
                ),
                rows,
                255,
            )
        self.assertIn("index_select", ops.names)
        self.assertEqual(scatters(ops.names), [])
        # The env=0 path does scatter (the check would catch one).
        with Ops() as ops:
            _scatter_zeroed(
                DecoderTail(gather=gather, positions=None, num_tokens=255), rows, 255
            )
        self.assertNotEqual(scatters(ops.names), [])

    def test_head_reads_the_identity_tail_after_the_model_scattered_back(self):
        """DeepseekV4ForCausalLM's compact-tail branch reads the backend's tail
        after the model; once scatter_dspark_capture put the rows back in the
        full layout it gets the identity, and only for that forward."""
        from unittest.mock import patch

        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            FullLayoutTail,
            clear_full_layout_head,
        )

        be = object.__new__(backend.DSV41TPUAttnBackend)
        be.forward_metadata = None
        # Read as the causal-LM forward does (decoder_replay._model_forward).
        be.in_model_forward = True
        fb = SimpleNamespace(
            extend_seq_lens=torch.tensor([3, 2]), extend_seq_lens_cpu=[3, 2]
        )
        be.head_full_layout = fb
        tail = be.tail_forward_metadata.late_layer_tail
        self.assertIsInstance(tail, FullLayoutTail)
        ids = torch.arange(5)
        self.assertIs(tail.rows(ids), ids)
        self.assertEqual(tail.extend_seq_lens_cpu, [3, 2])
        self.assertIsNone(tail.token_indices)
        with patch(
            "sglang.srt.model_executor.forward_context.get_attn_backend",
            return_value=be,
        ):
            self.assertEqual(clear_full_layout_head(lambda lm: "out", None), "out")
        self.assertIsNone(be.head_full_layout)
        self.assertIsNone(be.tail_forward_metadata.late_layer_tail)

    def test_capture_without_a_tail_still_gives_the_head_the_identity(self):
        """A DSpark-capture forward without a DecoderTail (bucket 0, warmup dec0,
        an extend without replay) must not leave the head reading a None tail."""
        from unittest.mock import patch

        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            FullLayoutTail,
            scatter_dspark_capture,
        )

        be = object.__new__(backend.DSV41TPUAttnBackend)
        be.forward_metadata = None
        # Inside the causal-LM forward, whose head reads the identity tail.
        be.in_model_forward = True
        model = SimpleNamespace(dspark_layers_to_capture=[2])
        fb = SimpleNamespace(extend_seq_lens=None, extend_seq_lens_cpu=None)
        out = ((torch.ones(3, 2), torch.ones(3, 4)), [torch.ones(3, 2)])
        with patch(
            "sglang.srt.model_executor.forward_context.get_attn_backend",
            return_value=be,
        ):
            got = scatter_dspark_capture(
                lambda *a, **k: out, model, torch.arange(3), None, fb
            )
        self.assertIs(got, out)
        self.assertIs(be.head_full_layout, fb)
        # A stale identity never reaches the next forward's layer loop.
        seen = []

        def loop(*a, **k):
            seen.append(be.tail_forward_metadata.late_layer_tail)
            return out

        with patch(
            "sglang.srt.model_executor.forward_context.get_attn_backend",
            return_value=be,
        ):
            scatter_dspark_capture(loop, model, torch.arange(3), None, fb)
        self.assertEqual(seen, [None])
        self.assertIsInstance(be.tail_forward_metadata.late_layer_tail, FullLayoutTail)

    def test_dp_wide_ids_take_the_regathered_decoder_rows(self):
        """Core takes tail.rows(input_ids) then tail.rows(input_ids_global): the
        second (another row count) is the DP-wide gather of the first's rows."""
        from unittest.mock import patch

        gather = torch.tensor([3, 1, 0], dtype=torch.int32)
        fb = object()
        tail = DecoderTail(
            gather=gather,
            positions=None,
            num_tokens=4,
            backend=SimpleNamespace(tail_forward_batch=fb),
        )
        seen = []

        def global_ids(ids, forward_batch):
            seen.append((ids.tolist(), forward_batch))
            return ids.repeat(8)

        with patch.object(DecoderTail, "global_ids", staticmethod(global_ids)):
            with self.assertRaises(RuntimeError):
                tail.rows(torch.arange(32))
            local = tail.rows(torch.tensor([10, 11, 12, 13]))
            out = tail.rows(torch.arange(32))
        self.assertEqual(local.tolist(), [13, 11, 10])
        self.assertEqual(seen, [([13, 11, 10], fb)])
        self.assertEqual(out.shape[0], 24)
        # Hidden rows (floating) are never taken for DP-wide ids.
        self.assertEqual(tail.rows(torch.randn(4, 2)).shape, (3, 2))

    def test_global_ids_are_regathered_for_the_decoder_bucket_under_dp(self):
        """The MoE routing reads one id per DP-gathered row: inside the decoder
        the gather spans the bucket on every rank, not the full batch."""
        from unittest.mock import patch

        calls = []

        def gather(out, local, fb):
            calls.append((tuple(out.shape), tuple(local.shape)))
            out.copy_(local.repeat(out.shape[0] // local.shape[0], 1))

        ids = torch.arange(128)
        none_a2a = SimpleNamespace(is_none=lambda: True)
        with (
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=8),
            ),
            patch("sglang.srt.layers.moe.get_moe_a2a_backend", return_value=none_a2a),
            patch(
                "sglang.srt.layers.dp_attention.get_global_dp_buffer_len",
                return_value=8 * 128,
            ),
            patch("sglang.srt.layers.dp_attention.dp_gather_replicate", gather),
        ):
            out = DecoderTail.global_ids(ids, object())
        self.assertEqual(calls, [((1024, 1), (128, 1))])
        self.assertEqual(tuple(out.shape), (1024,))
        with patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(attn_dp_size=1),
        ):
            self.assertIs(DecoderTail.global_ids(ids, object()), ids)


def _forward_batch(*, prefix, extend, fill, bucket, slots):
    positions = torch.cat([torch.arange(p, p + n) for p, n in zip(prefix, extend)])
    num_tokens = positions.shape[0] + 3  # token-bucket padding
    positions = torch.cat([positions, torch.zeros(3, dtype=torch.int64)])
    locs = torch.cat(
        [
            torch.arange(p, p + n) + 4096 * (s + 1)
            for p, n, s in zip(prefix, extend, slots)
        ]
    )
    return SimpleNamespace(
        batch_size=len(prefix),
        positions=positions,
        out_cache_loc=torch.cat([locs, torch.zeros(3, dtype=torch.int64)]),
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens_cpu=list(extend),
        req_pool_indices=torch.tensor(slots),
        encoder_swa_row_floor=None,
        decoder_fill_lens=fill,
        decoder_rows_bucket=bucket,
    ), num_tokens


class TestTailMetadata(CustomTestCase):
    """Expected rows below are the prompt rule's; test_chunk_rule_keeps_each_
    extends_last_window checks the default chunk rule."""

    def setUp(self):
        from sglang_tpu.envs import envs as tpu_envs

        ctx = tpu_envs.SGLANG_TPU_DSV41_DECODER_REPLAY_ROWS.override("prompt")
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)

    def test_chunk_rule_keeps_each_extends_last_window(self):
        from sglang_tpu.envs import envs as tpu_envs

        with tpu_envs.SGLANG_TPU_DSV41_DECODER_REPLAY_ROWS.override("chunk"):
            fb, num_tokens, md = self._build(
                prefix=[0, 1000],
                extend=[200, 300],
                fill=[200, 1500],
                bucket=256,
                slots=[1, 2],
            )
        tail = md.tail
        # Request 1 is mid-prompt but keeps its chunk's last 128 rows, floored there.
        self.assertEqual(tail.query_start_loc.tolist(), [0, 128, 256])
        self.assertEqual(tail.positions[128:256].tolist(), list(range(1172, 1300)))
        self.assertEqual(tail.swa_minimum.tolist(), [72, 1172])

    def _build(self, request_window=None, **kw):
        fb, num_tokens = _forward_batch(**kw)
        req_to_token = torch.arange(8 * 4096).view(8, 4096)
        return (
            fb,
            num_tokens,
            backend.build_forward_metadata(
                fb,
                req_to_token=req_to_token,
                full_to_swa_index_mapping=None,
                request_window=request_window,
                swa_ring_valid_from=None,
                max_context_len=4096,
                low_ratios=(1, 2),
                device=torch.device("cpu"),
                decoder_replay=True,
                decoder_low_ratios=(1,),
            ),
        )

    def test_tail_metadata_describes_the_decoder_rows(self):
        fb, num_tokens, md = self._build(
            prefix=[0, 1000],
            extend=[200, 300],
            fill=[200, 1500],
            bucket=256,
            slots=[1, 2],
        )
        tail = md.tail
        # Request 0 keeps 72..199; request 1 (fill 1500) is mid-prompt: no rows.
        self.assertEqual(tail.query_start_loc.tolist(), [0, 128, 128])
        self.assertEqual(tail.positions[:128].tolist(), list(range(72, 200)))
        self.assertEqual(int(tail.valid.sum()), 128)
        self.assertFalse(bool(tail.valid[128:].any()))
        self.assertFalse(bool(tail.compress_valid.any()))
        self.assertEqual(tail.swa_minimum.tolist(), [72, 1372])
        self.assertEqual(
            tail.out_cache_loc[:128].tolist(), fb.out_cache_loc[72:200].tolist()
        )
        self.assertEqual(tail.token_rows[:128].unique().tolist(), [0])
        self.assertEqual(set(tail.low_ratio), {1})
        self.assertEqual(tail.low_ratio[1].lengths[:128].tolist(), list(range(73, 201)))
        self.assertIs(tail.full_pages, md.full_pages)
        self.assertEqual(md.tail_gather.shape[0], 256)
        # Padding rows gather the last token-bucket padding row.
        self.assertEqual(md.tail_gather[128:].unique().tolist(), [num_tokens - 1])
        self.assertEqual(set(md.low_ratio), {1, 2})

    def test_inverse_map_is_built_by_default_and_off_with_env_0(self):
        from sglang_tpu.envs import envs as tpu_envs

        _, num_tokens, md = self._build(
            prefix=[0, 1000],
            extend=[200, 300],
            fill=[200, 1500],
            bucket=256,
            slots=[1, 2],
        )
        inverse = md.tail_inverse
        self.assertEqual(inverse.dtype, torch.int32)
        self.assertEqual(inverse.shape[0], num_tokens)
        real = int(md.tail.valid.sum())
        mapped = (inverse >= 0).nonzero().flatten()
        self.assertEqual(mapped.numel(), real)
        self.assertTrue(
            torch.equal(md.tail_gather[inverse[mapped].long()].long(), mapped)
        )
        with tpu_envs.SGLANG_TPU_DSV41_TAIL_GATHER_SCATTER.override(False):
            _, _, md = self._build(
                prefix=[0, 1000],
                extend=[200, 300],
                fill=[200, 1500],
                bucket=256,
                slots=[1, 2],
            )
        self.assertIsNone(md.tail_inverse)

    def test_no_kept_rows_run_the_smallest_bucket_on_padding(self):
        _, num_tokens, md = self._build(
            prefix=[0], extend=[512], fill=[4000], bucket=128, slots=[1]
        )
        self.assertEqual(md.tail_rows, 128)
        self.assertFalse(bool(md.tail.valid.any()))
        self.assertEqual(md.tail_gather.unique().tolist(), [num_tokens - 1])

    def test_request_window_commits_only_decoder_rows(self):
        window = SimpleNamespace(workspace_first_page=4 + 4 * 8, workspace_pages=64)
        fb, _, md = self._build(
            request_window=window,
            prefix=[0, 896],
            extend=[200, 64],
            fill=[200, 960],
            bucket=256,
            slots=[1, 2],
        )
        plan = md.tail.swa_window_plan
        commit = plan.commit_token.view(2, W)
        ok = plan.commit_dst.view(2, W) >= 0
        # Each request commits its decoder rows, as rows of the tail batch.
        self.assertEqual(commit[0][ok[0]].tolist(), list(range(0, 128)))
        self.assertEqual(commit[1][ok[1]].tolist(), list(range(128, 192)))
        # Request 1 continues: history [832, 896) from its ring, floored at 832.
        hist_ok = plan.history_dst.view(2, W)[1] >= 0
        self.assertEqual(int(hist_ok.sum()), 64)


class TestUnevenRanks(CustomTestCase):
    """One forward under DP: every rank plans its tail at the shared bucket, from
    a real-tail rank next to a zero-tail rank, a dummy-extend rank (every row
    invalid) and a decode-converted rank (one row per request, no fill lengths).
    Each rank's tail must be well formed and every rank must give the compiled
    forward the same structure and tensor shapes."""

    T = 512
    BUCKET = 256

    B = 4  # every rank pads to the same batch bucket, as the shim does

    def _batch(self, positions, locs, extend, slots, fill):
        pad = self.T - positions.shape[0]
        more = self.B - len(extend)
        extend = list(extend) + [0] * more
        slots = list(slots) + [0] * more
        if fill is not None:
            fill = list(fill) + [0] * more
        return SimpleNamespace(
            batch_size=self.B,
            positions=torch.cat([positions, torch.zeros(pad, dtype=torch.int64)]),
            out_cache_loc=torch.cat([locs, torch.zeros(pad, dtype=torch.int64)]),
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=list(extend),
            req_pool_indices=torch.tensor(slots),
            encoder_swa_row_floor=None,
            decoder_fill_lens=fill,
            decoder_rows_bucket=self.BUCKET,
        )

    def _rank_batches(self):
        def extend_rows(prefix, extend, slots):
            pos = torch.cat([torch.arange(p, p + n) for p, n in zip(prefix, extend)])
            loc = torch.cat(
                [
                    torch.arange(p, p + n) + 4096 * (s + 1)
                    for p, n, s in zip(prefix, extend, slots)
                ]
            )
            return pos, loc

        real = self._batch(
            *extend_rows([0, 1000], [200, 300], [1, 2]), [200, 300], [1, 2], [200, 1500]
        )
        zero = self._batch(*extend_rows([0], [300], [3]), [300], [3], [5000])
        # materialize_dp_idle_extend: one request over the bucket, every row
        # without a slot (out_cache_loc 0).
        dummy = self._batch(
            torch.zeros(self.T, dtype=torch.int64),
            torch.zeros(self.T, dtype=torch.int64),
            [self.T],
            [0],
            None,
        )
        decode = self._batch(
            *extend_rows([100, 2000, 37], [1, 1, 1], [4, 5, 6]),
            [1, 1, 1],
            [4, 5, 6],
            None,
        )
        return {"real": real, "zero": zero, "dummy": dummy, "decode": decode}

    def _metadata(self, fb, request_window):
        return backend.build_forward_metadata(
            fb,
            req_to_token=torch.arange(8 * 4096).view(8, 4096),
            full_to_swa_index_mapping=None,
            request_window=request_window,
            swa_ring_valid_from=None,
            max_context_len=4096,
            low_ratios=(1, 2),
            device=torch.device("cpu"),
            decoder_replay=True,
            decoder_low_ratios=(1,),
        )

    def _shapes(self, md):
        from jax import tree_util

        leaves, tree = tree_util.tree_flatten(md)
        return tree, [tuple(x.shape) for x in leaves if hasattr(x, "shape")]

    def test_every_rank_plans_a_well_formed_tail_of_one_shape(self):
        from sglang_tpu.envs import envs as tpu_envs

        windows = (
            None,
            SimpleNamespace(workspace_first_page=4 + 4 * 8, workspace_pages=64),
        )
        for window in windows:
            with (
                self.subTest(window=window is not None),
                tpu_envs.SGLANG_TPU_DSV41_DECODER_REPLAY_ROWS.override("prompt"),
            ):
                shapes = {}
                for name, fb in self._rank_batches().items():
                    md = self._metadata(fb, window)
                    tail = md.tail
                    self.assertEqual(md.tail_rows, self.BUCKET, name)
                    gather = md.tail_gather.long()
                    self.assertTrue(
                        bool(((gather >= 0) & (gather < self.T)).all()), name
                    )
                    valid = tail.valid
                    real = int(valid.sum())
                    qsl = tail.query_start_loc
                    self.assertEqual(int(qsl[-1]), real, name)
                    self.assertTrue(bool((qsl[1:] >= qsl[:-1]).all()), name)
                    self.assertFalse(bool(valid[real:].any()), name)
                    rows = gather[:real]
                    self.assertEqual(rows.unique().numel(), real, name)
                    self.assertTrue(bool(fb.out_cache_loc[rows].gt(0).all()), name)
                    self.assertTrue(bool((tail.token_rows[~valid] == -1).all()), name)
                    pos = tail.positions.long()
                    req = tail.token_rows.long().clamp_min(0)
                    floor = tail.swa_minimum.long()[req]
                    self.assertTrue(bool((pos[valid] >= floor[valid]).all()), name)
                    if window is not None:
                        plan = tail.swa_window_plan
                        n = tail.positions.shape[0]
                        self.assertTrue(bool((plan.commit_token < n).all()), name)
                    shapes[name] = self._shapes(md)
                trees = {name: s[0] for name, s in shapes.items()}
                leaf_shapes = {name: s[1] for name, s in shapes.items()}
                ref = shapes["real"]
                for name in shapes:
                    self.assertEqual(trees[name], ref[0], f"{name} pytree structure")
                    self.assertEqual(leaf_shapes[name], ref[1], f"{name} leaf shapes")


class TestDpStateAroundTheTail(CustomTestCase):
    def test_exit_restores_the_published_dp_gather_sizes_exactly(self):
        """Inside the tail the DP gather sizes are the bucket's; on exit every
        value published for the forward (padded counts, padding mode, device
        counts) is back, not the forward batch's unpadded counts."""
        from unittest.mock import patch

        from sglang.srt.layers import dp_attention

        be = object.__new__(backend.DSV41TPUAttnBackend)
        be.decoder_low_ratios = (1,)
        tail = SimpleNamespace(
            valid=torch.tensor([True, False]), selected={}, candidates=None
        )
        be.forward_metadata = SimpleNamespace(
            tail=tail,
            tail_gather=torch.tensor([0, 3], dtype=torch.int32),
            selected={},
            candidates=None,
        )
        gpu = torch.tensor([4, 4])
        published = (8, 4, True, [4, 4], gpu)
        state = {"value": published}
        calls = []

        def set_len(*args):
            calls.append(args)
            state["value"] = tuple(args) + (None,) * (5 - len(args))

        fb = SimpleNamespace(
            global_num_tokens_cpu=[3, 1],  # unpadded real counts
            global_num_tokens_gpu=torch.tensor([3, 1]),
            global_dp_buffer_len=8,
            dp_local_start_pos=None,
            dp_local_num_tokens=None,
        )
        with (
            patch.object(dp_attention, "is_dp_attention_enabled", lambda: True),
            patch.object(dp_attention, "set_dp_buffer_len", set_len),
            patch.object(
                dp_attention, "get_global_dp_buffer_len", lambda: state["value"][0]
            ),
            patch.object(
                dp_attention, "get_local_dp_buffer_len", lambda: state["value"][1]
            ),
            patch.object(dp_attention, "is_dp_max_padding", lambda: state["value"][2]),
            patch.object(
                dp_attention, "get_dp_global_num_tokens", lambda: state["value"][3]
            ),
            patch.object(
                dp_attention._DpGatheredBufferWrapper,
                "get_dp_global_num_tokens_gpu",
                classmethod(lambda cls: state["value"][4]),
            ),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=2),
            ),
        ):
            saved = be.enter_late_layer_tail(fb)
            self.assertEqual(calls[-1][:4], (4, 2, True, [2, 2]))
            self.assertEqual(fb.global_num_tokens_gpu.tolist(), [2, 2])
            be.exit_late_layer_tail(saved, fb)
        self.assertEqual(calls[-1][:4], published[:4])
        self.assertIs(calls[-1][4], gpu)
        self.assertEqual(fb.global_num_tokens_cpu, [3, 1])
        self.assertEqual(fb.global_dp_buffer_len, 8)


class TestWindowRecordsUnique(CustomTestCase):
    def _records(self, slots, counts):
        from sglang_tpu.models.deepseek_v41.request_window import extend_window_pages

        B = len(slots)
        _, plan = extend_window_pages(
            slots=torch.tensor(slots),
            first=torch.full((B,), 200, dtype=torch.long),
            counts=torch.tensor(counts),
            minimum=torch.zeros(B, dtype=torch.long),
            columns=8,
            base_page=40,
            capacity_pages=256,
        )
        c, h = plan.commit_dst, plan.history_dst
        return torch.cat([c[c >= 0], h[h >= 0]]).long(), int((c >= 0).sum())

    def test_padding_and_repeated_slots_publish_unique_records(self):
        """write_swa_pages needs unique records: the padding slot 0 commits and
        moves nothing, and a slot shared by several requests (dummy / warmup
        batches) moves rows for its first request only."""
        cases = {
            "padding": ([0, 0, 0, 0], [6, 6, 6, 6], 0),
            "real + padding": ([3, 0, 0, 0], [6, 6, 6, 6], 6),
            "warmup one slot": ([1, 1, 1, 1], [6, 6, 6, 6], 6),
            "distinct": ([1, 2, 3, 4], [6, 6, 6, 6], 24),
        }
        for name, (slots, counts, commits) in cases.items():
            with self.subTest(name):
                records, n = self._records(slots, counts)
                self.assertEqual(records.unique().numel(), records.numel())
                self.assertEqual(n, commits)


class TestTpuStartupRules(CustomTestCase):
    def _cfg(self, prefill_backend=None, **kw):
        from sglang.srt.model_executor.cuda_graph_config import Backend

        base = dict(
            enable_mixed_chunk=False,
            disable_radix_cache=False,
            enable_encoder_swa_bounded_replay=True,
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=prefill_backend or Backend.DISABLED)
            ),
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_core_validator_sees_the_flag_off_and_tpu_rules_apply(self):
        """Core refuses decoder replay under DP attention; the TPU wrapper runs
        the TPU rules, then core's validator with the flag hidden."""
        from unittest.mock import patch

        from sglang.srt.arg_groups import deepseek_v4_hook as validator
        from sglang_tpu.models.deepseek_v41 import hooks

        cfg = self._cfg(enable_decoder_swa_bounded_replay=True)
        seen = []

        def original(server_args):
            view = validator.resolving_view(server_args)
            seen.append(
                (view.enable_decoder_swa_bounded_replay, view.disable_radix_cache)
            )

        model = SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41"))
        with (
            patch.object(validator, "resolving_view", lambda args: cfg),
            patch.object(validator, "model_config_of", lambda args: model),
            patch(
                "sglang.srt.runtime_context.get_platform",
                return_value=SimpleNamespace(is_tpu=True),
            ),
        ):
            hooks.allow_tpu_dsv41_encoder_replay(original, object())
            self.assertEqual(seen, [(False, False)])
            cfg.enable_mixed_chunk = True
            with self.assertRaises(ValueError):
                hooks.allow_tpu_dsv41_encoder_replay(original, object())
        self.assertEqual(len(seen), 1)

    def test_rules(self):
        check_tpu_dsv41_decoder_replay(self._cfg())
        check_tpu_dsv41_decoder_replay(
            self._cfg(disable_radix_cache=True, enable_encoder_swa_bounded_replay=False)
        )
        with self.assertRaises(ValueError):
            check_tpu_dsv41_decoder_replay(self._cfg(enable_mixed_chunk=True))
        from sglang.srt.model_executor.cuda_graph_config import Backend

        graph = Backend.FULL
        with self.assertRaises(ValueError):
            check_tpu_dsv41_decoder_replay(self._cfg(prefill_backend=graph))
        with self.assertRaises(ValueError):
            check_tpu_dsv41_decoder_replay(
                self._cfg(enable_encoder_swa_bounded_replay=False)
            )

    def test_stateful_decoder_layers_are_refused(self):
        cfg = SimpleNamespace(
            kv_source_layer_ids=[2, 8, 14, 20],
            engram_layer_ids=[1, 14],
            compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
            num_hidden_layers=40,
            candidate_source_layer_id=20,
        )
        backend._check_decoder_replay(cfg)
        for bad in (
            dict(engram_layer_ids=[1, 30]),
            dict(candidate_source_layer_id=24),
            dict(compress_ratios=[0, 0] + [2] * 18 + [1] * 10 + [2] * 10 + [0, 0, 0]),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                backend._check_decoder_replay(SimpleNamespace(**{**vars(cfg), **bad}))


if __name__ == "__main__":
    unittest.main()


class TestTailStart(CustomTestCase):
    def test_tail_starts_after_the_last_kv_source_layer(self):
        """Core starts the decoder tail at the last kv_source layer and relies on
        the CUDA backend writing that layer's compressed KV for every row; the
        TPU backend runs that layer on every row instead, so the tail must start
        one layer later or later layers read main KV missing for most rows."""
        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            start_tail_after_last_source,
        )

        model = SimpleNamespace(tail_writes_source_kv=True, late_layer_start=20)
        start_tail_after_last_source(None, model)
        self.assertEqual(
            (model.tail_writes_source_kv, model.late_layer_start), (False, 21)
        )

        off = SimpleNamespace(tail_writes_source_kv=False, late_layer_start=None)
        start_tail_after_last_source(None, off)
        self.assertIsNone(off.late_layer_start)


class TestLateLayerDp(CustomTestCase):
    def test_model_keeps_the_backend_decoder_bucket_layout(self):
        """Regression: core's decoder-tail DP switch (SUM_LEN, per-rank counts)
        sent the decoder layers' DP gathers through a Triton memcpy, so compile
        tracing failed on TPU. Through the model's own imports, every rank's
        count is the shared decoder bucket and core's switch is skipped."""
        import sys
        from unittest.mock import patch

        from sglang.srt.models import deepseek_v4
        from sglang.srt.models.deepseek_common import decoder_swa_bounded_replay
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            keep_backend_late_layer_dp,
            lockstep_late_layer_dp_counts,
        )

        module = "sglang.srt.models.deepseek_common.decoder_swa_bounded_replay"
        originals = (
            decoder_swa_bounded_replay.late_layer_dp_counts,
            decoder_swa_bounded_replay.enter_late_layer_dp,
        )
        # The registry patches every module's reference; put them all back after.
        bound = [
            (mod, name, value)
            for mod in list(sys.modules.values())
            for name, value in list(vars(mod).items())
            if any(value is fn for fn in originals)
        ]
        HookRegistry.reset()
        try:
            HookRegistry.register(
                f"{module}.late_layer_dp_counts",
                lockstep_late_layer_dp_counts,
                HookType.AROUND,
            )
            HookRegistry.register(
                f"{module}.enter_late_layer_dp",
                keep_backend_late_layer_dp,
                HookType.AROUND,
            )
            HookRegistry.apply_hooks()
            with patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=8),
            ):
                counts = deepseek_v4.late_layer_dp_counts(
                    SimpleNamespace(), 128, in_graph=False
                )
            self.assertEqual(counts, [128] * 8)
            self.assertIsNone(
                deepseek_v4.enter_late_layer_dp(SimpleNamespace(), counts)
            )
        finally:
            HookRegistry.reset()
            for mod, name, value in bound:
                setattr(mod, name, value)
