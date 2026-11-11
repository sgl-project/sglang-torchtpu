"""SGLANG_TPU_PD_DECODE_DRAFT_PREFILL: the hidden rows a PD prefill ships and the decode reads."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
)
from sglang.srt.model_executor.pool_configurator import (
    DefaultPoolConfigurator,
    HybridSWAPoolConfigurator,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden import common as raiden_common
from sglang_tpu.disaggregation.raiden import pipeline as raiden_pipeline
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental import runs_draft_prompt_pass
from sglang_tpu.experimental.speculative import pd_draft_prefill, pd_hidden_pool
from sglang_tpu.experimental.speculative.pd_draft_prefill import (
    prebuilt_extend_groups,
)
from sglang_tpu.experimental.speculative.pd_hidden_pool import (
    PdHiddenPool,
    hidden_lanes,
)
from sglang_tpu.hooks.experimental_spec import (
    budget_pd_hidden_pool,
    reject_overlong_pd_draft_prompt,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

KV_HEADS = 8
PAGES, PAGE = 4, 4
FEATURE = 3 * 160


def _pool(attn_tp_size: int, group=None) -> PdHiddenPool:
    return PdHiddenPool(
        num_pages=PAGES,
        page_size=PAGE,
        feature_size=FEATURE,
        total_kv_heads=KV_HEADS,
        attn_tp_size=attn_tp_size,
        attn_tp_group=group,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )


class _FakeGroup:
    """all_gather over ranks whose local tensors are given up front."""

    def __init__(self, per_rank: list[torch.Tensor]) -> None:
        self.per_rank = per_rank

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.cat(self.per_rank, dim=dim)


class TestPdHiddenPool(CustomTestCase):
    def _shipped(self, slots: list[int]) -> tuple[PdHiddenPool, torch.Tensor]:
        prefill = _pool(attn_tp_size=1)
        hidden = torch.randn(len(slots) + 2, FEATURE).to(torch.bfloat16)
        segments = prefill.write_segments(torch.tensor(slots), hidden.shape[0])
        prefill.write(hidden, segments=segments)
        return prefill, hidden[: len(slots)]

    def test_bytes_per_token_is_the_buffer_row(self):
        for attn_tp_size in (1, 8):
            pool = _pool(attn_tp_size=attn_tp_size)
            row = pool.buffer[0, 0]
            self.assertEqual(
                PdHiddenPool.bytes_per_token(
                    feature_size=FEATURE,
                    total_kv_heads=KV_HEADS,
                    attn_tp_size=attn_tp_size,
                    dtype=torch.bfloat16,
                ),
                row.numel() * row.element_size(),
            )

    def test_lanes_hold_a_row_in_head_slices(self):
        lanes = hidden_lanes(feature_size=FEATURE, total_kv_heads=KV_HEADS)
        self.assertEqual(lanes % 128, 0)
        self.assertGreaterEqual(KV_HEADS * 2 * lanes, FEATURE)

    def test_prefill_rows_round_trip_without_padding_rows(self):
        """Bucket padding rows past the real tokens must not land in slot 0."""
        slots = [5, 6, 7, 9]
        prefill, hidden = self._shipped(slots)
        self.assertTrue(torch.equal(prefill.read(torch.tensor(slots)), hidden))
        self.assertEqual(prefill.buffer.flatten(0, 1)[0].abs().sum().item(), 0.0)

    def test_tp_decode_joins_the_head_slices_raiden_hands_each_rank(self):
        """Raiden gives decode rank r kv head r of every row; the read joins them."""
        slots = [1, 2, 3, 12]
        prefill, hidden = self._shipped(slots)
        tp = 8
        loc = torch.tensor(slots)
        locals_ = [
            prefill.buffer[:, :, rank : rank + 1]
            .flatten(0, 1)[loc]
            .reshape(len(slots), -1)
            for rank in range(tp)
        ]
        group = _FakeGroup([t.unsqueeze(0) for t in locals_])
        decode = _pool(attn_tp_size=tp, group=group)
        self.assertEqual(decode.buffer.shape[2], 1)
        self.assertTrue(torch.equal(decode.read(loc), hidden))


class TestRaidenCarriesHiddenNotDraftKv(CustomTestCase):
    def test_hidden_pool_replaces_the_draft_pool_in_its_own_class(self):
        target = SimpleNamespace(
            start_layer=0,
            layer_num=2,
            layers_mapping=None,
            get_fused_kv_buffer=lambda layer: torch.zeros(PAGES, PAGE, 8, 2, 128),
        )
        draft = SimpleNamespace(
            start_layer=0,
            layer_num=1,
            layers_mapping=None,
            get_fused_kv_buffer=lambda layer: torch.zeros(PAGES, PAGE, 8, 2, 128),
        )
        hidden = _pool(attn_tp_size=1)
        with patch.object(pd_hidden_pool, "_pd_hidden_pool", hidden):
            groups = raiden_common.get_raiden_kv_caches(target, draft_pool=draft)
        self.assertEqual(groups.num_draft_full, 0)
        self.assertEqual(groups.num_hidden_full, 1)
        self.assertIs(groups.full[-1], hidden.buffer)
        tags = [pool["tag"] for pool in raiden_pipeline.pool_manifest(groups)]
        self.assertEqual(tags, ["fa.l0", "fa.l1", "pdhid.fa.l0"])

    def test_stages_without_the_hidden_pool_fingerprint_alike(self):
        """Only the last prefill stage and the decode hold the hidden pool."""
        target = SimpleNamespace(
            start_layer=0,
            layer_num=1,
            layers_mapping=None,
            get_fused_kv_buffer=lambda layer: torch.zeros(PAGES, PAGE, 8, 2, 128),
        )
        layout = lambda tensor: ((4, 3, 2, 1, 0), ((2, 128),), 16)  # noqa: E731
        with patch.object(pd_hidden_pool, "_pd_hidden_pool", _pool(attn_tp_size=1)):
            with_hidden = raiden_common.get_raiden_kv_caches(target, draft_pool=target)
        without = raiden_common.get_raiden_kv_caches(target)
        fingerprints = [
            raiden_pipeline.layout_fingerprint(
                raiden_pipeline.pool_manifest(groups),
                groups.full,
                layout,
                ships_pd_hidden=False,
            )
            for groups in (with_hidden, without)
        ]
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_ends_disagreeing_on_the_flag_fingerprint_apart(self):
        groups = SimpleNamespace(
            full=[torch.zeros(PAGES, PAGE, 8, 2, 128)],
            full_layer_ids=[0],
            num_draft_full=0,
            num_hidden_full=0,
            swa=[],
            swa_layer_ids=[],
            num_draft_swa=0,
        )
        layout = lambda tensor: ((4, 3, 2, 1, 0), ((2, 128),), 16)  # noqa: E731
        pools = raiden_pipeline.pool_manifest(groups)
        self.assertNotEqual(
            raiden_pipeline.layout_fingerprint(
                pools, groups.full, layout, ships_pd_hidden=True
            ),
            raiden_pipeline.layout_fingerprint(
                pools, groups.full, layout, ships_pd_hidden=False
            ),
        )

    def test_without_the_flag_the_draft_pool_travels(self):
        target = SimpleNamespace(
            start_layer=0,
            layer_num=1,
            layers_mapping=None,
            get_fused_kv_buffer=lambda layer: torch.zeros(PAGES, PAGE, 8, 2, 128),
        )
        with patch.object(pd_hidden_pool, "_pd_hidden_pool", None):
            groups = raiden_common.get_raiden_kv_caches(target, draft_pool=target)
        self.assertEqual(groups.num_hidden_full, 0)


class TestDraftPromptPassPlacement(CustomTestCase):
    def test_flag_moves_the_prompt_pass_from_prefill_to_decode(self):
        for flag, prefill, decode in ((False, True, False), (True, False, True)):
            with tpu_envs.SGLANG_TPU_PD_DECODE_DRAFT_PREFILL.override(flag):
                self.assertEqual(runs_draft_prompt_pass("prefill"), prefill)
                self.assertEqual(runs_draft_prompt_pass("decode"), decode)
                self.assertTrue(runs_draft_prompt_pass("null"))

    def test_groups_pack_requests_under_the_bucket_and_cap(self):
        batch = SimpleNamespace(extend_lens=[3000, 3000, 3000, 100, 100, 100])
        groups = prebuilt_extend_groups(batch, max_tokens=8192, max_reqs=2)
        self.assertEqual([list(g) for g in groups], [[0, 1], [2, 3], [4, 5]])
        groups = prebuilt_extend_groups(batch, max_tokens=6000, max_reqs=8)
        self.assertEqual([list(g) for g in groups], [[0, 1], [2, 3, 4, 5]])
        with self.assertRaises(AssertionError):
            prebuilt_extend_groups(
                SimpleNamespace(extend_lens=[9000]), max_tokens=8192, max_reqs=8
            )


def _kvc(**overrides):
    fields = dict(
        is_draft_worker=False,
        spec_algorithm=SpeculativeAlgorithm.EAGLE3,
        spec_aux_config=SpecAuxHiddenStateConfig(
            eagle_use_aux_hidden_state=True, eagle_aux_hidden_state_layer_ids=[1, 2, 3]
        ),
        model_config=SimpleNamespace(
            hidden_size=160, get_total_num_kv_heads=lambda: KV_HEADS
        ),
        model_dtype=torch.bfloat16,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _configurator(cls, cell_size: int = 1000):
    configurator = object.__new__(cls)
    configurator._cell_size = cell_size
    return configurator


class TestHiddenPoolKvBudget(CustomTestCase):
    def setUp(self):
        super().setUp()
        for scope in (
            tpu_envs.SGLANG_TPU_PD_DECODE_DRAFT_PREFILL.override(True),
            tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(True),
            get_context().override_server_args(disaggregation_mode="decode"),
            get_parallel().override(pp_rank=0, pp_size=1),
        ):
            scope.__enter__()
            self.addCleanup(scope.__exit__, None, None, None)

    def test_the_budget_counts_the_hidden_row(self):
        row = PdHiddenPool.bytes_per_token(
            feature_size=FEATURE,
            total_kv_heads=KV_HEADS,
            attn_tp_size=1,
            dtype=torch.bfloat16,
        )
        configurator = budget_pd_hidden_pool(
            _configurator(DefaultPoolConfigurator), _kvc()
        )
        self.assertEqual(configurator._cell_size, 1000 + row)

    def test_servers_without_the_pool_keep_their_budget(self):
        cases = [
            (_kvc(is_draft_worker=True), {}),
            (_kvc(), dict(pp_rank=0, pp_size=2)),
        ]
        for kvc, parallel in cases:
            with get_parallel().override(**parallel):
                configurator = budget_pd_hidden_pool(
                    _configurator(DefaultPoolConfigurator), kvc
                )
            self.assertEqual(configurator._cell_size, 1000)
        with tpu_envs.SGLANG_TPU_PD_DECODE_DRAFT_PREFILL.override(False):
            configurator = budget_pd_hidden_pool(
                _configurator(DefaultPoolConfigurator), _kvc()
            )
        self.assertEqual(configurator._cell_size, 1000)

    def test_a_budget_it_cannot_size_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "HybridSWAPoolConfigurator"):
            budget_pd_hidden_pool(_configurator(HybridSWAPoolConfigurator), _kvc())


class TestOverlongPromptAbort(CustomTestCase):
    def _admit(self, prompt_len: int, cap: int | None):
        queue = SimpleNamespace(
            scheduler=SimpleNamespace(
                output_streamer=SimpleNamespace(stream_output=lambda *args: None)
            )
        )
        req = SimpleNamespace(
            rid="r", origin_input_ids=[0] * prompt_len, return_logprob=False
        )
        prompt_cap = None if cap is None else (lambda: cap)
        with patch.object(pd_draft_prefill, "_pd_draft_prompt_cap", prompt_cap):
            rejected = reject_overlong_pd_draft_prompt(
                lambda queue, req: False, queue, req
            )
        return rejected, req

    def test_only_the_long_prompt_is_aborted(self):
        rejected, req = self._admit(9000, cap=8192)
        self.assertTrue(rejected)
        self.assertEqual(req.finished_reason.status_code, 400)
        self.assertFalse(self._admit(8192, cap=8192)[0])
        self.assertFalse(self._admit(9000, cap=None)[0])


if __name__ == "__main__":
    unittest.main()
