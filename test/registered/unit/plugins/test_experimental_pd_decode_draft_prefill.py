"""SGLANG_TPU_PD_DECODE_DRAFT_PREFILL: the hidden rows a PD prefill ships and the decode reads."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden import common as raiden_common
from sglang_tpu.disaggregation.raiden import pipeline as raiden_pipeline
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental import runs_draft_prompt_pass
from sglang_tpu.experimental.speculative import pd_hidden_pool
from sglang_tpu.experimental.speculative.pd_draft_prefill import (
    prebuilt_extend_groups,
)
from sglang_tpu.experimental.speculative.pd_hidden_pool import (
    PdHiddenPool,
    hidden_lanes,
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
                raiden_pipeline.pool_manifest(groups), groups.full, layout
            )
            for groups in (with_hidden, without)
        ]
        self.assertEqual(fingerprints[0], fingerprints[1])

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
        with self.assertRaises(NotImplementedError):
            prebuilt_extend_groups(
                SimpleNamespace(extend_lens=[9000]), max_tokens=8192, max_reqs=8
            )


if __name__ == "__main__":
    unittest.main()
