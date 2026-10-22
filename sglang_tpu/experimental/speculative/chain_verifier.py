# SPDX-License-Identifier: Apache-2.0
"""Chain acceptance policy over an explicit platform operation bundle."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import msgspec

from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang_tpu.experimental.speculative import spec_ops


class SpecOps(NamedTuple):
    accept_drafts_greedy: Callable = spec_ops.accept_drafts_greedy
    accept_drafts_target_only: Callable = spec_ops.accept_drafts_target_only
    accept_drafts_sampled_filtered: Callable = spec_ops.accept_drafts_sampled_filtered
    accept_drafts_sampled: Callable = spec_ops.accept_drafts_sampled
    finalize_accept: Callable = spec_ops.finalize_accept


class ChainVerifier(msgspec.Struct, frozen=True):
    ops: SpecOps = SpecOps()

    def accept(
        self, *, target_logits, draft_result, bucket, threshold_single, threshold_acc
    ):
        if bucket.temperatures is None or bucket.top_ks is None:
            num_correct_drafts, bonus_tokens = self.ops.accept_drafts_greedy(
                target_logits, draft_result.draft_tokens, draft_result.num_verify_drafts
            )
            site = SpecTpSyncSite.SPEC_ACCEPT_GREEDY
        elif draft_result.draft_logits is None:
            # A deterministic draft is a one-hot q: TPU EAGLE's target-only rule.
            num_correct_drafts, bonus_tokens = self.ops.accept_drafts_target_only(
                target_logits,
                draft_result.draft_tokens,
                draft_result.num_verify_drafts,
                bucket.temperatures,
                bucket.top_ks,
                bucket.top_ps,
                bucket.min_ps,
                threshold_single,
                threshold_acc,
            )
            site = SpecTpSyncSite.SPEC_ACCEPT_SAMPLE
        elif bucket.need_filters:
            num_correct_drafts, bonus_tokens = self.ops.accept_drafts_sampled_filtered(
                target_logits,
                draft_result.draft_tokens,
                draft_result.num_verify_drafts,
                draft_result.draft_logits.float(),
                bucket.temperatures,
                bucket.top_ks,
                bucket.top_ps,
                bucket.min_ps,
            )
            site = SpecTpSyncSite.SPEC_ACCEPT_SAMPLE
        else:
            num_correct_drafts, bonus_tokens = self.ops.accept_drafts_sampled(
                target_logits,
                draft_result.draft_tokens,
                draft_result.num_verify_drafts,
                draft_result.draft_logits.float(),
                bucket.temperatures,
                bucket.top_ks,
            )
            site = SpecTpSyncSite.SPEC_ACCEPT_SAMPLE
        return num_correct_drafts, bonus_tokens, site
