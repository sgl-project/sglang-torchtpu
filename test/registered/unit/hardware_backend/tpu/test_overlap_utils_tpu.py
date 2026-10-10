"""CPU coverage for the TPU FutureMap relay helpers in overlap_utils_tpu."""

import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang_tpu import patches
from sglang_tpu.overlap_utils_tpu import (
    flush_direct_handoff_tpu,
    resolve_direct_handoff_tpu,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def tpu_future_map():
    """Install the TPU FutureMap patch on a CPU host, restoring it on exit."""
    with ExitStack() as stack:
        for name in (
            "stash",
            "publish",
            "stash_bonus_tokens",
            "resolve_mixed_spec_tails",
        ):
            stack.enter_context(patch.object(FutureMap, name, getattr(FutureMap, name)))
        for name in ("tpu_direct_handoff", "tpu_published_seq_lens"):
            stack.enter_context(patch.object(FutureMap, name, None, create=True))
        stack.enter_context(
            patch.object(patches, "_future_map_direct_handoff_patched", False)
        )
        patches._patch_future_map_direct_handoff()
        yield


def stash(fm, indices, payload, seq_lens, published=True) -> bool:
    """Publish then stash like the overlap forward; report whether deferred."""
    with tpu_future_map():
        if published is True:
            published = indices
        if published is not None:
            fm.publish(published, seq_lens)
        fm.stash(indices, payload)
        return fm.tpu_direct_handoff is not None


def make_case():
    algo = SimpleNamespace(
        is_eagle3=lambda: True,
        is_some=lambda: True,
        is_ngram=lambda: False,
        need_topk=lambda: True,
    )
    fm = FutureMap(
        device="cpu",
        spec_algo=algo,
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(8, 16)),
        needs_cpu_seq_lens=False,
    )
    fm.need_topk = fm.need_hidden_states = True
    fm.topk_p_buf = torch.zeros(8, 1)
    fm.topk_index_buf = torch.zeros(8, 1, dtype=torch.int64)
    fm.hidden_states_buf = torch.zeros(8, 3)
    fm.output_tokens_buf.zero_()
    fm.new_seq_lens_buf.zero_()
    indices = torch.tensor([2, 5])
    payload = RelayPayload(
        bonus_tokens=torch.tensor([11, 12]),
        topk_p=torch.ones(2, 1),
        topk_index=torch.tensor([[21], [22]]),
        hidden_states=torch.arange(6, dtype=torch.float32).reshape(2, 3),
    )
    draft = SimpleNamespace(
        future_indices=indices,
        accept_lens_device=torch.tensor([1, 3]),
        device_accept_candidate_locs=torch.arange(14).reshape(2, 7),
        device_accept_selected_cache_locs=None,
        future_dsa_topk_indices_available=False,
    )
    batch = SimpleNamespace(spec_info=draft)
    draft.bonus_tokens = payload.bonus_tokens
    return fm, batch, indices, payload, torch.tensor([101, 203])


class TestDirectHandoff(CustomTestCase):
    def test_exact_generation_consumes_without_store_and_pins_values(self):
        fm, batch, indices, payload, seq_lens = make_case()
        expected = payload.hidden_states
        real_store = fm._store

        def store_seq_lens_only(buf, indices, values):
            # Publish still writes seq_lens; only the payload store is skipped.
            assert buf is fm.new_seq_lens_buf, "unexpected payload store"
            real_store(buf, indices, values)

        with patch.object(fm, "_store", side_effect=store_seq_lens_only):
            self.assertTrue(stash(fm, indices, payload, seq_lens))
            self.assertTrue(resolve_direct_handoff_tpu(fm, batch))
            self.assertIs(batch.spec_info.hidden_states, expected)
            torch.testing.assert_close(
                batch.spec_info.device_accept_selected_cache_locs,
                torch.tensor([[0, 1, 2, 3], [9, 10, 11, 12]]),
            )
            self.assertIsNone(fm.tpu_direct_handoff)
            self.assertFalse(resolve_direct_handoff_tpu(fm, batch))
        payload.hidden_states = torch.full((2, 3), -10.0)
        self.assertIs(batch.spec_info.hidden_states, expected)
        torch.testing.assert_close(batch.seq_lens, seq_lens)

    def test_filter_reorder_or_generation_change_flushes_before_fallback(self):
        for change in [
            "filter",
            "reorder",
            "generation",
            "no_candidates",
            "no_accept_lens",
        ]:
            with self.subTest(change=change):
                fm, batch, indices, payload, seq_lens = make_case()
                self.assertTrue(stash(fm, indices, payload, seq_lens))
                if change == "filter":
                    batch.spec_info.future_indices = indices[:1]
                elif change == "reorder":
                    batch.spec_info.future_indices = indices.flip(0)
                elif change == "generation":
                    batch.spec_info.bonus_tokens = payload.bonus_tokens.clone()
                elif change == "no_accept_lens":
                    batch.spec_info.accept_lens_device = None
                else:
                    batch.spec_info.device_accept_candidate_locs = None
                self.assertFalse(resolve_direct_handoff_tpu(fm, batch))
                torch.testing.assert_close(
                    fm.hidden_states_buf[indices], payload.hidden_states
                )
                torch.testing.assert_close(
                    fm.topk_index_buf[indices], payload.topk_index
                )
                torch.testing.assert_close(
                    fm.output_tokens_buf[indices], payload.bonus_tokens
                )
                self.assertEqual(fm.hidden_states_buf[0].sum().item(), 0)
                self.assertIsNone(fm.tpu_direct_handoff)

    def test_unconsumed_replacement_preserves_old_rows(self):
        fm, batch, indices, payload, seq_lens = make_case()
        stash(fm, indices, payload, seq_lens)
        other = RelayPayload(
            bonus_tokens=torch.tensor([31, 32]),
            topk_p=payload.topk_p,
            topk_index=payload.topk_index,
            hidden_states=payload.hidden_states + 100,
        )
        stash(fm, torch.tensor([3, 4]), other, seq_lens)
        torch.testing.assert_close(fm.output_tokens_buf[indices], payload.bonus_tokens)
        with tpu_future_map():
            flush_direct_handoff_tpu(fm)
        torch.testing.assert_close(
            fm.output_tokens_buf[torch.tensor([3, 4])], other.bonus_tokens
        )

    def test_consumed_generation_then_filtered_successor_uses_new_values(self):
        fm, batch, indices, payload, seq_lens = make_case()
        stash(fm, indices, payload, seq_lens)
        self.assertTrue(resolve_direct_handoff_tpu(fm, batch))
        successor = SimpleNamespace(**vars(batch.spec_info))
        successor.device_accept_selected_cache_locs = None
        batch.spec_info = successor
        newer = RelayPayload(
            bonus_tokens=payload.bonus_tokens + 50,
            topk_p=payload.topk_p,
            topk_index=payload.topk_index + 50,
            hidden_states=payload.hidden_states + 50,
        )
        stash(fm, indices, newer, seq_lens)
        successor.bonus_tokens = newer.bonus_tokens
        successor.future_indices = indices[:1]
        self.assertFalse(resolve_direct_handoff_tpu(fm, batch))
        torch.testing.assert_close(
            fm.output_tokens_buf[indices[:1]], newer.bonus_tokens[:1]
        )
        torch.testing.assert_close(
            fm.hidden_states_buf[indices[:1]], newer.hidden_states[:1]
        )

    def test_unsupported_paths_do_not_defer(self):
        for change in [
            "no_publish",
            "other_indices",
            "not_eagle3",
            "wide_topk",
            "no_hidden",
            "probabilities",
        ]:
            with self.subTest(change=change):
                fm, batch, indices, payload, seq_lens = make_case()
                fallback = nullcontext()
                published = None if change == "no_publish" else indices
                if change == "other_indices":
                    # Beam tail slices the stash indices after publish.
                    indices = indices[:]
                elif change == "not_eagle3":
                    fm.spec_algo.is_eagle3 = lambda: False
                elif change == "wide_topk":
                    fm.topk_p_buf = torch.zeros(8, 2)
                    fm.topk_index_buf = torch.zeros(8, 2, dtype=torch.int64)
                    payload.topk_p = torch.ones(2, 2)
                    payload.topk_index = torch.tensor([[21, 23], [22, 24]])
                elif change == "no_hidden":
                    # The fallback store lazily asks the spec config whether
                    # hidden states are relayed; no config is published here.
                    fm.need_hidden_states = False
                    payload.hidden_states = None
                    fallback = patch(
                        "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
                        return_value=False,
                    )
                else:
                    payload.draft_probs = torch.ones(2, 5)
                with fallback:
                    self.assertFalse(
                        stash(fm, indices, payload, seq_lens, published=published)
                    )
                torch.testing.assert_close(
                    fm.output_tokens_buf[indices], payload.bonus_tokens
                )

    def test_mixed_batch_resolve_flushes_pending_rows_first(self):
        # A spec mixed batch reads decode-tail tokens straight from the buffer.
        fm, batch, indices, payload, seq_lens = make_case()
        self.assertTrue(stash(fm, indices, payload, seq_lens))
        seen = []

        def read_tails(self, batch):
            seen.append(self.output_tokens_buf[batch.mix_running_indices].clone())

        with patch.object(FutureMap, "resolve_mixed_spec_tails", read_tails):
            with tpu_future_map():
                fm.resolve_mixed_spec_tails(
                    SimpleNamespace(mix_running_indices=indices)
                )
        torch.testing.assert_close(seen[0], payload.bonus_tokens)
        self.assertIsNone(fm.tpu_direct_handoff)

    def test_bonus_token_stash_flushes_pending_rows_first(self):
        fm, batch, indices, payload, seq_lens = make_case()
        self.assertTrue(stash(fm, indices, payload, seq_lens))
        with tpu_future_map():
            fm.stash_bonus_tokens(indices[:1], torch.tensor([77]))
        # The flush lands first, so the newer bonus-token write wins.
        torch.testing.assert_close(
            fm.output_tokens_buf[indices], torch.tensor([77, 12])
        )
        self.assertIsNone(fm.tpu_direct_handoff)


if __name__ == "__main__":
    unittest.main()
