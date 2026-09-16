"""One-use relay identity, fallback ordering and owned tensor lifetime."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.hardware_backend.tpu.direct_handoff import (
    flush_direct_handoff_tpu,
    resolve_direct_handoff_tpu,
    stash_direct_handoff_tpu,
)
from sglang.srt.hardware_backend.tpu.overlap_utils_tpu import (
    _select_device_accept_cache_locs,
    resolve_device_accept_seq_lens_tpu,
)
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
    fm.set_tpu_precompiled_batch_sizes([2])
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
        tpu_draft_continuation=None,
        accept_lens_device=torch.tensor([1, 3]),
        device_accept_candidate_locs=torch.arange(14).reshape(2, 7),
        device_accept_selected_cache_locs=None,
        future_dsa_topk_indices_available=False,
    )
    batch = SimpleNamespace(
        enable_overlap=True,
        forward_mode=SimpleNamespace(is_draft_extend_v2=lambda: True),
        return_logprob=False,
        has_grammar=False,
        reqs=[SimpleNamespace(lora_id=None)] * 2,
        spec_info=draft,
    )
    result = SimpleNamespace(
        next_draft_input=draft, new_seq_lens=torch.tensor([101, 203])
    )
    return fm, batch, indices, payload, result


class TestDirectHandoff(CustomTestCase):
    def test_exact_generation_consumes_without_store_and_pins_values(self):
        fm, batch, indices, payload, result = make_case()
        expected = payload.hidden_states
        with patch.object(fm, "_store", side_effect=AssertionError("unexpected store")):
            self.assertTrue(
                stash_direct_handoff_tpu(fm, batch, indices, payload, result)
            )
            with patch(
                "sglang.srt.hardware_backend.tpu.overlap_utils_tpu.select_direct_accept_cache_tpu",
                side_effect=_select_device_accept_cache_locs,
            ):
                self.assertTrue(resolve_device_accept_seq_lens_tpu(fm, batch))
            relay = batch.spec_info.tpu_relay_inputs
            self.assertIs(relay.logical_seq_lens, batch.seq_lens)
            self.assertIs(relay.hidden_states, expected)
            self.assertIs(
                relay.cache_locs, batch.spec_info.device_accept_selected_cache_locs
            )
            self.assertIs(batch.spec_info.hidden_states, expected)
            torch.testing.assert_close(
                batch.spec_info.device_accept_selected_cache_locs,
                torch.tensor([[0, 1, 2, 3], [9, 10, 11, 12]]),
            )
            self.assertIsNone(fm.tpu_direct_handoff)
            self.assertFalse(resolve_direct_handoff_tpu(fm, batch))
        payload.hidden_states = torch.full((2, 3), -10.0)
        self.assertIs(batch.spec_info.hidden_states, expected)
        torch.testing.assert_close(batch.seq_lens, result.new_seq_lens)

    def test_filter_reorder_or_generation_change_flushes_before_fallback(self):
        for change in ["filter", "reorder", "generation", "no_candidates"]:
            with self.subTest(change=change):
                fm, batch, indices, payload, result = make_case()
                self.assertTrue(
                    stash_direct_handoff_tpu(fm, batch, indices, payload, result)
                )
                if change == "filter":
                    batch.spec_info.future_indices = indices[:1]
                elif change == "reorder":
                    batch.spec_info.future_indices = indices.flip(0)
                elif change == "generation":
                    batch.spec_info = SimpleNamespace(**vars(batch.spec_info))
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
        fm, batch, indices, payload, result = make_case()
        stash_direct_handoff_tpu(fm, batch, indices, payload, result)
        other = RelayPayload(
            bonus_tokens=torch.tensor([31, 32]),
            topk_p=payload.topk_p,
            topk_index=payload.topk_index,
            hidden_states=payload.hidden_states + 100,
        )
        stash_direct_handoff_tpu(fm, batch, torch.tensor([3, 4]), other, result)
        torch.testing.assert_close(fm.output_tokens_buf[indices], payload.bonus_tokens)
        flush_direct_handoff_tpu(fm)
        torch.testing.assert_close(
            fm.output_tokens_buf[torch.tensor([3, 4])], other.bonus_tokens
        )

    def test_consumed_generation_then_filtered_successor_uses_new_values(self):
        fm, batch, indices, payload, result = make_case()
        stash_direct_handoff_tpu(fm, batch, indices, payload, result)
        with patch(
            "sglang.srt.hardware_backend.tpu.overlap_utils_tpu.select_direct_accept_cache_tpu",
            side_effect=_select_device_accept_cache_locs,
        ):
            self.assertTrue(resolve_direct_handoff_tpu(fm, batch))
        successor = SimpleNamespace(**vars(result.next_draft_input))
        successor.device_accept_selected_cache_locs = None
        result.next_draft_input = successor
        batch.spec_info = successor
        newer = RelayPayload(
            bonus_tokens=payload.bonus_tokens + 50,
            topk_p=payload.topk_p,
            topk_index=payload.topk_index + 50,
            hidden_states=payload.hidden_states + 50,
        )
        stash_direct_handoff_tpu(fm, batch, indices, newer, result)
        successor.future_indices = indices[:1]
        self.assertFalse(resolve_direct_handoff_tpu(fm, batch))
        torch.testing.assert_close(
            fm.output_tokens_buf[indices[:1]], newer.bonus_tokens[:1]
        )
        torch.testing.assert_close(
            fm.hidden_states_buf[indices[:1]], newer.hidden_states[:1]
        )

    def test_padded_handoff_reuses_physical_rows_but_flushes_only_real_rows(self):
        from sglang.srt.hardware_backend.tpu.padded_spec_state import (
            DraftContinuation,
            install_draft_continuation_tpu,
        )

        for consume in (False, True):
            with self.subTest(consume=consume):
                fm, batch, indices, payload, result = make_case()
                fm.set_tpu_precompiled_batch_sizes([4])
                physical = DraftContinuation(
                    logical_batch_size=2,
                    physical_batch_size=4,
                    hidden_states=torch.cat(
                        (payload.hidden_states, torch.full((2, 3), 999.0))
                    ),
                    topk_p=torch.ones(4, 1),
                    topk_index=torch.cat((payload.topk_index, torch.full((2, 1), 999))),
                    bonus_tokens=torch.tensor([11, 12, 999, 999], dtype=torch.int32),
                    accept_lens=torch.tensor([1, 3, 4, 4]),
                )
                result.next_draft_input.tpu_draft_continuation = physical
                self.assertTrue(
                    stash_direct_handoff_tpu(fm, batch, indices, payload, result)
                )
                if consume:
                    with patch(
                        "sglang.srt.hardware_backend.tpu.overlap_utils_tpu.select_direct_accept_cache_tpu",
                        side_effect=_select_device_accept_cache_locs,
                    ):
                        self.assertTrue(resolve_device_accept_seq_lens_tpu(fm, batch))
                    self.assertIsNone(batch.spec_info.tpu_relay_inputs)
                    self.assertEqual(
                        batch.spec_info.device_accept_selected_cache_locs.shape, (2, 4)
                    )
                    self.assertEqual(batch.seq_lens.shape, (2,))
                    install_draft_continuation_tpu(batch.spec_info, 4)
                    self.assertIs(batch.spec_info.hidden_states, physical.hidden_states)
                    self.assertIsNone(batch.spec_info.tpu_draft_continuation)
                    self.assertEqual(
                        batch.spec_info.bonus_tokens.dtype, fm.output_tokens_buf.dtype
                    )
                else:
                    batch.spec_info.future_indices = indices[:1]
                    self.assertFalse(resolve_direct_handoff_tpu(fm, batch))
                    torch.testing.assert_close(
                        fm.hidden_states_buf[indices], payload.hidden_states
                    )
                    self.assertEqual(fm.hidden_states_buf[0].count_nonzero().item(), 0)
                    self.assertEqual(fm.output_tokens_buf[0].item(), 0)

    def test_other_algorithms_do_not_require_eagle_continuation_fields(self):
        fm, batch, indices, payload, result = make_case()
        fm.spec_algo.is_eagle3 = lambda: False
        result.next_draft_input = SimpleNamespace()
        self.assertFalse(stash_direct_handoff_tpu(fm, batch, indices, payload, result))

    def test_unsupported_paths_do_not_defer(self):
        for change in [
            "no_overlap",
            "prefill",
            "grammar",
            "logprob",
            "lora",
            "off_bucket",
            "probabilities",
        ]:
            with self.subTest(change=change):
                fm, batch, indices, payload, result = make_case()
                if change == "no_overlap":
                    batch.enable_overlap = False
                elif change == "prefill":
                    batch.forward_mode.is_draft_extend_v2 = lambda: False
                elif change == "grammar":
                    batch.has_grammar = True
                elif change == "logprob":
                    batch.return_logprob = True
                elif change == "lora":
                    batch.reqs[0].lora_id = "adapter"
                elif change == "off_bucket":
                    fm.set_tpu_precompiled_batch_sizes([4])
                else:
                    payload.draft_probs = torch.ones(2, 5)
                self.assertFalse(
                    stash_direct_handoff_tpu(fm, batch, indices, payload, result)
                )
                self.assertIsNone(fm.tpu_direct_handoff)


if __name__ == "__main__":
    unittest.main()
