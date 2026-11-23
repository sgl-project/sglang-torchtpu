"""Unit tests for the encode-disaggregation receiver."""

import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace

import torch
from fastapi import HTTPException

from sglang.srt.disaggregation.encoder.receiver import (
    MMReceiverBase,
    WaitingMMRequestStatus,
    _select_mm_processor_prompt,
)
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSelectMMProcessorPrompt(CustomTestCase):
    def test_tokens_in_prompt_is_list(self):
        """A tokens-in prompt reaches the processor as list[int], not array("q");
        HTTP sends input_text=None and the gRPC servicer sends ""."""
        processor = SimpleNamespace(prefer_tokenized_input=False)
        for input_text in (None, ""):
            with self.subTest(input_text=input_text):
                recv_req = SimpleNamespace(
                    input_text=input_text, input_ids=array("q", [1, 2, 3])
                )

                prompt = _select_mm_processor_prompt(recv_req, processor)

                self.assertIsInstance(prompt, list)
                self.assertEqual(prompt, [1, 2, 3])


class TestEncodeReceiverRequestConstruction(CustomTestCase):
    @staticmethod
    def _create_req(input_ids):
        scheduler = SimpleNamespace(
            model_config=SimpleNamespace(hf_eos_token_id={2}, vocab_size=128),
            disaggregation_mode=DisaggregationMode.NULL,
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            metrics_collector=None,
            dllm_config=None,
            tokenizer=object(),
        )
        receiver = SimpleNamespace(scheduler=scheduler)
        recv_req = SimpleNamespace(
            rid="request-1",
            input_text="hello",
            input_ids=input_ids,
            sampling_params=SamplingParams(max_new_tokens=1),
            return_logprob=False,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            lora_id=None,
            input_embeds=None,
            custom_logit_processor=None,
            require_reasoning=False,
            return_hidden_states=False,
            return_routed_experts=False,
            routed_experts_start_len=0,
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=None,
            routed_dp_rank=None,
            disagg_prefill_dp_rank=None,
            priority=None,
            extra_key="classification",
            cache_salt="tenant-a",
            http_worker_ipc=None,
        )

        return MMReceiverBase.create_req(receiver, recv_req)

    def test_abort_for_request_awaiting_encoder_has_input_ids(self):
        """A failed encode aborts a request that has no input_ids yet; streaming that abort must not crash the scheduler."""
        req = self._create_req(None)

        self.assertEqual(len(req.origin_input_ids), 0)


class TestEncodeDispatchWithoutEncoder(CustomTestCase):
    """With no encoder registered, a multimodal request fails instead of encoding locally."""

    @staticmethod
    def _dispatch(mm_data):
        receiver = SimpleNamespace(
            encode_urls=[], _extract_url_data=lambda obj: mm_data
        )
        obj = SimpleNamespace(rid="request-1", need_wait_for_mm_inputs=True)
        result = MMReceiverBase._send_encode_request(receiver, obj)
        return obj, result

    def test_rejects_mm_request(self):
        with self.assertRaises(HTTPException) as raised:
            self._dispatch(["http://example.invalid/a.png"])
        self.assertEqual(raised.exception.status_code, HTTPStatus.SERVICE_UNAVAILABLE)

    def test_text_only_request_passes_without_waiting(self):
        obj, result = self._dispatch([])
        self.assertIsNone(result)
        self.assertFalse(obj.need_wait_for_mm_inputs)


class TestHoldReadyRequests(CustomTestCase):
    """Ready requests wait for batch-mates still receiving only while decode runs, and only for a bounded number of ticks."""

    @staticmethod
    def _hold(statuses, held_ticks, *, receiving=None, decoding=True, max_ticks=4):
        if receiving is None:
            receiving = [s == WaitingMMRequestStatus.PENDING for s in statuses]
        receiver = SimpleNamespace(
            scheduler=SimpleNamespace(
                running_batch=SimpleNamespace(is_empty=lambda: not decoding)
            ),
            waiting_list=[SimpleNamespace(held_ticks=t) for t in held_ticks],
        )
        with envs.SGLANG_ENCODER_RECEIVER_COALESCE_TICKS.override(max_ticks):
            return MMReceiverBase._hold_ready_requests(
                receiver,
                torch.tensor([int(x) for x in statuses], dtype=torch.int32),
                torch.tensor(receiving),
            )

    def test_holds_ready_while_a_batch_mate_is_pending(self):
        S = WaitingMMRequestStatus
        self.assertTrue(self._hold([S.SUCCESS, S.PENDING], [0, 0]))

    def test_releases_when_pending_mates_only_wait_for_pool_space(self):
        """Held requests may be what fills the embedding pool; waiting on a pool-blocked request cannot help."""
        S = WaitingMMRequestStatus
        self.assertFalse(
            self._hold([S.SUCCESS, S.PENDING], [0, 0], receiving=[False, False])
        )

    def test_releases_once_nothing_is_pending(self):
        S = WaitingMMRequestStatus
        self.assertFalse(self._hold([S.SUCCESS, S.SUCCESS], [2, 0]))

    def test_releases_when_no_decode_would_be_interrupted(self):
        S = WaitingMMRequestStatus
        self.assertFalse(self._hold([S.SUCCESS, S.PENDING], [0, 0], decoding=False))

    def test_releases_after_the_tick_budget(self):
        S = WaitingMMRequestStatus
        self.assertFalse(self._hold([S.SUCCESS, S.PENDING], [4, 0]))

    def test_zero_ticks_disables_holding(self):
        S = WaitingMMRequestStatus
        self.assertFalse(self._hold([S.SUCCESS, S.PENDING], [0, 0], max_ticks=0))


if __name__ == "__main__":
    unittest.main()
