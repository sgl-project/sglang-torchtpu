"""prefill_driven reservation: the prefill picks which requests decodes reserve
KV for, and Raiden carries the request (RESERVE) and the decode's answer, a
pull request or BUSY."""

import errno
import pkgutil
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver, PrefillServerInfo
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.prefill_driven import validate
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager, RaidenKVReceiver
from sglang_tpu.disaggregation.raiden.receiver import prefill_driven_peer_mismatch
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.disaggregation.raiden.serde import (
    MessageTag,
    PullReq,
    Reserve,
    ReserveBusy,
)
from sglang_tpu.disaggregation.raiden.state import DecodeState, PrefillState
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks import prefill_driven

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOM = 7
DECODE_IP = "10.0.0.2"
DECODE_PORT = 5555
DECODE_ENDPOINT = f"tcp://{DECODE_IP}:{DECODE_PORT}"


def _sender(decode="tcp://decode:1"):
    sender = Mock()
    sender.reservation_busy.return_value = False
    sender.decode_endpoint.return_value = decode
    return sender


def _req(rid, prompt_tokens, decode="tcp://decode:1"):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[0] * prompt_tokens,
        extend_range=None,
        disagg_kv_sender=_sender(decode),
    )


class _Req:
    """Hashable and weakly referenceable, as the hooks' per-request state needs."""

    def __init__(self, rid, prompt_tokens=0, **fields):
        self.rid = rid
        self.origin_input_ids = [0] * prompt_tokens
        self.output_ids = []
        self.num_matched_prefix_tokens = 0
        self.arrival_processed_tokens = 0
        self.time_stats = Mock(wait_queue_entry_time=0)
        self.kv = SimpleNamespace(
            holds_mamba=False,
            holds_kv=False,
            mamba_cow_src_index=None,
            mamba_needs_clear=False,
        )
        self.session = None
        self.__dict__.update(fields)


class TestValidatePrefillDriven(unittest.TestCase):
    def setUp(self):
        enabled = tpu_envs.SGLANG_ENABLE_DISAGG_PREFILL_DRIVEN.override(True)
        enabled.__enter__()
        self.addCleanup(enabled.__exit__, None, None, None)

    def _args(self, **overrides):
        values = dict(
            enable_priority_scheduling=False,
            disable_priority_preemption=False,
            enable_lora_overlap_loading=False,
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="raiden",
            disaggregation_decode_allocation_policy="early",
            pp_size=1,
            enable_pdmux=False,
            decoupled_spec_role="null",
            enable_pd_role_switch=False,
            speculative_algorithm=None,
            encoder_only=False,
            language_only=False,
            disaggregation_decode_host_receive_threshold=0,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_supported_configuration_passes(self):
        validate(self._args())

    def test_rejects_preemption_and_lora_overlap_but_allows_priority_sort(self):
        with self.assertRaisesRegex(ValueError, "priority preemption"):
            validate(
                self._args(
                    enable_priority_scheduling=True, disable_priority_preemption=False
                )
            )
        validate(
            self._args(
                enable_priority_scheduling=True, disable_priority_preemption=True
            )
        )
        with self.assertRaisesRegex(ValueError, "LoRA overlap"):
            validate(self._args(enable_lora_overlap_loading=True))

    def test_negative_slots_are_rejected(self):
        with tpu_envs.SGLANG_DISAGG_PREFILL_DRIVEN_SLOTS.override(-1):
            with self.assertRaisesRegex(ValueError, "SLOTS.*nonnegative"):
                validate(self._args())

    def test_disabled_accepts_anything(self):
        with tpu_envs.SGLANG_ENABLE_DISAGG_PREFILL_DRIVEN.override(False):
            validate(self._args(pp_size=2))

    def test_rejects_what_the_reservation_cannot_cover(self):
        for bad in (
            {"disaggregation_mode": "null"},
            {"disaggregation_transfer_backend": "mooncake"},
            {"disaggregation_decode_allocation_policy": "prefill_complete"},
            {"pp_size": 2},
            {"enable_pdmux": True},
            {"enable_pd_role_switch": True},
            {"speculative_algorithm": "DSPARK"},
            {"encoder_only": True},
        ):
            with self.subTest(bad), self.assertRaises(ValueError):
                validate(self._args(**bad))

    def test_rejects_staging_buffers_and_the_rust_registry(self):
        for env in (envs.SGLANG_DISAGG_STAGING_BUFFER, envs.SGLANG_RUST_SERVER):
            with self.subTest(env), env.override(True):
                with self.assertRaises(ValueError):
                    validate(self._args())


class TestHookTargets(unittest.TestCase):
    def test_no_hooks_when_disabled(self):
        with tpu_envs.SGLANG_ENABLE_DISAGG_PREFILL_DRIVEN.override(False):
            self.assertEqual(prefill_driven.hook_specs(), [])

    def test_every_target_resolves(self):
        """A renamed core method would leave the feature silently half-applied."""
        with tpu_envs.SGLANG_ENABLE_DISAGG_PREFILL_DRIVEN.override(True):
            specs = prefill_driven.hook_specs()
        self.assertEqual(len(specs), 11)
        for target, _, _ in specs:
            with self.subTest(target):
                pkgutil.resolve_name(target)


class TestPeerLayout(unittest.TestCase):
    def _info(self, *, attn_tp_size, sources=1):
        return PrefillServerInfo(
            attn_tp_size=attn_tp_size,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=None,
            kv_cache_dtype=None,
            follow_bootstrap_room=True,
            decode_allocation_policy="early",
            required_prefill_response_num=sources,
        )

    def test_equal_tp_pairs(self):
        self.assertIsNone(
            prefill_driven_peer_mismatch(
                self._info(attn_tp_size=8), decode_attn_tp_size=8
            )
        )

    def test_unequal_tp_or_several_sources_fail_at_init(self):
        """One RESERVE per decode rank cannot ready a rank fed by several
        prefill ranks, or ranks that share one prefill rank."""
        for info, decode_tp in (
            (self._info(attn_tp_size=8, sources=2), 4),
            (self._info(attn_tp_size=4), 8),
            (self._info(attn_tp_size=8, sources=2), 8),
        ):
            with self.subTest(info=info, decode_tp=decode_tp):
                reason = prefill_driven_peer_mismatch(
                    info, decode_attn_tp_size=decode_tp
                )
                self.assertIn("equal attention TP", reason)


def _decode_req(rid, *, ready, receiver_cls=RaidenKVReceiver):
    receiver = Mock(spec=receiver_cls)
    receiver.conclude_state = None
    return SimpleNamespace(waiting_for_input=ready, kv_receiver=receiver, req=_Req(rid))


class TestDecodeAnswersBusy(unittest.TestCase):
    def _queue(self, reqs):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = list(reqs)
        queue.scheduler = SimpleNamespace()
        return queue

    def test_asked_rooms_the_pass_left_behind_answer_busy(self):
        """Core admits in order and stops at the first room that does not fit.
        Every asked room still in the queue after the pass, fitting or not,
        answers BUSY so the prefill can ask for others now and these later."""
        ready, behind = (
            _decode_req("ready", ready=True),
            _decode_req("behind", ready=True),
        )
        unasked = _decode_req("unasked", ready=False)
        queue = self._queue([ready, behind, unasked])

        prefill_driven.answer_busy_for_unadmitted(([], []), queue)

        for decode_req in (ready, behind):
            decode_req.kv_receiver.reject_reservation.assert_called_once_with()
            self.assertFalse(decode_req.waiting_for_input)
        unasked.kv_receiver.reject_reservation.assert_not_called()

    def test_busy_releases_mamba_cow_state_before_answering(self):
        self._check_busy_mamba_cleanup(session=None)

    def test_busy_releases_new_cow_state_on_first_turn_session(self):
        self._check_busy_mamba_cleanup(session=SimpleNamespace())

    def _check_busy_mamba_cleanup(self, session):
        import torch

        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req("mamba-busy", "", [1, 2, 3], SamplingParams(max_new_tokens=1))
        req.session = session
        req.kv.mamba_pool_idx = torch.tensor(7)
        req.kv.mamba_cow_src_index = 2
        req.kv.mamba_needs_clear = True
        receiver = Mock(spec=RaidenKVReceiver)
        queued = SimpleNamespace(req=req, kv_receiver=receiver, waiting_for_input=True)
        queue = self._queue([queued])
        queue.tree_cache = Mock()
        queue.tree_cache.supports_mamba.return_value = True
        free = queue.tree_cache.req_to_token_pool.mamba_allocator.free

        def busy():
            free.assert_called_once()
            self.assertEqual(free.call_args.args[0].tolist(), [7])
            self.assertFalse(req.kv.holds_mamba)
            self.assertIsNone(req.kv.mamba_cow_src_index)
            self.assertFalse(req.kv.mamba_needs_clear)

        receiver.reject_reservation.side_effect = busy
        prefill_driven.answer_busy_for_unadmitted(([], []), queue)
        self.assertFalse(queued.waiting_for_input)

    def test_fake_transfer_room_is_left_alone(self):
        """A warmup room reserves at arrival and has no prefill to answer."""
        fake = SimpleNamespace(
            waiting_for_input=True,
            kv_receiver=SimpleNamespace(conclude_state=None),
            req=_Req("fake"),
        )

        prefill_driven.answer_busy_for_unadmitted(([], []), self._queue([fake]))

        self.assertTrue(fake.waiting_for_input)

    def test_retracted_request_fails_instead_of_waiting_for_a_recompute(self):
        """A retract-mode pause hands decode requests back for a prefill
        recompute that prefill_driven never asks for. They must fail at once."""
        queue = self._queue([])
        queue.scheduler.output_streamer = Mock()
        req = SimpleNamespace(rid="r", return_logprob=False)
        original = Mock()

        with patch("sglang.srt.disaggregation.utils.prepare_abort") as abort:
            prefill_driven.fail_rebootstrap(original, queue, req, is_rebootstrap=True)

        original.assert_not_called()
        abort.assert_called_once()
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], False
        )

    def test_decode_resume_fails_held_rebootstrap_requests_without_requeue(self):
        from http import HTTPStatus

        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.plugins.hook_registry import HookType, _wrap_fn

        queue = self._queue([])
        queue.held_rebootstrap_reqs = []
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = queue
        scheduler.output_streamer = Mock()
        scheduler._engine_paused = True
        queue.scheduler = scheduler
        req = _Req("paused", return_logprob=False)
        queue.hold_rebootstrap(req)
        with patch.object(
            DecodePreallocQueue,
            "add",
            _wrap_fn(
                DecodePreallocQueue.add,
                prefill_driven.fail_rebootstrap,
                HookType.AROUND,
            ),
        ):
            scheduler.continue_generation(SimpleNamespace(torch_empty_cache=False))
        self.assertFalse(scheduler._engine_paused)
        self.assertEqual(queue.held_rebootstrap_reqs, [])
        self.assertEqual(queue.queue, [])
        self.assertEqual(
            req.finished_reason.status_code, HTTPStatus.SERVICE_UNAVAILABLE
        )
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)

    def test_other_requests_are_added_as_usual(self):
        queue = self._queue([])
        original = Mock()

        prefill_driven.fail_rebootstrap(original, queue, "req", is_retracted=True)

        original.assert_called_once_with(
            queue, "req", is_retracted=True, is_rebootstrap=False
        )


def _prefill_manager(prefill_driven=True):
    mgr = RaidenKVManager.__new__(RaidenKVManager)
    mgr._prefill = PrefillState(announce_pages=1)
    mgr._decode = None
    mgr._pipeline = None
    mgr._prefill_driven = prefill_driven
    mgr.attn_tp_rank = 0
    mgr.bootstrap_timeout = 300
    mgr.request_status = {ROOM: KVPoll.Bootstrapping}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._aborted = {}
    mgr._failed = set()
    mgr._send_multipart_locked = Mock()
    mgr._maybe_reply = Mock()
    # Empty native state used by real room cleanup in registration tests.
    mgr._pulling = None
    mgr._reshard_plans = {}
    mgr._cleared_plans = {}
    mgr._send_done_at = {}
    mgr._completed_send = set()
    mgr._completed_recv = set()
    mgr._native_failed = set()
    return mgr


def _bootstrap_frames(room=ROOM):
    return [
        MessageTag.BOOTSTRAP.value,
        str(room).encode(),
        DECODE_IP.encode(),
        str(DECODE_PORT).encode(),
    ]


def _sent_tags(mgr):
    return [call.args[1][0] for call in mgr._send_multipart_locked.call_args_list]


class TestRaidenPrefillReserve(unittest.TestCase):
    def test_endpoint_churn_does_not_retain_retired_workers(self):
        mgr = _prefill_manager()
        for index in range(100):
            room = ROOM + index
            frames = _bootstrap_frames(room)
            frames[3] = str(DECODE_PORT + index).encode()
            mgr._dispatch(frames)
            mgr.clear_room(room)
        self.assertEqual(mgr._prefill.registered_decodes, {})
        self.assertEqual(mgr._prefill.decode_heard_at, {})
        self.assertEqual(mgr._prefill.silent_decodes, set())
        self.assertEqual(mgr._prefill.decode_room_counts, {})

    def test_endpoint_liveness_survives_until_its_last_room_retires(self):
        mgr = _prefill_manager()
        mgr._dispatch(_bootstrap_frames())
        mgr._dispatch(_bootstrap_frames(ROOM + 1))
        mgr._prefill.silent_decodes.add(DECODE_ENDPOINT)
        mgr.clear_room(ROOM)
        self.assertIn(DECODE_ENDPOINT, mgr._prefill.decode_heard_at)
        self.assertIn(DECODE_ENDPOINT, mgr._prefill.silent_decodes)
        mgr.clear_room(ROOM + 1)
        self.assertNotIn(DECODE_ENDPOINT, mgr._prefill.decode_heard_at)
        self.assertNotIn(DECODE_ENDPOINT, mgr._prefill.silent_decodes)

    def test_duplicate_registration_keeps_one_endpoint_owner(self):
        mgr = _prefill_manager()
        mgr._dispatch(_bootstrap_frames())
        mgr._dispatch(_bootstrap_frames())
        mgr.clear_room(ROOM)
        self.assertEqual(mgr._prefill.decode_heard_at, {})

    def test_replacing_endpoint_keeps_other_rooms_and_retires_old_worker(self):
        mgr = _prefill_manager()
        mgr._dispatch(_bootstrap_frames())
        mgr._dispatch(_bootstrap_frames(ROOM + 1))
        frames = _bootstrap_frames()
        frames[3] = str(DECODE_PORT + 1).encode()
        mgr._dispatch(frames)
        new_endpoint = f"tcp://{DECODE_IP}:{DECODE_PORT + 1}"
        self.assertIn(DECODE_ENDPOINT, mgr._prefill.decode_heard_at)
        self.assertIn(new_endpoint, mgr._prefill.decode_heard_at)
        mgr.clear_room(ROOM + 1)
        self.assertNotIn(DECODE_ENDPOINT, mgr._prefill.decode_heard_at)
        self.assertIn(new_endpoint, mgr._prefill.decode_heard_at)
        mgr.clear_room(ROOM)
        self.assertEqual(mgr._prefill.decode_heard_at, {})

    def test_late_registration_cannot_recreate_retired_endpoint_state(self):
        mgr = _prefill_manager()
        mgr._aborted[ROOM] = time.time()
        mgr._dispatch(_bootstrap_frames())
        self.assertEqual(mgr._prefill.registered_decodes, {})
        self.assertEqual(mgr._prefill.decode_heard_at, {})

    def test_late_busy_cannot_recreate_a_retired_room(self):
        for retired in ("aborted", "cleared"):
            with self.subTest(retired=retired):
                mgr = _prefill_manager()
                if retired == "aborted":
                    mgr._aborted[ROOM] = time.time()
                    mgr._prefill.reserve_requested_at[ROOM] = time.time()
                mgr._on_reserve_busy(ReserveBusy(room=ROOM).to_frames())
                self.assertNotIn(ROOM, mgr._prefill.reserve_busy)

    def test_reserve_waits_for_the_decode_to_register(self):
        """The prefill may pick a request before its decode's BOOTSTRAP
        arrives. The RESERVE then goes out on registration."""
        mgr = _prefill_manager()

        mgr.request_decode_reservation(ROOM)
        self.assertEqual(_sent_tags(mgr), [])
        mgr._dispatch(_bootstrap_frames())

        self.assertEqual(_sent_tags(mgr), [Reserve.TAG])
        endpoint, frames = mgr._send_multipart_locked.call_args.args
        self.assertEqual(endpoint, DECODE_ENDPOINT)
        self.assertEqual(Reserve.from_frames(frames).room, ROOM)

    def test_busy_answer_frees_the_decode_until_the_next_ask(self):
        """BUSY stops the silence clock and marks the room for retry
        until the next ask."""
        mgr = _prefill_manager()
        mgr._dispatch(_bootstrap_frames())
        mgr.request_decode_reservation(ROOM)
        mgr._prefill.reserve_requested_at[ROOM] = time.time() - 1000
        mgr._prefill.registered_at[ROOM] = time.time() - 1000
        self.assertTrue(mgr.fail_decode_silence(ROOM, 0.0, 300))

        mgr = _prefill_manager()
        mgr._dispatch(_bootstrap_frames())
        mgr.request_decode_reservation(ROOM)
        mgr._prefill.reserve_requested_at[ROOM] = time.time() - 1000
        mgr._prefill.registered_at[ROOM] = time.time() - 1000
        mgr._on_reserve_busy(ReserveBusy(room=ROOM).to_frames())

        self.assertTrue(mgr.reservation_busy(ROOM))
        self.assertFalse(mgr.fail_decode_silence(ROOM, 0.0, 300))
        # A decode that registers again meanwhile is not asked.
        mgr._dispatch(_bootstrap_frames())
        self.assertEqual(_sent_tags(mgr), [Reserve.TAG])
        mgr.request_decode_reservation(ROOM)
        self.assertFalse(mgr.reservation_busy(ROOM))
        self.assertEqual(_sent_tags(mgr), [Reserve.TAG, Reserve.TAG])

    def test_pull_without_a_reservation_fails_the_room(self):
        """A decode running without the feature reserves at arrival and
        pulls unasked. It must fail with the reason, not run outside the
        window as if the feature were off on both sides."""
        pull = PullReq(
            room=ROOM, decode_ip=DECODE_IP, decode_port=DECODE_PORT, decode_pages=4
        ).to_frames()
        mgr = _prefill_manager()

        mgr._on_pull_req(pull)

        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)
        self.assertIn("never asked", mgr.failure_records[ROOM])
        self.assertEqual(_sent_tags(mgr), [MessageTag.ROOM_FAILED.value])
        self.assertNotIn(ROOM, mgr._prefill.decode_addr)

        mgr = _prefill_manager()
        mgr._dispatch(_bootstrap_frames())
        mgr.request_decode_reservation(ROOM)
        mgr._on_pull_req(pull)
        self.assertEqual(mgr._prefill.decode_addr[ROOM], (DECODE_IP, DECODE_PORT))

    def test_abort_for_an_unopened_room_drops_its_registration(self):
        """A request the prefill rejected at intake never gets a sender. Its
        decode gives up through its own timeout and aborts, which must take
        the registration with it."""
        mgr = _prefill_manager()
        mgr.request_status = {}
        mgr._dispatch(_bootstrap_frames())

        mgr._dispatch([MessageTag.ABORT.value, str(ROOM).encode()])

        self.assertNotIn(ROOM, mgr._prefill.registered_decodes)
        self.assertNotIn(ROOM, mgr._prefill.registered_at)
        self.assertNotIn(DECODE_ENDPOINT, mgr._prefill.decode_heard_at)
        self.assertIn(ROOM, mgr._aborted)

    def test_undeliverable_reserve_fails_the_room(self):
        """A RESERVE this rank cannot deliver fails its own room at once, as
        a CHUNK does, instead of waiting out the decode silence timeout. An
        error that is not about the one peer still reaches the scheduler."""
        mgr = _prefill_manager()
        mgr._dispatch(_bootstrap_frames())
        mgr._send_multipart_locked.side_effect = zmq.ZMQError(zmq.EAGAIN)

        mgr.request_decode_reservation(ROOM)

        self.assertEqual(mgr.request_status[ROOM], KVPoll.Failed)
        self.assertIn(ROOM, mgr._aborted)
        self.assertIn("RESERVE to", mgr.failure_records[ROOM])

        mgr = _prefill_manager()
        mgr._dispatch(_bootstrap_frames())
        mgr._send_multipart_locked.side_effect = zmq.ZMQError(errno.EBADF)
        with self.assertRaises(zmq.ZMQError):
            mgr.request_decode_reservation(ROOM)

    def test_silence_counts_from_arrival_until_the_decode_registers(self):
        """Asking must not restart the clock of a decode that never showed up."""
        mgr = _prefill_manager()
        created = time.time() - 301
        mgr._prefill.reserve_requested_at[ROOM] = time.time()

        self.assertTrue(mgr.fail_decode_silence(ROOM, created, 300))

    def test_silence_message_reports_the_wait_since_the_ask(self):
        mgr = _prefill_manager()
        mgr._prefill.registered_decodes[ROOM] = DECODE_ENDPOINT
        mgr._prefill.reserve_requested_at[ROOM] = time.time() - 301

        self.assertTrue(mgr.fail_decode_silence(ROOM, time.time() - 900, 300))

        self.assertIn("timed out after 301", mgr.failure_records[ROOM])

    def test_silence_counts_from_a_registration_later_than_the_ask(self):
        """An ask that waited for the decode's BOOTSTRAP gives the decode its
        full timeout from the registration."""
        mgr = _prefill_manager()
        mgr._prefill.registered_decodes[ROOM] = DECODE_ENDPOINT
        mgr._prefill.reserve_requested_at[ROOM] = time.time() - 1000
        mgr._prefill.registered_at[ROOM] = time.time() - 50
        self.assertFalse(mgr.fail_decode_silence(ROOM, time.time() - 1000, 300))

        mgr._prefill.registered_at[ROOM] = time.time() - 301
        self.assertTrue(mgr.fail_decode_silence(ROOM, time.time() - 1000, 300))
        self.assertIn("timed out after 301", mgr.failure_records[ROOM])

    def test_silent_decode_fails_the_rooms_it_registered_before(self):
        """A decode that dies after registering N rooms must not drain them
        through N asks of a full timeout each. A room it registers later
        comes from a live decode."""
        mgr = _prefill_manager()
        mgr.request_status = {ROOM: KVPoll.Bootstrapping, 8: KVPoll.Bootstrapping}
        mgr._dispatch(_bootstrap_frames())
        mgr._dispatch(_bootstrap_frames(8))
        mgr.request_decode_reservation(ROOM)
        mgr._prefill.reserve_requested_at[ROOM] = time.time() - 301
        mgr._prefill.registered_at[ROOM] = time.time() - 301
        mgr._prefill.decode_heard_at[DECODE_ENDPOINT] = time.time() - 301
        self.assertFalse(mgr.fail_decode_silence(8, time.time() - 1000, 300))

        self.assertTrue(mgr.fail_decode_silence(ROOM, time.time() - 1000, 300))

        self.assertTrue(mgr.fail_decode_silence(8, time.time(), 300))
        self.assertIn("stopped answering", mgr.failure_records[8])
        mgr.request_status[9] = KVPoll.Bootstrapping
        mgr._dispatch(_bootstrap_frames(9))
        self.assertFalse(mgr.fail_decode_silence(9, time.time(), 300))

    def test_decode_heard_during_the_timeout_is_not_silent(self):
        """A live decode can let one ask time out: restarted at the same
        address it drops the old incarnation's RESERVE, or admission paused.
        Its other rooms keep their own clocks as long as it keeps talking."""
        mgr = _prefill_manager()
        mgr.request_status = {ROOM: KVPoll.Bootstrapping, 8: KVPoll.Bootstrapping}
        mgr._dispatch(_bootstrap_frames())
        mgr._dispatch(_bootstrap_frames(8))
        mgr.request_decode_reservation(ROOM)
        mgr.request_decode_reservation(8)
        mgr._prefill.reserve_requested_at[ROOM] = time.time() - 301
        mgr._prefill.registered_at[ROOM] = time.time() - 301
        mgr._on_reserve_busy(ReserveBusy(room=8).to_frames())

        self.assertTrue(mgr.fail_decode_silence(ROOM, time.time() - 1000, 300))

        self.assertNotIn(DECODE_ENDPOINT, mgr._prefill.silent_decodes)
        self.assertFalse(mgr.fail_decode_silence(8, time.time(), 300))

    def test_decode_that_registers_again_is_asked_again(self):
        """A restarted decode or a retraction re-bootstrap lost the RESERVE."""
        mgr = _prefill_manager()
        mgr._dispatch(_bootstrap_frames())
        mgr.request_decode_reservation(ROOM)

        mgr._dispatch(_bootstrap_frames())

        self.assertEqual(_sent_tags(mgr), [Reserve.TAG, Reserve.TAG])

    def test_decode_that_registers_again_gets_a_full_timeout(self):
        """The earlier life's clock must not fail the decode just re-asked."""
        mgr = _prefill_manager()
        mgr._dispatch(_bootstrap_frames())
        mgr.request_decode_reservation(ROOM)
        mgr._prefill.reserve_requested_at[ROOM] = time.time() - 1000
        mgr._prefill.registered_at[ROOM] = time.time() - 1000
        self.assertTrue(
            mgr._prefill_driven_silence_start_locked(ROOM, 0.0) < time.time() - 300
        )

        mgr._dispatch(_bootstrap_frames())

        self.assertFalse(mgr.fail_decode_silence(ROOM, time.time() - 1000, 300))

    def test_room_reused_after_cleanup_is_asked_again(self):
        mgr = _prefill_manager()
        mgr._dispatch(_bootstrap_frames())
        mgr.request_decode_reservation(ROOM)
        # Transfer state the real cleanup walks, empty for a room never pulled.
        mgr._pulling = None
        mgr._reshard_plans = {}
        mgr._cleared_plans = {}
        mgr._send_done_at = {}
        for name in ("_completed_send", "_completed_recv", "_native_failed"):
            setattr(mgr, name, set())

        mgr.clear_room(ROOM)
        mgr._dispatch(_bootstrap_frames())
        self.assertEqual(_sent_tags(mgr), [Reserve.TAG])
        mgr.request_decode_reservation(ROOM)

        self.assertEqual(_sent_tags(mgr), [Reserve.TAG, Reserve.TAG])
        frames = mgr._send_multipart_locked.call_args.args[1]
        self.assertEqual(Reserve.from_frames(frames).room, ROOM)

    def test_decode_is_told_when_its_silence_fails_the_room(self):
        """Otherwise it waits out its reservation timeout for a dead room."""
        sender = RaidenKVSender.__new__(RaidenKVSender)
        sender.kv_mgr = Mock()
        sender.kv_mgr.is_dummy_cp_rank = False
        sender.kv_mgr._prefill_driven = True
        sender.kv_mgr.is_failed.return_value = False
        sender.kv_mgr.is_send_done.return_value = False
        sender.kv_mgr.check_status.return_value = KVPoll.Bootstrapping
        sender.kv_mgr.finish_bootstrap_if_decode_requested.return_value = False
        sender.kv_mgr.fail_decode_silence.return_value = True
        sender.bootstrap_room = ROOM
        sender.init_time = 0.0
        sender.conclude_state = None
        sender._decode_told = False

        self.assertEqual(sender.poll(), KVPoll.Failed)

        sender.kv_mgr.notify_decode_room_failed.assert_called_once_with(ROOM)

    def test_early_prefill_rejects_a_prefill_driven_decode(self):
        mgr = _prefill_manager(prefill_driven=False)

        mgr._dispatch(_bootstrap_frames())

        self.assertEqual(_sent_tags(mgr), [MessageTag.ROOM_FAILED.value])

    def test_decode_silence_clock_follows_the_reservation(self):
        """Before registering, the decode owes its BOOTSTRAP. Registered and
        not yet asked, it owes nothing however long the prefill queue is.
        Once asked, it owes the pull request within the bootstrap timeout."""
        created = time.time() - 1000
        mgr = _prefill_manager()
        self.assertTrue(mgr.fail_decode_silence(ROOM, created, 300))

        mgr = _prefill_manager()
        mgr._dispatch(_bootstrap_frames())
        self.assertFalse(mgr.fail_decode_silence(ROOM, created, 300))

        mgr.request_decode_reservation(ROOM)
        mgr._prefill.reserve_requested_at[ROOM] = time.time() - 301
        mgr._prefill.registered_at[ROOM] = time.time() - 301
        self.assertTrue(mgr.fail_decode_silence(ROOM, created, 300))

    def test_failure_reaches_a_registered_decode_before_its_pull(self):
        """Cancelling a request the decode has not reserved yet must free the
        decode's queue entry instead of leaving it to the long reserve wait."""
        mgr = _prefill_manager()
        mgr._dispatch(_bootstrap_frames())

        mgr.notify_decode_room_failed(ROOM)

        self.assertEqual(_sent_tags(mgr), [MessageTag.ROOM_FAILED.value])


def _decode_manager():
    mgr = RaidenKVManager.__new__(RaidenKVManager)
    mgr._prefill = None
    mgr._decode = DecodeState()
    mgr._prefill_driven = True
    mgr.request_status = {ROOM: KVPoll.Bootstrapping}
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._aborted = {}
    return mgr


def _decode_receiver(mgr, *, started_s_ago):
    recv = RaidenKVReceiver.__new__(RaidenKVReceiver)
    recv.kv_mgr = mgr
    recv.bootstrap_room = ROOM
    recv.init_time = None
    recv._handshake_start = time.time() - started_s_ago
    recv._handshake_timeout = 300
    return recv


class TestDummyCPReservation(unittest.TestCase):
    def test_non_transfer_rank_is_neutral_in_all_rank_busy_reduction(self):
        sender = RaidenKVSender.__new__(RaidenKVSender)
        sender.bootstrap_room = ROOM
        sender.kv_mgr = Mock(is_dummy_cp_rank=True)
        sender.kv_mgr.reservation_busy.return_value = False
        self.assertTrue(sender.reservation_busy())
        sender.request_decode_reservation()
        sender.kv_mgr.request_decode_reservation.assert_not_called()


class TestRaidenDecodeReserve(unittest.TestCase):
    def test_unasked_wait_keeps_existing_bootstrap_deadline(self):
        mgr = _decode_manager()
        recv = _decode_receiver(mgr, started_s_ago=10000)
        self.assertIn("timed out", recv._check_handshake_timeout(KVPoll.Bootstrapping))
        self.assertIn(
            "timed out", recv._check_handshake_timeout(KVPoll.WaitingForInput)
        )
        recent = _decode_receiver(mgr, started_s_ago=0)
        self.assertIsNone(recent._check_handshake_timeout(KVPoll.Bootstrapping))

    def test_reserve_turns_the_room_ready(self):
        mgr = _decode_manager()

        mgr._on_reserve(Reserve(room=ROOM).to_frames())

        self.assertEqual(mgr.request_status[ROOM], KVPoll.WaitingForInput)
        self.assertIn(ROOM, mgr._decode.reserve_held)

    def test_rejecting_answers_busy_and_holds_the_room_for_the_next_ask(self):
        mgr = _decode_manager()
        mgr._send_multipart_locked = Mock()
        mgr._on_reserve(Reserve(room=ROOM).to_frames())

        mgr.reject_reservation(ROOM, ["tcp://p0:1", "tcp://p1:1"])
        mgr.reject_reservation(ROOM, ["tcp://p0:1"])

        self.assertEqual(mgr.request_status[ROOM], KVPoll.Bootstrapping)
        self.assertNotIn(ROOM, mgr._decode.reserve_held)
        sent = mgr._send_multipart_locked.call_args_list
        self.assertEqual([call.args[0] for call in sent], ["tcp://p0:1", "tcp://p1:1"])
        self.assertEqual(ReserveBusy.from_frames(sent[0].args[1]).room, ROOM)

    def test_reserve_does_not_revive_an_aborted_room(self):
        mgr = _decode_manager()
        mgr._aborted[ROOM] = time.time()

        mgr._on_reserve(Reserve(room=ROOM).to_frames())

        self.assertEqual(mgr.request_status[ROOM], KVPoll.Bootstrapping)


class TestDecodeQueueReserveBusyReserve(unittest.TestCase):
    """Core's handshake waiter sets waiting_for_input from the poll and never
    clears it on Bootstrapping, so the BUSY hook must. The room then goes
    ready again on the prefill's next RESERVE."""

    def test_room_is_ready_only_while_a_reserve_is_held(self):
        mgr = _decode_manager()
        mgr._send_multipart_locked = Mock()
        mgr._drain_engine = Mock()
        mgr.is_failed = Mock(return_value=False)
        mgr.check_status = lambda room: mgr.request_status[room]
        mgr.is_recv_done = Mock(return_value=False)
        mgr.maybe_start_pull = Mock(return_value=False)
        recv = _decode_receiver(mgr, started_s_ago=0)
        recv.conclude_state = None
        recv._started = False
        recv.bootstrap_infos = [{"rank_ip": "10.0.0.1", "rank_port": 1}]
        decode_req = SimpleNamespace(
            waiting_for_input=False,
            kv_receiver=recv,
            req=_Req("r", bootstrap_room=ROOM, time_stats=Mock()),
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]
        queue.pp_size = 1
        queue.gloo_group = None
        queue.tp_rank = 0

        def waiters():
            with patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                side_effect=lambda receivers, *_, **__: [r.poll() for r in receivers],
            ):
                queue._update_handshake_waiters()
            return decode_req.waiting_for_input

        self.assertFalse(waiters())
        mgr._on_reserve(Reserve(room=ROOM).to_frames())
        self.assertTrue(waiters())

        prefill_driven.answer_busy_for_unadmitted(([], []), queue)
        self.assertEqual(_sent_tags(mgr), [ReserveBusy.TAG])
        self.assertFalse(waiters())

        mgr._on_reserve(Reserve(room=ROOM).to_frames())
        self.assertTrue(waiters())


class TestRaidenDecodeRegistration(unittest.TestCase):
    """Core's early policy opens a decode room ready to reserve. Under
    prefill_driven Raiden must hold it until the prefill asks."""

    def _receiver(self, *, prefill_attn_tp=8, sources=1):
        mgr = _decode_manager()
        mgr.attn_tp_size = 8
        mgr.local_ip = DECODE_IP
        mgr.rank_port = DECODE_PORT
        mgr._mark_aborted_locked = Mock()
        mgr._fail_room_unless_failed_locked = Mock()
        mgr.room_has_active_io = Mock(return_value=False)
        mgr.enable_deferred_decode_kv_release = False
        recv = _decode_receiver(mgr, started_s_ago=0)
        recv.conclude_state = None
        recv.abort_notified = False
        recv._abort_generation = None
        recv.prefill_info = SimpleNamespace(
            attn_tp_size=prefill_attn_tp, required_prefill_response_num=sources
        )
        recv.bootstrap_infos = [{"rank_ip": "10.0.0.1", "rank_port": 1}]
        self.sock = Mock()
        recv._connect_to_bootstrap_server = Mock(
            return_value=(self.sock, threading.Lock())
        )
        return recv

    def _init(self, recv):
        def early_init(receiver, prefill_dp_rank):
            receiver.kv_mgr.request_status[ROOM] = KVPoll.WaitingForInput

        with patch.object(CommonKVReceiver, "init", early_init):
            recv.init(prefill_dp_rank=0)

    def test_room_waits_for_reserve_and_registers_with_the_prefill(self):
        recv = self._receiver()

        self._init(recv)

        self.assertEqual(recv.kv_mgr.request_status[ROOM], KVPoll.Bootstrapping)
        frames = self.sock.send_multipart.call_args.args[0]
        self.assertEqual(frames[0], b"BOOTSTRAP")
        self.assertEqual(int(frames[1]), ROOM)

    def test_unequal_tp_fails_and_tells_the_prefill(self):
        """The prefill must drop the room now, not after its bootstrap
        timeout with a log that blames a silent decode."""
        recv = self._receiver(prefill_attn_tp=4)

        self._init(recv)

        self.assertEqual(recv.conclude_state, KVPoll.Failed)
        sent = [call.args[0][0] for call in self.sock.send_multipart.call_args_list]
        self.assertEqual(sent, [MessageTag.ABORT.value])

    def test_rejecting_answers_every_prefill_rank_of_the_room(self):
        recv = self._receiver()
        recv.bootstrap_infos = [
            {"rank_ip": "10.0.0.1", "rank_port": 1},
            {"rank_ip": "10.0.0.1", "rank_port": 2},
        ]
        recv.kv_mgr._send_multipart_locked = Mock()
        recv.kv_mgr._on_reserve(Reserve(room=ROOM).to_frames())

        recv.reject_reservation()

        sent = recv.kv_mgr._send_multipart_locked.call_args_list
        self.assertEqual(
            [call.args[0] for call in sent], ["tcp://10.0.0.1:1", "tcp://10.0.0.1:2"]
        )

    def test_disabled_keeps_core_early_behavior(self):
        recv = self._receiver()
        recv.kv_mgr._prefill_driven = False

        self._init(recv)

        self.assertEqual(recv.kv_mgr.request_status[ROOM], KVPoll.WaitingForInput)
        self.sock.send_multipart.assert_not_called()


if __name__ == "__main__":
    unittest.main()
