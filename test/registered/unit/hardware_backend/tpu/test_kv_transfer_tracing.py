"""CPU unit tests for Raiden P/D KV-cache transfer tracing spans."""

from __future__ import annotations

import contextlib
import pkgutil
import sys
import threading
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.common import (
    clean_req_id,
    is_profiler_enabled,
    record_trace_meta,
)
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.pool_transfer import RaidenPoolTransfer
from sglang_tpu.disaggregation.raiden.state import DecodeState, PrefillState
from sglang_tpu.hooks.raiden_trace_request_ids import bind_trace_request_id, hook_specs

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _settled(ids):
    return [
        SimpleNamespace(
            req_id=r,
            completed_ns=None,
            code=None,
            phase=None,
            peer=None,
            elapsed_ns=None,
            reason=None,
        )
        for r in ids
    ]


def _poll(sent=(), received=(), failed=()):
    return SimpleNamespace(
        sent=_settled(sent), received=_settled(received), failed=_settled(failed)
    )


@contextlib.contextmanager
def _capture_spans():
    with (
        patch.object(torch.autograd.profiler, "_is_profiler_enabled", True),
        patch("torch.profiler.record_function") as mock_rf,
    ):
        labels: list[str] = []
        yield labels
        labels.extend(c.args[0] for c in mock_rf.call_args_list)


def _start_read(mgr, engine, room, req_id, uuid, remote, local, port=31000):
    return mgr._start_read_or_fail(
        room=room,
        engine=engine,
        req_id=req_id,
        uuid=uuid,
        endpoint=f"tcp://127.0.0.1:{port}",
        remote=remote,
        local=local,
    )


class TestRaidenKVTransferTracing(CustomTestCase):
    def _make_manager(self, page_size: int = 16) -> tuple[RaidenKVManager, MagicMock]:
        mgr = object.__new__(RaidenKVManager)
        engine = MagicMock()
        engine.register_read.return_value = True
        mgr.__dict__.update(
            attn_tp_rank=0,
            attn_tp_size=2,
            kv_args=SimpleNamespace(page_size=page_size, total_kv_head_num=4),
            _state_lock=threading.Lock(),
            _status_lock=threading.Lock(),
            _drain_lock=threading.Lock(),
            _reshard_push_lock=threading.Lock(),
            failure_lock=threading.Lock(),
            failure_records={},
            request_status={},
            dram_offload_channel=None,
            chunked_transfer=False,
            _timeout_s=10.0,
            _transfer_parallelism=1,
            _monotonic=lambda: 0.0,
            _aborted={},
            _send_reads={},
            _send_done_at={},
            _cleared_plans={},
            _reshard_plans={},
            _engine_poll_failing_since={},
            _room_request_ids={},
            _transfer_trace_meta={},
            _completed_send=set(),
            _completed_recv=set(),
            _failed=set(),
            _native_failed=set(),
            _superseded_sender_reads=set(),
            _pulling_cleared=set(),
            _stale_plans=[],
            _pipeline=None,
            _pulling=None,
            _prefill=None,
            _decode=None,
            _swa_engine=None,
            _state_engine=None,
            _draft_engine=None,
            _kv_caches=[object(), object()],
            _swa_kv_caches=[object()],
            _engine=engine,
        )
        return mgr, engine

    def test_clean_req_id_sanitizes_metadata_delimiters(self):
        self.assertEqual(clean_req_id(" req,id#1=foo bar "), "req_id_1_foo_bar")
        self.assertEqual(clean_req_id("   "), "")

    def test_real_torch_profiler_captures_spans_and_clears_across_threads(self):
        mgr, engine = self._make_manager(page_size=16)
        mgr.register_trace_request_id(42, "req-alpha#1")
        engine.poll_stats_with_details.return_value = _poll(sent=["42#0"])

        bg_during, bg_after = [], []
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            mgr._register_sender_read(42, engine, "42#0", 9001, [10, 11, 12])
            t = threading.Thread(target=lambda: bg_during.append(is_profiler_enabled()))
            t.start()
            t.join()
            mgr._drain_engine()

        t = threading.Thread(target=lambda: bg_after.append(is_profiler_enabled()))
        t.start()
        t.join()

        self.assertEqual((bg_during, bg_after), ([True], [False]))
        event_names = {evt.key for evt in prof.key_averages()}
        self.assertIn(
            "KV_Cache_Prefill_Register_Blocks#uuid=9001,request_id=req-alpha_1,num_tokens=48,num_blocks=3#",
            event_names,
        )
        self.assertIn(
            "KV_Cache_Prefill_Send_Complete#uuid=9001,request_id=req-alpha_1,num_tokens=48#",
            event_names,
        )

    def test_request_id_hooks_resolve_bind_and_clear_room_releases(self):
        for target, _, _ in hook_specs():
            obj_path, attr = target.rsplit(".", 1)
            self.assertTrue(hasattr(pkgutil.resolve_name(obj_path), attr), target)

        mgr, engine = self._make_manager(page_size=16)
        engine.poll_stats_with_details.return_value = _poll(received=["42"])
        queue = SimpleNamespace(kv_manager=mgr)
        req = SimpleNamespace(bootstrap_room=42, rid="req,beta=2")
        self.assertEqual(bind_trace_request_id("ok", queue, req), "ok")

        with _capture_spans() as labels:
            self.assertTrue(_start_read(mgr, engine, 42, "42", 42, [1, 2], [5, 6]))
            mgr._drain_engine()

        self.assertEqual(
            labels,
            [
                "KV_Cache_Decode_Submit_Load#uuid=42,request_id=req_beta_2,num_tokens=32,num_blocks=2#",
                "KV_Cache_Decode_Recv_Complete#uuid=42,request_id=req_beta_2,num_tokens=32#",
            ],
        )
        mgr.clear_status = MagicMock()
        mgr.clear_room(42)
        self.assertNotIn(42, mgr._room_request_ids)
        self.assertEqual(mgr._transfer_trace_meta, {})

    def test_fallback_to_raiden_req_id_and_state_zero_tokens(self):
        mgr, engine = self._make_manager(page_size=16)
        mgr.register_trace_request_id(7, "req-state")
        engine.poll_stats_with_details.return_value = _poll(
            sent=["42#0", "state:7"], received=["state:7"]
        )

        with _capture_spans() as labels:
            mgr._register_sender_read(42, engine, "42#0", 9001, [10, 11, 12])
            mgr._register_sender_read(7, engine, "state:7", 7, [3])
            _start_read(mgr, engine, 7, "state:7", 7, [3], [8], port=31004)
            mgr._drain_engine()

        self.assertEqual(
            labels,
            [
                "KV_Cache_Prefill_Register_Blocks#uuid=9001,request_id=42_0,num_tokens=48,num_blocks=3#",
                "KV_Cache_Prefill_Register_Blocks#uuid=7,request_id=req-state,num_tokens=0,num_blocks=1#",
                "KV_Cache_Decode_Submit_Load#uuid=7,request_id=req-state,num_tokens=0,num_blocks=1#",
                "KV_Cache_Prefill_Send_Complete#uuid=9001,request_id=42_0,num_tokens=48#",
                "KV_Cache_Prefill_Send_Complete#uuid=7,request_id=req-state,num_tokens=0#",
                "KV_Cache_Decode_Recv_Complete#uuid=7,request_id=req-state,num_tokens=0#",
            ],
        )

    def test_noop_and_no_meta_writes_when_profiler_disabled(self):
        mgr, engine = self._make_manager(page_size=16)
        engine.poll_stats_with_details.return_value = _poll(
            sent=["42"], received=["42"]
        )

        with patch("torch.profiler.record_function") as mock_rf:
            mgr._register_sender_read(42, engine, "42", 42, [10])
            _start_read(mgr, engine, 42, "42", 42, [10], [20])
            self.assertEqual(mgr._transfer_trace_meta, {})
            mgr._drain_engine()

        mock_rf.assert_not_called()
        self.assertEqual(mgr._transfer_trace_meta, {})

    def test_failed_and_cleared_registrations_do_not_leak_trace_meta(self):
        mgr, engine = self._make_manager(page_size=16)
        mgr._decode = DecodeState(reshard_meta={10: {}})
        for attr in (
            "_unregister_plan",
            "_send_multipart_locked",
            "record_failure",
            "update_status",
            "_release_pull_slots_locked",
        ):
            setattr(mgr, attr, MagicMock())

        with _capture_spans():
            engine.register_read.return_value = False
            mgr._register_sender_read(1, engine, "1", 1, [10])
            self.assertEqual(mgr._transfer_trace_meta, {})

            engine.start_read.side_effect = RuntimeError("boom")
            self.assertFalse(_start_read(mgr, engine, 2, "2", 2, [1], [2]))
            self.assertEqual(mgr._transfer_trace_meta, {})

            engine.register_active_plan.side_effect = lambda *_a, **_k: (
                mgr._decode.reshard_meta.pop(10, None)
            )
            self.assertFalse(
                mgr._publish_reshard_plan(
                    room=10,
                    req_id="10",
                    uuid=10,
                    req=SimpleNamespace(dst_block_ids=[1, 2]),
                    ready_payload=[b"ready"],
                    ready_endpoints=["tcp://127.0.0.1:31000"],
                )
            )
            self.assertEqual(mgr._transfer_trace_meta, {})

            engine.register_read.return_value = True
            mgr._register_sender_read(3, engine, "3", 3, [5])
            self.assertIn("3", mgr._transfer_trace_meta)
            engine.poll_stats_with_details.return_value = _poll(failed=["3"])
            mgr._drain_engine()
            self.assertEqual(mgr._transfer_trace_meta, {})

    def test_reshard_receiver_and_sender_emit_complete_pairs(self):
        mgr, engine = self._make_manager(page_size=16)
        mgr.register_trace_request_id(55, "req-reshard")
        mgr._decode = DecodeState(reshard_meta={55: {}})
        mgr._send_multipart_locked = MagicMock()
        mgr._prefill = PrefillState(
            announce_pages=1,
            reshard_req={
                55: {
                    "dst_attn_tp": 1,
                    "push_endpoint": "tcp://127.0.0.1:32000",
                    "decode_blocks": [0, 1],
                    "decode_swa_blocks": [4],
                    "swa_push_endpoint": "tcp://127.0.0.1:32002",
                }
            },
            send_blocks={55: [10, 11]},
            reshard_ready_pages={55: [0, 1]},
        )
        mgr._make_geometry = MagicMock(return_value=SimpleNamespace(ratio=2))
        ready_fut = SimpleNamespace(is_ready=lambda: True, ok=lambda: True)
        engine.plan_host_blocks.return_value = [0, 1]
        engine.d2h.return_value = ready_fut
        swa_engine = MagicMock(transfer_address="tcp://127.0.0.1:32002")
        swa_engine.plan_host_blocks.return_value = [0]
        swa_engine.d2h.return_value = ready_fut
        mgr._swa_engine = swa_engine

        fake_pb2 = ModuleType("tpu_sync.rpc.raiden_service_pb2")
        fake_rpc = SimpleNamespace(raiden_service_pb2=fake_pb2)
        with (
            patch.dict(
                sys.modules,
                {
                    "tpu_sync": SimpleNamespace(rpc=fake_rpc),
                    "tpu_sync.rpc": fake_rpc,
                    "tpu_sync.rpc.raiden_service_pb2": fake_pb2,
                },
            ),
            patch(
                "sglang_tpu.disaggregation.raiden.reshard_schedule.build_sender_request",
                return_value=SimpleNamespace(),
            ),
            patch(
                "sglang_tpu.disaggregation.raiden.reshard_schedule.build_receiver_request",
                return_value=SimpleNamespace(),
            ),
            _capture_spans() as labels,
        ):
            mgr._run_reshard_push(55)
            mgr._run_reshard_swa_push(55, mgr._prefill.reshard_req[55], swa_blocks=[9])
            self.assertEqual(mgr._transfer_trace_meta, {})

            self.assertTrue(
                mgr._publish_reshard_plan(
                    room=55,
                    req_id="55",
                    uuid=55,
                    req=SimpleNamespace(dst_block_ids=[0, 1]),
                    ready_payload=[b"ready"],
                    ready_endpoints=["tcp://127.0.0.1:31000"],
                )
            )
            self.assertTrue(
                mgr._register_reshard_swa_receiver(
                    room=55,
                    src_attn_tp=2,
                    local_swa_blocks=[4],
                    swa_sources_map={0: [9], 1: [9]},
                )
            )
            engine.poll_stats_with_details.return_value = _poll(
                received=["resharded_transfer_55"]
            )
            swa_engine.poll_stats_with_details.return_value = _poll(
                received=["resharded_transfer_swa:55"]
            )
            mgr._drain_engine()

        self.assertEqual(
            labels,
            [
                "KV_Cache_Prefill_Register_Blocks#uuid=55,request_id=req-reshard,num_tokens=32,num_blocks=2#",
                "KV_Cache_Prefill_Send_Complete#uuid=55,request_id=req-reshard,num_tokens=32#",
                "KV_Cache_Prefill_Register_Blocks#uuid=55,request_id=req-reshard,num_tokens=16,num_blocks=1#",
                "KV_Cache_Prefill_Send_Complete#uuid=55,request_id=req-reshard,num_tokens=16#",
                "KV_Cache_Decode_Submit_Load#uuid=55,request_id=req-reshard,num_tokens=32,num_blocks=2#",
                "KV_Cache_Decode_Submit_Load#uuid=55,request_id=req-reshard,num_tokens=16,num_blocks=1#",
                "KV_Cache_Decode_Recv_Complete#uuid=55,request_id=req-reshard,num_tokens=32#",
                "KV_Cache_Decode_Recv_Complete#uuid=55,request_id=req-reshard,num_tokens=16#",
            ],
        )

    def test_pipeline_transfer_spans_and_non_duplicate_poll(self):
        mgr, _ = self._make_manager(page_size=16)
        mgr.register_trace_request_id(88, "req-pipe")

        pipe = object.__new__(RaidenPoolTransfer)
        pipe.manager = mgr
        pipe.lock = threading.Lock()
        pipe.pools, pipe.tensors = [], {}
        pipe.unit = SimpleNamespace()
        pipe.draft_kv_heads = pipe.draft_swa_kv_heads = None
        pipe.client = MagicMock()
        pipe.side = SimpleNamespace(
            family=SimpleNamespace(
                twin_req_id=lambda rid: f"swa:{rid}",
                twin_uuid=lambda rid: 880,
                spans=lambda *_a: [],
                label="swa",
            ),
            client=MagicMock(),
            unit=SimpleNamespace(),
            registered=set(),
            book=SimpleNamespace(expect=MagicMock()),
        )
        pipe._dst_attn_tp = lambda _room: 1
        pipe._thread_clients = threading.local()
        pipe.client_type = lambda _addr: MagicMock()
        pipe.poll = MagicMock(return_value=({"88"}, {"88"}, set()))
        mgr._pipeline = pipe

        with (
            patch(
                "sglang_tpu.disaggregation.raiden.pool_transfer.part_spans",
                return_value=[],
            ),
            _capture_spans() as labels,
        ):
            self.assertTrue(
                pipe._register_blocks(
                    88, req_id="88", uuid=88, full=[1, 2, 3], side=[9], with_side=True
                )
            )
            record_trace_meta(mgr, "88", 88, 3)
            pipe._coordinate(
                "127.0.0.1:31000", req_id="88", uuid=88, dst_block_counts=[3, 3, 3, 3]
            )
            pipe._coordinate(
                "127.0.0.1:31002", req_id="swa:88", uuid=880, dst_block_counts=[1, 1]
            )
            mgr._drain_engine()
            mgr._drain_engine()

        self.assertEqual(
            labels,
            [
                "KV_Cache_Prefill_Register_Blocks#uuid=88,request_id=req-pipe,num_tokens=48,num_blocks=3#",
                "KV_Cache_Prefill_Register_Blocks#uuid=880,request_id=req-pipe,num_tokens=16,num_blocks=1#",
                "KV_Cache_Decode_Submit_Load#uuid=88,request_id=req-pipe,num_tokens=48,num_blocks=3#",
                "KV_Cache_Decode_Submit_Load#uuid=880,request_id=req-pipe,num_tokens=16,num_blocks=1#",
                "KV_Cache_Prefill_Send_Complete#uuid=88,request_id=req-pipe,num_tokens=48#",
                "KV_Cache_Decode_Recv_Complete#uuid=88,request_id=req-pipe,num_tokens=48#",
            ],
        )


if __name__ == "__main__":
    unittest.main()
