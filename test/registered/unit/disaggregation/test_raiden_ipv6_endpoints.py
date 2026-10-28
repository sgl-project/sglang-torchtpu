"""The raiden rendezvous must reach peers that only have IPv6 addresses.

Every control message of the raiden transport (PULL_REQ, BLOCKS, CHUNK, the
reshard handshake) travels on a per-peer ZMQ PUSH socket whose endpoint is
built from a bare ip literal and a port. On a cluster whose pods carry IPv6
addresses (GKE IPv6-only node pools) a bare ``tcp://2a03::13:31000`` is
accepted by libzmq's connect, then its resolve fails inside the I/O thread on
the IPv4-only socket and, with reconnects disabled, is never retried: the PUSH
socket queues every message and delivers none. The decode never learns where
the prefill's blocks are, the prefill never hears the decode ask, and both
sides sit until their disaggregation timeouts fire.

These tests pin the two halves of the fix: endpoints bracket IPv6 literals,
and the send path opts the socket into IPv6 for them.
"""

import socket
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager, _tcp_endpoint
from sglang_tpu.disaggregation.raiden.serde import (
    Blocks,
    Chunk,
    PullReq,
)
from sglang_tpu.disaggregation.raiden.state import DecodeState, PrefillState

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOM = 11
V6_DECODE = "2a03:83e4:4021:302:0:15:0:13"
V6_PREFILL = "2a03:83e4:4021:302:0:13:0:12"


def _bare_manager(mode):
    """A RaidenKVManager with only the state the tested paths touch."""
    mgr = RaidenKVManager.__new__(RaidenKVManager)
    mgr._prefill = (
        PrefillState(announce_pages=1) if mode == DisaggregationMode.PREFILL else None
    )
    mgr._decode = DecodeState() if mode == DisaggregationMode.DECODE else None
    mgr.disaggregation_mode = mode
    mgr.attn_tp_rank = 0
    mgr.request_status = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._announce_lock = threading.Lock()
    mgr._engine_lock = threading.Lock()
    mgr._completed_send = set()
    mgr._completed_recv = set()
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._aborted = {}
    mgr._engine = Mock()
    mgr._swa_engine = None
    mgr._send_reads = {}
    mgr.chunked_transfer = False
    mgr._transfer_parallelism = 4
    mgr.prefix_pull_enabled = False
    return mgr


def _prefill_manager():
    mgr = _bare_manager(DisaggregationMode.PREFILL)
    mgr.raiden_endpoint = f"[{V6_PREFILL}]:31000"
    mgr.swa_raiden_endpoint = None
    mgr._uuid_seq = 0
    mgr._local_rank = 0
    return mgr


def _decode_manager():
    mgr = _bare_manager(DisaggregationMode.DECODE)
    mgr.local_ip = V6_DECODE
    mgr.rank_port = 7000
    mgr._dynamic_host_staging = True
    return mgr


class TestEndpointFormatting(CustomTestCase):
    def test_ipv6_literal_is_bracketed(self):
        self.assertEqual(_tcp_endpoint(V6_DECODE, 5555), f"tcp://[{V6_DECODE}]:5555")

    def test_ipv4_literal_is_unchanged(self):
        self.assertEqual(_tcp_endpoint("10.0.0.2", 5555), "tcp://10.0.0.2:5555")

    def test_already_bracketed_is_not_double_wrapped(self):
        self.assertEqual(
            _tcp_endpoint(f"[{V6_DECODE}]", 5555), f"tcp://[{V6_DECODE}]:5555"
        )

    def test_port_is_coerced_to_int(self):
        # rank_port arrives as a string from the bootstrap server's JSON.
        self.assertEqual(_tcp_endpoint("10.0.0.2", "5555"), "tcp://10.0.0.2:5555")


class TestSendOptsIntoIPv6(CustomTestCase):
    def _capture_sends(self, mgr):
        calls = []

        def fake_connect(endpoint, is_ipv6=False):
            calls.append((endpoint, is_ipv6))
            mgr._socket_send_locks.setdefault(endpoint, threading.Lock())
            sock = Mock()
            mgr._socket_cache[endpoint] = sock
            return sock

        mgr._socket_cache = {}
        mgr._socket_send_locks = {}
        mgr._socket_lock = threading.Lock()
        mgr._connect = fake_connect
        return calls

    def test_bracketed_endpoint_requests_ipv6_socket(self):
        mgr = _prefill_manager()
        calls = self._capture_sends(mgr)
        mgr._send_multipart_locked(f"tcp://[{V6_DECODE}]:5555", [b"x"])
        self.assertEqual(calls, [(f"tcp://[{V6_DECODE}]:5555", True)])

    def test_ipv4_endpoint_keeps_ipv4_socket(self):
        mgr = _prefill_manager()
        calls = self._capture_sends(mgr)
        mgr._send_multipart_locked("tcp://10.0.0.2:5555", [b"x"])
        self.assertEqual(calls, [("tcp://10.0.0.2:5555", False)])

    def test_explicit_flag_is_honoured(self):
        mgr = _prefill_manager()
        calls = self._capture_sends(mgr)
        mgr._send_multipart_locked("tcp://peer.example:5555", [b"x"], is_ipv6=True)
        self.assertEqual(calls, [("tcp://peer.example:5555", True)])


class TestHandshakeEndpoints(CustomTestCase):
    """The rendezvous messages address IPv6 peers with bracketed endpoints."""

    def test_prefill_reply_targets_bracketed_decode_endpoint(self):
        mgr = _prefill_manager()
        mgr._prefill.send_blocks[ROOM] = [1, 2, 3]
        mgr._prefill.send_aux[ROOM] = b""
        mgr._prefill.reshard_finalized.add(ROOM)
        # As stored by _on_pull_req from the PULL_REQ frames.
        mgr._prefill.decode_addr[ROOM] = (V6_DECODE, 5555)
        mgr._prefill.decode_pages[ROOM] = 3
        mgr._prefill.decode_swa_pages[ROOM] = 0
        mgr.request_status[ROOM] = KVPoll.Transferring
        mgr._register_sender_read = Mock()
        mgr._send_multipart_locked = Mock()

        mgr._maybe_reply(ROOM)

        (endpoint, payload), _ = mgr._send_multipart_locked.call_args
        self.assertEqual(endpoint, f"tcp://[{V6_DECODE}]:5555")
        Blocks.from_frames(payload)
        # The native engine endpoint rides along bracketed as well.
        self.assertEqual(Blocks.from_frames(payload).endpoint, f"[{V6_PREFILL}]:31000")

    def test_chunk_announcement_targets_bracketed_decode_endpoint(self):
        mgr = _prefill_manager()
        mgr.chunked_transfer = True
        mgr._prefill.decode_addr[ROOM] = (V6_DECODE, 5555)
        mgr._prefill.decode_pages[ROOM] = 4
        mgr._register_sender_read = Mock()
        mgr._send_multipart_locked = Mock()

        mgr._announce_chunk(ROOM, 0, [10, 11], total_pages=4)

        (endpoint, payload), _ = mgr._send_multipart_locked.call_args
        self.assertEqual(endpoint, f"tcp://[{V6_DECODE}]:5555")
        Chunk.from_frames(payload)

    def test_decode_pull_request_targets_bracketed_prefill_endpoint(self):
        mgr = _decode_manager()
        mgr._send_multipart_locked = Mock()
        infos = [{"rank_ip": V6_PREFILL, "rank_port": "6000"}]

        mgr.register_decode_request(ROOM, [1, 2], [], None, infos)

        (endpoint, payload), _ = mgr._send_multipart_locked.call_args
        self.assertEqual(endpoint, f"tcp://[{V6_PREFILL}]:6000")
        PullReq.from_frames(payload)
        # The reply address is the bare ip; the prefill brackets it itself.
        self.assertEqual(PullReq.from_frames(payload).decode_ip, V6_DECODE)

    def test_ipv4_peers_are_addressed_as_before(self):
        mgr = _decode_manager()
        mgr.local_ip = "10.0.0.3"
        mgr._send_multipart_locked = Mock()

        mgr.register_decode_request(
            ROOM, [1, 2], [], None, [{"rank_ip": "10.0.0.2", "rank_port": 6000}]
        )

        (endpoint, _), _ = mgr._send_multipart_locked.call_args
        self.assertEqual(endpoint, "tcp://10.0.0.2:6000")


class TestEngineEndpoint(CustomTestCase):
    """The endpoint the native engine is advertised under brackets IPv6."""

    def _build(self, local_ip, swa):
        mgr = _bare_manager(DisaggregationMode.PREFILL)
        mgr._engine = None
        mgr.swa_raiden_endpoint = None
        mgr.local_ip = local_ip
        mgr._control_port = 31000
        mgr._timeout_s = 120.0
        mgr._local_rank = 0
        mgr._kv_caches = [object()]
        mgr._swa_kv_caches = [object()] if swa else None
        mgr._max_request_blocks = lambda: 8
        mgr._swa_max_request_blocks = lambda: 2
        mgr._num_slots = lambda *a, **k: 3
        fake_api = types.ModuleType("tpu_sync.api.torch.kv_cache_manager")
        fake_api.KVCacheManager = Mock()
        modules = {
            "tpu_sync": types.ModuleType("tpu_sync"),
            "tpu_sync.api": types.ModuleType("tpu_sync.api"),
            "tpu_sync.api.torch": types.ModuleType("tpu_sync.api.torch"),
            "tpu_sync.api.torch.kv_cache_manager": fake_api,
        }
        with patch.dict(sys.modules, modules):
            mgr._build_engine()
        return mgr

    def test_ipv6_engine_endpoints_are_bracketed(self):
        mgr = self._build(V6_PREFILL, swa=True)
        self.assertEqual(mgr.raiden_endpoint, f"[{V6_PREFILL}]:31000")
        self.assertEqual(mgr.swa_raiden_endpoint, f"[{V6_PREFILL}]:31002")

    def test_ipv4_engine_endpoint_is_unchanged(self):
        mgr = self._build("10.0.0.2", swa=False)
        self.assertEqual(mgr.raiden_endpoint, "10.0.0.2:31000")
        self.assertIsNone(mgr.swa_raiden_endpoint)


def _ipv6_loopback_available():
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


@unittest.skipUnless(_ipv6_loopback_available(), "no IPv6 loopback on this host")
class TestIPv6Delivery(CustomTestCase):
    """End to end over real sockets: the manager's send path reaches an IPv6
    PULL socket, and the bare form it replaced does not."""

    def setUp(self):
        self.ctx = zmq.Context()
        self.pull = self.ctx.socket(zmq.PULL)
        self.pull.setsockopt(zmq.IPV6, 1)
        self.port = self.pull.bind_to_random_port("tcp://[::1]")
        self.mgr = _prefill_manager()
        self.mgr._zmq_ctx = self.ctx
        self.mgr._socket_cache = {}
        self.mgr._monitor_cache = {}
        self.mgr._socket_send_locks = {}
        self.mgr._socket_lock = threading.Lock()

    def tearDown(self):
        for sock in list(self.mgr._monitor_cache.values()) + list(
            self.mgr._socket_cache.values()
        ):
            sock.close(linger=0)
        self.pull.close(linger=0)
        self.ctx.term()

    def test_bracketed_endpoint_is_delivered(self):
        self.mgr._send_multipart_locked(_tcp_endpoint("::1", self.port), [b"hello"])
        self.assertTrue(self.pull.poll(5000), "message never arrived")
        self.assertEqual(self.pull.recv_multipart(), [b"hello"])

    def test_bare_ipv6_endpoint_is_silently_dropped(self):
        # The pre-fix formatting: libzmq accepts it and delivers nothing.
        self.mgr._send_multipart_locked(f"tcp://::1:{self.port}", [b"hello"])
        self.assertFalse(self.pull.poll(1000))


if __name__ == "__main__":
    unittest.main()
