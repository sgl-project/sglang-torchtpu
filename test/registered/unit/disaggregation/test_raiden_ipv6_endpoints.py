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
import threading
import unittest
from unittest.mock import Mock

import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.raiden.conn import (
    RaidenKVCacheGroups,
    _tcp_endpoint,
    build_engine_configs,
)
from sglang.srt.disaggregation.raiden.decode import RaidenDecodeKVManager
from sglang.srt.disaggregation.raiden.prefill import RaidenPrefillKVManager
from sglang.srt.disaggregation.raiden.serde import MessageTag
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOM = 11
V6_DECODE = "2a03:83e4:4021:302:0:15:0:13"
V6_PREFILL = "2a03:83e4:4021:302:0:13:0:12"


def _bare_manager(manager_cls, mode):
    """A role manager with only the state the tested paths touch."""
    mgr = manager_cls.__new__(manager_cls)
    mgr.disaggregation_mode = mode
    mgr.attn_tp_rank = 0
    mgr.request_status = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._state_lock = threading.Lock()
    mgr._status_lock = threading.Lock()
    mgr._announce_lock = threading.Lock()
    mgr._completed_send = set()
    mgr._completed_recv = set()
    mgr._failed = set()
    mgr._native_failed = set()
    mgr._aborted = {}
    mgr._full_attn_engine = Mock()
    mgr._sliding_window_attn_engine = None
    mgr._send_reads = {}
    mgr.chunked_transfer = False
    return mgr


def _prefill_manager():
    mgr = _bare_manager(RaidenPrefillKVManager, DisaggregationMode.PREFILL)
    mgr._full_attn_engine.endpoint = f"[{V6_PREFILL}]:31000"
    mgr._send_blocks = {}
    mgr._send_sliding_window_attn_blocks = {}
    mgr._send_aux = {}
    mgr._decode_addr = {}
    mgr._replied = set()
    mgr._reshard_finalized = set()
    mgr._reshard_req = {}
    mgr._decode_pages = {}
    mgr._send_total = {}
    mgr._chunk_seq = {}
    mgr._uuid_seq = 0
    mgr._local_rank = 0
    return mgr


def _decode_manager():
    mgr = _bare_manager(RaidenDecodeKVManager, DisaggregationMode.DECODE)
    mgr.local_ip = V6_DECODE
    mgr.rank_port = 7000
    mgr._dynamic_host_staging = True
    mgr._local_blocks = {}
    mgr._local_sliding_window_attn_blocks = {}
    mgr._decode_aux_index = {}
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
        mgr._send_blocks[ROOM] = [1, 2, 3]
        mgr._send_aux[ROOM] = b""
        mgr._reshard_finalized.add(ROOM)
        # As stored by _on_pull_req from the PULL_REQ ip/port frames.
        mgr._decode_addr[ROOM] = (V6_DECODE, 5555)
        mgr.request_status[ROOM] = KVPoll.Transferring
        mgr._register_sender_read = Mock()
        mgr._send_multipart_locked = Mock()

        mgr._maybe_reply(ROOM)

        (endpoint, payload), _ = mgr._send_multipart_locked.call_args
        self.assertEqual(endpoint, f"tcp://[{V6_DECODE}]:5555")
        self.assertEqual(payload[0], MessageTag.BLOCKS)
        # The native engine endpoint rides along bracketed as well.
        self.assertEqual(payload[3], f"[{V6_PREFILL}]:31000".encode("ascii"))

    def test_chunk_announcement_targets_bracketed_decode_endpoint(self):
        mgr = _prefill_manager()
        mgr.chunked_transfer = True
        mgr._decode_addr[ROOM] = (V6_DECODE, 5555)
        mgr._decode_pages[ROOM] = 4
        mgr._register_sender_read = Mock()
        mgr._send_multipart_locked = Mock()

        mgr._announce_chunk(ROOM, 0, [10, 11], total_pages=4)

        (endpoint, payload), _ = mgr._send_multipart_locked.call_args
        self.assertEqual(endpoint, f"tcp://[{V6_DECODE}]:5555")
        self.assertEqual(payload[0], MessageTag.CHUNK)

    def test_decode_pull_request_targets_bracketed_prefill_endpoint(self):
        mgr = _decode_manager()
        mgr._send_multipart_locked = Mock()
        infos = [{"rank_ip": V6_PREFILL, "rank_port": "6000"}]

        mgr.register_decode_request(ROOM, [1, 2], [], None, infos)

        (endpoint, payload), _ = mgr._send_multipart_locked.call_args
        self.assertEqual(endpoint, f"tcp://[{V6_PREFILL}]:6000")
        self.assertEqual(payload[0], MessageTag.PULL_REQ)
        # The reply address is the bare ip; the prefill brackets it itself.
        self.assertEqual(payload[2], V6_DECODE.encode("ascii"))

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

    def _build(self, local_ip, sliding_window_attn):
        cache = Mock(shape=(8, 4))
        cache.element_size.return_value = 4
        groups = RaidenKVCacheGroups(
            full_attn=[cache],
            sliding_window_attn=[cache] if sliding_window_attn else [],
        )
        return build_engine_configs(
            groups,
            local_ip=local_ip,
            local_rank=0,
            control_port=31000,
            timeout_s=120.0,
            context_length=32,
            page_size=4,
            sliding_window_size=8,
            slot_ceiling=3,
            local_ranks=1,
            rank=0,
        )

    def test_ipv6_engine_endpoints_are_bracketed(self):
        full_attn, sliding_window_attn = self._build(
            V6_PREFILL, sliding_window_attn=True
        )
        self.assertEqual(full_attn.endpoint, f"[{V6_PREFILL}]:31000")
        self.assertEqual(sliding_window_attn.endpoint, f"[{V6_PREFILL}]:31002")

    def test_ipv4_engine_endpoint_is_unchanged(self):
        full_attn, sliding_window_attn = self._build(
            "10.0.0.2", sliding_window_attn=False
        )
        self.assertEqual(full_attn.endpoint, "10.0.0.2:31000")
        self.assertIsNone(sliding_window_attn)


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
