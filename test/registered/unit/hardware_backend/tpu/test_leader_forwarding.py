import errno
import os
import signal
import socket
import socketserver
import threading
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import anyio

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.entrypoints import leader_forwarding

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class StreamingTarget(socketserver.BaseRequestHandler):
    def handle(self):
        request = b""
        while data := self.request.recv(4096):
            request += data
        self.request.sendall(b"port-%d:" % self.server.server_address[1])
        for index in range(3):
            self.request.sendall(b"chunk-%d:" % index + request)


class TestLeaderForwarding(unittest.TestCase):
    def test_serving_failure_signals_worker_before_context_exit(self):
        signalled = threading.Event()

        async def fail_serve(listener, handler):
            await anyio.sleep(0)
            raise OSError(errno.EMFILE, "Too many open files")

        with (
            patch.object(leader_forwarding.SocketListener, "serve", fail_serve),
            patch.object(
                leader_forwarding.os, "kill", side_effect=lambda *_: signalled.set()
            ) as kill,
            self.assertLogs(leader_forwarding.logger, level="ERROR") as logs,
            self.assertRaisesRegex(Exception, "unhandled errors in a TaskGroup"),
        ):
            with leader_forwarding._forward_ports("127.0.0.1", "127.0.0.2", [0]):
                self.assertTrue(signalled.wait(5))
                kill.assert_called_once_with(os.getpid(), signal.SIGQUIT)
        self.assertIn("Too many open files", "\n".join(logs.output))

    def test_streaming_half_close_and_listener_cleanup(self):
        with ExitStack() as stack:
            ports = []
            for _ in range(2):
                target = stack.enter_context(
                    socketserver.ThreadingTCPServer(("127.0.0.2", 0), StreamingTarget)
                )
                thread = threading.Thread(target=target.serve_forever, daemon=True)
                thread.start()
                stack.callback(thread.join)
                stack.callback(target.shutdown)
                ports.append(target.server_address[1])
            with (
                patch.object(leader_forwarding.os, "kill") as kill,
                leader_forwarding._forward_ports("127.0.0.1", "127.0.0.2", ports),
            ):
                for port in ports:
                    with socket.create_connection(
                        ("127.0.0.1", port), timeout=5
                    ) as client:
                        client.sendall(b"request")
                        client.shutdown(socket.SHUT_WR)
                        response = b""
                        while data := client.recv(4096):
                            response += data
                    self.assertEqual(
                        response,
                        b"port-%d:" % port
                        + b"".join(b"chunk-%d:request" % index for index in range(3)),
                    )
            kill.assert_not_called()
            for port in ports:
                with self.assertRaises(ConnectionRefusedError):
                    socket.create_connection(("127.0.0.1", port), timeout=5)

    def test_unready_target_rejects_connection_then_accepts_retry(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.2", 0))
            probe.settimeout(5)
            port = probe.getsockname()[1]
            with leader_forwarding._forward_ports("127.0.0.1", "127.0.0.2", [port]):
                with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
                    self.assertEqual(client.recv(4096), b"")
                probe.listen()
                with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
                    upstream, _ = probe.accept()
                    with upstream:
                        upstream.sendall(b"ready")
                        self.assertEqual(client.recv(4096), b"ready")

    def test_shutdown_closes_an_active_connection(self):
        accepted = threading.Event()

        class IdleTarget(socketserver.BaseRequestHandler):
            def handle(self):
                accepted.set()
                while self.request.recv(4096):
                    pass

        with socketserver.ThreadingTCPServer(("127.0.0.2", 0), IdleTarget) as target:
            thread = threading.Thread(target=target.serve_forever, daemon=True)
            thread.start()
            port = target.server_address[1]
            try:
                with self.assertRaisesRegex(RuntimeError, "scheduler failed"):
                    with leader_forwarding._forward_ports(
                        "127.0.0.1", "127.0.0.2", [port]
                    ):
                        client = socket.create_connection(
                            ("127.0.0.1", port), timeout=5
                        )
                        self.addCleanup(client.close)
                        self.assertTrue(accepted.wait(5))
                        raise RuntimeError("scheduler failed")
                self.assertEqual(client.recv(4096), b"")
            finally:
                target.shutdown()
                thread.join()

    def test_bind_failure_releases_already_started_listeners(self):
        with socket.socket() as occupied, socket.socket() as probe:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            probe.bind(("127.0.0.1", 0))
            first_port = probe.getsockname()[1]
            probe.close()
            with self.assertRaisesRegex(
                OSError, f"Could not bind port {occupied.getsockname()[1]}"
            ):
                with leader_forwarding._forward_ports(
                    "127.0.0.1", "127.0.0.2", [first_port, occupied.getsockname()[1]]
                ):
                    self.fail("bind failure must propagate")
            with socket.socket() as reusable:
                reusable.bind(("127.0.0.1", first_port))

    def test_only_moved_input_leader_forwards_role_ports(self):
        for initial, final, mode, switch, enabled, expected in (
            (0, 1, "null", False, True, [30000]),
            (0, 1, "prefill", False, True, [30000, 8998]),
            (0, 1, "decode", False, True, [30000]),
            (0, 1, "decode", True, True, [30000, 8998]),
            (0, 0, "prefill", False, True, []),
            (1, 0, "prefill", False, True, []),
            (1, 1, "prefill", False, True, []),
            (0, 1, "prefill", False, False, []),
        ):
            with self.subTest(
                initial=initial, final=final, mode=mode, switch=switch, enabled=enabled
            ):
                args = SimpleNamespace(node_rank=initial)
                cfg = SimpleNamespace(
                    node_rank=final,
                    nnodes=2,
                    tpu_node_normalization=enabled,
                    host="0.0.0.0",
                    port=30000,
                    dist_init_addr="normalized:29500",
                    disaggregation_mode=mode,
                    enable_pd_role_switch=switch,
                    disaggregation_bootstrap_port=8998,
                )
                with (
                    patch.object(leader_forwarding, "resolving_view", return_value=cfg),
                    patch.object(leader_forwarding, "_forward_ports") as forward,
                ):
                    with leader_forwarding.forward_normalized_leader(args) as active:
                        self.assertEqual(active, bool(expected))
                    if expected:
                        forward.assert_called_once_with(
                            "0.0.0.0", "normalized", expected
                        )
                        forward.return_value.__exit__.assert_called_once()
                    else:
                        forward.assert_not_called()

    def test_engine_listener_ownership_and_failure_cleanup(self):
        from sglang.srt.entrypoints import engine

        for device, moved, rust in (
            ("tpu", True, False),
            ("tpu", False, False),
            ("cuda", False, False),
            ("tpu", True, True),
        ):
            for fail in (False, True):
                with self.subTest(device=device, moved=moved, rust=rust, fail=fail):
                    forward = Mock()

                    @contextmanager
                    def forwarding(args):
                        forward("start")
                        try:
                            yield moved
                        finally:
                            forward("stop")

                    with (
                        patch.object(
                            engine,
                            "get_device",
                            return_value=SimpleNamespace(device=device),
                        ),
                        patch.object(
                            engine,
                            "get_serving",
                            return_value=SimpleNamespace(host="0.0.0.0", port=30000),
                        ),
                        patch.object(
                            engine,
                            "get_observability",
                            return_value=SimpleNamespace(enable_metrics=False),
                        ),
                        patch.object(
                            engine.envs.SGLANG_RUST_SERVER, "get", return_value=rust
                        ),
                        patch.object(
                            engine, "node_hosts_rust_server", return_value=rust
                        ),
                        patch.object(
                            leader_forwarding, "forward_normalized_leader", forwarding
                        ),
                        patch.object(
                            engine, "launch_dummy_health_check_server"
                        ) as dummy,
                    ):
                        with ExitStack() as stack:
                            if fail:
                                stack.enter_context(
                                    self.assertRaisesRegex(
                                        RuntimeError, "scheduler failed"
                                    )
                                )
                            stack.enter_context(
                                engine.Engine._worker_serving_listener(object())
                            )
                            if fail:
                                raise RuntimeError("scheduler failed")
                        self.assertEqual(dummy.call_count, int(not rust and not moved))
                        self.assertEqual(
                            forward.call_count, 2 if device == "tpu" and not rust else 0
                        )


if __name__ == "__main__":
    unittest.main()
