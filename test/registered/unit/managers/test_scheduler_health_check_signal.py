"""A busy scheduler acknowledges a deferred health probe once a batch with real
work completes.

The Python frontend reads the acknowledgement from the zmq tokenizer socket.
The embedded Rust server does not consume that socket, so in Rust mode the
acknowledgement goes on its response ring instead.

Which batches acknowledge a probe, and how a probe is deferred while the engine
is busy, is covered by test_health_completion_ack.py.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX  # noqa: E402
from sglang.srt.managers.io_struct import HealthCheckOutput  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.health_monitor import (  # noqa: E402
    SchedulerHealthMonitor,
)
from sglang.srt.model_executor.forward_mode import ForwardMode  # noqa: E402
from sglang.srt.rust_server.server import RustServer  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _decode_batch():
    req = SimpleNamespace(rid="prompt", to_finish=None, finished_reason=None)
    return SimpleNamespace(reqs=[req], forward_mode=ForwardMode.DECODE)


class TestHealthCheckSignal(CustomTestCase):
    def _scheduler(self, rust_server, probe_ipcs) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.rust_server = rust_server
        s.ipc_channels = MagicMock()
        s.health_monitor = SchedulerHealthMonitor(s.send_health_check_output, 0)
        for i, ipc in enumerate(probe_ipcs):
            s.health_monitor.register(
                SimpleNamespace(
                    rid=f"{HEALTH_CHECK_RID_PREFIX}_{i}", http_worker_ipc=ipc
                ),
                admitted=False,
            )
        return s

    def test_rust_mode_acknowledges_deferred_probes_on_the_ring(self):
        server = MagicMock()
        s = self._scheduler(RustServer(server, http_port=0), [None, None])

        self.assertTrue(s.health_monitor.on_batch_completed(_decode_batch()))

        server.push_health_ack.assert_called()
        self.assertFalse(s.health_monitor.pending_rids)
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_python_mode_acknowledges_each_probe_on_zmq(self):
        s = self._scheduler(None, ["ipc-a", "ipc-b"])

        self.assertTrue(s.health_monitor.on_batch_completed(_decode_batch()))

        outputs = [
            args[0]
            for args, _ in s.ipc_channels.send_to_tokenizer.send_output.call_args_list
        ]
        self.assertTrue(all(isinstance(o, HealthCheckOutput) for o in outputs))
        self.assertEqual([o.http_worker_ipc for o in outputs], ["ipc-a", "ipc-b"])
        self.assertFalse(s.health_monitor.pending_rids)


if __name__ == "__main__":
    unittest.main()
