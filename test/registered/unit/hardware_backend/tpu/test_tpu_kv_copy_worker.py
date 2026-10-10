"""Deterministic CPU checks for copy completion, FIFO order, and direction isolation."""

import threading
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.kv_copy_worker import TpuKvCopyWorker

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTpuKvCopyWorker(unittest.TestCase):
    def test_failed_backup_preserves_fifo_and_does_not_block_reload(self):
        backup = TpuKvCopyWorker("audit-backup")
        reload = TpuKvCopyWorker("audit-reload")
        started, release, reloaded = (threading.Event() for _ in range(3))
        order = []
        failure = RuntimeError("backup failed")

        def blocked_backup():
            started.set()
            if not release.wait(5):
                raise TimeoutError("test did not release the backup")
            order.append("failed")
            raise failure

        try:
            failed = backup.submit(blocked_backup)
            self.assertTrue(started.wait(5))
            successor = backup.submit(order.append, "successor")
            loaded = reload.submit(reloaded.set)
            self.assertTrue(reloaded.wait(5))
            self.assertFalse(failed.query())
            self.assertFalse(successor.query())
            self.assertEqual(order, [])
            # Shutdown must drain jobs accepted before its sentinel.
            backup.shutdown()
            release.set()
            backup._thread.join(5)
            self.assertFalse(backup._thread.is_alive())
            self.assertEqual(order, ["failed", "successor"])
            for observe in (failed.query, failed.synchronize, failed.wait):
                with self.assertRaises(RuntimeError) as caught:
                    observe()
                self.assertIs(caught.exception, failure)
            self.assertTrue(successor.query())
            successor.synchronize()
            self.assertTrue(loaded._e.wait(5))
            self.assertTrue(loaded.query())
        finally:
            release.set()
            backup.shutdown()
            reload.shutdown()
            backup._thread.join(5)
            reload._thread.join(5)


if __name__ == "__main__":
    unittest.main()
