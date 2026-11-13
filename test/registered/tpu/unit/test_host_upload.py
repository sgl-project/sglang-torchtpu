"""Host-to-TPU uploads on the decode hot path must not wait for queued TPU work.

On torch_tpu a blocking H2D copy returns only after every program already
queued on the TPU has finished. In GLM-5.3-Flash MTP the track-id upload in
``_pad_mamba_tracking`` and the row-count uploads waited for the draft or the
whole target verify every round, and the TPU then idled while the host caught
up. Each case below queues ~100 ms of TPU work and times the host call: a call
that waited for the queue takes ~100 ms instead of well under 5 ms.

The last case pins the torch_tpu behavior the helpers rely on: a non-blocking
upload copies the host bytes during the call, so callers may reuse or drop the
host tensor right away.
"""

import math
import time
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

DEV = torch.device("tpu")
QUEUED_MS = 50.0
MAX_HOST_MS = 5.0


def _sync() -> None:
    torch.accelerator.synchronize()


class TestHostUploadDoesNotWait(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Big enough that one step takes far longer on the TPU than to launch,
        # so launching the steps builds a backlog instead of draining as it goes.
        cls.a = torch.randn(4096, 4096, device=DEV)
        cls.b = torch.randn(4096, 4096, device=DEV) * 0.01
        cls.iters = 2
        cls._queue_work()  # compiles
        _sync()
        cls.iters = 20
        t0 = time.perf_counter()
        cls._queue_work()
        _sync()
        per_iter_ms = (time.perf_counter() - t0) * 1e3 / cls.iters
        cls.iters = max(20, math.ceil(2 * QUEUED_MS / per_iter_ms))

    @classmethod
    def _queue_work(cls) -> torch.Tensor:
        x = cls.a
        for _ in range(cls.iters):
            x = torch.tanh(x @ cls.b)
        return x

    def _assert_no_wait(self, name, call):
        call()  # the first call may compile; time a warm one
        _sync()
        self._queue_work()
        t0 = time.perf_counter()
        out = call()
        host_ms = (time.perf_counter() - t0) * 1e3
        remaining_t0 = time.perf_counter()
        _sync()
        remaining_ms = (time.perf_counter() - remaining_t0) * 1e3
        self.assertLess(
            host_ms,
            MAX_HOST_MS,
            f"{name} took {host_ms:.1f} ms while TPU work was queued "
            f"({remaining_ms:.1f} ms of it left afterwards): it waited for the queue",
        )
        # Guard the measurement itself: the work must still have been queued.
        self.assertGreater(
            remaining_ms,
            QUEUED_MS,
            f"only {remaining_ms:.1f} ms of TPU work was left after {name} "
            f"returned in {host_ms:.1f} ms; the test did not queue enough work",
        )
        return out

    def test_upload(self):
        from sglang_tpu.host_upload import upload

        src = torch.arange(8, dtype=torch.int32)
        out = self._assert_no_wait("upload", lambda: upload(src, DEV))
        self.assertEqual(out.cpu().tolist(), list(range(8)))

    def test_device_scalar_first_upload(self):
        from sglang_tpu.host_upload import device_scalar

        # Each call takes a value not cached yet, so it exercises the upload.
        values = iter(range(1_000_000, 1_000_010))
        out = self._assert_no_wait(
            "device_scalar", lambda: device_scalar(next(values), torch.int32, DEV)
        )
        self.assertEqual(out.item(), 1_000_001)

    def test_pad_mamba_tracking(self):
        """Bug regression: this upload ran just before every verify and waited
        for the draft still queued on the TPU (about 1.5 ms per round)."""
        from sglang_tpu.compilation.compiled_forward_shim import _pad_mamba_tracking

        def call():
            batch = SimpleNamespace(
                seq_lens=torch.zeros(8, dtype=torch.int64, device=DEV),
                mamba_track_indices=torch.tensor([3, 5, 7], dtype=torch.int64),
            )
            _pad_mamba_tracking(batch, 8)
            return batch

        batch = self._assert_no_wait("_pad_mamba_tracking", call)
        self.assertEqual(batch.mamba_track_indices.device.type, DEV.type)
        self.assertEqual(
            batch.mamba_track_indices.cpu().tolist(), [3, 5, 7, 0, 0, 0, 0, 0]
        )

    def test_upload_copies_host_bytes_at_call_time(self):
        from sglang_tpu.host_upload import upload

        for _ in range(5):
            src = torch.full((8,), 7, dtype=torch.int32)
            self._queue_work()
            out = upload(src, DEV)
            # The copy has not run yet; the bytes it sends must already be taken.
            src.fill_(-1)
            del src
            _sync()
            self.assertEqual(out.cpu().tolist(), [7] * 8)


if __name__ == "__main__":
    unittest.main()
