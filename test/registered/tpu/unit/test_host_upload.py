"""Host-to-TPU uploads on hot paths must not wait for queued TPU work.

On torch_tpu a blocking H2D copy returns only after every program already
queued on the TPU has finished. In GLM-5.3-Flash MTP the track-id upload in
``_pad_mamba_tracking`` and the row-count uploads waited for the draft or the
whole target verify every round, and the TPU then idled while the host caught
up. Multimodal row placement had the same wait right after the vision encoder
launch. Each case below queues ~100 ms of TPU work and times the host call: a call
that waited for the queue takes ~100 ms instead of well under 5 ms.

The last two cases check results, not timing. One pins the torch_tpu behavior
the helpers rely on: a non-blocking upload copies the host bytes during the call,
so callers may reuse or drop the host tensor right away. The other checks that
host placement's reused matrices survive being rewritten by the next placement
while TPU work is still queued.
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

    def test_scatter_mm_on_device(self):
        """Bug regression: the host-built row indices and mask were uploaded
        blocking right after the vision encoder launch, so the host waited for
        the encoder before it could prepare the next prefill chunk."""
        from sglang_tpu.mm_embed_tpu import scatter_mm_on_device

        rows = torch.tensor([1, 4, 6])
        mask = torch.zeros(8, 1, dtype=torch.bool)
        mask[rows] = True
        src = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).to(DEV)
        dest = torch.full((8, 4), -1.0, dtype=torch.bfloat16, device=DEV)

        self._assert_no_wait(
            "scatter_mm_on_device", lambda: scatter_mm_on_device(dest, mask, src)
        )
        expected = torch.full((8, 4), -1.0, dtype=torch.bfloat16)
        expected[rows] = src.cpu()
        self.assertTrue(torch.equal(dest.cpu(), expected))

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

    def test_host_placement_survives_the_next_placement(self):
        """The next prefill's placement rewrites embed_mm_on_host's reused
        _HostRowBuffer matrices in place, so each upload of them must be done
        with the host bytes before it returns. At the largest size a 16k-token
        chunk reaches, place one batch behind queued TPU work, place the next
        batch into the same matrices before that work drains, and check the
        first batch's device results bit for bit."""
        from sglang.srt.managers.schedule_batch import Modality
        from sglang_tpu.mm_embed_tpu import embed_mm_on_host

        rows, hidden, dtype = 16384, 12288, torch.bfloat16
        gen = torch.Generator().manual_seed(0)

        def batch(first_row, num_rows):
            mask = torch.zeros(rows, 1, dtype=torch.bool)
            mask[first_row : first_row + num_rows] = True
            embeds = torch.randn(num_rows, hidden, generator=gen).to(dtype)
            deepstack = torch.randn(num_rows, hidden, generator=gen).to(dtype)
            return mask, embeds, deepstack

        input_embeds = torch.full((rows, hidden), -1.0, dtype=dtype, device=DEV)

        def place(placed):
            mask, embeds, deepstack = placed
            return embed_mm_on_host(
                input_embeds=input_embeds,
                deepstack_shape=torch.Size((rows, hidden)),
                modalities=[Modality.IMAGE],
                embeddings=[embeds],
                masks=[mask],
                deepstack_embeddings=[deepstack],
                use_deepstack={Modality.IMAGE: True},
            )

        # The second batch's reset() zeroes the first batch's rows, and its own
        # rows overlap them, so a copy that read the matrices late would differ.
        first, second = batch(1000, 3000), batch(2000, 3000)
        place(second)  # compiles the merge at this shape
        _sync()
        t0 = time.perf_counter()
        place(first)
        place(second)
        two_placements_ms = (time.perf_counter() - t0) * 1e3
        _sync()

        # Queue several times the two placements' host time, so that with
        # non-blocking copies both placements finish while the first batch's
        # copies still wait behind this work.
        steps = max(5, math.ceil(4 * two_placements_ms / (2 * QUEUED_MS)))
        for _ in range(steps):
            self._queue_work()
        t0 = time.perf_counter()
        out, out_deepstack = place(first)
        first_ms = (time.perf_counter() - t0) * 1e3
        place(second)
        remaining_t0 = time.perf_counter()
        _sync()
        remaining_ms = (time.perf_counter() - remaining_t0) * 1e3
        # A blocking copy waits for the queue, so the second placement has
        # nothing to race; a placement that returned early must have left the
        # queue running for the second one.
        if first_ms < steps * QUEUED_MS:
            self.assertGreater(
                remaining_ms,
                QUEUED_MS,
                f"only {remaining_ms:.1f} ms of TPU work was left after the second "
                "placement; the test did not queue enough work",
            )

        mask, embeds, deepstack = first
        placed_rows = mask.reshape(-1).nonzero(as_tuple=True)[0]
        expected = torch.full((rows, hidden), -1.0, dtype=dtype)
        expected[placed_rows] = embeds
        expected_deepstack = torch.zeros(rows, hidden, dtype=dtype)
        expected_deepstack[placed_rows] = deepstack
        self.assertTrue(torch.equal(out.cpu(), expected))
        self.assertTrue(torch.equal(out_deepstack.cpu(), expected_deepstack))


if __name__ == "__main__":
    unittest.main()
