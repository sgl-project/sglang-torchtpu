"""The point-to-point pipeline hand-off (sglang_tpu/pp_buffered.py) over a fake
in-process transport."""

import contextlib
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch  # noqa: E402

from sglang.srt.layers.aux_hidden_states import (  # noqa: E402
    AUX_HIDDEN_STATES_PROXY_KEY,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (  # noqa: E402
    SpecAuxHiddenStateConfig,
)
from sglang_tpu import pp_scheduler  # noqa: E402
from sglang_tpu.compilation.bucketer import Bucketer  # noqa: E402
from sglang_tpu.compilation.compiled_forward_shim import (  # noqa: E402
    _fit_pp_proxy_rows,
)
from sglang_tpu.compilation.decoder_compile_policy import (  # noqa: E402
    DecoderTpuCompilePolicy,
)
from sglang_tpu.cp_forward import cp_local_rows_padded  # noqa: E402
from sglang_tpu.envs import envs as tpu_envs  # noqa: E402
from sglang_tpu.hooks.pp_buffered_kv_budget import (  # noqa: E402
    reserve_pp_buffered_handoff,
)
from sglang_tpu.pp_buffered import (  # noqa: E402
    _MAX_RECEIVES,
    _POST_LOG_EVERY,
    PPBufferedHandoff,
    Transport,
    hbm_reserve_bytes,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 4


class _Done:
    def wait(self):
        return True


class FakeNet:
    """Posted receives of every rank, matched by (src, dst, tag)."""

    def __init__(self):
        self.cv = threading.Condition()
        self.posted = {}
        self.sent_tags = []
        # Received tensors whose data has landed, by id. The tensors are kept
        # so their ids cannot be reused by later landing buffers.
        self.landed = {}


class FakeFence:
    def __init__(self, log):
        self.log = log

    def synchronize(self):
        self.log.append(self)

    def query(self):
        # Not known to have passed: a post past the bound waits on it.
        return False


class FakeTransport(Transport):
    def __init__(self, net, rank, *, send_gate=None):
        super().__init__()
        self.net = net
        self.rank = rank
        self.send_gate = send_gate
        self.synced = []
        self.flushed_all = 0
        self.checked = []
        self.fail_check = False

    def recv_list(self, tensors, src, tag):
        with self.net.cv:
            key = (src, self.rank, tag)
            assert key not in self.net.posted, key
            self.net.posted[key] = list(tensors)
            self.net.cv.notify_all()
        return _Done()

    def send_list(self, tensors, dst, tag):
        if self.send_gate is not None:
            self.send_gate.wait()
        key = (self.rank, dst, tag)
        with self.net.cv:
            # A send returns once the receiver has posted: the credit.
            self.net.cv.wait_for(lambda: key in self.net.posted, timeout=10)
            targets = self.net.posted.pop(key)
            assert len(targets) == len(tensors), (len(targets), len(tensors))
            for target, tensor in zip(targets, tensors):
                assert target.shape == tensor.shape, (target.shape, tensor.shape)
                target.copy_(tensor)
                self.net.landed[id(target)] = target
            self.net.sent_tags.append((self.rank, dst, tag))
            self.net.cv.notify_all()
        return _Done()

    def wait_landed(self, tensor):
        with self.net.cv:
            assert self.net.cv.wait_for(
                lambda: self.net.landed.get(id(tensor)) is tensor, timeout=10
            )

    def landing_buffer(self, shape, dtype, device):
        return torch.zeros(shape, dtype=dtype, device=device)

    def flush(self, tensors):
        list(tensors)

    def materialize(self, tensors):
        list(tensors)

    def flush_all(self):
        self.flushed_all += 1

    def fence(self):
        return FakeFence(self.synced)

    def check_free(self, nbytes, device):
        self.checked.append(nbytes)
        if self.fail_check:
            raise RuntimeError("RESOURCE_EXHAUSTED: out of HBM")


def make_handoff(
    rank, size, net, *, num_aux=0, depth=2, max_rows=64, aux_dtype=None, **kw
):
    return PPBufferedHandoff(
        rank=rank,
        size=size,
        pp_ranks=tuple(range(10, 10 + size)),
        hidden=HIDDEN,
        num_aux=num_aux,
        dtype=torch.float32,
        aux_dtype=aux_dtype,
        device=torch.device("cpu"),
        max_rows=max_rows,
        depth=depth,
        transport=FakeTransport(net, 10 + rank, **kw),
    )


def outgoing(rank, step, rows, num_aux, aux_dtype=torch.float32):
    tensors = {
        "hidden_states": torch.full((rows, HIDDEN), float(100 * rank + step)),
        # The handoff sends the sum, so a residual must be added in.
        "residual": torch.ones((rows, HIDDEN)),
    }
    if num_aux:
        tensors[AUX_HIDDEN_STATES_PROXY_KEY] = torch.full(
            (rows, HIDDEN * num_aux), float(-(100 * rank + step)), dtype=aux_dtype
        )
    return tensors


def run_stage(handoff, schedule, num_aux, received, errors):
    aux_dtype = handoff.aux_dtype
    try:
        if not handoff.probe():
            raise AssertionError(f"stage {handoff.rank}: self-test failed")
        for step, rows in enumerate(schedule):
            handoff.begin_iteration(rows is not None)
            if rows is None:
                handoff.end_iteration(None)
                continue
            got = handoff.receive(rows)
            if got is not None:
                received.append((step, got))
            out = None
            if handoff.rank != handoff.last:
                out = outgoing(handoff.rank, step, rows, num_aux, aux_dtype)
            handoff.end_iteration(out)
        handoff.close()
    except BaseException as exc:  # surfaced by the test thread
        errors.append(exc)


class TestPPBufferedHandoffPipeline(unittest.TestCase):
    def _run(self, size, schedule, num_aux=0, depth=2, aux_dtype=None):
        net = FakeNet()
        handoffs = [
            make_handoff(
                r, size, net, num_aux=num_aux, depth=depth, aux_dtype=aux_dtype
            )
            for r in range(size)
        ]
        received = [[] for _ in range(size)]
        errors = []
        threads = [
            threading.Thread(
                target=run_stage,
                args=(handoffs[r], schedule, num_aux, received[r], errors),
            )
            for r in range(size)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertFalse(any(t.is_alive() for t in threads), "a stage hung")
        self.assertEqual(errors, [])
        self.assertEqual(net.posted, {}, "receives left unmatched")
        return handoffs, received, net

    def test_each_stage_gets_the_previous_stage_output(self):
        schedule = [16, None, 64, 8, None, None, 32, 16, 16, 64, 8]
        handoffs, received, net = self._run(4, schedule)
        steps = [s for s, rows in enumerate(schedule) if rows is not None]
        self.assertEqual(received[0], [])
        for rank in range(1, 4):
            self.assertEqual([s for s, _ in received[rank]], steps)
            for step, got in received[rank]:
                rows = schedule[step]
                self.assertEqual(tuple(got["hidden_states"].shape), (rows, HIDDEN))
                want = float(100 * (rank - 1) + step) + 1.0
                self.assertTrue(torch.all(got["hidden_states"] == want))
                self.assertTrue(torch.all(got["residual"] == 0))
        for handoff in handoffs[:-1]:
            self.assertEqual(handoff.sent, len(steps))
        # Transfer 1 is the self-test; forward k travels as transfer k + 1.
        self.assertEqual(
            sorted(tag for src, dst, tag in net.sent_tags if src == 10),
            list(range(1, 2 + len(steps))),
        )

    def test_zero_residual_is_one_tensor_per_row_count(self):
        schedule = [16, 32, 16, 16]
        _, received, _ = self._run(2, schedule)
        residuals = [got["residual"] for _, got in received[1]]
        self.assertIs(residuals[0], residuals[2])
        self.assertIs(residuals[0], residuals[3])
        self.assertIsNot(residuals[0], residuals[1])
        self.assertIsNone(residuals[0]._base)

    def test_aux_slots_travel_with_the_activations(self):
        schedule = [8, 16, 8]
        _, received, net = self._run(3, schedule, num_aux=2)
        for rank in (1, 2):
            for step, got in received[rank]:
                aux = got[AUX_HIDDEN_STATES_PROXY_KEY]
                self.assertEqual(tuple(aux.shape), (schedule[step], 2 * HIDDEN))
                self.assertTrue(torch.all(aux == -float(100 * (rank - 1) + step)))
        # Both tensors of a transfer go in one call (the self-test included).
        self.assertEqual(
            len([t for t in net.sent_tags if t[0] == 10]), len(schedule) + 1
        )

    def test_aux_slots_keep_their_own_dtype(self):
        schedule = [8, 16]
        _, received, _ = self._run(2, schedule, num_aux=1, aux_dtype=torch.bfloat16)
        for step, got in received[1]:
            self.assertEqual(got["hidden_states"].dtype, torch.float32)
            aux = got[AUX_HIDDEN_STATES_PROXY_KEY]
            self.assertEqual(aux.dtype, torch.bfloat16)
            self.assertTrue(torch.all(aux == -float(step)))

    def test_depth_one_still_completes(self):
        self._run(4, [8] * 12, depth=1)


class TestPPBufferedHandoffBounds(unittest.TestCase):
    def test_receives_outstanding_are_bounded(self):
        """The last stage waits on the oldest forward's fence before posting
        a receive that would exceed the device's receive flags."""
        net = FakeNet()
        handoff = make_handoff(1, 2, net)
        # Nothing sends here; on a device the host never waits for landing.
        handoff._transport.wait_landed = lambda tensor: None
        for step in range(12):
            handoff.begin_iteration(True)
            handoff.receive(8)
            handoff.end_iteration(None)
            self.assertLessEqual(len(handoff._landing), _MAX_RECEIVES)
        # Twelve receives, at most seven outstanding at a post: the oldest
        # five fences were waited on.
        self.assertEqual(len(handoff._transport.synced), 12 - _MAX_RECEIVES)
        self.assertEqual(handoff._transport.flushed_all, 12)

    def test_an_aux_transfer_takes_one_receive_flag(self):
        """Both tensors of a transfer are posted in one call, so the hidden
        states and the aux slots together count once against the flags."""
        net = FakeNet()
        handoff = make_handoff(1, 2, net, num_aux=1)
        handoff._transport.wait_landed = lambda tensor: None
        for _ in range(12):
            handoff.begin_iteration(True)
            handoff.receive(8)
            handoff.end_iteration(None)
        self.assertEqual(len(handoff._transport.synced), 12 - _MAX_RECEIVES)
        self.assertEqual(len(net.posted), 12)
        self.assertTrue(all(len(posted) == 2 for posted in net.posted.values()))

    def test_send_queue_blocks_at_depth(self):
        net = FakeNet()
        gate = threading.Event()
        handoff = make_handoff(0, 2, net, depth=2, send_gate=gate)
        # Post the next stage's receives up front (no self-test here, so the
        # forwards are transfers 1 to 3).
        peer = FakeTransport(net, 11)
        for transfer in range(1, 4):
            peer.recv_list([torch.empty((8, HIDDEN))], src=10, tag=transfer)
        queued = []

        def produce():
            for step in range(3):
                handoff.begin_iteration(True)
                handoff.receive(8)
                handoff.end_iteration(outgoing(0, step, 8, 0))
                queued.append(step)

        producer = threading.Thread(target=produce)
        producer.start()
        producer.join(timeout=0.5)
        # Two queued, the third waits for room.
        self.assertTrue(producer.is_alive())
        self.assertEqual(queued, [0, 1])
        gate.set()
        producer.join(timeout=10)
        self.assertEqual(queued, [0, 1, 2])
        handoff.close()
        self.assertEqual(handoff.sent, 3)

    def test_aux_of_another_dtype_is_refused(self):
        handoff = make_handoff(0, 2, FakeNet(), num_aux=1, aux_dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            handoff.merge(outgoing(0, 0, 8, 1, aux_dtype=torch.float32))

    def test_residual_is_optional(self):
        handoff = make_handoff(0, 2, FakeNet(), num_aux=1)
        tensors = outgoing(0, 3, 8, 1)
        summed = handoff.merge(tensors)
        self.assertTrue(torch.equal(summed[0], torch.full((8, HIDDEN), 4.0)))
        for residual in ("missing", None):
            without = dict(tensors)
            if residual == "missing":
                del without["residual"]
            else:
                without["residual"] = None
            merged = handoff.merge(without)
            self.assertTrue(torch.equal(merged[0], tensors["hidden_states"]))
            self.assertTrue(
                torch.equal(merged[1], tensors[AUX_HIDDEN_STATES_PROXY_KEY])
            )
        with self.assertRaises(ValueError):
            handoff.merge({"residual": tensors["residual"]})

    def test_post_time_log_covers_one_window(self):
        handoff = make_handoff(1, 2, FakeNet())
        with self.assertLogs("sglang_tpu.pp_buffered", level="INFO") as logs:
            # One slow post in the first window, then a fast second window.
            handoff._note_post(0.050)
            for _ in range(2 * _POST_LOG_EVERY - 1):
                handoff._note_post(0.001)
        lines = [m for m in (r.getMessage() for r in logs.records) if "posted" in m]
        self.assertEqual(len(lines), 2)
        self.assertIn("50.000 ms max", lines[0])
        self.assertIn("1.000 ms max", lines[1])
        self.assertIn("1.000 ms mean", lines[1])

    def test_post_time_log_counts_threaded_posts_in_their_window(self):
        handoff = make_handoff(1, 2, FakeNet())
        with self.assertLogs("sglang_tpu.pp_buffered", level="INFO") as logs:
            # The window's last post is the threaded one: it counts in this
            # window's line, not the next.
            for _ in range(_POST_LOG_EVERY - 1):
                handoff._note_post(0.001)
            handoff._note_post(0.001, threaded=True)
            for _ in range(_POST_LOG_EVERY):
                handoff._note_post(0.001)
        lines = [m for m in (r.getMessage() for r in logs.records) if "posted" in m]
        self.assertIn("1 of them on the posting thread", lines[0])
        self.assertIn("0 of them on the posting thread", lines[1])

    def test_rows_outside_the_buffer_are_refused(self):
        handoff = make_handoff(1, 2, FakeNet(), max_rows=32)
        handoff.begin_iteration(True)
        with self.assertRaises(RuntimeError):
            handoff.receive(64)

    def test_receive_outside_a_batch_iteration_is_refused(self):
        handoff = make_handoff(1, 2, FakeNet())
        handoff.begin_iteration(False)
        with self.assertRaises(RuntimeError):
            handoff.receive(8)

    def test_send_failure_surfaces_on_the_scheduler_thread(self):
        net = FakeNet()
        handoff = make_handoff(0, 2, net)
        handoff._transport.send_list = Mock(side_effect=RuntimeError("link down"))
        handoff.begin_iteration(True)
        handoff.receive(8)
        handoff.end_iteration(outgoing(0, 0, 8, 0))
        with self.assertRaises(RuntimeError):
            handoff.close()


class TestWireShape(unittest.TestCase):
    def test_widths_off_128_travel_128_wide(self):
        self.assertEqual(PPBufferedHandoff._wire_shape(512, 2880), (11520, 128))
        # Aligned widths travel 128 wide too: XLA might lay them out
        # column-major at the same byte size, which would scramble silently.
        self.assertEqual(PPBufferedHandoff._wire_shape(512, 2048), (8192, 128))
        # Not a whole number of 128-wide rows: sent as is.
        self.assertEqual(PPBufferedHandoff._wire_shape(16, 4), (16, 4))

    def test_round_trip_keeps_values_and_owns_storage(self):
        handoff = make_handoff(1, 2, FakeNet())
        x = torch.arange(64 * HIDDEN, dtype=torch.float32).reshape(64, HIDDEN)
        wire = handoff._to_wire(x)
        self.assertEqual(tuple(wire.shape), (2, 128))
        self.assertIsNone(wire._base)
        back = handoff._from_wire(wire, 64, HIDDEN)
        self.assertTrue(torch.equal(back, x))
        self.assertIsNone(back._base)


class TestWarmup(unittest.TestCase):
    def _programs(self, rank, size):
        handoff = make_handoff(rank, size, FakeNet(), num_aux=1)
        flushed = []
        handoff._transport.flush = lambda tensors: flushed.extend(
            tuple(x.shape) for x in tensors
        )
        handoff.warmup_merge([16, 64, 128])
        return flushed

    def test_every_bucket_compiles_its_wire_programs(self):
        """A middle stage warms the outgoing sum and wire copy and the copy
        back from the wire for each bucket up to its buffer (64 rows)."""
        shapes = self._programs(1, 3)
        # Send side: hidden [rows, 4] and aux [rows, 4]; 16 x 4 is not a whole
        # number of 128-wide rows and travels as it is.
        self.assertEqual(shapes.count((16, 4)), 4)
        self.assertEqual(shapes.count((2, 128)), 2)
        # Receive side rebuilds [64, 4] from the wire for both tensors.
        self.assertEqual(shapes.count((64, 4)), 2)
        self.assertNotIn((128, 4), shapes)

    def test_first_and_last_stages_warm_only_their_side(self):
        first = self._programs(0, 3)
        last = self._programs(2, 3)
        self.assertIn((2, 128), first)
        self.assertNotIn((64, 4), first)
        self.assertIn((64, 4), last)
        self.assertNotIn((2, 128), last)


class TestHbmReserve(unittest.TestCase):
    def test_worst_case_counts_receives_sends_and_zero_residuals(self):
        # GPT-OSS-120B PP8 at 16384 rows: 7 receives + 3 sends of 94 MiB, plus
        # one zero residual per bucket (512..16384 tokens, 8..64 requests).
        transfer = 16384 * 2880 * 2
        zero_rows = 512 + 1024 + 2048 + 4096 + 8192 + 16384 + 8 + 16 + 32 + 64
        args = dict(
            max_rows=16384,
            hidden=2880,
            dtype=torch.bfloat16,
            aux_dtype=torch.bfloat16,
            depth=2,
            zero_residual_rows=zero_rows,
        )
        plain = hbm_reserve_bytes(num_aux=0, **args)
        self.assertEqual(plain, 10 * transfer + zero_rows * 2880 * 2)
        self.assertAlmostEqual(plain / (1 << 30), 1.05, places=2)
        # Three Eagle3 aux slots travel in the same transfers.
        self.assertEqual(
            hbm_reserve_bytes(num_aux=3, **args), 40 * transfer + zero_rows * 2880 * 2
        )

    def test_start_up_check_allocates_the_worst_case(self):
        handoff = make_handoff(1, 2, FakeNet(), max_rows=64, depth=2)
        checked = handoff.check_hbm(zero_residual_rows=96)
        self.assertEqual(checked, handoff.reserve_bytes(96))
        self.assertEqual(handoff._transport.checked, [checked])

    def test_start_up_check_fails_with_the_size_it_needs(self):
        handoff = make_handoff(1, 2, FakeNet(), max_rows=64)
        handoff._transport.fail_check = True
        with self.assertRaisesRegex(RuntimeError, "worst case of .* GiB"):
            handoff.check_hbm(zero_residual_rows=96)


class TestKvBudgetReserve(unittest.TestCase):
    def _kvc(self, *, pp_size=8, is_draft_worker=False):
        return SimpleNamespace(
            pp_size=pp_size,
            is_draft_worker=is_draft_worker,
            model=SimpleNamespace(),
            model_config=SimpleNamespace(context_len=4096, hidden_size=16),
            model_dtype=torch.bfloat16,
            server_args=SimpleNamespace(),
            spec_aux_config=SpecAuxHiddenStateConfig(),
        )

    def test_buffered_handoff_takes_its_worst_case_off_the_kv_budget(self):
        original = Mock(return_value=10 << 30)
        with (
            tpu_envs.SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF.override(True),
            patch.object(
                pp_scheduler, "buffered_handoff_reserve_bytes", return_value=1 << 30
            ) as reserve,
        ):
            self.assertEqual(
                reserve_pp_buffered_handoff(original, self._kvc()), 9 << 30
            )
        reserve.assert_called_once()

    def test_budget_is_unchanged_without_the_switch_one_stage_or_for_a_draft(self):
        cases = [
            (False, self._kvc()),
            (True, self._kvc(pp_size=1)),
            (True, self._kvc(is_draft_worker=True)),
        ]
        for switch, kvc in cases:
            with (
                self.subTest(switch=switch, pp=kvc.pp_size, draft=kvc.is_draft_worker),
                tpu_envs.SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF.override(switch),
                patch.object(pp_scheduler, "buffered_handoff_reserve_bytes") as reserve,
            ):
                original = Mock(return_value=10 << 30)
                self.assertEqual(reserve_pp_buffered_handoff(original, kvc), 10 << 30)
                reserve.assert_not_called()

    def test_reserve_uses_the_forward_buckets_and_transfer_dtype(self):
        kvc = self._kvc()
        kvc.model = SimpleNamespace(pp_transfer_dtype=torch.float32)
        buckets = (
            Bucketer(name="Token", buckets=[64, 256]),
            Bucketer(name="Bs", buckets=[1, 32]),
        )
        with patch(
            "sglang_tpu.compilation.decoder_compile_policy.forward_bucketers",
            return_value=buckets,
        ):
            got = pp_scheduler.buffered_handoff_reserve_bytes(
                model=kvc.model,
                model_config=kvc.model_config,
                model_dtype=kvc.model_dtype,
                server_args=kvc.server_args,
                spec_aux_config=kvc.spec_aux_config,
            )
        # 256 rows of 16 fp32 columns, 7 receives + 3 sends; zeros for 64+256+1+32 rows.
        self.assertEqual(got, 10 * 256 * 16 * 4 + (64 + 256 + 1 + 32) * 16 * 4)


class TestPPBufferedHandoffProbe(unittest.TestCase):
    def test_probe_checks_the_sender(self):
        net = FakeNet()
        handoffs = [make_handoff(r, 3, net, num_aux=1) for r in range(3)]
        verdicts = [None] * 3

        def probe(r):
            verdicts[r] = handoffs[r].probe()

        threads = [threading.Thread(target=probe, args=(r,)) for r in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(verdicts, [True, True, True])
        self.assertTrue(all(handoff.transfers == 1 for handoff in handoffs))

    def test_probe_pattern_is_exact_and_distinct_on_large_worlds(self):
        handoff = make_handoff(0, 2, FakeNet())
        patterns = {}
        for rank in range(256):
            pattern = handoff._probe_pattern(rank, 64, 256, torch.float32)
            # Every value survives bfloat16 unchanged, at any rank.
            self.assertTrue(
                torch.equal(pattern.to(torch.bfloat16).float(), pattern), rank
            )
            patterns[rank] = pattern
        for rank in range(256):
            for other in range(rank + 1, min(rank + 128, 256)):
                self.assertFalse(torch.equal(patterns[rank], patterns[other]))
        # A transposed square block does not compare equal.
        square = handoff._probe_pattern(15, 128, 128, torch.float32)
        self.assertFalse(torch.equal(square, square.t()))


class TestPostingThread(unittest.TestCase):
    @staticmethod
    def _recording(handoff):
        """Record which thread posts each receive. No sender here: what is
        tested is where the post runs, not the data."""
        posted_on = []
        recv_list = handoff._transport.recv_list

        def record_thread(tensors, src, tag):
            posted_on.append(threading.current_thread().name)
            return recv_list(tensors, src, tag)

        handoff._transport.recv_list = record_thread
        handoff._transport.wait_landed = lambda tensor: None
        return posted_on

    @staticmethod
    def _gate_fences(handoff):
        gate = threading.Event()

        class GatedFence:
            def __init__(self, log):
                self.log = log

            def synchronize(self):
                gate.wait(10)
                self.log.append(self)

            def query(self):
                return gate.is_set()

        handoff._transport.fence = lambda: GatedFence(handoff._transport.synced)
        return gate

    def test_post_without_a_fence_wait_runs_inline(self):
        handoff = make_handoff(1, 2, FakeNet())
        posted_on = self._recording(handoff)
        handoff.begin_iteration(True)
        pending = handoff.post_receive(8)
        self.assertTrue(callable(pending))
        tensors = pending()
        self.assertEqual(tuple(tensors["hidden_states"].shape), (8, HIDDEN))
        self.assertEqual(posted_on, [threading.current_thread().name])
        self.assertIsNone(handoff._post_thread)

    def test_a_post_that_waits_on_a_fence_runs_on_the_posting_thread(self):
        # With 7 receives outstanding, the next post waits for the oldest
        # forward's fence; it does so on the posting thread, and queuing it
        # returns at once.
        net = FakeNet()
        handoff = make_handoff(1, 2, net)
        posted_on = self._recording(handoff)
        gate = self._gate_fences(handoff)
        for _ in range(_MAX_RECEIVES):
            handoff.begin_iteration(True)
            handoff.receive(8)
            handoff.end_iteration(None)
        handoff.begin_iteration(True)
        pending = handoff.post_receive(8)
        # The post is held behind the fence, off this thread.
        time.sleep(0.2)
        self.assertEqual(len(net.posted), _MAX_RECEIVES)
        gate.set()
        pending()
        self.assertEqual(len(handoff._transport.synced), 1)
        self.assertEqual(posted_on[-1], "pp-buffered-post")
        self.assertEqual(set(posted_on[:-1]), {threading.current_thread().name})
        handoff.close()

    def test_an_inline_post_failure_raises_at_once(self):
        handoff = make_handoff(1, 2, FakeNet())

        def fail(tensors, src, tag):
            raise RuntimeError("store unavailable")

        handoff._transport.recv_list = fail
        handoff.begin_iteration(True)
        with self.assertRaisesRegex(RuntimeError, "posting transfer 1 failed"):
            handoff.post_receive(8)

    def test_landed_fences_let_the_post_run_inline(self):
        # Seven receives outstanding, but their forwards have all completed:
        # the next post drops those fences without a wait and runs inline.
        handoff = make_handoff(1, 2, FakeNet())
        posted_on = self._recording(handoff)
        gate = self._gate_fences(handoff)
        gate.set()
        for _ in range(_MAX_RECEIVES + 1):
            handoff.begin_iteration(True)
            handoff.receive(8)
            handoff.end_iteration(None)
        self.assertEqual(set(posted_on), {threading.current_thread().name})
        self.assertEqual(handoff._transport.synced, [])
        self.assertIsNone(handoff._post_thread)

    def test_a_threaded_post_failure_surfaces_on_the_scheduler_thread(self):
        handoff = make_handoff(1, 2, FakeNet())
        self._recording(handoff)
        gate = self._gate_fences(handoff)
        for _ in range(_MAX_RECEIVES):
            handoff.begin_iteration(True)
            handoff.receive(8)
            handoff.end_iteration(None)

        def fail(tensors, src, tag):
            raise RuntimeError("store unavailable")

        handoff._transport.recv_list = fail
        handoff.begin_iteration(True)
        pending = handoff.post_receive(8)
        gate.set()
        with self.assertRaisesRegex(RuntimeError, "posting transfer 8 failed"):
            pending()
        with self.assertRaisesRegex(RuntimeError, "posting thread failed"):
            handoff.begin_iteration(True)

    def test_launch_waits_for_the_pending_receive(self):
        scheduler = SchedulerPPMixin()
        scheduler.forward_stream_ctx = contextlib.nullcontext()
        scheduler.use_overlap_streams = False
        seen = []

        class Stop(Exception):
            pass

        def run_batch(batch, proxy):
            seen.append(proxy)
            raise Stop

        scheduler.run_batch = run_batch
        tensors = {"hidden_states": torch.ones(2, HIDDEN)}
        with self.assertRaises(Stop):
            scheduler._pp_launch_batch(
                0, SimpleNamespace(reqs=[]), lambda: tensors, [None], None
            )
        self.assertIsInstance(seen[0], PPProxyTensors)
        self.assertIs(seen[0].tensors, tensors)


class TestReceiveProxy(unittest.TestCase):
    def test_scheduler_receive_takes_the_batch_padded_rows(self):
        handoff = make_handoff(1, 2, FakeNet())
        tensors = {"hidden_states": torch.zeros(1)}
        handoff.post_receive = Mock(return_value=lambda: tensors)
        policy = Mock()
        policy.forward_rows.return_value = 128
        scheduler = SchedulerPPMixin()
        scheduler._pp_wave = handoff
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(compile_policy=policy)
        )
        handoff.rows_of = lambda batch: pp_scheduler.pp_forward_rows(scheduler, batch)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            reqs=[object(), object()],
            input_ids=torch.zeros(100, dtype=torch.int64),
            extend_num_tokens=100,
            spec_info=None,
        )
        proxy = scheduler._pp_recv_proxy(batch)
        policy.forward_rows.assert_called_once_with(ForwardMode.EXTEND, 100, 2, None)
        handoff.post_receive.assert_called_once_with(128)
        # A pending receive, waited for by _pp_launch_batch.
        self.assertTrue(callable(proxy))
        self.assertIs(proxy(), tensors)

    def test_first_stage_receives_nothing(self):
        handoff = make_handoff(0, 2, FakeNet())
        handoff.begin_iteration(True)
        self.assertIsNone(handoff.receive_batch(object()))

    def _cp_scheduler(self):
        policy = Mock()
        policy.cp_forward_rows.return_value = 512
        policy.forward_rows.return_value = 1024
        return SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(compile_policy=policy)
            )
        ), policy

    def _cp_batch(self, cp_extend=True):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_context_parallel_extend=lambda: cp_extend,
                is_decode=lambda: False,
            ),
            reqs=[object(), object()],
            input_ids=torch.zeros(900, dtype=torch.int64),
            extend_lens=[600, 300],
            extend_num_tokens=900,
            spec_info=None,
        )

    def test_context_parallel_rows_are_the_local_bucket(self):
        scheduler, policy = self._cp_scheduler()
        strategy = Mock()
        strategy.can_apply.return_value = True
        with patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy):
            rows = pp_scheduler.pp_forward_rows(scheduler, self._cp_batch())
        self.assertEqual(rows, 512)
        policy.cp_forward_rows.assert_called_once_with([600, 300])
        policy.forward_rows.assert_not_called()
        num_tokens, view = strategy.can_apply.call_args.args
        self.assertEqual(num_tokens, 900)
        self.assertEqual(view.extend_seq_lens_cpu, [600, 300])

    def test_rows_fall_back_when_the_cp_forward_does_not_apply(self):
        """A batch the CP strategy declines, or a non-extend batch, runs the
        plain forward and pads the global token count."""
        for can_apply, cp_extend in ((False, True), (True, False)):
            with self.subTest(can_apply=can_apply, cp_extend=cp_extend):
                scheduler, policy = self._cp_scheduler()
                strategy = Mock()
                strategy.can_apply.return_value = can_apply
                with patch(
                    "sglang.srt.layers.cp.base.get_cp_strategy", return_value=strategy
                ):
                    rows = pp_scheduler.pp_forward_rows(
                        scheduler, self._cp_batch(cp_extend)
                    )
                self.assertEqual(rows, 1024)
                policy.cp_forward_rows.assert_not_called()

    def test_decode_rows_follow_the_batch_size(self):
        policy = Mock()
        policy.forward_rows.return_value = 8
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(compile_policy=policy)
            ),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            reqs=[object()] * 5,
            input_ids=torch.zeros(40, dtype=torch.int64),
            extend_num_tokens=None,
            spec_info=None,
        )
        self.assertEqual(pp_scheduler.pp_forward_rows(scheduler, batch), 8)
        policy.forward_rows.assert_called_once_with(ForwardMode.DECODE, 5, 5, None)


class TestBufferedHandoffWiring(unittest.TestCase):
    """SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF: what it builds, what it refuses, and the shapes
    the compiled forward is warmed up and checked with."""

    def _scheduler(self):
        policy = DecoderTpuCompilePolicy(
            t_bucketer=Bucketer(name="Token", buckets=[64, 256]),
            b_bucketer=Bucketer(name="Bs", buckets=[1, 32]),
        )
        scheduler = SchedulerPPMixin()
        scheduler.pp_group = SimpleNamespace(
            is_first_rank=False, rank_in_group=1, world_size=2, ranks=[0, 1]
        )
        scheduler.tp_group = SimpleNamespace()
        scheduler.world_group = SimpleNamespace()
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                compile_policy=policy,
                model_config=SimpleNamespace(context_len=4096, hidden_size=16),
                spec_aux_config=SpecAuxHiddenStateConfig(),
                pp_group=scheduler.pp_group,
                model=SimpleNamespace(pp_transfer_dtype=torch.float32),
                dtype=torch.bfloat16,
                device="cpu",
            )
        )
        policy.model_runner = scheduler.tp_worker.model_runner
        return scheduler, policy

    def _no_cross_rank_padding(self):
        return (
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
            patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=None),
        )

    def test_switch_builds_the_buffered_handoff_instead_of_the_wave(self):
        scheduler, _ = self._scheduler()
        dp, cp = self._no_cross_rank_padding()
        with (
            tpu_envs.SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF.override(True),
            dp,
            cp,
            patch.object(pp_scheduler, "build_pp_buffered_handoff") as handoff,
            patch.object(pp_scheduler, "build_pp_wave") as wave,
        ):
            scheduler._pp_init_wave()
        wave.assert_not_called()
        kwargs = handoff.call_args.kwargs
        self.assertEqual(kwargs["max_rows"], 256)
        self.assertEqual(kwargs["dtype"], torch.float32)
        self.assertEqual(kwargs["aux_dtype"], torch.bfloat16)
        self.assertEqual(kwargs["depth"], 2)
        self.assertTrue(callable(kwargs["rows_of"]))
        # The start-up memory check covers one zero residual per bucket.
        handoff.return_value.check_hbm.assert_called_once_with(64 + 256 + 1 + 32)

    def test_wave_stays_the_default(self):
        scheduler, _ = self._scheduler()
        with (
            patch.object(pp_scheduler, "build_pp_buffered_handoff") as handoff,
            patch.object(pp_scheduler, "build_pp_wave") as wave,
        ):
            scheduler._pp_init_wave()
        handoff.assert_not_called()
        wave.assert_called_once()

    def test_cp_rows_match_the_cp_forward_padding(self):
        """The receive size of a context-parallel stage comes from the same
        function the CP forward pads with (cp_local_rows_padded)."""
        policy = DecoderTpuCompilePolicy(
            t_bucketer=Bucketer(name="Token", buckets=[64, 256, 512]),
            b_bucketer=Bucketer(name="Bs", buckets=[1, 32]),
        )
        with patch(
            "sglang_tpu.compilation.decoder_compile_policy.get_cp_strategy",
            return_value=SimpleNamespace(cp_size=2),
        ):
            rows = policy.cp_forward_rows([600, 300])
        # Zigzag: each rank holds half of each request, 300 + 150 = 450 rows.
        self.assertEqual(rows, 512)
        self.assertEqual(
            rows,
            cp_local_rows_padded(
                [600, 300],
                pad_b=policy._extend_compile_cap(),
                t_bucketer=policy.t_bucketer,
                cp_size=2,
            ),
        )

    def test_buffered_refuses_dp_attention(self):
        scheduler, _ = self._scheduler()
        with (
            tpu_envs.SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF.override(True),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=True,
            ),
            patch.object(pp_scheduler, "build_pp_buffered_handoff") as handoff,
        ):
            with self.assertRaisesRegex(NotImplementedError, "DP attention"):
                scheduler._pp_init_wave()
            handoff.assert_not_called()

    def test_buffered_accepts_context_parallel_stages(self):
        """A context-parallel stage derives each receive from its local rows."""
        scheduler, _ = self._scheduler()
        with (
            tpu_envs.SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF.override(True),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
            patch("sglang.srt.layers.cp.base.get_cp_strategy", return_value=object()),
            patch.object(pp_scheduler, "build_pp_buffered_handoff") as handoff,
        ):
            scheduler._pp_init_wave()
        handoff.assert_called_once()

    def test_shim_takes_exactly_the_padded_rows_under_the_buffered_handoff(self):
        proxy = PPProxyTensors(
            {"hidden_states": torch.zeros(16, 2), "residual": torch.zeros(16, 2)}
        )
        with tpu_envs.SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF.override(True):
            with self.assertRaisesRegex(ValueError, "buffered hand-off"):
                _fit_pp_proxy_rows(proxy, torch.zeros(8), None)
            self.assertIs(_fit_pp_proxy_rows(proxy, torch.zeros(16), None), proxy)
        # The wave trims its full-height buffer instead.
        trimmed = _fit_pp_proxy_rows(proxy, torch.zeros(8), None)
        self.assertEqual(trimmed.tensors["hidden_states"].shape[0], 8)

    def test_warmup_proxies_are_whole_tensors_under_the_buffered_handoff(self):
        _, policy = self._scheduler()
        with tpu_envs.SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF.override(True):
            buffered_proxy = policy._pp_warmup_kwargs(64)["pp_proxy_tensors"]
        wave_proxy = policy._pp_warmup_kwargs(64)["pp_proxy_tensors"]
        for name in ("hidden_states", "residual"):
            self.assertEqual(buffered_proxy[name].shape[0], 64)
            self.assertIsNone(buffered_proxy[name]._base)
            self.assertEqual(buffered_proxy[name].dtype, torch.float32)
            # The wave warms row views of its full-height buffer.
            self.assertIsNotNone(wave_proxy[name]._base)


if __name__ == "__main__":
    unittest.main()
