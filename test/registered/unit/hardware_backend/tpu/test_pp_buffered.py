"""The point-to-point pipeline hand-off (sglang_tpu/pp_buffered.py) over a fake
in-process transport."""

import threading
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
from sglang_tpu.envs import envs as tpu_envs  # noqa: E402
from sglang_tpu.pp_buffered import (  # noqa: E402
    _MAX_RECEIVES,
    PPBufferedHandoff,
    Transport,
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


class FakeTransport(Transport):
    def __init__(self, net, rank, *, send_gate=None):
        super().__init__()
        self.net = net
        self.rank = rank
        self.send_gate = send_gate
        self.synced = []
        self.flushed_all = 0

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


class TestReceiveProxy(unittest.TestCase):
    def test_scheduler_receive_takes_the_batch_padded_rows(self):
        handoff = make_handoff(1, 2, FakeNet())
        handoff.receive = Mock(return_value={"hidden_states": torch.zeros(1)})
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
        handoff.receive.assert_called_once_with(128)
        self.assertIn("hidden_states", proxy.tensors)

    def test_first_stage_receives_nothing(self):
        handoff = make_handoff(0, 2, FakeNet())
        handoff.begin_iteration(True)
        self.assertIsNone(handoff.receive_batch(object()))

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

    def test_wave_stays_the_default(self):
        scheduler, _ = self._scheduler()
        with (
            patch.object(pp_scheduler, "build_pp_buffered_handoff") as handoff,
            patch.object(pp_scheduler, "build_pp_wave") as wave,
        ):
            scheduler._pp_init_wave()
        handoff.assert_not_called()
        wave.assert_called_once()

    def test_buffered_refuses_dp_attention_and_prefill_cp(self):
        scheduler, _ = self._scheduler()
        for dp_on, cp_strategy, needle in (
            (True, None, "DP attention"),
            (False, object(), "context parallelism"),
        ):
            with (
                self.subTest(needle=needle),
                tpu_envs.SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF.override(True),
                patch(
                    "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                    return_value=dp_on,
                ),
                patch(
                    "sglang.srt.layers.cp.base.get_cp_strategy",
                    return_value=cp_strategy,
                ),
                patch.object(pp_scheduler, "build_pp_buffered_handoff") as handoff,
            ):
                with self.assertRaisesRegex(NotImplementedError, needle):
                    scheduler._pp_init_wave()
                handoff.assert_not_called()

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
