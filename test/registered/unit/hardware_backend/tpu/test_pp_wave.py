"""Unit tests for the TPU pipeline hand-off wave.

CPU only: the collective-permute is a threaded fake that pairs launch k on
every stage and hands each stage's block to the next around the cycle,
blocking a stage until its peers reach the same launch, the way the device
does.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import threading
import unittest

import torch

from sglang.srt.hardware_backend.tpu.pp_wave import PPWave, pipeline_permutation
from sglang.test.test_utils import CustomTestCase

ROWS, HIDDEN = 8, 4


class _FakeIci:
    """Launch k on every stage exchanges blocks: stage r receives what stage
    r-1 sent at launch k, stage 0 what the last stage sent."""

    def __init__(self, size: int, timeout: float = 10.0):
        self.size = size
        self.timeout = timeout
        self.cond = threading.Condition()
        self.sent: dict[tuple[int, int], tuple] = {}
        self.count = [0] * size
        self.failed = False

    def launcher(self, rank: int):
        def launch(x):
            with self.cond:
                k = self.count[rank]
                self.sent[(k, rank)] = x.clone()
                self.count[rank] += 1
                self.cond.notify_all()
                ok = self.cond.wait_for(
                    lambda: self.failed or all(c > k for c in self.count),
                    timeout=self.timeout,
                )
                if not ok or self.failed:
                    self.failed = True
                    self.cond.notify_all()
                    raise RuntimeError(
                        f"stage {rank} launch {k}: peers never launched it "
                        f"(counts {self.count})"
                    )
                return self.sent[(k, (rank - 1) % self.size)]

        return launch


def _wave(rank, size, ici=None):
    return PPWave(
        rank=rank,
        size=size,
        rows=ROWS,
        hidden=HIDDEN,
        dtype=torch.float32,
        device=torch.device("cpu"),
        launcher=ici.launcher(rank) if ici else (lambda x: x),
    )


def _expected_counts(size, schedule):
    """Per iteration, the launch count every stage must have reached: stage 0
    launches once per iteration with a batch or while it owes the drain of
    2 * (size - 2), stage r keeps min(r-1, owed) launches ahead of it."""
    counts = []
    stage0, owed = 0, 0
    for has_batch in schedule:
        if has_batch:
            stage0, owed = stage0 + 1, 2 * (size - 2)
        elif owed:
            stage0, owed = stage0 + 1, owed - 1
        counts.append([stage0] + [stage0 + min(r - 1, owed) for r in range(1, size)])
    return counts


def _run_schedule(size, schedule, rows_per_batch=None):
    """Every stage runs ``schedule`` (True = an iteration with a batch). Stage 0
    emits batch j as hidden states 100 + j with residual 0.5, so the sum
    100.5 + j arrives; each later stage adds one. Returns the per-stage
    received sums, the per-stage launch count after every iteration, the
    waves, any stage error and the fake ICI."""
    ici = _FakeIci(size)
    waves = [_wave(r, size, ici) for r in range(size)]
    received = [[] for _ in range(size)]
    counts = [[] for _ in range(size)]
    errors = []

    def run(rank):
        wave = waves[rank]
        batch = 0
        try:
            for has_batch in schedule:
                wave.begin_iteration(has_batch)
                tensors = None
                if has_batch:
                    rows = rows_per_batch[batch] if rows_per_batch else ROWS
                    incoming = wave.receive()
                    if rank == 0:
                        out = torch.full((rows, HIDDEN), 100.0 + batch)
                    else:
                        received[rank].append(float(incoming["hidden_states"][0, 0]))
                        assert float(incoming["residual"].abs().max()) == 0.0, rank
                        assert (
                            float(incoming["hidden_states"][rows - 1, HIDDEN - 1])
                            == received[rank][-1]
                        ), (rank, batch)
                        out = incoming["hidden_states"][:rows] + 1.0
                    if rank < size - 1:
                        tensors = {
                            "hidden_states": out,
                            "residual": torch.full_like(out, 0.5),
                        }
                    batch += 1
                wave.end_iteration(tensors)
                counts[rank].append(wave.launches)
        except Exception as exc:  # noqa: BLE001
            errors.append((rank, exc))
            with ici.cond:
                ici.failed = True
                ici.cond.notify_all()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return received, counts, waves, errors, ici


def _drained(size, schedule):
    return schedule + [False] * (2 * (size - 2))


class TestWaveProtocol(CustomTestCase):
    def test_activations_reach_every_stage_in_order_across_gaps(self):
        """Batch j leaves stage 0 as the sum 100.5+j and must arrive at stage
        r as 100.5+j+1.5*(r-1) for every j: across gaps inside the drain, a
        gap longer than the drain, and single batches."""
        size = 4
        schedule = _drained(
            size,
            [True, True, True, False, False, True, False, True, True]
            + [False] * 5
            + [True],
        )
        received, counts, waves, errors, ici = _run_schedule(size, schedule)
        self.assertEqual(errors, [])
        n = schedule.count(True)
        for rank in range(1, size):
            self.assertEqual(
                received[rank],
                [100.5 + j + 1.5 * (rank - 1) for j in range(n)],
                rank,
            )
        self.assertEqual(len(set(ici.count)), 1)
        for wave in waves:
            self.assertIsNone(wave.pending)
            self.assertEqual(wave.owed, 0)
            self.assertEqual(wave.launches, wave.stage0_launches)

    def test_stage_zero_launches_one_hop_per_iteration_and_stages_keep_pace(self):
        """Stage 0 issues at most one launch per iteration (a send, or a zero
        launch while draining), and stage r is at most r-1 launches ahead."""
        size = 5
        schedule = _drained(size, [True] * 3 + [False] * 7 + [True])
        _, counts, _, errors, _ = _run_schedule(size, schedule)
        self.assertEqual(errors, [])
        expected = _expected_counts(size, schedule)
        for rank in range(size):
            self.assertEqual(
                counts[rank], [row[rank] for row in expected], f"stage {rank}"
            )
        # A gap longer than the drain stops every stage; the run restarts
        # with stage r launching r-1 hops ahead of stage 0 again.
        self.assertEqual(expected[9], [9, 9, 9, 9, 9])
        self.assertEqual(expected[10], [10, 10, 11, 12, 13])

    def test_drain_is_twice_stages_minus_two_launches_after_the_last_batch(self):
        size = 5
        schedule = _drained(size, [True] * 6 + [False])
        _, _, waves, errors, ici = _run_schedule(size, schedule)
        self.assertEqual(errors, [])
        self.assertEqual(ici.count, [6 + 2 * (size - 2)] * size)
        for wave in waves:
            self.assertEqual(wave.owed, 0)
            self.assertIsNone(wave.pending)

    def test_smaller_batches_are_padded_to_the_buffer_and_trimmed_by_rows(self):
        size = 3
        schedule = _drained(size, [True, True, False])
        received, _, _, errors, _ = _run_schedule(size, schedule, rows_per_batch=[3, 5])
        self.assertEqual(errors, [])
        self.assertEqual(received[2], [102.0, 103.0])

    def test_two_stages_launch_once_per_batch_and_nothing_else(self):
        size = 2
        schedule = [True, True, False, True, False]
        received, _, _, errors, ici = _run_schedule(size, schedule)
        self.assertEqual(errors, [])
        self.assertEqual(received[1], [100.5, 101.5, 102.5])
        self.assertEqual(ici.count, [3, 3])


class TestWaveLaunchSlots(CustomTestCase):
    """torch_tpu runs every multi-device program on the whole slice's device
    assignment, matched by launch count; every hand-off must sit at the same
    launch index on every device, whatever the stages launch in between."""

    def _run(self, size, schedule, forward_launches, slot):
        ici = _FakeIci(size)
        streams = [[] for _ in range(size)]
        waves = []
        for r in range(size):
            base = ici.launcher(r)

            def launcher(x, r=r, base=base):
                streams[r].append("H")
                return base(x)

            wave = PPWave(
                rank=r,
                size=size,
                rows=ROWS,
                hidden=HIDDEN,
                dtype=torch.float32,
                device=torch.device("cpu"),
                launcher=launcher,
                slot_size=slot,
                slot_filler=lambda r=r: streams[r].append("f"),
            )
            wave.forward_launches = forward_launches[r]
            waves.append(wave)
        errors = []

        def run(rank):
            wave = waves[rank]
            try:
                for has_batch in schedule:
                    wave.begin_iteration(has_batch)
                    tensors = None
                    if has_batch:
                        wave.receive()
                        streams[rank].extend("L" * wave.forward_launches)
                        wave.note_stage_launches(wave.forward_launches)
                        if rank < size - 1:
                            out = torch.zeros((ROWS, HIDDEN))
                            tensors = {"hidden_states": out, "residual": out}
                    wave.end_iteration(tensors)
            except Exception as exc:  # noqa: BLE001
                errors.append((rank, exc))
                with ici.cond:
                    ici.failed = True
                    ici.cond.notify_all()

        threads = [threading.Thread(target=run, args=(r,)) for r in range(size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        return streams, waves, errors

    def test_hand_offs_sit_at_the_same_launch_index_on_every_stage(self):
        size, slot = 4, 3
        schedule = _drained(size, [True, True, False, True] + [False] * 5 + [True])
        streams, waves, errors = self._run(size, schedule, [2, 1, 3, 0], slot)
        self.assertEqual(errors, [])
        hand_offs = [[i for i, p in enumerate(s) if p == "H"] for s in streams]
        for r in range(1, size):
            # Stage r runs up to r-1 hand-offs ahead; compare the common prefix.
            n = min(len(hand_offs[0]), len(hand_offs[r]))
            self.assertEqual(hand_offs[r][:n], hand_offs[0][:n], f"stage {r}")
        # Drained: every stage issued the same programs count.
        self.assertEqual(len({len(s) for s in streams}), 1, streams)
        for s in streams:
            self.assertEqual(s[-1], "H")
        for wave in waves:
            self.assertEqual(wave.slot_used, 0)

    def test_a_forward_above_the_slot_is_an_error(self):
        wave = PPWave(
            rank=1,
            size=3,
            rows=ROWS,
            hidden=HIDDEN,
            dtype=torch.float32,
            device=torch.device("cpu"),
            launcher=lambda x: x,
            slot_size=2,
            slot_filler=lambda: None,
        )
        wave.note_stage_launches(2)
        with self.assertRaisesRegex(RuntimeError, "above the slot of 2"):
            wave.note_stage_launches(1)

    def test_per_graph_counts_keep_hand_offs_aligned(self):
        # Each stage's forwards issue different counts per graph (mode), as
        # reported by the measured-table counter.
        size, slot = 4, 3
        per_forward = [[2, 1, 3], [1, 1, 2], [0, 3, 1], [2, 2, 2]]
        schedule = _drained(size, [True, True, True, False, True] + [False] * 3)
        ici = _FakeIci(size)
        streams = [[] for _ in range(size)]
        waves = []
        for r in range(size):
            base = ici.launcher(r)

            def launcher(x, r=r, base=base):
                streams[r].append("H")
                return base(x)

            wave = PPWave(
                rank=r,
                size=size,
                rows=ROWS,
                hidden=HIDDEN,
                dtype=torch.float32,
                device=torch.device("cpu"),
                launcher=launcher,
                slot_size=slot,
                slot_filler=lambda r=r: streams[r].append("f"),
            )
            waves.append(wave)
        errors = []

        def run(rank):
            wave = waves[rank]
            forwards = iter(per_forward[rank] * 4)
            try:
                for has_batch in schedule:
                    wave.begin_iteration(has_batch)
                    tensors = None
                    if has_batch:
                        wave.receive()
                        n = next(forwards)
                        streams[rank].extend("L" * n)
                        wave.forward_launch_counter = lambda n=n: n
                        wave.note_forward()
                        if rank < size - 1:
                            out = torch.zeros((ROWS, HIDDEN))
                            tensors = {"hidden_states": out, "residual": out}
                    wave.end_iteration(tensors)
            except Exception as exc:  # noqa: BLE001
                errors.append((rank, exc))
                with ici.cond:
                    ici.failed = True
                    ici.cond.notify_all()

        threads = [threading.Thread(target=run, args=(r,)) for r in range(size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(errors, [])
        hand_offs = [[i for i, p in enumerate(s) if p == "H"] for s in streams]
        for r in range(1, size):
            n = min(len(hand_offs[0]), len(hand_offs[r]))
            self.assertEqual(hand_offs[r][:n], hand_offs[0][:n], f"stage {r}")
        self.assertEqual(len({len(s) for s in streams}), 1, streams)

    def test_a_slot_needs_a_filler(self):
        with self.assertRaisesRegex(ValueError, "needs a filler"):
            PPWave(
                rank=0,
                size=2,
                rows=ROWS,
                hidden=HIDDEN,
                dtype=torch.float32,
                device=torch.device("cpu"),
                launcher=lambda x: x,
                slot_size=1,
            )


class TestWaveGuards(CustomTestCase):
    def test_rejects_rows_above_the_buffer(self):
        wave = _wave(0, 2)
        wave.begin_iteration(True)
        big = torch.zeros(ROWS + 1, HIDDEN)
        with self.assertRaisesRegex(ValueError, "above the wave"):
            wave.end_iteration({"hidden_states": big, "residual": big})

    def test_rejects_a_hand_off_the_sum_would_change(self):
        """Only hidden states plus residual in the buffer's dtype pass: another
        key would be dropped, and a wider sum does not fit the buffer."""
        wave = _wave(0, 2)
        wave.begin_iteration(True)
        rows = torch.zeros(ROWS, HIDDEN)
        with self.assertRaisesRegex(ValueError, "hands on exactly"):
            wave.end_iteration(
                {"hidden_states": rows, "residual": rows, "topk_indices": rows}
            )
        with self.assertRaisesRegex(ValueError, "carries torch.float32"):
            wave.end_iteration({"hidden_states": rows, "residual": rows.double()})

    def test_receive_needs_an_iteration_with_a_batch(self):
        wave = _wave(1, 3)
        wave.begin_iteration(False)
        with self.assertRaisesRegex(RuntimeError, "outside an iteration"):
            wave.receive()

    def test_a_second_output_without_a_launch_between_is_an_error(self):
        wave = _wave(1, 3)
        out = {
            "hidden_states": torch.zeros(ROWS, HIDDEN),
            "residual": torch.zeros(ROWS, HIDDEN),
        }
        wave.begin_iteration(True)
        wave.receive()
        wave.end_iteration(out)
        with self.assertRaisesRegex(RuntimeError, "two forward outputs"):
            wave.end_iteration(out)

    def test_missing_receive_is_reported_at_iteration_end(self):
        wave = _wave(2, 3)
        wave.begin_iteration(True)
        with self.assertRaisesRegex(RuntimeError, "expected .*end of this iteration"):
            wave.end_iteration(None)

    def test_an_extra_launch_is_reported_at_iteration_end(self):
        wave = _wave(0, 3)
        wave.begin_iteration(False)
        wave._launch(wave.zero)
        with self.assertRaisesRegex(RuntimeError, "expected .*end of this iteration"):
            wave.end_iteration(None)

    def test_probe_checks_the_received_stage_number(self):
        size = 3
        ici = _FakeIci(size)
        waves = [_wave(r, size, ici) for r in range(size)]
        verdicts = [None] * size

        def run(rank):
            verdicts[rank] = waves[rank].probe()

        threads = [threading.Thread(target=run, args=(r,)) for r in range(size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(verdicts, [True] * size)
        self.assertEqual(len(set(ici.count)), 1)

        def wrong(x):
            return torch.full_like(x, 1.0)

        wave = _wave(2, 3)
        wave._launcher = wrong
        self.assertFalse(wave.probe())

    def test_probe_tells_the_device_columns_apart(self):
        def from_device(index):
            return lambda x: torch.full_like(x, float(index + 1))

        # Stage 2, column 1 of 2 receives from stage 1: device 3 is its column.
        wave = _wave(2, 3)
        wave._launcher = from_device(3)
        self.assertTrue(wave.probe(column=1, columns=2))
        wave = _wave(2, 3)
        wave._launcher = from_device(2)
        self.assertFalse(wave.probe(column=1, columns=2))

    def test_permutation_forms_one_cycle_per_column(self):
        physical = (4, 5, 0, 1, 6, 7, 2, 3)  # rank -> device, stages of two
        perm = pipeline_permutation(physical=physical, size=4, columns=2)
        # Column 0: ranks 0,2,4,6 -> devices 4,0,6,2; column 1: ranks 1,3,5,7.
        self.assertEqual(
            perm,
            ((4, 0), (0, 6), (6, 2), (2, 4), (5, 1), (1, 7), (7, 3), (3, 5)),
        )
        self.assertEqual(sorted(s for s, _ in perm), sorted(physical))
        self.assertEqual(sorted(t for _, t in perm), sorted(physical))
        self.assertEqual(
            pipeline_permutation(physical=(0, 1, 2), size=3, columns=1),
            ((0, 1), (1, 2), (2, 0)),
        )
        with self.assertRaisesRegex(ValueError, "cannot form"):
            pipeline_permutation(physical=(0, 1, 2), size=2, columns=2)


class _CountMatchedSlice:
    """Devices that start their k-th multi-device program only once every
    device issued k: a program's output is ready iff every count reached it.
    With ``latency``, an output reads not ready on its first query (still
    computing)."""

    def __init__(self, counts, latency=False):
        self.counts = list(counts)
        self.latency = latency
        self.queried = set()
        self.lock = threading.Lock()

    def filler(self, rank):
        def launch():
            with self.lock:
                self.counts[rank] += 1
                return (rank, self.counts[rank])

        return launch

    def is_ready(self, token):
        with self.lock:
            if self.latency and token not in self.queried:
                self.queried.add(token)
                return False
            return all(c >= token[1] for c in self.counts)


class _HostGroup:
    def __init__(self, n):
        self.buf = [None] * n
        self.barrier = threading.Barrier(n, timeout=10)

    def member(self, rank):
        group = self

        class _Member:
            def all_gather_object(self, obj):
                group.buf[rank] = obj
                group.barrier.wait()
                out = list(group.buf)
                group.barrier.wait()
                return out

        return _Member()


class TestLaunchLeveling(CustomTestCase):
    def _level(self, stage_counts, columns, latency=False):
        import sys
        import types
        from unittest import mock

        from sglang.srt.hardware_backend.tpu.pp_wave import level_stage_launches

        size = len(stage_counts)
        n = size * columns
        dev = _CountMatchedSlice(
            [stage_counts[r // columns] for r in range(n)], latency=latency
        )
        sync = types.ModuleType("torch_tpu._internal.sync.sync")
        sync.is_ready = dev.is_ready
        mods = {
            "torch_tpu": types.ModuleType("torch_tpu"),
            "torch_tpu._internal": types.ModuleType("torch_tpu._internal"),
            "torch_tpu._internal.sync": types.ModuleType("torch_tpu._internal.sync"),
            "torch_tpu._internal.sync.sync": sync,
        }
        host = _HostGroup(n)
        stage_group = types.SimpleNamespace(world_size=columns)
        fillers = [None] * n
        errors = []

        def run(rank):
            try:
                fillers[rank] = level_stage_launches(
                    stage_group=stage_group,
                    world_group=host.member(rank),
                    filler=dev.filler(rank),
                    pp_rank=rank // columns,
                    pp_size=size,
                    poll_s=0.0,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append((rank, exc))

        with mock.patch.dict(sys.modules, mods):
            threads = [threading.Thread(target=run, args=(r,)) for r in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
        return dev.counts, fillers, errors

    def test_the_lowest_stages_fill_up_to_the_highest_count(self):
        counts, fillers, errors = self._level([5, 3, 3, 7], columns=2)
        self.assertEqual(errors, [])
        self.assertEqual(len(set(counts)), 1, counts)
        # One uniform filler, then exactly the gap to the highest stage.
        self.assertEqual(counts[0], 7 + 1)
        self.assertEqual(fillers, [2, 2, 4, 4, 4, 4, 0, 0])

    def test_converges_when_launches_are_still_computing(self):
        counts, fillers, errors = self._level([2, 6, 4], columns=2, latency=True)
        self.assertEqual(errors, [])
        self.assertEqual(len(set(counts)), 1, counts)
        # A stage's devices always issue the same fillers.
        for s in range(3):
            self.assertEqual(fillers[2 * s], fillers[2 * s + 1])

    def test_level_counts_issue_no_filler(self):
        counts, fillers, errors = self._level([4, 4], columns=2)
        self.assertEqual(errors, [])
        self.assertEqual(fillers, [0, 0, 0, 0])
        self.assertEqual(counts, [5, 5, 5, 5])


if __name__ == "__main__":
    unittest.main()
