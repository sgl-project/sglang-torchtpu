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

from sglang.srt.layers.aux_hidden_states import AUX_HIDDEN_STATES_PROXY_KEY
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.pp_wave import PPAuxWave, PPWave, pipeline_permutation

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
                self.sent[(k, rank)] = (
                    tuple(t.clone() for t in x) if isinstance(x, tuple) else x.clone()
                )
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


def _wave(rank, size, ici=None, num_aux=0):
    launcher = ici.launcher(rank) if ici else (lambda x: x)
    if num_aux:
        return PPAuxWave(
            rank=rank,
            size=size,
            rows=ROWS,
            hidden=HIDDEN,
            num_aux=num_aux,
            dtype=torch.float32,
            device=torch.device("cpu"),
            launcher=launcher,
        )
    return PPWave(
        rank=rank,
        size=size,
        rows=ROWS,
        hidden=HIDDEN,
        dtype=torch.float32,
        device=torch.device("cpu"),
        launcher=launcher,
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


class TestWaveWarmup(CustomTestCase):
    def test_warmup_merges_every_bucket_the_buffer_holds(self):
        """Each forward bucket's sum and pad compiles at warmup, not on its first hand-off."""
        for num_aux in (0, 2):
            wave = _wave(0, 2, num_aux=num_aux)
            seen = []
            merge = wave.merge
            wave.merge = lambda tensors: (
                seen.append({k: tuple(v.shape) for k, v in tensors.items()})
                or merge(tensors)
            )
            wave.warmup_merge({2, ROWS, ROWS + 4})
            self.assertEqual(
                sorted(rows["hidden_states"][0] for rows in seen), [2, ROWS]
            )
            for rows in seen:
                self.assertEqual(len(rows), 2 + (num_aux > 0))


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
        # Column 0: ranks 0,2,4,6; column 1: ranks 1,3,5,7.
        cycles = ((0, 2), (2, 4), (4, 6), (6, 0), (1, 3), (3, 5), (5, 7), (7, 1))
        for stage in range(4):
            for column in range(2):
                with self.subTest(stage=stage, column=column):
                    perm = pipeline_permutation(
                        pp_ranks=tuple(range(column, 8, 2)),
                        stage_ranks=(2 * stage, 2 * stage + 1),
                        column=column,
                    )
                    self.assertEqual(perm, cycles)
        self.assertEqual(
            pipeline_permutation(pp_ranks=(2, 0, 1), stage_ranks=(2,), column=0),
            ((2, 0), (0, 1), (1, 2)),
        )


class TestWaveAuxSlots(CustomTestCase):
    """Eagle3's capture layers sit on different stages, so their slots ride the
    same permute as the activation sum and must stay in their own columns."""

    def test_every_stage_slot_reaches_the_last_stage(self):
        size, num_aux = 3, 2
        schedule = _drained(size, [True])
        ici = _FakeIci(size)
        waves = [_wave(r, size, ici, num_aux=num_aux) for r in range(size)]
        arrived = [None] * size
        errors = []

        def run(rank):
            wave = waves[rank]
            try:
                for has_batch in schedule:
                    wave.begin_iteration(has_batch)
                    tensors = None
                    if has_batch:
                        incoming = wave.receive()
                        aux = (
                            torch.zeros((ROWS, HIDDEN * num_aux))
                            if rank == 0
                            else incoming[AUX_HIDDEN_STATES_PROXY_KEY].clone()
                        )
                        if rank < num_aux:
                            aux[:, rank * HIDDEN : (rank + 1) * HIDDEN] = float(
                                rank + 1
                            )
                        arrived[rank] = aux
                        if rank < size - 1:
                            hidden = torch.full((ROWS, HIDDEN), 7.0)
                            tensors = {
                                "hidden_states": hidden,
                                "residual": torch.zeros_like(hidden),
                                AUX_HIDDEN_STATES_PROXY_KEY: aux,
                            }
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
        slots = torch.split(arrived[size - 1], HIDDEN, dim=-1)
        self.assertEqual([float(slot[0, 0]) for slot in slots], [1.0, 2.0])

    def test_a_payload_of_the_wrong_width_is_rejected(self):
        """The model and the wave derive the slot count separately; a mismatch
        would otherwise land as silently misaligned columns."""
        wave = _wave(0, 3, num_aux=2)
        tensors = {
            "hidden_states": torch.zeros((ROWS, HIDDEN)),
            "residual": torch.zeros((ROWS, HIDDEN)),
            AUX_HIDDEN_STATES_PROXY_KEY: torch.zeros((ROWS, HIDDEN)),
        }
        with self.assertRaisesRegex(ValueError, "carries 8 aux columns, got 4"):
            wave.merge(tensors)

    def test_received_pieces_are_their_own_buffers(self):
        """torch_tpu hands a view of a shared buffer to an executable as the
        buffer's base, so a piece sliced out of one hand-off buffer fails at the
        first forward that reads it."""
        wave = _wave(1, 3, num_aux=2)
        wave.begin_iteration(True)
        got = wave.receive()
        for key, piece in got.items():
            head = piece[: ROWS // 2]
            self.assertTrue(head.is_contiguous(), key)
            self.assertEqual(head.storage_offset(), 0, key)
        self.assertEqual(got[AUX_HIDDEN_STATES_PROXY_KEY].shape, (ROWS, 2 * HIDDEN))

    def test_a_payload_without_the_aux_slots_is_rejected(self):
        wave = _wave(0, 3, num_aux=2)
        tensors = {
            "hidden_states": torch.zeros((ROWS, HIDDEN)),
            "residual": torch.zeros((ROWS, HIDDEN)),
        }
        with self.assertRaisesRegex(ValueError, "hands on exactly"):
            wave.merge(tensors)


if __name__ == "__main__":
    unittest.main()
