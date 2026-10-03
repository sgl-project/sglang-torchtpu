# SPDX-License-Identifier: Apache-2.0
"""Pipeline-stage hand-off over ICI as a lockstep wave.

torch_tpu has no device send/recv, and a collective over any subset of the
slice's devices only runs when every device launches it in the same order.
So every hand-off is one collective-permute over all pipeline stages at once,
around closed cycles: stage r hands the sum of its hidden states and
residual to stage r+1 and receives stage r-1's, and the last stage's block
lands on stage 0, which ignores it. A stage of several devices (a prefill
context-parallel group) forms one cycle per device column: each device hands
its rows to the device holding the same rows in the next stage, and all
columns move in the same launch. The stage after a boundary must add
hidden states and residual before its first norm, as a pre-norm decoder's
fused add-norm does, so the sum with a zero residual carries the same
residual stream at half the bytes. That norm reads the sum rounded to the
model dtype instead of the float32 sum, a small difference at every stage
boundary. The closed cycles let XLA lower the permute to full-bandwidth
ring transfers. A batch's activations
leave stage 0 at some launch k and reach stage r at launch k+r-1, so every
stage issues launches by one count:

- Stage 0 launches once per iteration while the wave is active: after its
  forward, carrying the batch's activations, or a zero launch while it still
  owes launches after its last send. It owes 2 * (stages - 2): the last batch
  needs stages-2 more hops to reach the last stage, and the trailing stages,
  which run up to stages-2 iterations behind stage 0, need as many again so
  that every launch they issued ahead of stage 0's count is matched before
  stage 0 goes quiet.
- Stage r>0 launches at the start of an iteration until its count is
  stage 0's count plus min(r-1, launches stage 0 still owes): the last launch
  of an iteration with a batch receives that batch, and the first launch after
  its forward carries the forward's output.

Stage 0 launches one hop per iteration, never more, because the schedulers'
hosts are coupled: the last stage's outputs travel last -> 0 -> 1 -> ... one
hop per iteration over the host channel, and every stage waits on its device
when it reads a result (the decode allocation, the forward's metadata, the
ring's outputs) while the device runs launches in order. A stage that
launched further ahead than the trailing stages can issue before its next
device wait would block for good.

Every launch runs the same compiled program: one buffer padded to the largest
batch a forward can hand on.

Stage-local launch slots. torch_tpu compiles every program holding a
collective, a stage group's included, for the whole slice's device
assignment, and a device starts its i-th such program only once every device
of the slice has started its i-th, whichever program that is. A hand-off must
therefore sit at the same launch index on every device. With ``slot_size``
set, every device issues exactly ``slot_size`` stage-local multi-device
programs between consecutive hand-offs: a stage reports the programs its
forward launched (``note_stage_launches``) and the wave tops the slot up with
a filler program over the stage group (a tiny all-reduce) right before the
next hand-off. Stage-local programs of different stages may differ in content
and order; only the count must match. Every forward is followed by at least
one hand-off of its wave (the drain), so the counts are level when the wave
goes quiet. Single-device programs do not count.
"""

import logging
import os
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

_OP_NAME = "sglang_tpu::pp_permute"
_OPS: dict[tuple[int, ...], Any] = {}


def physical_device_by_rank(world_group: Any) -> tuple[int, ...]:
    """Physical TPU device id of every rank of the slice, gathered over the
    world group's host channel."""
    from torch_tpu._internal.distributed import tpu_distributed

    ids: list = [None] * int(world_group.world_size)
    torch.distributed.all_gather_object(
        ids, int(tpu_distributed.global_device_id()), group=world_group.cpu_group
    )
    return tuple(int(x) for x in ids)


def pipeline_permutation(
    *, physical: tuple[int, ...], size: int, columns: int
) -> tuple[tuple[int, int], ...]:
    """Source/target physical device pairs of the hand-off: rank
    ``s * columns + j`` (stage s, column j) sends to stage s+1's column j,
    the last stage back to stage 0, one closed cycle per column."""
    if len(physical) != size * columns:
        raise ValueError(
            f"{len(physical)} ranks cannot form {size} stages of {columns} devices"
        )
    return tuple(
        (physical[s * columns + j], physical[((s + 1) % size) * columns + j])
        for j in range(columns)
        for s in range(size)
    )


def build_pp_permute_op(perm: tuple[tuple[int, int], ...]) -> Any:
    """Torch op ``x_out = pp_permute(x)``: one collective-permute over the
    slice's devices moving a [rows, hidden] block along ``perm`` (mesh
    positions), compiled once per input shape."""
    key = tuple(x for pair in perm for x in pair)
    op = _OPS.get(key)
    if op is not None:
        return op
    import jax
    import numpy as np
    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh, PartitionSpec
    from torch_tpu._internal.pallas import jax_op

    from sglang.srt.hardware_backend.tpu.pallas_export import (
        get_torchtpu_export_devices,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    mesh = Mesh(np.asarray(get_torchtpu_export_devices(world_size)), ("pp",))
    spec = PartitionSpec()

    def shift(x: jax.Array) -> jax.Array:
        # Each device passes its own block as a "replicated" array, so the
        # exported program sees this rank's shape and returns this rank's
        # received block at the same shape.
        return shard_map(
            lambda xb: jax.lax.ppermute(xb, "pp", list(perm)),
            mesh=mesh,
            in_specs=(spec,),
            out_specs=spec,
            check_rep=False,
        )(x)

    op = jax_op(f"{_OP_NAME}_{len(_OPS)}", shift)
    _OPS[key] = op
    return op


def launch_now(op: Any, x: torch.Tensor) -> torch.Tensor:
    """Run one hand-off as its own device program, submitted immediately.

    Pending lazy work is flushed first so the launch order across stages is
    the order the scheduler issues it; neither flush waits.
    """
    from torch_tpu._internal.sync.sync import synchronize

    synchronize(None, wait=False)
    out = op(x)
    synchronize([out], wait=False)
    return out


def make_stage_filler(stage_group: Any, device: torch.device) -> Callable[[], Any]:
    """A filler program: a tiny all-reduce over the stage group, flushed on
    both sides so it is one multi-device program of its own. Returns its
    output, whose readiness tells when the program ran."""
    from torch_tpu._internal.sync.sync import synchronize

    stage_pg = stage_group.device_group

    def filler() -> torch.Tensor:
        synchronize(None, wait=False)
        x = torch.zeros((8, 128), dtype=torch.float32, device=device)
        torch.distributed.all_reduce(x, group=stage_pg)
        synchronize([x], wait=False)
        return x

    return filler


def ready_probe(x: torch.Tensor) -> Callable[[], bool]:
    """A non-blocking check that the program producing ``x`` has run.

    ``sync.is_ready`` where torch_tpu has it; otherwise a daemon thread
    blocks reading ``x`` back and the check reports whether the read
    returned. (The experimental ``TpuEvent.query`` crashed on the torch_tpu
    this was validated on.) The read always returns once the slice is
    levelled, so the thread does not outlive the leveling.
    """
    try:
        from torch_tpu._internal.sync.sync import is_ready
    except ImportError:
        import threading

        done = threading.Event()

        def read() -> None:
            x.cpu()
            done.set()

        threading.Thread(target=read, daemon=True, name="pp-ready-probe").start()
        return done.is_set
    return lambda: bool(is_ready(x))


def level_stage_launches(
    *,
    stage_group: Any,
    world_group: Any,
    filler: Callable[[], Any],
    pp_rank: int,
    pp_size: int,
    poll_s: float = 0.005,
    report_s: float = 60.0,
) -> int:
    """Bring every device of the slice to the same multi-device launch count
    without knowing the counts; returns the fillers this stage issued.

    A device's latest launch completes only once every device has issued at
    least as many, so the stages whose latest launch is ready are exactly
    the stages at the lowest count (a stage still computing reads as not
    ready and simply waits a round). Each round every rank shares its
    readiness over the host channel and the lowest stages add one filler,
    until every stage is ready. A stage acts only when all its devices are
    ready, so its devices issue the same fillers. Must be called at the same
    point on every rank, with no device wait pending.
    """
    import time

    columns = int(stage_group.world_size)
    # Uniform: every stage's latest launch is now its own.
    ready = ready_probe(filler())
    fillers = 0
    start = time.monotonic()
    next_report = start + report_s
    while True:
        flags = world_group.all_gather_object(ready())
        stage_ready = [
            all(flags[s * columns : (s + 1) * columns]) for s in range(pp_size)
        ]
        if all(stage_ready):
            return fillers
        if stage_ready[pp_rank]:
            ready = ready_probe(filler())
            fillers += 1
        # Let fresh launches finish so the next round's readiness is not
        # stale; a stale "not ready" only costs extra (harmless) fillers.
        time.sleep(poll_s)
        if time.monotonic() > next_report:
            next_report += report_s
            logger.warning(
                "pipeline launch leveling stage %d: %d fillers after %.0fs, "
                "stage readiness %s",
                pp_rank,
                fillers,
                time.monotonic() - start,
                stage_ready,
            )


class PPWave:
    """Per-stage bookkeeping of the hand-off wave; see the module docstring.

    ``launcher(x)`` performs one launch and returns the received block.
    Per scheduler iteration: ``begin_iteration``, then ``receive`` before the
    forward when the iteration carries a batch, then ``end_iteration``.
    """

    KEYS = ("hidden_states", "residual")

    def __init__(
        self,
        *,
        rank: int,
        size: int,
        rows: int,
        hidden: int,
        dtype: torch.dtype,
        device: torch.device,
        launcher: Callable[[torch.Tensor], torch.Tensor],
        slot_size: int = 0,
        slot_filler: Callable[[], None] | None = None,
    ) -> None:
        if size < 2:
            raise ValueError("the pipeline wave needs at least two stages")
        if slot_size < 0 or (slot_size > 0 and slot_filler is None):
            raise ValueError(
                f"a launch slot of {slot_size} needs a filler program"
            )
        self.rank = int(rank)
        self.size = int(size)
        self.last = self.size - 1
        self.rows = int(rows)
        self.hidden = int(hidden)
        self.dtype = dtype
        self._launcher = launcher
        self.zero = torch.zeros((self.rows, self.hidden), dtype=dtype, device=device)
        # Output of this stage's last forward, sent at its next launch.
        self.pending: torch.Tensor | None = None
        self.launches = 0
        # Stage 0's launch count through the current iteration, and the zero
        # launches it still owes after its last send; mirrored on every stage.
        self.stage0_launches = 0
        self.owed = 0
        self.forwards = 0
        self.runs = 0
        self._has_batch = False
        # Stage-local programs since the last hand-off (module docstring).
        self.slot_size = int(slot_size)
        self._slot_filler = slot_filler
        self.slot_used = 0
        self.fillers = 0
        # Stage-local programs one forward of this stage issues (set by the
        # scheduler; counted with note_stage_launches after each forward).
        # ``forward_launch_counter``, when set, reports the forward just run
        # instead (its graph's measured count).
        self.forward_launches = 0
        self.forward_launch_counter: Callable[[], int] | None = None

    # -- launches -------------------------------------------------------------
    def note_forward(self) -> None:
        """Count the stage-local programs of the forward this stage just ran."""
        if self.slot_size == 0:
            return
        counter = self.forward_launch_counter
        self.note_stage_launches(
            counter() if counter is not None else self.forward_launches
        )

    def note_stage_launches(self, n: int) -> None:
        """Count ``n`` stage-local multi-device programs (a forward's) issued
        since the last hand-off."""
        if self.slot_size == 0:
            return
        self.slot_used += int(n)
        if self.slot_used > self.slot_size:
            raise RuntimeError(
                f"pipeline wave stage {self.rank}: {self.slot_used} stage-local "
                f"launches between two hand-offs, above the slot of "
                f"{self.slot_size}; raise SGLANG_TPU_PP_SLOT_LAUNCHES"
            )

    def _fill_slot(self) -> None:
        for _ in range(self.slot_size - self.slot_used):
            self._slot_filler()
            self.fillers += 1
        self.slot_used = 0

    def _launch(self, x: torch.Tensor) -> torch.Tensor:
        self._fill_slot()
        out = self._launcher(x)
        self.launches += 1
        return out

    def _target(self) -> int:
        """The launch count this stage must reach in the current iteration."""
        if self.rank == 0:
            return self.stage0_launches
        return self.stage0_launches + min(self.rank - 1, self.owed)

    def _pad(self, t: torch.Tensor) -> torch.Tensor:
        n = int(t.shape[0])
        if n == self.rows:
            return t
        if n > self.rows:
            raise ValueError(
                f"pipeline hand-off tensor has {n} rows, above the wave's "
                f"{self.rows}-row buffer (the largest compiled token bucket)"
            )
        return torch.cat([t, self.zero[n:]])

    def _merge(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        """The padded sum of a forward's outgoing hidden states and residual."""
        if set(tensors) != set(self.KEYS):
            raise ValueError(
                f"the pipeline wave hands on exactly {self.KEYS}, got {sorted(tensors)}"
            )
        merged = tensors[self.KEYS[0]] + tensors[self.KEYS[1]]
        if merged.dtype != self.dtype:
            raise ValueError(
                f"the pipeline wave carries {self.dtype}, got a {merged.dtype} sum"
            )
        return self._pad(merged)

    # -- protocol -------------------------------------------------------------
    def begin_iteration(self, has_batch: bool) -> None:
        """Called once per scheduler iteration, before any hand-off of it."""
        self._has_batch = bool(has_batch)
        if has_batch:
            self.forwards += 1
            self.stage0_launches += 1
            self.owed = 2 * (self.size - 2)
        elif self.owed > 0:
            self.stage0_launches += 1
            self.owed -= 1
            if self.owed == 0:
                self.runs += 1
                if self.runs <= 3 or self.runs % 500 == 0:
                    logger.info(
                        "pipeline wave stage %d: drain %d complete, %d forwards, "
                        "%d hand-offs and %d slot fillers so far",
                        self.rank,
                        self.runs,
                        self.forwards,
                        self._target(),
                        self.fillers,
                    )
        if self.rank == 0:
            return
        due = self._target() - self.launches
        if self.pending is not None and due < 1:
            raise RuntimeError(
                f"pipeline wave stage {self.rank}: a forward's output is held "
                "with no launch due to carry it"
            )
        # With a batch, the iteration's last launch is the receive().
        for _ in range(due - (1 if has_batch else 0)):
            self._launch(self._take_pending())

    def receive(self) -> dict[str, torch.Tensor] | None:
        """Before a forward: launch (sending the output still held) and return
        the incoming activations at the buffer's row count; None on stage 0.
        The sum arrives as the hidden states; the residual is zero."""
        if self.rank == 0:
            return None
        if not self._has_batch:
            raise RuntimeError(
                f"pipeline wave stage {self.rank}: receive() outside an "
                "iteration with a batch"
            )
        x = self._launch(self._take_pending())
        return {self.KEYS[0]: x, self.KEYS[1]: self.zero}

    def _check_count(self) -> None:
        expected = self._target()
        if self.launches != expected:
            raise RuntimeError(
                f"pipeline wave stage {self.rank}: issued {self.launches} launches, "
                f"expected {expected} at the end of this iteration"
            )

    def end_iteration(self, tensors: dict[str, torch.Tensor] | None) -> None:
        """After the forward (``tensors`` = its outgoing activations; None
        without a batch or on the last stage): stage 0 launches its one hop of
        the iteration, a middle stage holds the output for its next launch."""
        if self.rank == self.last:
            self._check_count()
            return
        padded = None
        if tensors is not None:
            padded = self._merge(tensors)
        if self.rank == 0:
            if self.launches < self._target():
                self._launch(padded if padded is not None else self.zero)
            self._check_count()
            return
        if padded is not None:
            if self.pending is not None:
                raise RuntimeError(
                    f"pipeline wave stage {self.rank}: two forward outputs "
                    "held without a launch between them"
                )
            self.pending = padded
        self._check_count()

    def _take_pending(self) -> torch.Tensor:
        x = self.pending if self.pending is not None else self.zero
        self.pending = None
        return x

    # -- self-test ------------------------------------------------------------
    def probe(self, *, column: int = 0, columns: int = 1) -> bool:
        """One batch iteration and its full drain on every stage, each device
        sending its stage-major index plus one; True when this device received
        the index of the previous stage's device in its own column (stage 0
        receives the last stage's block and ignores it).
        Every stage's launch count is level again afterwards."""
        payload = torch.full_like(self.zero, float(self.rank * columns + column + 1))
        got = None
        for it in range(1 + 2 * (self.size - 2)):
            has_batch = it == 0
            self.begin_iteration(has_batch)
            tensors = None
            if has_batch:
                got = self.receive()
                if self.rank != self.last:
                    tensors = {self.KEYS[0]: payload, self.KEYS[1]: self.zero}
            self.end_iteration(tensors)
        if self.rank == 0:
            return True
        want = float((self.rank - 1) * columns + column + 1)
        x = got[self.KEYS[0]]
        return float(x[0, 0].item()) == want and float(x[-1, -1].item()) == want


def build_pp_wave(
    pp_group: Any,
    *,
    stage_group: Any,
    world_group: Any,
    rows: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    slot_size: int = 0,
) -> PPWave:
    """Create the stage's wave on the device and verify the hand-off routing.

    The permute is routed by mesh position, which follows physical device
    order; each column's cycle closes from the last stage back to the first,
    and every device of the slice launches the same program. ``slot_size``
    stage-local launches separate consecutive hand-offs (module docstring).
    """
    rank = int(pp_group.rank_in_group)
    size = int(pp_group.world_size)
    columns = int(stage_group.world_size)
    column = int(stage_group.rank_in_group)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if size * columns != world_size:
        raise ValueError(
            f"the pipeline wave needs the pipeline stages to cover the whole slice "
            f"(pp {size} stages of {columns} devices, world {world_size})"
        )
    if slot_size > 0 and columns < 2:
        raise ValueError(
            "pipeline launch slots need a stage group of at least two devices"
        )
    physical = physical_device_by_rank(world_group)
    perm = pipeline_permutation(physical=physical, size=size, columns=columns)
    op = build_pp_permute_op(perm)
    filler = make_stage_filler(stage_group, device) if slot_size > 0 else None
    if columns > 1:
        # Start-up (model load, warmup) may leave the stages at different
        # launch counts; the first hand-off must sit at one index everywhere.
        levelled = level_stage_launches(
            stage_group=stage_group,
            world_group=world_group,
            filler=make_stage_filler(stage_group, device),
            pp_rank=rank,
            pp_size=size,
        )
        logger.info(
            "pipeline wave stage %d: %d fillers levelled the start-up launches",
            rank,
            levelled,
        )
    wave = PPWave(
        rank=rank,
        size=size,
        rows=rows,
        hidden=hidden,
        dtype=dtype,
        device=device,
        launcher=lambda x: launch_now(op, x),
        slot_size=slot_size,
        slot_filler=filler,
    )
    ok = wave.probe(column=column, columns=columns)
    # Every device must agree on the routing, so the verdict is gathered.
    verdicts: list = [None] * int(world_group.world_size)
    torch.distributed.all_gather_object(verdicts, bool(ok), group=world_group.cpu_group)
    if not all(verdicts):
        raise RuntimeError(
            f"pipeline wave stage {rank} column {column}: the hand-off probe "
            f"failed (rank verdicts {verdicts}, devices {physical})"
        )
    logger.info(
        "pipeline wave stage %d/%d column %d/%d: hand-off cycles over devices %s, "
        "buffer [%d, %d] %s",
        rank,
        size,
        column,
        columns,
        physical,
        rows,
        hidden,
        dtype,
    )
    return wave
