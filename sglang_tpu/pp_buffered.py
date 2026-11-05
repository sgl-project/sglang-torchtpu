# SPDX-License-Identifier: Apache-2.0
"""Buffered point-to-point hand-off of activations between pipeline stages.

The alternative to the lockstep wave (``pp_wave``), enabled with
``SGLANG_USE_TPU_PP_BUFFERED_HANDOFF=1``. Each device sends a forward's output
to the device that holds the same token rows in the next stage, its column,
with a process-group send, and that device takes it with a receive. A
transfer involves only the two devices at its ends and runs no program on
either: the receiver posts a landing buffer, and once the sender's data is
ready the runtime copies it there. So a stage waits only for its own input,
never for the other stages to reach the same point; there are no zero
launches to drain the pipeline and a small forward is not held behind a
large one on another stage. The scheduler hands a forward's output on right
after it launches the forward, not at the end of its iteration as the wave
must. Port of vllm-project/vllm-torchtpu#2527 ("buffered pipeline hand-off")
to sglang's per-stage scheduler loops.

What travels is what the wave carries: the hidden states plus any residual
in the transfer dtype, plus the aux capture slots of Eagle3 or DFlash/DSpark
in their own dtype. The next stage starts from that sum and a zero residual,
one kept per padded row count.

Every stage runs the same micro-batches in the same order, so a count of
transfers kept by each process numbers them alike on every stage: the
self-test is transfer 1 and each forward takes the next number, which is the
tag it travels with. All tensors of a transfer go in one send and one
receive call. An iteration without a forward transfers nothing.

A receive names the exact shape it takes: the padded token count of the
forward (``pp_scheduler.pp_forward_rows``), which every stage derives alike
from the batch. The posted tensors are fresh landing buffers; the host does
not wait for the data, the forward reads the tensors and the runtime runs the
forward once the data has landed. Every tensor travels as [n, 128]
(``_wire_shape``): the receive lands row-major, and only that shape is laid
out row-major on both sides whatever XLA picks for the sender's output.

A device has 8 receive flags, and a receive posted beyond them stalls its
sender without an error. A receive is outstanding until its data lands,
which is known once the forward that read it has run: the stage records a
device fence after each forward and, before posting, waits on the oldest
fences until at most ``_MAX_RECEIVES`` receive calls are outstanding. The
scheduler loop can run several micro-batches ahead of the device, so this
bound is not implied by the loop.

The sender hands its output on from a helper thread. ``end_iteration``
submits the ops that produce the outgoing tensors and queues the transfer;
the helper waits until the tensors exist on the device (a send reads the
device buffer as it stands), sends them and waits until the copies are done,
keeping the tensors alive until then. A send returns only once the receiver
has posted the matching receive. At most ``depth`` transfers wait in a
stage's queue, the one being sent included; ``end_iteration`` blocks until
there is room, which bounds the memory a stage that runs ahead holds for its
outputs.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.aux_hidden_states import AUX_HIDDEN_STATES_PROXY_KEY

logger = logging.getLogger(__name__)

HIDDEN_STATES = "hidden_states"
RESIDUAL = "residual"

# Tags are C ints; transfer numbers wrap well before they would overflow one.
_TAG_SPAN = 1 << 30

# Receive calls a device may have posted whose data has not landed. A device
# has 8 receive flags, and a receive posted beyond them stalls its sender
# without an error; the hand-off keeps one spare.
_MAX_RECEIVES = 7

# Receive posts between two logs of their time on the scheduler thread.
_POST_LOG_EVERY = 500

# Width of the [n, 128] shape a tensor travels in when its own width is not a
# multiple of 128; see PPBufferedHandoff._wire_shape.
_WIRE_WIDTH = 128

_LIBTPU_REQUIREMENT = (
    "SGLANG_USE_TPU_PP_BUFFERED_HANDOFF needs a libtpu whose device transfers land in the "
    "posted receive, as libtpu 0.0.50 and later do with torch_tpu 0.2.0 and later"
)


class Transport:
    """The device primitives the hand-off uses; replaced by a fake in tests."""

    def __init__(self) -> None:
        pass

    def recv_list(self, tensors: list[torch.Tensor], src: int, tag: int) -> Any:
        """Posts the receives of all ``tensors`` in one runtime call: one
        receive flag and one store round trip per transfer, not per tensor."""
        from torch.distributed.distributed_c10d import _get_default_group

        return _get_default_group().recv(tensors, src, tag)

    def send_list(self, tensors: list[torch.Tensor], dst: int, tag: int) -> Any:
        """Sends all ``tensors`` of one transfer in one runtime call."""
        from torch.distributed.distributed_c10d import _get_default_group

        return _get_default_group().send(tensors, dst, tag)

    def landing_buffer(
        self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """A buffer to receive into. The receive lays it out row-major whatever
        it is, so a plain allocation; a program-made buffer posted before its
        program ran crashed the sender's isend."""
        return torch.empty(shape, dtype=dtype, device=device)

    def wait_landed(self, tensor: torch.Tensor) -> None:
        """Before the host reads a received tensor. The runtime runs a read
        after the data lands, so there is nothing to wait for here."""

    def flush(self, tensors: Iterable[torch.Tensor]) -> None:
        """Submits the pending ops producing ``tensors`` without blocking."""
        from torch_tpu._internal.sync.sync import synchronize

        synchronize(list(tensors), wait=False)

    def flush_all(self) -> None:
        """Submits every pending op without blocking."""
        from torch_tpu._internal.sync.sync import synchronize

        synchronize(None, wait=False)

    def materialize(self, tensors: Iterable[torch.Tensor]) -> None:
        """Blocks until ``tensors`` exist on the device. isend reads the
        tensor's device buffer as it stands: a non-blocking flush only submits
        the producing ops to torch_tpu's materialization worker, and a send
        issued before the worker has produced them reads a buffer that is
        then replaced ("CopyToRemoteDevice() called on deleted or donated
        buffer", or a segfault in ProcessGroupTpu::CrossHostSendBuffers)."""
        from torch_tpu._internal.sync.sync import synchronize

        synchronize(list(tensors), wait=True)

    def fence(self) -> Any:
        """A device event for all work submitted so far."""
        from torch_tpu._internal.stream.streams import TpuEvent

        event = TpuEvent()
        event.record()
        return event


class PPBufferedHandoff:
    """Per-stage state of the point-to-point hand-off; the scheduler drives it
    like ``PPWave``: ``begin_iteration``, then ``receive(rows)`` before a
    forward, then ``end_iteration`` after it."""

    KEYS = (HIDDEN_STATES, RESIDUAL)
    # The scheduler hands a forward's activations on right after it launches
    # the forward (scheduler_pp_mixin._pp_hands_off_after_launch).
    hands_off_after_launch = True

    def __init__(
        self,
        *,
        rank: int,
        size: int,
        pp_ranks: tuple[int, ...],
        hidden: int,
        num_aux: int,
        dtype: torch.dtype,
        device: torch.device,
        aux_dtype: torch.dtype | None = None,
        max_rows: int,
        depth: int = 2,
        rows_of: Callable[[Any], int] | None = None,
        transport: Transport | None = None,
    ) -> None:
        if size < 2:
            raise ValueError("the buffered hand-off needs at least two stages")
        if depth < 1:
            raise ValueError(
                f"the buffered hand-off needs a depth of at least 1, got {depth}"
            )
        self.rank = int(rank)
        self.size = int(size)
        self.last = self.size - 1
        # The global ranks of this device's column, by stage.
        self.pp_ranks = tuple(int(r) for r in pp_ranks)
        self.global_rank = self.pp_ranks[self.rank]
        self.prev = self.pp_ranks[self.rank - 1] if self.rank > 0 else None
        self.next = self.pp_ranks[self.rank + 1] if self.rank < self.last else None
        self.hidden = int(hidden)
        self.num_aux = int(num_aux)
        # Widths of the carried tensors: the activation sum, then the aux slots.
        self.widths = (self.hidden,) + (
            (self.hidden * self.num_aux,) if self.num_aux else ()
        )
        self.dtype = dtype
        # The aux capture slots stay in the compute dtype (PPAuxWave).
        self.aux_dtype = aux_dtype if aux_dtype is not None else dtype
        # Dtypes of the carried tensors, in the order of widths.
        self.dtypes = (self.dtype,) + ((self.aux_dtype,) if self.num_aux else ())
        self.device = device
        self.max_rows = int(max_rows)
        self.depth = int(depth)
        # The padded token count of a batch's forward (pp_scheduler).
        self.rows_of = rows_of
        self._transport = transport or Transport()
        # Transfers numbered alike on every stage: 1 is the self-test.
        self.transfers = 0
        self.forwards = 0
        self.sent = 0
        self._has_batch = False
        # The transfer of the forward between receive() and end_iteration().
        self._current = 0
        self._received_this_iteration = False
        # Fences after forwards that read a receive, oldest first:
        # (transfer, fence); a receive has landed once its fence has passed.
        self._landing: deque[tuple[int, Any]] = deque()
        # Zero residuals by padded row count, each its own storage: a view of
        # one larger tensor would be copied out before every forward.
        self._zero_residuals: dict[int, torch.Tensor] = {}
        # Time posting receives holds the scheduler thread (_note_post).
        self._post_count = 0
        self._post_total_s = 0.0
        self._post_max_s = 0.0
        self._ones: dict[torch.dtype, torch.Tensor] = {}
        # Sends queued for the helper thread, the one in progress first.
        self._cv = threading.Condition()
        self._sends: deque[tuple[int, tuple[torch.Tensor, ...]]] = deque()
        self._closing = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _tag(transfer: int) -> int:
        return transfer % _TAG_SPAN

    def _check_rows(self, transfer: int, rows: int) -> None:
        if not 0 < rows <= self.max_rows:
            raise RuntimeError(
                f"buffered hand-off stage {self.rank}: transfer {transfer} of {rows} "
                f"rows, the hand-off holds 1 to {self.max_rows}"
            )

    def _raise_send_failure(self) -> None:
        with self._cv:
            if self._error is not None:
                raise RuntimeError(
                    f"buffered hand-off stage {self.rank}: the send thread failed"
                ) from self._error

    def _wait_for_receive_flags(self) -> None:
        """Waits until posting one more transfer keeps at most
        _MAX_RECEIVES receive calls outstanding (one per transfer: its
        tensors are posted together)."""
        while self._landing and len(self._landing) + 1 > _MAX_RECEIVES:
            _, fence = self._landing.popleft()
            with torch.profiler.record_function("pp_buffered_wait_receive_flags"):
                fence.synchronize()

    def _one(self, dtype: torch.dtype) -> torch.Tensor:
        """A device scalar one of ``dtype``: multiplying by it makes a tensor a
        program's output."""
        one = self._ones.get(dtype)
        if one is None:
            one = torch.ones((), dtype=dtype, device=self.device)
            self._ones[dtype] = one
        return one

    @staticmethod
    def _wire_shape(rows: int, width: int) -> tuple[int, int]:
        """The shape a [rows, width] tensor travels in.

        A transfer is a byte copy between the sender's buffer and a buffer the
        receive allocates row-major. XLA lays a program output out by its shape
        (GPT-OSS: bf16[512, 2880] tiles column-major, 2949120 bytes, against
        3014656 row-major), and a layout that differs at the same byte size
        would scramble the copy silently. So every tensor travels as
        [rows * width / 128, 128], which both sides lay out row-major without
        padding, aligned widths included; a tensor whose size is not a multiple
        of 128 travels as it is.
        """
        if rows * width % _WIRE_WIDTH:
            return rows, width
        return rows * width // _WIRE_WIDTH, _WIRE_WIDTH

    def _to_wire(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` in its wire shape, as a tensor of its own."""
        shape = self._wire_shape(int(x.shape[0]), int(x.shape[1]))
        if tuple(x.shape) == shape:
            return x
        # A reshape alone is a view, which is handed on as its base buffer; a
        # clone() of it is not safe to send either (its device buffer can be
        # deleted under isend: "CopyToRemoteDevice() called on deleted or
        # donated buffer", or a segfault), while a program's output is.
        return x.reshape(shape) * self._one(x.dtype)

    def _from_wire(self, x: torch.Tensor, rows: int, width: int) -> torch.Tensor:
        """A received tensor back in [rows, width], as a program's output: the
        layout the compiled forward takes."""
        if tuple(x.shape) == (rows, width):
            return x
        return x.reshape(rows, width) * self._one(x.dtype)

    def _post(self, transfer: int, rows: int) -> tuple[torch.Tensor, ...]:
        """Posts the receives of one transfer of ``rows`` rows and returns the
        tensors they land in, in their wire shapes."""
        self._check_rows(transfer, rows)
        started = time.perf_counter()
        received = [
            self._transport.landing_buffer(
                self._wire_shape(rows, width), dtype, self.device
            )
            for width, dtype in zip(self.widths, self.dtypes)
        ]
        # The work does not wait for the data; the forward that reads the
        # tensors runs after they land.
        self._transport.recv_list(received, src=self.prev, tag=self._tag(transfer))
        self._note_post(time.perf_counter() - started)
        return tuple(received)

    def _note_post(self, seconds: float) -> None:
        """Track how long posting a receive holds the scheduler thread (the
        allocation and a store round trip), logged every _POST_LOG_EVERY."""
        self._post_count += 1
        self._post_total_s += seconds
        self._post_max_s = max(self._post_max_s, seconds)
        if self._post_count % _POST_LOG_EVERY == 0:
            logger.info(
                "buffered hand-off stage %d: %d receives posted, %.3f ms mean, "
                "%.3f ms max on the scheduler thread",
                self.rank,
                self._post_count,
                1000 * self._post_total_s / self._post_count,
                1000 * self._post_max_s,
            )

    def _zero_residual(self, rows: int) -> torch.Tensor:
        zero = self._zero_residuals.get(rows)
        if zero is None:
            zero = torch.zeros(
                (rows, self.hidden), dtype=self.dtype, device=self.device
            )
            self._zero_residuals[rows] = zero
        return zero

    # -- protocol -------------------------------------------------------------
    def begin_iteration(self, has_batch: bool) -> None:
        """Called once per scheduler iteration, before its forward if any."""
        self._raise_send_failure()
        self._has_batch = bool(has_batch)
        self._received_this_iteration = False
        if has_batch:
            self.transfers += 1
            self.forwards += 1
            self._current = self.transfers

    def receive(self, rows: int) -> dict[str, torch.Tensor] | None:
        """Before a forward of ``rows`` padded tokens: post its receives and
        return the incoming activations; None on stage 0."""
        if not self._has_batch:
            raise RuntimeError(
                f"buffered hand-off stage {self.rank}: receive() outside an "
                "iteration with a batch"
            )
        if self.rank == 0:
            return None
        self._wait_for_receive_flags()
        with torch.profiler.record_function("pp_buffered_post"):
            received = self._post(self._current, rows)
        self._received_this_iteration = True
        for x in received:
            self._transport.wait_landed(x)
        return self.unpack(
            tuple(
                self._from_wire(x, rows, width)
                for x, width in zip(received, self.widths)
            ),
            rows,
        )

    def receive_batch(self, batch: Any) -> dict[str, torch.Tensor] | None:
        """receive() at the padded token count of ``batch``'s forward."""
        if self.rank == 0:
            return self.receive(0)
        assert self.rows_of is not None, (
            "the buffered hand-off needs rows_of to size receives"
        )
        return self.receive(self.rows_of(batch))

    def unpack(
        self, received: tuple[torch.Tensor, ...], rows: int
    ) -> dict[str, torch.Tensor]:
        """The forward's inputs: the sum arrives as the hidden states, the
        residual is zero."""
        tensors = {HIDDEN_STATES: received[0], RESIDUAL: self._zero_residual(rows)}
        if self.num_aux:
            tensors[AUX_HIDDEN_STATES_PROXY_KEY] = received[1]
        return tensors

    def merge(
        self, tensors: dict[str, torch.Tensor | None]
    ) -> tuple[torch.Tensor, ...]:
        """The outgoing tensors of a forward: its hidden states plus the
        residual when it returns one, then the aux slots; each its own
        storage."""
        aux = {AUX_HIDDEN_STATES_PROXY_KEY} if self.num_aux else set()
        if set(tensors) not in ({HIDDEN_STATES} | aux, set(self.KEYS) | aux):
            raise ValueError(
                f"the buffered hand-off requires {sorted({HIDDEN_STATES} | aux)} "
                f"and an optional {RESIDUAL}, got {sorted(tensors)}"
            )
        total = tensors[HIDDEN_STATES]
        if tensors.get(RESIDUAL) is not None:
            total = total + tensors[RESIDUAL]
        if total.dtype != self.dtype:
            raise ValueError(
                f"the buffered hand-off carries {self.dtype}, got {total.dtype}"
            )
        outgoing = [total]
        if self.num_aux:
            aux = tensors[AUX_HIDDEN_STATES_PROXY_KEY]
            if int(aux.shape[-1]) != self.widths[1]:
                raise ValueError(
                    f"the buffered hand-off carries {self.widths[1]} aux columns, "
                    f"got {int(aux.shape[-1])}"
                )
            if aux.dtype != self.aux_dtype:
                raise ValueError(
                    f"the buffered hand-off carries {self.aux_dtype} aux activations, "
                    f"got {aux.dtype}"
                )
            outgoing.append(aux)
        # Sending a view would make its copy on the send thread; copy here,
        # on the thread that traces the forward.
        # (A program's output, not clone(): see _to_wire.)
        return tuple(x if x._base is None else x * self._one(x.dtype) for x in outgoing)

    def end_iteration(self, tensors: dict[str, torch.Tensor] | None) -> None:
        """After the forward (``tensors`` = its outgoing activations; None
        without a forward or on the last stage): queue them for the next
        stage."""
        self._raise_send_failure()
        if not self._has_batch:
            if tensors is not None:
                raise RuntimeError(
                    f"buffered hand-off stage {self.rank}: outgoing activations "
                    "from an iteration without a batch"
                )
            return
        if self._received_this_iteration:
            # The forward that read the receive must be submitted before the
            # fence that tells when its data has landed.
            self._transport.flush_all()
            self._landing.append((self._current, self._transport.fence()))
        if self.rank == self.last:
            return
        if tensors is None:
            raise RuntimeError(
                f"buffered hand-off stage {self.rank}: forward {self._current} "
                "produced nothing to hand on"
            )
        outgoing = self.merge(tensors)
        rows = int(outgoing[0].shape[0])
        self._check_rows(self._current, rows)
        if any(int(x.shape[0]) != rows for x in outgoing):
            raise RuntimeError(
                f"buffered hand-off stage {self.rank}: forward {self._current} hands "
                f"on tensors of different row counts {[tuple(x.shape) for x in outgoing]}"
            )
        outgoing = tuple(self._to_wire(x) for x in outgoing)
        self._transport.flush(outgoing)
        self._queue_send(self._current, outgoing)

    # -- sending --------------------------------------------------------------
    def _queue_send(self, transfer: int, outgoing: tuple[torch.Tensor, ...]) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._send_loop, name="pp-buffered-send", daemon=True
            )
            self._thread.start()
        with self._cv:
            if len(self._sends) >= self.depth and self._error is None:
                with torch.profiler.record_function("pp_buffered_full"):
                    self._cv.wait_for(
                        lambda: len(self._sends) < self.depth or self._error is not None
                    )
            if self._error is not None:
                raise RuntimeError(
                    f"buffered hand-off stage {self.rank}: the send thread failed"
                ) from self._error
            self._sends.append((transfer, outgoing))
            self._cv.notify_all()

    def _send(self, transfer: int, outgoing: tuple[torch.Tensor, ...]) -> None:
        """Sends one transfer once its tensors exist on the device and returns
        when the copies are done; runs on the send thread."""
        self._transport.materialize(outgoing)
        self._transport.send_list(
            list(outgoing), dst=self.next, tag=self._tag(transfer)
        ).wait()

    def _send_loop(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: bool(self._sends) or self._closing)
                if not self._sends:
                    return
                # Stays queued, and counts against the depth, until sent.
                transfer, outgoing = self._sends[0]
            error: BaseException | None = None
            try:
                self._send(transfer, outgoing)
            except BaseException as exc:  # re-raised on the scheduler thread
                logger.exception(
                    "buffered hand-off stage %d: sending transfer %d failed",
                    self.rank,
                    transfer,
                )
                error = exc
            with self._cv:
                if error is not None:
                    self._error = error
                    self._sends.clear()
                    self._cv.notify_all()
                    return
                # Releases the tensors: the copies are done.
                self._sends.popleft()
                self.sent += 1
                self._cv.notify_all()

    def close(self) -> None:
        """Waits until every queued send is done, stops the send thread and
        raises its failure."""
        if self._thread is not None:
            with self._cv:
                self._closing = True
                self._cv.notify_all()
            self._thread.join()
            self._thread = None
            self._closing = False
        self._raise_send_failure()

    # -- warmup and self-test -------------------------------------------------
    def warmup_merge(self, row_buckets: Iterable[int]) -> None:
        """Compile, for every bucket a forward hands on, the programs a
        transfer runs around the forward: the outgoing sum and its wire copy
        on a sending stage, and the copy back from the wire on a receiving
        one. Otherwise each compiles at the first request of its bucket."""
        for rows in row_buckets:
            if rows > self.max_rows:
                continue
            programs: list[torch.Tensor] = []
            if self.rank != self.last:
                zeros = {
                    key: torch.zeros(
                        (rows, self.hidden), dtype=self.dtype, device=self.device
                    )
                    for key in self.KEYS
                }
                if self.num_aux:
                    zeros[AUX_HIDDEN_STATES_PROXY_KEY] = torch.zeros(
                        (rows, self.widths[1]), dtype=self.aux_dtype, device=self.device
                    )
                programs.extend(self._to_wire(x) for x in self.merge(zeros))
            if self.rank > 0:
                for width, dtype in zip(self.widths, self.dtypes):
                    wire = self._transport.landing_buffer(
                        self._wire_shape(rows, width), dtype, self.device
                    )
                    programs.append(self._from_wire(wire, rows, width))
            self._transport.flush(programs)

    def _probe_pattern(
        self, global_rank: int, rows: int, width: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """What the self-test sends from ``global_rank``: its rank plus one,
        plus sixteenths that vary along both axes, so a transposed or padded
        layout does not compare equal. Exact in bfloat16."""
        r = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
        c = torch.arange(width, dtype=torch.float32).unsqueeze(0)
        pattern = float(global_rank + 1) + torch.remainder(3 * r + c, 16) / 16
        return pattern.to(dtype)

    def probe(self) -> bool:
        """One transfer of every carried tensor from each device to its column
        in the next stage, at a real bucket size and a pattern that varies by
        row and column; True when all of what arrived is what the previous
        stage's device of this column sent (stage 0 receives nothing)."""
        assert self.transfers == 0
        self.transfers = 1
        rows = min(512, self.max_rows)
        received: tuple[torch.Tensor, ...] = ()
        try:
            # Every stage posts before it sends, and a send waits only for the
            # next stage's post, so the chain completes from the last stage.
            if self.prev is not None:
                received = self._post(self.transfers, rows)
            if self.next is not None:
                # Made on the device by a program, as a forward's output is.
                sent = tuple(
                    self._to_wire(
                        self._probe_pattern(self.global_rank, rows, width, dtype).to(
                            self.device
                        )
                        * self._one(dtype)
                    )
                    for width, dtype in zip(self.widths, self.dtypes)
                )
                self._transport.materialize(sent)
                self._transport.send_list(
                    list(sent), dst=self.next, tag=self._tag(self.transfers)
                ).wait()
            ok = True
            for x, width, dtype in zip(received, self.widths, self.dtypes):
                self._transport.wait_landed(x)
                want = self._probe_pattern(self.prev, rows, width, dtype)
                got = self._from_wire(x, rows, width).cpu()
                ok = ok and bool(torch.equal(got, want))
        except Exception as exc:
            raise RuntimeError(
                f"buffered hand-off stage {self.rank}: the hand-off self-test failed. "
                f"{_LIBTPU_REQUIREMENT}."
            ) from exc
        return ok


def build_pp_buffered_handoff(
    pp_group: GroupCoordinator,
    *,
    world_group: GroupCoordinator,
    max_rows: int,
    hidden: int,
    num_aux: int = 0,
    dtype: torch.dtype,
    aux_dtype: torch.dtype | None = None,
    device: torch.device,
    depth: int,
    rows_of: Callable[[Any], int],
) -> PPBufferedHandoff:
    """Create the stage's buffered hand-off on the device and verify the routing."""
    rank = pp_group.rank_in_group
    size = pp_group.world_size
    handoff = PPBufferedHandoff(
        rank=rank,
        size=size,
        pp_ranks=tuple(pp_group.ranks),
        hidden=hidden,
        num_aux=num_aux,
        dtype=dtype,
        aux_dtype=aux_dtype,
        device=device,
        max_rows=max_rows,
        depth=depth,
        rows_of=rows_of,
    )
    ok = handoff.probe()
    verdicts: list = [None] * world_group.world_size
    torch.distributed.all_gather_object(verdicts, bool(ok), group=world_group.cpu_group)
    if not all(verdicts):
        raise RuntimeError(
            f"buffered hand-off stage {rank}: the hand-off self-test failed (rank "
            f"verdicts {verdicts}, column ranks {tuple(pp_group.ranks)}). "
            f"{_LIBTPU_REQUIREMENT}."
        )
    logger.info(
        "buffered hand-off stage %d/%d: point-to-point hand-off from %s to %s, "
        "depth %d, carries widths %s in %s, up to %d rows",
        rank,
        size,
        handoff.prev,
        handoff.next,
        depth,
        handoff.widths,
        handoff.dtypes,
        max_rows,
    )
    return handoff
