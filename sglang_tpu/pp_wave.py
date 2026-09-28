# SPDX-License-Identifier: Apache-2.0
"""Pipeline-stage hand-off over ICI as a lockstep wave.

torch_tpu has no device send/recv, so each hand-off is one collective-permute
that every device of the slice launches in the same order: each device sends
the sum of its hidden states and residual to its column's device in the next
stage, one closed cycle per column. Stage 0 launches once per iteration while
the wave is active and owes 2 * (stages - 2) zero launches after its last send;
stage r > 0 launches up to stage 0's count plus min(r - 1, launches owed).

Every launch runs the same compiled program: one buffer padded to the largest
batch a forward can hand on. Eagle3 widens that buffer by one hidden-wide slot
per capture layer, since the capture layers sit on different stages and the
draft needs all of them on the last one.
"""

import logging
import os
from typing import Any, Callable

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.aux_hidden_states import AUX_HIDDEN_STATES_PROXY_KEY
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
)

logger = logging.getLogger(__name__)

_OP_NAME = "sglang_tpu::pp_permute"
_OPS: dict[tuple[int, ...], Any] = {}


def pipeline_permutation(
    *, pp_ranks: tuple[int, ...], stage_ranks: tuple[int, ...], column: int
) -> tuple[tuple[int, int], ...]:
    """Source/target rank pairs of the hand-off: each device sends to its
    column's device in the next stage, the last stage back to stage 0, one
    closed cycle per column. ``pp_ranks`` are the ranks of this device's column
    by stage and ``stage_ranks`` those of its stage by column; every stage
    orders its columns alike, so each device derives the same pairs."""
    size = len(pp_ranks)
    return tuple(
        (pp_ranks[s] + shift, pp_ranks[(s + 1) % size] + shift)
        for shift in (r - stage_ranks[column] for r in stage_ranks)
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
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec
    from torch_tpu._internal.pallas import get_pallas_mesh, jax_op

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    mesh = get_pallas_mesh(("pp",), (world_size,))
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

    op = jax_op(
        f"{_OP_NAME}_{len(_OPS)}",
        shift,
        mesh=mesh,
        input_partition_specs=(spec,),
        output_partition_specs=(spec,),
    )
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
        num_aux: int = 0,
        dtype: torch.dtype,
        device: torch.device,
        launcher: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        if size < 2:
            raise ValueError("the pipeline wave needs at least two stages")
        self.rank = int(rank)
        self.size = int(size)
        self.last = self.size - 1
        self.rows = int(rows)
        self.hidden = int(hidden)
        # Eagle3 capture layers sit on different stages, so their slot buffer
        # rides the same permute: block 0 is the activation sum, then one
        # hidden-wide slot per capture.
        self.num_aux = int(num_aux)
        self.width = self.hidden * (1 + self.num_aux)
        self.dtype = dtype
        self._launcher = launcher
        self.zero = torch.zeros((self.rows, self.width), dtype=dtype, device=device)
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

    # -- launches -------------------------------------------------------------
    def _launch(self, x: torch.Tensor) -> torch.Tensor:
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

    def _expected_keys(self) -> set[str]:
        keys = set(self.KEYS)
        if self.num_aux:
            keys.add(AUX_HIDDEN_STATES_PROXY_KEY)
        return keys

    def _merge(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        """A forward's outgoing activation sum, the aux slots appended, padded."""
        expected = self._expected_keys()
        if set(tensors) != expected:
            raise ValueError(
                f"the pipeline wave hands on exactly {sorted(expected)}, "
                f"got {sorted(tensors)}"
            )
        merged = tensors[self.KEYS[0]] + tensors[self.KEYS[1]]
        if merged.dtype != self.dtype:
            raise ValueError(
                f"the pipeline wave carries {self.dtype}, got a {merged.dtype} sum"
            )
        if self.num_aux:
            merged = torch.cat([merged, tensors[AUX_HIDDEN_STATES_PROXY_KEY]], dim=-1)
        if int(merged.shape[-1]) != self.width:
            raise ValueError(
                f"the pipeline wave carries {self.width} columns, "
                f"got {int(merged.shape[-1])}"
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
                        "pipeline wave stage %d: drain %d complete, %d forwards "
                        "and %d launches so far",
                        self.rank,
                        self.runs,
                        self.forwards,
                        self._target(),
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
        if not self.num_aux:
            return {self.KEYS[0]: x, self.KEYS[1]: self.zero}
        return {
            self.KEYS[0]: x[:, : self.hidden],
            self.KEYS[1]: self.zero[:, : self.hidden],
            AUX_HIDDEN_STATES_PROXY_KEY: x[:, self.hidden :],
        }

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
        sent = {
            self.KEYS[0]: payload[:, : self.hidden],
            self.KEYS[1]: self.zero[:, : self.hidden],
        }
        if self.num_aux:
            sent[AUX_HIDDEN_STATES_PROXY_KEY] = payload[:, self.hidden :]
        got = None
        for it in range(1 + 2 * (self.size - 2)):
            has_batch = it == 0
            self.begin_iteration(has_batch)
            tensors = None
            if has_batch:
                got = self.receive()
                if self.rank != self.last:
                    tensors = sent
            self.end_iteration(tensors)
        if self.rank == 0:
            return True
        want = float((self.rank - 1) * columns + column + 1)
        blocks = [got[self.KEYS[0]]]
        if self.num_aux:
            blocks.append(got[AUX_HIDDEN_STATES_PROXY_KEY])
        return all(
            float(x[0, 0].item()) == want and float(x[-1, -1].item()) == want
            for x in blocks
        )


def num_relayed_aux_slots(config: SpecAuxHiddenStateConfig) -> int:
    """Eagle3 capture slots a pipeline stage hand-off carries; 0 when none.

    The capture layers sit on different stages, so each one occupies a
    hidden-wide slot that rides the hand-off to the stage holding the draft.
    """
    if not config.eagle_use_aux_hidden_state:
        return 0
    layer_ids = config.eagle_aux_hidden_state_layer_ids
    if layer_ids is None:
        raise NotImplementedError(
            "Pipeline parallelism sizes its stage hand-off from the draft's "
            "eagle_aux_hidden_state_layer_ids, which this draft does not declare."
        )
    return len(layer_ids)


def build_pp_wave(
    pp_group: GroupCoordinator,
    *,
    stage_group: GroupCoordinator,
    world_group: GroupCoordinator,
    rows: int,
    hidden: int,
    num_aux: int = 0,
    dtype: torch.dtype,
    device: torch.device,
) -> PPWave:
    """Create the stage's wave on the device and verify the hand-off routing.

    The permute is routed by mesh position, which follows distributed rank
    order; each column's cycle closes from the last stage back to the first,
    and every device of the slice launches the same program.
    """
    rank = pp_group.rank_in_group
    size = pp_group.world_size
    columns = stage_group.world_size
    column = stage_group.rank_in_group
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if size * columns != world_size:
        raise ValueError(
            f"the pipeline wave needs the pipeline stages to cover the whole slice "
            f"(pp {size} stages of {columns} devices, world {world_size})"
        )
    perm = pipeline_permutation(
        pp_ranks=tuple(pp_group.ranks),
        stage_ranks=tuple(stage_group.ranks),
        column=column,
    )
    op = build_pp_permute_op(perm)
    wave = PPWave(
        rank=rank,
        size=size,
        rows=rows,
        hidden=hidden,
        num_aux=num_aux,
        dtype=dtype,
        device=device,
        launcher=lambda x: launch_now(op, x),
    )
    ok = wave.probe(column=column, columns=columns)
    # Every device must agree on the routing, so the verdict is gathered.
    verdicts: list = [None] * world_group.world_size
    torch.distributed.all_gather_object(verdicts, bool(ok), group=world_group.cpu_group)
    if not all(verdicts):
        raise RuntimeError(
            f"pipeline wave stage {rank} column {column}: the hand-off probe "
            f"failed (rank verdicts {verdicts}, cycles {perm})"
        )
    logger.info(
        "pipeline wave stage %d/%d column %d/%d: hand-off cycles %s, "
        "buffer [%d, %d] %s (%d aux slot(s))",
        rank,
        size,
        column,
        columns,
        perm,
        rows,
        wave.width,
        dtype,
        num_aux,
    )
    return wave
