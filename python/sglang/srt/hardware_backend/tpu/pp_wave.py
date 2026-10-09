# SPDX-License-Identifier: Apache-2.0
"""Pipeline-stage hand-off over ICI as a lockstep wave.

torch_tpu has no device send/recv, and a collective over any subset of the
slice's devices only runs when every device launches it in the same order.
So every hand-off is one collective-permute over all pipeline stages at once,
around one closed cycle: stage r hands the sum of its hidden states and
residual to stage r+1 and receives stage r-1's, and the last stage's block
lands on stage 0, which ignores it. The stage after a boundary must add
hidden states and residual before its first norm, as a pre-norm decoder's
fused add-norm does, so the sum with a zero residual carries the same
residual stream at half the bytes. That norm reads the sum rounded to the
model dtype instead of the float32 sum, a small difference at every stage
boundary. The closed cycle lets XLA lower the permute to a full-bandwidth
ring transfer. A batch's activations
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
"""

import logging
import os
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

_OP_NAME = "sglang_tpu::pp_permute"
_OPS: dict[tuple[int, ...], Any] = {}


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
    from torch_tpu._internal.pallas import get_pallas_mesh

    from sglang.srt.hardware_backend.tpu.patches import get_tpu_jax_op

    jax_op = get_tpu_jax_op()

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
    def probe(self) -> bool:
        """One batch iteration and its full drain on every stage, each stage
        sending its own number plus one; True when this stage received its
        number (stage 0 receives the last stage's block and ignores it).
        Every stage's launch count is level again afterwards."""
        payload = torch.full_like(self.zero, float(self.rank + 1))
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
        want = float(self.rank)
        x = got[self.KEYS[0]]
        return float(x[0, 0].item()) == want and float(x[-1, -1].item()) == want


def build_pp_wave(
    pp_group: Any, *, rows: int, hidden: int, dtype: torch.dtype, device: torch.device
) -> PPWave:
    """Create the stage's wave on the device and verify the hand-off routing.

    The permute is routed by mesh position, which follows distributed rank
    order; the cycle closes from the last stage back to the first.
    """
    rank = int(pp_group.rank_in_group)
    size = int(pp_group.world_size)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if size != world_size:
        raise ValueError(
            f"the pipeline wave needs the pipeline group to span the whole slice "
            f"(pp {size}, world {world_size}); run pipeline stages at tp 1"
        )
    ranks = tuple(pp_group.ranks)
    perm = tuple((ranks[r], ranks[(r + 1) % size]) for r in range(size))
    op = build_pp_permute_op(perm)
    wave = PPWave(
        rank=rank,
        size=size,
        rows=rows,
        hidden=hidden,
        dtype=dtype,
        device=device,
        launcher=lambda x: launch_now(op, x),
    )
    ok = wave.probe()
    # Every stage must agree on the routing, so the verdict is gathered.
    verdicts: list = [None] * size
    torch.distributed.all_gather_object(verdicts, bool(ok), group=pp_group.cpu_group)
    if not all(verdicts):
        raise RuntimeError(
            f"pipeline wave stage {rank}: the hand-off probe failed "
            f"(stage verdicts {verdicts}, ranks {ranks})"
        )
    logger.info(
        "pipeline wave stage %d/%d: hand-off cycle over ranks %s, buffer [%d, %d] %s",
        rank,
        size,
        ranks,
        rows,
        hidden,
        dtype,
    )
    return wave
