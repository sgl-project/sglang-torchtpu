# Adapted from vllm-project/vllm-torchtpu (Apache-2.0), commit c78529c; modified.
# Source: src/vllm_torchtpu/megakernel/kimi_k3/draft_ring.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark draft's ring, computed by the prefill program.

The decoder's draft (:mod:`draft`) attends a ring of the last 2048 positions'
projected keys and values of the target's residual streams entering the aux
layers (Kimi-K3's speculator: 24, 48, 72, 88, 92; Kimi-K3.1-Pro-DSpark: 76, 80,
84, 88, 92); a decoder that starts it empty drafts poorly until it fills.
:class:`PrefillRing` computes it window by window from the prefill stack's
residual streams, the pool writer stores it in the draft rows of the request's
KDA state blocks (``pools.write_draft_rings``), and the KV transfer (colocated:
the decoder's import from the same pools) carries it to
``KdaImporter.import_ring``. Under simulated acceptance the emitted tokens at
accepted positions are the draft's, so the ring must hold the prompt's context
before the first decode step. Each rank gathers the draft's
``fc``, ``kv``, ``kvnorm`` and ``hidden_norm`` shards (``dspark-<revision>`` in
the pre-shard directory) into replicated weights and projects its own window
rows for every KV head (``kernels.dspark.prefill_ring_update``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from sglang_tpu.k3_programs._support.logger import init_logger
from sglang_tpu.k3_programs.bridge import hbm, materialize, register_op, upload
from sglang_tpu.k3_programs.kimi_k3.program import kernel_modules

logger = init_logger(__name__)

# The residual stream entering an aux layer comes from the prefill's AttnRes
# block snapshots (``blocks`` slot L // 12, L a multiple of 12) or from its
# ``entering`` streams (each pair's input -- kept only on request -- middle
# and tap, the final op's input): ``draft_mla_prefill.stream_source``.
WEIGHT_ARRAYS = ("fc", "kv", "kvnorm", "hidden_norm")


def aux_streams(aux_layers, blocks, taps: dict):
    """``[len(aux_layers), rows, hidden]``: each aux layer's stream in aux order,
    ``taps[layer]`` (this rank's rows of an ``entering`` stream) or the block
    snapshot ``blocks[layer // 12]`` (``blocks`` ``[8, rows, hidden]``)."""
    return jnp.stack(
        [taps[layer] if layer in taps else blocks[layer // 12] for layer in aux_layers]
    )


class PrefillRing:
    """The prefill side of the DSpark ring (see the module docstring)."""

    def __init__(
        self, mesh, rank: int, axis: str, checkpoint: Path, presharded: Path
    ) -> None:
        (self.dspark,) = kernel_modules("dspark")
        self.config = self.dspark.DraftConfig.from_checkpoint(checkpoint)
        from sglang_tpu.k3_programs.kimi_k3.draft_mla_prefill import stream_source

        # Each aux layer's stream exists (stream_source raises otherwise).
        self.sources = {layer: stream_source(layer) for layer in self.config.aux_layers}
        self.mesh, self.rank, self.axis = mesh, rank, axis
        self._directory = (
            Path(presharded) / f"dspark-{Path(checkpoint).resolve().name[:8]}"
        )
        self.weights: tuple[torch.Tensor, ...] = ()
        self._ops: dict[Any, Any] = {}
        self._zero: torch.Tensor | None = None

    def load_weights(self) -> None:
        """Read this rank's four arrays of the draft's kernel shard and gather
        the replicated weights."""
        import time

        started = time.perf_counter()
        index = json.loads((self._directory / "index.json").read_text())
        if index.get("format") != self.dspark.KERNEL_SHARD_FORMAT:
            raise ValueError(f"{self._directory}: not a DSpark kernel shard set")
        raw = np.memmap(
            self._directory / f"rank{self.rank:02d}.bin", dtype=np.uint8, mode="r"
        )
        shards = []
        for name in WEIGHT_ARRAYS:
            spec = index["arrays"][name]
            dtype = (
                np.dtype(jnp.bfloat16)
                if spec["dtype"] == "bfloat16"
                else np.dtype(spec["dtype"])
            )
            count = int(np.prod(spec["shape"]))
            value = np.array(
                raw[spec["offset"] : spec["offset"] + count * dtype.itemsize]
            )
            shards.append(upload(value.view(dtype).reshape(spec["shape"])))
        c, axis = self.config, self.axis

        def gather(fc, kv, kvnorm, hidden_norm):  # [1, ...]: this rank's shards
            out = self.dspark.ring_weights_from_shards(
                c, fc[0], kv[0], kvnorm[0], hidden_norm[0], axis
            )
            return tuple(out[name] for name in ("fc", "kv", "k_norm", "hidden_norm"))

        op = register_op(
            "pallas::k3_prefill_draft_ring_weights",
            self.mesh,
            gather,
            (P(axis),) * 4,
            (P(),) * 4,
        )
        self.weights = tuple(op(*shards))
        materialize(self.weights[0])
        if self.rank == 0:
            gib = sum(t.numel() * t.element_size() for t in self.weights) / 2**30
            logger.info(
                "K3 prefill: DSpark ring weights resident (%.2f GiB per rank) in "
                "%.1f s; HBM %s",
                gib,
                time.perf_counter() - started,
                hbm(self.rank),
            )

    def empty(self, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        """A request's ring before any of its rows: ``(cache [1, layers,
        window, 128], meta [2])`` -- it covers ``start`` tokens and holds
        none of them."""
        c = self.config
        cache = torch.zeros(
            (1, c.layers, c.window, 2 * c.head_dim), dtype=torch.bfloat16
        ).to("tpu")
        meta = torch.tensor([start, start], dtype=torch.int32).to("tpu")
        return cache, meta

    def zero(self) -> tuple[torch.Tensor, torch.Tensor]:
        """A shared empty ring for a window's unused sequence slots (never
        donated)."""
        if self._zero is None:
            self._zero = self.empty(0)
        return self._zero

    @property
    def entering_layers(self) -> tuple[int, ...]:
        """The aux layers whose streams come from ``entering``, in aux order
        (the pair inputs among them must be kept: ``run_packed``'s ``keep``)."""
        return tuple(
            layer
            for layer in self.config.aux_layers
            if self.sources[layer] == "entering"
        )

    def first_aux(
        self, window: int, blocks, entering: dict, picked: np.ndarray
    ) -> tuple[torch.Tensor, ...]:
        """First-step drafts (KernelConfig.ring_draft_first_step): the decode's
        first-speculation aux operand of each of the window's slots
        (``draft_mla_prefill.first_aux_fn``, as the MLA draft's prefill keeps
        it); ``picked[s]`` slot ``s``'s last prompt row in ownership order, -1
        for none."""
        from sglang_tpu.k3_programs.kimi_k3.draft_mla_prefill import (
            FIRST_AUX_ROWS,
            first_aux_fn,
        )

        key = ("first_aux", window, len(picked))
        if key not in self._ops:
            c, axis = self.config, self.axis
            self._ops[key] = register_op(
                f"pallas::k3_prefill_draft_ring_first_aux_w{window}_s{len(picked)}",
                self.mesh,
                first_aux_fn(tuple(c.aux_layers), axis, len(picked), FIRST_AUX_ROWS),
                (P(axis),) * (1 + len(self.entering_layers)) + (P(),),
                (P(),) * len(picked),
                export_key=repr(
                    ("prefill_draft_ring_first_aux", window, len(picked), c.aux_layers)
                ),
            )
        return tuple(
            self._ops[key](
                blocks,
                *(entering[layer] for layer in self.entering_layers),
                torch.from_numpy(np.ascontiguousarray(picked, np.int32)).to("tpu"),
            )
        )

    def update(
        self,
        blocks: torch.Tensor,
        entering: dict,
        positions: np.ndarray,
        sequences: np.ndarray,
        ends: list[int],
        stops: list[int],
        rings: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[list, list]:
        """One packed window (``slots`` sequences): its residual streams
        (``blocks`` and the ``entering`` streams ``run_packed`` passes its ring
        hook) into each sequence's ring. ``positions`` / ``sequences``
        ``[window]`` per window row (``-1``: padding); ``ends`` / ``stops``
        per sequence (``-1``: no checkpoint). Returns the ``(cache, meta)``
        rings after the window and at the checkpoints."""
        slots, window = len(rings), int(positions.shape[0])

        def device(values) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(values, np.int32)).to("tpu")

        out = self._update_op(window, slots)(
            blocks,
            *(entering[layer] for layer in self.entering_layers),
            device(positions),
            device(sequences),
            device(ends),
            device(stops),
            *self.weights,
            *(cache for cache, _ in rings),
            *(meta for _, meta in rings),
        )
        after = list(zip(out[:slots], out[2 * slots : 3 * slots], strict=True))
        at = list(zip(out[slots : 2 * slots], out[3 * slots :], strict=True))
        return after, at

    def _update_op(self, window: int, slots: int) -> Any:
        """:meth:`update`'s op for a ``window`` of ``slots`` sequences."""
        key = ("update", window, slots)
        if key not in self._ops:
            c, axis, dspark = self.config, self.axis, self.dspark
            taps = self.entering_layers

            def body(blocks, *args):
                streams = {layer: tap[0] for layer, tap in zip(taps, args)}
                positions, sequences, ends, stops = args[len(taps) : len(taps) + 4]
                rest = args[len(taps) + 4 :]
                weights = dict(
                    zip(("fc", "kv", "k_norm", "hidden_norm"), rest[:4], strict=True)
                )
                caches, metas = rest[4 : 4 + slots], rest[4 + slots :]
                aux = aux_streams(c.aux_layers, blocks[0], streams)
                after, at = dspark.prefill_ring_update(
                    c,
                    weights,
                    aux,
                    positions,
                    sequences,
                    jnp.stack([cache[0] for cache in caches]),
                    ends,
                    stops,
                    axis,
                )
                firsts = [meta[1] for meta in metas]
                return (
                    *(after[s][None] for s in range(slots)),
                    *(at[s][None] for s in range(slots)),
                    *(jnp.stack((ends[s], firsts[s])) for s in range(slots)),
                    *(jnp.stack((stops[s], firsts[s])) for s in range(slots)),
                )

            self._ops[key] = register_op(
                f"pallas::k3_prefill_draft_ring_w{window}_s{slots}",
                self.mesh,
                body,
                (P(axis),) * (1 + len(taps))
                + (P(),) * 4
                + (P(),) * 4
                + (P(axis),) * slots
                + (P(),) * slots,
                (P(axis),) * (2 * slots) + (P(),) * (2 * slots),
                export_key=repr(
                    ("prefill_draft_ring", window, slots, dataclasses.astuple(c))
                ),
            )
        return self._ops[key]
