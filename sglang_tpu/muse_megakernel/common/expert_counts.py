# SPDX-License-Identifier: Apache-2.0
"""Running per-layer routed-expert counts of the SP prefill program, for offline EPLB placement
(SGLANG_TPU_MUSE_EXPERT_COUNTS_DIR). Each rank sums its program outputs on device and dumps
``counts_rank<r>.npy`` ([n_layers, n_experts] int64, valid rows only) every DUMP_EVERY steps."""

import os

import numpy as np
import torch

from sglang_tpu.envs import envs as tpu_envs

# Each dump syncs the device; arbitrary, small next to a 4096-row prefill chunk every ~300 ms.
DUMP_EVERY = 32


class ExpertCountRecorder:
    def __init__(self, out_dir: str, rank: int):
        self._path = os.path.join(out_dir, f"counts_rank{rank}.npy")
        self._sum: torch.Tensor | None = None
        self._steps = 0
        os.makedirs(out_dir, exist_ok=True)

    def add(self, counts: torch.Tensor) -> None:
        """counts: the program's [n_layers, n_experts] int32 output (or [tp * n_layers, ...]
        when the runtime hands back the global array, which the dump folds)."""
        c = counts.to(torch.int64)
        self._sum = c if self._sum is None else self._sum + c
        self._steps += 1
        if self._steps % DUMP_EVERY == 0:
            self.dump()

    def dump(self) -> None:
        if self._sum is None:
            return
        tmp = self._path + ".tmp.npy"
        np.save(tmp, self._sum.cpu().numpy())
        os.replace(tmp, self._path)


_RECORDER: list[ExpertCountRecorder] = []


def expert_counts_enabled() -> bool:
    return bool(tpu_envs.SGLANG_TPU_MUSE_EXPERT_COUNTS_DIR.get())


def record_expert_counts(*, counts: torch.Tensor, rank: int) -> None:
    if not _RECORDER:
        _RECORDER.append(
            ExpertCountRecorder(tpu_envs.SGLANG_TPU_MUSE_EXPERT_COUNTS_DIR.get(), rank)
        )
    _RECORDER[0].add(counts)
