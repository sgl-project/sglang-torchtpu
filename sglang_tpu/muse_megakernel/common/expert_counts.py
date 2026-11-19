# SPDX-License-Identifier: Apache-2.0
"""Running per-layer routed-expert counts of the SP prefill program, for offline EPLB placement
(SGLANG_TPU_MUSE_EXPERT_COUNTS_DIR).

The compiled graph replays the prefill program's executable without re-entering the op's Python
body, so the sum lives on device: an int32 [n_layers, n_experts] buffer rides along with the KV
pools (donated and aliased), and the program adds each chunk's counts of its valid rows. A daemon
thread copies it to counts_rank<r>.npy every DUMP_SECONDS."""

import os
import threading
import time

import numpy as np
import torch

from sglang_tpu.envs import envs as tpu_envs

# Each dump syncs the buffer to the host; arbitrary, long next to a prefill chunk.
DUMP_SECONDS = 120


def expert_counts_enabled() -> bool:
    return bool(tpu_envs.SGLANG_TPU_MUSE_EXPERT_COUNTS_DIR.get())


def _dump_forever(buf: torch.Tensor, path: str) -> None:
    while True:
        time.sleep(DUMP_SECONDS)
        tmp = path + ".tmp.npy"
        np.save(tmp, buf.cpu().numpy().astype(np.int64))
        os.replace(tmp, path)


def count_buffer(*, n_layers: int, n_experts: int, device, rank: int) -> torch.Tensor:
    """The on-device running counts of this rank, with its dump thread started."""
    out_dir = tpu_envs.SGLANG_TPU_MUSE_EXPERT_COUNTS_DIR.get()
    os.makedirs(out_dir, exist_ok=True)
    buf = torch.zeros((n_layers, n_experts), dtype=torch.int32, device=device)
    path = os.path.join(out_dir, f"counts_rank{rank}.npy")
    threading.Thread(target=_dump_forever, args=(buf, path), daemon=True).start()
    return buf
