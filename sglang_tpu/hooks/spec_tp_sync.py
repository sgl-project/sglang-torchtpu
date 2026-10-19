# SPDX-License-Identifier: Apache-2.0
"""Speculative TP sync broadcasts run with eager deferral off.

broadcast_undeferred
  Why: under SGLANG_TPU_EAGER_MODE=DEFER_AND_FUSE (the Kimi-K3 serving mode), the
       in-place torch.distributed.broadcast in SpecTpSync.sync does not deliver
       rank 0's tensor: receivers read garbage (int64 token 227 arrives as
       0xe300e300e3) or zeros, and rank 0 later blocks in the collective, which
       hangs TpuBlockSpecWorker's sampled-draft warmup. DEFER_NEVER is correct.
  How: AROUND SpecTpSync.sync: run it inside
       execution_mode.set_eager_mode(EagerMode.DEFER_NEVER).
  Upstream PR or issue: not filed yet (torch_tpu in-place collective under
       DEFER_AND_FUSE).
  Remove when: torch_tpu materializes deferred inputs of in-place collectives
       under DEFER_AND_FUSE.
"""

from __future__ import annotations

import torch


def broadcast_undeferred(original, sync, site, values: torch.Tensor) -> torch.Tensor:
    from torch_tpu._internal import execution_mode

    with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_NEVER):
        return original(sync, site, values)
