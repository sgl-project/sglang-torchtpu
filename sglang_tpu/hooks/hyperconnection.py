# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp gated hyper-connections inside the TPU compiled forward.

eager_gated_residual
  Why: GatedResidual.__init__ wraps its mix and combine computations in
       torch.compile. TPU compiles the whole decoder forward, so the nested
       compiled function executes its own graph on the outer trace's
       placeholders and the first Qwen4-Exp warmup fails on every rank:
       "execute(): failed to prepare compiled mode arguments: cannot
       Materialize() a placeholder tensor" at hyperconnection.py _mix_compute.
  How: AFTER GatedResidual.__init__: swap the torch.compile wrappers for the
       functions they wrap, which the TPU trace compiles with the forward.
  Upstream PR or issue: not filed.
  Remove when: core asks the platform whether to compile nested helpers.
"""

from __future__ import annotations

from typing import Any


def _eager(fn: Any) -> Any:
    """The function a torch.compile wrapper wraps, or fn itself."""
    return getattr(fn, "_torchdynamo_orig_callable", fn)


def eager_gated_residual(result: None, residual: Any, *args, **kwargs) -> None:
    """AFTER GatedResidual.__init__."""
    for name in ("_mix_compute", "_combine_compute"):
        fn = getattr(residual, name, None)
        if fn is not None:
            setattr(residual, name, _eager(fn))
