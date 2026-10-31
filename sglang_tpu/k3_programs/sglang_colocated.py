# SPDX-License-Identifier: Apache-2.0
"""Both Kimi-K3 programs in one engine: prefill and decode on one slice.

Under P/D the decode slice is the bottleneck on AgentX while the prefill slice
mostly waits (prefix hits leave ~2.6% of the input to compute). Two engines
that each prefill and decode put both slices to decoding. The programs' HBM
fits once they share the routed experts, nearly all of the weights: at
equal expert parallelism -- 8, the pre-shard's layout, or 16, the fused prefill
MoE's (the decode runs it as half experts) -- both programs read the same
tensors, so the prefill takes the decode program's copy instead of loading its
own.

Prefill and decode bind the same unified pools (each with its own tables).
A request needs no transfer: the prefill leaves its KDA state in the
request's slot blocks and its KV and the draft's in its pages, where the
decode program imports them on the request's first decode step.
"""

from __future__ import annotations

import contextlib
from typing import Any

EXPERT_FAMILIES = ("expert_gu", "expert_gus", "expert_down", "expert_ds")
SHARED_EXPERT_PARALLEL = (8, 16)


@contextlib.contextmanager
def _no_expert_jobs(rank_loader: Any):
    """The rank loader without the expert families' read jobs."""
    original = rank_loader._experts_at
    rank_loader._experts_at = lambda *args, **kwargs: iter(())
    try:
        yield
    finally:
        rank_loader._experts_at = original


class _SharedExpertLoader:
    """A rank loader whose prefill iterator reads no expert families."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def iter_prefill_rank(self, *args: Any, **kwargs: Any):
        with _no_expert_jobs(self._loader):
            for group, name, array in self._loader.iter_prefill_rank(*args, **kwargs):
                if name.split("@", 1)[0] in EXPERT_FAMILIES:
                    continue
                yield group, name, array


def share_experts(prefiller: Any, decoder: Any) -> None:
    """Make ``prefiller`` (a ``KimiPrefiller``) use ``decoder``'s routed
    experts: load the decoder's weights, then load the prefill program's
    other weights around them."""
    from sglang_tpu.k3_programs.kimi_k3.config import active

    ep, decode_ep = active().prefill_expert_parallel, active().decode_expert_parallel
    if ep != decode_ep or ep not in SHARED_EXPERT_PARALLEL:
        raise ValueError(
            f"colocated K3 programs share the experts at prefill and decode "
            f"expert parallelism 8 or 16 alike, not {ep} and {decode_ep}"
        )
    decoder.ops.load_weights()
    experts = {name: decoder.ops.weights[name] for name in EXPERT_FAMILIES}
    program = prefiller.program
    program._rank_loader = _SharedExpertLoader(program._rank_loader)
    program.shared_experts = True
    load = program.load_weights

    def load_weights(presharded):
        program.weights.setdefault("moe", {}).update(experts)
        return load(presharded)

    program.load_weights = load_weights


class ColocatedSteps:
    """Extend batches to the prefill program, decode batches to the decode
    program, over one set of unified pools."""

    def __init__(self, prefill: Any, decode: Any) -> None:
        self.prefill, self.decode = prefill, decode
        self.plan = decode.plan
        # A request's first decode step takes the prefill's token from here
        # when the (overlapped) scheduler has not processed its prefill yet.
        prefill.first_tokens = decode.first_tokens = {}

    def forget_all(self) -> None:
        self.prefill.forget_all()
        self.decode.forget_all()
