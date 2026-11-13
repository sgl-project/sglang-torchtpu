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
# The dense families both programs read as the same pre-shard arrays, by the
# prefill's weight group: all but the KDA and MLA output projections, which
# the prefill rebuilds sequence-parallel from the chip pair's shards (and its
# MoE latent and shared projections, from its own cache). ~3.8 GiB a rank,
# ~230 more unified blocks.
SHARED_DENSE = {
    "io": ("embedding", "lm_head", "output_res", "final_norm"),
    "common": (
        "attn_res",
        "attn_norm",
        "mlp_res",
        "ffn_norm",
        "dense_gu",
        "dense_down",
    ),
    "kda": ("k_gate", "k_fb", "k_projection", "k_conv", "k_a_log", "k_dt", "k_norm"),
    "mla": ("m_qa", "m_qb", "m_ka", "m_kb", "m_gate", "m_qnorm", "m_knorm"),
    "moe": ("router", "router_bias", "latent_norm"),
}


def shared_dense() -> dict[str, tuple[str, ...]]:
    """The dense families the prefill takes from the decode program: under
    attention DP the decode holds several head shards of some KDA and MLA
    families (``ATTENTION_DP_WEIGHTS``), so the prefill keeps its own."""
    from sglang_tpu.k3_programs.kimi_k3.config import active

    if getattr(active(), "attention_dp", 1) == 1:
        return SHARED_DENSE
    from sglang_tpu.k3_programs.kimi_k3.kernels.decode_megakernel import (
        ATTENTION_DP_WEIGHTS,
    )

    return {
        group: tuple(n for n in names if n not in ATTENTION_DP_WEIGHTS)
        for group, names in SHARED_DENSE.items()
    }


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
    """A rank loader whose prefill iterator reads no expert families and
    skips the dense families taken from the decode program."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def iter_prefill_rank(self, *args: Any, **kwargs: Any):
        with _no_expert_jobs(self._loader):
            for group, name, array in self._loader.iter_prefill_rank(*args, **kwargs):
                if name.split("@", 1)[0] in EXPERT_FAMILIES:
                    continue
                if name in shared_dense().get(group, ()):
                    continue
                yield group, name, array


def share_experts(prefiller: Any, decoder: Any) -> None:
    """Make ``prefiller`` (a ``KimiPrefiller``) use ``decoder``'s routed
    experts and shared dense families: load the decoder's weights, then load
    the prefill program's other weights around them."""
    from sglang_tpu.k3_programs.kimi_k3.config import active

    ep, decode_ep = active().prefill_expert_parallel, active().decode_expert_parallel
    if ep != decode_ep or ep not in SHARED_EXPERT_PARALLEL:
        raise ValueError(
            f"colocated K3 programs share the experts at prefill and decode "
            f"expert parallelism 8 or 16 alike, not {ep} and {decode_ep}"
        )
    decoder.ops.load_weights()
    experts = {name: decoder.ops.weights[name] for name in EXPERT_FAMILIES}
    dense = {
        group: {name: decoder.ops.weights[name] for name in names}
        for group, names in shared_dense().items()
    }
    program = prefiller.program
    program._rank_loader = _SharedExpertLoader(program._rank_loader)
    program.shared_experts = True
    load = program.load_weights

    def load_weights(presharded):
        program.weights.setdefault("moe", {}).update(experts)
        for group, tensors in dense.items():
            program.weights.setdefault(group, {}).update(tensors)
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
