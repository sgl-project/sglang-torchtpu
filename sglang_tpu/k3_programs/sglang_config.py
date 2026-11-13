# SPDX-License-Identifier: Apache-2.0
"""A Kimi-K3 program's ``MegakernelEngineConfig`` from sglang's settings.

The programs are built from plain values (``config.MegakernelEngineConfig``):
vLLM fills them from its configuration, sglang from its server arguments here.
The decode program speculates with DSpark by a per-batch-size schedule; the
default (7 drafts up to four requests, 3 up to eight, none above) is the one
the compiled verify shapes support (``docs/megakernel/README.md`` upstream).
"""

from __future__ import annotations

from collections.abc import Sequence

from sglang_tpu.k3_programs.config import MegakernelEngineConfig
from sglang_tpu.k3_programs.kimi_k3.layout import cache_layout
from sglang_tpu.k3_programs.kimi_k3.program import MAX_STEP_TOKENS

TP_SIZE = 32
DCP_SIZE = 32
ROLES = ("prefill", "decode")
# (min batch, max batch, drafts): 7 drafts through batch four, 3 through eight.
DEFAULT_DSPARK_SCHEDULE = ((1, 4, 7), (5, 8, 3))


def program_kv_cache_dtype(kv_cache_dtype: str | None) -> str:
    """sglang's ``--kv-cache-dtype`` as the programs spell it."""
    value = str(kv_cache_dtype or "auto")
    return {"bf16": "bfloat16"}.get(value, value)


def acceptance_length_to_rates(length: float, drafts: int) -> tuple[float, ...]:
    """A mean acceptance length (bonus token included) as per-position
    unconditional acceptance rates, by vLLM's minimum-variance rule
    (``SpeculativeConfig._acceptance_length_to_rates``): every position up to
    ``length - 1`` accepted, the fraction on the next one."""
    if not 1.0 <= length <= drafts + 1:
        raise ValueError(
            f"a mean acceptance length over {drafts} drafts is in [1, {drafts + 1}], "
            f"not {length}"
        )
    accepted = length - 1
    full = int(accepted)
    rates = [1.0] * full + [accepted - full] + [0.0] * (drafts - full - 1)
    return tuple(rates[:drafts])


def engine_config(
    *,
    role: str,
    max_running_requests: int,
    context_length: int,
    chunked_prefill_size: int,
    kv_cache_dtype: str = "auto",
    speculative_tokens: int = 0,
    schedule: Sequence[tuple[int, int, int]] | None = DEFAULT_DSPARK_SCHEDULE,
    simulate_acc_lengths: dict[int, float] | None = None,
    attention_dp: int = 1,
) -> MegakernelEngineConfig:
    """The program configuration for ``role``.

    ``speculative_tokens``: DSpark drafts per step at the schedule's widest
    (0: no speculation). ``simulate_acc_lengths``: per draft width, the mean
    acceptance length to accept synthetically (InferenceX AgentX pins it, e.g.
    ``{7: 3.78, 3: 2.87}`` for Kimi-K3 DSpark); None verifies for real.
    """
    if role not in ROLES:
        raise ValueError(f"the Kimi-K3 programs have roles {ROLES}, not {role!r}")
    kv_cache_dtype = program_kv_cache_dtype(kv_cache_dtype)
    block_size = cache_layout(kv_cache_dtype).tokens_per_block
    spec_schedule = None
    by_width = None
    rates = None
    if speculative_tokens:
        spec_schedule = tuple(tuple(int(v) for v in entry) for entry in schedule or ())
        widths = sorted({drafts for _, _, drafts in spec_schedule if drafts})
        if simulate_acc_lengths is not None:
            missing = [w for w in widths if w not in simulate_acc_lengths]
            if missing:
                raise ValueError(
                    f"no simulated acceptance length for {missing} drafts "
                    f"(the schedule runs {widths})"
                )
            by_width = tuple(
                (w, acceptance_length_to_rates(simulate_acc_lengths[w], w))
                for w in widths
            )
            rates = acceptance_length_to_rates(
                simulate_acc_lengths[speculative_tokens], speculative_tokens
            )
    budget = chunked_prefill_size
    if role == "prefill":
        # The prefill's windows go up to MAX_STEP_TOKENS; a longer chunk (a
        # 16384-token fp8 page) runs in consecutive windows of one step.
        budget = min(budget, MAX_STEP_TOKENS)
    return MegakernelEngineConfig(
        role=role,
        tensor_parallel_size=TP_SIZE,
        decode_context_parallel_size=DCP_SIZE // attention_dp,
        block_size=block_size,
        max_num_seqs=max_running_requests,
        max_model_len=context_length,
        max_num_batched_tokens=budget,
        num_speculative_tokens=speculative_tokens,
        num_speculative_tokens_per_batch_size=spec_schedule,
        synthetic_acceptance_rates=rates,
        synthetic_acceptance_rates_by_width=by_width,
        kv_cache_dtype=kv_cache_dtype,
        cp_kv_cache_interleave_size=block_size,
        attention_dp=attention_dp,
    )
