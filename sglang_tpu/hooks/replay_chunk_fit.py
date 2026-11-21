# SPDX-License-Identifier: Apache-2.0
"""Admit an encoder-replay hit whole when its extend fits beside the replay rows.

Why: with --enable-encoder-swa-bounded-replay a prefix hit's first chunk also
     runs up to 128 replay rows, so PrefillAdder lowers the hit's chunk limit by
     replay_len (1024 -> 896 with page 256). The fits-whole check compares the
     page-ceiled extend against that off-page limit, so a hit with 769-896 new
     tokens (ceiled 1024 > 896) was split into a 768-token chunk plus a second
     step although its rows fit one step; under DP lockstep every rank pays it.
How: the exact-chunk-fill path already compares and bills the raw extend.
     AROUND PrefillAdder._select_prefill_admission turns exact_chunk_fill on for
     one call when a replay hit's raw extend fits the reduced limit (and any
     per-request chunk_tokens_cap) but its ceiled extend does not, and keeps the
     result only if it admits the request whole (a tighter budget cap that still
     splits it falls back to core's page-floored chunk). AROUND PrefillAdder._update_prefill_budget turns it on
     while billing a replay hit, so the compute budgets (rem_chunk_tokens,
     rem_input_tokens, rem_dllm_tokens) take the rows the step runs; the KV
     budgets still reserve page-ceiled tokens. Without replay rows both hooks
     call core unchanged.
Upstream PR or issue: none.
Remove when: core compares and bills the raw extend beside replay rows.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def _exact_chunk_fill(adder):
    saved = adder.exact_chunk_fill
    adder.exact_chunk_fill = True
    try:
        yield
    finally:
        adder.exact_chunk_fill = saved


def admit_fitting_replay_hit_whole(original, adder, req, **kwargs):
    if (
        adder.exact_chunk_fill
        or adder.dllm_config is not None
        or adder.rem_chunk_tokens is None
    ):
        return original(adder, req, **kwargs)
    prefix_len = req.prefix_len + kwargs["host_hit_length"]
    replay_len = adder._replay_len(req, prefix_len)
    if not replay_len:
        return original(adder, req, **kwargs)
    extend_len = len(req.full_untruncated_fill_ids) - prefix_len
    chunk_limit = adder.rem_chunk_tokens - replay_len
    if kwargs.get("chunk_tokens_cap") is not None:
        # Core applies the per-request cap (dynamic chunking) after the budget.
        chunk_limit = min(chunk_limit, kwargs["chunk_tokens_cap"])
    if not extend_len <= chunk_limit < adder.ceil_paged_tokens(extend_len):
        # Core already admits it whole, or it does not fit either way.
        return original(adder, req, **kwargs)
    with _exact_chunk_fill(adder):
        admission = original(adder, req, **kwargs)
    if getattr(admission, "is_chunked", False):
        # A budget cap below the chunk limit splits it: keep the page-floored chunk.
        return original(adder, req, **kwargs)
    return admission


def bill_replay_hit_raw_extend(original, adder, *args, **kwargs):
    if adder.exact_chunk_fill or not kwargs.get("replay_len"):
        return original(adder, *args, **kwargs)
    with _exact_chunk_fill(adder):
        return original(adder, *args, **kwargs)


def hook_specs() -> list[tuple]:
    from sglang.srt.plugins.hook_registry import HookType

    adder = "sglang.srt.managers.schedule_policy.PrefillAdder"
    return [
        (
            f"{adder}._select_prefill_admission",
            admit_fitting_replay_hit_whole,
            HookType.AROUND,
        ),
        (
            f"{adder}._update_prefill_budget",
            bill_replay_hit_raw_extend,
            HookType.AROUND,
        ),
    ]
