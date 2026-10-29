# SPDX-License-Identifier: Apache-2.0
"""sglang requests in and out of a Kimi-K3 program step.

A program step takes every request of the batch as host data
(``program.RequestInput``) and returns each request's tokens, ``-1`` padded:
one for a plain step, up to the accepted drafts and the bonus token for a
DSpark step. sglang's spec workers return the same thing as flat accepted
tokens and per-request accept lengths, which is what :func:`accepted` builds.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang_tpu.k3_programs.program import RequestInput, SamplingSpec


def sampling_spec(params: Any, rid: str) -> SamplingSpec:
    """sglang ``SamplingParams`` as the programs sample.

    sglang writes greedy as ``top_k == 1`` (and resets the temperature to 1),
    and the whole vocabulary as ``TOP_K_ALL``; the programs want a temperature
    of 0 for greedy and ``top_k <= 0`` for the whole vocabulary. Every rank
    samples with the same seeded generator, so an unseeded request is seeded
    from its id.
    """
    greedy = params.top_k == 1
    seed = getattr(params, "sampling_seed", None)
    return SamplingSpec(
        temperature=0.0 if greedy else float(params.temperature),
        top_p=float(params.top_p),
        top_k=0 if greedy or params.top_k >= TOP_K_ALL else int(params.top_k),
        seed=int(seed) if seed is not None else zlib.crc32(rid.encode()),
    )


def request_input(
    *,
    rid: str,
    origin_input_ids: Sequence[int],
    output_ids: Sequence[int],
    computed: int,
    scheduled: int,
    block_tables: tuple[np.ndarray, ...],
    sampling_params: Any,
) -> RequestInput:
    """One request of a step: ``computed`` tokens already in the KV pools and
    ``scheduled`` new ones (a decode step schedules one, the next token)."""
    token_ids = np.asarray([*origin_input_ids, *output_ids], dtype=np.int32)
    if computed + scheduled > len(token_ids):
        raise ValueError(
            f"request {rid}: {computed} computed + {scheduled} scheduled tokens "
            f"exceed its {len(token_ids)} known tokens"
        )
    return RequestInput(
        req_id=rid,
        computed=computed,
        scheduled=scheduled,
        token_ids=token_ids,
        num_tokens=len(token_ids),
        num_prompt_tokens=len(origin_input_ids),
        block_tables=block_tables,
        sampling=sampling_spec(sampling_params, rid),
    )


def accepted(tokens: np.ndarray, num_requests: int) -> tuple[list[int], list[int]]:
    """A step's ``[>= num_requests, width]`` token rows (``-1`` padded) as
    sglang's spec result: the accepted tokens of every request back to back,
    and how many each request produced."""
    rows = np.asarray(tokens)[:num_requests]
    flat: list[int] = []
    lens: list[int] = []
    for row in rows:
        kept = row[row >= 0]
        if kept.size and (row[: kept.size] < 0).any():
            raise ValueError(f"step tokens {row.tolist()} are not -1 padded at the end")
        flat.extend(int(t) for t in kept)
        lens.append(int(kept.size))
    return flat, lens
