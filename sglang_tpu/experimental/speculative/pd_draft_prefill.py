"""Decode-side draft prompt pass of SGLANG_TPU_PD_DECODE_DRAFT_PREFILL.

A PD decode receives each request's target KV and its packed target hidden rows
(pd_hidden_pool.py) instead of a draft KV; before the first decode step it runs
the draft over the prompt, in extend-shaped groups cut from the prebuilt batch; a
prompt longer than the largest extend bucket runs in successive bucket-sized chunks.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Sequence

import msgspec
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.experimental.speculative.pd_hidden_pool import PdHiddenPool

# (prebuilt ScheduleBatch) -> None.
PdDraftPrefiller = Callable[[ScheduleBatch], None]

_pd_draft_prefiller: PdDraftPrefiller | None = None


def register_pd_draft_prefiller(prefiller: PdDraftPrefiller) -> None:
    """Set the draft prompt pass a PD decode runs on each prebuilt batch."""
    global _pd_draft_prefiller
    _pd_draft_prefiller = prefiller


def get_pd_draft_prefiller() -> PdDraftPrefiller:
    if _pd_draft_prefiller is None:
        raise RuntimeError("No TpuSpecWorker registered a PD draft prefiller.")
    return _pd_draft_prefiller


def draft_feature_size(config: SpecAuxHiddenStateConfig, hidden_size: int) -> int:
    """Width of the target hidden the draft reads: its aux layers, or the final hidden."""
    if config.eagle_use_aux_hidden_state:
        layer_ids = config.eagle_aux_hidden_state_layer_ids
    elif config.dflash_use_aux_hidden_state:
        layer_ids = config.dflash_target_layer_ids
    else:
        return hidden_size
    if layer_ids is None:
        raise NotImplementedError(
            "SGLANG_TPU_PD_DECODE_DRAFT_PREFILL sizes the hidden rows from the "
            "draft's aux layer ids, which this draft does not declare."
        )
    return len(layer_ids) * hidden_size


def pd_hidden_bytes_per_token(
    config: SpecAuxHiddenStateConfig, model_config: ModelConfig, dtype: torch.dtype
) -> int:
    """The hidden pool's HBM per KV token slot on this rank."""
    return PdHiddenPool.bytes_per_token(
        feature_size=draft_feature_size(config, model_config.hidden_size),
        total_kv_heads=model_config.get_total_num_kv_heads(),
        attn_tp_size=get_parallel().attn_tp_size,
        dtype=dtype,
    )


def build_pd_hidden_pool(
    target_runner: ModelRunner, *, paged_by_swa: bool
) -> PdHiddenPool:
    """A hidden pool paged like the target's SWA pages when paged_by_swa (a windowed
    draft), else like its full-attention pages."""
    kv_pool = target_runner.token_to_kv_pool
    draft_kv_is_swa = target_runner.spec_aux_config.dflash_draft_kv_is_swa
    if isinstance(kv_pool, SWAKVPool) and paged_by_swa != draft_kv_is_swa:
        # budget_pd_hidden_pool sized the rows per SWA or full token by this flag.
        raise RuntimeError(
            f"The PD hidden pool pages by {'SWA' if paged_by_swa else 'full'} slots "
            f"but the KV budget sized it by dflash_draft_kv_is_swa={draft_kv_is_swa}."
        )
    groups = get_raiden_kv_caches(kv_pool)
    kind = "SWA" if paged_by_swa else "full-attention"
    pages = groups.swa if paged_by_swa else groups.full
    if not pages:
        raise RuntimeError(
            f"The PD hidden pool pages like the target's {kind} KV, which this "
            "stage does not hold."
        )
    parallel = get_parallel()
    return PdHiddenPool(
        num_pages=int(pages[0].shape[0]),
        # Not pages[0].shape[1]: a SEQ_ALONG_LANE page keeps its tokens on the last axis.
        page_size=kv_pool.page_size,
        feature_size=draft_feature_size(
            target_runner.spec_aux_config, target_runner.model_config.hidden_size
        ),
        total_kv_heads=target_runner.model_config.get_total_num_kv_heads(),
        attn_tp_size=parallel.attn_tp_size,
        attn_tp_group=parallel.attn_tp_group if parallel.attn_tp_size > 1 else None,
        dtype=target_runner.dtype,
        device=target_runner.device,
        paged_by_swa=paged_by_swa,
    )


class DraftPromptChunk(msgspec.Struct, frozen=True):
    """Extend tokens [start, start + length) of request req of a prebuilt batch."""

    req: int
    start: int
    length: int


def prebuilt_extend_groups(
    extend_lens: Sequence[int],
    starts: Sequence[int],
    *,
    max_tokens: int,
    max_reqs: int,
) -> list[list[DraftPromptChunk]]:
    """Groups of at most max_tokens extend tokens and max_reqs requests covering each
    request's extend tokens from its start, in order."""
    groups: list[list[DraftPromptChunk]] = []
    group: list[DraftPromptChunk] = []
    tokens = 0
    for req, (extend_len, start) in enumerate(zip(extend_lens, starts, strict=True)):
        # A prompt longer than a bucket runs alone in full buckets; its tail packs
        # with the next requests like a short prompt.
        while extend_len - start > max_tokens:
            if group:
                groups.append(group)
                group, tokens = [], 0
            groups.append([DraftPromptChunk(req=req, start=start, length=max_tokens)])
            start += max_tokens
        length = extend_len - start
        if group and (tokens + length > max_tokens or len(group) == max_reqs):
            groups.append(group)
            group, tokens = [], 0
        group.append(DraftPromptChunk(req=req, start=start, length=length))
        tokens += length
    if group:
        groups.append(group)
    return groups


def swa_mapped_starts(swa_loc: torch.Tensor, extend_lens: Sequence[int]) -> list[int]:
    """Per request, the extend offset of its first token with an SWA slot (nonzero
    swa_loc); every later token must have one too."""
    starts = []
    offsets = itertools.accumulate(extend_lens, initial=0)
    for offset, length in zip(offsets, extend_lens):
        mapped = swa_loc[offset : offset + length] != 0
        start = int(torch.argmax(mapped.to(torch.int8))) if length else 0
        if length and not bool(mapped[start:].all()):
            raise RuntimeError(
                "The windowed draft needs SWA slots for a suffix of each prompt; "
                f"got a mapping with holes for a {length}-token extend."
            )
        starts.append(start)
    return starts


def prebuilt_extend_view(
    batch: ScheduleBatch, chunks: Sequence[DraftPromptChunk], *, vocab_size: int
) -> ScheduleBatch:
    """An EXTEND copy of batch over these chunks' prebuilt KV slots; a chunk that
    stops short of its prompt's end is the view's chunked request."""
    view = copy.copy(batch)
    reqs = [batch.reqs[chunk.req] for chunk in chunks]
    prefix_lens = [batch.prefix_lens[chunk.req] + chunk.start for chunk in chunks]
    extend_lens = [chunk.length for chunk in chunks]
    loc_offsets = list(itertools.accumulate(batch.extend_lens, initial=0))
    view.reqs = reqs
    view.forward_mode = ForwardMode.EXTEND
    view.input_ids = torch.tensor(
        [
            token
            for req, prefix, length in zip(reqs, prefix_lens, extend_lens)
            for token in req.origin_input_ids[prefix : prefix + length]
        ],
        dtype=torch.int64,
    )
    loc_starts = [loc_offsets[chunk.req] + chunk.start for chunk in chunks]
    view.out_cache_loc = torch.cat(
        [
            batch.out_cache_loc[start : start + length]
            for start, length in zip(loc_starts, extend_lens)
        ]
    )
    # Host copies: the shim pads them on host and uploads at the bucket, while a
    # device slice would compile a pad per group size and offset.
    rows = torch.tensor([chunk.req for chunk in chunks], dtype=torch.int64)
    view.req_pool_indices_cpu = batch.req_pool_indices_cpu[rows]
    view.req_pool_indices = view.req_pool_indices_cpu.clone()
    view.seq_lens_cpu = torch.tensor(
        [prefix + length for prefix, length in zip(prefix_lens, extend_lens)],
        dtype=batch.seq_lens_cpu.dtype,
    )
    view.seq_lens = view.seq_lens_cpu.clone()
    view.orig_seq_lens = view.seq_lens_cpu.to(torch.int32)
    view.seq_lens_sum = int(view.seq_lens_cpu.sum())
    view.prefix_lens = prefix_lens
    view.extend_lens = extend_lens
    view.extend_num_tokens = sum(extend_lens)
    view.multimodal_inputs = [req.multimodal_inputs for req in reqs]
    view.return_logprob = False
    view.spec_info = None
    view.chunked_req = None
    view.chunked_req_next_prompt_token = None
    last = chunks[-1]
    if last.start + last.length < batch.extend_lens[last.req]:
        # Only a group's last chunk can stop short (prebuilt_extend_groups).
        view.chunked_req = reqs[-1]
        view.chunked_req_next_prompt_token = reqs[-1].origin_input_ids[
            prefix_lens[-1] + last.length
        ]
    view.sampling_info = SamplingBatchInfo.from_schedule_batch(view, vocab_size)
    return view
