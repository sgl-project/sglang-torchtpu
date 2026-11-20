# SPDX-License-Identifier: Apache-2.0
"""Two-pass draft prompt pass of a DSA NextN draft: KV-only rows, then last rows.

TpuChainSpecWorker's draft prompt pass runs the NextN layer over every padded
prompt row but keeps one output row per request; the other rows only leave
their draft KV (MLA latent and index keys) behind for later queries. Where the
token bucket is large enough the pass is split in two:

pass 1: the layer input and attention input of every row, then the cache-only
        store of the layer's latent and index keys (dsa_layer_kv_write). No
        query, attention, indexer scoring or MoE; the draft's logits step still
        runs on the extend cap's rows, unused. It replaces the NextN backbone in
        the compiled draft forward under either tracer (chain_compile_policy).
pass 2: the draft prompt program on each request's last prompt row only, over
        the prefix pass 1 stored. Its hidden, logits and DSA top-k capture are
        the pass's results.

The proposals match the single pass's up to rounding: pass 1 stores the bytes
the full layer stores, and a row's output reads only cache entries rows up to
it wrote, but pass 2 computes the last rows in a smaller program, so their
low-order bits can differ and flip a near-tie draft token.
"""

from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.layers.layer_boundary.adapters.attention import get_attn_tp_context
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.deepseek_nextn import (
    DeepseekModelNextN,
    DeepseekV3ForCausalLMNextN,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang_tpu._logging import get_logger
from sglang_tpu.dsa_layer_kv_write import (
    dsa_layer_kv_unsupported_reason,
    write_dsa_layer_kv,
)
from sglang_tpu.envs import envs as tpu_envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = get_logger(__name__)

# Off switch for debugging and bisecting: False keeps the single prompt pass.
# Outputs never depend on it; the startup log says which way the pass goes.
KV_ONLY_PROMPT_ENABLED = True

# Smallest padded prompt bucket that takes two passes. Carried from the same
# split's TPU7x measurements at GLM-5.2 dims in vllm-torchtpu, where two passes
# won at every measured cell from 1024 tokens. On GLM-5.3-Flash pass 2 runs at
# 1024, so 2048 is the first two-pass bucket; its TTFT measured neutral there.
KV_ONLY_PROMPT_MIN_TOKENS = 1024


def kv_only_prompt_unsupported_reason(
    *, draft_model: torch.nn.Module, attn_backend: object
) -> str | None:
    """Why this draft keeps the single prompt pass, or None to take two passes.

    Each reason is a path the two passes do not cover yet or where the draft
    stores its KV differently, so they would change what later steps read.
    """
    from sglang.srt.runtime_context import get_disagg, get_parallel
    from sglang_tpu.pallas_export import get_pallas_tpu_device_kind

    if not KV_ONLY_PROMPT_ENABLED:
        return "KV_ONLY_PROMPT_ENABLED is False (kv_only_prompt.py)"
    if not isinstance(draft_model, DeepseekV3ForCausalLMNextN) or not isinstance(
        draft_model.model, DeepseekModelNextN
    ):
        return f"the draft is {type(draft_model).__name__}, not a DeepSeek NextN layer"
    attn = draft_model.model.decoder.self_attn
    if not isinstance(attn, DeepseekV2AttentionMLA):
        return f"the draft layer's attention is {type(attn).__name__}, not MLA"
    # An idle DP rank would have to replay both passes in the busy ranks' order.
    if get_parallel().attn_dp_enabled:
        return "DP attention is on"
    # A P/D prefill ships hidden states instead; the decode's draft prefill is
    # not covered yet. (Pipeline parallelism is refused with P/D off.)
    if get_disagg().disaggregation_mode != "null":
        return "P/D disaggregation is on"
    if tpu_envs.SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT.get():
        return "SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT is on"
    reason = dsa_layer_kv_unsupported_reason(attn=attn, backend=attn_backend)
    if reason is not None:
        return reason
    device_kind = get_pallas_tpu_device_kind()
    if device_kind != "TPU7x":
        return f"its token floor is measured on TPU7x, not {device_kind}"
    return None


class KvOnlyPromptPlan(msgspec.Struct, frozen=True):
    """How a drafter's prompt pass takes two passes: the token buckets that do,
    pass 2's bucket, and the pass-1 backbone the compile policy dispatches to."""

    tokens: frozenset[int]
    last_row_tokens: int
    backbone: Any


def kv_only_prompt_plan(
    *,
    draft_model: torch.nn.Module,
    attn_backend: object,
    buckets: list[int],
    last_row_tokens: int,
) -> KvOnlyPromptPlan | None:
    """The two-pass plan of a draft, or None to keep the single pass; logged."""
    reason = kv_only_prompt_unsupported_reason(
        draft_model=draft_model, attn_backend=attn_backend
    )
    min_tokens = kv_only_prompt_min_tokens(last_row_tokens=last_row_tokens)
    tokens = sorted(t for t in buckets if t >= min_tokens) if reason is None else []
    log_kv_only_prompt(reason=reason, tokens=tokens, last_row_tokens=last_row_tokens)
    if not tokens:
        return None
    return KvOnlyPromptPlan(
        tokens=frozenset(tokens),
        last_row_tokens=last_row_tokens,
        backbone=functools.partial(forward_kv_only_backbone, draft_model.model),
    )


def kv_only_prompt_min_tokens(*, last_row_tokens: int) -> int:
    """Smallest padded prompt bucket that takes two passes.

    Pass 2 runs the draft prompt program at ``last_row_tokens``, the bucket a
    batch of one row per request pads to, so a bucket below twice that saves
    less than pass 2 costs.
    """
    return max(KV_ONLY_PROMPT_MIN_TOKENS, 2 * last_row_tokens)


def log_kv_only_prompt(
    *, reason: str | None, tokens: list[int], last_row_tokens: int
) -> None:
    """Say once at startup which way the draft prompt pass goes, and why."""
    if reason is not None:
        logger.info("Draft prompt pass keeps a single pass: %s.", reason)
    elif not tokens:
        logger.info(
            "Draft prompt pass keeps a single pass: no token bucket reaches "
            "the two-pass floor of %d tokens.",
            kv_only_prompt_min_tokens(last_row_tokens=last_row_tokens),
        )
    else:
        logger.info(
            "Draft prompt pass stores the draft KV in a KV-only pass, then runs "
            "the last rows at %d tokens, for token buckets %s.",
            last_row_tokens,
            tokens,
        )


def nextn_layer_input(
    nextn: DeepseekModelNextN,
    *,
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor | None,
) -> torch.Tensor:
    """The NextN decoder layer's input rows, as DeepseekModelNextN.forward builds them.

    A test runs both on the same inputs; change the two together.
    """
    if input_embeds is None:
        assert not (
            forward_batch.forward_mode.is_extend()
            and forward_batch.contains_mm_inputs()
        ), "the draft prompt pass hands a multimodal prompt its embeds"
        input_embeds = forward_batch.mm_input_embeds
        if input_embeds is None:
            input_embeds = nextn.embed_tokens(input_ids)
    previous_hidden_states = forward_batch.spec_info.hidden_states
    if nextn.rot_weight is not None:
        previous_hidden_states = torch.matmul(previous_hidden_states, nextn.rot_weight)
    eh_input = torch.cat(
        (nextn.enorm(input_embeds), nextn.hnorm(previous_hidden_states)), dim=-1
    )
    if isinstance(nextn.eh_proj, ReplicatedLinear):
        hidden_states, _ = nextn.eh_proj(eh_input)
        return hidden_states
    return nextn.eh_proj(eh_input)


def forward_kv_only_backbone(
    nextn: DeepseekModelNextN,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pass 1, in place of DeepseekModelNextN.forward: store the NextN layer's
    draft KV for every row and nothing else.

    The attention input comes from the decoder layer's own boundary, so norms,
    residual streams and an input-scattered gather are the full layer's.
    Returns the layer input rows, which only the logits step after it reads
    (pass 1's logits are not used); the caches are the result.
    """
    # The cache-only store mirrors a layer that runs its own indexer.
    assert not forward_batch.reuse_dsa_topk_indices, "pass 1 reuses no top-k"
    hidden_states = nextn_layer_input(
        nextn,
        input_ids=input_ids,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
    )
    decoder = nextn.decoder
    residual_batch.start(forward_batch)
    # quant_format is left at its default: it is set on gfx95 GPUs only.
    attn_input = decoder.attn_boundary.prepare(hidden_states, forward_batch)
    attn_tp_context = get_attn_tp_context()
    write_dsa_layer_kv(
        decoder.self_attn,
        backend=get_attn_backend(),
        forward_batch=forward_batch,
        qkv_latent=attn_tp_context.fetch_qkv_latent(),
        hidden_states=attn_input,
        positions=positions,
    )
    attn_tp_context.clear_attn_inputs()
    forward_batch.residual_stream = None
    return hidden_states


def last_prompt_rows(extend_lens: list[int]) -> torch.Tensor:
    """Host int64 [bs]: the row of each request's last prompt token."""
    return torch.cumsum(torch.tensor(extend_lens, dtype=torch.int64), dim=0) - 1


def last_row_view(batch: ScheduleBatch) -> ScheduleBatch:
    """Pass 2's batch: an EXTEND copy of batch over each request's last prompt row.

    One token per request at its last position, over the full prefix (which
    pass 1 stored), writing that row's KV slot again. The draft input ids and
    target hidden of those rows are set on the forward batch, gathered on device.
    """
    out_cache_loc = batch.out_cache_loc
    assert out_cache_loc.device.type == "cpu", "HOST_PADDED extend stages slots on host"
    bs = len(batch.extend_lens)
    view = copy.copy(batch)
    view.input_ids = torch.zeros((bs,), dtype=torch.int64)
    view.out_cache_loc = out_cache_loc[last_prompt_rows(batch.extend_lens)]
    view.prefix_lens = [int(n) - 1 for n in batch.seq_lens_cpu.tolist()]
    view.extend_lens = [1] * bs
    view.extend_num_tokens = bs
    view.extend_logprob_start_lens = [0] * bs
    # The single pass also embeds each last row from its token id, not from mm
    # embeds (TpuChainSpecWorker._draft_prompt_embeds); the plain program does that.
    view.multimodal_inputs = [None] * bs
    view.return_logprob = False
    view.spec_info = None
    view.chunked_req = None
    view.chunked_req_next_prompt_token = None
    return view


@compiled_op()
def gather_last_rows(
    draft_ids: torch.Tensor, target_hidden: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pass 2's [rows] draft input ids and [rows, H] target hidden, from pass 1's."""
    return draft_ids.index_select(0, rows), target_hidden.index_select(0, rows)
