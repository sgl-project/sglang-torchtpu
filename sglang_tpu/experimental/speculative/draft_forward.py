"""One draft forward over a slice of the verify rows, shared by the block and chain drafters."""

from __future__ import annotations

import torch

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    LogitsOutputMode,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.experimental.speculative.spec_types import (
    DraftBlockInputs,
    TpuSpecVerifyInput,
    VerifyRows,
)
from sglang_tpu.host_upload import device_scalar


def set_bucket_dp_metadata(
    forward_batch: ForwardBatch,
    *,
    runner: ModelRunner,
    num_tokens: int,
    bs: int,
    mode: ForwardMode,
) -> None:
    """Under DP attention, give a forward at the agreed batch bucket the DP metadata warmup traced.

    The scheduler's per-rank counts are live rows; the worker runs bucket rows,
    so the MLP sync would otherwise pad past them.
    """
    if not is_dp_attention_enabled():
        return
    compile_policy = runner.compile_policy
    assert isinstance(compile_policy, DecoderTpuCompilePolicy)
    compile_policy.set_dp_attention_metadata(forward_batch, num_tokens, bs, mode)


def build_draft_rows_forward_batch(
    *,
    draft_runner: ModelRunner,
    spec_algorithm: SpeculativeAlgorithm,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor | None,
    cache_locs: torch.Tensor | None,
    seq_lens: torch.Tensor,
    extend_prefix_lens: torch.Tensor | None,
    extend_seq_lens: torch.Tensor | None,
    req_pool_indices: torch.Tensor,
    verify_input: TpuSpecVerifyInput,
    offset: int,
    width: int,
    spec_info: DFlashVerifyInput,
    capture_hidden_mode: CaptureHiddenMode,
    publish_dsa_topk: bool = False,
) -> ForwardBatch:
    """Build the bucketed TARGET_VERIFY ForwardBatch for draft block/chain forwards."""
    padded_bs = req_pool_indices.shape[0]
    bs = verify_input.seq_lens_cpu.shape[0]
    device = req_pool_indices.device
    # Padded rows carry kv len 0, so RPA keeps them inactive.
    kv_lens_cpu, kv_lens_sum = verify_input.padded_kv_lens_after_rows(offset + width)
    num_live_tokens = bs * width
    # This runs right after the verify launch, so a blocking upload would wait
    # for the whole verify; the cached scalar needs no upload at all.
    num_token_non_padded = (
        device_scalar(num_live_tokens, torch.int32, device)
        if enable_num_token_non_padded()
        else None
    )
    num_pad = padded_bs - bs
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        out_cache_loc_is_physical=True,
        batch_size=padded_bs,
        input_ids=input_ids,
        req_pool_indices=req_pool_indices,
        req_pool_indices_cpu=verify_input.req_pool_indices_cpu,
        seq_lens=seq_lens,
        out_cache_loc=cache_locs,
        seq_lens_sum=kv_lens_sum,
        seq_lens_cpu=kv_lens_cpu,
        positions=positions,
        input_embeds=None,
        spec_algorithm=spec_algorithm,
        spec_info=spec_info,
        capture_hidden_mode=capture_hidden_mode,
        # Pallas RPA plans the draft rows as an extend from host lengths.
        extend_prefix_lens=extend_prefix_lens,
        extend_seq_lens=extend_seq_lens,
        extend_prefix_lens_cpu=(verify_input.seq_lens_cpu + offset).tolist()
        + [0] * num_pad,
        extend_seq_lens_cpu=[width] * bs + [0] * num_pad,
        extend_num_tokens=num_live_tokens,
        # Only the local field, as in the warmup batch: direct_compile pins
        # Tensor-vs-None per input.
        num_token_non_padded=num_token_non_padded,
        global_num_token_non_padded_cpu=num_live_tokens,
        # Handed no shared top-k, a reusing NextN layer runs its indexer and sets
        # spec_info.dsa_topk_indices (IndexTopKShareState.publish).
        reuse_dsa_topk_indices=publish_dsa_topk,
    )
    set_bucket_dp_metadata(
        forward_batch,
        runner=draft_runner,
        num_tokens=padded_bs * width,
        bs=padded_bs,
        mode=ForwardMode.TARGET_VERIFY,
    )
    return forward_batch


def forward_draft_rows(
    *,
    draft_runner: ModelRunner,
    spec_algorithm: SpeculativeAlgorithm,
    inputs: DraftBlockInputs,
    req_pool_indices: torch.Tensor,
    verify_input: TpuSpecVerifyInput,
    offset: int,
    width: int,
    spec_info: DFlashVerifyInput,
    capture_hidden_mode: CaptureHiddenMode,
    publish_dsa_topk: bool = False,
) -> LogitsProcessorOutput:
    """Draft forward of width rows per request, offset rows past seq_lens, at the batch bucket.

    Writes draft KV into the verify slots of those rows; inputs come from
    build_draft_block_inputs / build_chain_block_inputs. With publish_dsa_topk a
    DSA NextN draft leaves its indexer top-k rows in spec_info.dsa_topk_indices.
    """
    forward_batch = build_draft_rows_forward_batch(
        draft_runner=draft_runner,
        spec_algorithm=spec_algorithm,
        input_ids=inputs.input_ids,
        positions=inputs.positions,
        cache_locs=inputs.cache_locs,
        seq_lens=inputs.seq_lens,
        extend_prefix_lens=inputs.extend_prefix_lens,
        extend_seq_lens=inputs.extend_seq_lens,
        req_pool_indices=req_pool_indices,
        verify_input=verify_input,
        offset=offset,
        width=width,
        spec_info=spec_info,
        capture_hidden_mode=capture_hidden_mode,
        publish_dsa_topk=publish_dsa_topk,
    )
    # Dynamo guards grad mode; the TPU warmup traced under no_grad.
    with torch.no_grad():
        return draft_runner.forward(forward_batch).logits_output


def prep_and_draft_rows(
    *,
    draft_runner: ModelRunner,
    spec_algorithm: SpeculativeAlgorithm,
    packed_addressing: torch.Tensor,
    bonus_tokens: torch.Tensor,
    state_seq_lens: torch.Tensor,
    live_mask: torch.Tensor,
    req_pool_indices: torch.Tensor,
    verify_input: TpuSpecVerifyInput,
    num_verify_rows: int,
    mask_token_id: int,
    width: int,
    has_swa: bool,
    spec_info: DFlashVerifyInput,
) -> tuple[
    VerifyRows,
    VerifyRows | None,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor] | None,
    torch.Tensor,
]:
    """Fused Launch 1: unpack verify addressing + draft block forward in one dispatch."""
    from sglang_tpu.experimental.compilation.block_compile_policy import (
        TpuBlockSpecDecodeCompilePolicyMixin,
    )

    compile_policy = draft_runner.compile_policy
    assert isinstance(compile_policy, TpuBlockSpecDecodeCompilePolicyMixin)
    padded_bs = req_pool_indices.shape[0]
    bs = verify_input.seq_lens_cpu.shape[0]
    num_live_tokens = bs * width
    forward_batch = build_draft_rows_forward_batch(
        draft_runner=draft_runner,
        spec_algorithm=spec_algorithm,
        input_ids=None,
        positions=None,
        cache_locs=None,
        seq_lens=state_seq_lens,
        extend_prefix_lens=None,
        extend_seq_lens=None,
        req_pool_indices=req_pool_indices,
        verify_input=verify_input,
        offset=0,
        width=width,
        spec_info=spec_info,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    # Tensors are already bucket-padded by build_draft_rows_forward_batch; record
    # logical row bounds directly instead of calling prepare_forward_batch
    # (which assumes non-None input_ids).
    setattr(forward_batch, "_tpu_real_b", bs)
    setattr(forward_batch, "_tpu_real_q", num_live_tokens)
    spec_info.record_padding(
        logical_batch_size=bs,
        physical_batch_size=padded_bs,
        logical_num_tokens=num_live_tokens,
        physical_num_tokens=padded_bs * width,
    )
    forward_batch.logits_output_mode = LogitsOutputMode.FULL_VOCAB
    forward_batch.is_prefill_only = False
    draft_runner.attn_backend.init_forward_metadata(forward_batch)
    with (
        forward_context(
            ForwardContext(attn_backend=draft_runner.attn_backend),
            forward_batch=forward_batch,
        ),
        torch.no_grad(),
    ):
        return compile_policy.run_prep_and_draft(
            forward_batch=forward_batch,
            packed_addressing=packed_addressing,
            bonus_tokens=bonus_tokens,
            state_seq_lens=state_seq_lens,
            live_mask=live_mask,
            num_verify_rows=num_verify_rows,
            mask_token_id=mask_token_id,
            draft_width=width,
            has_swa=has_swa,
        )
