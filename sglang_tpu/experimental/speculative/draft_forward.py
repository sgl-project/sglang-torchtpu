"""One draft forward over a slice of the verify rows, shared by the block and chain drafters."""

from __future__ import annotations

import torch

from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.experimental.speculative.spec_types import (
    DraftBlockInputs,
    TpuSpecVerifyInput,
)


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
) -> LogitsProcessorOutput:
    """Draft forward of width rows per request, offset rows past seq_lens, at the batch bucket.

    Writes draft KV into the verify slots of those rows; inputs come from
    build_draft_block_inputs / build_chain_block_inputs.
    """
    padded_bs = req_pool_indices.shape[0]
    bs = verify_input.seq_lens_cpu.shape[0]
    device = req_pool_indices.device
    # Padded rows carry kv len 0, so RPA keeps them inactive.
    kv_lens_cpu, kv_lens_sum = verify_input.padded_kv_lens_after_rows(offset + width)
    num_live_tokens = bs * width
    num_token_non_padded = (
        torch.tensor(num_live_tokens, dtype=torch.int32, device=device)
        if enable_num_token_non_padded()
        else None
    )
    num_pad = padded_bs - bs
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.TARGET_VERIFY,
        out_cache_loc_is_physical=True,
        batch_size=padded_bs,
        input_ids=inputs.input_ids,
        req_pool_indices=req_pool_indices,
        req_pool_indices_cpu=verify_input.req_pool_indices_cpu,
        seq_lens=inputs.seq_lens,
        out_cache_loc=inputs.cache_locs,
        seq_lens_sum=kv_lens_sum,
        seq_lens_cpu=kv_lens_cpu,
        positions=inputs.positions,
        input_embeds=None,
        spec_algorithm=spec_algorithm,
        spec_info=spec_info,
        capture_hidden_mode=capture_hidden_mode,
        # Pallas RPA plans the draft rows as an extend from host lengths.
        extend_prefix_lens=inputs.extend_prefix_lens,
        extend_seq_lens=inputs.extend_seq_lens,
        extend_prefix_lens_cpu=(verify_input.seq_lens_cpu + offset).tolist()
        + [0] * num_pad,
        extend_seq_lens_cpu=[width] * bs + [0] * num_pad,
        extend_num_tokens=num_live_tokens,
        # Only the local field, as in the warmup batch: direct_compile pins
        # Tensor-vs-None per input.
        num_token_non_padded=num_token_non_padded,
        global_num_token_non_padded_cpu=num_live_tokens,
    )
    set_bucket_dp_metadata(
        forward_batch,
        runner=draft_runner,
        num_tokens=padded_bs * width,
        bs=padded_bs,
        mode=ForwardMode.TARGET_VERIFY,
    )
    # Dynamo guards grad mode; the TPU warmup traced under no_grad.
    with torch.no_grad():
        return draft_runner.forward(forward_batch).logits_output
