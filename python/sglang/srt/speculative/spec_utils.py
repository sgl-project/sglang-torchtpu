from __future__ import annotations

import contextlib
import functools
import logging
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Tuple

import torch
import triton
import triton.language as tl
from huggingface_hub import snapshot_download

# V2 (triton) path: re-export upstream's maintained kernels from cache_locs.
from sglang.kernels.ops.sampling import softmax as sampling_softmax
from sglang.kernels.ops.speculative.cache_locs import (
    align_evict_mask_to_page_size as align_evict_mask_to_page_size,
)
from sglang.kernels.ops.speculative.cache_locs import (
    assign_draft_cache_locs_contiguous as assign_draft_cache_locs_contiguous,
)
from sglang.kernels.ops.speculative.cache_locs import (
    assign_extend_cache_locs as assign_extend_cache_locs,
)
from sglang.kernels.ops.speculative.cache_locs import (
    filter_finished_cache_loc_kernel as filter_finished_cache_loc_kernel,
)
from sglang.kernels.ops.speculative.cache_locs import (
    generate_draft_decode_kv_indices as generate_draft_decode_kv_indices,
)
from sglang.kernels.ops.speculative.cache_locs import (
    get_target_cache_loc as get_target_cache_loc,
)

# Non-TPU backends (triton / CPU) dispatch through the kernel-side ``*_func``
# wrappers; the ``*_func`` defined below add the TPU torch-native path on top.
from sglang.kernels.ops.speculative.eagle import (
    fill_accept_out_cache_loc_func as _fill_accept_out_cache_loc_func,
)
from sglang.kernels.ops.speculative.eagle import fill_bonus_tokens as fill_bonus_tokens
from sglang.kernels.ops.speculative.eagle import (
    fill_bonus_tokens_func as _fill_bonus_tokens_func,
)
from sglang.kernels.ops.speculative.gather_spec_extras import gather_spec_extras
from sglang.kernels.ops.speculative.row_argmax import row_argmax
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.constrained.base_grammar_backend import GrammarMask
from sglang.srt.distributed.parallel_state import (
    get_self_pp_group,
    patch_pipeline_parallel_group,
    patch_tensor_parallel_group,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.utils import pp_spec_stable_rows_enabled
from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs
from sglang.srt.mem_cache.allocation import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sglang.srt.mem_cache.allocation import (
    assign_req_to_token_pool as assign_req_to_token_pool,
)
from sglang.srt.mem_cache.allocation import (
    assign_req_to_token_pool_func as _assign_req_to_token_pool_func,
)
from sglang.srt.mem_cache.allocation import (
    get_last_loc,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_spec,
    mamba_track_grid,
    max_speculative_num_draft_tokens,
)
from sglang.srt.utils import (
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_tpu,
    is_xpu,
    next_power_of_2,
)
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.utils.common import fast_topk
from sglang.srt.utils.nvtx_utils import profile_range

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()
_is_tpu = is_tpu()
_is_xpu = is_xpu()
_is_cpu = is_cpu()
# TPU/XLA cannot run the triton kernels; V1 dispatches to torch-native impls.
_use_torch_native = _is_tpu

if TYPE_CHECKING:
    from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


if _is_cpu:
    from sgl_kernel import assign_extend_cache_locs_cpu


logger = logging.getLogger(__name__)

HostTensor = torch.Tensor | Callable[[], torch.Tensor] | None


def _gather_spec_extras_torch(
    indices: torch.Tensor,
    topk_p_buf: torch.Tensor,
    topk_index_buf: torch.Tensor,
    output_tokens_buf: torch.Tensor,
    hidden_states_buf: Optional[torch.Tensor],
):
    row_indices = indices.to(dtype=torch.int64)
    return (
        torch.index_select(topk_p_buf, 0, row_indices),
        torch.index_select(topk_index_buf, 0, row_indices),
        torch.index_select(output_tokens_buf, 0, row_indices),
        (
            torch.index_select(hidden_states_buf, 0, row_indices)
            if hidden_states_buf is not None
            else None
        ),
    )


def gather_spec_extras_func(
    indices: torch.Tensor,
    topk_p_buf: torch.Tensor,
    topk_index_buf: torch.Tensor,
    output_tokens_buf: torch.Tensor,
    hidden_states_buf: Optional[torch.Tensor],
):
    if _use_torch_native:
        return _gather_spec_extras_torch(
            indices,
            topk_p_buf,
            topk_index_buf,
            output_tokens_buf,
            hidden_states_buf,
        )
    return gather_spec_extras(
        indices,
        topk_p_buf,
        topk_index_buf,
        output_tokens_buf,
        hidden_states_buf,
    )


def _route_to_req_to_token_device(
    req_to_token: torch.Tensor,
    device_tensor: torch.Tensor,
    host_tensor: HostTensor,
) -> torch.Tensor:
    """Use an existing host shadow when the request table is host-resident."""
    if req_to_token.device.type != "cpu" or host_tensor is None:
        return device_tensor
    return host_tensor() if callable(host_tensor) else host_tensor


def _assign_req_to_token_pool_torch(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
) -> None:
    """Pure-Torch request-table update for TPU/XLA."""
    if batch_size == 0:
        return
    device = req_to_token.device
    req_pool_indices = req_pool_indices[:batch_size].to(device)
    start_offset = start_offset[:batch_size].to(device)
    end_offset = end_offset[:batch_size].to(device)
    out_cache_loc = out_cache_loc.to(device)
    lengths = end_offset - start_offset
    max_len = min(
        int(req_to_token.shape[1]),
        int(out_cache_loc.numel()) if out_cache_loc.numel() > 0 else 0,
    )
    if max_len == 0:
        return
    offsets = torch.arange(max_len, device=device, dtype=start_offset.dtype)
    mask = offsets[None, :] < lengths[:, None]
    src_starts = torch.cumsum(lengths, dim=0) - lengths
    src_index = (src_starts[:, None] + offsets[None, :]).clamp(
        max=out_cache_loc.numel() - 1
    )
    dst_index = start_offset[:, None] + offsets[None, :]
    row_index = req_pool_indices[:, None].expand(batch_size, max_len)
    dst_index = torch.where(mask, dst_index, torch.zeros_like(dst_index))
    row_index = torch.where(mask, row_index, torch.zeros_like(row_index))
    values = torch.where(
        mask,
        out_cache_loc[src_index].to(req_to_token.dtype),
        req_to_token[row_index, dst_index],
    )
    req_to_token[row_index, dst_index] = values


def assign_req_to_token_pool_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
) -> None:
    if _use_torch_native:
        _assign_req_to_token_pool_torch(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            batch_size,
        )
        return
    _assign_req_to_token_pool_func(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        batch_size,
    )


def allocate_and_assign_decode_cache(
    *,
    tree_cache,
    req_to_token_pool,
    req_pool_indices: torch.Tensor,
    cur_kv_lens_cpu: torch.Tensor,
    nxt_kv_lens_cpu: torch.Tensor,
    num_needed_tokens: int,
    batch_size: int,
    device,
    host_req_pool_indices: HostTensor = None,
) -> torch.Tensor:
    """Allocate SpecV2 decode slots and update the request page table."""
    req_to_token = req_to_token_pool.req_to_token
    host_resident = req_to_token_pool.host_resident
    if host_resident:
        rows = _route_to_req_to_token_device(
            req_to_token, req_pool_indices, host_req_pool_indices
        )
        cur_kv_lens = cur_kv_lens_cpu
        nxt_kv_lens = nxt_kv_lens_cpu
    else:
        rows = req_pool_indices
        cur_kv_lens = cur_kv_lens_cpu.to(device=device)
        nxt_kv_lens = nxt_kv_lens_cpu.to(device=device)

    allocator = tree_cache.token_to_kv_pool_allocator
    if allocator.page_size == 1:
        out_cache_loc = alloc_token_slots(tree_cache, num_needed_tokens)
        out_cache_loc_host = None
    else:
        out_cache_loc_host = (
            torch.empty(num_needed_tokens, dtype=torch.int32, device="cpu")
            if host_resident
            else None
        )
        last_loc = get_last_loc(req_to_token, rows, cur_kv_lens)
        out_cache_loc = alloc_paged_token_slots_extend(
            tree_cache,
            cur_kv_lens,
            cur_kv_lens_cpu,
            nxt_kv_lens,
            nxt_kv_lens_cpu,
            last_loc,
            num_needed_tokens,
            out_cache_loc_host=out_cache_loc_host,
        )

    assign_locs = out_cache_loc
    if host_resident:
        assign_locs = (
            out_cache_loc.cpu() if out_cache_loc_host is None else out_cache_loc_host
        )
    assign_req_to_token_pool_func(
        rows,
        req_to_token,
        cur_kv_lens,
        nxt_kv_lens,
        assign_locs,
        batch_size,
    )
    return out_cache_loc


def _assign_draft_cache_locs_contiguous_torch(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    topk: int,
    speculative_num_steps: int,
    *,
    host_req_pool_indices: HostTensor = None,
    host_seq_lens: HostTensor = None,
) -> None:
    """Torch-native draft-write slot gather for a dense chain."""
    bs = seq_lens.shape[0]
    if bs == 0:
        return
    rows = _route_to_req_to_token_device(
        req_to_token, req_pool_indices, host_req_pool_indices
    ).to(req_to_token.device, dtype=torch.int64)
    starts = _route_to_req_to_token_device(req_to_token, seq_lens, host_seq_lens).to(
        req_to_token.device, dtype=torch.int64
    )
    width = topk * speculative_num_steps
    offsets = torch.arange(width, device=req_to_token.device, dtype=torch.int64)
    gathered = req_to_token[rows[:, None], starts[:, None] + offsets[None, :]]
    out_cache_loc.view(bs, width).copy_(
        gathered.to(out_cache_loc.device, dtype=out_cache_loc.dtype)
    )


def assign_draft_cache_locs_contiguous_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    topk: int,
    speculative_num_steps: int,
    *,
    host_req_pool_indices: HostTensor = None,
    host_seq_lens: HostTensor = None,
    positions: Optional[torch.Tensor] = None,
    mrope: Optional[torch.Tensor] = None,
) -> None:
    """Gather the draft-write slots for a dense chain.

    ``positions`` / ``mrope`` are CUDA-only outputs the triton kernel also
    writes (the draft positions of a chain); the torch-native path never gets
    them since its callers do not prepare them off-device.
    """
    bs = seq_lens.shape[0]
    if _use_torch_native:
        assert positions is None and mrope is None
        _assign_draft_cache_locs_contiguous_torch(
            req_pool_indices,
            req_to_token,
            seq_lens,
            out_cache_loc,
            topk,
            speculative_num_steps,
            host_req_pool_indices=host_req_pool_indices,
            host_seq_lens=host_seq_lens,
        )
        return
    assign_draft_cache_locs_contiguous[(bs,)](
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc,
        req_to_token.shape[1],
        topk,
        speculative_num_steps,
        positions=positions,
        mrope=mrope,
        BS=bs,
        WRITE_POSITIONS=positions is not None,
        WRITE_MROPE=mrope is not None,
    )


def assign_extend_cache_locs_kernel_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> None:
    """Fixed-shape ragged gather used for verify and accepted-slot moves."""
    bs = req_pool_indices.shape[0]
    if _use_torch_native:
        if bs == 0:
            return
        r2t_device = req_to_token.device
        out_device = out_cache_loc.device
        cap = out_cache_loc.numel()
        starts = start_offset.to(r2t_device, dtype=torch.int64)
        ends = end_offset.to(r2t_device, dtype=torch.int64)
        rows = req_pool_indices.to(r2t_device, dtype=torch.int64)
        lengths = (ends - starts).clamp(min=0)
        inclusive_ends = torch.cumsum(lengths, dim=0)
        exclusive_starts = inclusive_ends - lengths
        positions = torch.arange(cap, device=r2t_device, dtype=torch.int64)
        segments = (
            (positions[:, None] >= inclusive_ends[None, :])
            .to(torch.int64)
            .sum(dim=1)
            .clamp(max=bs - 1)
        )
        columns = (starts[segments] + positions - exclusive_starts[segments]).clamp(
            min=0, max=req_to_token.shape[1] - 1
        )
        gathered = req_to_token[rows[segments], columns].to(
            out_device, dtype=out_cache_loc.dtype
        )
        valid = torch.arange(cap, device=out_device) < inclusive_ends[-1].to(out_device)
        out_cache_loc.copy_(torch.where(valid, gathered, out_cache_loc))
        return
    if _is_cpu:
        assign_extend_cache_locs_cpu(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
        )
        return
    assign_extend_cache_locs[(bs,)](
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],
        next_power_of_2(bs),
    )


def assign_extend_cache_locs_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device,
    *,
    host_req_pool_indices: HostTensor = None,
    host_seq_lens: HostTensor = None,
) -> torch.Tensor:
    """Allocate and gather verify cache locations on the supported backend."""
    dtype = torch.int32 if _is_npu else torch.int64
    out = torch.zeros(batch_size * draft_token_num, dtype=dtype, device=device)
    if _is_npu:
        torch.ops.npu.cache_loc_update(
            req_pool_indices, req_to_token, start_offset, end_offset, out
        )
        return out
    if _use_torch_native:
        req_pool_indices = _route_to_req_to_token_device(
            req_to_token, req_pool_indices, host_req_pool_indices
        )
        start_offset = _route_to_req_to_token_device(
            req_to_token, start_offset, host_seq_lens
        )
        end_offset = start_offset + draft_token_num
    assign_extend_cache_locs_kernel_func(
        req_pool_indices, req_to_token, start_offset, end_offset, out
    )
    return out


def fill_bonus_tokens_func(
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,
    num_draft_tokens: int,
) -> None:
    bs = accept_lens.shape[0]
    if _use_torch_native:
        if bs == 0:
            return
        indices = (
            torch.arange(bs, device=accept_lens.device, dtype=torch.int64)
            * num_draft_tokens
            + accept_lens.to(torch.int64)
            - 1
        )
        bonus_tokens.copy_(accept_tokens.reshape(-1)[indices].to(bonus_tokens.dtype))
        return
    _fill_bonus_tokens_func(
        accept_tokens, accept_lens, bonus_tokens, num_draft_tokens, bs
    )


def fill_accept_out_cache_loc_func(
    accept_index: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_out_cache_loc: torch.Tensor,
) -> None:
    """Compact accepted source slots to the front, preserving order."""
    size = accept_index.numel()
    if _use_torch_native:
        if size == 0:
            return
        flat = accept_index.reshape(-1).to(torch.int64)
        mask = flat != -1
        mask_i = mask.to(torch.int64)
        destinations = torch.cumsum(mask_i, dim=0) - mask_i
        values = out_cache_loc.reshape(-1)[flat.clamp(min=0)].to(
            accept_out_cache_loc.dtype
        )
        discard = torch.full_like(destinations, size)
        buffer = torch.zeros(
            size + 1,
            dtype=accept_out_cache_loc.dtype,
            device=accept_out_cache_loc.device,
        )
        buffer.scatter_(0, torch.where(mask, destinations, discard), values)
        accept_out_cache_loc.copy_(buffer[:size])
        return
    _fill_accept_out_cache_loc_func(
        accept_index,
        out_cache_loc,
        accept_out_cache_loc,
        size,
    )


def resolve_num_tokens_per_req(
    *,
    phase: Literal["draft_decode", "draft_extend", "target_verify"],
    spec_algorithm=None,
    is_draft_worker: bool = False,
    num_draft_tokens: Optional[int] = None,
) -> int:
    """Single static derivation point for a spec phase's per-request token
    width (sizes capture shapes / buffers); the per-forward dynamic width
    lives on ``SpecInput.num_tokens_per_req``. Draft phases are
    EAGLE-family-only; "target_verify" is algorithm-generic via the hook.

    The widths come from the bags: adaptive spec captures each candidate step
    config with that config's leaves overridden, so the buffers being sized
    must follow the override rather than the startup values.
    """
    spec = get_spec()
    if phase == "draft_decode":
        return spec.speculative_eagle_topk
    if phase == "draft_extend":
        return spec.speculative_num_draft_tokens
    if phase == "target_verify":
        if num_draft_tokens is None:
            num_draft_tokens = spec.speculative_num_draft_tokens
        return spec_algorithm.get_num_tokens_per_req_for_target_verify(
            num_draft_tokens, is_draft_worker
        )
    raise ValueError(f"Unknown speculative phase: {phase}")


def fast_sample(probs: torch.Tensor, num_samples: int = 1):
    """Draw from `probs` via the Gumbel-max trick: argmax(probs / Exp(1)).

    Distributionally equivalent to torch.multinomial, but avoids multinomial's
    device-side distribution-validity assert, which the draft CUDA graph would
    otherwise capture and replay every step. q is clamped off zero so a zero
    draw can't yield inf/NaN scores that argmax would wrongly select; fp32
    avoids bf16 argmax ties biasing the draw. Set SGLANG_OPT_USE_GUMBEL_SAMPLE=0
    to fall back to torch.multinomial.
    """
    if not envs.SGLANG_OPT_USE_GUMBEL_SAMPLE.get():
        sample_index = torch.multinomial(probs, num_samples=num_samples)
        return probs.gather(1, sample_index), sample_index
    q = torch.empty_like(probs, dtype=torch.float32).exponential_(1.0)
    q.clamp_min_(torch.finfo(torch.float32).tiny)
    scores = probs.float() / q
    if num_samples == 1:
        sample_index = row_argmax(scores).unsqueeze(-1)
    else:
        sample_index = scores.topk(num_samples, dim=-1).indices
    sample_p = probs.gather(1, sample_index)
    return sample_p, sample_index


def renorm_draft_probs(
    next_token_logits: torch.Tensor,
    sampling_info,
    use_rejection_sampling: bool,
) -> torch.Tensor:
    """Draft-side next-token distribution.

    Plain softmax, except under rejection sampling where logits are
    temperature-scaled so the draft proposal q tracks the target sampling
    temperature (higher acceptance; correctness holds for any q).
    """
    if not use_rejection_sampling or not next_token_logits.size(0):
        return sampling_softmax(next_token_logits)
    return sampling_softmax(next_token_logits, temperatures=sampling_info.temperatures)


def sample_draft_proposal(
    next_token_logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: Optional[torch.Tensor] = None,
):
    """Leviathan draft proposal: q = softmax(logits / T), X ~ q.

    Returns (q, q(X), X). The verify's accept test coin*q(X) < p(X) is unbiased
    only if q is exactly the distribution X was drawn from, so callers must hand
    the returned q (not a recomputed one) to the verify.

    A greedy row (``top_k == 1``) proposes its argmax instead. SamplingParams
    rewrites temperature 0 to ``temperature=1.0, top_k=1``, so T alone cannot
    tell a greedy request from a T=1 one, and sampling a sharp-but-not-
    degenerate distribution proposes a non-argmax token often enough to cost
    real accept length.

    That row's X is then not drawn from the q returned beside it, which the
    unbiasedness argument above otherwise rests on. It stays correct because
    eagle_sample renormalises the target by the same per-row ``top_ks`` before
    the accept test, so a greedy row's p is one-hot: X equal to the target
    argmax accepts (p(X) = 1), any other X rejects (p(X) = 0) and the residual
    (p - q)+ it resamples from is p itself. Both arms commit the target argmax,
    which is what greedy means. Drop that renorm and this stops holding.
    """
    probs = sampling_softmax(next_token_logits, temperatures=temperatures)
    topk_p, topk_index = fast_sample(probs, num_samples=1)
    if top_ks is not None:
        # Assert rather than skip on a device mismatch: a host-side top_ks would
        # make this correction silently vanish, and the symptom -- draft accept
        # length quietly dropping about 20% -- reads as a model problem, not a
        # plumbing one.
        assert top_ks.device == probs.device, (
            f"top_ks must be on {probs.device} to reach the draft proposal, "
            f"got {top_ks.device}; the caller has to carry the real per-request "
            "top_k, not a host placeholder"
        )
        greedy = (top_ks <= 1).view(-1, 1)
        topk_index = torch.where(greedy, probs.argmax(dim=-1, keepdim=True), topk_index)
        topk_p = probs.gather(1, topk_index)
    return probs, topk_p, topk_index


# Simulate acceptance length for benchmarking purposes
SIMULATE_ACC_LEN = envs.SGLANG_SIMULATE_ACC_LEN.get()  # turn off if < 0
SIMULATE_ACC_METHOD = envs.SGLANG_SIMULATE_ACC_METHOD.get()
SIMULATE_ACC_TOKEN_MODE = envs.SGLANG_SIMULATE_ACC_TOKEN_MODE.get()

TREE_TRAVERSE_TIME_THRESHOLD = 1  # TODO: set this properly
TREE_SPEC_KERNEL_AVAILABLE = (
    _is_cuda or _is_musa
)  # This kernel is only available for CUDA and MUSA now


def draft_kv_indices_buffer_width(
    num_seqs: int, topk: int, max_context_len: int
) -> int:
    """Per-step row width of the EAGLE draft-decode kv_indices buffer.

    num_seqs * topk branches each attend up to max_context_len KV slots; the topk
    factor is mandatory -- dropping it under-allocates and overflows the row (#27338, #27460).
    """
    assert num_seqs * topk * max_context_len < 2**31, (
        "kv_indices flat offset would overflow int32; reduce batch/topk/context"
    )
    return num_seqs * topk * max_context_len


def draft_kv_indices_used_len(
    seq_lens_sum: int, topk: int, bs: int, num_steps: int
) -> int:
    """kv_indices length used through num_steps draft-decode steps.

    bs = topk * num_seqs branches, one index appended per branch per step. Called with
    num_steps = i + 1 (per-step slice) and speculative_num_steps (capacity assert).
    """
    return seq_lens_sum * topk + bs * num_steps


def resolve_draft_decode_window(model_runner) -> Tuple[int, int]:
    """Resolve (window_size, sink_size) for generate_draft_decode_kv_indices.

    Returns (0, 0) -- full draft attention, the pristine read plan -- when
    --speculative-draft-window-size is unset, and also when the draft model
    already has a sliding window of its own: the index builder emits one KV list
    shared by every draft layer, so it cannot express a per-layer window, and the
    draft model's own window is authoritative -- whether it comes from the
    checkpoint or, for LlamaForCausalLMEagle3, from this same flag.
    """
    # Read through the resolving view: handle_speculative_decoding declares both
    # fields rather than assigning them, so the raw field holds the unvalidated input.
    cfg = resolving_view(model_runner.server_args)
    window_size = int(cfg.speculative_draft_window_size or 0)
    if window_size <= 0:
        return 0, 0
    # The runner's resolved window, not the raw config field: config keys
    # (sliding_window / window_size) are overloaded across model families, while
    # this is the same value the attention backends key their own SWA paths on.
    native_window = getattr(model_runner, "sliding_window_size", None)
    if native_window is not None and native_window > 0:
        # An equal window is the one that was asked for, applied per layer instead
        # of here (LlamaForCausalLMEagle3 routes this flag into its own window).
        if native_window != window_size:
            logger.warning(
                "Ignoring --speculative-draft-window-size=%d: this draft model has a "
                "sliding window of %d, which the attention backend applies per layer. "
                "Draft-decode windowing stays off.",
                window_size,
                native_window,
            )
        return 0, 0
    return window_size, int(cfg.speculative_draft_sink_size or 0)


def record_stream_each(tensors, stream):
    """Call record_stream(stream) on each cuda tensor in `tensors`, skipping
    non-tensor / non-cuda entries. Tells the caching allocator that the
    tensors are also used on `stream`, so memory is not recycled while
    queued work is still in flight after Python refs drop.
    """
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            t.record_stream(stream)


def record_stream_for_v2_verify(batch, verify_input, fwd_stream):
    """Mark pre-prepare SB / verify_input GPU tensors as used on `fwd_stream`.

    Spec V2 mutates SB mid-forward (`prepare_for_verify` rebinds
    `batch.input_ids` / `out_cache_loc`; `_draft_extend_for_decode` later
    replaces `batch.input_ids` again). Each rebind drops the only SB Python
    ref to the old tensor while the verify forward kernel may still be
    reading its memory on `fwd_stream`; `record_stream` tells the caching
    allocator to wait for `fwd_stream` before recycling the block.

    Covers pre-prepare tensors only; caller must also `record_stream_each`
    the post-prepare rebinds (new `batch.input_ids` / `out_cache_loc`).
    """
    candidates = [
        batch.seq_lens,
        batch.req_pool_indices,
        batch.input_ids,
        batch.out_cache_loc,
    ]
    if verify_input is not None:
        candidates.extend(
            [
                getattr(verify_input, attr, None)
                for attr in (
                    "draft_token",
                    "custom_mask",
                    "positions",
                    "prepared_out_cache_loc",
                    "prepared_mrope_positions",
                    "retrieve_index",
                    "retrieve_next_token",
                    "retrieve_next_sibling",
                )
            ]
        )
    record_stream_each(candidates, fwd_stream)


def spec_need_hidden_states() -> bool:
    # STANDALONE drafts don't consume `spec_info.hidden_states` (vanilla LLM).
    # multi_layer_eagle, DFLASH, and DSPARK don't relay hidden_states through FutureMap.
    # TODO(lsyin): also skip when step == 1.
    spec = get_spec()
    if spec.speculative_algorithm in ("STANDALONE", "DFLASH", "DSPARK"):
        return False
    return not spec.enable_multi_layer_eagle


@triton.jit
def create_extend_after_decode_spec_info(
    accept_tokens,
    seq_lens,
    accept_lens,
    positions,
    bonus_tokens_ptr,
    bs_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, bs_upper)
    seq_length = tl.load(seq_lens + pid)
    # `accept_lens` includes the bonus token; load this req's value.
    accept_len = tl.load(accept_lens + pid)

    accept_len_cumsum = tl.sum(
        tl.load(accept_lens + offsets, mask=offsets < pid, other=0)
    )
    positions_ptr = positions + accept_len_cumsum
    mask = offsets < accept_len
    tl.store(positions_ptr + offsets, seq_length - accept_len + offsets, mask)

    accept_len_cumsum += accept_len - 1
    bonus_token = tl.load(accept_tokens + accept_len_cumsum)
    tl.store(bonus_tokens_ptr + pid, bonus_token)


def _create_extend_after_decode_spec_info_torch(
    accept_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    positions: torch.Tensor,
    bonus_tokens: torch.Tensor,
):
    """Pure-PyTorch implementation of create_extend_after_decode_spec_info.

    For each request *pid*, compute:
      - positions[start:start+accept_len] = seq_len - accept_len .. seq_len - 1
      - bonus_tokens[pid] = accept_tokens[start + accept_len - 1]
    where start = sum(accept_lens[:pid]).
    """
    batch_size = seq_lens.numel()
    if batch_size == 0:
        return

    cumsum = torch.cumsum(accept_lens, dim=0)

    # bonus_tokens: last accepted token per request (vectorised, no collisions)
    bonus_idx = (cumsum - 1).clamp(min=0)
    bonus_tokens.copy_(accept_tokens[bonus_idx].to(bonus_tokens.dtype))

    total = positions.numel()
    if total == 0:
        return
    flat = torch.arange(total, device=positions.device, dtype=accept_lens.dtype)
    # Only the first sum(accept_lens) rows are real accepted tokens; the remaining rows
    # are padding up to the compiled token bucket.
    real_mask = flat < cumsum[-1]
    req_id = (flat[:, None] >= cumsum[None, :]).sum(dim=1).clamp(max=batch_size - 1)
    start_excl = cumsum - accept_lens  # exclusive prefix offset per request
    within = flat - start_excl[req_id]
    real_positions = seq_lens[req_id] - accept_lens[req_id] + within
    positions.copy_(
        torch.where(real_mask, real_positions, torch.zeros_like(real_positions)).to(
            positions.dtype
        )
    )


def create_extend_after_decode_spec_info_func(
    accept_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    positions: torch.Tensor,
    bonus_tokens: torch.Tensor,
    speculative_num_steps: int,
):
    if _use_torch_native:
        _create_extend_after_decode_spec_info_torch(
            accept_tokens,
            seq_lens,
            accept_lens,
            positions,
            bonus_tokens,
        )
        return

    create_extend_after_decode_spec_info[(len(seq_lens),)](
        accept_tokens,
        seq_lens,
        accept_lens,
        positions,
        bonus_tokens,
        next_power_of_2(max(speculative_num_steps + 1, len(seq_lens))),
    )


@triton.jit
def assign_draft_cache_locs(
    req_pool_indices,
    req_to_token,
    seq_lens,
    extend_lens,
    num_new_pages_per_topk,
    out_cache_loc,
    source_cache_loc,
    target_cache_loc,
    last_page_lens_cumsum,
    duplicate_cache_len: tl.constexpr,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
    page_size: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    if page_size == 1 or topk == 1:
        copy_len = topk * speculative_num_steps
        out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps
    else:
        bs_offset = tl.arange(0, bs_upper)
        copy_len = tl.load(extend_lens + pid)
        cum_copy_len = tl.sum(tl.load(extend_lens + bs_offset, mask=bs_offset < pid))
        out_cache_ptr = out_cache_loc + cum_copy_len

    # Part 1: Copy from out_cache_loc to req_to_token
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(out_cache_ptr + copy_offset, mask=mask)
        tl.store(token_pool + kv_start + copy_offset, data, mask=mask)
    # XXX (MUSA): Triton issue: chained boolean operators (A or B or C) are not supported.
    if (page_size != 1 and topk != 1) and duplicate_cache_len > 0:
        # Part 2: Copy indices into source_cache_loc and target_cache_loc
        # Expected output: src:[8,9,10,8,9,10...] tgt:[16,17,18,24,25,26...]
        prefix_len = tl.load(seq_lens + pid)
        last_page_len = prefix_len % page_size
        offsets = tl.arange(0, page_size)
        mask = offsets < last_page_len
        num_new_pages_per_topk_ = tl.load(num_new_pages_per_topk + pid)
        prefix_base = token_pool + prefix_len - last_page_len
        src_indices = tl.load(prefix_base + offsets, mask=mask)
        last_page_lens_cumsum_ = tl.load(last_page_lens_cumsum + pid)
        # Skip the first one since no copy is needed
        for topk_id in range(1, topk):
            tl.store(
                source_cache_loc
                + (topk - 1) * (last_page_lens_cumsum_ - last_page_len)
                + (topk_id - 1) * last_page_len
                + offsets,
                src_indices,
                mask=mask,
            )
            tgt_indices = tl.load(
                prefix_base + topk_id * num_new_pages_per_topk_ * page_size + offsets,
                mask=mask,
            )
            tl.store(
                target_cache_loc
                + (topk - 1) * (last_page_lens_cumsum_ - last_page_len)
                + (topk_id - 1) * last_page_len
                + offsets,
                tgt_indices,
                mask=mask,
            )
        # Part 3: Copy and remove the used indices for duplication
        # speculative_num_steps=5, page_size=4, num_new_pages_per_topk_=2, last_page_len=1
        #  - xxxxx .. | - xxxxx .. |
        #   topk=0        topk=1
        #  "-" means prefix tokens
        #  "x" means speculative draft tokens
        #  "." means padded tokens
        # we only want to copy the "x" part.
        iter_offset = tl.arange(0, iter_upper)
        for topk_id in range(topk):
            mask_upper = iter_offset < (speculative_num_steps + last_page_len)
            mask_lower = iter_offset >= last_page_len
            combined_mask = mask_upper & mask_lower
            indices = tl.load(
                prefix_base
                + topk_id * num_new_pages_per_topk_ * page_size
                + iter_offset,
                mask=combined_mask,
                other=0,
            )
            # Shift from previous batches
            ptr_offset = pid * speculative_num_steps * topk
            # Subtract last_page_len to fill the gap of duplicated last page tokens.
            # For example, token pool is (1, 2, 3, 4 ,5) and last page is 1,
            # we write 2, 3, 4 to the front of out_cache_loc.
            tl.store(
                out_cache_loc
                + ptr_offset
                + topk_id * speculative_num_steps
                - last_page_len
                + iter_offset,
                indices,
                mask=combined_mask,
            )


def _assign_draft_cache_locs_torch(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    num_new_pages_per_topk: torch.Tensor,
    out_cache_loc: torch.Tensor,
    source_cache_loc: torch.Tensor,
    target_cache_loc: torch.Tensor,
    last_page_lens_cumsum: torch.Tensor,
    duplicate_cache_len: int,
    pool_len: int,
    topk: int,
    speculative_num_steps: int,
    page_size: int,
    batch_size: int,
):
    """Pure-PyTorch implementation of assign_draft_cache_locs for TPU/XLA.

    Currently supports only page_size=1 or topk=1.
    """
    if batch_size == 0:
        return

    if page_size != 1 and topk != 1:
        raise NotImplementedError(
            "TPU assign_draft_cache_locs currently supports only "
            f"page_size=1 or topk=1, got {page_size=} {topk=}."
        )

    device = req_to_token.device
    copy_len = topk * speculative_num_steps

    req_pool_indices = req_pool_indices[:batch_size]
    seq_lens = seq_lens[:batch_size]
    out_cache_loc = out_cache_loc[: batch_size * copy_len]
    # Out-of-bounds safety: cols stay within req_to_token.shape[1] because
    # the KV pool allocator guarantees seq_lens + copy_len <= pool width.
    offsets = torch.arange(copy_len, device=device, dtype=seq_lens.dtype)
    rows = req_pool_indices.to(device)[:, None]  # (bs, 1)
    cols = seq_lens.to(device)[:, None] + offsets[None, :]  # (bs, copy_len)
    src = out_cache_loc.view(batch_size, copy_len).to(device).to(req_to_token.dtype)
    req_to_token[rows, cols] = src


def assign_draft_cache_locs_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    num_new_pages_per_topk: torch.Tensor,
    out_cache_loc: torch.Tensor,
    source_cache_loc: torch.Tensor,
    target_cache_loc: torch.Tensor,
    last_page_lens_cumsum: torch.Tensor,
    duplicate_cache_len: int,
    pool_len: int,
    topk: int,
    speculative_num_steps: int,
    page_size: int,
    batch_size: int,
):
    if _use_torch_native:
        _assign_draft_cache_locs_torch(
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            pool_len,
            topk,
            speculative_num_steps,
            page_size,
            batch_size,
        )
        return

    assign_draft_cache_locs[(batch_size,)](
        req_pool_indices,
        req_to_token,
        seq_lens,
        extend_lens,
        num_new_pages_per_topk,
        out_cache_loc,
        source_cache_loc,
        target_cache_loc,
        last_page_lens_cumsum,
        duplicate_cache_len,
        pool_len,
        topk,
        speculative_num_steps,
        page_size,
        next_power_of_2(batch_size),
        next_power_of_2(speculative_num_steps + page_size),
    )


@functools.lru_cache(maxsize=None)
def _draft_token_cols(num_draft_tokens: int, device: torch.device) -> torch.Tensor:
    """Cached ``[1, num_draft_tokens]`` arange (num_draft_tokens is static per run)."""
    return torch.arange(num_draft_tokens, device=device).unsqueeze(0)


def _align_evict_mask_to_page_size_torch(
    seq_lens: torch.Tensor,
    evict_mask: torch.Tensor,
    page_size: int,
    num_draft_tokens: int,
):
    """Vectorized torch equivalent of ``align_evict_mask_to_page_size``."""
    bs = seq_lens.shape[0]
    device = evict_mask.device
    mask2d = evict_mask.view(bs, num_draft_tokens)
    num_false = (~mask2d).sum(dim=1).to(seq_lens.dtype)  # accepted tokens / req
    start = (
        torch.div(seq_lens + num_false - 1, page_size, rounding_mode="floor")
        * page_size
        - seq_lens
    )
    lo = torch.clamp(start, min=0)
    hi = torch.clamp(start + page_size, max=num_draft_tokens)
    cols = _draft_token_cols(num_draft_tokens, device)
    keep = (cols >= lo.unsqueeze(1)) & (cols < hi.unsqueeze(1))
    mask2d &= ~keep


def align_evict_mask_to_page_size_func(
    seq_lens: torch.Tensor,
    evict_mask: torch.Tensor,
    page_size: int,
    num_draft_tokens: int,
):
    """Dispatch to the torch fallback on TPU, triton kernel otherwise."""
    if _use_torch_native:
        _align_evict_mask_to_page_size_torch(
            seq_lens, evict_mask, page_size, num_draft_tokens
        )
        return
    align_evict_mask_to_page_size[len(seq_lens),](
        seq_lens,
        evict_mask,
        page_size,
        num_draft_tokens,
        next_power_of_2(num_draft_tokens),
    )


@torch.compile(dynamic=True, disable=(_is_npu or _is_tpu))
def get_src_tgt_cache_loc(
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_index: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    draft_token_num: int,
    page_size: int,
):
    src_cache_loc = out_cache_loc[accept_index]
    # zeros_like, not empty_like: any uncovered tail stays at slot 0 (padding)
    # instead of caching-allocator garbage.
    tgt_cache_loc = torch.zeros_like(src_cache_loc)
    extended_len = seq_lens + draft_token_num
    keep_len = torch.minimum(
        (seq_lens + num_correct_drafts + 1 + page_size - 1) // page_size * page_size,
        extended_len,
    )
    to_free_num_slots = extended_len - keep_len
    return src_cache_loc, tgt_cache_loc, to_free_num_slots


@torch.compile(dynamic=True, disable=(_is_npu or _is_tpu or _is_xpu))
def create_num_accept_tokens_filter(
    num_correct_drafts: torch.Tensor,
    unfinished_index_device: torch.Tensor,
    seq_lens: torch.Tensor,
):
    num_accept_tokens_filter = torch.zeros_like(num_correct_drafts)
    num_accept_tokens_filter[unfinished_index_device] = (
        num_correct_drafts[unfinished_index_device] + 1
    )
    seq_lens.add_(num_correct_drafts + 1)
    return num_accept_tokens_filter


def _select_top_k_tokens_first(
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    topk: int,
):
    input_ids = topk_index.flatten()
    if hidden_states is not None:
        hidden_states = hidden_states.repeat_interleave(topk, dim=0)

    tree_info = (
        topk_p.unsqueeze(1),  # (b, 1, topk)
        topk_index,  # (b, topk)
        torch.arange(-1, topk, dtype=torch.long, device=input_ids.device).expand(
            topk_p.shape[0], -1
        ),  # (b, topk + 1) — expand avoids the allocation of repeat
    )
    return input_ids, hidden_states, topk_p, tree_info


@torch.compile(dynamic=True, disable=(_is_npu or _is_tpu or _is_xpu))
def _select_top_k_tokens_later(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    topk_sq = topk * topk

    expand_scores = scores.unsqueeze(2) * topk_p.view(-1, topk, topk)
    # (b, topk, 1) * (b, topk, topk) -> (b, topk, topk)

    topk_cs_p, topk_cs_index = fast_topk(
        expand_scores.flatten(start_dim=1), topk, dim=-1
    )  # (b, topk)

    topk_index = topk_index.view(-1, topk_sq)
    input_ids = torch.gather(topk_index, 1, topk_cs_index).flatten()

    if hidden_states is not None and hidden_states.shape[0] > 0:
        flat_cs = topk_cs_index.flatten()
        batch_offsets = torch.arange(
            0, hidden_states.shape[0], step=topk, device=flat_cs.device
        )
        selected_input_index = flat_cs // topk + batch_offsets.repeat_interleave(topk)
        hidden_states = hidden_states[selected_input_index]

    tree_info = (
        expand_scores,  # (b, topk, topk)
        topk_index,  # (b, topk * topk)
        topk_cs_index + (topk_sq * (i - 1) + topk),  # (b, topk)
    )
    return input_ids, hidden_states, topk_cs_p, tree_info


def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    if i == 0:
        return _select_top_k_tokens_first(topk_p, topk_index, hidden_states, topk)
    return _select_top_k_tokens_later(
        i, topk_p, topk_index, hidden_states, scores, topk
    )


def sample_simulated_acc_len(
    simulate_acc_len: float,
    simulate_acc_method: str,
    max_len: int,
) -> int:
    """Sample a simulated acceptance length in [1, max_len]."""
    if simulate_acc_method == "multinomial":
        simulated_values = torch.normal(
            mean=simulate_acc_len,
            std=1.0,
            size=(1,),
            device="cpu",
        )
        # clamp simulated values to be between 1 and max_len
        simulated_values = torch.clamp(simulated_values, min=1.0, max=max_len)
        simulate_acc_len = int(simulated_values.round().item())
    elif simulate_acc_method == "match-expected":
        # multinomial sampling does not match the expected length
        # we keep it for the sake of compatibility of existing tests
        # but it's better to use "match-expected" for the cases that need to
        # match the expected length, One caveat is that this will only sample
        # either round down or round up of the expected length
        simulate_acc_len = max(1.0, min(max_len, simulate_acc_len))
        lower = int(simulate_acc_len // 1)
        upper = lower + 1 if lower < max_len else lower
        if lower == upper:
            simulate_acc_len = lower
        else:
            weight_upper = simulate_acc_len - lower
            weight_lower = 1.0 - weight_upper
            probs = torch.tensor([weight_lower, weight_upper], device="cpu")
            sampled_index = torch.multinomial(probs, num_samples=1)
            simulate_acc_len = lower if sampled_index == 0 else upper
    else:
        raise ValueError(f"Invalid simulate_acc_method: {simulate_acc_method}")
    return int(simulate_acc_len)


def generate_simulated_accept_index(
    accept_index,
    predict,
    num_correct_drafts,
    candidates,
    target_predict,
    bs,
    spec_steps,
    simulate_acc_len: float = SIMULATE_ACC_LEN,
    simulate_acc_method: str = SIMULATE_ACC_METHOD,
    simulate_acc_token_mode: str = SIMULATE_ACC_TOKEN_MODE,
):
    use_real_draft_tokens = simulate_acc_token_mode == "real-draft-token"

    assert simulate_acc_len > 0.0
    simulate_acc_len = sample_simulated_acc_len(
        simulate_acc_len, simulate_acc_method, spec_steps + 1
    )

    accept_indx_first_col = accept_index[:, 0].view(-1, 1)
    sim_accept_index = torch.full(
        (bs, spec_steps + 1), -1, dtype=torch.int32, device=accept_index.device
    )
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + torch.arange(
        simulate_acc_len, device=accept_index.device
    )
    num_correct_drafts.fill_(simulate_acc_len - 1)

    if not use_real_draft_tokens:
        predict.fill_(100)  # some legit token id
        return sim_accept_index

    # Use the topk=1 draft chain for forced acceptance, then a target-derived bonus.
    if simulate_acc_len > 1:
        draft_node_indices = sim_accept_index[:, : simulate_acc_len - 1].long()
        predict[draft_node_indices] = candidates[:, 1:simulate_acc_len].to(
            dtype=predict.dtype
        )
    bonus_node_indices = sim_accept_index[:, simulate_acc_len - 1].long()
    predict[bonus_node_indices] = target_predict[:, simulate_acc_len - 1].to(
        dtype=predict.dtype
    )
    return sim_accept_index


def traverse_tree(
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    draft_tokens: torch.Tensor,
    grammar: BaseGrammarObject,
    allocate_token_bitmask: torch.Tensor,
    vocab_size: Optional[int] = None,
):
    """
    Traverse the tree constructed by the draft model to generate the logits mask.
    """
    assert (
        retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape
    )

    def dfs(
        curr: int,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        parent_pos: int,
    ):
        if curr == 0:
            # the first token generated by the target model, and thus it is always
            # accepted from the previous iteration
            is_accepted = True
        else:
            parent_bitmask = allocate_token_bitmask[parent_pos]
            current_token = draft_tokens[curr]
            if vocab_size and current_token >= vocab_size:
                is_accepted = False
            else:
                # 32 boolean bitmask values are packed into 32-bit integers
                is_accepted = (
                    parent_bitmask[current_token // 32] & (1 << (current_token % 32))
                ) != 0

        if is_accepted:
            if curr != 0:
                # Accept the current token
                grammar.accept_token(int(draft_tokens[curr]))
            if not grammar.is_terminated():
                # Generate the bitmask for the current token
                grammar.fill_vocab_mask(allocate_token_bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    # Visit the child node
                    dfs(
                        int(retrieve_next_token[curr]),
                        retrieve_next_token,
                        retrieve_next_sibling,
                        curr,
                    )

            if curr != 0:
                # Rollback the current token
                grammar.rollback(1)

        if retrieve_next_sibling[curr] != -1:
            # Visit the sibling node
            dfs(
                int(retrieve_next_sibling[curr]),
                retrieve_next_token,
                retrieve_next_sibling,
                parent_pos,
            )

    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)


def generate_token_bitmask(
    reqs: List[Req],
    retrieve_next_token_cpu: torch.Tensor,
    retrieve_next_sibling_cpu: torch.Tensor,
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
) -> Tuple[Optional[torch.Tensor], Optional[BaseGrammarObject]]:
    """
    Generate the logit mask for structured output.
    Draft model's token can be either valid or invalid with respect to the grammar.
    We need to perform DFS to
    1. figure out which tokens are accepted by the grammar.
    2. if so, what is the corresponding logit mask.
    """

    num_draft_tokens = draft_tokens_cpu.shape[-1]

    allocate_token_bitmask = None
    assert len(reqs) == retrieve_next_token_cpu.shape[0]
    grammar = None
    for i, req in enumerate(reqs):
        if req.grammar is not None:
            if allocate_token_bitmask is None:
                allocate_token_bitmask = req.grammar.allocate_vocab_mask(
                    vocab_size=vocab_size,
                    batch_size=draft_tokens_cpu.numel(),
                    device="cpu",
                )
            grammar = req.grammar
            s = time.perf_counter()
            traverse_tree(
                retrieve_next_token_cpu[i],
                retrieve_next_sibling_cpu[i],
                draft_tokens_cpu[i],
                req.grammar,
                allocate_token_bitmask[
                    i * num_draft_tokens : (i + 1) * num_draft_tokens
                ],
                vocab_size=vocab_size,
            )
            tree_traverse_time = time.perf_counter() - s
            if tree_traverse_time > TREE_TRAVERSE_TIME_THRESHOLD:
                logger.warning(
                    f"Bit mask generation took {tree_traverse_time} seconds with "
                    f"grammar: {req.grammar}"
                )

    return allocate_token_bitmask, grammar


class GrammarTree:
    """The verify tree the grammar bitmask is built over, on the host.

    ``from_device`` starts an async copy, so build it before the target verify
    launch; ``from_host`` is for algorithms that build the tree there (NGRAM).
    """

    def __init__(self, host: Tuple[torch.Tensor, ...], done_event):
        self._host = host
        self._done = done_event

    @classmethod
    def from_device(
        cls,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        draft_token: torch.Tensor,
    ) -> GrammarTree:
        # Function-local: avoids the schedule_batch -> overlap_utils ->
        # spec_utils import cycle.
        from sglang.srt.managers.utils import _async_d2h

        tensors = (retrieve_next_token, retrieve_next_sibling, draft_token)
        host = tuple(_async_d2h(t) for t in tensors)
        # Sources may be mixed -- an algorithm can synthesize part of the tree on
        # the host -- so the event has to key off whichever one is on device.
        device = next((t.device for t in tensors if t.device.type != "cpu"), None)
        if device is None:
            return cls(host, None)
        done = torch.get_device_module(device).Event()
        done.record()
        return cls(host, done)

    @classmethod
    def from_host(
        cls,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        draft_token: torch.Tensor,
    ) -> GrammarTree:
        return cls((retrieve_next_token, retrieve_next_sibling, draft_token), None)

    @classmethod
    def from_linear_chain(cls, verify_ids_2d: torch.Tensor) -> GrammarTree:
        """Degenerate tree for chain-verify algorithms: node i's only child is i + 1.

        ``verify_ids_2d`` is (bs, chain_len) with column 0 the already-committed
        token, so mask rows line up with the target's logits rows one-for-one.
        Only the ids need a copy; the links are fixed by the shape.
        """
        bs, chain_len = verify_ids_2d.shape
        next_token = torch.full((bs, chain_len), -1, dtype=torch.int64)
        next_token[:, :-1] = torch.arange(1, chain_len, dtype=torch.int64)
        next_sibling = torch.full((bs, chain_len), -1, dtype=torch.int64)
        return cls.from_device(next_token, next_sibling, verify_ids_2d)

    def resolve(self) -> Tuple[torch.Tensor, ...]:
        if self._done is not None:
            self._done.synchronize()
        return self._host


def build_grammar_vocab_mask(
    *,
    reqs: List[Req],
    tree: GrammarTree,
    sampling_info: SamplingBatchInfo,
    device,
    barrier: Optional[Callable[[], None]],
) -> Optional[GrammarMask]:
    """Build the constrained-decoding bitmask over a verify tree and stage it on device.

    Call it after the target verify launch -- every step here is host work, so it all
    overlaps that forward. ``barrier`` advances the previous batch's FSM over its
    committed tokens, which the traversal then reads, so it has to run first.
    """
    if barrier is not None:
        barrier()
    vocab_mask, grammar = generate_token_bitmask(
        reqs,
        *tree.resolve(),
        sampling_info.vocab_size,
    )
    if vocab_mask is None:
        return None

    # non_blocking is safe: the bitmask is pinned (see xgrammar_backend), and stream
    # order keeps the copy ahead of the sampler's apply_vocab_mask.
    vocab_mask = vocab_mask.to(device, non_blocking=True)
    # Otherwise the extend stage's leftover mask is applied instead.
    sampling_info.grammar_mask = None
    return GrammarMask(grammar, vocab_mask)


def load_token_map(token_map_path: str) -> List[int]:
    if not os.path.exists(token_map_path):
        repo_id = os.path.dirname(token_map_path)
        file_name = os.path.basename(token_map_path)

        cache_dir = None
        if envs.SGLANG_USE_MODELSCOPE.get():
            from modelscope.utils.file_utils import get_model_cache_root

            cached_repo_path = os.path.join(get_model_cache_root(), repo_id)
            if os.path.exists(cached_repo_path):
                cache_dir = cached_repo_path

        if cache_dir is None:
            if envs.SGLANG_USE_MODELSCOPE.get():
                from modelscope.hub.snapshot_download import (
                    snapshot_download as download_func,
                )
            else:
                download_func = snapshot_download
            cache_dir = download_func(
                repo_id,
                ignore_patterns=["*.bin", "*.safetensors"],
            )

        token_map_path = os.path.join(cache_dir, file_name)
    hot_token_id = torch.load(token_map_path, weights_only=True)
    return torch.tensor(hot_token_id, dtype=torch.int64)


@contextmanager
def draft_pp_context():
    # The draft model is one layer and never spans pipeline stages; give it a
    # single-member pp group so it initializes as if pp were off.
    with patch_pipeline_parallel_group(get_self_pp_group()):
        yield


def draft_tp_context(owns_attention: bool):
    """Enter the TP placement that draft work runs under.

    Work that owns attention (an attention-owning draft, or DSpark's DP-MoE
    sync) runs on the target's attention-TP group; other draft work keeps the
    target's TP placement. Enter it from outside any draft scope, where
    ``attn_tp_group`` is still the target's.
    """
    if not owns_attention:
        return contextlib.nullcontext()
    return patch_tensor_parallel_group(
        get_parallel().attn_tp_group, owns_attention=True
    )


def spec_stage_span(name: str):
    """Profiler span for a coarse speculative-decoding stage (``draft`` /
    ``draft_extend`` / ``verify``).
    """
    return profile_range(name)


def move_accept_tokens_to_target_kvcache(
    batch: ScheduleBatch,
    accept_index: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
):
    """
    Move accepted tokens (drafts + bonus) to the target KV cache.

    Args:
        batch: The batch to run.
        accept_index: The index of the accepted tokens (incl. bonus).
        num_correct_drafts: Per-req count of correct drafts (excludes bonus);
            seq_lens is advanced by ``num_correct_drafts + 1`` to cover the bonus slot.
    """
    bs = len(batch.seq_lens)
    device = batch.seq_lens.device
    # accept_index element count, NOT bs * num_draft_tokens: for topk > 1 the
    # tree exceeds the accepted chain, over-reading accept_index (illegal memory).
    size = bs * accept_index.shape[1]

    # fill_accept_out_cache_loc reads out_cache_loc[accept_index]; -1 sentinel ok.
    maybe_detect_oob(
        accept_index,
        -1,
        batch.out_cache_loc.size(0),
        "spec v2 move_accept_tokens accept_index",
    )

    tgt_cache_loc = torch.zeros(
        size,
        dtype=torch.int64,
        device=device,
    )
    accept_out_cache_loc = torch.zeros(size, dtype=torch.int64, device=device)
    # The CPU / triton dispatch (and the TPU torch-native path) lives in the
    # kernel wrapper; ``size`` there is accept_index.numel() == this ``size``.
    assign_extend_cache_locs_kernel_func(
        batch.req_pool_indices,
        batch.req_to_token_pool.req_to_token,
        batch.seq_lens,
        batch.seq_lens + num_correct_drafts + 1,
        tgt_cache_loc,
    )
    fill_accept_out_cache_loc_func(
        accept_index,
        batch.out_cache_loc,
        accept_out_cache_loc,
    )
    token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
        tgt_cache_loc, accept_out_cache_loc
    )


def prepare_mamba_track_for_verify(batch: ScheduleBatch) -> None:
    """Rebuild mamba track indices from reqs before a TARGET_VERIFY forward.

    Spec batches skip the refresh in prepare_for_decode, and filter/merge
    null these fields, so they must be rebuilt right before verify. Clearing
    the mask also keeps a stale extend-time mask from triggering in-forward
    tracking during TARGET_VERIFY; tracking is done in
    commit_mamba_states_after_verify instead.

    Lazy: gather the positions planned by mamba_lazy_spec_prepare. Runs
    inside forward isolation, so it must not mutate req/pool state.
    """
    if not get_exec().mamba.enable_mamba_extra_buffer:
        return
    track_positions = None
    if get_exec().mamba.enable_mamba_extra_buffer_lazy:
        track_positions = batch.mamba_lazy_spec_track_positions_cpu
        assert track_positions is not None and len(track_positions) == len(
            batch.reqs
        ), (
            "lazy spec verify without a track plan: mamba_lazy_spec_prepare "
            "must run in prepare_for_decode for every spec decode iteration"
        )
    set_mamba_track_indices_from_reqs(batch, track_positions)
    batch.mamba_track_mask = None
    batch.mamba_track_seqlens = None
    batch.mamba_prefill_track_mask_cpu = None
    batch.mamba_track_seqlens_cpu = None


def _verify_commit_step_indices(
    *,
    batch: ScheduleBatch,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    draft_token_num: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Step indices for a post-verify state commit: per req, the tree step of
    the last accepted node (reduces to accept_lens - 1 for topk == 1), and the
    mamba-track interval-crossing step (-1 = no crossing; None when tracking
    is off)."""
    bs = accept_lens.shape[0]
    if accept_index.is_cuda:
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_commit_track_indices,
        )

        track_grid = (
            mamba_track_grid(batch.tree_cache.page_size)
            if batch.mamba_track_indices is not None
            else 0
        )
        return fused_commit_track_indices(
            accept_index,
            accept_lens,
            batch.seq_lens if track_grid > 0 else None,
            draft_token_num,
            track_grid,
        )
    accept_indices_offset = torch.arange(
        0,
        bs * draft_token_num,
        step=draft_token_num,
        dtype=accept_lens.dtype,
        device=accept_lens.device,
    )
    req_idx = torch.arange(bs, dtype=torch.int64, device=accept_lens.device)
    last_correct_step_indices = (
        accept_index[req_idx, (accept_lens - 1).to(torch.int64)] - accept_indices_offset
    )
    if batch.mamba_track_indices is None:
        return last_correct_step_indices, None
    seq_lens_pre_verify = batch.seq_lens
    seq_lens_post_verify = batch.seq_lens + accept_lens
    mamba_track_interval = mamba_track_grid(batch.tree_cache.page_size)
    to_track_mask = (
        seq_lens_pre_verify // mamba_track_interval
        != seq_lens_post_verify // mamba_track_interval
    )
    tracking_point = seq_lens_post_verify // mamba_track_interval * mamba_track_interval
    to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0).to(
        torch.int64
    )
    candidate_track_steps = accept_index[req_idx, to_track_ith] - accept_indices_offset
    mamba_steps_to_track = torch.where(
        to_track_mask,
        candidate_track_steps,
        torch.full_like(candidate_track_steps, -1),
    )
    return last_correct_step_indices, mamba_steps_to_track


def commit_mamba_states_after_verify(
    target_worker: TpModelWorker,
    batch: ScheduleBatch,
    accept_lens: torch.Tensor,
    accept_index: torch.Tensor,
    draft_token_num: int,
    prepared_step_indices: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
) -> None:
    """Commit accepted per-step mamba states into the persistent caches.

    During TARGET_VERIFY, hybrid linear attention backends keep per-step
    states in intermediate caches instead of advancing the persistent
    conv/ssm caches. After acceptance, the state of each request's last
    accepted step is committed back, plus the interval-crossing state used
    for prefix-cache tracking (mamba extra_buffer mode).

    No-op for models without mamba-style state or backends without the
    commit hook.
    """
    model_runner = target_worker.model_runner
    if mambaish_config(model_runner.model_config) is None:
        return

    # ReplaySSM spec-verify path (Part B of #28511): the accepted drafts already
    # live in the per-slot circular ring (written during verify). Instead of
    # scattering an intermediate full SSM state into `temporal`, advance the
    # block-keyed cursors by the accepted count (the ring owns the SSM state; the
    # verify/flush kernel folds it into `temporal` periodically). The CONV state
    # still needs its usual accept-rollback, so we keep the conv-window scatter and
    # skip only the SSM scatter. GDN-only + linear-chain (topk<=1) -- the runtime
    # ring is allocated only then; KDA never allocates the cursors.
    req_pool = model_runner.req_to_token_pool
    mamba_pool = getattr(req_pool, "mamba_pool", None)
    bs = accept_lens.shape[0]
    src_indices_raw = (
        batch.req_pool_indices[:bs] if pp_spec_stable_rows_enabled() else None
    )

    # Fold-every-commit: replay the accepted prefix from the ring into
    # `temporal`; the same fold stores the interval-crossing state to the
    # track slot, so no SSM scatter or force-flush is needed here.
    if (
        mamba_pool is not None
        and getattr(mamba_pool, "replayssm_spec_fold", False)
        and not getattr(mamba_pool, "replayssm_is_kda", False)
    ):
        if batch.forward_mode.is_idle() or accept_index.numel() == 0:
            return
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
            commit_gdn_replayssm_fold_after_verify,
        )

        spec_state = req_pool.get_speculative_mamba2_params_all_layers()
        state_batch_indices = req_pool.get_mamba_indices(batch.req_pool_indices)
        if prepared_step_indices is None:
            last_correct_step_indices, mamba_steps_to_track = (
                _verify_commit_step_indices(
                    batch=batch,
                    accept_index=accept_index,
                    accept_lens=accept_lens,
                    draft_token_num=draft_token_num,
                )
            )
        else:
            last_correct_step_indices, mamba_steps_to_track = prepared_step_indices
        commit_gdn_replayssm_fold_after_verify(
            spec_state=spec_state,
            state_batch_indices=state_batch_indices,
            accept_lens=accept_lens,
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            src_indices_raw=src_indices_raw,
            null_block_id=-1,
        )
        return

    if (
        mamba_pool is not None
        and getattr(mamba_pool, "replayssm_cache_base", None) is not None
        and not getattr(mamba_pool, "replayssm_is_kda", False)
    ):
        if batch.forward_mode.is_idle() or accept_index.numel() == 0:
            return
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
            commit_gdn_replayssm_circular,
            commit_gdn_replayssm_spec,
        )
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_conv_window_scatter_with_mask,
        )

        spec_state = req_pool.get_speculative_mamba2_params_all_layers()
        state_batch_indices = req_pool.get_mamba_indices(batch.req_pool_indices)
        replay_indices = batch.req_pool_indices
        if prepared_step_indices is None:
            last_correct_step_indices, mamba_steps_to_track = (
                _verify_commit_step_indices(
                    batch=batch,
                    accept_index=accept_index,
                    accept_lens=accept_lens,
                    draft_token_num=draft_token_num,
                )
            )
        else:
            last_correct_step_indices, mamba_steps_to_track = prepared_step_indices
        # Advance the per-request circular cursors by the accepted count (incl. the
        # bonus token). max_cache_len = ring length L = replayssm_d.shape[-2].
        commit_gdn_replayssm_spec(
            write_pos=mamba_pool.replayssm_spec_write_pos,
            cache_base=mamba_pool.replayssm_cache_base,
            is_flush=mamba_pool.replayssm_is_flush,
            num_accepted=accept_lens,  # [bs], includes the bonus token
            replay_indices=replay_indices,
            max_cache_len=spec_state.replayssm_d.shape[-2],
            max_spec_len=draft_token_num,
            fold_every_commit=spec_state.temporal.dtype != torch.float32,
            null_block_id=-1,  # SGLang: valid slots >= 0, padding == -1
        )
        # Capacity rows fold all layers in one launch; track rows snapshot the
        # exact crossing state without disturbing the active circular history.
        commit_gdn_replayssm_circular(
            checkpoint_state=spec_state.temporal,
            d_cache=spec_state.replayssm_d,
            k_cache=spec_state.replayssm_k,
            g_cache=spec_state.replayssm_g,
            d_residual_cache=spec_state.replayssm_rawv,
            k_residual_cache=spec_state.replayssm_rawk,
            state_batch_indices=state_batch_indices,
            replay_indices=replay_indices,
            write_pos=mamba_pool.replayssm_spec_write_pos,
            cache_base=mamba_pool.replayssm_cache_base,
            is_flush=mamba_pool.replayssm_is_flush,
            accept_lens=accept_lens,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            null_block_id=-1,
        )
        # Roll back active conv state and snapshot its interval-crossing window.
        fused_conv_window_scatter_with_mask(
            spec_state.conv[0],
            spec_state.intermediate_conv_window[0],
            state_batch_indices,
            last_correct_step_indices,
            src_indices_raw,
        )
        if batch.mamba_track_indices is not None:
            fused_conv_window_scatter_with_mask(
                spec_state.conv[0],
                spec_state.intermediate_conv_window[0],
                batch.mamba_track_indices,
                mamba_steps_to_track,
                src_indices_raw,
            )
        return

    # KDA ReplaySSM (fold-every-commit): KDA keeps its own recurrent verify kernel
    # for the OUTPUT, so we replay the accepted window into the fp32 checkpoint
    # (`temporal`) here on commit -- `temporal` is always the current committed
    # state. The draft window's raw inputs were written to the ring during verify
    # by the KDA backend. Gate on the fold flag + is_kda (the cursor tensors are
    # never allocated under fold, so they cannot serve as the signal).
    if (
        mamba_pool is not None
        and getattr(mamba_pool, "replayssm_spec_fold", False)
        and getattr(mamba_pool, "replayssm_is_kda", False)
    ):
        if batch.forward_mode.is_idle() or accept_index.numel() == 0:
            return
        from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
            commit_kda_replayssm_after_verify,
        )

        spec_state = req_pool.get_speculative_mamba2_params_all_layers()
        state_batch_indices = req_pool.get_mamba_indices(batch.req_pool_indices)
        accept_indices_offset = torch.arange(
            0,
            bs * draft_token_num,
            step=draft_token_num,
            dtype=accept_lens.dtype,
            device=accept_lens.device,
        )
        req_idx = torch.arange(bs, dtype=torch.int64, device=accept_lens.device)
        last_correct_step_indices = (
            accept_index[req_idx, (accept_lens - 1).to(torch.int64)]
            - accept_indices_offset
        )
        # extra_buffer: the interval-crossing step whose state must snapshot into
        # the track ping-pong slot (mirrors the regular commit's
        # mamba_steps_to_track); commit_kda_replayssm_spec folds it in one pass, so
        # `temporal` stays current and no device-side force-flush is needed.
        mamba_track_indices = batch.mamba_track_indices
        mamba_steps_to_track = None
        if mamba_track_indices is not None:
            ti = mamba_track_grid(batch.tree_cache.page_size)
            seq_pre = batch.seq_lens
            seq_post = batch.seq_lens + accept_lens
            to_track_mask = seq_pre // ti != seq_post // ti
            tracking_point = seq_post // ti * ti
            to_track_ith = torch.clamp(tracking_point - seq_pre - 1, min=0).to(
                torch.int64
            )
            candidate = accept_index[req_idx, to_track_ith] - accept_indices_offset
            mamba_steps_to_track = torch.where(
                to_track_mask, candidate, torch.full_like(candidate, -1)
            )
        commit_kda_replayssm_after_verify(
            spec_state=spec_state,
            state_batch_indices=state_batch_indices,
            accept_lens=accept_lens,  # incl. bonus token
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            src_indices_raw=src_indices_raw,
            null_block_id=-1,  # SGLang: valid slots >= 0, padding == -1
        )
        return

    attn_backend = model_runner.attn_backend

    # `accept_lens` already includes the bonus token (drafts + 1 per req).
    if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
        if prepared_step_indices is None:
            last_correct_step_indices, mamba_steps_to_track = (
                _verify_commit_step_indices(
                    batch=batch,
                    accept_index=accept_index,
                    accept_lens=accept_lens,
                    draft_token_num=draft_token_num,
                )
            )
        else:
            last_correct_step_indices, mamba_steps_to_track = prepared_step_indices

        if hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            attn_backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                model=model_runner.model,
                req_pool_indices=batch.req_pool_indices[:bs],
            )


def spec_prepare_for_decode(batch: ScheduleBatch) -> None:
    """eagle/ngram share a stateless free function; dflash keeps stateful
    prep on its draft input -- the dispatcher routes.
    """
    if get_exec().mamba.enable_mamba_extra_buffer_lazy:
        # Scheduler phase (outside forward isolation).
        batch.mamba_lazy_spec_prepare(
            mamba_track_grid(batch.tree_cache.page_size),
            max_speculative_num_draft_tokens(),
        )
    if batch.spec_algorithm.is_dflash_family():
        batch.spec_info.prepare_for_decode(batch)
    elif batch.spec_algorithm.is_uno():
        from sglang.srt.speculative.uno_info import UnoDraftInput

        if not isinstance(batch.spec_info, UnoDraftInput):
            raise RuntimeError("UNO decode preparation requires UnoDraftInput")
        batch.spec_info.prepare_for_decode(batch)
    else:
        from sglang.srt.speculative.eagle_utils import eagle_prepare_for_decode

        eagle_prepare_for_decode(batch)


# Disable torch.compile for this function because it will be
# even slower.
# @torch.compile(dynamic=True)
def get_last_loc_large_page_size_large_top_k(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    speculative_num_steps: int,
    topk: int,
    page_size: int,
):
    prefix_lens = seq_lens
    last_page_lens = prefix_lens % page_size
    num_new_pages_per_topk = (
        last_page_lens + speculative_num_steps + page_size - 1
    ) // page_size
    seq_lens = prefix_lens // page_size * page_size + num_new_pages_per_topk * (
        page_size * topk
    )
    extend_lens = seq_lens - prefix_lens
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )

    return (
        prefix_lens,
        seq_lens,
        last_loc,
        num_new_pages_per_topk,
        extend_lens,
        last_page_lens,
    )


def get_plan_stream(
    device: str,
) -> Tuple[Any, contextlib.AbstractContextManager]:
    # Platforms without real multi-stream execution (e.g. TPU) get no plan
    # stream: a non-default stream there only relabels fences.
    if not current_platform.supports_overlap_streams():
        return None, contextlib.nullcontext()
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()
