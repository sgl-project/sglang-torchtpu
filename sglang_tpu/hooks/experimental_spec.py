"""Narrow hook adapters for the experimental host-padded spec path.

Why: filter/merge and H2D staging must be intercepted before device operations,
not repaired afterward. All non-experimental calls delegate unchanged.
kimi_spec_defaults: core refuses speculation for Kimi hybrid (KDA) targets on
TPU and otherwise pins GPU verify/draft backends; the experimental worker
serves DSPARK on the TPU KDA checkpoint verify, so it skips both.
Upstream seam issue: not filed (local prototype). Before proposing upstream,
replace these hooks with agreed factories/overrides for the extracted operations.
Remove when: the experiment is removed or those upstream seams are available.
"""

from __future__ import annotations

import torch

from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.speculative.experimental_spec import (
    experimental_spec_enabled,
    get_host_batch_padder,
    is_experimental_chain,
    uses_host_batch,
)


def initialize_host_batch(result, *args, **kwargs):
    # Every batch of a process a TpuSpecWorker serves is host padded, whatever the algorithm.
    padder = get_host_batch_padder()
    if padder is not None:
        if not result.enable_overlap:
            raise ValueError(
                "The experimental TPU spec worker requires overlap scheduling"
            )
        result.host_pad_bs = padder
    return result


def copy_host_batch(result, batch):
    if uses_host_batch(batch):
        result.host_pad_bs = batch.host_pad_bs
    return result


def sampling_from_batch(
    original, cls, batch, vocab_size, *, sampling_params_on_cpu=False
):
    if uses_host_batch(batch):
        from sglang_tpu.sampling.sampling_batch_info import (
            HostPaddedSamplingInfo,
        )

        cls = HostPaddedSamplingInfo
    return original(
        cls, batch, vocab_size, sampling_params_on_cpu=sampling_params_on_cpu
    )


def _rebuild_relay_rows(batch):
    assert batch.enable_overlap and not batch.spec_algorithm.is_none()
    assert batch.input_ids is None
    batch.req_pool_indices = batch.req_pool_indices_cpu.to(
        batch.device, non_blocking=True
    )
    # FutureMap resolves these before forward; direct compile ignores orig_seq_lens.
    batch.seq_lens = None
    batch.orig_seq_lens = None


def filter_model_inputs(original, batch, keep_indices, keep_indices_device):
    if not uses_host_batch(batch):
        return original(batch, keep_indices, keep_indices_device)
    _rebuild_relay_rows(batch)


def merge_model_inputs(original, batch, other):
    assert uses_host_batch(batch) == uses_host_batch(other)
    if not uses_host_batch(batch):
        return original(batch, other)
    assert other.input_ids is None
    _rebuild_relay_rows(batch)


def filter_spec_input(original, batch, keep_indices, keep_indices_device):
    if not uses_host_batch(batch):
        return original(batch, keep_indices, keep_indices_device)
    from sglang_tpu.speculative.draft_info import HostPaddedDraftInput

    assert isinstance(batch.spec_info, HostPaddedDraftInput)
    batch.spec_info.filter_batch(
        new_indices=keep_indices_device,
        new_indices_cpu=keep_indices,
        future_indices=batch.req_pool_indices,
    )


def merge_spec_input(original, batch, other):
    if not uses_host_batch(batch):
        return original(batch, other)
    from sglang_tpu.speculative.draft_info import HostPaddedDraftInput

    assert isinstance(batch.spec_info, HostPaddedDraftInput)
    batch.spec_info.merge_batch(other.spec_info, future_indices=batch.req_pool_indices)


def prepare_spec_decode(original, batch):
    # Core routes only DFlash-family decode prep to the stateful draft input;
    # every TpuSpecWorker input is a HostPaddedDraftInput, which reserves KV and
    # plans the bucketed verify window.
    if not uses_host_batch(batch) or batch.spec_algorithm.is_dflash_family():
        return original(batch)
    from sglang_tpu.speculative.draft_info import HostPaddedDraftInput

    assert isinstance(batch.spec_info, HostPaddedDraftInput)
    batch.spec_info.prepare_for_decode(batch)


def build_disagg_eagle_input(original, batch, last_tokens_tensor, future_map):
    # Core builds and relays the PD seed; the experimental worker reads it back
    # through its own input type.
    spec_info = original(batch, last_tokens_tensor, future_map)
    if not uses_host_batch(batch):
        return spec_info
    from sglang_tpu.speculative.eagle_info import (
        HostPaddedEagleDraftInput,
    )

    if spec_info.dsa_topk_indices is not None:
        raise NotImplementedError("TpuEagle3Worker does not support DSA PD seeds.")
    draft_input = HostPaddedEagleDraftInput(
        bonus_tokens=spec_info.bonus_tokens,
        new_seq_lens=batch.seq_lens.to(torch.int64),
        topk_p=spec_info.topk_p,
        topk_index=spec_info.topk_index,
        hidden_states=spec_info.hidden_states,
    )
    draft_input.future_indices = spec_info.future_indices
    return draft_input


def stage_prefill_tokens(original, batch):
    if not uses_host_batch(batch):
        return original(batch)
    assert batch.mix_running_indices is None, (
        "Host-padded spec does not support mixed chunks"
    )
    return batch.prefill_input_ids_cpu


def extend_metadata_device(original, batch, device):
    from sglang_tpu.sampling.sampling_batch_info import is_host_padded

    if is_host_padded(batch.sampling_info):
        return "cpu"
    return original(batch, device)


def alloc_extend(
    original,
    tree_cache,
    prefix_lens,
    prefix_lens_cpu,
    seq_lens,
    seq_lens_cpu,
    last_loc,
    extend_num_tokens,
    req_pool_indices=None,
    batch=None,
    out_cache_loc_host=None,
    return_host=False,
):
    if uses_host_batch(batch):
        return_host = True
        if out_cache_loc_host is None:
            out_cache_loc_host = torch.empty((extend_num_tokens,), dtype=torch.int32)
    return original(
        tree_cache=tree_cache,
        prefix_lens=prefix_lens,
        prefix_lens_cpu=prefix_lens_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        last_loc=last_loc,
        extend_num_tokens=extend_num_tokens,
        req_pool_indices=req_pool_indices,
        batch=batch,
        out_cache_loc_host=out_cache_loc_host,
        return_host=return_host,
    )


def keep_host_padded_extend_outputs(original, forward_batch, logits_output):
    # finalize_forward_batch left HOST_PADDED extend logits at the batch bucket
    # and hidden at the token bucket; DP attention's unpadding would trim them
    # to the live rows, which the samplers and the draft prefill do not take.
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang_tpu.sampling.sampling_batch_info import is_host_padded

    sampling_info = forward_batch.sampling_info
    if (
        logits_output is None
        or sampling_info is None
        or not is_host_padded(sampling_info)
        or forward_batch.forward_mode != ForwardMode.EXTEND
    ):
        return original(forward_batch, logits_output)
    logits, hidden = logits_output.next_token_logits, logits_output.hidden_states
    result = original(forward_batch, logits_output)
    logits_output.next_token_logits, logits_output.hidden_states = logits, hidden
    return result


def sample_draft_tokens(original, model, hidden_states, input_ids):
    from sglang_tpu.speculative.experimental_spec import experimental_spec_enabled

    if not experimental_spec_enabled():
        return original(model, hidden_states, input_ids)
    from sglang_tpu.speculative.dspark.dspark import sample_draft_tokens as sample

    return sample(model, hidden_states, input_ids)


def sample_dflash2_draft_tokens(original, model, hidden_states, input_ids):
    from sglang_tpu.speculative.experimental_spec import experimental_spec_enabled

    if not experimental_spec_enabled():
        return original(model, hidden_states, input_ids)
    from sglang_tpu.speculative.dflash.dflash2 import sample_draft_tokens as sample

    return sample(model, hidden_states, input_ids)


def supports_kv_free_fence(original, spec_algorithm, *, copies_draft_tail):
    if is_experimental_chain(spec_algorithm):
        # commit writes draft KV after the accept tensors the result copies.
        return False
    return original(spec_algorithm, copies_draft_tail=copies_draft_tail)


def need_topk(original, spec_algorithm):
    # HostPaddedDraftInput carries an empty topk placeholder; an EAGLE topk relay
    # would route it through the EAGLE device-accept resolve.
    if is_experimental_chain(spec_algorithm):
        return False
    return original(spec_algorithm)


def supports_tpu_direct_compile(original, spec_algorithm):
    # The chain worker's fused draft and draft rows are direct-compiled.
    if is_experimental_chain(spec_algorithm):
        return True
    return original(spec_algorithm)


def spec_need_hidden_states(original):
    # The chain carry lives in a req-pool-indexed device table, never in FutureMap.
    algorithm = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
    if is_experimental_chain(algorithm):
        return False
    return original()


def kimi_spec_defaults(original, server_args):
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang_tpu.speculative.experimental_spec import experimental_spec_enabled

    cfg = resolving_view(server_args)
    if cfg.speculative_algorithm == "DSPARK" and experimental_spec_enabled():
        return None
    return original(server_args)


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    schedule = "sglang.srt.managers.schedule_batch.ScheduleBatch."
    hooks = [
        (
            "sglang.srt.arg_groups.kimi_k3_hook.apply_kimi_k3_spec_backend_defaults",
            kimi_spec_defaults,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.dspark.DSparkDraftModel.sample_draft_tokens_tpu",
            sample_draft_tokens,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.dflash.DFlash2DraftModel.sample_draft_tokens_tpu",
            sample_dflash2_draft_tokens,
            HookType.AROUND,
        ),
        (
            "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
            spec_need_hidden_states,
            HookType.AROUND,
        ),
        (
            "sglang.srt.speculative.spec_info.SpeculativeAlgorithm.need_topk",
            need_topk,
            HookType.AROUND,
        ),
        (
            "sglang.srt.speculative.spec_info.SpeculativeAlgorithm.supports_kv_free_fence",
            supports_kv_free_fence,
            HookType.AROUND,
        ),
        (
            "sglang.srt.speculative.spec_info.SpeculativeAlgorithm.supports_tpu_direct_compile",
            supports_tpu_direct_compile,
            HookType.AROUND,
        ),
        (schedule + "init_new", initialize_host_batch, HookType.AFTER),
        (schedule + "copy", copy_host_batch, HookType.AFTER),
        (schedule + "_filter_model_inputs", filter_model_inputs, HookType.AROUND),
        (schedule + "_merge_model_inputs", merge_model_inputs, HookType.AROUND),
        (schedule + "_filter_spec_input", filter_spec_input, HookType.AROUND),
        (schedule + "_merge_spec_input", merge_spec_input, HookType.AROUND),
        (
            "sglang.srt.sampling.sampling_batch_info.SamplingBatchInfo.from_schedule_batch",
            sampling_from_batch,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.overlap_utils._stage_prefill_tokens",
            stage_prefill_tokens,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.forward_batch_info._extend_metadata_device",
            extend_metadata_device,
            HookType.AROUND,
        ),
        (
            "sglang.srt.mem_cache.allocation.alloc_paged_token_slots_extend",
            alloc_extend,
            HookType.AROUND,
        ),
    ]
    if experimental_spec_enabled():
        # Hot paths every spec batch takes; registered only with the flag on so
        # the default path runs no extra wrappers.
        hooks += [
            (
                "sglang.srt.speculative.spec_utils.spec_prepare_for_decode",
                prepare_spec_decode,
                HookType.AROUND,
            ),
            (
                "sglang.srt.speculative.eagle_disaggregation."
                "build_eagle_disagg_draft_input",
                build_disagg_eagle_input,
                HookType.AROUND,
            ),
            (
                "sglang.srt.model_executor.forward_batch_info.ForwardBatch."
                "post_forward_mlp_sync_batch",
                keep_host_padded_extend_outputs,
                HookType.AROUND,
            ),
        ]
    return hooks
