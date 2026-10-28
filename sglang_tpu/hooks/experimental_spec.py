"""Narrow hook adapters for the experimental host-padded spec path.

Why: filter/merge and H2D staging must be intercepted before device operations,
not repaired afterward. All non-experimental calls delegate unchanged.
kimi_spec_defaults: core refuses speculation for Kimi hybrid (KDA) targets on
TPU and otherwise pins GPU verify/draft backends; the experimental worker
serves DSPARK on the TPU KDA checkpoint verify, so it skips both.
materialize_future_map: the live FutureMap's lazy torch.empty buffers would fuse
into their first store, compiling at whatever shape the first batch has.
Upstream seam issue: not filed (local prototype). Before proposing upstream,
replace these hooks with agreed factories/overrides for the extracted operations.
Remove when: the experiment is removed or those upstream seams are available.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.runtime_context import get_disagg, get_spec
from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm
from sglang_tpu.experimental import (
    experimental_spec_enabled,
    get_host_batch_padder,
    is_experimental_chain,
    is_experimental_spec,
    is_experimental_spec_worker,
    uses_host_batch,
)
from sglang_tpu.experimental.speculative.draft_info import HostPaddedDraftInput


def initialize_host_batch(result, *args, **kwargs):
    # Every batch of a process a TpuSpecWorker serves is host padded, whatever the algorithm.
    padder = get_host_batch_padder()
    if padder is not None:
        # Only decode relays through FutureMap; a PD prefill (PP runs it
        # without overlap) never decodes.
        if not result.enable_overlap and get_disagg().disaggregation_mode != "prefill":
            raise ValueError(
                "The experimental TPU spec worker requires overlap scheduling "
                "outside --disaggregation-mode prefill"
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
        from sglang_tpu.experimental.sampling.sampling_batch_info import (
            HostPaddedSamplingInfo,
        )

        cls = HostPaddedSamplingInfo
    return original(
        cls, batch, vocab_size, sampling_params_on_cpu=sampling_params_on_cpu
    )


def _relays_rows(batch) -> bool:
    """Whether FutureMap owns batch's rows; a non-overlap PD prefill keeps core's."""
    return uses_host_batch(batch) and batch.enable_overlap


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
    if not _relays_rows(batch):
        return original(batch, keep_indices, keep_indices_device)
    _rebuild_relay_rows(batch)


def merge_model_inputs(original, batch, other):
    assert uses_host_batch(batch) == uses_host_batch(other)
    if not _relays_rows(batch):
        return original(batch, other)
    assert other.input_ids is None
    _rebuild_relay_rows(batch)


def _draft_state_on_host(spec_info: SpecInput | None) -> bool:
    from sglang.srt.speculative.eagle_info import EagleDraftInput

    return (
        isinstance(spec_info, EagleDraftInput)
        and spec_info.bonus_tokens is not None
        and spec_info.bonus_tokens.device.type == "cpu"
    )


def filter_spec_input(original, batch, keep_indices, keep_indices_device):
    if not _relays_rows(batch):
        if _draft_state_on_host(batch.spec_info):
            # A PP PD prefill keeps the relayed draft state on host; index it there.
            keep_indices_device = torch.tensor(keep_indices, dtype=torch.int64)
        return original(batch, keep_indices, keep_indices_device)
    from sglang_tpu.experimental.speculative.draft_info import HostPaddedDraftInput

    assert isinstance(batch.spec_info, HostPaddedDraftInput)
    batch.spec_info.filter_batch(
        new_indices=keep_indices_device,
        new_indices_cpu=keep_indices,
        future_indices=batch.req_pool_indices,
    )


def merge_spec_input(original, batch, other):
    if not _relays_rows(batch):
        return original(batch, other)
    from sglang_tpu.experimental.speculative.draft_info import HostPaddedDraftInput

    assert isinstance(batch.spec_info, HostPaddedDraftInput)
    batch.spec_info.merge_batch(other.spec_info, future_indices=batch.req_pool_indices)


def prepare_spec_decode(original, batch):
    # Core routes only DFlash-family decode prep to the stateful draft input;
    # every TpuSpecWorker input is a HostPaddedDraftInput, which reserves KV and
    # plans the bucketed verify window.
    if not uses_host_batch(batch) or batch.spec_algorithm.is_dflash_family():
        return original(batch)
    from sglang_tpu.experimental.speculative.draft_info import HostPaddedDraftInput

    assert isinstance(batch.spec_info, HostPaddedDraftInput)
    batch.spec_info.prepare_for_decode(batch)


SeedBuilder = Callable[[ScheduleBatch, torch.Tensor, FutureMap], SpecInput]


def build_disagg_eagle_input(
    original: SeedBuilder,
    batch: ScheduleBatch,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> SpecInput:
    # Core stacks the seed on device at the live bs, one compile per bs. The chain
    # worker (MTP, EAGLE3) keeps the draft hidden in its carry table instead.
    if not uses_host_batch(batch):
        return original(batch, last_tokens_tensor, future_map)
    from sglang_tpu.experimental import get_pd_carry_seeder
    from sglang_tpu.experimental.speculative.chain_info import (
        HostPaddedChainDraftInput,
    )

    assert is_experimental_chain(batch.spec_algorithm), batch.spec_algorithm
    if any(req.output_dsa_topk_indices is not None for req in batch.reqs):
        raise NotImplementedError("TpuChainSpecWorker does not support DSA PD seeds.")
    get_pd_carry_seeder()(
        batch.req_pool_indices_cpu,
        torch.stack([req.hidden_states_tensor for req in batch.reqs]),
    )
    draft_input = HostPaddedChainDraftInput(
        bonus_tokens=last_tokens_tensor,
        new_seq_lens=batch.seq_lens.to(torch.int64),
    )
    return _relay_disagg_seed(batch, draft_input, future_map)


def build_disagg_block_input(
    original: SeedBuilder,
    batch: ScheduleBatch,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> SpecInput:
    # DFlash/DSpark seed only the bonus token; the draft context KV arrives with
    # the PD transfer of the draft pool.
    if not uses_host_batch(batch):
        return original(batch, last_tokens_tensor, future_map)
    from sglang_tpu.experimental.speculative.block_info import (
        HostPaddedBlockDraftInput,
    )

    draft_input = HostPaddedBlockDraftInput(
        bonus_tokens=last_tokens_tensor,
        new_seq_lens=batch.seq_lens.to(torch.int64),
    )
    return _relay_disagg_seed(batch, draft_input, future_map)


def _relay_disagg_seed(
    batch: ScheduleBatch, draft_input: HostPaddedDraftInput, future_map: FutureMap
) -> HostPaddedDraftInput:
    """Publish and stash the seed by pool row, as core's PD seed builders do."""
    from sglang.srt.managers.overlap_utils import RelayPayload

    assert batch.enable_overlap
    _reset_pd_checkpoints(batch)
    draft_input.future_indices = batch.req_pool_indices
    future_map.publish(draft_input.future_indices, batch.seq_lens)
    future_map.stash(
        draft_input.future_indices, RelayPayload.from_draft_input(draft_input)
    )
    return draft_input


def _reset_pd_checkpoints(batch: ScheduleBatch) -> None:
    # The transferred GDN state lands in each slot's own row, but a reused decode
    # slot keeps its previous read offset; prebuilt skips extend's reset.
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang_tpu.mamba_pool import TpuMambaPool

    pool = batch.req_to_token_pool
    if not isinstance(pool, HybridReqToTokenPool):
        return
    mamba_pool = pool.mamba_pool
    assert isinstance(mamba_pool, TpuMambaPool), type(mamba_pool)
    if mamba_pool.checkpoint_read_offsets is not None:
        mamba_pool.clear_slots(
            torch.stack([req.kv.mamba_pool_idx.reshape(()) for req in batch.reqs])
        )


def materialize_future_map(
    result: FutureMap, algorithm: SpeculativeAlgorithm, *args, **kwargs
) -> FutureMap:
    # A torch.empty buffer stays lazy and fuses into its first store, one program
    # per first-batch shape; an uploaded buffer is concrete.
    if algorithm.is_none():
        return result
    from sglang_tpu.compilation.recompilation_checker import (
        snapshot_recompilation_baseline,
    )

    result.output_tokens_buf = _uploaded_zeros_like(result.output_tokens_buf)
    result.new_seq_lens_buf = _uploaded_zeros_like(result.new_seq_lens_buf)
    # The map is built after the scheduler's warmup baseline.
    torch.accelerator.synchronize()
    snapshot_recompilation_baseline()
    return result


def _uploaded_zeros_like(buf: torch.Tensor) -> torch.Tensor:
    return torch.zeros(buf.shape, dtype=buf.dtype).to(buf.device)


def stage_prefill_tokens(original, batch):
    if not uses_host_batch(batch):
        return original(batch)
    assert batch.mix_running_indices is None, (
        "Host-padded spec does not support mixed chunks"
    )
    return batch.prefill_input_ids_cpu


def extend_metadata_device(original, batch, device):
    from sglang_tpu.experimental.sampling.sampling_batch_info import is_host_padded

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
    from sglang_tpu.experimental.sampling.sampling_batch_info import is_host_padded

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
    from sglang_tpu.experimental import experimental_spec_enabled

    if not experimental_spec_enabled():
        return original(model, hidden_states, input_ids)
    from sglang_tpu.experimental.models.dspark import sample_draft_tokens as sample

    return sample(model, hidden_states, input_ids)


def sample_dflash2_draft_tokens(original, model, hidden_states, input_ids):
    from sglang_tpu.experimental import experimental_spec_enabled

    if not experimental_spec_enabled():
        return original(model, hidden_states, input_ids)
    from sglang_tpu.experimental.models.dflash2 import sample_draft_tokens as sample

    return sample(model, hidden_states, input_ids)


def supports_kv_free_fence(original, spec_algorithm, *, copies_draft_tail):
    if is_experimental_spec(spec_algorithm):
        # commit writes draft KV after the accept tensors the result copies;
        # the copied draft tail (kv_fence_tail) is an output of that write.
        return copies_draft_tail
    return original(spec_algorithm, copies_draft_tail=copies_draft_tail)


def supports_pd_decode_swa_hicache(original, spec_algorithm):
    if is_experimental_spec_worker(spec_algorithm):
        # The block draft's KV at i is projected from the target hidden at i and
        # written for every prompt and committed token, so unigram radix keys and
        # the full-anchored draft sidecar cover it; this extends DFlash to DSpark.
        return True
    return original(spec_algorithm)


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
    from sglang_tpu.experimental import experimental_spec_enabled

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
            "sglang.srt.speculative.spec_info.SpeculativeAlgorithm.supports_pd_decode_swa_hicache",
            supports_pd_decode_swa_hicache,
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
        # Hot paths every spec batch takes (non-spec batches return immediately);
        # skipped when SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER=0 at import time.
        hooks += [
            (
                "sglang.srt.speculative.spec_utils.spec_prepare_for_decode",
                prepare_spec_decode,
                HookType.AROUND,
            ),
            (
                "sglang.srt.speculative.spec_info.SpeculativeAlgorithm."
                "create_future_map",
                materialize_future_map,
                HookType.AFTER,
            ),
            (
                "sglang.srt.speculative.eagle_disaggregation."
                "build_eagle_disagg_draft_input",
                build_disagg_eagle_input,
                HookType.AROUND,
            ),
            (
                "sglang.srt.speculative.dflash_disaggregation."
                "build_dflash_disagg_draft_input",
                build_disagg_block_input,
                HookType.AROUND,
            ),
            (
                "sglang.srt.speculative.dspark_disaggregation."
                "build_dspark_disagg_draft_input",
                build_disagg_block_input,
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
