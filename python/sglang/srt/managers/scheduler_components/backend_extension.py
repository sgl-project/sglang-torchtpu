# SPDX-License-Identifier: Apache-2.0
"""Per-scheduler backend collaborators; shared scheduling remains in Scheduler."""

from contextlib import nullcontext
from typing import NamedTuple


class PipelineTransferStatus(NamedTuple):
    terminal_rids: list[str]
    failed_rids: list[str]


class SchedulerExtension:
    def create_future_map(self, algorithm, device, req_to_token_pool, **kwargs):
        return algorithm.create_future_map(device, req_to_token_pool, **kwargs)

    def init_result_copier(self, *, enable_overlap, needs_kv_free_fence):
        pass

    def copy_result_to_cpu(
        self,
        result,
        *,
        return_logprob,
        return_hidden_states,
        use_overlap_streams,
        copy_stream,
        forward_stream,
        copy_stream_ctx,
    ):
        if use_overlap_streams:
            copy_stream.wait_stream(forward_stream)
        with copy_stream_ctx:
            result.copy_to_cpu(
                return_logprob=return_logprob, return_hidden_states=return_hidden_states
            )

    def needs_kv_free_fence(
        self, *, enable_overlap, uses_raiden_hicache, disaggregation_mode
    ):
        return False

    def create_kv_free_fence(
        self,
        *,
        allocator,
        device_module,
        spec_algorithm,
        enable_overlap,
        uses_raiden_hicache,
        disaggregation_mode,
        tree_cache,
    ):
        return None

    def graph_warmup_context(self):
        return nullcontext()

    def precompile_spec_ops(self, *, target_runner, draft_worker, vocab_size):
        pass

    def validate_pending_results(self, spec_algorithm, num_pending):
        pass

    def init_chunked_prefill(
        self, *, runner, page_size, chunked_prefill_size, world_group
    ):
        pass

    def prefill_adder_options(
        self, *, waiting_queue, chunked_req, mbs, last_mbs, tree_cache
    ):
        from sglang.srt.managers.schedule_policy import PrefillAdder

        return PrefillAdder, {}

    def hicache_load_page_ids(
        self, *, disaggregation_mode, requests, tree_cache, page_size
    ):
        return None

    def group_drained(self, senders, groups):
        return [sender.drained() for sender in senders]

    def poll_drained(self, senders, groups):
        for sender in senders:
            sender.poll()
            yield sender.drained()

    def create_pipeline_transport(
        self, *, model_runner, pp_group, stage_group, world_group, depth
    ):
        return None

    def close(self):
        pass
