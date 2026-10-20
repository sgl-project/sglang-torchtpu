# SPDX-License-Identifier: Apache-2.0
"""Scheduler-instance TPU state and backend lifecycle adapters."""

from contextlib import nullcontext

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.backend_extension import (
    SchedulerExtension,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.eagle_utils import get_draft_recurrent_hidden_state_spec
from sglang_tpu.envs import envs as tpu_envs


class TpuSchedulerExtension(SchedulerExtension):
    def __init__(self):
        self.result_copier = None
        self.kv_free_fence = None
        self.pipeline_transport = None
        self.chunk_cost = None
        self.cached_prefix = {}

    def create_future_map(self, algorithm, device, req_to_token_pool, **kwargs):
        from sglang_tpu.future_map import TpuFutureMap
        from sglang_tpu.host_padding_relay import HostPaddingFutureMap

        cls = TpuFutureMap
        if (
            tpu_envs.SGLANG_TPU_HOST_PADDING.get()
            and algorithm.is_none()
            and getattr(req_to_token_pool, "_tpu_relay_bucketer", None) is not None
        ):
            cls = HostPaddingFutureMap
        future_map = cls(device, algorithm, req_to_token_pool, **kwargs)
        from sglang.srt.runtime_context import get_context
        from sglang_tpu.spec_execution import resolve_spec_execution

        # Only a plugin spec worker serves this relay through compiled programs.
        if (
            not algorithm.is_none()
            and resolve_spec_execution(
                algorithm, pp_size=get_context().config_bag("parallel").pp_size
            ).host_padded
        ):
            from sglang_tpu.hooks.experimental_spec import materialize_future_map

            future_map = materialize_future_map(future_map, algorithm)
        return future_map

    def init_result_copier(self, *, enable_overlap, needs_kv_free_fence):
        if enable_overlap and envs.SGLANG_TPU_ASYNC_COPY_WORKER.get():
            from sglang_tpu.result_copy_worker import TPUAsyncCopyWorker

            self.result_copier = TPUAsyncCopyWorker(copy_draft_tail=needs_kv_free_fence)

    def copy_result_to_cpu(
        self, result, *, return_logprob, return_hidden_states, **kwargs
    ):
        if self.result_copier is None:
            return super().copy_result_to_cpu(
                result,
                return_logprob=return_logprob,
                return_hidden_states=return_hidden_states,
                **kwargs,
            )
        self.result_copier.submit(
            result,
            return_logprob=return_logprob,
            return_hidden_states=return_hidden_states,
        )

    def needs_kv_free_fence(
        self, *, enable_overlap, uses_raiden_hicache, disaggregation_mode
    ):
        from sglang.srt.disaggregation.utils import DisaggregationMode

        return enable_overlap and (
            uses_raiden_hicache or disaggregation_mode == DisaggregationMode.DECODE
        )

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
        if not self.needs_kv_free_fence(
            enable_overlap=enable_overlap,
            uses_raiden_hicache=uses_raiden_hicache,
            disaggregation_mode=disaggregation_mode,
        ):
            return None
        from sglang_tpu.kv_free_fence import (
            fence_hicache_backups,
            install_kv_free_fence,
            kv_free_fence_unsupported_reason,
        )

        kwargs = dict(
            spec_algorithm=spec_algorithm,
            copies_draft_tail=self.result_copier is not None
            and self.result_copier.copy_draft_tail,
        )
        reason = kv_free_fence_unsupported_reason(allocator, **kwargs)
        if reason is not None:
            needed_by = (
                "The raiden HiCache backend"
                if uses_raiden_hicache
                else "A PD decode server under overlap scheduling"
            )
            raise ValueError(
                f"{needed_by} needs the KV free fence: {reason} Pass --disable-overlap-schedule to run without it."
            )
        self.kv_free_fence = install_kv_free_fence(
            allocator, synchronize_device=device_module.synchronize, **kwargs
        )
        if uses_raiden_hicache:
            fence_hicache_backups(tree_cache.cache_controller, self.kv_free_fence)
        return self.kv_free_fence

    def graph_warmup_context(self):
        if envs.SGLANG_DIRECT_COMPILE_ASYNC.get():
            from sglang_tpu.compilation.warmup_queue import collect_warmups

            return collect_warmups()
        return nullcontext()

    def precompile_spec_ops(self, *, target_runner, draft_worker, vocab_size):
        if draft_worker.draft_worker is None:
            return
        from sglang_tpu.compilation.spec_decode_compile_policy import (
            BaseSpecDecodeCompilePolicyMixin,
        )
        from sglang_tpu.spec_op_precompile import (
            precompile_spec_v2_ops,
        )

        compile_policy = target_runner.compile_policy
        if not isinstance(compile_policy, BaseSpecDecodeCompilePolicyMixin):
            return
        assert compile_policy.b_bucketer is not None

        draft_runner = draft_worker.draft_worker.draft_runner
        hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(draft_runner)
        if hidden_size is None or hidden_dtype is None:
            return
        draft_vocab_size = (
            getattr(draft_runner.model.config, "draft_vocab_size", None)
            or draft_runner.model.config.vocab_size
        )

        precompile_spec_v2_ops(
            device=torch.device(target_runner.device),
            physical_batch_buckets=compile_policy.b_bucketer.buckets,
            num_verify_tokens=compile_policy.draft_token_num,
            vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
            hidden_size=hidden_size,
            hidden_dtype=hidden_dtype,
        )

    def validate_pending_results(self, spec_algorithm, num_pending):
        if spec_algorithm.is_dflash():
            # The verify window spans one commit; a second unprocessed forward
            # would address KV slots from the wrong committed base.
            assert num_pending <= 1

    def init_chunked_prefill(
        self, *, runner, page_size, chunked_prefill_size, world_group
    ):
        if get_parallel().pp_size > 1:
            from sglang_tpu.pp_chunk_scheduler import profile_cost_model

            self.chunk_cost = profile_cost_model(
                runner=runner,
                page_size=page_size,
                chunked_prefill_size=chunked_prefill_size,
                world_group=world_group,
            )

    def prefill_adder_options(
        self, *, waiting_queue, chunked_req, mbs, last_mbs, tree_cache
    ):
        if self.chunk_cost is None:
            return super().prefill_adder_options(
                waiting_queue=waiting_queue,
                chunked_req=chunked_req,
                mbs=mbs,
                last_mbs=last_mbs,
                tree_cache=tree_cache,
            )
        from sglang_tpu.pp_chunk_scheduler import (
            TpuPipelinePrefillAdder,
            pending_step_limit,
        )

        limit = pending_step_limit(
            cost=self.chunk_cost,
            cached_prefix=self.cached_prefix,
            waiting_queue=waiting_queue,
            chunked_req=chunked_req,
            mbs=mbs,
            last_mbs=last_mbs,
            tree_cache=tree_cache,
        )
        return TpuPipelinePrefillAdder, {"cost": self.chunk_cost, "step_limit": limit}

    def hicache_load_page_ids(
        self, *, disaggregation_mode, requests, tree_cache, page_size
    ):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang_tpu.prefill_early_send import tpu_hicache_load_page_ids

        if (
            disaggregation_mode != DisaggregationMode.PREFILL
            or not envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get()
            or not any(req.host_loaded_length > 0 for req in requests)
        ):
            return None
        return tpu_hicache_load_page_ids(
            page_size=page_size,
            load_indices=[
                tree_cache.path_device_indices(req.last_node)
                for req in requests
                if req.host_loaded_length > 0
            ],
        )

    def group_drained(self, senders, groups):
        bits = torch.tensor(
            [sender.drained() for sender in senders], dtype=torch.uint8, device="cpu"
        )
        for group in groups:
            if group is not None and torch.distributed.get_world_size(group) > 1:
                torch.distributed.all_reduce(
                    bits, op=torch.distributed.ReduceOp.MIN, group=group
                )
        return [bool(bit) for bit in bits.tolist()]

    def poll_drained(self, senders, groups):
        for sender in senders:
            sender.poll()
        return self.group_drained(senders, groups)

    def create_pipeline_transport(
        self, *, model_runner, pp_group, stage_group, world_group, depth
    ):
        from sglang_tpu.pp_scheduler import (
            TpuPipelineTransport,
            build_scheduler_pp_wave,
        )

        wave = build_scheduler_pp_wave(
            model_runner=model_runner,
            pp_group=pp_group,
            stage_group=stage_group,
            world_group=world_group,
        )
        self.pipeline_transport = TpuPipelineTransport(wave, depth)
        return self.pipeline_transport

    def close(self):
        if self.result_copier is not None:
            self.result_copier.close()
