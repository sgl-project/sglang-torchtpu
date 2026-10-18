# SPDX-License-Identifier: Apache-2.0
"""TPU draft execution shared by composed and compatibility EAGLE workers."""

from __future__ import annotations

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_flags
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.spec_utils import select_top_k_tokens
from sglang.srt.utils.async_probe import maybe_detect_nan
from sglang.srt.utils.common import fast_topk
from sglang_tpu.eagle_utils_tpu import (
    build_topk1_verify_chain_torch,
    fused_draft_extend_select_argmax_func,
)


class TpuEagleDraftWorker(EagleDraftWorker):
    def _fused_draft_forward(self, forward_batch: ForwardBatch):
        """Run all draft steps in one compiled call."""
        spec_info: EagleDraftInput = forward_batch.spec_info
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        maybe_detect_nan(topk_p, "fused_draft: NaN in initial topk_p")
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        attn_backends = self.draft_attn_backend.attn_backends
        # The loop derives step i's KV length as seq_lens + i + 1 in-graph
        # instead of stacking each step backend's eagerly built metadata. That
        # holds only while step i's backend carries speculative_step_id == i,
        # which is how TpuPallasRpaMultiStepDraftBackend constructs them.
        assert all(b.speculative_step_id == i for i, b in enumerate(attn_backends))
        out_cache_loc = per_step_draft_out_cache_loc(
            forward_batch.out_cache_loc,
            forward_batch.batch_size,
            self.topk,
            self.speculative_num_steps,
        )

        steps_fn = self._compiled_fused_steps
        if steps_fn is None:
            steps_fn = self.draft_runner.compile_policy.compile_fused_draft_steps(
                self._fused_draft_steps
            )
            self._compiled_fused_steps = steps_fn

        # The fused loop inlines CompiledForwardShim into an outer Dynamo
        # graph. Gate the batch's HiCache reload eagerly so the shim can omit
        # the mutable transfer counter while that graph is being traced.
        compiled_forward_shim = self.draft_runner.compile_policy.compiled_forward_shim
        if compiled_forward_shim is not None:
            compiled_forward_shim.wait_for_hicache_reload()

        # Draft decode is never a prefill-only operation. A request completing
        # chunked prefill may retain that scheduler marker. Pin the whole fused
        # loop to decode while preserving the scheduler-owned value afterward.
        orig_prefill_only = forward_batch.is_prefill_only
        forward_batch.is_prefill_only = False
        try:
            with forward_context(ForwardContext(attn_backend=attn_backends[0])):
                tree_outputs = steps_fn(
                    forward_batch,
                    topk_p,
                    topk_index,
                    hidden_states,
                    out_cache_loc,
                    self.hot_token_id,
                    spec_info.bonus_tokens,
                    forward_batch.seq_lens,
                )
        finally:
            forward_batch.is_prefill_only = orig_prefill_only
        spec_info.rpa_draft_kv_lens = None

        return tree_outputs

    def _should_fuse_draft_steps(self) -> bool:
        return (
            get_flags().capture.enable_torch_compile and self.speculative_num_steps > 1
        )

    def _fused_draft_steps(
        self,
        forward_batch: ForwardBatch,
        topk_p: torch.Tensor,
        topk_index: torch.Tensor,
        hidden_states: torch.Tensor,
        out_cache_loc: torch.Tensor,
        hot_token_id: torch.Tensor | None,
        bonus_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        """Run the compiled draft loop."""
        num_steps = self.speculative_num_steps
        model = self.draft_runner.model
        # The draft policy compiles this loop, so its tracer -- not the global
        # server arg -- decides whether hidden states may be rebound. The two
        # agree while tracer selection is global; a per-worker split would
        # silently desync them and drop the mutation below.
        draft_tracer = self.draft_runner.compile_policy.tracer
        base_positions = forward_batch.positions
        base_kv_lens = seq_lens.to(torch.int32)
        tokens: tuple = ()
        scores = None
        for i in range(num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            tokens += (tree_info[1],)
            if i == num_steps - 1:
                break

            forward_batch.spec_info.rpa_draft_kv_lens = base_kv_lens + i + 1
            if draft_tracer == "direct":
                # direct_compile treats explicit inputs as immutable graph
                # arguments. Rebind the traced ForwardBatch field instead of
                # turning its hidden-state input into a deferred in-place op.
                forward_batch.spec_info.hidden_states = hidden_states
            else:
                forward_batch.spec_info.hidden_states.copy_(hidden_states)
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            logits_output = model.forward(input_ids, base_positions + i, forward_batch)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if hot_token_id is not None:
                topk_index = hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states
        return build_topk1_verify_chain_torch(
            bonus_tokens,
            torch.cat(tokens, dim=1),
            seq_lens,
        )

    def _prepare_step_cache_loc(self, cache_loc: torch.Tensor) -> torch.Tensor:
        # XLA graph inputs cannot change storage offset between calls.
        # https://github.com/google-pytorch/torch_tpu/issues/3370
        return cache_loc.clone()

    def _draft_extend_select_index(
        self, accept_lens: torch.Tensor
    ) -> torch.Tensor | None:
        # Fused selection computes the indices; only the DSA seed gather needs them eagerly.
        if self.seed_dsa_topk_from_draft_extend:
            return super()._draft_extend_select_index(accept_lens)
        return None

    def _select_draft_extend_output(
        self,
        draft_logits_output,
        *,
        forward_batch,
        select_index,
        sampling_info,
        can_run_decode_cuda_graph,
    ):
        logits, hidden, topk_p, topk_index = (
            self.draft_runner.compile_policy.finalize_spec_forward_batch(
                outputs=fused_draft_extend_select_argmax_func(
                    draft_logits_output.next_token_logits,
                    draft_logits_output.hidden_states,
                    forward_batch.spec_info.num_accept_tokens,
                ),
                forward_batch=forward_batch,
            )
        )
        draft_logits_output.next_token_logits = logits
        draft_logits_output.hidden_states = hidden
        return None, topk_p, topk_index, hidden


class TpuEagleWorker(EAGLEWorkerV2):
    draft_worker_cls = TpuEagleDraftWorker
