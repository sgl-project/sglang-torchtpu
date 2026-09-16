"""Fuse Llama EAGLE draft extension with selected-row logits and sampling."""

import functools

import torch

from sglang.srt.environ import envs


def install_fused_draft_extend(policy) -> None:
    from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
    from sglang.srt.speculative.eagle_utils import (
        fused_draft_extend_select_argmax_func,
    )

    # Diagnostic probes inspect every logit row on the original worker path.
    if envs.SGLANG_ENABLE_ASYNC_ASSERT.get() or not isinstance(
        policy.owner, LlamaForCausalLMEagle3
    ):
        return
    owner = policy.owner
    tokens_per_req = policy.draft_token_num

    def draft_extend(forward_batch, counts, *args, **kwargs):
        spec = forward_batch.spec_info
        # The direct compiler's spec pytree retains only hidden_states. Counts
        # are explicit graph inputs; the window width is fixed by this policy.
        starts = torch.arange(
            0,
            counts.shape[0] * tokens_per_req,
            tokens_per_req,
            device=counts.device,
            dtype=torch.int64,
        )
        selector = starts + counts.to(torch.int64).clamp_min(1) - 1
        previous = spec.select_index
        spec.select_index = selector
        try:
            out = owner.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                *args,
                **kwargs,
            )
        finally:
            spec.select_index = previous
        out.tpu_draft_extend_outputs = (
            fused_draft_extend_select_argmax_func.__wrapped__(
                next_token_logits=out.next_token_logits,
                hidden_states=out.hidden_states,
                num_accept_tokens=counts,
                num_tokens_per_req=tokens_per_req,
                logits_are_selected=True,
            )
        )
        out.next_token_logits = None
        out.hidden_states = None
        return out

    compiled = policy._direct_compile_fused_draft_steps(draft_extend)
    shim = policy.compiled_forward_shim
    original = shim.fn

    @functools.wraps(original)
    def dispatch(input_ids, positions, forward_batch, *args, **kwargs):
        if (
            forward_batch.forward_mode.is_draft_extend_v2()
            and forward_batch.capture_hidden_mode.is_full()
        ):
            return compiled(
                forward_batch,
                forward_batch.spec_info.num_accept_tokens,
                *args,
                **kwargs,
            )
        return original(input_ids, positions, forward_batch, *args, **kwargs)

    shim.fn = dispatch
