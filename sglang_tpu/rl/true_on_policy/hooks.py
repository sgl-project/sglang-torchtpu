# SPDX-License-Identifier: Apache-2.0
"""True on-policy mode: engine ops whose bytes could depend on the batch run batch-invariant kernels.

Registered only with SGLANG_TPU_ENABLE_TRUE_ON_POLICY=1.

Why: an RL trainer recomputes the rollout's log-probs in one packed forward; they equal the engine's only if
     every op gives a row the same bytes in any batch.
How: AROUND the unquantized linear and the lm_head (matmul), the RMSNorm variants (norm kernels),
     fused_conv1d_gdn (per-token recurrence), fused_moe_call (slot-order combine), Sampler.forward and
     InputLogprobProcessor.forward (log-prob kernel); AFTER batched RPA calculate_block_sizes (KV block = page)
     and SamplingParams.verify (refuses logit transforms). Not covered: TP above 1, other attention backends,
     quantized weights.
Upstream PR or issue: not filed (prototype).
Remove when: core exposes batch-invariant op seams on TPU.
"""

from __future__ import annotations

import inspect
from typing import Any

import jax.numpy as jnp
import torch

from sglang_tpu.envs import envs
from sglang_tpu.rl.true_on_policy import (
    attention_kernel,
    logprob_kernel,
    matmul_kernel,
    moe_kernel,
    norm_kernel,
)


def true_on_policy_linear(original, self, layer, x: torch.Tensor, bias=None):
    """AROUND UnquantizedLinearMethod.apply: the fixed-tile matmul for every bf16 linear."""
    if x.device.type != "tpu":
        return original(self, layer, x, bias)
    if (
        bias is not None
        or x.dtype != torch.bfloat16
        or layer.weight.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"TPU true on-policy linear needs bf16 x and weight and no bias, got {x.dtype=} "
            f"{layer.weight.dtype=} bias={bias is not None}"
        )
    out = matmul_kernel.matmul(x.reshape(-1, x.shape[-1]), layer.weight)
    return out.reshape(*x.shape[:-1], out.shape[-1])


def true_on_policy_lm_head(
    original, self, hidden_states: torch.Tensor, lm_head, embedding_bias=None
):
    """AROUND LogitsProcessor._compute_lm_head_rows: bf16 logits from the fixed-tile matmul."""
    if hidden_states.device.type != "tpu":
        return original(self, hidden_states, lm_head, embedding_bias)
    if embedding_bias is not None or lm_head.weight.dtype != torch.bfloat16:
        raise ValueError(
            f"TPU true on-policy lm_head needs an unquantized bf16 head without bias, got "
            f"{lm_head.weight.dtype=} bias={embedding_bias is not None}"
        )
    return matmul_kernel.matmul(hidden_states.to(torch.bfloat16), lm_head.weight)


def true_on_policy_rms_norm(
    original,
    self,
    x: torch.Tensor,
    residual=None,
    post_residual_addition=None,
    quant_linear=None,
):
    """AROUND RMSNorm.forward_native: the fixed-tile norm, rounding where forward_native rounds."""
    if x.device.type != "tpu":
        return original(self, x, residual, post_residual_addition, quant_linear)
    if (
        post_residual_addition is not None
        or self.variance_size_override is not None
        or self.cast_x_before_out_mul
        or self.fp32_residual
        or self.override_orig_dtype is not None
        or x.dtype != torch.bfloat16
        or self.weight.dtype != torch.bfloat16
    ):
        raise ValueError(
            "TPU true on-policy RMSNorm covers the plain bf16 norm (with or without residual) only"
        )
    return _fixed_tile_norm(x, residual, self.weight, self.variance_epsilon)


def true_on_policy_gemma_rms_norm(
    original, self, x: torch.Tensor, residual=None, post_residual_addition=None
):
    """AROUND GemmaRMSNorm.forward_native: the fixed-tile norm with the Gemma weight and rounding."""
    if x.device.type != "tpu":
        return original(self, x, residual, post_residual_addition)
    if (
        post_residual_addition is not None
        or x.dtype != torch.bfloat16
        or self.weight.dtype != torch.bfloat16
    ):
        raise ValueError(
            "TPU true on-policy GemmaRMSNorm covers the bf16 norm (with or without residual) only"
        )
    return _fixed_tile_norm(x, residual, self.weight, self.variance_epsilon, gemma=True)


def _fixed_tile_norm(x, residual, weight, eps, *, gemma=False):
    h = x.shape[-1]
    flat = x.reshape(-1, h)
    if residual is None:
        out = norm_kernel.rms_norm(flat, None, weight, eps, gemma=gemma)
        return out.reshape(x.shape)
    out, new_residual = norm_kernel.rms_norm(
        flat, residual.reshape(-1, h), weight, eps, gemma=gemma
    )
    return out.reshape(x.shape), new_residual.reshape(x.shape)


def true_on_policy_gated_rms_norm(original, self, x: torch.Tensor, z=None):
    """AROUND RMSNormGated.forward: the fixed-tile norm of x, gated by silu(z), rounded once."""
    if x.device.type != "tpu":
        return original(self, x, z)
    if z is None or x.dtype != torch.bfloat16:
        raise ValueError(
            "TPU true on-policy RMSNormGated covers the bf16 norm with its gate only"
        )
    h = x.shape[-1]
    out = norm_kernel.gated_rms_norm(
        x.reshape(-1, h), z.reshape(-1, h), self.weight, self.eps
    )
    return out.reshape(x.shape)


def recurrent_gated_delta_net(original, *args, **kwargs):
    """AROUND fused_conv1d_gdn: every tile runs the per-token recurrence, as a decode step does."""
    call = inspect.signature(original).bind(*args, **kwargs)
    a_log = call.arguments["a_log"]
    # A trainer's forward sees A_log in bf16.
    call.arguments["a_log"] = a_log.astype(jnp.bfloat16).astype(a_log.dtype)
    return original(*call.args, **{**call.kwargs, "recurrent_scan": True})


def true_on_policy_moe(original, **kwargs):
    """AROUND fused_moe_call: one kernel for every row count, its combine in slot order on the TensorCore."""
    moe_kernel.refuse_moe_rounding_flags()
    with (
        envs.SGLANG_TPU_FUSED_MOE_SC_GATHER.override(False),
        envs.SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD.override(0),
        envs.SGLANG_TPU_FORCE_MOE_GATHER_PERMUTE_WITH_SC_COMBINE.override(False),
        # A single-row call would otherwise take its own kernel.
        envs.SGLANG_TPU_ENABLE_SINGLE_TOKEN_MOE_KERNEL.override(False),
    ):
        return original(**kwargs)


def true_on_policy_sampler(
    original, self, logits_output, sampling_info, return_logprob, *args, **kwargs
):
    """AROUND Sampler.forward: sample as before; the returned token log-probs come from the log-prob kernel."""
    if not return_logprob:
        return original(
            self, logits_output, sampling_info, return_logprob, *args, **kwargs
        )
    # The sampler scales and softmaxes the logits in place.
    logits = logits_output.next_token_logits.clone()
    token_ids = original(
        self, logits_output, sampling_info, return_logprob, *args, **kwargs
    )
    # An all-greedy batch keeps its sampling parameters on the host.
    temperature = sampling_info.temperatures.reshape(-1).to(
        logits.device, torch.float32
    )
    logits_output.next_token_logprobs = logprob_kernel.logprob(
        logits, temperature, token_ids.to(torch.int32)
    )
    return token_ids


def true_on_policy_input_logprobs(
    original,
    self,
    pruned_states,
    sample_indices,
    input_logprob_indices,
    token_to_seq_idx,
    lm_head,
    get_logits_fn,
    logits_metadata,
    *args,
    **kwargs,
):
    """AROUND InputLogprobProcessor.forward: the prompt token log-probs come from the log-prob kernel."""
    result, sampled_logits = original(
        self,
        pruned_states,
        sample_indices,
        input_logprob_indices,
        token_to_seq_idx,
        lm_head,
        get_logits_fn,
        logits_metadata,
        *args,
        **kwargs,
    )
    targets = logits_metadata.extend_input_logprob_token_ids_gpu.to(torch.int32)
    token_logprobs = []
    # The chunk only bounds the [rows, vocab] logits.
    for start in range(0, input_logprob_indices.shape[0], self.logprobs_chunk_size):
        chunk = slice(start, start + self.logprobs_chunk_size)
        logits = get_logits_fn(
            pruned_states[input_logprob_indices[chunk]],
            lm_head,
            logits_metadata,
            use_logits_buffer=False,
        )
        # Prompt log-probs carry no temperature.
        ones = torch.ones(logits.shape[0], dtype=torch.float32, device=logits.device)
        token_logprobs.append(logprob_kernel.logprob(logits, ones, targets[chunk]))
    result.token_logprobs = torch.cat(token_logprobs)
    return result, sampled_logits


def refuse_transformed_sampling(result, self, vocab_size) -> Any:
    """AFTER SamplingParams.verify: the returned log-prob is log_softmax(logits / temperature); refuse the rest."""
    from sglang.srt.sampling.sampling_params import TOP_K_ALL

    transforms = {
        "top_p": self.top_p < 1.0,
        # greedy is top_k 1
        "top_k": 1 < self.top_k < TOP_K_ALL,
        "min_p": self.min_p > 0.0,
        "penalties": (
            self.frequency_penalty,
            self.presence_penalty,
            self.repetition_penalty,
        )
        != (0.0, 0.0, 1.0),
        "logit_bias": self.logit_bias is not None,
        # masks the stop tokens' logits until the minimum length
        "min_new_tokens": self.min_new_tokens > 0,
        "a grammar": any(
            g is not None
            for g in (self.json_schema, self.regex, self.ebnf, self.structural_tag)
        ),
    }
    if refused := [name for name, on in transforms.items() if on]:
        raise ValueError(
            f"SGLANG_TPU_ENABLE_TRUE_ON_POLICY serves plain temperature sampling; refused: {', '.join(refused)}. "
            "Chat requests take top_p and top_k from the model's generation config unless the server runs "
            "--sampling-defaults openai."
        )
    return result


def pin_kv_block(result, model_cfgs, serve_cfgs, vmem_limit_bytes) -> Any:
    """AFTER batched RPA calculate_block_sizes: one KV block (the page) for decode and prefill alike."""
    attention_kernel.refuse_rpa_rounding_flags()
    return attention_kernel.pin_kv_block(*result, serve_cfgs)


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply",
            true_on_policy_linear,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head_rows",
            true_on_policy_lm_head,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.layernorm.RMSNorm.forward_native",
            true_on_policy_rms_norm,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.layernorm.GemmaRMSNorm.forward_native",
            true_on_policy_gemma_rms_norm,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.attention.native_torch.layernorm_gated.RMSNormGated.forward",
            true_on_policy_gated_rms_norm,
            HookType.AROUND,
        ),
        (
            "sglang_tpu.attention.pallas.gdn.fused_conv1d_gdn",
            recurrent_gated_delta_net,
            HookType.AROUND,
        ),
        (
            "sglang_tpu.moe.pallas.fused_moe_op.fused_moe_call",
            true_on_policy_moe,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.sampler.Sampler.forward",
            true_on_policy_sampler,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.logprob_processor.InputLogprobProcessor.forward",
            true_on_policy_input_logprobs,
            HookType.AROUND,
        ),
        (
            "sglang_tpu.attention.pallas.batched_rpa.wrapper.calculate_block_sizes",
            pin_kv_block,
            HookType.AFTER,
        ),
        (
            "sglang.srt.sampling.sampling_params.SamplingParams.verify",
            refuse_transformed_sampling,
            HookType.AFTER,
        ),
    ]
