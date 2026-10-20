# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 decode megakernel (experimental, opt-in).

attach_k3_megakernel, k3_megakernel_forward
  Why: the megakernel replaces the whole decoder stack forward (embedding output
       to final norm) with one multi-device Pallas call; the model has no
       backend seam for swapping its layer loop, and the kernel must stay out
       of core.
  How: AFTER KimiK3LinearForCausalLM.post_load_weights builds the per-rank
       runtime (kernel-layout weights read from the checkpoint, the exported
       calls). AROUND KimiK3LinearModel.forward sends bs=1 decode steps, bs=1
       extend (prefill) chunks and bs=1 DSpark target-verify steps to the
       kernel. Both pass through unless SGLANG_TPU_ENABLE_K3_DECODE_MEGAKERNEL=1
       when the model is loaded.
  Upstream PR or issue: <to be filed> (a model-level seam to replace a decoder
       stack's layers and forward).
  Remove when: that seam exists, or the megakernel is dropped.

skip_k3_decoder_layers, load_k3_non_layer_weights
  Why: the full model does not fit twice in HBM, so with the megakernel on it
       owns the decoder layers' weights (attention, MLP/MoE, per-layer norms)
       and sglang keeps only embed_tokens, the final attention-residual/norm
       weights, lm_head and the vision tower.
  How: AROUND sglang.srt.models.kimi_k3.make_pp_layers (the module-level name
       KimiK3LinearModel.__init__ calls, and nothing else in that module) builds
       every decoder layer as a PPMissingLayer (the pipeline-parallel
       placeholder that load_weights and post_load_weights already skip). AROUND KimiK3ForConditionalGeneration /
       KimiK3LinearForCausalLM.load_weights feeds only the non-layer tensors
       (the multimodal entry reads them from the checkpoint's index itself, so
       the decoder shards are never read).
       SGLANG_TPU_K3_MEGAKERNEL_KEEP_SGLANG_WEIGHTS=1 (dev only)
       keeps sglang's layers and its prefill path for A/B comparisons.
  Upstream PR or issue: <to be filed> (same seam).
  Remove when: as above.

prepare_k3_capture_kernels
  Why: DSpark sets the layers to capture after the model is built, and the
       kernel variants that capture them (prefill and verify) must exist before
       the compiled forward traces: dynamo cannot construct them.
  How: AFTER KimiK3LinearForCausalLM.set_dspark_layers_to_capture (the
       multimodal entry delegates to it) builds those variants, with one verify
       row per draft token plus the bonus row.
  Upstream PR or issue: <to be filed> (same seam).
  Remove when: as above.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    from sglang_tpu.envs import envs as tpu_envs

    return tpu_envs.SGLANG_TPU_ENABLE_K3_DECODE_MEGAKERNEL.get()


def _owns_weights() -> bool:
    from sglang_tpu.envs import envs as tpu_envs

    return (
        _enabled() and not tpu_envs.SGLANG_TPU_K3_MEGAKERNEL_KEEP_SGLANG_WEIGHTS.get()
    )


def skip_k3_decoder_layers(original, num_hidden_layers: int, layer_fn, *args, **kwargs):
    """AROUND sglang.srt.models.kimi_k3.make_pp_layers."""
    if not _owns_weights():
        return original(num_hidden_layers, layer_fn, *args, **kwargs)
    from sglang.srt.layers.utils import PPMissingLayer

    return original(
        num_hidden_layers, lambda idx, prefix: PPMissingLayer(), *args, **kwargs
    )


def _checkpoint_non_layer_tensors(model_path: str):
    """(name, tensor) for every checkpoint tensor outside the decoder layers, reading only
    the shards that hold them. ``model_path`` must be a local checkpoint directory (the
    megakernel reads the same files for its own weights)."""
    import json

    import safetensors

    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if ".layers." not in name or name.startswith(
            ("vision_tower.", "mm_projector.")
        ):
            by_shard.setdefault(shard, []).append(name)
    for shard in sorted(by_shard):
        with safetensors.safe_open(
            os.path.join(model_path, shard), framework="pt", device="cpu"
        ) as f:
            for name in sorted(by_shard[shard]):
                yield name, f.get_tensor(name)


def load_k3_non_layer_weights(original, causal_lm: Any, weights, *args, **kwargs):
    """AROUND KimiK3ForConditionalGeneration.load_weights and
    KimiK3LinearForCausalLM.load_weights."""
    if not _owns_weights():
        return original(causal_lm, weights, *args, **kwargs)
    from sglang.srt.layers.utils import get_layer_id

    if type(causal_lm).__name__ == "KimiK3ForConditionalGeneration":
        from sglang.srt.runtime_context import get_model

        # The loader's iterator is lazy: leaving it unconsumed skips reading the decoder shards.
        weights = _checkpoint_non_layer_tensors(get_model().model_path)
    else:
        weights = (w for w in weights if get_layer_id(w[0]) is None)
    return original(causal_lm, weights, *args, **kwargs)


def attach_k3_megakernel(result: None, causal_lm: Any, *args, **kwargs) -> None:
    """AFTER KimiK3LinearForCausalLM.post_load_weights."""
    if not _enabled():
        return
    from sglang_tpu.k3_megakernel.ops import build_runtime

    if _owns_weights():
        _check_server_args()
    causal_lm.model._k3_megakernel = build_runtime(
        causal_lm, owns_weights=_owns_weights()
    )
    causal_lm.model._k3_megakernel_owns_weights = _owns_weights()
    # The kernel holds its own weight copies read from the checkpoint at load time; the
    # weight updater's derived-cache guard rejects every online update path.
    causal_lm.model._derived_weight_cache_error = (
        "Online weight updates are not supported with "
        "SGLANG_TPU_ENABLE_K3_DECODE_MEGAKERNEL=1: the megakernel reads weight copies "
        "built at load time. Restart the server to change weights."
    )


# Serving layout the kernel is built for (sglang's TPU MLA pool: 128-token pages, bf16 latent).
_PAGE_SIZE = 128
_ATTENTION_BACKENDS = ("pallas_tpu_mla",)
_KV_CACHE_DTYPES = ("auto", "bf16", "bfloat16")


def prepare_k3_capture_kernels(
    result: None, causal_lm: Any, layer_ids, *args, **kwargs
) -> None:
    """AFTER KimiK3LinearForCausalLM.set_dspark_layers_to_capture."""
    runtime = causal_lm.model.__dict__.get("_k3_megakernel")
    if runtime is None:
        return
    from sglang.srt.runtime_context import get_spec

    # Verify rows per request: the DSpark block plus the bonus row.
    rows = get_spec().speculative_num_draft_tokens
    runtime.prepare(tuple(sorted(layer_ids)), verify_rows=rows)


def _check_server_args() -> None:
    """The megakernel owns the decoder weights, so every forward must be one it can run: one
    request at a time, no speculation other than DSpark, no radix reuse of a prefix whose KDA state the
    kernel did not see (the token loop resets the state only for a request without a prefix),
    and the pool layout the kernel reads in place."""
    from sglang.srt.runtime_context import (
        get_exec,
        get_memory,
        get_model,
        get_parallel,
        get_schedule,
        get_spec,
    )

    schedule, parallel = get_schedule(), get_parallel()
    problems = []
    if schedule.max_running_requests != 1:
        problems.append(
            f"--max-running-requests 1 (got {schedule.max_running_requests})"
        )
    if not get_memory().disable_radix_cache:
        problems.append("--disable-radix-cache")
    if get_spec().speculative_algorithm not in (None, "DSPARK"):
        problems.append(
            "no speculative decoding other than DSPARK "
            f"(got {get_spec().speculative_algorithm})"
        )
    if schedule.enable_mixed_chunk:
        problems.append("no --enable-mixed-chunk")
    chunk = schedule.chunked_prefill_size
    if chunk is None or chunk <= 0:
        problems.append(
            f"--chunked-prefill-size > 0 (the kernel's prefill rows; got {chunk})"
        )
    if parallel.pp_size != 1 or parallel.dp_size != 1:
        problems.append(
            f"pp_size = dp_size = 1 (got {parallel.pp_size}, {parallel.dp_size})"
        )
    if parallel.attn_cp_size != 1 or parallel.dcp_size != 1:
        problems.append(
            f"no context parallelism (attn_cp_size {parallel.attn_cp_size}, "
            f"dcp_size {parallel.dcp_size})"
        )
    if schedule.page_size != _PAGE_SIZE:
        problems.append(f"--page-size {_PAGE_SIZE} (got {schedule.page_size})")
    if get_model().kv_cache_dtype not in _KV_CACHE_DTYPES:
        problems.append(f"bf16 KV cache (got {get_model().kv_cache_dtype})")
    backend = get_exec().kernel.attention_backend
    if backend not in _ATTENTION_BACKENDS:
        problems.append(f"--attention-backend pallas_tpu_mla (got {backend})")
    compile_bs = get_exec().graph.tpu_compile_bs
    if compile_bs not in (None, [1]):
        problems.append(f"TPU compile batch sizes [1] (got {compile_bs})")
    if problems:
        raise ValueError(
            "SGLANG_TPU_ENABLE_K3_DECODE_MEGAKERNEL=1 (the megakernel owns the decoder weights) "
            "needs: " + "; ".join(problems)
        )


def _eligible(
    model: Any, input_ids, forward_batch, inputs_embeds, pp_proxy_tensors
) -> bool:
    mode = forward_batch.forward_mode
    owns = model.__dict__.get("_k3_megakernel_owns_weights", False)
    if pp_proxy_tensors is not None:
        return False
    # Embeddings (multimodal prefill, the VL warmup) only when sglang has no layers to run them.
    if input_ids is None and not (owns and inputs_embeds is not None):
        return False
    if inputs_embeds is not None and not owns:
        return False
    if mode.is_decode():
        return (input_ids if input_ids is not None else inputs_embeds).shape[0] == 1
    if mode.is_target_verify():  # DSpark: one request's verify rows
        return owns and _real_batch_size(forward_batch) == 1
    # Prefill through the kernel only when sglang has no layer weights to run it.
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return owns and mode == ForwardMode.EXTEND and _real_batch_size(forward_batch) == 1


def _real_batch_size(forward_batch) -> int:
    # The TPU runner pads batches to a bucket and records the pre-bucket size.
    return int(getattr(forward_batch, "_tpu_real_b", forward_batch.batch_size))


def k3_megakernel_forward(
    original,
    model: Any,
    input_ids,
    positions,
    forward_batch,
    inputs_embeds=None,
    pp_proxy_tensors=None,
):
    """AROUND KimiK3LinearModel.forward."""
    runtime = model.__dict__.get("_k3_megakernel")
    if runtime is None:
        return original(
            model, input_ids, positions, forward_batch, inputs_embeds, pp_proxy_tensors
        )
    if not _eligible(model, input_ids, forward_batch, inputs_embeds, pp_proxy_tensors):
        if model.__dict__.get("_k3_megakernel_owns_weights", False):
            raise NotImplementedError(
                f"K3 megakernel owns the decoder weights and runs only bs=1 decode / extend / "
                f"DSpark verify; got "
                f"{forward_batch.forward_mode} with {_real_batch_size(forward_batch)} requests, "
                f"{(input_ids if input_ids is not None else inputs_embeds).shape[0]} tokens."
            )
        return original(
            model, input_ids, positions, forward_batch, inputs_embeds, pp_proxy_tensors
        )
    return runtime.forward(model, input_ids, forward_batch, inputs_embeds)
