# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 megakernel programs (``--speculative-algorithm K3_PROGRAM``).

allow_k3_program_spec
  Why: core rejects every speculative algorithm for Kimi-K3 on TPU (no KDA
       target-verify path); K3_PROGRAM verifies DSpark drafts inside the decode
       program and never runs sglang's KDA verify.
  How: AROUND sglang.srt.arg_groups.kimi_k3_hook.apply_kimi_k3_spec_backend_defaults
       returns without the check under K3_PROGRAM; any other algorithm keeps it.
  Upstream PR or issue: <to be filed> (a platform seam for which speculative
       algorithms a hybrid model supports).
  Remove when: that seam exists.

k3_program_kv_layer_ids
  Why: the unified pools register the DSpark draft's five KV pools after the
       24 MLA pools, and HybridLinearKVPool derives one layer id per entry from
       the full-attention layers alone, which P/D pairing rejects (29 entries,
       24 ids).
  How: AROUND sglang.srt.mem_cache.memory_pool.HybridLinearKVPool.get_kv_layer_ids
       returns K3UnifiedKVPool's ids (the MLA layers, then the draft band
       after the 93 layers) when that is the full-attention pool.
  Upstream PR or issue: <to be filed> (a pool-reported layer id list).
  Remove when: HybridLinearKVPool asks its full-attention pool for the ids.

attach_k3_program
  Why: the program owns the decoder weights (pre-sharded, read at load) and
       must be resident before sglang sizes the KV pools from free HBM; the
       model has no seam for a whole-model replacement.
  How: AFTER KimiK3LinearForCausalLM.post_load_weights builds the program for
       the server's P/D role and loads its weights (sglang_program). The
       decoder layers are never built: hooks/k3_megakernel's weight
       ownership covers K3_PROGRAM (skip_k3_decoder_layers,
       load_k3_non_layer_weights).
  Upstream PR or issue: <to be filed> (a model-level seam to replace a decoder
       stack's layers and forward).
  Remove when: that seam exists.

size_k3_program_pools
  Why: DefaultPoolConfigurator sizes the full-attention pool from the MLA
       latent's bytes per token on every rank (about 27.6 KB for Kimi-K3); the
       unified pools hold a rank's share of each logical block (1160 bytes a
       token in bf16), so sized from free memory the pool would come out about
       24 times too small.
  How: AFTER DefaultPoolConfigurator.__init__ sets the cell size from
       K3UnifiedKVPool.bytes_per_token under K3_PROGRAM (the state slots are
       costed through K3StatePool.cache_bytes_per_request).
  Upstream PR or issue: sgl-project/sglang-torchtpu#1072 (the configurator
       asking the platform pool for its bytes per token).
  Remove when: that lands.

k3_program_no_spec_state_reserve
  Why: with a speculative algorithm the KV cache configurator reserves, ahead
       of the KV pool, sglang's intermediate SSM state for each capped running
       request times the draft tokens (its own KDA target-verify scratch);
       K3_PROGRAM verifies inside the decode program and never allocates it.
       On hardware the reserve took ~5 GB a rank on colocated engines at
       AgentX concurrency 8 and left a P/D prefill at concurrency 32 no KV
       ("Loaded weights leave no GPU memory for the KV cache").
  How: AROUND sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator
       ._handle_max_mamba_cache sizes the state pool as without speculation
       under K3_PROGRAM (the configurator's spec_algorithm reads NONE for the
       call).
  Upstream PR or issue: <to be filed> (a speculative worker's say in the
       state scratch it needs).
  Remove when: that seam exists.

k3_program_decode_tree
  Why: a P/D decode's prefix cache needs no KDA states under K3_PROGRAM (the
       transfer delivers each request's state into its slot; a state copied
       from the tree would race it), but sglang's tree keeps Mamba state for
       every hybrid model.
  How: AROUND sglang.srt.mem_cache.registry.create_unified_radix_cache installs
       k3_programs.sglang_decode_cache's stateless Mamba component on the
       decode role with its prefix cache on.
  Upstream PR or issue: <to be filed> (a model's choice of tree components).
  Remove when: that choice exists.

reject_unsupported_k3_requests
  Why: the programs return tokens only and keep each request's next input on
       the device, so a request asking for logprobs, hidden states or
       structured output cannot be served; raised inside the step it would
       stop the scheduler.
  How: AROUND sglang.srt.managers.scheduler.Scheduler._add_request_to_queue
       finishes such a request with an abort (an error to its client) under
       K3_PROGRAM before it is queued, as the scheduler does for requests
       DFlash cannot serve.
  Upstream PR or issue: <to be filed> (a speculative algorithm's request
       validation seam).
  Remove when: that seam exists.

k3_program_prefill_wait
  Why: AgentX's interactivity averages each request's inter-token latency, and
       a colocated engine's prefill batch (0.3-1 s a chunk) stops every
       decoding request: a stall in a short answer multiplies its latency
       (ITL mean 3.11 ms against a 2.79 ms median at concurrency 4 on
       hardware), while the trace's sessions sit idle most of the time.
  How: AROUND sglang.srt.managers.scheduler.Scheduler._should_defer_prefill
       defers a new prefill while requests decode until the oldest waiting
       (or chunked) request has waited SGLANG_TPU_K3_PROGRAM_PREFILL_WAIT_S;
       otherwise the scheduler's own interval decides.
  Upstream PR or issue: <to be filed> (a scheduling policy seam for
       prefill/decode priority).
  Remove when: that seam exists.

k3_program_prefill_share
  Why: a colocated engine's prefill batches stop its decoding requests, and
       --prefill-decode-interval defers the next prefill by a fixed number of
       decode rounds whatever the batch cost: 16 rounds after each ~0.4 s
       chunk of a cold prompt halved decoding during bursts on hardware
       (ITL 9.8 ms at AgentX concurrency 8), while a short turn's batch
       needs few.
  How: AROUND sglang.srt.managers.scheduler.Scheduler
       ._arm_prefill_decode_interval sets the deferral from the batch's
       extend tokens and the running batch so prefill takes
       SGLANG_TPU_K3_PROGRAM_PREFILL_SHARE of the engine while requests
       decode (none when nothing decodes); from batch contents only, so every
       rank defers alike. Off (0): the fixed interval.
  Upstream PR or issue: <to be filed> (a cost-aware prefill/decode cadence).
  Remove when: that cadence exists.

skip_k3_program_backends, skip_k3_program_graphs
  Why: the program runs the whole model, so sglang's forward never runs, but
       the model runner still builds the KDA/MLA attention backends over the
       unified pools (which are not TpuMambaPool or MLA token pools) and
       captures graphs whose warmup runs that forward.
  How: AROUND sglang.srt.model_executor.model_runner.ModelRunner
       .init_attention_backends and .init_cuda_graphs return without building
       under K3_PROGRAM, leaving no backends or graph runners (the compile
       policy stays the runner's no-op one).
  Upstream PR or issue: <to be filed> (a model-runner seam for a worker that
       owns the forward).
  Remove when: that seam exists.
"""

from __future__ import annotations

from typing import Any


def k3_program_selected(server_args: Any) -> bool:
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang_tpu.k3_programs.sglang_worker import ALGORITHM

    algorithm = resolving_view(server_args).speculative_algorithm
    return str(algorithm or "").upper() == ALGORITHM


def allow_k3_program_spec(original, server_args: Any, *args, **kwargs) -> None:
    """AROUND apply_kimi_k3_spec_backend_defaults."""
    if k3_program_selected(server_args):
        return None
    return original(server_args, *args, **kwargs)


def k3_program_kv_layer_ids(original, pool: Any, *args, **kwargs) -> list[int]:
    """AROUND HybridLinearKVPool.get_kv_layer_ids."""
    from sglang_tpu.k3_programs.sglang_pools import K3UnifiedKVPool

    full = getattr(pool, "full_kv_pool", None)
    if isinstance(full, K3UnifiedKVPool):
        return full.get_kv_layer_ids()
    return original(pool, *args, **kwargs)


def k3_program_active() -> bool:
    """Whether this server runs K3_PROGRAM (from the runtime context; no
    published configuration, as before server arguments resolve: no)."""
    from sglang.srt.runtime_context import get_context, get_spec
    from sglang_tpu.k3_programs.sglang_worker import ALGORITHM

    if not get_context().is_config_namespace_published("spec"):
        return False
    return str(get_spec().speculative_algorithm or "").upper() == ALGORITHM


def attach_k3_program(result: None, causal_lm: Any, *args, **kwargs) -> None:
    """AFTER KimiK3LinearForCausalLM.post_load_weights."""
    if not k3_program_active():
        return
    from sglang_tpu.k3_programs.sglang_program import build_from_server

    causal_lm.model._k3_program = build_from_server()
    causal_lm.model._derived_weight_cache_error = (
        "Online weight updates are not supported with K3_PROGRAM: the program "
        "reads pre-sharded weights at load time. Restart the server to change "
        "weights."
    )


def size_k3_program_pools(result: None, configurator, kvc, *args, **kwargs) -> None:
    """AFTER DefaultPoolConfigurator.__init__."""
    if not k3_program_active():
        return result
    from sglang.srt.runtime_context import get_model
    from sglang_tpu.k3_programs.sglang_pools import K3UnifiedKVPool

    configurator._cell_size = K3UnifiedKVPool.bytes_per_token(
        get_model().kv_cache_dtype
    )
    return result


def k3_program_decode_tree(original, ctx: Any, *args, **kwargs):
    """AROUND create_unified_radix_cache."""
    from sglang.srt.runtime_context import get_disagg

    disagg = get_disagg()
    if (
        k3_program_active()
        and disagg.disaggregation_mode == "decode"
        and disagg.disaggregation_decode_enable_radix_cache
        and not ctx.disable_radix_cache
    ):
        from sglang_tpu.k3_programs.sglang_decode_cache import install

        install(ctx.params)
    return original(ctx, *args, **kwargs)


def k3_program_unsupported(req: Any) -> str | None:
    """Why K3_PROGRAM cannot serve ``req``, or None."""
    if getattr(req, "return_logprob", False):
        return "logprobs"
    if getattr(req, "return_hidden_states", False):
        return "hidden states"
    params = getattr(req, "sampling_params", None)
    if params is not None and any(
        getattr(params, name, None)
        for name in ("json_schema", "regex", "ebnf", "structural_tag")
    ):
        return "structured output"
    return None


def reject_unsupported_k3_requests(
    original, scheduler: Any, req: Any, *args, **kwargs
) -> None:
    """AROUND Scheduler._add_request_to_queue."""
    if k3_program_active() and getattr(req, "finished_reason", None) is None:
        reason = k3_program_unsupported(req)
        if reason is not None:
            req.set_finish_with_abort(
                f"--speculative-algorithm K3_PROGRAM serves without {reason}"
            )
    return original(scheduler, req, *args, **kwargs)


def k3_program_no_spec_state_reserve(original, configurator: Any, *args, **kwargs):
    """AROUND KVCacheConfigurator._handle_max_mamba_cache."""
    if not k3_program_active():
        return original(configurator, *args, **kwargs)
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algorithm = configurator.spec_algorithm
    configurator.spec_algorithm = SpeculativeAlgorithm.NONE
    try:
        return original(configurator, *args, **kwargs)
    finally:
        configurator.spec_algorithm = algorithm


def k3_program_prefill_share(original, scheduler: Any, batch: Any, *args, **kwargs):
    """AROUND Scheduler._arm_prefill_decode_interval."""
    result = original(scheduler, batch, *args, **kwargs)
    from sglang_tpu.envs import envs as tpu_envs

    share = tpu_envs.SGLANG_TPU_K3_PROGRAM_PREFILL_SHARE.get()
    if (
        not share
        or batch is None
        or not scheduler.prefill_decode_interval
        or not batch.forward_mode.is_extend()
        or not k3_program_active()
    ):
        return result
    from sglang_tpu.k3_programs.sglang_worker import prefill_deferral_rounds

    running = scheduler.running_batch
    scheduler._prefill_decode_interval_remaining = prefill_deferral_rounds(
        int(batch.extend_num_tokens or 0),
        len(running.reqs) if running is not None else 0,
        share,
    )
    return result


def prefill_wait_defers(scheduler: Any, wait_s: float, now: float) -> bool:
    """Whether a colocated engine holds its next prefill back: requests are
    decoding and the oldest waiting (or chunked) request has waited less than
    ``wait_s`` seconds."""
    running = scheduler.running_batch
    if running is None or running.is_empty() or running.is_prefill_only:
        return False
    pending = list(scheduler.waiting_queue)
    if scheduler.chunked_req is not None:
        pending.append(scheduler.chunked_req)
    entries = [
        req.time_stats.wait_queue_entry_time
        for req in pending
        if req.time_stats.wait_queue_entry_time
    ]
    return bool(entries) and now - min(entries) < wait_s


def k3_program_prefill_wait(original, scheduler: Any, *args, **kwargs) -> bool:
    """AROUND Scheduler._should_defer_prefill."""
    import time

    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang_tpu.envs import envs as tpu_envs

    wait_s = tpu_envs.SGLANG_TPU_K3_PROGRAM_PREFILL_WAIT_S.get()
    if (
        wait_s > 0
        and k3_program_active()
        and scheduler.disaggregation_mode == DisaggregationMode.NULL
        and prefill_wait_defers(scheduler, wait_s, time.perf_counter())
    ):
        return True
    return original(scheduler, *args, **kwargs)


def skip_k3_program_backends(original, model_runner: Any, *args, **kwargs) -> None:
    """AROUND ModelRunner.init_attention_backends."""
    if not k3_program_active():
        return original(model_runner, *args, **kwargs)
    model_runner.attn_backend = None
    model_runner.decode_attn_backend = None
    model_runner.decode_attn_backend_group = []
    return None


def skip_k3_program_graphs(original, model_runner: Any, *args, **kwargs) -> None:
    """AROUND ModelRunner.init_cuda_graphs."""
    if not k3_program_active():
        return original(model_runner, *args, **kwargs)
    model_runner.eager_runner = None
    model_runner.prefill_cuda_graph_runner = None
    model_runner.decode_cuda_graph_runner = None
    return None


def hook_specs() -> list[tuple]:
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.arg_groups.kimi_k3_hook.apply_kimi_k3_spec_backend_defaults",
            allow_k3_program_spec,
            HookType.AROUND,
        ),
        (
            "sglang.srt.mem_cache.memory_pool.HybridLinearKVPool.get_kv_layer_ids",
            k3_program_kv_layer_ids,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.post_load_weights",
            attach_k3_program,
            HookType.AFTER,
        ),
        (
            "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator.__init__",
            size_k3_program_pools,
            HookType.AFTER,
        ),
        (
            "sglang.srt.mem_cache.registry.create_unified_radix_cache",
            k3_program_decode_tree,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.scheduler.Scheduler._add_request_to_queue",
            reject_unsupported_k3_requests,
            HookType.AROUND,
        ),
        (
            "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._handle_max_mamba_cache",
            k3_program_no_spec_state_reserve,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.scheduler.Scheduler._arm_prefill_decode_interval",
            k3_program_prefill_share,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.scheduler.Scheduler._should_defer_prefill",
            k3_program_prefill_wait,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.model_runner.ModelRunner.init_attention_backends",
            skip_k3_program_backends,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.model_runner.ModelRunner.init_cuda_graphs",
            skip_k3_program_graphs,
            HookType.AROUND,
        ),
    ]
