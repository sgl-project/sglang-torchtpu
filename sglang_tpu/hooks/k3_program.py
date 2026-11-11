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
       defers a new prefill while requests decode for up to
       SGLANG_TPU_K3_PROGRAM_PREFILL_WAIT_ROUNDS scheduler rounds in a row
       (rounds, not seconds: every TP rank must take the same branch);
       otherwise the scheduler's own interval decides.
  Upstream PR or issue: <to be filed> (a scheduling policy seam for
       prefill/decode priority).
  Remove when: that seam exists.

k3_program_dp_allocator / k3_program_dp_prealloc / k3_program_dp_transfer_rows /
k3_program_dp_admission / k3_program_dp_release / k3_program_dp_group_cap
  Why: under attention DP2 (SGLANG_TPU_K3_PROGRAM_ATTENTION_DP=2) the decode's
       two DCP16 groups each hold their own requests' pages, while the
       prefill's DCP32 rows reach the decode rank of the same index
       (k3_programs/sglang_dp.py); core's allocator and transfer know one id
       space and one destination list for every rank.
  How: AROUND TpuSRTPlatform.get_paged_allocator_cls returns
       K3DpPagedAllocator; AROUND DecodePreallocQueue._pre_alloc places the
       request in a group and takes its landing pages in the other one;
       AROUND _transfer_page_indices gives each rank its destination rows;
       AROUND _required_alloc_tokens reserves the landing pages too; AFTER
       release_kv_cache releases the request's group and landing pages.
  Upstream PR or issue: <to be filed> (allocator regions and per-rank
       transfer destinations for attention DP on P/D).
  Remove when: those seams exist.

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
    import os

    if os.environ.get("K3_DP_DEBUG") == "1":
        from sglang_tpu.k3_programs import sglang_dp_debug

        sglang_dp_debug.install()
    from sglang.srt.runtime_context import get_model
    from sglang_tpu.k3_programs import sglang_hicache
    from sglang_tpu.k3_programs.sglang_pools import K3UnifiedKVPool

    # The host tier's stack for the unified radix cache built after the pools.
    sglang_hicache.install()
    from sglang_tpu.k3_programs.sglang_pools import is_k31

    configurator._cell_size = K3UnifiedKVPool.bytes_per_token(
        get_model().kv_cache_dtype,
        dsa=is_k31(getattr(getattr(kvc, "model_config", None), "hf_config", None)),
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
    from sglang_tpu.k3_programs.sglang_pools import K3StatePool, is_k31

    # The state slots' budget counts Kimi-K3.1's index-key rows.
    K3StatePool.dsa = is_k31(
        getattr(getattr(configurator, "model_config", None), "hf_config", None)
    )
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


def prefill_wait_defers(scheduler: Any, rounds: int) -> bool:
    """Whether a colocated engine holds its next prefill back: requests are
    decoding and requests wait, for at most ``rounds`` scheduler rounds in a
    row. Counted in rounds, not seconds: every TP rank must take the same
    branch (a rank prefilling while another decodes mismatches their
    collectives and hangs the slice), and the ranks see the same queues each
    round but not the same clocks."""
    running = scheduler.running_batch
    decoding = not (running is None or running.is_empty() or running.is_prefill_only)
    waiting = bool(scheduler.waiting_queue) or scheduler.chunked_req is not None
    held = getattr(scheduler, "_k3_prefill_wait_rounds", 0)
    if decoding and waiting and held < rounds:
        scheduler._k3_prefill_wait_rounds = held + 1
        return True
    scheduler._k3_prefill_wait_rounds = 0
    return False


def k3_program_prefill_wait(original, scheduler: Any, *args, **kwargs) -> bool:
    """AROUND Scheduler._should_defer_prefill."""
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang_tpu.envs import envs as tpu_envs

    rounds = tpu_envs.SGLANG_TPU_K3_PROGRAM_PREFILL_WAIT_ROUNDS.get()
    if (
        rounds > 0
        and k3_program_active()
        and scheduler.disaggregation_mode == DisaggregationMode.NULL
        and prefill_wait_defers(scheduler, rounds)
    ):
        return True
    return original(scheduler, *args, **kwargs)


def _dp2() -> bool:
    from sglang_tpu.envs import envs as tpu_envs

    return tpu_envs.SGLANG_TPU_K3_PROGRAM_ATTENTION_DP.get() == 2


def _dp_allocator(queue: Any):
    from sglang_tpu.k3_programs.sglang_dp_alloc import K3DpPagedAllocator

    allocator = queue.token_to_kv_pool_allocator
    return allocator if isinstance(allocator, K3DpPagedAllocator) else None


def k3_program_dp_allocator(original, platform: Any, *args, **kwargs) -> type:
    """AROUND TpuSRTPlatform.get_paged_allocator_cls (the P/D decode's
    regions; a colocated server keeps sglang's allocator, sglang_dp.ColoRows)."""
    from sglang_tpu.k3_programs.sglang_program import server_role

    if _dp2() and k3_program_active() and server_role() == "decode":
        from sglang_tpu.k3_programs.sglang_dp_alloc import K3DpPagedAllocator

        return K3DpPagedAllocator
    return original(platform, *args, **kwargs)


def k3_program_dp_prealloc(original, queue: Any, req: Any, *args, **kwargs):
    """AROUND DecodePreallocQueue._pre_alloc."""
    allocator = _dp_allocator(queue)
    if allocator is None:
        return original(queue, req, *args, **kwargs)
    from sglang_tpu.k3_programs import sglang_dp

    # A decode prefix hit (sglang_decode_cache) keeps the request in the
    # group holding its cached pages; only the suffix is transferred.
    prefix_indices = args[0] if args else kwargs.get("prefix_indices")
    prefix_len = int((args[1] if len(args) > 1 else kwargs.get("prefix_len")) or 0)
    group = None
    if prefix_len > 0 and prefix_indices is not None and len(prefix_indices):
        group = allocator.group_of_page(int(prefix_indices[0]) // allocator.page_size)
    group = allocator.assign(req.rid, group)
    allocator.pending_group = group
    try:
        kv_loc = original(queue, req, *args, **kwargs)
    finally:
        allocator.pending_group = None
    pages = -(-int(kv_loc.numel()) // allocator.page_size)
    landing = allocator.alloc_landing(1 - group, sglang_dp.landing_pages_needed(pages))
    if landing is None:
        raise RuntimeError(
            f"K3 decode DP2: no landing pages left in group {1 - group} for "
            f"request {req.rid} ({pages} pages)"
        )
    req._k3_dp_landing = landing
    return kv_loc


def k3_program_dp_transfer_rows(original, queue: Any, decode_req: Any, kv_indices, page_size):
    """AROUND DecodePreallocQueue._transfer_page_indices."""
    pages = original(queue, decode_req, kv_indices, page_size)
    allocator = _dp_allocator(queue)
    if allocator is None:
        return pages
    import numpy as np

    from sglang_tpu.k3_programs import sglang_dp

    req = decode_req.req
    landing = [int(p) for p in getattr(req, "_k3_dp_landing", ())]
    pool = allocator.get_kvcache()
    plan = getattr(pool, "plan", None) or pool.full_kv_pool.plan  # HybridLinearKVPool
    layout = sglang_dp.DpPages(allocator.dp_pages, plan.num_blocks)
    page_list = [int(p) for p in pages]
    req._k3_dp_transfer = (page_list, landing)
    rows = layout.transfer_rows(int(queue.tp_rank), page_list, landing)
    return np.asarray(rows, dtype=np.int32)


def k3_program_dp_admission(original, queue: Any, *args, **kwargs) -> int:
    """AROUND DecodePreallocQueue._required_alloc_tokens: landing pages too."""
    tokens = original(queue, *args, **kwargs)
    if _dp_allocator(queue) is None:
        return tokens
    page = queue.token_to_kv_pool_allocator.page_size
    pages = -(-tokens // page)
    return tokens + -(-pages // 2) * page


def dp_group_room(running, waiting, group_of, cap: int):
    """Split ``waiting`` (in order) into the requests a DP2 decode may start
    now and those held back: at most ``cap`` running requests a group
    (``group_of(req)``: its group, None: no group), so the compiled batch
    stays the smallest that holds both halves."""
    count = [0, 0]
    for req in running:
        g = group_of(req)
        if g is not None:
            count[g] += 1
    take, held = [], []
    for req in waiting:
        g = group_of(req)
        if g is None or count[g] < cap:
            take.append(req)
            if g is not None:
                count[g] += 1
        else:
            held.append(req)
    return take, held


def k3_program_dp_group_cap(original, scheduler: Any, running_batch: Any, *args, **kwargs):
    """AROUND SchedulerDisaggregationDecodeMixin._get_new_prebuilt_batch: a
    P/D DP2 decode starts a transferred request only while its group has
    room (half the running requests), as groups are fixed at prealloc."""
    from sglang_tpu.k3_programs import sglang_dp_alloc

    allocator = sglang_dp_alloc.active()
    if allocator is None or not scheduler.waiting_queue:
        return original(scheduler, running_batch, *args, **kwargs)
    cap = -(-int(scheduler.max_running_requests) // 2)
    running = list(getattr(running_batch, "reqs", None) or ())
    take, held = dp_group_room(
        running, list(scheduler.waiting_queue), lambda r: allocator.request_group.get(r.rid), cap
    )
    if not held:
        return original(scheduler, running_batch, *args, **kwargs)
    scheduler.waiting_queue = take
    try:
        return original(scheduler, running_batch, *args, **kwargs)
    finally:
        scheduler.waiting_queue = held + list(scheduler.waiting_queue)


def k3_program_hicache_layout(original, server_args: Any, *args, **kwargs):
    """AROUND arg_groups.hicache_hook.resolve_layout_io_compatibility: the
    K3 programs' host tier (k3_programs/sglang_hicache.py) copies their own
    unified pools over a TPU DMA engine, not an MHA fused pool or an RPA
    backend's; it needs the fused page layout and no L3 storage."""
    from sglang.srt.arg_groups.overrides import resolving_view

    cfg = resolving_view(server_args)
    if (
        str(getattr(cfg, "speculative_algorithm", None) or "").upper() != "K3_PROGRAM"
        or not cfg.enable_hierarchical_cache
    ):
        # (A P/D decode normalizes HiCache knobs without a host tier.)
        return original(server_args, *args, **kwargs)
    from sglang_tpu.tpu_dma_hicache import is_tpu_dma_hicache_backend

    if not is_tpu_dma_hicache_backend(cfg.hicache_io_backend):
        raise ValueError(
            f"K3_PROGRAM's host tier copies over a TPU DMA io backend (raiden), "
            f"not {cfg.hicache_io_backend!r}"
        )
    if cfg.hicache_mem_layout != "page_first_fused":
        raise ValueError("K3_PROGRAM's host tier needs --hicache-mem-layout page_first_fused")
    if cfg.hicache_storage_backend is not None:
        raise ValueError("K3_PROGRAM's host tier has no L3 storage path")
    return None


def k3_program_dp_release(result: Any, req: Any, *args, **kwargs) -> None:
    """AFTER mem_cache.common.release_kv_cache."""
    from sglang_tpu.k3_programs import sglang_dp_alloc, sglang_worker

    steps = sglang_worker.colo_steps()
    if steps is not None:
        # Colocated DP2: the request's decode-written halves back to the
        # DCP32 rows the radix cache now holds, before any prefill reads them.
        steps.colo_release(req.rid)
    allocator = sglang_dp_alloc.active()
    if allocator is None:
        return result
    landing = getattr(req, "_k3_dp_landing", None)
    transfer = getattr(req, "_k3_dp_transfer", None)
    if landing is not None and (transfer is None or len(transfer[1])):
        # Not swapped yet (the request left before its first decode step).
        allocator.free_landing(landing)
    req._k3_dp_landing = None
    allocator.release(req.rid)
    return result


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
            "sglang_tpu.tpu_platform.TpuSRTPlatform.get_paged_allocator_cls",
            k3_program_dp_allocator,
            HookType.AROUND,
        ),
        (
            "sglang.srt.disaggregation.decode.DecodePreallocQueue._pre_alloc",
            k3_program_dp_prealloc,
            HookType.AROUND,
        ),
        (
            "sglang.srt.disaggregation.decode.DecodePreallocQueue._transfer_page_indices",
            k3_program_dp_transfer_rows,
            HookType.AROUND,
        ),
        (
            "sglang.srt.disaggregation.decode.DecodePreallocQueue._required_alloc_tokens",
            k3_program_dp_admission,
            HookType.AROUND,
        ),
        (
            "sglang.srt.mem_cache.common.release_kv_cache",
            k3_program_dp_release,
            HookType.AFTER,
        ),
        (
            "sglang.srt.arg_groups.hicache_hook.resolve_layout_io_compatibility",
            k3_program_hicache_layout,
            HookType.AROUND,
        ),
        (
            "sglang.srt.disaggregation.decode.SchedulerDisaggregationDecodeMixin._get_new_prebuilt_batch",
            k3_program_dp_group_cap,
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
