"""TPU decoder compile policies for annotation-driven torch.compile."""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable, Optional

import torch
import tqdm

from sglang.srt.compilation.support_torch_compile import CompilePolicy
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
    _REAL_B_KEY,
    _REAL_Q_KEY,
    CompiledForwardShim,
    _extend_batch_cap,
    _pad_forward_batch_inputs,
    _restore_forward_batch_inputs,
    build_bucketers,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.platforms import current_platform

logger = get_logger(__name__)

# Pure-text progress format (no animated bar) so log files and
# multi-process wrappers don't garble the output.
_COMPILE_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}% Completed | "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)


def _materialize_out(out: Any, attn_backend: Any, token_to_kv_pool: Any) -> None:
    """Wait for the precompile forward's PJRT execution to finish.

    attn_backend / token_to_kv_pool come from the model_runner: upstream moved
    both off ForwardBatch (attn_backend now flows through ForwardContext, and the
    KV pool is held by the attention backend / model_runner), so warmup reads them
    from the runner rather than the batch.

    Under torch.compile the XLA compile itself is synchronous — AOTAutograd's
    SerializableAOTDispatchCompiler runs the backend during graph tracing and
    writes the compiled PjRtLoadedExecutable into the Dynamo cache. What is
    async is the PJRT execution the compiled graph dispatches when invoked:
    ``TorchTpuCompiledExecutable.__call__`` just enqueues the execution and
    returns handles. Without a ``synchronize(wait=True)`` here the precompile
    loop would keep launching bucket-N+1's forward before bucket-N's PJRT
    execution completes, which (a) inflates peak HBM (multiple in-flight
    graphs holding activation buffers) and (b) makes the per-bucket tqdm
    timings meaningless. This applies whether ``out`` is a raw ``torch.Tensor``
    (CP-v2 precompile calls ``model.model(...)`` and gets back the inner
    hidden_states directly) or a ``LogitsProcessorOutput`` (non-CP path
    through ``Qwen3MoeForCausalLM.forward``).
    """
    tensors: list[torch.Tensor] = []

    if isinstance(out, torch.Tensor):
        tensors.append(out)
    elif isinstance(out, PPProxyTensors):
        # Non-last PP rank: out is a proxy holding the pipeline stage's
        # tensor payload. Sync every constituent so peak HBM stays bounded
        # (same accounting reason as the LogitsProcessorOutput branch).
        for t in out.tensors.values():
            if isinstance(t, torch.Tensor):
                tensors.append(t)
    else:
        # LogitsProcessorOutput path. Optional fields access via getattr
        # to accommodate SimpleNamespace stubs in unit tests that mock a
        # partial output — real LogitsProcessorOutput declares both fields
        # as Optional[torch.Tensor].
        next_token_logits = getattr(out, "next_token_logits", None)
        if isinstance(next_token_logits, torch.Tensor):
            tensors.append(next_token_logits)
        hidden_states = getattr(out, "hidden_states", None)
        if isinstance(hidden_states, torch.Tensor):
            tensors.append(hidden_states)

    if getattr(attn_backend, "requires_kv_sync_after_compile", False):
        pool = token_to_kv_pool
        for layer_id in range(pool.start_layer, pool.start_layer + pool.layer_num):
            tensors.append(pool.get_fused_kv_buffer(layer_id))

    if not tensors:
        return
    from torch_tpu._internal.sync.sync import synchronize as tpu_sync

    tpu_sync(tensors, wait=True)


def _get_hbm_usage_mb() -> Optional[tuple[float, float]]:
    """Return (used_mb, total_mb) for the current TPU device, or None."""
    try:
        info = torch.accelerator.get_memory_info(torch.device("tpu"))
        free = info["free"] if isinstance(info, dict) else info[0]
        total = info["total"] if isinstance(info, dict) else info[1]
        used = total - free
        return (used / 1024 / 1024, total / 1024 / 1024)
    except Exception:
        return None


def _hbm_delta_str(
    before: Optional[tuple[float, float]],
    after: Optional[tuple[float, float]],
) -> str:
    """Format HBM delta + total as a suffix string, or empty if unavailable.

    Also emits a WARNING when the after-bucket HBM usage exceeds 90% of
    total — PCP precompile bakes many shapes and a runaway warmup would
    otherwise OOM mid-loop without an actionable message.
    """
    if before is None or after is None:
        return " HBM: N/A"
    delta = after[0] - before[0]
    used_pct = after[0] / max(after[1], 1.0)
    if used_pct > 0.9:
        logger.warning(
            "[decoder compile policy] HBM usage %.1f%% (%.0f/%.0f MB) — "
            "warmup nearing OOM. Consider SGLANG_TPU_PCP_KEEP_NONCP_COMPILE=0 "
            "if it's set, or reducing --mem-fraction-static.",
            100.0 * used_pct,
            after[0],
            after[1],
        )
    return f" HBM: +{delta:.0f} MB ({after[0]:.0f}/{after[1]:.0f} MB)"


def _unpad_decoder_outputs(out: Any, forward_batch: Any) -> Any:
    real_q = getattr(forward_batch, _REAL_Q_KEY, None)
    real_b = getattr(forward_batch, _REAL_B_KEY, None)
    if real_q is None or real_b is None:
        return out

    # Handle raw Tensor return (from ``Qwen3MoeModel.forward`` when a
    # compile boundary is installed on it — e.g. under PCP CP-v2, where
    # the eager runner calls ``model.model(...)`` directly). Without this
    # branch the padded ghost slots leak past the shim into the CP-gather.
    if isinstance(out, torch.Tensor):
        # Under PCP CP-v2 the compiled model.model returns a per-rank
        # rank-major-packed tensor of shape ``(local_padded_num_tokens,
        # hidden_dim)``. Real tokens and padding are interleaved per the
        # ``PcpSequenceLayoutPlan.token_order`` reorder (rank r owns positions
        # ``[k*cycle + r*interleave, k*cycle + (r+1)*interleave)``), so the
        # padding is NOT contiguous at the tail. ``out[:real_q]`` would blindly
        # take the first ``real_q`` rows, mixing padding into the sample and
        # violating the ``pcp_size * local_padded_num_tokens`` invariant that
        # :func:`sglang.srt.layers.cp.pcp_sequence_layout.gather_and_unpack_pcp`
        # relies on to reverse the pack (via ``inverse_order``). Skip the
        # tensor-level unpad in that case — the downstream cp_gather does the
        # real unpacking. Other CP-v2 strategies (Zigzag/Interleave) do not
        # set ``pcp_attn_active``, so they keep the original truncation.
        is_pcp_active = forward_batch.pcp_attn_active
        if not is_pcp_active and real_q < out.shape[0]:
            out = out[:real_q]
        _restore_forward_batch_inputs(forward_batch, real_q, real_b)
        return out

    next_token_logits = getattr(out, "next_token_logits", None)
    if next_token_logits is not None and real_b < next_token_logits.shape[0]:
        out.next_token_logits = next_token_logits[:real_b]

    hidden_states = getattr(out, "hidden_states", None)
    if hidden_states is not None and real_q < hidden_states.shape[0]:
        out.hidden_states = hidden_states[:real_q]

    # Restore real token/batch metadata so eager outer-model processing and
    # downstream consumers never observe static-shape ghost slots.
    _restore_forward_batch_inputs(forward_batch, real_q, real_b)
    return out


class DecoderTpuCompilePolicy(CompilePolicy):
    """Compile, pad, unpad, and precompile one decoder boundary."""

    platform = "tpu"
    fullgraph = True
    dynamic = False

    def __init__(
        self,
        *,
        model_runner: Any | None = None,
        t_bucketer: Bucketer | None = None,
        b_bucketer: Bucketer | None = None,
        owner: Any | None = None,
        method_name: str | None = None,
        method: Callable[..., Any] | None = None,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        super().__init__(
            model_runner=model_runner,
            owner=owner,
            method_name=method_name or "forward",
            method=method,
            name=name,
            label=label,
        )
        self.t_bucketer = t_bucketer
        self.b_bucketer = b_bucketer
        self._logged_vl_warmup_variants: set[str] = set()

    def install(self) -> bool:
        """Install runtime compile wrappers for the annotated decoder target."""
        if self.model_runner is None or self.owner is None or self.method_name is None:
            logger.warning(
                "[decoder compile policy] missing model_runner/owner/method; "
                "cannot install."
            )
            return False

        fn = getattr(self.owner, self.method_name, None)
        if fn is None:
            logger.warning(
                "[decoder compile policy] %s target %s has no method %s; skipping.",
                self.name,
                type(self.owner).__name__,
                self.method_name,
            )
            return False

        signature = inspect.signature(fn)
        backend = self.backend or current_platform.get_compile_backend()
        compiled = torch.compile(
            torch.no_grad()(fn),
            backend=backend,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )
        compiled.__signature__ = signature
        setattr(self.owner, self.method_name, compiled)

        server_args = self.model_runner.server_args
        # max_t = smallest positive prefill/context limit (extend never runs
        # longer). max_b = the reconciled compile-batch ceiling, already grown to
        # cover the scheduler concurrency in ServerArgs.__post_init__ via
        # reconcile_tpu_batch_limits, so the grid covers every decode batch the
        # scheduler can form (Task #40).
        t_candidates = [
            c
            for c in (
                server_args.chunked_prefill_size,
                server_args.max_prefill_tokens,
                self.model_runner.model_config.context_len,
            )
            if c and c > 0
        ]
        if not t_candidates:
            raise ValueError(
                "Cannot determine max_t for TPU compile bucketing: "
                "chunked_prefill_size, max_prefill_tokens, and context_len are all unset."
            )
        max_t = min(t_candidates)
        max_b = server_args.torch_compile_max_bs
        t_bucketer, b_bucketer = build_bucketers(server_args, max_t=max_t, max_b=max_b)
        self.t_bucketer = t_bucketer
        self.b_bucketer = b_bucketer

        outer_model = self.model_runner.model
        existing_prep = getattr(outer_model, "prepare_forward_batch", None)

        def _prepare(forward_batch: ForwardBatch) -> None:
            if existing_prep is not None:
                existing_prep(forward_batch)
            real_q, real_b = _pad_forward_batch_inputs(
                forward_batch, t_bucketer, b_bucketer
            )
            setattr(forward_batch, _REAL_Q_KEY, real_q)
            setattr(forward_batch, _REAL_B_KEY, real_b)

        outer_model.prepare_forward_batch = _prepare

        self.compiled_forward_shim = CompiledForwardShim(compiled, policy=self)
        setattr(self.owner, self.method_name, self.compiled_forward_shim)
        self.model_runner.compile_policy = self

        if is_dp_attention_enabled():
            self._install_dp_idle_as_decode()

        logger.info(
            "[decoder compile policy] Installed %s on %s.%s "
            "(t-bucketer=%s, b-bucketer=%s).",
            self.name,
            type(self.owner).__name__,
            self.method_name,
            self.t_bucketer.name,
            self.b_bucketer.name,
        )
        return True

    def _install_dp_idle_as_decode(self) -> None:
        """Run idle DP batches as DECODE.

        An idle rank may take a different code path than decode, but with DP
        attention all ranks must run the same code. Relabel idle as DECODE.

        Upstream removed ``ModelRunner.forward_idle`` (idle now flows through
        ``ModelRunner.forward`` -> the eager runner), so hook ``forward`` and
        relabel idle batches at entry; the ``is_idle()`` guard keeps decode /
        extend batches untouched.
        """
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        model_runner = self.model_runner
        # Idempotent: a re-install (tests, warmups) must not nest a second
        # wrapper around an already-wrapped forward.
        if getattr(model_runner.forward, "_tpu_idle_as_decode", False):
            return
        orig_forward = model_runner.forward

        def forward(forward_batch, *args, **kwargs):
            # Relabel only; forward()'s decode path (eager runner) runs
            # prepare_forward_batch exactly once. Do NOT prepare here — the
            # shim's _prepare is not idempotent, and a second call overwrites
            # _REAL_Q/_REAL_B with the already-padded size, so outputs never get
            # unpadded (garbage decode under DP attention).
            if (
                forward_batch.forward_mode.is_idle()
                and is_dp_attention_enabled()
                and (
                    # This is necessary for preventing all Idle.
                    max(forward_batch.global_num_tokens_cpu, default=0)
                    > 0
                )
            ):
                forward_batch.forward_mode = ForwardMode.DECODE
            return orig_forward(forward_batch, *args, **kwargs)

        forward._tpu_idle_as_decode = True
        model_runner.forward = forward

    def set_recompilation_checker(self, checker: Any, model_label: str = "") -> None:
        """Forward serving-time recompilation checks to the installed shim."""
        self.compiled_forward_shim.set_recompilation_checker(
            checker, model_label=model_label
        )

    def precompile(self) -> None:
        """Warm decoder graphs for this policy's bucket grid.

        Under ``--enable-prefill-cp`` with the PCP strategy every extend /
        decode routes through ``eager_runner``'s CP-v2 branch, which
        dispatches ``model.model(input_ids, positions, forward_batch,
        input_embeds=<sharded>, ...)``. The non-CP entry point uses
        ``input_embeds=None``; Dynamo caches on that type mismatch so the
        non-CP graphs are unreachable at serve time. Skip them — each
        compile is 30-60s on 30B / minutes on 480B and leaves dead XLA
        program cache behind. Set ``SGLANG_TPU_PCP_KEEP_NONCP_COMPILE=1``
        to retain them (mixed workloads that route some traffic through
        the non-CP path).
        """
        from sglang.srt.environ import envs

        overall = time.perf_counter()
        pcp_active = self._pcp_streaming_enabled()
        keep_noncp = bool(envs.SGLANG_TPU_PCP_KEEP_NONCP_COMPILE.get())
        if not pcp_active or keep_noncp:
            self._precompile_extend()
            self._precompile_decode()
        if pcp_active:
            self._precompile_extend_pcp()
            self._precompile_decode_pcp()
        logger.info(
            "[decoder compile policy] precompile done in %.1fs.",
            time.perf_counter() - overall,
        )

    def _pcp_streaming_enabled(self) -> bool:
        """True iff the CP-v2 PCP streaming branch will fire at serve time."""
        from sglang.srt.layers.cp.base import is_pcp
        from sglang.srt.layers.cp.utils import enable_cp_v2

        sa = self.model_runner.server_args
        return enable_cp_v2() and (sa.attn_cp_size or 1) > 1 and is_pcp()

    def _precompile_extend_pcp(self) -> None:
        """Compile extend at each PCP-aligned token bucket + the extend cap."""
        assert self.t_bucketer is not None and self.b_bucketer is not None
        pcp_size = int(self.model_runner.server_args.attn_cp_size or 1)
        bs = _extend_batch_cap(self.t_bucketer, self.b_bucketer)
        # PCP kernel requires ``padded_num_tokens % (pcp_size * q_block_size)
        # == 0`` (per-rank shard must be a multiple of the kernel's lockstep
        # floor). Rounding only to ``pcp_size`` here would let a bucket like
        # 1032 (=8*129) pass and blow up at Pallas trace time. Filter with a
        # visible log line rather than silently dropping shapes.
        from sglang.srt.layers.cp.base import get_cp_strategy

        strategy = get_cp_strategy()
        kernel_bucket = pcp_size * int(strategy.local_compile_token_multiple)
        # Also require num_tokens % bs == 0 — the dummy CP forward splits
        # tokens equally across the ``bs`` dummy requests and any leftover
        # produces a shape mismatch between ``extend_num_tokens`` (bs*floor)
        # and ``input_ids.shape[0]`` (num_tokens). Drop mis-aligned buckets
        # here with a visible log line so we never dispatch one.
        valid_shapes = [
            (t, bs)
            for t in self.t_bucketer.buckets
            if t >= bs and t % kernel_bucket == 0 and t % bs == 0
        ]
        dropped = [
            t
            for t in self.t_bucketer.buckets
            if t >= bs and not (t % kernel_bucket == 0 and t % bs == 0)
        ]
        if dropped:
            # WARNING (not INFO) because dropped buckets = ignored
            # user-configured shapes. Operators should see this by default.
            logger.warning(
                "[decoder compile policy] EXTEND-PCP: dropping %d buckets "
                "not aligned to pcp_size*q_block_size=%d or bs=%d: %s",
                len(dropped),
                kernel_bucket,
                bs,
                dropped,
            )
        if not valid_shapes:
            raise ValueError(
                "No PCP-aligned extend buckets: token buckets "
                f"{list(self.t_bucketer.buckets)} yield no shape with "
                f"t>={bs}, t%{kernel_bucket}==0, and t%{bs}==0."
            )
        logger.info(
            "[decoder compile policy] EXTEND-PCP: %d shapes (bs=%d, pcp=%d)",
            len(valid_shapes),
            bs,
            pcp_size,
        )
        compile_range = (
            tqdm.tqdm(
                valid_shapes,
                desc="Compiling EXTEND-PCP",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_tensor_model_parallel_rank() == 0
            else valid_shapes
        )
        for num_tokens, bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one_pcp(num_tokens=num_tokens, bs=bs)
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled EXTEND-PCP (tokens={num_tokens}, bs={bs}) "
                    f"in {elapsed:.1f}s{hbm_str}"
                )

    def _run_one_pcp(
        self, *, num_tokens: int, bs: int, mode: ForwardMode = ForwardMode.EXTEND
    ) -> None:
        """One dummy CP-v2 forward, mirroring eager_runner's CP branch.

        The eager CP branch calls ``model_runner.model.model(...)`` with
        rank-sharded ``input_embeds`` and ``positions`` — not
        ``Qwen3MoeForCausalLM.forward``. The compile policy still installs
        the shim on ``Qwen3MoeModel.forward`` (models/qwen3_moe.py), so
        ``self.owner.forward`` (owner is ``Qwen3MoeModel``) enters the same
        compiled graph the eager CP path hits at serve time.

        Handles both extend (``mode=EXTEND``) and decode (``mode=DECODE``);
        decode reuses the same rank-sharded dispatch since PCP now routes
        decode through the streaming kernel too (see is_cp_v2_active in
        layers/cp/utils.py).
        """
        from sglang.srt.layers.cp.utils import (
            cp_split_before_forward,
            is_cp_v2_active,
            prepare_cp_forward,
        )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        args, _ = self._make_warmup_inputs(num_tokens=num_tokens, bs=bs, mode=mode)
        forward_batch = args[2]
        # For EXTEND: rewrite the dummy batch's extend_* fields so the
        # per-req length reflects num_tokens // bs (rather than the base
        # 1-token-per-req dummy). Without this, ``_build_pcp_layout_plan``
        # sizes ``plan.inverse_order`` at (bs,) and torch_tpu re-JITs the
        # cp_gather + logits_processor + argmax paths at (num_tokens,) on
        # the first real request.
        # For DECODE: extend_* fields stay None (that's how the base
        # dummy factory constructs them); ``_build_pcp_layout_plan``
        # branches on ``extend_lens_cpu is None`` and takes the pure-
        # decode path with scheduled=ones(bs), which is what serve time
        # decodes look like — so no rewrite is needed.
        if mode.is_extend(include_draft_extend_v2=True):
            if num_tokens % bs != 0:
                # Splitting num_tokens into equal per-req lengths requires
                # exact divisibility — otherwise ``extend_num_tokens`` (=
                # bs*floor) would not equal input_ids.shape[0], producing a
                # shape mismatch inside the model that shows up as either a
                # Pallas trace assert or (worse) a silently wrong compiled
                # graph shape. Filter buckets in _precompile_extend_pcp.
                raise ValueError(
                    "_run_one_pcp: num_tokens must be divisible by bs "
                    f"(got num_tokens={num_tokens}, bs={bs}). Filter the "
                    "bucket at _precompile_extend_pcp before dispatch."
                )
            _tokens_per_req = num_tokens // bs
            _dev = forward_batch.input_ids.device
            forward_batch.extend_seq_lens = torch.full(
                (bs,), _tokens_per_req, dtype=torch.int32, device=_dev
            )
            forward_batch.extend_num_tokens = int(bs * _tokens_per_req)
            forward_batch.extend_seq_lens_cpu = [_tokens_per_req] * bs
            forward_batch.extend_start_loc = torch.arange(
                0,
                bs * _tokens_per_req,
                _tokens_per_req,
                dtype=torch.int32,
                device=_dev,
            )
        self._seed_dummy_req_to_token(num_tokens)
        if is_dp_attention_enabled():
            self._set_dp_attention_metadata(forward_batch, num_tokens, bs, mode)
        self.model_runner.model.prepare_forward_batch(forward_batch)
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        assert is_cp_v2_active(forward_batch), (
            f"PCP precompile: is_cp_v2_active(dummy_batch) is False; check "
            f"SGLANG_ENABLE_CP_V2, cp_strategy=='pcp', attn_cp_size>1, and "
            f"that the current mode ({mode.name}) satisfies "
            f"is_context_parallel_extend() or is_decode()."
        )
        prepare_cp_forward(forward_batch)
        embed_layer = self.model_runner.model.get_input_embeddings()
        complete_hidden = embed_layer(forward_batch.input_ids)
        sharded_hidden, sharded_positions = cp_split_before_forward(
            complete_hidden,
            forward_batch.positions,
            forward_batch,
        )
        from sglang.srt.layers.cp.utils import cp_gather_after_forward

        with forward_context(
            ForwardContext(attn_backend=self.model_runner.attn_backend)
        ), torch.no_grad():
            hidden_states = self.owner.forward(
                forward_batch.input_ids,
                sharded_positions,
                forward_batch,
                input_embeds=sharded_hidden,
                pp_proxy_tensors=None,
            )
            # Warm the post-forward pipeline (cp_gather + logits_processor +
            # final norm + LM head projection + argmax). Without this, the
            # first CP serve request pays ~1.8s of TPU-idle time while the
            # host JIT-compiles these eager-mode kernels
            # (``tt_jit_argmax``, ``tt_jit_layernorm_L527C0``,
            # ``tt_jit_all``, ``tt_jit_index``, etc. — verified via TPU
            # profile diff).
            model = self.model_runner.model
            # capture_aux_hidden_states=True (EAGLE-3): the model forward
            # returns (hidden_states, aux_hidden_states) tuple instead of
            # a single tensor. Mirror the eager_runner CP-v2 branch unpack
            # so the precompile shape matches serve time.
            aux_hidden_states = None
            if model.capture_aux_hidden_states:
                hidden_states, aux_hidden_states = hidden_states
            if model.pp_group.is_last_rank:
                gathered = cp_gather_after_forward(
                    hidden_states,
                    forward_batch,
                    stream=None,
                )
                out = model.logits_processor(
                    forward_batch.input_ids,
                    gathered,
                    model.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
                # Warm the sampler-side kernels (argmax + validity check)
                # that fire in every serve request but aren't triggered
                # by the model forward chain. Without this, the first
                # serve pays ~185ms of ``tt_jit_argmax`` JIT compile.
                # LogitsProcessorOutput.next_token_logits is Optional; the
                # isinstance narrows the None branch (LM head can return
                # None when spec-decode is off; see logits_processor.py).
                if isinstance(out.next_token_logits, torch.Tensor):
                    _ = torch.argmax(out.next_token_logits, dim=-1)
                    _ = torch.all(torch.isfinite(out.next_token_logits))
            else:
                out = hidden_states
            _materialize_out(
                out,
                self.model_runner.attn_backend,
                self.model_runner.token_to_kv_pool,
            )

    def _precompile_extend(self) -> None:
        """Compile extend at each token bucket and the fixed batch cap."""
        assert self.t_bucketer is not None
        assert self.b_bucketer is not None

        bs = _extend_batch_cap(self.t_bucketer, self.b_bucketer)
        valid_shapes = [(t, bs) for t in self.t_bucketer.buckets if t >= bs]
        if not valid_shapes:
            raise ValueError(
                "No TPU extend compile shapes are valid: token buckets "
                f"{list(self.t_bucketer.buckets)} are all smaller than the "
                f"extend batch cap {bs}."
            )

        logger.info(
            "[decoder compile policy] EXTEND: %d shapes (bs=%d, token_buckets=%s)",
            len(valid_shapes),
            bs,
            list(self.t_bucketer.buckets),
        )
        compile_range = (
            tqdm.tqdm(
                valid_shapes,
                desc="Compiling EXTEND",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_tensor_model_parallel_rank() == 0
            else valid_shapes
        )
        for num_tokens, bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(num_tokens=num_tokens, bs=bs, mode=ForwardMode.EXTEND)
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled EXTEND (tokens={num_tokens}, bs={bs}) "
                    f"in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_decode_pcp(self) -> None:
        """Warm decode on the PCP CP-v2 path (mirrors _precompile_extend_pcp).

        For decode, num_tokens == bs (one token per request). PCP requires
        the global bucket to be a multiple of pcp_size * q_block_size, so
        the effective PCP dispatch pads up to that minimum regardless of
        the caller-visible bucket. Warming at each configured decode bs
        ensures the first served decode step doesn't retrace.
        """
        assert self.b_bucketer is not None
        bs_buckets = list(self.b_bucketer.buckets)
        logger.info(
            "[decoder compile policy] DECODE-PCP: %d shapes (bs=%s)",
            len(bs_buckets),
            bs_buckets,
        )
        compile_range = (
            tqdm.tqdm(
                bs_buckets,
                desc="Compiling DECODE-PCP",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_tensor_model_parallel_rank() == 0
            else bs_buckets
        )
        for bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one_pcp(num_tokens=bs, bs=bs, mode=ForwardMode.DECODE)
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled DECODE-PCP (bs={bs}) in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_decode(self) -> None:
        """Compile decode mode: vary batch size (tokens == batch size)."""
        assert self.b_bucketer is not None

        bs_buckets = list(self.b_bucketer.buckets)
        logger.info(
            "[decoder compile policy] DECODE: %d shapes (bs=%s)",
            len(bs_buckets),
            bs_buckets,
        )
        compile_range = (
            tqdm.tqdm(
                bs_buckets,
                desc="Compiling DECODE",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_tensor_model_parallel_rank() == 0
            else bs_buckets
        )
        for bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(num_tokens=bs, bs=bs, mode=ForwardMode.DECODE)
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled DECODE (bs={bs}) in {elapsed:.1f}s{hbm_str}"
                )

    def _make_warmup_inputs(
        self,
        *,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        variant: str = "plain",
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: Any | None = None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Build generic text decoder warmup inputs.

        The 4-positional + ``pp_proxy_tensors=None`` kwarg shape matches
        the dispatch that ``Qwen2/3MoeForCausalLM.forward`` produces when
        it calls ``self.model(input_ids, positions, forward_batch,
        input_embeds, pp_proxy_tensors=pp_proxy_tensors)``. The compile
        shim is installed on the INNER model (``Qwen3MoeModel.forward``),
        so serve-time non-CP dispatch enters the shim with ``len(args)
        == 4`` and ``pp_proxy_tensors`` as the only kwarg. Precompiling
        the same shape prevents a serve-time recompile (Dynamo guards
        on len(args) and on the kwarg dict identity).
        """
        forward_batch = self._make_dummy_forward_batch(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )
        return (
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            None,  # input_embeds — positional (None for non-CP)
        ), {"pp_proxy_tensors": None}

    def _run_one(
        self,
        *,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        variant: str = "plain",
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: Any | None = None,
    ) -> None:
        """Build and execute one dummy forward pass for compilation."""
        args, kwargs = self._make_warmup_inputs(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            variant=variant,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )
        forward_batch = args[2]
        self._seed_dummy_req_to_token(num_tokens)

        model = self.model_runner.model
        # Set the DP metadata before the pad shim runs, just like serving does,
        # so precompile traces the same DP graph serving uses. Without it,
        # precompile traces a different graph.
        if is_dp_attention_enabled():
            self._set_dp_attention_metadata(forward_batch, num_tokens, bs, mode)
        model.prepare_forward_batch(forward_batch)
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)

        # The compiled forward resolves its attention backend from the forward
        # context (upstream moved attn_backend off ForwardBatch into a
        # ForwardContext). Serving establishes it in ModelRunner.forward, but the
        # warmup path invokes the compiled forward directly, so set it here too.
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        # Match TpuGraphRunner's no_grad wrapper for Dynamo cache reuse.
        with forward_context(
            ForwardContext(attn_backend=self.model_runner.attn_backend)
        ), torch.no_grad():
            out = self.owner.forward(*args, **kwargs)
            # Materialize lazy outputs and backend-marked side effects.
            _materialize_out(
                out,
                self.model_runner.attn_backend,
                self.model_runner.token_to_kv_pool,
            )

    def unpad_outputs(self, out: Any, forward_batch: Any) -> Any:
        return _unpad_decoder_outputs(out, forward_batch)

    def _set_dp_attention_metadata(
        self, forward_batch: ForwardBatch, num_tokens: int, bs: int, mode: ForwardMode
    ) -> None:
        """Set the DP metadata in the same way as serving.

        Serving fills in the DP padding mode and global per-rank token counts
        before the shim runs. Precompile must set the same values so the shim
        traces the identical DP graph. All ranks compile the same bucket in
        lockstep.
        """
        from sglang.srt.layers.dp_attention import DpPaddingMode

        dp_size = self.model_runner.server_args.dp_size
        device = self.model_runner.device
        global_num_tokens = [num_tokens] * dp_size
        forward_batch.dp_padding_mode = DpPaddingMode.MAX_LEN
        forward_batch.is_extend_in_batch = mode == ForwardMode.EXTEND
        forward_batch.global_num_tokens_cpu = list(global_num_tokens)
        forward_batch.original_global_num_tokens_cpu = list(global_num_tokens)
        forward_batch.global_num_tokens_gpu = torch.tensor(
            global_num_tokens, dtype=torch.int64, device=device
        )
        forward_batch.global_num_tokens_for_logprob_cpu = [bs] * dp_size
        forward_batch.global_num_tokens_for_logprob_gpu = torch.tensor(
            [bs] * dp_size, dtype=torch.int64, device=device
        )

    def _dummy_kv_slot(self) -> int:
        """KV-pool slot for dummy ``req_pool_indices``/``out_cache_loc``."""
        return 0

    def _dummy_tokens_per_req(self, mode: ForwardMode, spec_info: Any | None) -> int:
        """Extend tokens per request in the dummy batch (1 for plain extend)."""
        return 1

    def _dummy_mm_inputs(self, bs: int) -> Any:
        """``mm_inputs`` field for the dummy batch (``None`` for plain decoders)."""
        return None

    def _seed_dummy_req_to_token(self, num_tokens: int) -> None:
        """Seed req_to_token rows for the dummy batch.

        No-op for plain decoders: row 0 stays all-zeros (page 0, the reserved
        padding page), which is exactly the harmless state padded batches expect.
        The spec policy overrides this to seed slot 1 with a valid page index.
        """
        return

    def _make_dummy_forward_batch(
        self,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        *,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: Any | None = None,
    ) -> ForwardBatch:
        """Construct a dummy ForwardBatch at the target bucket shape."""
        device = self.model_runner.device
        is_extend = mode.is_extend(include_draft_extend_v2=True)
        slot = self._dummy_kv_slot()

        # Token-level inputs
        input_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        input_ids[:bs] = 1
        positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        out_cache_loc = torch.full(
            (num_tokens,), slot, dtype=torch.int64, device=device
        )

        # MROPE runtime positions are real [3, num_tokens] tensors. Keep strides
        # identical here so Dynamo does not guard-miss after precompile.
        mrope_positions = None
        if self.model_runner.model_is_mrope:
            mrope_positions = positions.unsqueeze(0).expand(3, -1).contiguous()

        # Batch-level inputs
        seq_lens = torch.ones(bs, dtype=torch.int64, device=device)
        seq_lens_cpu = torch.ones(bs, dtype=torch.int64)
        req_pool_indices = torch.full((bs,), slot, dtype=torch.int64, device=device)

        tokens_per_req = self._dummy_tokens_per_req(mode, spec_info)
        extend_seq_lens = (
            torch.full((bs,), tokens_per_req, dtype=torch.int32, device=device)
            if is_extend
            else None
        )
        extend_prefix_lens = (
            torch.zeros(bs, dtype=torch.int32, device=device) if is_extend else None
        )
        extend_start_loc = (
            torch.arange(
                0, bs * tokens_per_req, tokens_per_req, dtype=torch.int32, device=device
            )
            if is_extend
            else None
        )
        extend_num_tokens = bs * tokens_per_req if is_extend else None
        extend_seq_lens_cpu = [tokens_per_req] * bs if is_extend else None
        extend_prefix_lens_cpu = [0] * bs if is_extend else None
        # Match runtime extend metadata; None vs list creates guard misses.
        extend_logprob_start_lens_cpu = [0] * bs if is_extend else None

        return ForwardBatch(
            forward_mode=mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=bs,
            orig_seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            return_logprob=False,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_logprob_start_lens_cpu=extend_logprob_start_lens_cpu,
            mrope_positions=mrope_positions,
            # Model accesses ``capture_hidden_mode.need_capture()``; default
            # of None would raise AttributeError inside the compiled region.
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
            mm_inputs=self._dummy_mm_inputs(bs),
        )


class Qwen3VLDecoderTpuCompilePolicy(DecoderTpuCompilePolicy):
    """Qwen3-VL decoder policy that warms multimodal prefill call shapes."""

    def install(self) -> bool:
        if is_dp_attention_enabled():
            raise NotImplementedError(
                "Qwen3-VL TPU decoder compilation does not support DP attention; "
                "disable --enable-torch-compile for this configuration."
            )
        return super().install()

    @property
    def _num_deepstack(self) -> int:
        model = getattr(self.model_runner, "model", None)
        return int(getattr(model, "num_deepstack_embeddings", 0) or 0)

    @property
    def _model_is_mrope(self) -> bool:
        return bool(getattr(self.model_runner, "model_is_mrope", False))

    def _uses_input_embeds(self, target: Any) -> bool:
        """Whether to warm ``target`` via the Qwen3-VL input_embeds path."""
        outer = getattr(self.model_runner, "model", None)
        is_multimodal = hasattr(outer, "get_image_feature")
        return is_multimodal and callable(getattr(target, "get_input_embeddings", None))

    def unpad_outputs(self, out: Any, forward_batch: Any) -> Any:
        """Slice the inner VL decoder output before outer logits processing."""
        real_q = getattr(forward_batch, _REAL_Q_KEY, None)
        out = super().unpad_outputs(out, forward_batch)
        if real_q is None:
            return out
        if isinstance(out, torch.Tensor):
            return out[:real_q]
        if isinstance(out, PPProxyTensors):
            # Intermediate pipeline stages must transmit the padded proxy. The
            # next stage pads ForwardBatch again but consumes this proxy as-is.
            return out
        if isinstance(out, tuple) and len(out) == 2:
            hidden_states, aux_hidden_states = out
            if isinstance(hidden_states, torch.Tensor):
                hidden_states = hidden_states[:real_q]
            if isinstance(aux_hidden_states, (list, tuple)):
                aux_hidden_states = type(aux_hidden_states)(
                    value[:real_q] if isinstance(value, torch.Tensor) else value
                    for value in aux_hidden_states
                )
            return hidden_states, aux_hidden_states
        return out

    def _precompile_extend(self) -> None:
        if not self._uses_input_embeds(self.owner):
            return super()._precompile_extend()

        assert self.t_bucketer is not None
        assert self.b_bucketer is not None

        bs = _extend_batch_cap(self.t_bucketer, self.b_bucketer)
        valid_shapes = [(t, bs) for t in self.t_bucketer.buckets if t >= bs]
        if not valid_shapes:
            raise ValueError(
                "No TPU extend compile shapes are valid: token buckets "
                f"{list(self.t_bucketer.buckets)} are all smaller than the "
                f"extend batch cap {bs}."
            )

        variants = ["text_only"]
        if self._num_deepstack > 0:
            variants = ["deepstack", "text_only"]

        logger.info(
            "[decoder compile policy] VL EXTEND: %d shapes x %d variants "
            "(bs=%d, token_buckets=%s)",
            len(valid_shapes),
            len(variants),
            bs,
            list(self.t_bucketer.buckets),
        )
        compile_items = [
            (num_tokens, bs, variant)
            for num_tokens, bs in valid_shapes
            for variant in variants
        ]
        compile_range = (
            tqdm.tqdm(
                compile_items,
                desc="Compiling VL EXTEND",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_tensor_model_parallel_rank() == 0
            else compile_items
        )
        for num_tokens, bs, variant in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.EXTEND,
                variant=variant,
            )
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled VL EXTEND (tokens={num_tokens}, bs={bs}, "
                    f"variant={variant}) in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_decode(self) -> None:
        if not self._uses_input_embeds(self.owner):
            return super()._precompile_decode()

        assert self.b_bucketer is not None

        bs_buckets = list(self.b_bucketer.buckets)
        logger.info(
            "[decoder compile policy] VL DECODE: %d shapes (bs=%s)",
            len(bs_buckets),
            bs_buckets,
        )
        compile_range = (
            tqdm.tqdm(
                bs_buckets,
                desc="Compiling VL DECODE",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_tensor_model_parallel_rank() == 0
            else bs_buckets
        )
        for bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(
                num_tokens=bs,
                bs=bs,
                mode=ForwardMode.DECODE,
                variant="text_only",
            )
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled VL DECODE (bs={bs}) in {elapsed:.1f}s{hbm_str}"
                )

    def _make_warmup_inputs(
        self,
        *,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        variant: str = "plain",
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: Any | None = None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not self._uses_input_embeds(self.owner):
            return super()._make_warmup_inputs(
                num_tokens=num_tokens,
                bs=bs,
                mode=mode,
                variant=variant,
                capture_hidden_mode=capture_hidden_mode,
                spec_info=spec_info,
            )

        forward_batch = self._make_dummy_forward_batch(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )

        positions = (
            forward_batch.mrope_positions
            if self._model_is_mrope
            else forward_batch.positions
        )

        embed_tokens = self.owner.get_input_embeddings()
        input_embeds = embed_tokens(forward_batch.input_ids)

        call_kwargs: dict[str, Any] = {}
        num_deepstack = self._num_deepstack
        if variant == "deepstack" and mode == ForwardMode.EXTEND and num_deepstack > 0:
            call_kwargs["input_deepstack_embeds"] = torch.zeros(
                input_embeds.shape[0],
                input_embeds.shape[1] * num_deepstack,
                dtype=input_embeds.dtype,
                device=input_embeds.device,
            )

        logged_variant = (
            "deepstack" if "input_deepstack_embeds" in call_kwargs else "text_only"
        )
        if logged_variant not in self._logged_vl_warmup_variants:
            self._logged_vl_warmup_variants.add(logged_variant)
            logger.info(
                "[decoder padding] VL warmup (%s) uses input_embeds path: "
                "input_ids=None, input_embeds=%s, input_deepstack_embeds=%s",
                logged_variant,
                tuple(input_embeds.shape),
                (
                    tuple(call_kwargs["input_deepstack_embeds"].shape)
                    if "input_deepstack_embeds" in call_kwargs
                    else None
                ),
            )

        call_kwargs["input_embeds"] = input_embeds
        return (None, positions, forward_batch), call_kwargs
