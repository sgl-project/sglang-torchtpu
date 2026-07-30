# SPDX-License-Identifier: Apache-2.0
"""PCP debug hooks.

Env-var-gated diagnostics for chasing PCP correctness / cold-latency bugs.
Every hook is a no-op when its env var is unset, and each env var is read
exactly once at module import so gating conditions are eligible for
constant-folding under ``torch.compile`` (a runtime ``envs.get()`` call
would graph-break).

Env vars (all default-off; declared in ``sglang.srt.environ``):

* ``SGLANG_TPU_PCP_DUMP_INPUTS=<dir>`` — torch-save each PCP op's inputs
  and post-op state per-layer per-forward.
* ``SGLANG_TPU_PCP_LAYER_TRACE=<file>`` — append one line per PCP layer
  with cheap host-side hashes of q/k/v/kv_pre/kv_post/output.
* ``SGLANG_TPU_PCP_LAYERS=<comma-list>`` — restrict PCP dispatch to the
  listed layer ids; excluded layers return zeros. Contaminates output
  tokens (bisection only).
* ``SGLANG_TPU_PCP_FORCE_SERIAL=1`` — data-dependency barrier
  (``out += kv_cache.sum() * 0``) after each PCP layer to force ordering.
  Compile-safe.
* ``SGLANG_TPU_PCP_STUB_ATTN=1`` — replace every PCP attention call with
  zeros. Contaminates output tokens (test whether attention is the source
  of a bug).
* ``SGLANG_TPU_PCP_JIT_TRACE_HASH=<file>`` — monkey-patch
  ``torch_tpu._internal.pallas.pallas.JaxCallable`` to log the first
  invocation of each ``(name, invocation_key)`` pair. Used to find
  cold-only JIT compiles missed by the precompile pass.
* ``SGLANG_TPU_PCP_DUMP_HS=1`` — emit ``[HS_LOCAL_S]`` lines with per-rank
  hidden-state norm + head8 at slots 0..3, from the eager runner around
  ``model.model()``.
* ``SGLANG_TPU_PCP_DUMP_KV_SLOT=1`` — emit ``[KV_SLOT]`` lines with
  per-rank layer-0 KV values at the first 4 ``out_cache_loc`` slots after
  each forward, from the eager runner.
"""

from __future__ import annotations

import os
import threading
import traceback
from typing import Any, Optional

import torch

from sglang.srt.environ import envs

# ---------------------------------------------------------------------------
# Read env vars ONCE at import so downstream gating checks are module-level
# constants (Dynamo constant-folds ``if _DUMP_INPUTS_DIR is None: pass``
# to a no-op; a runtime ``envs.X.get()`` would graph-break).
# ---------------------------------------------------------------------------

_DUMP_INPUTS_DIR: Optional[str] = envs.SGLANG_TPU_PCP_DUMP_INPUTS.get() or None
_LAYER_TRACE_PATH: Optional[str] = envs.SGLANG_TPU_PCP_LAYER_TRACE.get() or None

_LAYERS_ALLOWLIST_RAW: Optional[str] = envs.SGLANG_TPU_PCP_LAYERS.get() or None
_LAYERS_ALLOWLIST: Optional[frozenset[int]] = (
    frozenset(int(x) for x in _LAYERS_ALLOWLIST_RAW.split(",") if x.strip())
    if _LAYERS_ALLOWLIST_RAW
    else None
)

_FORCE_SERIAL: bool = envs.SGLANG_TPU_PCP_FORCE_SERIAL.get()
_STUB_ATTN: bool = envs.SGLANG_TPU_PCP_STUB_ATTN.get()
_JIT_TRACE_PATH: Optional[str] = envs.SGLANG_TPU_PCP_JIT_TRACE_HASH.get() or None
_DUMP_HS: bool = envs.SGLANG_TPU_PCP_DUMP_HS.get()
_DUMP_KV_SLOT: bool = envs.SGLANG_TPU_PCP_DUMP_KV_SLOT.get()


# ---------------------------------------------------------------------------
# Module state — kept trivial to minimise the risk of the debug path itself
# introducing races.
# ---------------------------------------------------------------------------

_TRACE_LOCK = threading.Lock()
_DUMP_LOCK = threading.Lock()
_FORWARD_COUNT = 0
_JIT_TRACE_INSTALLED = False


def forward_started() -> int:
    """Increment the per-process forward counter and return the new value.

    Called at the first PCP layer of each forward from
    :func:`sglang.srt.layers.cp.pcp_streaming.pcp_streaming_attention`
    (gated on ``dump_inputs_active() or layer_trace_active()``), so
    per-forward dumps land in a distinct file / trace section. Non-atomic
    across ranks (each rank keeps its own counter), which matches how the
    smoke test consumes dumps (per-rank compare).

    Ownership: exactly one call site increments the counter per forward.
    ``forward_started`` (layer 0) is the primary owner. When it fires,
    :func:`log_hidden_states_slot0` (post_model tag) MUST NOT also bump —
    otherwise per-forward the counter advances by 2 and downstream tools
    that treat the index as a forward number silently mis-align.
    :func:`log_hidden_states_slot0` bumps only when the layer-level hooks
    are off; see there.
    """
    global _FORWARD_COUNT
    _FORWARD_COUNT += 1
    return _FORWARD_COUNT


def current_forward_index() -> int:
    """Return the current forward index without incrementing."""
    return _FORWARD_COUNT


# ---------------------------------------------------------------------------
# Dump PCP op inputs & post-state per layer per forward.
# ---------------------------------------------------------------------------


def dump_inputs_active() -> bool:
    """True iff SGLANG_TPU_PCP_DUMP_INPUTS is set."""
    return _DUMP_INPUTS_DIR is not None


def dump_pcp_state(
    *,
    layer_id: int,
    phase: str,
    forward_index: int,
    kv_cache: torch.Tensor,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    cu_q_lens: Optional[torch.Tensor] = None,
    distribution: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> None:
    """Torch-save the PCP op's inputs / outputs for one layer.

    ``phase='pre'`` dumps the inputs; ``phase='post'`` dumps kv_cache
    after op + the output tensor. The two-phase split lets Angle A
    diff pre-vs-post kv_cache to confirm the kernel wrote what it
    was supposed to.
    """
    if _DUMP_INPUTS_DIR is None:
        return

    def _cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        return t.detach().to("cpu")

    payload = {
        "layer_id": int(layer_id),
        "phase": phase,
        "forward_index": int(forward_index),
        "kv_cache": _cpu(kv_cache),
        "q": _cpu(q),
        "k": _cpu(k),
        "v": _cpu(v),
        "seq_lens": _cpu(seq_lens),
        "block_tables": _cpu(block_tables),
        "cu_q_lens": _cpu(cu_q_lens),
        "distribution": _cpu(distribution),
        "output": _cpu(output),
    }
    with _DUMP_LOCK:
        os.makedirs(_DUMP_INPUTS_DIR, exist_ok=True)
        rank = os.environ.get("LOCAL_RANK", "0")
        fname = (
            f"pid{os.getpid()}_rank{rank}_fwd{forward_index:04d}"
            f"_layer{layer_id:03d}_{phase}.pt"
        )
        torch.save(payload, os.path.join(_DUMP_INPUTS_DIR, fname))


# ---------------------------------------------------------------------------
# Per-layer hash trace.
# ---------------------------------------------------------------------------


def layer_trace_active() -> bool:
    """True iff SGLANG_TPU_PCP_LAYER_TRACE is set."""
    return _LAYER_TRACE_PATH is not None


def _tensor_fingerprint(t: Optional[torch.Tensor]) -> str:
    """Cheap host-side fingerprint: fp32 sum + first/last 8-byte checksum.

    Deterministic across runs iff the tensor bytes are identical. Fast
    enough (~1ms per KV-cache tensor) that the trace is usable on real
    forwards. Rounds sum to 8 sig figs so tiny fp jitter across identical
    bit patterns still hashes equal.
    """
    if t is None:
        return "None"
    try:
        t_cpu = t.detach().to("cpu").contiguous()
    except Exception as e:
        return f"err({e!r})"
    try:
        s = float(t_cpu.to(torch.float32).sum().item())
    except Exception:
        s = float("nan")
    b = t_cpu.numpy().tobytes() if t_cpu.numel() > 0 else b""
    if len(b) >= 8:
        head = int.from_bytes(b[:8], "little", signed=False)
        tail = int.from_bytes(b[-8:], "little", signed=False)
    elif b:
        head = int.from_bytes(b.ljust(8, b"\0"), "little", signed=False)
        tail = head
    else:
        head = 0
        tail = 0
    return f"sum={s:.8g},head={head:016x},tail={tail:016x},shape={tuple(t.shape)}"


def log_layer_trace(
    *,
    layer_id: int,
    forward_index: int,
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    kv_cache_pre: Optional[torch.Tensor],
    kv_cache_post: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
) -> None:
    """Append one trace line per PCP layer per forward."""
    if _LAYER_TRACE_PATH is None:
        return
    rank = os.environ.get("LOCAL_RANK", "0")
    line = (
        f"fwd={forward_index:04d} rank={rank} layer={layer_id:03d} "
        f"q=[{_tensor_fingerprint(q)}] "
        f"k=[{_tensor_fingerprint(k)}] "
        f"v=[{_tensor_fingerprint(v)}] "
        f"kv_pre=[{_tensor_fingerprint(kv_cache_pre)}] "
        f"kv_post=[{_tensor_fingerprint(kv_cache_post)}] "
        f"out=[{_tensor_fingerprint(output)}]\n"
    )
    with _TRACE_LOCK:
        with open(_LAYER_TRACE_PATH, "a", buffering=1) as f:
            f.write(line)


# ---------------------------------------------------------------------------
# Per-layer allow list. Diagnostic-only (contaminates output).
# ---------------------------------------------------------------------------


def layer_allowed(layer_id: int) -> bool:
    """True iff PCP dispatch is permitted for ``layer_id``.

    When SGLANG_TPU_PCP_LAYERS is unset (no allow-list), always True.
    When set, only listed layers run the real PCP kernel; excluded
    layers are stubbed (see :func:`stub_output_for_layer`).
    """
    if _LAYERS_ALLOWLIST is None:
        return True
    return int(layer_id) in _LAYERS_ALLOWLIST


# ---------------------------------------------------------------------------
# Force serialisation after each layer.
# ---------------------------------------------------------------------------


def force_serial_active() -> bool:
    """True iff SGLANG_TPU_PCP_FORCE_SERIAL is set."""
    return _FORCE_SERIAL


def force_serial_output(output: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
    """Add a zero-effect kv_cache stat to ``output`` to force ordering.

    Cross-layer serialisation via data dependency: after this call,
    ``output`` value-depends on ``kv_cache`` (with a mul-by-zero fudge).
    The next layer's Q comes from ``output`` through the residual path,
    so XLA cannot schedule the next PCP op ahead of this layer's KV
    update. Also implicitly waits for any pending async cross-rank
    copies inside the kernel to complete, because ``kv_cache`` must be
    reduced before the sum lands.

    A ``tpu_sync`` barrier would be more direct but is not Dynamo-
    traceable (pybind builtin) and cannot Materialize a placeholder
    tensor at compile time — this data-dep trick avoids both issues.
    """
    if not _FORCE_SERIAL:
        return output
    stat = kv_cache.to(output.dtype).sum() * 0
    return output + stat


# ---------------------------------------------------------------------------
# Stub attention (return zeros).
# ---------------------------------------------------------------------------


def stub_attn_active() -> bool:
    """True iff SGLANG_TPU_PCP_STUB_ATTN is set."""
    return _STUB_ATTN


def stub_output_for_layer(q: torch.Tensor, v_head_dim: int) -> torch.Tensor:
    """Zero-output tensor shaped like the real PCP kernel result.

    Real kernel returns ``[num_tokens, tp_q_head_num * v_head_dim]`` after
    the reshape+slice; ``q`` at this call site has already been reshaped
    to ``[num_tokens, tp_q_head_num, qk_head_dim]``. We return the final
    2-D shape directly.
    """
    num_tokens = q.shape[0]
    tp_q_head_num = q.shape[1]
    return torch.zeros(
        num_tokens,
        tp_q_head_num * v_head_dim,
        dtype=q.dtype,
        device=q.device,
    )


# ---------------------------------------------------------------------------
# Dump per-rank hidden_states[slots 0..3] around the CP-v2 model forward.
# ---------------------------------------------------------------------------


def dump_hs_active() -> bool:
    """True iff SGLANG_TPU_PCP_DUMP_HS is set."""
    return _DUMP_HS


def log_hidden_states_slot0(*, tag: str, hidden_states: torch.Tensor) -> None:
    """Emit [HS_LOCAL_S] lines for slots 0..3 of the local sharded hidden_states.

    ``hidden_states`` is the per-rank rank-major-packed tensor of shape
    ``(local_padded_num_tokens, hidden_dim)`` returned by
    ``Qwen3MoeModel.forward`` under CP-v2. Slot 0 holds rank r's first-owned
    global position (rank 0 → pos 0, rank 1 → pos 1, etc. with interleave=1),
    slot 1 = rank r's second-owned (rank 0 → pos 4, rank 1 → pos 5, ...), etc.

    Prints norm + first 8 fp32 elements per rank per (slot in [0,1,2,3]) to
    stderr with the tag ``[HS_LOCAL_S tag=<tag> rank=r fwd=n slot=k]``. No-op
    when the env is unset, which lets the compiler constant-fold the call
    site. Runs entirely in eager code (called from the eager runner after
    ``model.model()`` returns) so tolist/item are safe.
    """
    global _FORWARD_COUNT
    if not _DUMP_HS:
        return
    if hidden_states.dim() < 2 or hidden_states.shape[0] == 0:
        return
    # First tag of a forward increments the counter so per-rank lines share
    # a fwd index even when the PCP-layer-trace hooks are off. When the
    # layer-level hooks ARE on, ``forward_started`` (layer 0) has already
    # bumped for this forward — don't double-bump.
    if tag == "post_model" and not (dump_inputs_active() or layer_trace_active()):
        _FORWARD_COUNT += 1
    rank = os.environ.get("LOCAL_RANK", "0")
    fwd = _FORWARD_COUNT
    n_slots_to_dump = min(4, int(hidden_states.shape[0]))
    for slot in range(n_slots_to_dump):
        try:
            row = hidden_states[slot].detach().to("cpu").to(torch.float32)
            norm = float(row.norm().item())
            head = row.reshape(-1)[:8].tolist()
        except Exception as e:  # pragma: no cover — diagnostic only
            norm = float("nan")
            head = [repr(e)]
        head_str = ",".join(f"{x:+.4e}" for x in head if isinstance(x, float))
        line = (
            f"[HS_LOCAL_S tag={tag} rank={rank} fwd={fwd} slot={slot} "
            f"shape={tuple(hidden_states.shape)} "
            f"norm={norm:.4f} head8=[{head_str}]]"
        )
        with _TRACE_LOCK:
            print(line, flush=True)


# ---------------------------------------------------------------------------
# After a forward completes, dump layer-0's KV cache values at the first 4
# out_cache_loc slots for this rank. Runs in eager runner (post model.model())
# so free of torch.compile interactions.
# ---------------------------------------------------------------------------


def dump_kv_slot_active() -> bool:
    """True iff SGLANG_TPU_PCP_DUMP_KV_SLOT is set."""
    return _DUMP_KV_SLOT


def log_kv_slot0(*, token_to_kv_pool: Any, forward_batch: Any) -> None:
    """Emit [KV_SLOT] lines with layer-0 K/V at the first 4 out_cache_loc slots.

    Prints norm + first 8 fp32 elements of the fused K/V slice at
    ``forward_batch.out_cache_loc[i]`` for ``i in range(min(4, n_slots))``.
    Runs in the eager runner post-forward, so torch.compile is not active.
    """
    if not _DUMP_KV_SLOT:
        return
    rank = os.environ.get("LOCAL_RANK", "0")
    fwd = _FORWARD_COUNT
    try:
        kv_buffer = token_to_kv_pool.kv_buffer[0]
    except Exception as e:  # pragma: no cover
        print(f"[KV_SLOT rank={rank} fwd={fwd} err_get_kv_buffer={e!r}]", flush=True)
        return
    # kv_buffer shape: [num_blocks, page_size, num_kv_heads_x2 // kv_packing,
    #                   kv_packing, head_dim]  (5D preferred)
    # OR fused pool layout: [num_blocks, page_size, num_kv_heads_x2, head_dim]
    try:
        # Use the forward_batch's own out_cache_loc — this is the actual
        # slot id assigned by the allocator for req 0's first token, and it
        # points at the physical page where the PCP kernel just wrote K/V.
        # Falling back to req_to_token[0, 0] reads a stale warmup slot on a
        # fresh server (the warmup used a dummy req; real reqs get fresh
        # allocations and land at different physical pages).
        out_cache_loc = forward_batch.out_cache_loc
        page_size = token_to_kv_pool.page_size
        # Log the first 4 slots' K/V so we can see slot 0 explicitly per rank.
        for real_pos in range(min(4, int(out_cache_loc.shape[0]))):
            slot_id = int(out_cache_loc[real_pos].item())
            page_idx = slot_id // page_size
            page_offset = slot_id % page_size
            slice_kv = kv_buffer[page_idx, page_offset]
            flat = slice_kv.detach().to("cpu").to(torch.float32).reshape(-1)
            norm = float(flat.norm().item())
            head = ",".join(f"{x:+.4e}" for x in flat[:8].tolist())
            line = (
                f"[KV_SLOT rank={rank} fwd={fwd} req_pos={real_pos} "
                f"slot_id={slot_id} page={page_idx} off={page_offset} "
                f"kv_shape={tuple(kv_buffer.shape)} slice_shape={tuple(slice_kv.shape)} "
                f"norm={norm:.4f} head8=[{head}]]"
            )
            with _TRACE_LOCK:
                print(line, flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[KV_SLOT rank={rank} fwd={fwd} err={e!r}]", flush=True)


def install_jit_trace_hook_if_enabled() -> None:
    """Monkey-patch JaxCallable.__call__ to log first-call hashes.

    Called once at module import (see module footer). No-op when
    SGLANG_TPU_PCP_JIT_TRACE_HASH is unset. Env is read once at import;
    setting it after import has no effect. Idempotent.
    """
    global _JIT_TRACE_INSTALLED
    if _JIT_TRACE_PATH is None or _JIT_TRACE_INSTALLED:
        return
    _JIT_TRACE_INSTALLED = True

    from torch_tpu._internal.pallas import pallas as pallas_impl

    JaxCallable = pallas_impl.JaxCallable
    orig_call = JaxCallable.__call__
    seen: set[str] = set()
    seen_lock = threading.Lock()

    def _traced_call(self, *args, **kwargs):
        try:
            key = pallas_impl._get_kernel_invocation_key(
                self.trace_key, args, kwargs, self.static_argnums
            )
        except Exception as e:  # pragma: no cover — diagnostic only
            key = f"unknown({e!r})"
        fingerprint = f"{self.name}::{key}"
        emit = False
        with seen_lock:
            if fingerprint not in seen:
                seen.add(fingerprint)
                emit = True
        if emit:
            tb = "".join(traceback.format_stack(limit=25))
            shape_desc = "; ".join(
                (
                    f"a{i}:{tuple(a.shape)}:{a.dtype}"
                    if isinstance(a, torch.Tensor)
                    else f"a{i}:{type(a).__name__}"
                )
                for i, a in enumerate(args)
            )
            rank = os.environ.get("LOCAL_RANK", "0")
            with _TRACE_LOCK:
                with open(_JIT_TRACE_PATH, "a", buffering=1) as f:
                    f.write(
                        f"\n===== FIRST CALL rank={rank} "
                        f"pid={os.getpid()} fwd={_FORWARD_COUNT} "
                        f"{self.name} key={key} =====\n"
                    )
                    f.write(f"shapes: {shape_desc}\n")
                    f.write(f"traceback:\n{tb}\n")
        return orig_call(self, *args, **kwargs)

    JaxCallable.__call__ = _traced_call


# Install the JIT trace hook at import. No-op unless
# SGLANG_TPU_PCP_JIT_TRACE_HASH is set at Python launch time (setting it
# after import has no effect — see the function docstring). Auto-install
# is required so the hook wraps JaxCallable.__call__ before any Pallas
# kernel invocation fires.
install_jit_trace_hook_if_enabled()
