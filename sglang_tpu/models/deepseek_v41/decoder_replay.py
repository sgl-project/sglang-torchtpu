# SPDX-License-Identifier: Apache-2.0
"""Decoder SWA bounded replay (CED prefill) on the DeepSeek-V4.1 TPU backend.

Under ``--enable-decoder-swa-bounded-replay`` a pure extend runs the layers past
the last kv_source layer (core ``late_layer_start``) only on each request's
decoder rows (``plan_decoder_rows``), with their SWA window floored at the
request's replay start. Those layers keep no state but that window: the
compressed records, index keys and candidates they read come from the
kv_source / candidate source layers, which still see every row.

Row rule (``SGLANG_TPU_DSV41_DECODER_REPLAY_ROWS``):

- ``chunk`` (default): ``max(end - min(window, extend), floor)``, each extend's
  last window (core ``late_layer_tail_layout``, as GPU SGLang serves it).
- ``prompt``: ``max(fill_len - window, floor)``, the prompt's last window. A
  chunk wholly before it keeps no rows, so the result does not depend on
  chunking (vLLM-TorchTPU's CED rule).

``floor`` is the first row of an extend that restarts the request's SWA window
(a new slot or a retraction; a folded prefix hit starts at its replay start),
else 0. The scheduler (MLP-sync row count, before the fold) and the backend
(after the fold) both use ``plan_decoder_rows``.

The row count is a compiled shape. ``DeepseekV4DecoderTpuCompilePolicy`` picks a
bucket every DP rank shares (``decoder_rows_bucket``: 0, 128, 256, 512 or the
token bucket) from the MLP-sync decoder row counts, so the EP MoE of the decoder
layers runs the same rows on every rank; bucket 0 skips the decoder layers
(``skip_empty_decoder``). The
backend builds a second ``DSV41TPUForwardMetadata`` for those rows (its own
positions, request rows, SWA floor and request-window moves) and the model
switches onto it through ``enter_late_layer_tail`` / ``exit_late_layer_tail``.

Padding rows of the bucket gather a full-batch row the decoder does not keep
and are invalid in the tail metadata: they write no cache, and the scatter
puts them back on that row. The full layout starts zeroed
(``scatter_decoder_rows``), so rows the decoder skipped read as zeros. The
decoder output and the DSpark aux rows go back to the full row layout (core
scatters without DSpark capture; ``scatter_dspark_capture`` with it), so logits
and the spec worker index rows as without replay.
"""

import contextlib
from typing import List, NamedTuple, Optional, Sequence

import torch

# =============================================================================
# Row plan
# =============================================================================

DECODER_REPLAY_WINDOW = 128
ROW_RULES = ("prompt", "chunk")


def decoder_row_rule() -> str:
    from sglang_tpu.envs import envs as tpu_envs

    rule = tpu_envs.SGLANG_TPU_DSV41_DECODER_REPLAY_ROWS.get()
    if rule not in ROW_RULES:
        raise ValueError(
            "SGLANG_TPU_DSV41_DECODER_REPLAY_ROWS must be one of "
            f"{ROW_RULES}, got {rule!r}"
        )
    return rule


class DecoderRowPlan(NamedTuple):
    # [bs] absolute position of each request's first decoder row (its SWA floor).
    replay_starts: List[int]
    # [bs] decoder rows of each request: its extend rows at or after replay_start.
    kept: List[int]

    @property
    def num_rows(self) -> int:
        return sum(self.kept)


def plan_decoder_rows(
    *,
    starts: Sequence[int],
    extend_lens: Sequence[int],
    fill_lens: Sequence[int],
    floors: Sequence[int],
    window: int = DECODER_REPLAY_WINDOW,
    rule: Optional[str] = None,
) -> DecoderRowPlan:
    """``starts``: first extend position of each request (after any fold);
    ``extend_lens``: its extend rows; ``fill_lens``: its full fill length (prompt
    plus output ids of a retracted request); ``floors``: its window floor."""
    rule = rule or decoder_row_rule()
    replay_starts, kept = [], []
    for start, n, fill, floor in zip(starts, extend_lens, fill_lens, floors):
        end = start + n
        if rule == "prompt":
            if end > fill:
                raise ValueError(f"extend ends at {end}, past the fill length {fill}")
            replay = max(fill - window, floor, 0)
        else:
            replay = max(end - min(window, n), floor, 0)
        replay_starts.append(replay)
        kept.append(min(n, max(0, end - max(replay, start))))
    return DecoderRowPlan(replay_starts, kept)


def scheduled_decoder_rows(batch, *, folds: bool) -> int:
    """This rank's MLP-sync decoder row count of a scheduled batch: an extend's
    decoder rows, counting the rows the encoder-replay fold will prepend
    (``folds``); a decode batch's rows (a peer's extend turns it into a
    one-row-per-request extend under MAX_LEN padding); else 0."""
    if batch is None:
        return 0
    if batch.forward_mode.is_decode():
        return batch.batch_size()
    if not batch.forward_mode.is_extend_without_speculative():
        return 0
    return scheduled_decoder_plan(batch, folds=folds).num_rows


def scheduled_decoder_plan(batch, *, folds: bool) -> DecoderRowPlan:
    """``plan_decoder_rows`` of a scheduled extend batch (before the fold)."""
    from sglang.srt.model_executor.encoder_swa_replay import replay_len

    resets = batch.encoder_swa_reset or [False] * len(batch.reqs)
    starts, ext, floors = [], [], []
    for prefix, n, reset in zip(batch.prefix_lens, batch.extend_lens, resets):
        replay = replay_len(prefix) if (folds and reset and prefix) else 0
        starts.append(prefix - replay)
        ext.append(n + replay)
        floors.append(prefix - replay if reset else 0)
    return plan_decoder_rows(
        starts=starts,
        extend_lens=ext,
        fill_lens=fill_lens_or_ends(
            getattr(batch, "decoder_fill_lens", None), starts, ext
        ),
        floors=floors,
    )


def fill_lens_or_ends(fill_lens, starts, extend_lens) -> List[int]:
    """A batch built without fill lengths (an extend converted from decode)
    keeps each extend's last window: its end stands in for the fill length."""
    if fill_lens is not None:
        return list(fill_lens)
    return [s + n for s, n in zip(starts, extend_lens)]


# =============================================================================
# Buckets and tail rows
# =============================================================================

# Decoder row buckets below the token bucket; the token bucket itself is the last.
DECODER_ROW_BUCKETS = (128, 256, 512)


def decoder_rows_bucket(max_rows: int, token_bucket: int) -> int:
    """The smallest shared bucket holding ``max_rows`` decoder rows: 0 when no
    rank keeps a row (every request mid-prompt under the prompt rule; the model
    then skips the decoder layers, ``skip_empty_decoder``), else one of
    DECODER_ROW_BUCKETS below the token bucket, else the token bucket."""
    if max_rows <= 0:
        return 0
    if max_rows > token_bucket:
        raise ValueError(f"{max_rows} decoder rows exceed the {token_bucket} bucket")
    for bucket in DECODER_ROW_BUCKETS:
        if max_rows <= bucket < token_bucket:
            return bucket
    return token_bucket


class TailRows(NamedTuple):
    """Host plan of one forward's decoder rows."""

    # [D] int64 full-batch row of each decoder row. Padding rows read (and core's
    # scatter writes back to) one full-batch row the decoder does not keep, which
    # no reader consumes.
    gather: torch.Tensor
    # [B] int64 decoder rows of each batch row (bucket-padding requests 0).
    counts: torch.Tensor
    # [B] int64 replay start of each request: the decoder SWA floor.
    replay_starts: torch.Tensor
    num_real: int


def _spare_row(real: torch.Tensor, num_tokens: int) -> int:
    """A full-batch row the decoder does not keep (exists whenever padding does:
    the real rows are fewer than the bucket, which is at most the token bucket)."""
    kept = torch.zeros(num_tokens, dtype=torch.bool)
    kept[real] = True
    spare = (~kept).nonzero()
    return int(spare[-1]) if spare.numel() else 0


def plan_tail_rows(
    *,
    fill_lens: Optional[List[int]],
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    valid: torch.Tensor,
    floors: torch.Tensor,
    bucket: Optional[int],
    warmup: bool = False,
) -> TailRows:
    """``query_start_loc`` [B + 1], ``positions`` [T], ``valid`` [T] and
    ``floors`` [B] (the fold's SWA floor) are host tensors of the full batch.
    Only valid rows are decoder rows, so a DP rank materialized as a dummy
    extend runs the decoder on padding rows only. A ``warmup`` dummy compiles
    the bucket's shape with padding rows only."""
    bs = query_start_loc.shape[0] - 1
    num_tokens = positions.shape[0]
    if warmup:
        assert bucket is not None
        return TailRows(
            gather=torch.full((bucket,), num_tokens - 1, dtype=torch.int64),
            counts=torch.zeros(bs, dtype=torch.int64),
            replay_starts=torch.zeros(bs, dtype=torch.int64),
            num_real=0,
        )
    counts = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
    live = [b for b in range(bs) if counts[b] > 0]
    starts = [int(positions[query_start_loc[b]]) for b in live]
    ext = [counts[b] for b in live]
    fills = fill_lens_or_ends(
        None if fill_lens is None else [fill_lens[b] for b in live], starts, ext
    )
    plan = plan_decoder_rows(
        starts=starts,
        extend_lens=ext,
        fill_lens=fills,
        floors=[int(floors[b]) for b in live],
    )
    rows: List[torch.Tensor] = []
    tail_counts = torch.zeros(bs, dtype=torch.int64)
    replay_starts = torch.zeros(bs, dtype=torch.int64)
    for b, kept, replay in zip(live, plan.kept, plan.replay_starts):
        end = int(query_start_loc[b + 1])
        idx = torch.arange(end - kept, end, dtype=torch.int64)
        idx = idx[valid[idx]]
        rows.append(idx)
        tail_counts[b] = idx.shape[0]
        replay_starts[b] = replay
    real = torch.cat(rows) if rows else torch.zeros(0, dtype=torch.int64)
    num_real = int(real.shape[0])
    size = num_real if bucket is None else int(bucket)
    if num_real > size:
        raise RuntimeError(
            f"{num_real} decoder rows on this rank exceed the shared bucket {size}; "
            "the MLP-sync decoder row count disagrees with the attention backend"
        )
    pad = size - num_real
    spare = _spare_row(real, num_tokens) if pad else 0
    return TailRows(
        gather=torch.cat([real, real.new_full((pad,), spare)]),
        counts=tail_counts,
        replay_starts=replay_starts,
        num_real=num_real,
    )


class DecoderTail:
    """Core ``LateLayerTail`` for the TPU backend: the decoder rows of the full
    batch, padded to their bucket (core's ``_scatter_tail_rows`` brings them back
    through ``token_indices``).

    ``rows`` also serves core's ``tail.rows(input_ids_global)``: under attention
    DP that tensor is the DP-wide gather of the batch's ids, which the MoE
    routing reads one per gathered row. Core takes ``tail.rows(input_ids)`` right
    before it, so the tail re-gathers the DP-wide ids of its own rows (the DP
    buffer is the decoder bucket's by then).
    """

    pad_rows = 0
    cp_metadata = None
    contiguous_start = None
    # Core LateLayerTail fields of CUDA prefill graphs and CP; None keeps
    # core's token_indices path.
    scatter_indices = None
    global_token_indices = None
    real_rows_count = None

    def __init__(
        self, *, gather, positions, num_tokens: int, backend=None, inverse=None
    ):
        self.inverse = inverse
        self.token_indices = gather
        self.positions = positions
        self.num_rows = int(gather.shape[0])
        self._num_tokens = int(num_tokens)
        self._backend = backend
        self._last_ids: Optional[torch.Tensor] = None

    def _is_dp_global(self, t: torch.Tensor) -> bool:
        # Only the DP-wide id gather has another row count than the batch.
        return (
            t.dim() == 1
            and not t.is_floating_point()
            and t.shape[0] != self._num_tokens
        )

    def rows(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        if self._is_dp_global(t):
            if self._last_ids is None:
                raise RuntimeError("the DP-wide ids came before this rank's ids")
            return self.global_ids(self._last_ids, self._backend.tail_forward_batch)
        out = t.index_select(0, self.token_indices.to(torch.int64))
        if t.dim() == 1 and not t.is_floating_point():
            self._last_ids = out
        return out

    real_rows = rows

    @staticmethod
    def global_ids(ids: torch.Tensor, forward_batch) -> torch.Tensor:
        """The decoder rows' DP-wide ids, gathered like core's forward start."""
        from sglang.srt.layers.dp_attention import (
            dp_gather_replicate,
            get_global_dp_buffer_len,
        )
        from sglang.srt.layers.moe import get_moe_a2a_backend
        from sglang.srt.runtime_context import get_parallel

        if get_parallel().attn_dp_size <= 1 or not get_moe_a2a_backend().is_none():
            return ids
        out = torch.empty(
            (get_global_dp_buffer_len(), 1), dtype=ids.dtype, device=ids.device
        )
        dp_gather_replicate(out, ids[:, None].clone(), forward_batch)
        return out.squeeze(-1)


def start_tail_after_last_source(result, model, *args, **kwargs) -> None:
    """AFTER DeepseekV4Model.__init__: start the decoder tail after the last
    kv_source layer. Core starts it at that layer and has the CUDA backend write
    its compressed KV for every row (write_tail_source_kv); this backend runs that
    layer on every row instead, so its main KV covers all of them."""
    if model.tail_writes_source_kv:
        model.tail_writes_source_kv = False
        model.late_layer_start += 1


def lockstep_late_layer_dp_counts(
    original, forward_batch, local_late_rows, *, in_graph
):
    """AROUND late_layer_dp_counts: every DP rank runs the decoder layers on one
    shared row bucket (DSV41TPUAttnBackend.enter_late_layer_tail), so each rank's
    count is that bucket, with no exchange. Core still re-gathers the decoder
    rows' DP-wide ids (vision top-k reads them) with these counts."""
    from sglang.srt.runtime_context import get_parallel

    return [int(local_late_rows)] * get_parallel().attn_dp_size


def keep_backend_late_layer_dp(original, forward_batch, counts):
    """AROUND enter_late_layer_dp: the backend already switched the batch's DP
    layout to the decoder bucket (MAX_LEN). Core's SUM_LEN switch would send the
    decoder layers' DP gathers through the Triton memcpy; None skips its exit."""
    return None


def skip_empty_decoder(
    original,
    model,
    positions,
    hidden_states,
    forward_batch,
    input_ids,
    input_ids_global,
    capture_dspark,
    dspark_aux_hidden_states,
):
    """AROUND DeepseekV4Model._forward_layers_hc_pre_from_prev: when no DP rank
    keeps a decoder row (bucket 0), stop the layer loop at the first decoder
    layer. Every row's output is then unread (each request is mid-prompt), so
    the decoder layers and their MoE collectives are skipped on every rank
    together; the DSpark aux rows of skipped capture layers are zeros."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    start = getattr(model, "late_layer_start", None)
    if start is None or not getattr(get_attn_backend(), "skips_decoder", False):
        return original(
            model,
            positions,
            hidden_states,
            forward_batch,
            input_ids,
            input_ids_global,
            capture_dspark,
            dspark_aux_hidden_states,
        )
    end = model.end_layer
    model.end_layer = min(end, start)
    try:
        out = original(
            model,
            positions,
            hidden_states,
            forward_batch,
            input_ids,
            input_ids_global,
            capture_dspark,
            dspark_aux_hidden_states,
        )
    finally:
        model.end_layer = end
    if capture_dspark:
        rows = hidden_states.shape[0]
        width = hidden_states.shape[-1]
        for i in sorted(model.dspark_layers_to_capture):
            if start <= i < end:
                dspark_aux_hidden_states.append(hidden_states.new_zeros((rows, width)))
    return out


def tail_gather_scatter_enabled() -> bool:
    """SGLANG_TPU_DSV41_TAIL_GATHER_SCATTER (default on): the decoder rows return
    to the full layout by a gather over a host-built inverse map; 0 scatters."""
    from sglang_tpu.envs import envs as tpu_envs

    return tpu_envs.SGLANG_TPU_DSV41_TAIL_GATHER_SCATTER.get()


def tail_inverse_rows(
    gather: torch.Tensor, num_real: int, num_tokens: int
) -> torch.Tensor:
    """[num_tokens] int32: the decoder row of each full-batch row, -1 if none.
    Only the ``num_real`` real decoder rows map back; padding rows map nowhere."""
    inverse = torch.full((num_tokens,), -1, dtype=torch.int32)
    real = gather[:num_real].to(torch.int64)
    inverse[real] = torch.arange(num_real, dtype=torch.int32)
    return inverse


def _scatter_zeroed(tail: "DecoderTail", rows: torch.Tensor, num_tokens: int):
    """The decoder rows back in the full layout; rows the decoder skipped are
    zeros. With the inverse map this is a gather (padding rows are dropped);
    without it a scatter, where padding rows land on one row the decoder does
    not keep."""
    inverse = tail.inverse
    if inverse is not None:
        # A gather, not a scatter: every full-batch row takes its decoder row, or
        # zero where the decoder skipped it (rows padding gathered from included).
        keep = (inverse >= 0).reshape(-1, *([1] * (rows.dim() - 1)))
        taken = rows.index_select(0, inverse.clamp(min=0).to(torch.int64))
        return torch.where(keep, taken, torch.zeros_like(taken))
    full = rows.new_zeros((num_tokens, *rows.shape[1:]))
    full[tail.token_indices] = rows[: tail.token_indices.shape[0]]
    return full


def scatter_decoder_rows(original, *args, **kwargs):
    """AROUND deepseek_v4._scatter_tail_rows: a DecoderTail's full layout starts
    zeroed. Under the ``prompt`` rule a chunk may keep no decoder rows, yet the
    logits processor still reads its last row (the sample is discarded later),
    so every row it can read must be defined. Other tails go to core unchanged."""
    tail = kwargs["tail"] if "tail" in kwargs else args[0]
    if not isinstance(tail, DecoderTail):
        return original(*args, **kwargs)
    rows = kwargs["rows"] if "rows" in kwargs else args[1]
    num_tokens = kwargs["num_tokens"] if "num_tokens" in kwargs else args[2]
    return _scatter_zeroed(tail, rows, num_tokens)


class FullLayoutTail:
    """What DeepseekV4ForCausalLM's compact-tail branch reads once
    ``scatter_dspark_capture`` put the rows back in the full layout: the
    identity (the batch's own rows and extend lengths)."""

    token_indices = None
    global_token_indices = None
    real_rows_count = None

    def __init__(self, forward_batch):
        self.extend_seq_lens = forward_batch.extend_seq_lens
        self.extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu

    def rows(self, t):
        return t


@contextlib.contextmanager
def _model_forward(backend):
    """Mark ``backend`` as inside a model forward, where core reads its tail."""
    outer = backend.in_model_forward
    backend.in_model_forward = True
    try:
        yield
    finally:
        backend.in_model_forward = outer


def scatter_dspark_capture(
    original, model, input_ids, positions, forward_batch, *args, **kwargs
):
    """AROUND DeepseekV4Model.forward: under DSpark capture core returns the
    hidden, pre-head and aux rows in the tail layout (its compact rebuild in the
    causal-LM head expects a tail without padding); a DecoderTail is padded to a
    compiled bucket, so put them back in the full layout here and have the head
    read its compact-tail branch as the identity (``FullLayoutTail``)."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    with _model_forward(get_attn_backend()):
        return _scatter_dspark_capture(
            original, model, input_ids, positions, forward_batch, *args, **kwargs
        )


def _scatter_dspark_capture(
    original, model, input_ids, positions, forward_batch, *args, **kwargs
):
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    # The layer loop reads the real tail: never a previous forward's identity.
    if getattr(backend, "head_full_layout", None) is not None:
        backend.head_full_layout = None
    out = original(model, input_ids, positions, forward_batch, *args, **kwargs)
    if getattr(model, "dspark_layers_to_capture", None) is None:
        return out
    view = getattr(backend, "tail_forward_metadata", None)
    tail = None if view is None else view.late_layer_tail
    # After this hook every DSpark capture is in the full layout (no tail, a
    # skipped decoder, or scattered back below): the head reads the identity.
    backend.head_full_layout = forward_batch
    if not isinstance(tail, DecoderTail):
        return out
    (hidden, pre_hc_head), aux = out
    num_tokens = input_ids.shape[0]
    # Core packs the captures into one [rows, K * hidden] tensor.
    if isinstance(aux, torch.Tensor):
        aux = _scatter_zeroed(tail, aux, num_tokens)
    else:
        aux = [_scatter_zeroed(tail, a, num_tokens) for a in aux]
    return (
        _scatter_zeroed(tail, hidden, num_tokens),
        _scatter_zeroed(tail, pre_hc_head, num_tokens),
    ), aux


def clear_full_layout_head(original, lm, *args, **kwargs):
    """AROUND DeepseekV4ForCausalLM.forward: the identity tail lasts one forward,
    and the backend reports a tail only inside it (``in_model_forward``)."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    try:
        with _model_forward(backend):
            return original(lm, *args, **kwargs)
    finally:
        if getattr(backend, "head_full_layout", None) is not None:
            backend.head_full_layout = None


class TailView(NamedTuple):
    """What core reads as ``attn_backend.tail_forward_metadata``."""

    late_layer_tail: Optional[DecoderTail]
