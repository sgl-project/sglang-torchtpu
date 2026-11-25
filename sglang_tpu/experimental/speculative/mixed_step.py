# SPDX-License-Identifier: Apache-2.0
"""Host-side layout of a mixed prefill + verify forward (Stage B worker side).

A MIXED step runs one target forward per DP rank holding the rank's prompt chunk
rows (its prompt batch, folded for encoder SWA replay) followed by its verify
requests' ``num_verify_rows`` rows each (``sglang.srt.speculative.mixed_verify``).
The batch dimension is ``[prompt real][verify real][tail pad]``; the shim pads
only the tail. The logits row set is fixed-size on every rank,
``L = ext_cap + num_verify_rows * pad_bv`` (``compiled_forward_shim.
mixed_verify_caps``): each prompt request's last row in ``[0, ext_cap)`` and every
verify row in ``[ext_cap, L)``, padded with a dummy row, so a vocab-parallel LM
head (a cross-DP collective) runs the same rows everywhere.

Positions are built on the host: the prompt rows from their (folded) prefix and
extend lengths, the verify rows from ``StepHostRows`` (the host twin of
``select_verify_rows``), so the host input mirror matches the device inputs.
"""

from copy import copy
from typing import NamedTuple, Optional, Sequence

import torch

# The dummy row the logits index pads with: row 0 always exists.
DUMMY_ROW = 0


def mixed_logits_index(
    prompt_lens: Sequence[int],
    n_verify: int,
    *,
    ext_cap: int,
    pad_bv: int,
    num_verify_rows: int,
) -> torch.Tensor:
    """[L] int64 host index into the forward's rows.

    ``prompt_lens``: each prompt request's (folded) row count, in batch order;
    the verify rows follow the prompt rows."""
    if len(prompt_lens) > ext_cap:
        raise ValueError(f"{len(prompt_lens)} prompt requests exceed ext_cap {ext_cap}")
    if n_verify > pad_bv:
        raise ValueError(f"{n_verify} verify requests exceed the bucket {pad_bv}")
    lens = torch.tensor(list(prompt_lens), dtype=torch.int64)
    prompt_rows = int(lens.sum())
    index = torch.full(
        (ext_cap + num_verify_rows * pad_bv,), DUMMY_ROW, dtype=torch.int64
    )
    if lens.numel():
        index[: lens.numel()] = torch.cumsum(lens, 0) - 1
    nv = n_verify * num_verify_rows
    index[ext_cap : ext_cap + nv] = prompt_rows + torch.arange(nv, dtype=torch.int64)
    return index


def prompt_positions(
    prefix_lens: Sequence[int], extend_lens: Sequence[int]
) -> torch.Tensor:
    """[sum(extend_lens)] int64 positions of a (folded) prompt batch's rows."""
    parts = [
        torch.arange(p, p + n, dtype=torch.int64)
        for p, n in zip(prefix_lens, extend_lens)
    ]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.int64)


class FoldedRowMaps(NamedTuple):
    """The encoder-replay fold's row maps over the whole mixed forward."""

    row_floor: torch.Tensor  # [rows] int64 SWA floor (verify rows: 0)
    compress_rows: torch.Tensor  # [kept rows] int64 rows that compress/publish


def extend_folded_row_maps(folded, n_verify_rows: int) -> FoldedRowMaps:
    """``folded``'s host row maps (prompt part) extended over the verify rows that
    follow the folded prompt rows: floor 0, and they compress (they are not
    replay rows)."""
    base = int(folded.num_rows)
    verify = torch.arange(base, base + n_verify_rows, dtype=torch.int64)
    return FoldedRowMaps(
        row_floor=torch.cat(
            [
                folded.row_floor.to("cpu", torch.int64),
                torch.zeros(n_verify_rows, dtype=torch.int64),
            ]
        ),
        compress_rows=torch.cat([folded.keep_rows.to("cpu", torch.int64), verify]),
    )


class VerifySegment(NamedTuple):
    """The verify part's rows of a mixed forward (host twins for the mirror)."""

    reqs: list
    req_pool_indices: torch.Tensor  # [n_verify] device
    req_pool_indices_cpu: torch.Tensor  # [n_verify] host
    seq_lens: torch.Tensor  # [n_verify] device, KV length after the verify rows
    seq_lens_cpu: torch.Tensor  # [n_verify] host
    committed_lens_cpu: list  # [n_verify] host committed length (prefix)
    input_ids: torch.Tensor  # [n_verify * R] device verify tokens
    positions_cpu: torch.Tensor  # [n_verify * R] host (StepHostRows)
    out_cache_loc: torch.Tensor  # [n_verify * R] device
    num_verify_rows: int


def merge_mixed_batch(
    prompt,
    verify: Optional[VerifySegment],
    *,
    prompt_input_ids: torch.Tensor,
    spec_info,
    mixed_mode,
):
    """A ScheduleBatch view of the mixed forward: ``prompt`` (already folded),
    then the verify segment. Per-request lists are extended over the verify
    requests; the fold is not applied again (encoder_swa_reset is cleared)."""
    merged = copy(prompt)
    merged.forward_mode = mixed_mode
    # A MIXED forward always samples. An empty prompt view (idle_like:
    # ScheduleBatch.init_new([])) carries is_prefill_only = all([]) = True,
    # which would send init_new to the pooled-output request fields.
    merged.is_prefill_only = False
    merged.encoder_swa_reset = None
    merged.spec_info = spec_info
    merged.return_logprob = False
    merged.top_logprobs_nums = None
    merged.token_ids_logprobs = None
    merged.extend_input_logprob_token_ids = None
    # Ignored by direct compile on TPU: None on every rank kind, so the policy
    # never pads a prompt-only rank's per prompt request count.
    merged.orig_seq_lens = None
    if verify is None:
        merged.input_ids = prompt_input_ids
        return merged
    n_v, R = len(verify.reqs), verify.num_verify_rows
    device = verify.input_ids.device
    merged.reqs = list(prompt.reqs) + list(verify.reqs)
    merged.input_ids = torch.cat([prompt_input_ids.to(device), verify.input_ids])
    merged.out_cache_loc = torch.cat(
        [
            prompt.out_cache_loc.to(
                verify.out_cache_loc.device, verify.out_cache_loc.dtype
            ),
            verify.out_cache_loc,
        ]
    )
    merged.prefix_lens = list(prompt.prefix_lens) + list(verify.committed_lens_cpu)
    merged.extend_lens = list(prompt.extend_lens) + [R] * n_v
    merged.extend_num_tokens = int(prompt.extend_num_tokens) + R * n_v
    merged.extend_logprob_start_lens = (
        None
        if prompt.extend_logprob_start_lens is None
        else list(prompt.extend_logprob_start_lens) + [R] * n_v
    )
    merged.seq_lens = torch.cat(
        [prompt.seq_lens.to(device), verify.seq_lens.to(prompt.seq_lens.dtype)]
    )
    merged.seq_lens_cpu = torch.cat(
        [prompt.seq_lens_cpu, verify.seq_lens_cpu.to(prompt.seq_lens_cpu.dtype)]
    )
    merged.seq_lens_sum = int(merged.seq_lens_cpu.sum())
    merged.req_pool_indices = torch.cat(
        [
            prompt.req_pool_indices.to(device),
            verify.req_pool_indices.to(prompt.req_pool_indices.dtype),
        ]
    )
    merged.req_pool_indices_cpu = torch.cat(
        [
            prompt.req_pool_indices_cpu,
            verify.req_pool_indices_cpu.to(prompt.req_pool_indices_cpu.dtype),
        ]
    )
    if prompt.multimodal_inputs is not None:
        merged.multimodal_inputs = list(prompt.multimodal_inputs) + [None] * n_v
    if getattr(prompt, "decoder_fill_lens", None) is not None:
        # Verify rows end their request's fill: a planner would keep them all.
        merged.decoder_fill_lens = list(prompt.decoder_fill_lens) + [
            int(c) + R for c in verify.committed_lens_cpu
        ]
    merged.prefill_input_ids_cpu = None
    return merged


def mixed_positions_cpu(prompt, verify: Optional[VerifySegment]) -> torch.Tensor:
    """Host positions of the mixed forward's real rows (prompt, then verify)."""
    pos = prompt_positions(prompt.prefix_lens, prompt.extend_lens)
    if verify is None:
        return pos
    return torch.cat([pos, verify.positions_cpu.to(torch.int64)])


# The fields the DP MLP sync writes on a batch (scheduler_components/dp_attn.py),
# carried onto an idle stand-in so its forward joins the same step.
_MLP_SYNC_FIELDS = (
    "global_num_tokens",
    "global_num_tokens_for_logprob",
    "draft_global_num_tokens",
    "draft_global_num_tokens_for_logprob",
    "is_extend_in_batch",
    "is_tpu_mm_extend_in_batch",
    "tbo_split_seq_index",
    "global_forward_mode",
    "can_run_decode_cuda_graph",
    "can_run_dp_draft_cuda_graph",
    "can_run_dp_prefill_cuda_graph",
    "dp_prefill_cuda_graph_max_prefix_len",
    "dp_cooperation_info",
    # DSV4.1 decoder replay's MLP-sync column (decoder_replay_sync.batch_decoder_rows):
    # it picks the extend's decoder-row bucket, which every rank must agree on.
    "global_decoder_rows",
)


def idle_like(batch, **overrides):
    """An IDLE ScheduleBatch on ``batch``'s pools, with its DP sync fields, for a
    rank that has no part of one half of a mixed step but must enter that half's
    collectives. ``overrides`` replace sync fields (e.g. the verify half's
    ``global_num_tokens``)."""
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    idle = ScheduleBatch.init_new(
        [],
        batch.req_to_token_pool,
        batch.token_to_kv_pool_allocator,
        batch.tree_cache,
        batch.model_config,
        batch.enable_overlap,
        batch.spec_algorithm,
    )
    idle.prepare_for_idle()
    for name in _MLP_SYNC_FIELDS:
        if hasattr(batch, name):
            setattr(idle, name, getattr(batch, name))
    for name, value in overrides.items():
        setattr(idle, name, value)
    return idle


def verify_sync_fields(plan) -> dict:
    """The DP sync fields of a mixed step's verify half: every rank's verify
    request count, so the decode bucket (_decode_bucket) is the plan's verify
    bucket on every rank."""
    counts = list(plan.global_verify_bs)
    return dict(
        global_num_tokens=counts,
        global_num_tokens_for_logprob=counts,
        is_extend_in_batch=False,
    )


def fused_mixed_forward() -> bool:
    """Whether a mixed step runs as one fused forward (SGLANG_TPU_MIXED_VERIFY_FUSED)
    rather than its verify half then its prompt half. Under decoder replay (CED) the
    fused forward runs the decoder layers on its prompt rows' replay rows and every
    verify row."""
    from sglang_tpu.envs import envs as tpu_envs

    return bool(tpu_envs.SGLANG_TPU_MIXED_VERIFY_FUSED.get())


# The widest step a fine bucket may add over the prompt's own token bucket.
FINE_BUCKET_ROWS = 128


def mixed_row_cap(server_args, context_len: int):
    """Row cap of fused mixed steps (platform hook spec_mixed_verify_row_cap):
    the token bucket the step's largest prompt row count pads to, so verify rows
    mix only into its slack; with SGLANG_TPU_MIXED_VERIFY_FINE_BUCKET, the next
    bucket instead when it is at most FINE_BUCKET_ROWS wider."""
    from sglang_tpu.compilation.decoder_compile_policy import forward_bucketers
    from sglang_tpu.envs import envs as tpu_envs

    t_bucketer, _ = forward_bucketers(server_args, context_len=context_len)
    fine = bool(tpu_envs.SGLANG_TPU_MIXED_VERIFY_FINE_BUCKET.get())
    return row_cap_from_bucketer(t_bucketer, fine=fine)


def row_cap_from_bucketer(t_bucketer, *, fine: bool):
    def cap(prompt_rows: int) -> int:
        bucket = t_bucketer.next_bucket(max(prompt_rows, 1))
        if fine and bucket < t_bucketer.max_val:
            wider = t_bucketer.next_bucket(bucket + 1)
            if wider - bucket <= FINE_BUCKET_ROWS:
                return wider
        return bucket

    return cap


# ---------------------------------------------------------------------------
# Fused mixed forward (SGLANG_TPU_MIXED_VERIFY_FUSED): one MIXED target forward
# holding the rank's (folded) prompt rows and its verify rows.
#
# TorchTPU compiles a view input as its whole base buffer, so every path that
# feeds the MIXED graph (prompt + verify, verify-only, prompt-only and idle
# ranks, and the warmup) hands it fresh tensors: the per-request and per-token
# fields are built on the host by torch.cat (fresh) and uploaded at the bucket
# by the compile policy, the input ids are spliced on the device into a fresh
# tensor, and the logits row index is a fresh upload.


class HostRows(NamedTuple):
    """Host twins of a verify bucket's rows (HostVerifyRows' layout)."""

    positions: torch.Tensor  # [pbs * R] int64
    out_cache_loc: torch.Tensor  # [pbs * R] int64
    req_pool_indices: torch.Tensor  # [pbs] int64


class MixedHostRows(NamedTuple):
    """What host Engram staging needs to hash a fused MIXED forward's rows:
    the prompt rows as an extend (from the host token lists, history commit at
    the forward) then the verify rows as a verify (history + verify tokens, no
    commit). Set on the ForwardBatch as an attribute, not a field."""

    prompt_reqs: list
    prompt_starts: list  # folded prefix of each prompt request
    prompt_lengths: list  # folded rows of each prompt request
    prompt_slots: list  # pool row of each prompt request
    prompt_folded: bool
    verify_tokens: Optional[torch.Tensor]  # device [pbs * R] verify tokens
    verify_positions: torch.Tensor  # host [n * R] int64 (n: verify requests)
    verify_slots: torch.Tensor  # host [n] int64
    num_verify_rows: int

    @property
    def prompt_rows(self) -> int:
        return int(sum(self.prompt_lengths))

    @property
    def verify_rows(self) -> int:
        return int(self.verify_positions.shape[0])


class MixedLayout(NamedTuple):
    """The host side of one fused MIXED forward."""

    batch: object  # merged ScheduleBatch view (MIXED), host fields
    positions: torch.Tensor  # [rows] int64 host, prompt rows then verify rows
    logits_index: torch.Tensor  # [L] int64 host
    row_floor: Optional[torch.Tensor]  # encoder-replay fold maps, or None
    compress_rows: Optional[torch.Tensor]
    prompt_rows: int  # folded prompt rows (the verify rows' offset)
    verify_rows: int  # verify rows in the forward (incl. an idle dummy request)
    real_rows: int  # rows that are not padding (num_token_non_padded)
    host_rows: MixedHostRows


class _DummyVerifyReq:
    """The idle rank's one dummy verify request (pool row 0, KV slot 0): the
    MIXED forward must hold a request; its rows are masked (non-padded 0)."""

    rid = "mixed-verify-idle"
    lora_id = None
    extend_end = 0
    full_untruncated_fill_ids = ()
    origin_input_ids = ()
    output_ids = ()
    multimodal_inputs = None
    # Optional per-request fields ForwardBatch.init_new reads on every request.
    token_type_ids = None
    dimensions = None
    multi_item_delimiter_indices = None
    return_pooled_hidden_states = False
    token_indices_to_pool = None
    decision_layout = None

    def __getattr__(self, name):
        # Any other optional request field: absent (the dummy's rows are masked,
        # non-padded 0). Private names stay errors.
        if name.startswith("_"):
            raise AttributeError(name)
        return None

    def __init__(self):
        from types import SimpleNamespace

        self.kv = SimpleNamespace(kv_committed_len=0, req_pool_idx=0)


# The merged batch's tensor fields that become MIXED forward inputs.
MIXED_INPUT_FIELDS = (
    "input_ids",
    "out_cache_loc",
    "seq_lens",
    "seq_lens_cpu",
    "req_pool_indices",
    "req_pool_indices_cpu",
)


def fresh(t):
    """``t`` as a tensor of its own (TorchTPU compiles a view input as its base)."""
    if isinstance(t, torch.Tensor) and t._base is not None:
        return t.clone()
    return t


def empty_prompt_view(batch):
    """A prompt part with no rows on ``batch``'s pools and sampling info, for
    a rank whose MIXED forward holds only verify rows."""
    view = copy(batch)
    empty = torch.zeros((0,), dtype=torch.int64)
    view.reqs = []
    view.input_ids = empty
    view.prefill_input_ids_cpu = None
    view.out_cache_loc = empty
    view.prefix_lens = []
    view.extend_lens = []
    view.extend_num_tokens = 0
    view.extend_logprob_start_lens = []
    view.seq_lens = empty
    view.seq_lens_cpu = empty
    view.seq_lens_sum = 0
    view.orig_seq_lens = None
    view.req_pool_indices = empty
    view.req_pool_indices_cpu = empty
    view.multimodal_inputs = None
    view.encoder_swa_reset = None
    view.decoder_fill_lens = None
    view.engram_history = None
    view.decoding_reqs = None
    return view


def host_prompt_ids(prompt) -> torch.Tensor:
    """A prompt part's input ids on the host (the spec worker stages them there)."""
    ids = prompt.input_ids
    if isinstance(ids, torch.Tensor) and ids.device.type == "cpu":
        return ids
    cpu = getattr(prompt, "prefill_input_ids_cpu", None)
    if isinstance(cpu, torch.Tensor):
        return cpu
    raise NotImplementedError(
        "A fused mixed step needs the prompt's input ids on the host "
        "(host-staged prefill tokens)."
    )


def build_mixed_layout(
    *,
    base,
    prompt,
    folded,
    host_rows: Optional[HostRows],
    n_verify: int,
    verify_reqs: Sequence,
    verify_tokens: Optional[torch.Tensor],
    num_verify_rows: int,
    ext_cap: int,
    pad_bv: int,
    out_cache_loc_dtype: torch.dtype,
    spec_info,
) -> MixedLayout:
    """The merged MIXED batch of a rank, on the host.

    ``prompt``: the rank's unfolded prompt batch or None; ``folded``: its
    encoder-replay fold (FoldedExtend) or None. ``base``: the batch whose
    pools and sampling info a prompt-less rank uses. ``host_rows``: the verify
    bucket's host rows; the first ``n_verify`` requests are real. A rank with
    neither part holds one dummy verify request (the idle view's row 0), masked
    by the non-padded row count.
    """
    R = num_verify_rows
    if prompt is None:
        prompt_view = empty_prompt_view(base)
        prompt_ids = prompt_view.input_ids
    else:
        prompt_view = copy(folded.batch if folded is not None else prompt)
        prompt_ids = host_prompt_ids(prompt_view)
        # Host per-request fields: the policy pads and uploads them at the
        # bucket (no ragged device ops, no device reads).
        prompt_view.seq_lens = prompt_view.seq_lens_cpu
        prompt_view.req_pool_indices = prompt_view.req_pool_indices_cpu
        prompt_view.engram_history = None
    if prompt_view.out_cache_loc.device.type != "cpu":
        raise NotImplementedError(
            "A fused mixed step needs the prompt's cache locations on the host."
        )
    dummy = prompt is None and n_verify == 0
    n_seg = 1 if dummy else n_verify
    segment = None
    if n_seg:
        assert host_rows is not None
        nv_rows = n_seg * R
        positions = host_rows.positions[:nv_rows].to(torch.int64).clone()
        firsts = positions.view(n_seg, R)[:, 0]
        seq_lens = positions.view(n_seg, R)[:, -1] + 1
        slots = host_rows.req_pool_indices[:n_seg].to(torch.int64).clone()
        segment = VerifySegment(
            reqs=[_DummyVerifyReq()] if dummy else list(verify_reqs)[:n_verify],
            req_pool_indices=slots,
            req_pool_indices_cpu=slots,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            committed_lens_cpu=firsts.tolist(),
            # Placeholder rows; the device verify tokens are spliced in after
            # the upload (MixedVerifyInput.record_padding).
            input_ids=torch.zeros((nv_rows,), dtype=torch.int64),
            positions_cpu=positions,
            out_cache_loc=host_rows.out_cache_loc[:nv_rows].to(torch.int64).clone(),
            num_verify_rows=R,
        )
    merged = merge_mixed_batch(
        prompt_view,
        segment,
        prompt_input_ids=prompt_ids.to(torch.int64),
        spec_info=spec_info,
        mixed_mode=_mixed_mode(),
    )
    merged.out_cache_loc = merged.out_cache_loc.to(out_cache_loc_dtype)
    for name in MIXED_INPUT_FIELDS:
        setattr(merged, name, fresh(getattr(merged, name)))
    merged.input_tensor_device = "cpu"
    merged.engram_history = None
    merged.has_grammar = False
    prompt_rows = int(prompt_view.extend_num_tokens) if prompt is not None else 0
    verify_rows = n_seg * R
    positions = mixed_positions_cpu(prompt_view, segment)
    assert positions.shape[0] == prompt_rows + verify_rows
    maps = extend_folded_row_maps(folded, verify_rows) if folded is not None else None
    index = mixed_logits_index(
        list(prompt_view.extend_lens) if prompt is not None else [],
        n_seg,
        ext_cap=ext_cap,
        pad_bv=pad_bv,
        num_verify_rows=R,
    )
    host = MixedHostRows(
        prompt_reqs=list(prompt.reqs) if prompt is not None else [],
        prompt_starts=[int(p) for p in prompt_view.prefix_lens]
        if prompt is not None
        else [],
        prompt_lengths=[int(n) for n in prompt_view.extend_lens]
        if prompt is not None
        else [],
        prompt_slots=prompt_view.req_pool_indices_cpu.tolist()
        if prompt is not None
        else [],
        prompt_folded=folded is not None,
        verify_tokens=verify_tokens if n_seg else None,
        verify_positions=segment.positions_cpu
        if segment is not None
        else torch.zeros((0,), dtype=torch.int64),
        verify_slots=segment.req_pool_indices_cpu
        if segment is not None
        else torch.zeros((0,), dtype=torch.int64),
        num_verify_rows=R,
    )
    # Host Engram addresses rows as [prompt rows | verify rows]: its prompt part
    # must be exactly the forward's prompt segment (no padding before verify).
    assert host.prompt_rows == prompt_rows, (host.prompt_rows, prompt_rows)
    return MixedLayout(
        batch=merged,
        positions=fresh(positions),
        logits_index=fresh(index),
        row_floor=None if maps is None else fresh(maps.row_floor),
        compress_rows=None if maps is None else fresh(maps.compress_rows),
        prompt_rows=prompt_rows,
        verify_rows=verify_rows,
        real_rows=prompt_rows + n_verify * R,
        host_rows=host,
    )


def _mixed_mode():
    from sglang.srt.model_executor.forward_mode import ForwardMode

    return ForwardMode.MIXED


def verify_hidden_index(prompt_rows: int, pbs: int, R: int, rows: int) -> torch.Tensor:
    """[pbs * R] int64 rows of the MIXED hidden holding the verify bucket's rows
    (padding requests clamp into the forward; the accept masks them)."""
    index = prompt_rows + torch.arange(pbs * R, dtype=torch.int64)
    return index.clamp_(max=max(rows - 1, 0))


def prompt_hidden_index(folded, n_prompt_rows: int, inject_rows: int, rows: int):
    """[inject_rows] int64 rows of the MIXED hidden the draft KV inject reads:
    the scheduled prompt rows (folded replay rows dropped, as drop_folded_rows),
    then the last row of the forward, as drop_tpu_dsv41_folded_rows' tail
    (padding draft rows write slot 0)."""
    tail = min(inject_rows, rows) - 1
    index = torch.full((inject_rows,), max(tail, 0), dtype=torch.int64)
    keep = (
        folded.keep_rows.to("cpu", torch.int64)
        if folded is not None
        else torch.arange(n_prompt_rows, dtype=torch.int64)
    )
    index[: keep.shape[0]] = keep
    return index


def splice_index(num_tokens: int, start: int, count: int) -> torch.Tensor:
    """[num_tokens] int64 gather index over cat([ids, verify_tokens]): row j
    reads ids[j], except the verify rows [start, start + count), which read
    verify token j - start."""
    index = torch.arange(num_tokens, dtype=torch.int64)
    index[start : start + count] = num_tokens + torch.arange(count, dtype=torch.int64)
    return index


def splice_verify_tokens(
    input_ids: torch.Tensor, verify_tokens: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """The MIXED input ids: the uploaded prompt ids with the device verify
    tokens in their rows; fixed shapes (token bucket, verify bucket) and a
    fresh tensor."""
    return torch.cat([input_ids, verify_tokens.to(input_ids.dtype)])[index]


def _dflash_verify_input_cls():
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    return DFlashVerifyInput


def _make_mixed_verify_input_cls():
    from dataclasses import dataclass

    base = _dflash_verify_input_cls()

    @dataclass
    class MixedVerifyInput(base):
        """spec_info of a fused MIXED forward.

        ``logits_select_index``: the [L] device logits row index, also on
        ForwardBatch.logit_rows (a field, so the direct tracer lifts it; the
        target ignores spec_info). ``verify_tokens``: the device verify tokens
        spliced into the uploaded input ids at rows [verify_row_start,
        verify_row_start + num_verify_tokens) once the policy has padded and
        uploaded them (record_padding runs right after). The MLP-sync counts
        are already token counts: the per-request widths are 1.
        """

        logits_select_index: Optional[torch.Tensor] = None
        verify_tokens: Optional[torch.Tensor] = None
        verify_row_start: int = 0
        num_verify_tokens: int = 0

        def __post_init__(self):
            super().__post_init__()
            self.num_tokens_per_req = 1
            self.num_tokens_for_logprob_per_req = 1
            self._batch_ref = None
            self._device = None
            self._finished = False

        @property
        def finished(self) -> bool:
            return self._finished

        def bind(self, forward_batch, device) -> None:
            # Dropped once used, so the batch and its spec input do not keep
            # each other alive.
            self._batch_ref = forward_batch
            self._device = device

        def record_padding(self, **kwargs):
            state = super().record_padding(**kwargs)
            forward_batch, self._batch_ref = self._batch_ref, None
            if forward_batch is not None and not self._finished:
                self._finished = True
                finish_mixed_inputs(forward_batch, self, self._device)
            return state

    return MixedVerifyInput


_MIXED_VERIFY_INPUT_CLS = None


def mixed_verify_input_cls():
    global _MIXED_VERIFY_INPUT_CLS
    if _MIXED_VERIFY_INPUT_CLS is None:
        _MIXED_VERIFY_INPUT_CLS = _make_mixed_verify_input_cls()
    return _MIXED_VERIFY_INPUT_CLS


def make_mixed_verify_input(
    *,
    positions: torch.Tensor,
    num_verify_rows: int,
    logits_select_index: Optional[torch.Tensor] = None,
    verify_tokens: Optional[torch.Tensor] = None,
    verify_row_start: int = 0,
    num_verify_tokens: int = 0,
):
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    return mixed_verify_input_cls()(
        draft_token=torch.zeros((0,), dtype=torch.int64),
        positions=positions,
        draft_token_num=num_verify_rows,
        custom_mask=None,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        logits_select_index=logits_select_index,
        verify_tokens=verify_tokens,
        verify_row_start=verify_row_start,
        num_verify_tokens=num_verify_tokens,
    )


# Fields the policy uploads at the bucket (host-padded batches); a batch whose
# sampling info is not host-padded leaves them on the host, uploaded here.
_UPLOADED_FIELDS = ("positions", "out_cache_loc", "req_pool_indices", "seq_lens")


def finish_mixed_inputs(forward_batch, spec_info, device) -> None:
    """After the policy padded the MIXED batch to its bucket: every host field
    on the device (recorded for host-built metadata), then the verify tokens
    spliced into the input ids."""
    from sglang_tpu.compilation import host_input_mirror

    if device is None:
        return
    for name in _UPLOADED_FIELDS:
        value = getattr(forward_batch, name, None)
        if isinstance(value, torch.Tensor) and value.device.type == "cpu":
            uploaded = value.to(device, non_blocking=True)
            setattr(forward_batch, name, uploaded)
            host_input_mirror.record_upload(forward_batch, name, value, uploaded)
    ids = forward_batch.input_ids
    if ids.device.type == "cpu":
        ids = ids.to(device, non_blocking=True)
    count = int(spec_info.num_verify_tokens)
    if count:
        index = splice_index(ids.shape[0], int(spec_info.verify_row_start), count)
        ids = splice_verify_tokens(
            ids, spec_info.verify_tokens, index.to(device, non_blocking=True)
        )
    forward_batch.input_ids = ids


def mixed_sync_fields(batch) -> dict:
    """The DP sync fields of a mixed step's container batch, for its MIXED forward."""
    return {
        name: getattr(batch, name) for name in _MLP_SYNC_FIELDS if hasattr(batch, name)
    }


# The draft KV append kernel takes row buckets in multiples of this.
DRAFT_KV_ROW_MULTIPLE = 256


def inject_rows(plan, t_bucketer, ext_cap: int) -> int:
    """Rows of the fused step's prompt draft KV inject, the same on every rank:
    the token bucket of the busiest rank's prompt rows (with replay rows), as
    its plain extend would pad to, but never a fine bucket (1152): the draft
    KV append kernel takes DRAFT_KV_ROW_MULTIPLE-row buckets only."""
    return inject_bucket(
        t_bucketer, max(max(plan.global_prompt_rows, default=0), ext_cap, 1)
    )


def inject_bucket(t_bucketer, rows: int) -> int:
    bucket = t_bucketer.next_bucket(rows)
    while bucket % DRAFT_KV_ROW_MULTIPLE and bucket < t_bucketer.max_val:
        bucket = t_bucketer.next_bucket(bucket + 1)
    return bucket


def verify_buckets(b_bucketer, attn_tp_size: int) -> list:
    """Every verify bucket a mixed step can pick (mixed_verify.verify_bucket)."""
    from sglang.srt.speculative.mixed_verify import ceil_align

    return sorted(
        {
            b_bucketer.next_bucket(ceil_align(n, attn_tp_size))
            for n in range(1, b_bucketer.max_val + 1)
            if ceil_align(n, attn_tp_size) <= b_bucketer.max_val
        }
    )


# Under decoder replay a fused MIXED forward runs at this token bucket at least:
# each token bucket multiplies the MIXED programs by the decoder buckets, and the
# smaller one served few steps (536 of 6304 at 512 in the c32 trace, j-73704d80).
MIXED_MIN_TOKEN_BUCKET_DECODER_REPLAY = 1024


def mixed_min_token_rows() -> int:
    """The fewest token rows a fused MIXED forward pads to (0: no floor)."""
    from sglang.srt.runtime_context import get_exec

    if fused_mixed_forward() and get_exec().features.enable_decoder_swa_bounded_replay:
        return MIXED_MIN_TOKEN_BUCKET_DECODER_REPLAY
    return 0


def fused_compile_cases(t_bucketer, b_bucketer, attn_tp_size: int) -> list:
    """(token bucket, ext_cap, verify bucket) of every fused MIXED forward: the
    batch is the prompt cap plus the verify bucket, and the token bucket holds
    at least that many rows (and mixed_min_token_rows)."""
    from sglang_tpu.compilation.compiled_forward_shim import _extend_batch_cap

    ext_cap = _extend_batch_cap(t_bucketer, b_bucketer)
    floor = min(mixed_min_token_rows(), t_bucketer.max_val)
    return [
        (num_tokens, ext_cap, pbs)
        for pbs in verify_buckets(b_bucketer, attn_tp_size)
        for num_tokens in t_bucketer.buckets
        if num_tokens >= ext_cap + pbs and num_tokens >= floor
    ]


class MixedPromptPart:
    """The prompt part of a fused mixed step, stashed on the worker while its
    verify half runs, and the prompt outputs of the one MIXED forward."""

    def __init__(self, *, plan, prompt, sync_fields: dict):
        self.plan = plan
        self.prompt = prompt
        self.sync_fields = sync_fields
        self.ran = False
        # Filled by the MIXED forward.
        self.folded = None
        self.prompt_logits: Optional[torch.Tensor] = None  # [ext_cap, V] fresh
        self.prompt_hidden: Optional[torch.Tensor] = None  # [inject_rows, H]


# Non-field ForwardBatch attributes the spec worker sets before the forward and
# the policy / model read after EagerRunner.load_batch copies the batch: the plan
# (the shim's batch rows) and host Engram's row map. Every such attribute the
# DSpark worker path sets must be listed here (test_mixed_fused_worker scans for
# new ones).
_MIXED_ATTRS = ("mixed_verify", "mixed_host_rows")


def carry_mixed_fields(result, runner, forward_batch, *args, **kwargs):
    """AFTER EagerRunner.load_batch: the runner hands the model a new ForwardBatch
    (dataclasses.replace or its static buffers) that keeps dataclass fields only;
    carry a fused MIXED batch's attributes over. Without the plan the shim pads
    the forward as an extend, and without the host rows host Engram would hash
    the verify rows as an extend (committing draft tokens)."""
    # Fused MIXED batches only: every other forward keeps the runner's behavior.
    if getattr(forward_batch, "mixed_verify", None) is None:
        return result
    if result is not None and result is not forward_batch:
        for name in _MIXED_ATTRS:
            value = getattr(forward_batch, name, None)
            if value is not None:
                setattr(result, name, value)
        # The fused spec input splices the device verify tokens into the input ids
        # of the batch it is bound to, at the policy's record_padding: bind it to
        # the copy the policy pads and the model runs (bound to the original, the
        # copy kept placeholder ids in the verify rows: check run j-e8de0bbc,
        # every verify row wrong).
        spec_info = getattr(result, "spec_info", None)
        if getattr(spec_info, "_batch_ref", None) is forward_batch:
            spec_info._batch_ref = result
        # Host copies recorded for the original (verify rows, uploads) serve the
        # copy: without this every host_copy misses and the metadata reads the
        # device back.
        from sglang_tpu.compilation import host_input_mirror

        host_input_mirror.rebind(forward_batch, result)
    return result


def execute_fused_mixed(original, runner, forward_batch, *args, **kwargs):
    """AROUND EagerRunner.execute: the eager runner runs a MIXED batch as EXTEND
    (decode tails as 1-token extends); a fused MIXED forward keeps MIXED, the mode
    its graph is warmed and compiled in and the one the logits processor selects
    its fixed-size row set for."""
    if (
        forward_batch.forward_mode.is_mixed()
        and getattr(forward_batch, "mixed_verify", None) is not None
        and fused_mixed_forward()
    ):
        pp_proxy_tensors = args[0] if args else kwargs.get("pp_proxy_tensors")
        with runner.model_context():
            return runner._execute_extend(forward_batch, pp_proxy_tensors)
    return original(runner, forward_batch, *args, **kwargs)


def fine_bucket_extra_rows() -> int:
    """Rows a fused mixed step may add over the prompt chunk on the fine token
    bucket: FINE_BUCKET_ROWS with fused steps and SGLANG_TPU_MIXED_VERIFY_FINE_BUCKET
    on, else 0. Sizes the SWA request-window workspace (request_window)."""
    from sglang_tpu.envs import envs as tpu_envs

    if fused_mixed_forward() and tpu_envs.SGLANG_TPU_MIXED_VERIFY_FINE_BUCKET.get():
        return FINE_BUCKET_ROWS
    return 0
