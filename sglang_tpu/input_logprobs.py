# SPDX-License-Identifier: Apache-2.0
"""Bound prompt-logprob compilation by physical row and output-width buckets."""

from __future__ import annotations

import bisect
import functools

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.layers.logprob_processor import (
    InputLogprobProcessor,
    LogprobResult,
    _append_top_logprobs_chunk,
    compute_row_log_normalizer,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.utils import LazyImport
from sglang_tpu.compilation.compiled_forward_shim import _pad_tensor
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.output_logprobs import _compiled_sort, _sort_logprobs_impl
from sglang_tpu.row_pad import row_bucket

direct_compile = LazyImport("sglang_tpu.compilation.direct_compile", "direct_compile")
_SORT_ROWS = 128


def _gather_rows(logprobs, indices):
    return logprobs.index_select(0, indices)


def _select_columns(values, indices, width):
    return values[:, :width].contiguous(), indices[:, :width].contiguous()


@functools.cache
def _compiled_gather():
    return direct_compile(
        _gather_rows,
        specialization_key=lambda values, indices: (
            tuple(values.shape),
            tuple(indices.shape),
            values.dtype,
            values.device,
        ),
    )


def _token_logits(logits, token_ids):
    return logits[torch.arange(logits.shape[0], device=logits.device), token_ids]


def _sample_logits(logits, indices, previous, mask):
    selected = logits.index_select(0, indices)
    return (
        selected if previous is None else torch.where(mask[:, None], selected, previous)
    )


def _gather_pairs(logits, rows, columns):
    return logits[rows[:, None], columns]


@functools.cache
def _compiled_token_logits():
    return direct_compile(
        _token_logits,
        specialization_key=lambda logits, ids: (
            tuple(logits.shape),
            logits.dtype,
            logits.device,
        ),
    )


@functools.cache
def _compiled_sample_logits():
    return direct_compile(
        _sample_logits,
        specialization_key=lambda logits, indices, previous, mask: (
            tuple(logits.shape),
            tuple(indices.shape),
            logits.dtype,
            logits.device,
            previous is None,
        ),
    )


@functools.cache
def _compiled_gather_pairs():
    return direct_compile(
        _gather_pairs,
        specialization_key=lambda logits, rows, columns: (
            tuple(logits.shape),
            logits.dtype,
            logits.device,
        ),
    )


def _device_indices(indices, size, device):
    return torch.tensor(
        [*indices, *([0] * (size - len(indices)))],
        dtype=torch.int64,
        device="cpu",
    ).to(device)


def _gather(values, indices):
    gather = _compiled_gather() if values.device.type == "tpu" else _gather_rows
    return gather(values, indices)


@functools.cache
def _compiled_select():
    return direct_compile(
        _select_columns,
        specialization_key=lambda values, indices, width: (
            tuple(values.shape),
            values.dtype,
            values.device,
            width,
        ),
    )


def _copy_top_logprobs(logprobs, max_k, log_normalizer, precomputed_topk):
    rows, vocab_size = logprobs.shape
    if not 0 <= max_k <= vocab_size:
        raise RuntimeError("selected index k out of range")
    if not rows or not max_k:
        return [[] for _ in range(rows)], [[] for _ in range(rows)]
    if precomputed_topk is not None and log_normalizer is not None:
        values, indices = (tensor.cpu() for tensor in precomputed_topk)
        row_max, row_log_sum = (tensor.cpu() for tensor in log_normalizer)
        values = (values.float() - row_max[:, None]) - row_log_sum[:, None]
        return values.tolist(), indices.tolist()

    width = min(1 << (max_k - 1).bit_length(), vocab_size)
    logprobs = _pad_tensor(logprobs, 1 << (rows - 1).bit_length())
    native = logprobs.device.type == "tpu"
    gather = _compiled_gather() if native else _gather_rows
    sort = _compiled_sort() if native else _sort_logprobs_impl
    select = _compiled_select() if native else _select_columns
    chunks = []
    for start in range(0, rows, _SORT_ROWS):
        count = min(_SORT_ROWS, rows - start)
        # Offset is tensor data, so later chunks reuse the first chunk's gather.
        row_indices = torch.arange(
            start, start + _SORT_ROWS, dtype=torch.int64, device="cpu"
        ).clamp_max(rows - 1)
        chunk = gather(logprobs, row_indices.to(logprobs.device))
        values, indices = sort(chunk)
        values, indices = select(values, indices, width)
        chunks.append((start, count, values, indices))

    normalizer = (
        tuple(tensor.cpu() for tensor in log_normalizer)
        if log_normalizer is not None
        else None
    )
    all_values, all_indices = [], []
    for start, count, values, indices in chunks:
        values = values.cpu()[:count, :max_k]
        indices = indices.cpu()[:count, :max_k]
        if normalizer is not None:
            row_max, row_log_sum = normalizer
            values = (
                values.float() - row_max[start : start + count, None]
            ) - row_log_sum[start : start + count, None]
        all_values.extend(values.tolist())
        all_indices.extend(indices.tolist())
    return all_values, all_indices


def get_top_logprobs_chunk(
    logprobs,
    top_k_nums,
    pruned_lens,
    top_logprobs_val,
    top_logprobs_idx,
    split_pruned_len,
    log_normalizer=None,
    precomputed_topk=None,
    copy_to_pinned_cpu=False,
):
    values, indices = _copy_top_logprobs(
        logprobs, max(top_k_nums), log_normalizer, precomputed_topk
    )
    if copy_to_pinned_cpu:
        values = torch.tensor(
            values, dtype=torch.float32, device="cpu", pin_memory=True
        )
        indices = torch.tensor(
            indices, dtype=torch.int64, device="cpu", pin_memory=True
        )
    return _append_top_logprobs_chunk(
        values,
        indices,
        len(values),
        top_k_nums,
        pruned_lens,
        top_logprobs_val,
        top_logprobs_idx,
        split_pruned_len,
        copy_to_pinned_cpu,
    )


class TpuInputLogprobProcessor(InputLogprobProcessor):
    def __init__(
        self, vocab_size, chunking_group=None, *, skip_chunking=False, model=None
    ):
        super().__init__(
            vocab_size,
            chunking_group,
            get_top_logprobs_chunk_fn=get_top_logprobs_chunk,
        )
        self.skip_chunking = skip_chunking
        self.model = model
        self._compiled_logits = None

    def prune_states(self, hidden_states, ranges):
        if self.skip_chunking:
            return super().prune_states(hidden_states, ranges)
        # Keep the physical row bucket through the LM head. Logical row counts
        # remain in token_to_seq_idx and never specialize a device slice.
        indices = [index for start, end in ranges for index in range(start, end)]
        hidden_states = _pad_tensor(hidden_states, row_bucket(hidden_states.shape[0]))
        return _gather(
            hidden_states,
            _device_indices(indices, hidden_states.shape[0], hidden_states.device),
        )

    def _logits_for_rows(self, hidden_states):
        return self.model.logits_processor._get_logits(
            hidden_states,
            self.model.lm_head,
            LogitsMetadata(forward_mode=ForwardMode.EXTEND, extend_return_logprob=True),
            use_logits_buffer=False,
        )

    def forward(
        self,
        pruned_states,
        sample_indices,
        input_logprob_indices,
        token_to_seq_idx,
        lm_head,
        get_logits_fn,
        logits_metadata,
        skip_chunking_for_dp_attn=False,
    ):
        if skip_chunking_for_dp_attn:
            return super().forward(
                pruned_states,
                sample_indices,
                input_logprob_indices,
                token_to_seq_idx,
                lm_head,
                get_logits_fn,
                logits_metadata,
                skip_chunking_for_dp_attn=skip_chunking_for_dp_attn,
            )
        total = len(token_to_seq_idx)
        chunk_size = (
            min(total, self.logprobs_chunk_size)
            if self.enable_logprobs_chunk and not self.skip_chunking
            else total
        )
        native = pruned_states.device.type == "tpu"
        select = _compiled_token_logits() if native else _token_logits
        sample = _compiled_sample_logits() if native else _sample_logits
        sample_rows = logits_metadata.sample_indices_cpu
        input_rows = logits_metadata.input_logprob_indices_cpu
        input_ids = logits_metadata.extend_input_logprob_token_ids_cpu.tolist()
        device = pruned_states.device
        sampled_logits = None
        token_chunks = []
        top_values, top_indices = (
            ([], []) if logits_metadata.extend_return_top_logprob else (None, None)
        )
        named_values, named_indices = (
            ([], []) if logits_metadata.extend_token_ids_logprob else (None, None)
        )
        split_top = split_named = 0
        pruned_states = _pad_tensor(pruned_states, row_bucket(pruned_states.shape[0]))
        for start in range(0, total, max(chunk_size, 1)):
            end = min(start + chunk_size, total)
            rows = row_bucket(end - start)
            states = _gather(
                pruned_states,
                _device_indices(list(range(start, end)), rows, device),
            )
            logits = (
                self._compiled_logits(states)
                if native and self._compiled_logits is not None
                else get_logits_fn(
                    states, lm_head, logits_metadata, use_logits_buffer=False
                )
            )
            sample_offsets = [
                index - start if start <= index < end else 0 for index in sample_rows
            ]
            sample_mask = torch.tensor(
                [start <= index < end for index in sample_rows],
                dtype=torch.bool,
                device="cpu",
            ).to(device)
            sampled_logits = sample(
                logits,
                _device_indices(sample_offsets, len(sample_rows), device),
                sampled_logits,
                sample_mask,
            )
            lo = bisect.bisect_left(input_rows, start)
            hi = bisect.bisect_left(input_rows, end)
            logical_rows = hi - lo
            local_input_rows = [index - start for index in input_rows[lo:hi]]
            selected_ids = [0] * rows
            for row, token_id in zip(local_input_rows, input_ids[lo:hi]):
                selected_ids[row] = token_id
            # Normalize the original rows and reorder only small host results.
            # Gathering full-vocabulary rows here doubles the projection memory.
            # Keep eager reduction boundaries to preserve fp32 rounding.
            if self.enable_fast_input_logprobs:
                values = logits
                normalizer = compute_row_log_normalizer(values)
            else:
                values = torch.nn.functional.log_softmax(logits, dim=-1)
                normalizer = None
            del logits
            selected = select(values, _device_indices(selected_ids, rows, device))
            chunk_slice = slice(token_to_seq_idx[start], token_to_seq_idx[end - 1] + 1)
            lengths = logits_metadata.extend_logprob_pruned_lens_cpu[chunk_slice]
            if top_values is not None:
                widths = logits_metadata.top_logprobs_nums[chunk_slice]
                host_values, host_indices = _copy_top_logprobs(
                    values, max(widths), normalizer, None
                )
                split_top = _append_top_logprobs_chunk(
                    [host_values[row] for row in local_input_rows],
                    [host_indices[row] for row in local_input_rows],
                    logical_rows,
                    widths,
                    lengths,
                    top_values,
                    top_indices,
                    split_top,
                    False,
                )
            if named_values is not None:
                named = logits_metadata.token_ids_logprobs
                per_row = [
                    named[token_to_seq_idx[index]] or [] for index in input_rows[lo:hi]
                ]
                host_values, host_indices = self._copy_named_logprobs(
                    values, per_row, local_input_rows, normalizer
                )
                split_named = _append_top_logprobs_chunk(
                    host_values,
                    host_indices,
                    logical_rows,
                    [len(ids or []) for ids in named[chunk_slice]],
                    lengths,
                    named_values,
                    named_indices,
                    split_named,
                    False,
                )
            chosen = selected.cpu()
            if normalizer is not None:
                row_max, row_sum = (tensor.cpu() for tensor in normalizer)
                chosen = (chosen.float() - row_max) - row_sum
            token_chunks.append(chosen[local_input_rows])
            # The next LM-head projection needs its own full-vocabulary output.
            # Release this chunk after its small host results have completed.
            del states, values, selected, normalizer
        return (
            LogprobResult(
                token_logprobs=torch.cat(token_chunks),
                top_logprobs_val=top_values,
                top_logprobs_idx=top_indices,
                token_ids_logprobs_val=named_values,
                token_ids_logprobs_idx=named_indices,
            ),
            sampled_logits,
        )

    @staticmethod
    def _copy_named_logprobs(values, per_row, input_rows, normalizer):
        gather = (
            _compiled_gather_pairs() if values.device.type == "tpu" else _gather_pairs
        )
        host_normalizer = (
            tuple(tensor.cpu() for tensor in normalizer)
            if normalizer is not None
            else None
        )
        result = [[] for _ in per_row]
        for start in range(0, len(per_row), _SORT_ROWS):
            requests = per_row[start : start + _SORT_ROWS]
            width = max((len(ids) for ids in requests), default=0)
            host_rows = torch.tensor(
                [
                    *input_rows[start : start + len(requests)],
                    *([0] * (_SORT_ROWS - len(requests))),
                ],
                dtype=torch.int64,
                device="cpu",
            )
            row_ids = host_rows.to(values.device)
            for column in range(0, width, _SORT_ROWS):
                columns = torch.zeros(
                    (_SORT_ROWS, _SORT_ROWS), dtype=torch.int64, device="cpu"
                )
                for row, ids in enumerate(requests):
                    selected = ids[column : column + _SORT_ROWS]
                    columns[row, : len(selected)] = torch.tensor(
                        selected, dtype=torch.int64, device="cpu"
                    )
                selected = gather(values, row_ids, columns.to(values.device)).cpu()
                if host_normalizer is not None:
                    row_max, row_sum = host_normalizer
                    selected = (selected.float() - row_max[host_rows, None]) - row_sum[
                        host_rows, None
                    ]
                for row, ids in enumerate(requests):
                    result[start + row].extend(
                        selected[row, : len(ids[column : column + _SORT_ROWS])].tolist()
                    )
        return result, per_row

    def precompile(
        self, max_rows, device, max_batch_size=1, *, hidden_size=None, hidden_dtype=None
    ):
        """Queue shape-only examples; no full-vocabulary output is copied to CPU."""
        max_prefill_rows = row_bucket(max_rows)
        if self.enable_logprobs_chunk and not self.skip_chunking:
            max_rows = min(max_rows, self.logprobs_chunk_size)
        max_rows = 1 << (max_rows - 1).bit_length()
        async_compile = envs.SGLANG_DIRECT_COMPILE_ASYNC.get()
        pending = []
        if self.model is not None and not self.skip_chunking:
            if hidden_size is None or hidden_dtype is None:
                raise ValueError(
                    "Prompt-logprob warmup needs the model activation shape and dtype"
                )
            self._compiled_logits = direct_compile(
                self._logits_for_rows,
                specialization_key=lambda hidden_states: (
                    tuple(hidden_states.shape),
                    hidden_states.dtype,
                    hidden_states.device,
                ),
                model=self.model.lm_head,
                lift_rank=torch.distributed.is_initialized(),
            )
            source_rows = 1
            while source_rows <= max_prefill_rows:
                example = torch.empty(
                    source_rows, hidden_size, dtype=hidden_dtype, device=device
                )
                pending.append(
                    _compiled_gather().precompile(
                        example,
                        torch.empty(source_rows, dtype=torch.int64, device=device),
                        async_compile=async_compile,
                    )
                )
                if hidden_dtype != torch.float32:
                    pending.append(
                        _compiled_gather().precompile(
                            torch.empty(
                                source_rows,
                                hidden_size,
                                dtype=torch.float32,
                                device=device,
                            ),
                            torch.empty(source_rows, dtype=torch.int64, device=device),
                            async_compile=async_compile,
                        )
                    )
                rows = 1
                while rows <= min(source_rows, max_rows):
                    pending.append(
                        _compiled_gather().precompile(
                            example,
                            torch.empty(rows, dtype=torch.int64, device=device),
                            async_compile=async_compile,
                        )
                    )
                    rows *= 2
                source_rows *= 2
        indices = torch.empty(_SORT_ROWS, dtype=torch.int64, device=device)
        rows = 1
        while rows <= max_rows:
            example = torch.empty(
                rows, self.vocab_size, dtype=torch.float32, device=device
            )
            pending.append(
                _compiled_gather().precompile(
                    example, indices, async_compile=async_compile
                )
            )
            if self.skip_chunking:
                rows *= 2
                continue
            same_rows = torch.empty(rows, dtype=torch.int64, device=device)
            pending.append(
                _compiled_token_logits().precompile(
                    example,
                    same_rows,
                    async_compile=async_compile,
                )
            )
            pending.append(
                PendingCall(functools.partial(self._warmup_normalizer, rows, device))
            )
            pending.append(
                _compiled_gather_pairs().precompile(
                    example,
                    indices,
                    torch.empty(
                        (_SORT_ROWS, _SORT_ROWS), dtype=torch.int64, device=device
                    ),
                    async_compile=async_compile,
                )
            )
            for batch_size in range(1, max_batch_size + 1):
                sample_ids = torch.empty(batch_size, dtype=torch.int64, device=device)
                mask = torch.empty(batch_size, dtype=torch.bool, device=device)
                previous = torch.empty(
                    batch_size, self.vocab_size, dtype=torch.float32, device=device
                )
                for old in (None, previous):
                    pending.append(
                        _compiled_sample_logits().precompile(
                            example, sample_ids, old, mask, async_compile=async_compile
                        )
                    )
            if self._compiled_logits is not None:
                pending.append(
                    self._compiled_logits.precompile(
                        torch.empty(
                            rows, hidden_size, dtype=hidden_dtype, device=device
                        ),
                        async_compile=async_compile,
                    )
                )
            rows *= 2
        values = torch.empty(
            _SORT_ROWS, self.vocab_size, dtype=torch.float32, device=device
        )
        indices = torch.empty(
            _SORT_ROWS, self.vocab_size, dtype=torch.int64, device=device
        )
        pending.append(_compiled_sort().precompile(values, async_compile=async_compile))
        width = 1
        while width < self.vocab_size:
            pending.append(
                _compiled_select().precompile(
                    values, indices, width, async_compile=async_compile
                )
            )
            width *= 2
        pending.append(
            _compiled_select().precompile(
                values, indices, self.vocab_size, async_compile=async_compile
            )
        )
        # Exercise first-use host/runtime setup after compilation has joined.
        pending.append(
            PendingCall(
                lambda: _copy_top_logprobs(
                    torch.zeros(
                        min(max_rows, _SORT_ROWS),
                        self.vocab_size,
                        dtype=torch.float32,
                        device="cpu",
                    ).to(device),
                    1,
                    None,
                    None,
                )
            )
        )
        return pending

    def _warmup_normalizer(self, rows, device):
        values = torch.zeros(
            rows, self.vocab_size, dtype=torch.float32, device="cpu"
        ).to(device)
        if self.enable_fast_input_logprobs:
            for value in compute_row_log_normalizer(values):
                value.cpu()
        else:
            normalized = torch.nn.functional.log_softmax(values, dim=-1)
            _compiled_token_logits()(
                normalized, _device_indices([], rows, device)
            ).cpu()
