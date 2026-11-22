# SPDX-License-Identifier: Apache-2.0
"""Scheduler side of decoder SWA bounded replay on TPU (decoder_replay.py).

The backend needs, per extend forward, each request's full fill length (the
``prompt`` row rule) and every DP rank's decoder row count (the shared decoder
row bucket). Core carries neither, so these hooks attach them as plain
attributes:

- ``ScheduleBatch.decoder_fill_lens``: set by ``prepare_for_extend``, cleared by
  ``convert_decode_to_extend`` (its rows keep each extend's end), carried by
  ``copy``; ``ForwardBatch.init_new`` copies it for extend batches, and
  ``EagerRunner.load_batch`` (which rebuilds the ForwardBatch from its
  dataclass fields) carries the ForwardBatch attributes to the batch the model
  runs.
- ``global_decoder_rows``: one extra MLP-sync column (this rank's
  ``scheduled_decoder_rows``) while ``prepare_mlp_sync_batch_raw`` runs with a
  decoder-replay backend; the gathered column lands on the synced batch and its
  ForwardBatch. Without a sync (or a skipped gather) it stays None and the
  compile policy falls back to the token bucket under DP.
- ``deepseek_v4._scatter_tail_rows``: a DecoderTail's full layout starts zeroed,
  so rows the decoder skipped (and the logits processor may still read) are
  defined.
"""

from __future__ import annotations

import threading
from typing import Optional

from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# ForwardBatch attributes these hooks set (as engram_host declares its own):
# defaults for forwards built without them. ScheduleBatch readers use getattr.
ForwardBatch.decoder_fill_lens = None
ForwardBatch.global_decoder_rows = None
# The shared decoder row bucket the compile policy picks for a forward.
ForwardBatch.decoder_rows_bucket = None

_SYNC = threading.local()


def _pending_rows() -> Optional[int]:
    return getattr(_SYNC, "rows", None)


def _local_decoder_rows(model_runner, batch) -> Optional[int]:
    backend = getattr(model_runner, "attn_backend", None)
    if not getattr(backend, "decoder_replay", False):
        return None
    if batch is not None and batch.forward_mode.is_prebuilt():
        batch = None
    folds = (
        batch is not None
        and getattr(batch, "encoder_swa_reset", None) is not None
        and getattr(backend, "folds_encoder_swa_replay", False)
    )
    return int(backend.scheduled_decoder_rows(batch, folds=folds))


def sync_decoder_rows(original, local_batch, model_runner, *args, **kwargs):
    """AROUND dp_attn.prepare_mlp_sync_batch_raw: send this rank's decoder rows
    as one more MLP-sync column for the duration of the call."""
    _SYNC.rows = _local_decoder_rows(model_runner, local_batch)
    try:
        return original(local_batch, model_runner, *args, **kwargs)
    finally:
        _SYNC.rows = None


def _with_column(tensor, value: Optional[int]):
    if value is None:
        return tensor
    import torch

    return torch.cat([tensor, tensor.new_tensor([value])])


def local_tensor_with_decoder_rows(result, *args, **kwargs):
    """AFTER MLPSyncBatchInfo._get_local_tensor."""
    return _with_column(result, _pending_rows())


def fallback_tensor_with_decoder_rows(result, *args, **kwargs):
    """AFTER MLPSyncBatchInfo._get_fallback_tensor (an idle slot: 0 rows)."""
    return _with_column(result, None if _pending_rows() is None else 0)


def gathered_decoder_rows(result, info, *args, **kwargs):
    """AFTER MLPSyncBatchInfo.all_gather."""
    info.global_decoder_rows = (
        None if _pending_rows() is None else info.tp0_info_cpu[:, -1].tolist()
    )
    return result


def local_decoder_rows(result, info, *args, **kwargs):
    """AFTER MLPSyncBatchInfo.finalize_local."""
    rows = _pending_rows()
    info.global_decoder_rows = None if rows is None else [rows]
    return result


def batch_decoder_rows(result, batch, mlp_sync_info, *args, **kwargs):
    """AFTER dp_attn._update_gather_batch."""
    batch.global_decoder_rows = getattr(mlp_sync_info, "global_decoder_rows", None)
    return result


def record_fill_lens(result, batch, *args, **kwargs):
    """AFTER ScheduleBatch.prepare_for_extend."""
    from sglang.srt.runtime_context import get_exec

    if get_exec().features.enable_decoder_swa_bounded_replay:
        batch.decoder_fill_lens = [len(r.full_untruncated_fill_ids) for r in batch.reqs]
    return result


def drop_fill_lens(result, batch, *args, **kwargs):
    """AFTER ScheduleBatch.convert_decode_to_extend."""
    batch.decoder_fill_lens = None
    return result


def copy_decoder_fields(result, batch, *args, **kwargs):
    """AFTER ScheduleBatch.copy."""
    result.decoder_fill_lens = getattr(batch, "decoder_fill_lens", None)
    result.global_decoder_rows = getattr(batch, "global_decoder_rows", None)
    return result


def forward_batch_decoder_fields(result, *args, **kwargs):
    """AFTER ForwardBatch.init_new (a classmethod: ``args`` start with the class)."""
    batch = kwargs.get("batch")
    if batch is None:
        batch = args[1] if args and isinstance(args[0], type) else args[0]
    # Extend batches only: a decode batch DP turns into an extend keeps each
    # row (its extend end stands in for the fill length).
    result.decoder_fill_lens = (
        getattr(batch, "decoder_fill_lens", None)
        if batch.forward_mode.is_extend()
        else None
    )
    result.global_decoder_rows = getattr(batch, "global_decoder_rows", None)
    return result


_FORWARD_BATCH_FIELDS = (
    "decoder_fill_lens",
    "global_decoder_rows",
    "decoder_rows_bucket",
)


def carry_decoder_fields(result, runner, forward_batch, *args, **kwargs):
    """AFTER EagerRunner.load_batch: the runner hands the model a new
    ForwardBatch (``dataclasses.replace`` or its static buffers), which keeps
    dataclass fields only; carry these attributes over to it."""
    if result is not None and result is not forward_batch:
        for name in _FORWARD_BATCH_FIELDS:
            value = getattr(forward_batch, name, None)
            if value is not None:
                setattr(result, name, value)
    return result


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType
    from sglang_tpu.models.deepseek_v41.decoder_replay import (
        clear_full_layout_head,
        keep_backend_late_layer_dp,
        lockstep_late_layer_dp_counts,
        scatter_decoder_rows,
        scatter_dspark_capture,
        skip_empty_decoder,
        start_tail_after_last_source,
    )

    sync = "sglang.srt.managers.scheduler_components.dp_attn."
    info = sync + "MLPSyncBatchInfo."
    schedule = "sglang.srt.managers.schedule_batch.ScheduleBatch."
    return [
        (sync + "prepare_mlp_sync_batch_raw", sync_decoder_rows, HookType.AROUND),
        (info + "_get_local_tensor", local_tensor_with_decoder_rows, HookType.AFTER),
        (
            info + "_get_fallback_tensor",
            fallback_tensor_with_decoder_rows,
            HookType.AFTER,
        ),
        (info + "all_gather", gathered_decoder_rows, HookType.AFTER),
        (info + "finalize_local", local_decoder_rows, HookType.AFTER),
        (sync + "_update_gather_batch", batch_decoder_rows, HookType.AFTER),
        (schedule + "prepare_for_extend", record_fill_lens, HookType.AFTER),
        (schedule + "convert_decode_to_extend", drop_fill_lens, HookType.AFTER),
        (schedule + "copy", copy_decoder_fields, HookType.AFTER),
        (
            "sglang.srt.model_executor.forward_batch_info.ForwardBatch.init_new",
            forward_batch_decoder_fields,
            HookType.AFTER,
        ),
        (
            "sglang.srt.model_executor.runner.eager_runner.EagerRunner.load_batch",
            carry_decoder_fields,
            HookType.AFTER,
        ),
        (
            "sglang.srt.models.deepseek_v4.DeepseekV4Model.__init__",
            start_tail_after_last_source,
            HookType.AFTER,
        ),
        (
            "sglang.srt.models.deepseek_common.decoder_swa_bounded_replay"
            ".late_layer_dp_counts",
            lockstep_late_layer_dp_counts,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_common.decoder_swa_bounded_replay"
            ".enter_late_layer_dp",
            keep_backend_late_layer_dp,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_v4.DeepseekV4Model"
            "._forward_layers_hc_pre_from_prev",
            skip_empty_decoder,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_v4.DeepseekV4Model.forward",
            scatter_dspark_capture,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.forward",
            clear_full_layout_head,
            HookType.AROUND,
        ),
        (
            "sglang.srt.models.deepseek_v4._scatter_tail_rows",
            scatter_decoder_rows,
            HookType.AROUND,
        ),
    ]
