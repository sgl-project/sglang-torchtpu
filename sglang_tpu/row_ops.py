"""Merge and filter scheduler rows with bounded TPU programs."""

import torch

from sglang.srt.compilation.compiled_op import compiled_op, warmup_compiled_op
from sglang.utils import LazyImport

# The forward shim imports scheduler types that can themselves use row helpers.
sync = LazyImport("torch_tpu._internal", "sync")
_pad_tensor = LazyImport("sglang_tpu.compilation.compiled_forward_shim", "_pad_tensor")


@compiled_op()
def _merge_padded_rows(
    left: torch.Tensor, right: torch.Tensor, split: torch.Tensor
) -> torch.Tensor:
    rows = torch.arange(left.shape[0], device=left.device)
    right_rows = (rows - split).clamp_min(0)
    mask = (rows < split).reshape((-1,) + (1,) * (left.ndim - 1))
    return torch.where(mask, left, right[right_rows])


def merge_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Keep the split in data, so every split shares its output bucket's graph."""
    if left.device.type == "cpu" or not left.shape[0] or not right.shape[0]:
        return torch.cat((left, right))
    rows = left.shape[0] + right.shape[0]
    bucket = 1 << (rows - 1).bit_length()
    split = torch.tensor(left.shape[0], dtype=torch.int64, device="cpu").to(left.device)
    merged = _merge_padded_rows(
        _pad_tensor(left, bucket), _pad_tensor(right, bucket), split
    )
    return _logical_rows(merged, rows)


@compiled_op()
def _filter_padded_rows(
    values: torch.Tensor, indices: torch.Tensor, logical_rows: torch.Tensor
) -> torch.Tensor:
    indices = torch.where(indices < 0, indices + logical_rows, indices)
    return values[indices]


def filter_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Keep both gather dimensions bucketed, including compact merged batches."""
    if values.device.type == "cpu":
        return values[indices]
    rows = indices.numel()
    if not rows:
        return torch.empty((0, *values.shape[1:]), dtype=values.dtype, device="cpu").to(
            values.device
        )
    source_rows = values.shape[0]
    source_bucket = 1 << (source_rows - 1).bit_length()
    output_bucket = 1 << (rows - 1).bit_length()
    logical_rows = torch.tensor(source_rows, dtype=torch.int64, device="cpu").to(
        values.device
    )
    # History may be a logical view of a model output's larger physical bucket.
    # Materialize the same compact copy exercised by row warmup before JIT input
    # preparation, which otherwise creates a new eager view program.
    compact = _logical_rows(values, source_rows)
    selected = _filter_padded_rows(
        _pad_tensor(compact, source_bucket),
        _pad_tensor(indices, output_bucket),
        logical_rows,
    )
    return _logical_rows(selected, rows)


def _logical_rows(merged: torch.Tensor, rows: int) -> torch.Tensor:
    # Preserve cat's compact, independently owned output; compiled consumers
    # must not materialize a fresh view of the padded allocation.
    return merged[:rows].clone()


def _row_bucket_inputs(templates: list[torch.Tensor], max_rows: int):
    for template in templates:
        if template.device.type != "tpu":
            continue
        bucket = 1
        while bucket < max_rows * 2:
            shape = (bucket, *template.shape[1:])
            left = torch.zeros(shape, dtype=template.dtype, device="cpu").to(
                template.device
            )
            right = torch.ones(shape, dtype=template.dtype, device="cpu").to(
                template.device
            )
            split = torch.tensor(1, dtype=torch.int64, device="cpu").to(template.device)
            yield left, right, split
            bucket *= 2


def _filter_inputs(left: torch.Tensor):
    bucket = left.shape[0]
    logical_rows = torch.tensor(bucket, dtype=torch.int64, device="cpu").to(left.device)
    output_bucket = 1
    while output_bucket <= bucket:
        indices = torch.zeros(output_bucket, dtype=torch.int64, device="cpu").to(
            left.device
        )
        yield left, indices, logical_rows
        output_bucket *= 2


def precompile_row_ops(templates: list[torch.Tensor], max_rows: int) -> None:
    """Queue row programs while model compilation can overlap their native work."""
    for left, right, split in _row_bucket_inputs(templates, max_rows):
        if left.shape[0] > 1:
            warmup_compiled_op(_merge_padded_rows, left, right, split)
        for args in _filter_inputs(left):
            warmup_compiled_op(_filter_padded_rows, *args)


def warmup_row_ops(templates: list[torch.Tensor], max_rows: int) -> None:
    """Finish row programs and compact copies before serving admission."""
    for left, right, split in _row_bucket_inputs(templates, max_rows):
        bucket = left.shape[0]
        merged = _merge_padded_rows(left, right, split) if bucket > 1 else left
        for rows in range(1, min(bucket, max_rows) + 1):
            sync.synchronize([_logical_rows(merged, rows)])
        for args in _filter_inputs(left):
            _filter_padded_rows(*args)
