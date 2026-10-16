"""Warm the scheduler relay using scratch buffers and the serving entry point."""

from types import SimpleNamespace

import torch

from sglang.srt.managers.overlap_utils import (
    FutureMap,
    RelayPayload,
    resolve_forward_inputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang_tpu.compilation.compiled_forward_shim import _pad_token_fields


@torch.no_grad()
def warmup_sequence_lengths(device, batch_buckets):
    # RPA converts uploaded scheduler lengths to its int32 kernel metadata.
    for rows in batch_buckets:
        torch.zeros(rows, dtype=torch.int64).to(device).to(torch.int32).cpu()


@torch.no_grad()
def warmup_host_relay(runner, bucketer, max_live, *, extend_cap):
    for rows in range(1, max_live + 1):
        host_indices = torch.arange(1, rows + 1, dtype=torch.int64)
        indices = host_indices.to(runner.device, non_blocking=True)
        padded_rows = bucketer.next_bucket(rows)
        # A view keeps its producer's extent in the TPU program signature.
        source_sizes = {rows, padded_rows}
        if rows <= extend_cap:
            source_sizes.add(extend_cap)
        for source_rows in sorted(source_sizes):
            relay = FutureMap(
                runner.device,
                runner.spec_algorithm,
                runner.req_to_token_pool,
                needs_cpu_seq_lens=False,
            )
            for dtype in (torch.int64, torch.int32):
                values = torch.arange(source_rows, dtype=dtype).to(runner.device)
                # The first store into an empty TPU buffer has no destination
                # parameter; subsequent stores update an existing buffer.
                for _ in range(2):
                    relay.stash(indices, RelayPayload(bonus_tokens=values[:rows]))
                    batch = SimpleNamespace(
                        input_ids=None,
                        prefill_input_ids_cpu=None,
                        req_pool_indices=host_indices,
                        enable_overlap=True,
                        spec_algorithm=runner.spec_algorithm,
                        forward_mode=ForwardMode.DECODE,
                        spec_info=None,
                        out_cache_loc=torch.arange(rows, dtype=torch.int32).to(
                            runner.device
                        ),
                    )
                    resolve_forward_inputs(batch, relay)
                    _pad_token_fields(batch, padded_rows, is_draft_extend_v2=False)
                    batch.input_ids.cpu()
                    batch.out_cache_loc.cpu()
