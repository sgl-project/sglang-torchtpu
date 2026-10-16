"""Warm the scheduler relay using scratch buffers and the serving entry point."""

import torch

from sglang.srt.managers.overlap_utils import RelayPayload


@torch.no_grad()
def warmup_sequence_lengths(device, batch_buckets):
    # RPA converts uploaded scheduler lengths to its int32 kernel metadata.
    for rows in batch_buckets:
        torch.zeros(rows, dtype=torch.int64).to(device).to(torch.int32).cpu()


@torch.no_grad()
def warmup_host_relay(runner, bucketer, max_live, *, extend_cap):
    from sglang_tpu.host_padding_relay import HostPaddingFutureMap

    relay = HostPaddingFutureMap(
        runner.device,
        runner.spec_algorithm,
        runner.req_to_token_pool,
        needs_cpu_seq_lens=False,
    )
    for rows in sorted(set(bucketer.buckets) | {extend_cap}):
        live = min(rows, max_live)
        indices = torch.arange(1, live + 1, dtype=torch.int64)
        for dtype in (torch.int64, torch.int32):
            values = torch.arange(rows, dtype=dtype).to(runner.device)
            relay.stash(indices, RelayPayload(bonus_tokens=values))
            relay.gather(indices).cpu()
