# SPDX-License-Identifier: Apache-2.0
"""Compare compiled Pallas and torch LFM2 ShortConv, including cache writes.

  python test/manual/kernels/bench_lfm2_short_conv_tpu.py --channels 1536 \
      --batch 8 --length 1024 --mode both

Each sample measures host dispatch + device completion. Compilation, warmup
and state reset are excluded. This is wall time, not xprof device-only time.
"""

import argparse
import statistics
import time

import torch

from sglang_tpu.attention_short_conv import short_conv_decode, short_conv_prefill
from sglang_tpu.kernels.mamba.short_conv import build_short_conv_op


def _sync(tensors):
    from torch_tpu._internal.sync.sync import synchronize

    synchronize(tensors, wait=True)


def _time(fn, initial, iters):
    samples = []
    for _ in range(iters):
        state = initial.to("tpu")
        _sync([state])
        begin = time.perf_counter()
        out = fn(state)
        _sync([out, state])
        samples.append((time.perf_counter() - begin) * 1e6)
    return statistics.median(samples)


def bench(args, decode):
    device = torch.device("tpu")
    torch.manual_seed(0)
    length = 1 if decode else args.length
    tokens = args.batch * length
    x = torch.randn(tokens, args.channels).bfloat16().to(device)
    weight = torch.randn(args.channels, args.kernel_size).bfloat16().to(device)
    bias = torch.randn(args.channels).bfloat16().to(device)
    initial = torch.randn(args.slots, args.kernel_size - 1, 1, args.channels)
    slots = torch.arange(1, args.batch + 1, dtype=torch.int64, device=device)
    starts = torch.arange(0, tokens + 1, length, dtype=torch.int64, device=device)
    prefix = torch.ones(args.batch, dtype=torch.bool, device=device)
    requests = torch.arange(args.batch, device=device).repeat_interleave(length)
    valid = torch.ones(tokens, dtype=torch.bool, device=device)
    op = build_short_conv_op()

    def pallas(state):
        return op(x, weight, bias, state, slots, starts, prefix, decode)

    def reference(state):
        if decode:
            return short_conv_decode(x, weight, bias, state, slots)
        return short_conv_prefill(
            x, weight, bias, state, slots, starts, prefix, requests, valid
        )

    pallas_fn = torch.compile(pallas, backend="tpu", fullgraph=True, dynamic=False)
    reference_fn = torch.compile(
        reference, backend="tpu", fullgraph=True, dynamic=False
    )
    # Compare outputs and mutations before timing, and materialize every input
    # before donation. Independent host snapshots avoid lazy-clone aliasing.
    states = [initial.to(device), initial.to(device)]
    _sync([x, weight, bias, starts, slots, prefix, requests, valid, *states])
    actual, expected = pallas_fn(states[0]), reference_fn(states[1])
    _sync([actual, expected, *states])
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(states[0].cpu(), states[1].cpu(), rtol=0, atol=0)
    for fn in (pallas_fn, reference_fn):
        for _ in range(3):
            state = initial.to(device)
            _sync([state])
            out = fn(state)
            _sync([out, state])
    pallas_us = _time(pallas_fn, initial, args.iters)
    reference_us = _time(reference_fn, initial, args.iters)
    print(
        f"{'decode' if decode else 'prefill'} B={args.batch} T={tokens} "
        f"C={args.channels} K={args.kernel_size} slots={args.slots}: "
        f"Pallas={pallas_us:.2f} us, torch={reference_us:.2f} us, "
        f"speedup={reference_us / pallas_us:.2f}x (median, BF16, compiled)"
    )


def main():
    import torch_tpu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--slots", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--mode", choices=("prefill", "decode", "both"), default="both")
    args = parser.parse_args()
    assert 0 < args.batch < args.slots
    for decode in (False, True):
        if args.mode == "both" or args.mode == ("decode" if decode else "prefill"):
            bench(args, decode)


if __name__ == "__main__":
    main()
