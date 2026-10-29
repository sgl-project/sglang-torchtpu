#!/usr/bin/env python3
"""CPU parity check: online-quantized BF16 == the FP8 checkpoint, byte for byte.

Reproduces the load-time quantization of launch_beam_bf16_fp8_online.sh
(sglang.srt.layers.quantization.compressed_tensors.online_fp8) on CPU and
compares against the official FP8 checkpoint:

  * every FP8 (float8_e4m3fn) weight + its weight_scale in the FP8 checkpoint
    must equal quantize_fp8_per_channel(bf16_weight) bit for bit
  * every tensor on the quant ignore list (kept BF16/FP32) must be identical
    in both checkpoints

  python check_online_fp8_parity.py \
      --bf16 /path/Beam-1.0-501B-A23B --fp8 /path/Beam-1.0-501B-A23B-FP8 \
      --sample 400
"""
import argparse
import json
import os
import random
import sys

import torch
from safetensors import safe_open

FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(FP8_DTYPE).max


def quantize_fp8_per_channel(weight: torch.Tensor):
    """Mirror of online_fp8.quantize_fp8_per_channel (keep in sync!)."""
    amax = weight.abs().amax(dim=1, keepdim=True)
    scale = amax / _FP8_MAX
    scale = torch.where(scale == 0, torch.finfo(scale.dtype).eps, scale)
    scaled = weight / scale
    scaled += 0  # compressed-tensors adds the (zero) zero point: -0.0 -> +0.0
    qweight = torch.clamp(scaled, -_FP8_MAX, _FP8_MAX).to(FP8_DTYPE)
    return qweight, scale


class Checkpoint:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.index = json.load(f)["weight_map"]
        self._handles = {}

    def get(self, name) -> torch.Tensor:
        shard = self.index[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                os.path.join(self.path, shard), framework="pt", device="cpu"
            )
        return self._handles[shard].get_tensor(name)

    def dtype(self, name):
        shard = self.index[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                os.path.join(self.path, shard), framework="pt", device="cpu"
            )
        return self._handles[shard].get_slice(name).get_dtype()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--fp8", required=True)
    ap.add_argument("--sample", type=int, default=0, help="N random FP8 tensors; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bf16, fp8 = Checkpoint(args.bf16), Checkpoint(args.fp8)

    fp8_weights = [
        n
        for n in fp8.index
        if n.endswith(".weight") and f"{n[:-len('.weight')]}.weight_scale" in fp8.index
    ]
    names = sorted(fp8_weights)
    if args.sample and args.sample < len(names):
        names = sorted(random.Random(args.seed).sample(names, args.sample))

    print(f"checking {len(names)} of {len(fp8_weights)} FP8-quantized tensors", flush=True)
    bad = 0
    for i, name in enumerate(names):
        mod = name[: -len(".weight")]
        ref_q, ref_s = fp8.get(name), fp8.get(f"{mod}.weight_scale")
        if name not in bf16.index:
            print(f"MISSING in bf16 ckpt: {name}", flush=True)
            bad += 1
            continue
        q, s = quantize_fp8_per_channel(bf16.get(name))
        ok = (
            q.dtype == ref_q.dtype
            and s.dtype == ref_s.dtype
            and torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
            and torch.equal(s.view(torch.uint8), ref_s.view(torch.uint8))
        )
        if not ok:
            bad += 1
            mism = (q.view(torch.uint8) != ref_q.view(torch.uint8)).sum().item()
            print(f"MISMATCH {name}: {mism}/{q.numel()} elements differ", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(names)}] ok so far, {bad} bad", flush=True)

    # Ignore-list tensors stay full precision and must be byte-identical.
    bf16_only = [
        n
        for n in bf16.index
        if n.endswith(".weight") and n not in fp8_weights and n in fp8.index
    ]
    print(f"checking {len(bf16_only)} full-precision (ignore-list) tensors", flush=True)
    for name in bf16_only:
        a, b = bf16.get(name), fp8.get(name)
        if a.dtype != b.dtype or not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
            bad += 1
            print(f"MISMATCH (full-precision) {name}", flush=True)

    print("PARITY OK" if bad == 0 else f"PARITY FAILED: {bad} mismatched tensors")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
