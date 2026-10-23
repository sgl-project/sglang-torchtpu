# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark + numerics check for the TPU Pallas splash attention ops.

Times, at one non-causal ``[B, H, S, D]`` shape:

* the flash kernel op (autotuned blocks), which ``tpu_attention`` selects for
  non-causal attention over at least 512 keys;
* the ``make_splash_mha`` op at the same shape (query and KV padded to its
  blocks, padded keys masked);
* torch SDPA (the ``torch_sdpa`` backend).

The default shape is Wan2.2-T2V-A14B at 720p/81f under Ulysses 4:
S = 21 latent frames x 45 x 80 = 75600 tokens, 40 heads / 4 = 10 local heads,
head_dim 128.

Run on one TPU core from the repository root:

    PYTHONPATH=python:. python3 benchmark/kernels/attention/bench_tpu_splash_attn.py
"""

import argparse
import os
import time
import traceback

import torch
import torch.nn.functional as F


def _sync(tensors):
    """Wait for lazily scheduled torch_tpu work."""
    from torch_tpu._internal.sync.sync import synchronize

    synchronize(tensors, wait=True)


def _reference_attention(q, k, v, scale, chunk=4096):
    """Chunked float32 attention on CPU; never materializes the full score matrix."""
    q32, k32, v32 = q.float(), k.float(), v.float()
    outs = []
    for start in range(0, q32.shape[2], chunk):
        scores = torch.matmul(q32[:, :, start : start + chunk], k32.transpose(-2, -1))
        outs.append(torch.matmul(torch.softmax(scores * scale, dim=-1), v32))
    return torch.cat(outs, dim=2)


def _time(fn, args, warmup, iters, label, flops, keep_output):
    print(f"\n=== {label} ===")
    try:
        t0 = time.perf_counter()
        out = fn(*args)
        _sync([out])
        print(f"  first call (incl. compile): {time.perf_counter() - t0:.2f}s")
        for _ in range(warmup):
            out = fn(*args)
        _sync([out])
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        _sync([out])
    except Exception:
        print(f"  FAILED:\n{traceback.format_exc()}")
        return None, None
    per_call = (time.perf_counter() - t0) / iters
    print(
        f"  steady state: {per_call * 1e3:.2f} ms/call "
        f"({flops / per_call / 1e12:.1f} TFLOP/s)"
    )
    return per_call, (out.cpu() if keep_output else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=75600, help="query sequence length")
    ap.add_argument(
        "--kv-seq", type=int, default=None, help="key/value length (default: --seq)"
    )
    ap.add_argument(
        "--heads", type=int, default=10, help="local heads after the Ulysses all-to-all"
    )
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument(
        "--check-numerics",
        action="store_true",
        help="compare each op against a chunked float32 CPU reference (slow at S=75600)",
    )
    args = ap.parse_args()

    # As in the diffusion TPU worker: JAX only traces and lowers on the host.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import torch_tpu  # noqa: F401  (registers the 'tpu' PrivateUse1 device)

    from sglang_tpu.multimodal_gen import tpu_splash_attn as tsa

    head_dim = 128
    kv_seq = args.kv_seq or args.seq
    print(
        f"shape: B=1 H={args.heads} Sq={args.seq} Skv={kv_seq} D={head_dim} bf16\n"
        f"flash kernel blocks: {tsa._autotune_block_sizes(args.seq, kv_seq, head_dim)}"
    )
    tsa._register_ops()

    torch.manual_seed(0)
    q_cpu = torch.randn(1, args.heads, args.seq, head_dim, dtype=torch.bfloat16)
    k_cpu, v_cpu = (
        torch.randn(1, args.heads, kv_seq, head_dim, dtype=torch.bfloat16)
        for _ in range(2)
    )
    scale = head_dim**-0.5
    q, k, v = (t.to("tpu") for t in (q_cpu, k_cpu, v_cpu))
    _sync([q, k, v])
    # QK^T + PV.
    flops = 4 * args.seq * kv_seq * head_dim * args.heads

    results = {}
    for name, label, fn in (
        (
            "flash",
            "flash kernel op",
            lambda q_, k_, v_: tsa._FLASH_OP(q_, k_, v_, scale),
        ),
        (
            "splash_mha",
            "make_splash_mha op",
            lambda q_, k_, v_: tsa._SPLASH_MHA_OP(q_, k_, v_, scale, False),
        ),
        (
            "sdpa",
            "torch SDPA",
            lambda q_, k_, v_: F.scaled_dot_product_attention(q_, k_, v_, scale=scale),
        ),
    ):
        results[name] = _time(
            fn, (q, k, v), args.warmup, args.iters, label, flops, args.check_numerics
        )

    if args.check_numerics:
        print("\n=== numerics vs float32 ===")
        ref = _reference_attention(q_cpu, k_cpu, v_cpu, scale)
        for name, (_, out) in results.items():
            if out is not None:
                err = (out.float() - ref).abs()
                print(
                    f"  {name:<10} max_abs={err.max().item():.3e} "
                    f"mean_abs={err.mean().item():.3e}"
                )

    print("\n=== summary ===")
    flash_t = results["flash"][0]
    for name, (t, _) in results.items():
        if t is None:
            print(f"  {name:<10} failed")
            continue
        speedup = f"  flash speedup {t / flash_t:.2f}x" if flash_t else ""
        print(f"  {name:<10} {t * 1e3:8.2f} ms{speedup}")


if __name__ == "__main__":
    main()
