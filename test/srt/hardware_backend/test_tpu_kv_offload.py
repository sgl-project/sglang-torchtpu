"""Round-trip tests for the TPU HiCache KV-offload data plane (overlapped).

Run on a TPU host with the full torch_tpu + jax runtime:

    python test/srt/hardware_backend/test_tpu_kv_offload.py

Covers:
  1. Pallas scatter -> gather round-trip (layout + kernel + 128-align + pow2 pad).
  2. D2H -> host -> H2D round-trip incl. the PCIe ``copy_`` legs.
  3. End-to-end host pool: backup -> evict -> ASYNC load restores bit-exact
     (load issues the scatter without a blocking sync; the ``.cpu()`` readback
     forces materialization, validating correctness of the overlapped path).

The pure host-side indexing/pad logic is covered separately on CPU. Whether the
overlapped load DMA actually *overlaps* compute is an integration/profiler
question, not asserted here.
"""

import numpy as np
import torch

try:
    import torch_tpu  # noqa: F401

    from sglang.srt.hardware_backend.tpu.kernels.kv_offload import (
        _gather_fn,
        _scatter_fn,
        _tpu_sync,
        pad_to_pow2,
    )

    _HAS_TPU = True
except Exception as e:  # pragma: no cover
    print(f"[skip] TPU/jax runtime not available: {e}")
    _HAS_TPU = False

DEV = "tpu"
PAGE_SIZE, KV_HEADS_PACKED, KV_PACKING, PADDED_HD, NUM_BLOCKS = 16, 2, 2, 128, 64
BLOCK = (PAGE_SIZE, KV_HEADS_PACKED, KV_PACKING, PADDED_HD)


def _rand_blocks(n):
    return torch.randn((n,) + BLOCK, dtype=torch.float32).to(torch.bfloat16)


def _pad_src(src, p):
    n = src.shape[0]
    if p <= n:
        return src
    return torch.cat([src, src[-1:].expand((p - n,) + tuple(src.shape[1:]))], dim=0)


def test_scatter_gather_roundtrip():
    cache = torch.zeros((NUM_BLOCKS,) + BLOCK, dtype=torch.bfloat16, device=DEV)
    block_ids = np.array([3, 7, 1, 18, 9, 40, 2], dtype=np.int64)
    n = len(block_ids)
    src = _rand_blocks(n)
    padded, _ = pad_to_pow2(block_ids)
    p = len(padded)
    ids = torch.from_numpy(padded.astype(np.int32)).to(DEV)
    cache[:] = _scatter_fn(cache, _pad_src(src, p).to(DEV), ids)
    _tpu_sync([cache], wait=True)
    gathered = _gather_fn(cache, ids)
    _tpu_sync([gathered], wait=True)
    assert torch.equal(gathered[:n].cpu(), src.cpu()), "scatter/gather mismatch"
    print(f"[ok] scatter/gather round-trip (n={n}, p={p})")


def test_d2h_h2d_pcie_roundtrip():
    cache = torch.zeros((NUM_BLOCKS,) + BLOCK, dtype=torch.bfloat16, device=DEV)
    block_ids = np.array([5, 11, 30], dtype=np.int64)
    n = len(block_ids)
    src = _rand_blocks(n)
    padded, _ = pad_to_pow2(block_ids)
    p = len(padded)
    ids = torch.from_numpy(padded.astype(np.int32)).to(DEV)
    cache[:] = _scatter_fn(cache, _pad_src(src, p).to(DEV), ids)
    _tpu_sync([cache], wait=True)

    gathered = _gather_fn(cache, ids)
    _tpu_sync([gathered], wait=False)
    host = torch.empty(gathered.shape, dtype=torch.bfloat16, device="cpu")
    host.copy_(gathered)

    cache[:] = _scatter_fn(
        cache, torch.zeros((p,) + BLOCK, dtype=torch.bfloat16, device=DEV), ids
    )
    _tpu_sync([cache], wait=True)

    dev_stage = torch.empty(host.shape, dtype=torch.bfloat16, device=DEV)
    dev_stage.copy_(host)
    cache[:] = _scatter_fn(cache, dev_stage, ids)
    _tpu_sync([cache], wait=True)
    restored = _gather_fn(cache, ids)
    _tpu_sync([restored], wait=True)
    assert torch.equal(restored[:n].cpu(), src.cpu()), "D2H/H2D mismatch"
    print(f"[ok] D2H->host->H2D PCIe round-trip (n={n}, p={p})")


def test_host_pool_overlapped_roundtrip():
    """Real TpuFusedMHATokenToKVPool + host pool: backup -> evict -> async load
    restores bit-exact (overlapped load issues the scatter without blocking)."""
    from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
    from sglang.srt.mem_cache.memory_pool_host import TpuFusedMHATokenToKVPoolHost

    size, page_size, layer_num, head_num, head_dim = 256, 16, 2, 2, 64
    dev = TpuFusedMHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        device=DEV,
        enable_memory_saver=False,
    )
    host = TpuFusedMHATokenToKVPoolHost(
        dev,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=page_size,
        layout="page_first_fused",
        device="cpu",
    )
    dev_pages = [1, 2]
    orig = {}
    for L in range(layer_num):
        r = (
            torch.randn((len(dev_pages),) + tuple(dev.kv_buffer[L].shape[1:]))
            .to(torch.bfloat16)
            .to(DEV)
        )
        dev.kv_buffer[L][dev_pages] = r
        orig[L] = r.cpu().clone()
    device_indices = torch.arange(
        dev_pages[0] * page_size, (dev_pages[-1] + 1) * page_size, device=DEV
    )
    host_indices = host.alloc(len(device_indices))
    assert host_indices is not None
    host.backup_from_device_all_layer(dev, host_indices, device_indices, "pallas_tpu")
    for L in range(layer_num):
        dev.kv_buffer[L][dev_pages] = 0
        assert dev.kv_buffer[L][dev_pages].cpu().abs().sum().item() == 0.0
    for L in range(layer_num):  # async load (no blocking sync)
        host.load_to_device_per_layer(
            dev, host_indices, device_indices, L, "pallas_tpu"
        )
    for L in range(layer_num):  # .cpu() forces materialization
        assert torch.equal(dev.kv_buffer[L][dev_pages].cpu(), orig[L]), f"layer {L}"
    print(
        f"[ok] overlapped host-pool backup->evict->async-load round-trip (layers={layer_num})"
    )


if __name__ == "__main__":
    if not _HAS_TPU:
        raise SystemExit(0)
    test_scatter_gather_roundtrip()
    test_d2h_h2d_pcie_roundtrip()
    test_host_pool_overlapped_roundtrip()
    print("ALL TPU ROUND-TRIP TESTS PASSED")
