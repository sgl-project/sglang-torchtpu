"""On-device raiden D2h/H2d round-trip test for the TPU KV-offload path.

Requires a TPU + a built tpu_sync extension + torch_tpu. NOT a CI unit test
(see test_tpu_raiden_offload_unit.py for the host-only logic). Run with:

    PYTHONPATH=/path/to/sglang-torchtpu/python:/path/to/tpu-sync \
        python3 test/manual/tpu/test_tpu_raiden_roundtrip_ondevice.py

IMPORTANT — why this test never writes the KV buffer with eager ops:
On torch_tpu eager mode, ANY write (``buf[i]=x``, ``copy_``, ``fill_``)
reallocates the tensor's underlying PJRT buffer (its ``data_ptr`` changes), so
raiden's one-time buffer registration would go stale and read/write the wrong
storage. Production never hits this: the Pallas RPA-v3 kernel updates the KV
cache via ``donate_argnums`` (XLA aliases input==output, in place), keeping the
same PJRT buffer. So this test validates the transfer mechanism by registering
a clean, pre-populated buffer and doing transfers ONLY (no eager writes between
registration and the D2h/H2d), which faithfully exercises RaidenKVTransfer,
page-id addressing, and poll-only completion.
"""

import time

import torch

from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang_tpu.kv_pool_host import TpuDMAHostKVPool
from sglang_tpu.raiden_transfer import RaidenKVTransfer

DEVICE = "tpu:0"
PAGE_SIZE = 64


def _poll(ev, what):
    t0 = time.monotonic()
    while not ev.query():
        if time.monotonic() - t0 > 30:
            raise TimeoutError(f"{what} did not complete within 30s")
        time.sleep(0.001)


class _SingleBufferPool:
    """Minimal device-pool surface RaidenKVTransfer needs, over one fused
    buffer pre-populated with a known pattern (no eager writes after register).
    """

    def __init__(self, buf):
        self._buf = buf
        self.layer_num = 1
        self.start_layer = 0
        self.page_size = PAGE_SIZE

    def get_fused_kv_buffer(self, layer_id):
        return self._buf


def test_raiden_dma_roundtrip():
    """D2h device pages -> host, H2d host -> DIFFERENT device pages, verify."""
    print("== test 1: raiden D2h/H2d data round-trip (stable buffer) ==")
    # 8 blocks; pattern in pages 1,2; pages 4,5 start at zero (the H2d targets).
    pat = torch.zeros(8, PAGE_SIZE, 2, 2, 128, dtype=torch.bfloat16)
    pat[1] = 101.0
    pat[2] = 102.0
    data = pat.to(DEVICE)  # clean block-major device buffer
    transfer = RaidenKVTransfer(_SingleBufferPool(data), num_host_pages=8)

    assert float(data[4, 0, 0, 0, 0].cpu()) == 0.0

    _poll(transfer.backup([1, 2], [0, 1]), "D2h")  # device 1,2 -> host 0,1
    _poll(transfer.load([0, 1], [4, 5]), "H2d")  # host 0,1 -> device 4,5
    torch.accelerator.synchronize()

    got4 = float(data[4, 0, 0, 0, 0].cpu())
    got5 = float(data[5, 0, 0, 0, 0].cpu())
    assert abs(got4 - 101.0) < 1e-2 and abs(got5 - 102.0) < 1e-2, (
        f"round-trip mismatch: pages 4,5 = {got4},{got5} (expected 101,102)"
    )
    print(f"  pages 4,5 after D2h->H2d = {got4},{got5} OK")


def test_host_pool_wiring():
    """TpuDMAHostKVPool over the real fused pool: token->page conversion,
    page-granular alloc/free, and that backup/load EXECUTE and complete on
    hardware (data is zeros here — fidelity is covered by test 1)."""
    print("== test 2: TpuDMAHostKVPool wiring on the real pool ==")
    pool = TpuFusedMHATokenToKVPool(
        size=256,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=128,
        layer_num=2,
        device=DEVICE,
        enable_memory_saver=False,
    )
    host = TpuDMAHostKVPool(
        pool,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=PAGE_SIZE,
        layout="fused_raiden",
    )
    dev_pages = [1, 2]
    device_indices = torch.tensor(
        [p * PAGE_SIZE + i for p in dev_pages for i in range(PAGE_SIZE)],
        dtype=torch.int64,
    )
    host_indices = host.alloc(len(device_indices))
    assert host_indices is not None
    assert host._to_page_ids(host_indices) == [0, 1]
    assert host._to_page_ids(device_indices) == [1, 2]

    _poll(
        host.backup_from_device_all_layer(pool, host_indices, device_indices, "raiden"),
        "host backup",
    )
    _poll(host.load_all_layer(pool, host_indices, device_indices), "host load")
    torch.accelerator.synchronize()
    print("  alloc/backup/load executed + completed OK")

    free_before = host.available_size()
    host.free(host_indices)
    assert host.available_size() == free_before + len(host_indices)
    print("  page-granular free OK")


if __name__ == "__main__":
    print(f"TPU chips: {torch.accelerator.device_count()}")
    test_raiden_dma_roundtrip()
    test_host_pool_wiring()
    print("\nALL ON-DEVICE RAIDEN TESTS PASSED")
