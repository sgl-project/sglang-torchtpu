"""On-device test: hybrid raiden KV offload piggybacks the draft sidecar pool.

Requires a TPU + a built tpu_sync extension + torch_tpu. NOT a CI unit test
(see test_tpu_raiden_hybrid_offload_unit.py for the host-only logic). Run with:

    PYTHONPATH=/path/to/sglang-torchtpu/python \
        python3 test/manual/tpu/test_tpu_raiden_hybrid_spec_offload_ondevice.py

Drives a write round + a load round through ``TpuDMAHybridCacheController``
with the hybrid pool set — a full-attention anchor pool, a SWA pool with its
own page-id space, and an EAGLE-style 1-layer draft sidecar riding the
anchor's indices — each backed by its own raiden manager (real DMA), and
checks bit-exact data movement on all three pools.

Like test_tpu_raiden_roundtrip_ondevice.py, the KV buffers are seeded with
eager writes BEFORE the host pools register them with raiden and never written
again (eager writes reallocate the PJRT buffer and would orphan the
registration; production updates KV in place via kernel donation). The load
round therefore targets DIFFERENT device pages instead of evicting the source
pages: source pages -> host -> destination pages, verify the destination.
"""

import time

import torch

from sglang.srt.managers.cache_controller import BatchDoneCounter
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import CacheOperation
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang_tpu.kv_pool_host import TpuDMAHostKVPool
from sglang_tpu.tpu_dma_hybrid_cache_controller import (
    TpuDMAHybridCacheController,
)

DEVICE = "tpu:0"
PAGE_SIZE = 16
NUM_PAGES = 16
SRC_PAGES = [1, 2, 3]
DST_PAGES = [8, 9, 10]
# The SWA pool has its own page-id space; use different pages to prove the
# indices are consumed per pool, not shared.
SWA_SRC_PAGES = [4, 5]
SWA_DST_PAGES = [11, 12]


def _poll(ev, what, timeout_s=30):
    t0 = time.monotonic()
    while not ev.query():
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(f"{what} did not complete within {timeout_s}s")
        time.sleep(0.001)


def _make_pool(layer_num):
    return TpuFusedMHATokenToKVPool(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=64,
        layer_num=layer_num,
        device=DEVICE,
        enable_memory_saver=False,
    )


def _seed(pool, tag):
    """Give every (layer, page) a distinct constant; returns expected values.

    Eager writes rebind the underlying PJRT buffers, so this must run BEFORE
    the host pool registers them with raiden. Constants stay below 256 so
    bfloat16 represents each one exactly (bit-exact compare after the DMAs).
    """
    expected = {}
    for layer in range(pool.layer_num):
        buf = pool.get_fused_kv_buffer(layer)
        pattern = torch.zeros(buf.shape, dtype=buf.dtype)
        for page in range(NUM_PAGES):
            pattern[page] = tag + 40 * layer + page
        buf.copy_(pattern.to(DEVICE))
        expected[layer] = pattern
    return expected


def _page_tokens(pages):
    return torch.tensor(
        [p * PAGE_SIZE + i for p in pages for i in range(PAGE_SIZE)],
        dtype=torch.int64,
    )


def _make_host_pool(pool, label):
    return TpuDMAHostKVPool(
        pool,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=PAGE_SIZE,
        layout="fused_raiden",
        pool_label=label,
    )


class _Entry:
    def __init__(self, name, host_pool, device_pool):
        self.name = name
        self.host_pool = host_pool
        self.device_pool = device_pool


class _Group:
    def __init__(self, anchor, extra):
        self.anchor_entry = anchor
        self.entry_map = {anchor.name: anchor}
        for entry in extra:
            self.entry_map[entry.name] = entry


def _make_controller(anchor, swa, draft):
    controller = object.__new__(TpuDMAHybridCacheController)
    controller.io_backend = "raiden"
    controller.mem_pool_host = _Group(anchor, [swa, draft])
    controller.layer_done_counter = BatchDoneCounter()
    controller.write_queue = []
    controller.load_queue = []
    controller.ack_write_queue = []
    controller.ack_load_queue = []
    # The draft device pool gates its forward on the whole-batch counter,
    # mirroring register_host_pool_entry's wiring in the assembled stack.
    draft.device_pool.register_layer_transfer_counter(controller.layer_done_counter)
    return controller


def _op(anchor_host, swa_host, anchor_pages, swa_pages, draft_pages):
    return CacheOperation(
        anchor_host,
        _page_tokens(anchor_pages),
        node_id=0,
        pool_transfers=[
            PoolTransfer(
                name=PoolName.SWA,
                host_indices=swa_host,
                device_indices=_page_tokens(swa_pages),
            ),
            # Draft sidecar rides the anchor's host slots
            # (SidecarPoolSpec(indices_from_pool=PoolName.KV)).
            PoolTransfer(
                name=PoolName.DRAFT,
                host_indices=anchor_host,
                device_indices=_page_tokens(draft_pages),
                indices_from_pool=PoolName.KV,
            ),
        ],
    )


def _verify_moved(pool, expected, src_pages, dst_pages, what):
    """dst_pages must now hold src_pages' data; all other pages untouched."""
    for layer in range(pool.layer_num):
        got = pool.get_fused_kv_buffer(layer).cpu()
        want = expected[layer].clone()
        want[dst_pages] = want[src_pages]
        assert torch.equal(got, want), (
            f"{what} layer {layer}: KV mismatch after backup->load "
            f"(dst pages {dst_pages} should hold src pages {src_pages} data)"
        )
    print(f"  {what}: bit-exact on all {pool.layer_num} layer(s)")


def test_hybrid_round_piggybacks_draft():
    print("== hybrid raiden write+load round: full + SWA + draft sidecar ==")
    full_dev = _make_pool(layer_num=2)
    swa_dev = _make_pool(layer_num=1)
    draft_dev = _make_pool(layer_num=1)
    expected_full = _seed(full_dev, tag=1.0)
    expected_swa = _seed(swa_dev, tag=101.0)
    expected_draft = _seed(draft_dev, tag=201.0)

    anchor = _Entry(PoolName.KV, _make_host_pool(full_dev, "kv"), full_dev)
    swa = _Entry(PoolName.SWA, _make_host_pool(swa_dev, "swa"), swa_dev)
    draft = _Entry(PoolName.DRAFT, _make_host_pool(draft_dev, "draft"), draft_dev)
    controller = _make_controller(anchor, swa, draft)
    assert draft_dev.layer_transfer_counter is controller.layer_done_counter

    anchor_host = anchor.host_pool.alloc(len(SRC_PAGES) * PAGE_SIZE)
    swa_host = swa.host_pool.alloc(len(SWA_SRC_PAGES) * PAGE_SIZE)
    assert anchor_host is not None and swa_host is not None

    # Write round: every pool's SRC pages -> host (one ack over all DMAs).
    controller.write_queue.append(
        _op(anchor_host, swa_host, SRC_PAGES, SWA_SRC_PAGES, SRC_PAGES)
    )
    controller.start_writing()
    (write_ack,) = controller.ack_write_queue
    _poll(write_ack.finish_event, "write-back (full+swa+draft)")

    # Load round: host -> every pool's DST pages.
    controller.load_queue.append(
        _op(anchor_host, swa_host, DST_PAGES, SWA_DST_PAGES, DST_PAGES)
    )
    producer_id = controller.start_loading()
    assert producer_id >= 0
    (load_ack,) = controller.ack_load_queue
    _poll(load_ack.finish_event, "load-back (full+swa+draft)")
    torch.accelerator.synchronize()

    _verify_moved(full_dev, expected_full, SRC_PAGES, DST_PAGES, "full pool (2 layers)")
    _verify_moved(
        swa_dev, expected_swa, SWA_SRC_PAGES, SWA_DST_PAGES, "swa pool (1 layer)"
    )
    _verify_moved(
        draft_dev, expected_draft, SRC_PAGES, DST_PAGES, "draft pool (1 layer)"
    )


if __name__ == "__main__":
    print(f"TPU chips: {torch.accelerator.device_count()}")
    test_hybrid_round_piggybacks_draft()
    print("\nON-DEVICE HYBRID RAIDEN SPEC-DRAFT OFFLOAD TEST PASSED")
