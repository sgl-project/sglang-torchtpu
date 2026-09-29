"""On-device raiden D2H/H2D round trip for the TPU KV-offload path.

Needs a TPU, torch_tpu and tpu-sync-torch (all in the TPU image). Skips only when
torch_tpu is absent or sees no TPU, so a missing tpu_sync on a TPU host fails
instead of skipping.

The device buffers are never written with eager ops after raiden registers
them: on torch_tpu eager mode any write (``buf[i] = x``, ``copy_``, ``fill_``)
reallocates the tensor's PJRT buffer, so raiden's one-time registration by raw
pointer would go stale. Production never hits this, because the attention
kernels update the KV cache in place through ``donate_argnums``. The tests
therefore register a pre-populated buffer and only transfer.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

import importlib.util
import time
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

DEVICE = "tpu:0"
# 128 fills the lane axis of the page-on-lane layouts, so no lane padding.
PAGE_SIZE = 128
NUM_BLOCKS = 8
NUM_LAYERS = 2

# raiden addresses a page by its dim-0 block id and sizes it from the buffer's
# own shape, so any block-major layout should round-trip. One shape per layout:
BLOCK_MAJOR_SHAPES = {
    # pallas_tpu_rpa_v3 fused row:
    # [blocks, page, 2*kv_heads/packing, packing, head_dim]
    "v3_fused": (NUM_BLOCKS, PAGE_SIZE, 2, 2, 128),
    # SEQ_ALONG_LANE (TpuSeqAlongLaneKVPool):
    # [blocks, 2*kv_heads, head_dim/packing, packing, page]
    "seq_along_lane": (NUM_BLOCKS, 2, 64, 2, PAGE_SIZE),
    # rank-4 token-on-lane pool of out-of-tree RPA kernels:
    # [blocks, 2*kv_heads, head_dim, page]
    "rank4_seq": (NUM_BLOCKS, 2, 128, PAGE_SIZE),
}


def _has_tpu() -> bool:
    # ``import torch_tpu`` registers the ``tpu`` backend; gating it on find_spec
    # keeps collection side-effect free on CPU hosts.
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    import torch_tpu  # noqa: F401

    return torch.tpu.is_available()


def _poll(test, event, what):
    deadline = time.monotonic() + 30
    while not event.query():
        test.assertLess(time.monotonic(), deadline, f"{what} did not complete in 30s")
        time.sleep(0.001)


def _page_pattern(shape, page_id, layer_id):
    # Integers below 256 are exact in bf16, so a scrambled element changes value.
    numel = torch.Size(shape[1:]).numel()
    values = (torch.arange(numel) + 7 * page_id + 31 * layer_id) % 251 + 1
    return values.reshape(shape[1:]).to(torch.bfloat16)


def _seeded(shape, layer_id):
    """Pages 1 and 2 patterned, the rest zero; and the buffer expected after
    pages 1,2 are backed up and loaded into pages 4,5."""
    initial = torch.zeros(shape, dtype=torch.bfloat16)
    initial[1] = _page_pattern(shape, 1, layer_id)
    initial[2] = _page_pattern(shape, 2, layer_id)
    expected = initial.clone()
    expected[4] = initial[1]
    expected[5] = initial[2]
    return initial, expected


class _StaticPool:
    """The device-pool surface RaidenKVTransfer reads, over fixed buffers."""

    def __init__(self, buffers):
        self._buffers = buffers
        self.layer_num = len(buffers)
        self.start_layer = 0
        self.page_size = PAGE_SIZE

    def get_fused_kv_buffer(self, layer_id):
        return self._buffers[layer_id]


@unittest.skipUnless(_has_tpu(), "Requires a TPU visible to torch.tpu")
class TestRaidenRoundTrip(CustomTestCase):
    def test_pages_round_trip_on_every_block_major_layout(self):
        """D2H device pages 1,2 to host, H2D them into device pages 4,5: both
        layers arrive element-exact and no other page is touched."""
        from sglang.srt.hardware_backend.tpu.raiden_transfer import RaidenKVTransfer

        for name, shape in BLOCK_MAJOR_SHAPES.items():
            with self.subTest(layout=name, shape=shape):
                device, expected = [], []
                for layer in range(NUM_LAYERS):
                    initial, after = _seeded(shape, layer)
                    device.append(initial.to(DEVICE))
                    expected.append(after)

                transfer = RaidenKVTransfer(
                    _StaticPool(device), num_host_pages=NUM_BLOCKS
                )
                _poll(self, transfer.backup([1, 2], [0, 1]), "D2H")
                _poll(self, transfer.load([0, 1], [4, 5]), "H2D")
                torch.tpu.synchronize()

                for layer in range(NUM_LAYERS):
                    self.assertTrue(
                        torch.equal(device[layer].cpu(), expected[layer]),
                        f"layer {layer} differs after the round trip",
                    )

    def test_host_pool_round_trip_on_fused_pool(self):
        """TpuRaidenHostKVCache over the real fused pool: token indices become
        the right raiden page ids on both sides, and the pool's own buffers are
        the ones raiden registered."""
        from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
        from sglang.srt.hardware_backend.tpu.kv_pool_host import TpuRaidenHostKVCache

        pool = TpuFusedMHATokenToKVPool(
            size=NUM_BLOCKS * PAGE_SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            layer_num=NUM_LAYERS,
            device=DEVICE,
            enable_memory_saver=False,
        )
        expected = []
        for layer in range(NUM_LAYERS):
            buf = pool.get_fused_kv_buffer(layer)
            initial, after = _seeded(buf.shape, layer)
            # Seeded before TpuRaidenHostKVCache registers the buffer.
            buf.copy_(initial.to(DEVICE))
            expected.append(after)
        host = TpuRaidenHostKVCache(
            pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PAGE_SIZE,
            layout="fused_raiden",
        )

        def tokens(pages):
            return torch.tensor(
                [p * PAGE_SIZE + i for p in pages for i in range(PAGE_SIZE)],
                dtype=torch.int64,
            )

        host_indices = host.alloc(2 * PAGE_SIZE)
        self.assertIsNotNone(host_indices)
        _poll(
            self,
            host.backup_from_device_all_layer(
                pool, host_indices, tokens([1, 2]), "raiden"
            ),
            "host backup",
        )
        _poll(
            self, host.load_all_layer(pool, host_indices, tokens([4, 5])), "host load"
        )
        torch.tpu.synchronize()

        for layer in range(NUM_LAYERS):
            self.assertTrue(
                torch.equal(pool.get_fused_kv_buffer(layer).cpu(), expected[layer]),
                f"layer {layer} differs after the host-pool round trip",
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
