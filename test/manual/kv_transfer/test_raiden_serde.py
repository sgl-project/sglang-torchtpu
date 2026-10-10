"""Unit tests for the raiden disaggregation wire format and page mapping.

These cover the pure byte-protocol helpers (no TPU / torch_tpu / zmq required),
so they run anywhere. The on-device end-to-end path is validated separately on
a TPU VM.

Run: python -m pytest test/manual/kv_transfer/test_raiden_serde.py
"""

import ctypes
import unittest

import numpy as np

from sglang_tpu.disaggregation.raiden.serde import (
    MessageTag,
    read_aux_blob,
    write_aux_blob,
)


def kv_to_page_indices(kv_indices, page_size):
    # Mirror of sglang.srt.mem_cache.common.kv_to_page_indices.
    if page_size == 1:
        return kv_indices
    return kv_indices[::page_size] // page_size


class TestRaidenSerde(unittest.TestCase):
    def test_headers_distinct(self):
        self.assertNotEqual(MessageTag.PULL_REQ, MessageTag.BLOCKS)

    def test_page_indices_preserve_token_order(self):
        # 3 full pages (page_size=4); page ids must stay in token order, NOT be
        # sorted -- prefill and decode each derive the list the same way, so the
        # i-th remote page maps to the i-th local page.
        toks = np.array([80, 81, 82, 83, 32, 33, 34, 35, 8, 9, 10, 11], dtype=np.int32)
        self.assertEqual(kv_to_page_indices(toks, 4).tolist(), [20, 8, 2])

    def test_aux_blob_roundtrip(self):
        # Two host aux buffers with different item lengths; copy slot @src_idx on
        # the "prefill" side into slot @dst_idx on the "decode" side.
        item_lens = [16, 8]
        src_bufs = [(ctypes.c_byte * (il * 8))() for il in item_lens]
        dst_bufs = [(ctypes.c_byte * (il * 8))() for il in item_lens]
        src_ptrs = [ctypes.addressof(b) for b in src_bufs]
        dst_ptrs = [ctypes.addressof(b) for b in dst_bufs]

        src_idx, dst_idx = 2, 5
        payload0, payload1 = bytes(range(16)), bytes(range(100, 108))
        ctypes.memmove(src_ptrs[0] + 16 * src_idx, payload0, 16)
        ctypes.memmove(src_ptrs[1] + 8 * src_idx, payload1, 8)

        blob = read_aux_blob(src_ptrs, item_lens, src_idx)
        write_aux_blob(dst_ptrs, item_lens, dst_idx, blob)

        got0 = bytes((ctypes.c_byte * 16).from_address(dst_ptrs[0] + 16 * dst_idx))
        got1 = bytes((ctypes.c_byte * 8).from_address(dst_ptrs[1] + 8 * dst_idx))
        self.assertEqual(got0, payload0)
        self.assertEqual(got1, payload1)
        # Keep buffers alive until after the reads.
        del src_bufs, dst_bufs


if __name__ == "__main__":
    unittest.main()
