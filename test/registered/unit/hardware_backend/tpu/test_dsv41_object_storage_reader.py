# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for reading DeepSeek-V4.1 Engram rows from object storage.

The Engram row reader parses each shard's safetensors header before reading this
rank's Engram rows. Guards that a gs:// shard's header is read through RunAI's
ranged reads and matches what a local open() of the same bytes gives.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import os
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import engram_row_reader as reader


class TestSafetensorsHeader(CustomTestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "model-00001.safetensors")
        save_file(
            {
                "model.layers.1.engram.embed.weight": torch.zeros(16, 4),
                "model.layers.1.engram.embed.scale": torch.ones(16, 1),
            },
            self.path,
        )
        with open(self.path, "rb") as f:
            self.raw = f.read()

    def _ranged(self, path, offset, out):
        self.assertEqual(path, "gs://bucket/ckpt/model-00001.safetensors")
        out.copy_(
            torch.frombuffer(
                bytearray(self.raw[offset : offset + out.numel()]), dtype=torch.uint8
            )
        )

    def test_object_storage_header_matches_local(self):
        local = reader._read_safetensors_header(self.path)
        with mock.patch.object(reader, "_read_range", side_effect=self._ranged):
            remote = reader._read_safetensors_header(
                "gs://bucket/ckpt/model-00001.safetensors"
            )
        self.assertEqual(remote, local)
        self.assertIn("model.layers.1.engram.embed.weight", local[1])


class TestEngramRowRanges(CustomTestCase):
    def test_reads_the_rows_engram_embedding_holds(self):
        """Each rank reads its EngramEmbedding's hash-column-aligned rows, not an
        even row split, or loading rejects every block as outside the rank."""
        from sglang.srt.layers.engram import EngramLayout

        # Layer 1: column bounds (0, 5, 12, 23, 36); layer 4: (0, 17, 36, 59, 88).
        layout = EngramLayout(
            max_ngram_size=3,
            layer_ids=(1, 4),
            num_embeddings=(36, 88),
            primes=(((5, 7), (11, 13)), ((17, 19), (23, 29))),
            n_heads=2,
            head_dim=4,
        )
        names = ("layers.1.engram.embed.weight", "layers.4.engram.embed.scale")
        ranges = reader.engram_row_ranges(names, layout=layout, tp_rank=1, tp_size=2)
        self.assertEqual(ranges, {names[0]: (12, 36), names[1]: (36, 88)})

        path = os.path.join(tempfile.mkdtemp(), "model-00001.safetensors")
        table = torch.arange(36, dtype=torch.float32)[:, None].repeat(1, 4)
        save_file({names[0]: table}, path)
        with open(path, "rb") as f:
            raw = f.read()

        def read_range(_, offset, out):
            data = bytearray(raw[offset : offset + out.numel()])
            out.copy_(torch.frombuffer(data, dtype=torch.uint8))

        with mock.patch.object(reader, "_read_range", read_range):
            blocks = list(
                reader.iter_row_tables(
                    shard_of={names[0]: path}, row_ranges={names[0]: ranges[names[0]]}
                )
            )
        self.assertEqual(len(blocks), 1)
        name, rows = blocks[0]
        self.assertEqual(name, names[0])
        self.assertEqual(getattr(rows, reader.ROW_START_ATTR), 12)
        self.assertTrue(torch.equal(rows, table[12:36]))


if __name__ == "__main__":
    unittest.main()
