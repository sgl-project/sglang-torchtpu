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


if __name__ == "__main__":
    unittest.main()
