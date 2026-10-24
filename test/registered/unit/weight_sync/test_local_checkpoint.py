"""local_checkpoint.pull: the host-side half of disk-delta weight sync (/pull_weights).

Builds a tiny safetensors checkpoint, publishes full and delta versions the way the
trainer does (zstd-compressed "overwrite" or "xor" deltas plus a checksum of each
tensor's new bytes), and checks that pull() reproduces every version byte for byte.
"""

import json
import os
import struct
import tempfile
import unittest
import zlib

import numpy as np
import zstandard

from sglang.srt.weight_sync import local_checkpoint
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _write_safetensors(path: str, tensors: dict, metadata: dict | None = None) -> None:
    """Minimal safetensors writer: every tensor stored as raw U8 bytes."""
    header, offset, blobs = {}, 0, []
    for name, data in tensors.items():
        raw = np.ascontiguousarray(data).view(np.uint8).reshape(-1).tobytes()
        header[name] = {"dtype": "U8", "shape": [len(raw)], "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    if metadata:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for raw in blobs:
            f.write(raw)


def _read_safetensors(path: str) -> dict:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        data = f.read()
    return {
        name: np.frombuffer(data[info["data_offsets"][0] : info["data_offsets"][1]], dtype=np.uint8)
        for name, info in header.items()
        if name != "__metadata__"
    }


def _adler32(buf: np.ndarray) -> str:
    return f"{zlib.adler32(buf.tobytes(), 1):08x}"


def _encode(encoding: str, old: np.ndarray, new: np.ndarray) -> bytes:
    """Trainer-side encodings (miles/utils/disk_delta.py): the bytes local_checkpoint decodes."""
    if encoding == "xor":
        payload = (old ^ new).tobytes()
    else:  # overwrite: changed count (u4), positions (u4 each), then the new bytes
        positions = np.flatnonzero(old != new).astype("<u4")
        payload = struct.pack("<I", positions.size) + positions.tobytes() + new[positions].tobytes()
    return zstandard.ZstdCompressor().compress(payload)


class TestLocalCheckpointPull(CustomTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = self._tmp.name
        self.base_dir = os.path.join(root, "base")
        self.source_dir = os.path.join(root, "published")
        self.local_dir = os.path.join(root, "local")
        os.makedirs(self.base_dir)
        os.makedirs(self.source_dir)
        rng = np.random.default_rng(0)
        # two shards, as in a real multi-file checkpoint
        self.state = {
            "model-00001-of-00002.safetensors": {
                "a.weight": rng.integers(0, 256, 4096, dtype=np.uint8),
                "b.weight": rng.integers(0, 256, 1000, dtype=np.uint8),
            },
            "model-00002-of-00002.safetensors": {
                "c.weight": rng.integers(0, 256, 2048, dtype=np.uint8),
            },
        }
        for fname, tensors in self.state.items():
            _write_safetensors(os.path.join(self.base_dir, fname), tensors)
        self.rng = rng

    def tearDown(self):
        self._tmp.cleanup()

    def _version_dir(self, version: int) -> str:
        path = os.path.join(self.source_dir, f"weight_v{version:06d}")
        os.makedirs(path, exist_ok=True)
        return path

    def _publish_delta(self, version: int, encoding: str, corrupt: str | None = None) -> None:
        """Change a few bytes of a.weight and c.weight; publish them as one delta version."""
        delta, checksums = {}, {}
        for fname, tensors in self.state.items():
            for name in ("a.weight", "c.weight"):
                if name not in tensors:
                    continue
                old = tensors[name]
                new = old.copy()
                idx = self.rng.choice(new.size, 37, replace=False)
                new[idx] = self.rng.integers(0, 256, idx.size, dtype=np.uint8)
                delta[name] = np.frombuffer(_encode(encoding, old, new), dtype=np.uint8)
                checksums[name] = "00000000" if name == corrupt else _adler32(new)
                tensors[name] = new
        vdir = self._version_dir(version)
        _write_safetensors(os.path.join(vdir, "model-00000-of-00001.safetensors"), delta, checksums)
        index = {
            "metadata": {
                "version": f"{version:06d}",
                "base_version": f"{version - 1:06d}",
                "delta_encoding": encoding,
                "compression_format": "zstd",
                "checksum_format": "adler32",
            },
            "weight_map": {name: "model-00000-of-00001.safetensors" for name in delta},
        }
        with open(os.path.join(vdir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f)

    def _publish_full(self, version: int) -> None:
        """Publish an ordinary checkpoint (new random weights) as a full version."""
        vdir = self._version_dir(version)
        for fname, tensors in self.state.items():
            for name in tensors:
                tensors[name] = self.rng.integers(0, 256, tensors[name].size, dtype=np.uint8)
            _write_safetensors(os.path.join(vdir, fname), tensors)

    def _pull(self, version: int) -> None:
        local_checkpoint.pull(
            local_checkpoint_dir=self.local_dir,
            base_dir=self.base_dir,
            source_dir=self.source_dir,
            target_version=version,
        )

    def _assert_local_matches_state(self) -> None:
        for fname, tensors in self.state.items():
            local = _read_safetensors(os.path.join(self.local_dir, fname))
            for name, expected in tensors.items():
                np.testing.assert_array_equal(local[name], expected, err_msg=f"{fname}:{name}")

    def _applied_version(self):
        return local_checkpoint._read_applied_version(self.local_dir)

    def test_seed_from_base_then_apply_overwrite_and_xor_deltas(self):
        self._pull(0)
        self.assertEqual(self._applied_version(), 0)
        self._assert_local_matches_state()

        self._publish_delta(1, "overwrite")
        self._publish_delta(2, "xor")
        self._pull(2)  # applies v1 then v2 in order
        self.assertEqual(self._applied_version(), 2)
        self._assert_local_matches_state()

        self._pull(2)  # already there: a no-op
        self.assertEqual(self._applied_version(), 2)
        self._assert_local_matches_state()

    def test_fresh_host_pulls_a_delta_chain_from_the_base(self):
        self._publish_delta(1, "overwrite")
        self._publish_delta(2, "overwrite")
        self._pull(2)  # no local checkpoint yet: seeds from base_dir, then applies both deltas
        self.assertEqual(self._applied_version(), 2)
        self._assert_local_matches_state()

    def test_full_version_resets_then_deltas_chain_on_it(self):
        self._pull(0)
        self._publish_delta(1, "overwrite")
        self._publish_full(2)
        self._publish_delta(3, "overwrite")
        self._pull(3)  # skips v1: resets to the full v2, then applies v3
        self.assertEqual(self._applied_version(), 3)
        self._assert_local_matches_state()

    def test_checksum_mismatch_fails_and_keeps_the_applied_version(self):
        self._pull(0)
        self._publish_delta(1, "overwrite", corrupt="c.weight")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self._pull(1)
        self.assertEqual(self._applied_version(), 0)


if __name__ == "__main__":
    unittest.main()
