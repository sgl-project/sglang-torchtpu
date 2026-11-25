# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for reading DeepSeek-V4.1 Engram rows.

Guards that a gs:// shard's header read through RunAI matches a local open();
that shared host tables are filled in place by a child process that starts when
the loader plans its stream, writes exactly this rank's rows, never leaves a
table silently incomplete, dies with its parent and cleans up on every exit
path; and that other layouts get the same rows as tagged blocks when consumed.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import mmap
import os
import signal
import subprocess
import sys
import tempfile
import time
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


def _engram_dirs() -> set[str]:
    return {d for d in os.listdir("/dev/shm") if d.startswith("dsv41-engram-")}


class _Shard(CustomTestCase):
    """A local shard with one Engram table: 40 rows x 32 B weight, 40 x 1 B scale."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "model-00047.safetensors")
        self.weight = (
            torch.arange(40 * 32, dtype=torch.int32)
            .remainder(251)
            .to(torch.uint8)
            .view(40, 32)
        )
        self.scale = torch.arange(40, dtype=torch.uint8).view(40, 1)
        self.names = ("layers.1.engram.embed.weight", "layers.1.engram.embed.scale")
        save_file(dict(zip(self.names, (self.weight, self.scale))), self.path)
        self.shard_of = dict.fromkeys(self.names, self.path)
        self.ranges = dict.fromkeys(self.names, (10, 30))
        self.dirs_before = _engram_dirs()


class TestSharedTables(_Shard):
    """The child-process path: rows land in the table's memfd at its offsets."""

    def _table(self):
        # A per_node host table stand-in: one memfd, weight rows then scale rows.
        fd = os.memfd_create("test_engram", 0)
        nbytes = 40 * 32 + 40
        os.ftruncate(fd, nbytes)
        self.addCleanup(os.close, fd)
        raw = torch.frombuffer(
            mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED), dtype=torch.uint8
        )
        weight, scale = raw[: 40 * 32].view(40, 32), raw[40 * 32 :].view(40, 1)
        shared = {
            name: reader.SharedRows(
                fd=fd,
                dest_offset=t[10:30].data_ptr() - raw.data_ptr(),
                dest_row_bytes=t.stride(0),
                marker=t[10:10],
            )
            for name, t in zip(self.names, (weight, scale))
        }
        return weight, scale, shared

    def _rows(self, shared, shard_of=None):
        return reader.iter_row_tables(
            shard_of=shard_of or self.shard_of, ranges=self.ranges, shared=shared
        )

    def test_child_writes_this_ranks_rows_in_place(self):
        weight, scale, shared = self._table()
        got = list(self._rows(shared))
        self.assertTrue(torch.equal(weight[10:30], self.weight[10:30]))
        self.assertTrue(torch.equal(scale[10:30], self.scale[10:30]))
        self.assertEqual(int(weight[:10].sum()) + int(weight[30:].sum()), 0)
        # One empty block per table, tagged with the first row, for load_local_rows.
        self.assertEqual(sorted(n for n, _ in got), sorted(self.names))
        for _, block in got:
            self.assertEqual(block.shape[0], 0)
            self.assertEqual(getattr(block, reader.ROW_START_ATTR), 10)
        self.assertEqual(_engram_dirs(), self.dirs_before)

    def test_rows_are_written_before_the_iterator_is_consumed(self):
        weight, _, shared = self._table()
        rows = self._rows(shared)
        # The main stream would run here: nothing has been consumed yet.
        deadline = time.monotonic() + 30
        while not torch.equal(weight[10:30], self.weight[10:30]):
            self.assertLess(
                time.monotonic(), deadline, "the child never wrote the rows"
            )
            time.sleep(0.05)
        list(rows)

    def test_child_failure_raises_and_cleans_up(self):
        _, _, shared = self._table()
        bad = dict.fromkeys(self.names, os.path.join(self.dir, "missing.safetensors"))
        with self.assertRaisesRegex(
            RuntimeError, r"(?s)exited with 1.*missing\.safetensors"
        ):
            list(self._rows(shared, shard_of=bad))
        self.assertEqual(_engram_dirs(), self.dirs_before)

    def test_wrong_row_size_fails_without_writing(self):
        weight, _, shared = self._table()
        w = shared[self.names[0]]
        shared[self.names[0]] = reader.SharedRows(
            fd=w.fd, dest_offset=w.dest_offset, dest_row_bytes=16, marker=w.marker
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint rows are 32 B"):
            list(self._rows(shared))
        self.assertEqual(int(weight.sum()), 0)

    def test_hung_child_times_out_and_is_killed(self):
        _, _, shared = self._table()
        script = os.path.join(self.dir, "hang.py")
        with open(script, "w") as f:
            f.write("import time\ntime.sleep(600)\n")
        with (
            mock.patch.object(reader, "_CHILD_SCRIPT", script),
            mock.patch.object(reader, "_CHILD_DEADLINE_S", 0.5),
            mock.patch.object(reader, "_CHILD_GRACE_S", 0.5),
        ):
            with self.assertRaises(TimeoutError):
                list(self._rows(shared))
        self.assertEqual(_engram_dirs(), self.dirs_before)

    def test_child_dies_with_its_parent(self):
        # A parent forks a child that arms the death signal and blocks on a FIFO;
        # SIGKILL on the parent must take the child down.
        fifo = os.path.join(self.dir, "fifo")
        os.mkfifo(fifo)
        code = (
            "import importlib.util, os, time\n"
            f"spec = importlib.util.spec_from_file_location('c', {str(reader._CHILD_SCRIPT)!r})\n"
            "c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    c._die_with_parent(os.getppid())\n"
            f"    open({fifo!r}).read()\n"
            "    os._exit(0)\n"
            "print(pid, flush=True)\n"
            "time.sleep(600)\n"
        )
        parent = subprocess.Popen(
            [sys.executable, "-I", "-c", code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
        )
        child_pid = int(parent.stdout.readline())
        time.sleep(0.5)
        parent.send_signal(signal.SIGKILL)
        parent.wait()
        deadline = time.monotonic() + 10
        while os.path.exists(f"/proc/{child_pid}"):
            if open(f"/proc/{child_pid}/stat").read().split()[2] == "Z":
                break
            self.assertLess(time.monotonic(), deadline, "the child outlived its parent")
            time.sleep(0.1)


class TestSerialTables(_Shard):
    """Layouts without a shared host table: tagged blocks, read when consumed."""

    def _ranged(self, path, offset, out):
        raw = open(self.path, "rb").read()
        out.copy_(
            torch.frombuffer(
                bytearray(raw[offset : offset + out.numel()]), dtype=torch.uint8
            )
        )

    def test_blocks_cover_this_ranks_rows(self):
        # 32 B rows: 3 weight rows a block.
        with (
            mock.patch.object(reader, "_read_range", side_effect=self._ranged),
            mock.patch.object(reader, "_ROW_BLOCK_BYTES", 96),
        ):
            got = list(
                reader.iter_row_tables(shard_of=self.shard_of, ranges=self.ranges)
            )
        for name, full in zip(self.names, (self.weight, self.scale)):
            blocks = [b for n, b in got if n == name]
            self.assertEqual(getattr(blocks[0], reader.ROW_START_ATTR), 10)
            self.assertTrue(torch.equal(torch.cat(blocks), full[10:30]))
        self.assertGreater(len([n for n, _ in got if n == self.names[0]]), 1)

    def test_no_read_before_consumption(self):
        # Serial reads start when the loader reaches them, after the main stream,
        # so no second Run:AI streamer runs beside it in this process.
        with mock.patch.object(reader, "_read_range", side_effect=self._ranged) as read:
            rows = reader.iter_row_tables(shard_of=self.shard_of, ranges=self.ranges)
            self.assertEqual(read.call_count, 0)
            list(rows)
            self.assertGreater(read.call_count, 0)


if __name__ == "__main__":
    unittest.main()
