# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the DeepSeek-V4.1 Engram and warmup inputs of the direct tracer.

The direct tracer rebuilds ForwardBatch from its dataclass fields only, so the
host-staged Engram rows (class attributes the V4 compile policy sets) reach the
traced graph on the attention metadata. Guards that the Engram hooks read the
batch first and the metadata inside a trace, and that the policy's compiled
call carries the rows onto the metadata.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import compile_policy, engram_host


def _staged():
    ids = torch.zeros(3, 2, 2, dtype=torch.int32)
    rows = ((torch.zeros(6, 4), torch.zeros(6, 1)),)
    return ids, rows


def _context(metadata):
    backend = SimpleNamespace(forward_metadata=metadata)
    return SimpleNamespace(attn_backend=backend)


class TestStagedInputs(CustomTestCase):
    def test_batch_rows_win(self):
        ids, rows = _staged()
        batch = SimpleNamespace(tpu_engram_local_ids=ids, tpu_engram_staged=rows)
        self.assertEqual(engram_host.staged_inputs(batch), (ids, rows))

    def test_traced_batch_reads_the_metadata(self):
        carried = _staged()
        rebuilt = SimpleNamespace(tpu_engram_local_ids=None, tpu_engram_staged=None)
        with (
            mock.patch(
                "sglang.srt.model_executor.forward_context.has_forward_context",
                return_value=True,
            ),
            mock.patch(
                "sglang.srt.model_executor.forward_context.get_forward_context",
                return_value=_context(SimpleNamespace(engram=carried)),
            ),
        ):
            self.assertIs(engram_host.staged_inputs(rebuilt), carried)

    def test_nothing_staged(self):
        rebuilt = SimpleNamespace(tpu_engram_local_ids=None, tpu_engram_staged=None)
        with mock.patch(
            "sglang.srt.model_executor.forward_context.has_forward_context",
            return_value=False,
        ):
            self.assertEqual(engram_host.staged_inputs(rebuilt), (None, None))


class TestHasherHook(CustomTestCase):
    def test_host_hashing_ignores_the_carried_ids(self):
        from sglang_tpu.models.deepseek_v41 import hooks

        ids, rows = _staged()
        batch = SimpleNamespace(tpu_engram_local_ids=None, tpu_engram_staged=None)
        original = mock.Mock(return_value="hashed")
        with mock.patch(
            "sglang.srt.model_executor.forward_context.get_forward_context",
            return_value=_context(SimpleNamespace(engram=(ids, rows))),
        ):
            out = hooks.tpu_dsv41_hasher_forward(
                original, "hasher", torch.zeros(3, dtype=torch.int64), batch
            )
        self.assertEqual(out, "hashed")
        original.assert_called_once()


class TestCarryEngram(CustomTestCase):
    def _call(self, batch, metadata):
        calls = []
        run = compile_policy._carrying_engram(lambda *args: calls.append(args) or "out")
        with mock.patch.object(
            compile_policy, "get_forward_context", return_value=_context(metadata)
        ):
            return run("ids", "pos", batch, "extra"), calls

    def test_puts_the_staged_rows_on_the_metadata(self):
        ids, rows = _staged()
        batch = SimpleNamespace(tpu_engram_local_ids=ids, tpu_engram_staged=list(rows))
        metadata = SimpleNamespace(engram=None)
        out, calls = self._call(batch, metadata)
        self.assertEqual(out, "out")
        self.assertEqual(calls, [("ids", "pos", batch, "extra")])
        self.assertIs(metadata.engram[0], ids)
        self.assertIsInstance(metadata.engram[1], tuple)
        self.assertIs(metadata.engram[1][0][0], rows[0][0])

    def test_needs_built_metadata(self):
        ids, rows = _staged()
        batch = SimpleNamespace(tpu_engram_local_ids=ids, tpu_engram_staged=rows)
        with self.assertRaises(RuntimeError):
            self._call(batch, None)


if __name__ == "__main__":
    unittest.main()
