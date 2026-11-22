# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the scheduler hooks of decoder SWA bounded replay on the
DeepSeek-V4.1 TPU backend (decoder_replay_sync.py): the per-rank decoder row
MLP-sync column and the fill lengths carried to the ForwardBatch."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.forward_mode import ForwardMode  # noqa: E402
from sglang.srt.plugins.hook_registry import _wrap_fn  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang_tpu.models.deepseek_v41 import decoder_replay_sync as sync  # noqa: E402

_SYNC_TARGETS = {
    "prepare_mlp_sync_batch_raw",
    "_get_local_tensor",
    "_get_fallback_tensor",
    "all_gather",
    "finalize_local",
    "_update_gather_batch",
}


@contextlib.contextmanager
def _sync_hooks():
    """The MLP-sync hooks of decoder_replay_sync, applied for one test."""
    with contextlib.ExitStack() as stack:
        for target, hook, kind in sync.hook_specs():
            owner_path, name = target.rsplit(".", 1)
            if name not in _SYNC_TARGETS:
                continue
            owner = (
                dp_attn.MLPSyncBatchInfo
                if owner_path.endswith("MLPSyncBatchInfo")
                else dp_attn
            )
            original = getattr(owner, name)
            stack.enter_context(
                patch.object(owner, name, _wrap_fn(original, hook, kind))
            )
        yield


def _info(num_dp_ranks=2):
    return dp_attn.MLPSyncBatchInfo(
        num_dp_ranks=num_dp_ranks,
        tp_size=1,
        cp_size=1,
        num_tokens=3,
        num_tokens_for_logprob=1,
        can_run_decode_cuda_graph=False,
        can_run_draft_cuda_graph=False,
        can_run_prefill_cuda_graph=False,
        is_extend_in_batch=True,
        local_can_run_tbo=False,
        local_forward_mode=ForwardMode.EXTEND.value,
    )


class TestDecoderRowsColumn(CustomTestCase):
    def _sync(self, attn_backend):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=lambda: 4,
            spec_info=None,
            reqs=[],
            seq_lens_cpu=torch.tensor([5, 6, 7, 8]),
        )
        tbo = Mock()
        tbo.prepare_all_gather.return_value = (True, ForwardMode.DECODE.value)
        tbo.compute_output.return_value = (0, ForwardMode.DECODE)
        runner = SimpleNamespace(
            prefill_cuda_graph_runner=None,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            model_config=object(),
        )
        if attn_backend is not None:
            runner.attn_backend = attn_backend
        with (
            _sync_hooks(),
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            patch.object(dp_attn, "TboDPAttentionPreparer", return_value=tbo),
            patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
            patch.object(
                dp_attn,
                "get_parallel",
                return_value=SimpleNamespace(
                    num_dp_ranks=1,
                    attn_tp_size=1,
                    attn_cp_size=1,
                    tp_group=SimpleNamespace(
                        device_group=object(), device="cpu", cpu_group=object()
                    ),
                ),
            ),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
        ):
            return dp_attn.prepare_mlp_sync_batch_raw(
                batch,
                model_runner=runner,
                get_idle_batch=Mock(),
                disable_cuda_graph=False,
                require_mlp_tp_gather=False,
                disable_overlap_schedule=True,
                offload_tags=set(),
            )

    def test_backend_count_rides_the_sync(self):
        calls = []

        def rows(batch, *, folds):
            calls.append((batch.batch_size(), folds))
            return 4

        backend = SimpleNamespace(
            decoder_replay=True,
            folds_encoder_swa_replay=False,
            scheduled_decoder_rows=rows,
        )
        result = self._sync(backend)
        self.assertEqual(result.global_decoder_rows, [4])
        self.assertEqual(calls, [(4, False)])
        self.assertIsNone(sync._pending_rows())

    def test_no_column_without_a_decoder_replay_backend(self):
        for backend in (None, SimpleNamespace(decoder_replay=False)):
            result = self._sync(backend)
            self.assertIsNone(getattr(result, "global_decoder_rows", None))

    def test_payload_width(self):
        """One more column only while a decoder-replay sync runs; idle slots 0."""
        width = _info()._get_local_tensor("cpu").numel()
        with _sync_hooks():
            info = _info()
            self.assertEqual(info._get_local_tensor("cpu").numel(), width)
            sync._SYNC.rows = 7
            try:
                local = info._get_local_tensor("cpu")
                fallback = info._get_fallback_tensor("cpu")
            finally:
                sync._SYNC.rows = None
        self.assertEqual((local.numel(), fallback.numel()), (width + 1, width + 1))
        self.assertEqual((local[-1].item(), fallback[-1].item()), (7, 0))

    def test_gathered_column(self):
        info = _info()
        info.tp0_info_cpu = torch.tensor([[0] * 10 + [5], [0] * 10 + [9]])
        sync._SYNC.rows = 5
        try:
            sync.gathered_decoder_rows(None, info)
        finally:
            sync._SYNC.rows = None
        self.assertEqual(info.global_decoder_rows, [5, 9])
        sync.gathered_decoder_rows(None, info)
        self.assertIsNone(info.global_decoder_rows)


class TestFillLens(CustomTestCase):
    def test_extend_records_fill_lens_and_decode_conversion_drops_them(self):
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(full_untruncated_fill_ids=[0] * 9),
                SimpleNamespace(full_untruncated_fill_ids=[0] * 4),
            ]
        )
        features = SimpleNamespace(enable_decoder_swa_bounded_replay=True)
        with patch(
            "sglang.srt.runtime_context.get_exec",
            return_value=SimpleNamespace(features=features),
        ):
            sync.record_fill_lens(None, batch)
        self.assertEqual(batch.decoder_fill_lens, [9, 4])
        copied = sync.copy_decoder_fields(SimpleNamespace(), batch)
        self.assertEqual(copied.decoder_fill_lens, [9, 4])
        sync.drop_fill_lens(None, batch)
        self.assertIsNone(batch.decoder_fill_lens)

    def test_flag_off_records_nothing(self):
        batch = SimpleNamespace(reqs=[])
        features = SimpleNamespace(enable_decoder_swa_bounded_replay=False)
        with patch(
            "sglang.srt.runtime_context.get_exec",
            return_value=SimpleNamespace(features=features),
        ):
            sync.record_fill_lens(None, batch)
        self.assertFalse(hasattr(batch, "decoder_fill_lens"))

    def test_forward_batch_takes_extend_fill_lens_only(self):
        class FB:
            pass

        for mode, expected in ((ForwardMode.EXTEND, [9]), (ForwardMode.DECODE, None)):
            batch = SimpleNamespace(
                forward_mode=mode, decoder_fill_lens=[9], global_decoder_rows=[3, 1]
            )
            fb = sync.forward_batch_decoder_fields(SimpleNamespace(), FB, batch, None)
            self.assertEqual(fb.decoder_fill_lens, expected)
            self.assertEqual(fb.global_decoder_rows, [3, 1])
            fb = sync.forward_batch_decoder_fields(SimpleNamespace(), batch=batch)
            self.assertEqual(fb.decoder_fill_lens, expected)


class TestLoadBatchCarry(CustomTestCase):
    def test_rebuilt_forward_batch_keeps_the_decoder_fields(self):
        """EagerRunner.load_batch rebuilds the ForwardBatch from its dataclass
        fields (dataclasses.replace); the decoder attributes must survive, or
        the compile policy falls back to the token bucket under DP."""
        import dataclasses

        @dataclasses.dataclass
        class FB:
            input_ids: object = None

        fb = FB()
        fb.decoder_fill_lens, fb.global_decoder_rows = [9], [3, 1]
        rebuilt = dataclasses.replace(fb)
        self.assertFalse(hasattr(rebuilt, "global_decoder_rows"))
        out = sync.carry_decoder_fields(rebuilt, object(), fb)
        self.assertIs(out, rebuilt)
        self.assertEqual(
            (out.decoder_fill_lens, out.global_decoder_rows), ([9], [3, 1])
        )
        self.assertFalse(hasattr(out, "decoder_rows_bucket"))
        # The same object passes through.
        self.assertIs(sync.carry_decoder_fields(fb, object(), fb), fb)

    def test_load_batch_hook_is_registered(self):
        from sglang.srt.plugins.hook_registry import HookType

        self.assertIn(
            (
                "sglang.srt.model_executor.runner.eager_runner.EagerRunner.load_batch",
                sync.carry_decoder_fields,
                HookType.AFTER,
            ),
            sync.hook_specs(),
        )


class TestHookTargets(CustomTestCase):
    def test_scatter_zero_fill_is_registered(self):
        from sglang.srt.plugins.hook_registry import HookType
        from sglang_tpu.models.deepseek_v41 import hooks
        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            scatter_decoder_rows,
        )

        spec = (
            "sglang.srt.models.deepseek_v4._scatter_tail_rows",
            scatter_decoder_rows,
            HookType.AROUND,
        )
        self.assertIn(spec, sync.hook_specs())
        self.assertIn(spec, hooks.hook_specs())

    def test_every_target_resolves(self):
        import importlib

        for target, _hook, _kind in sync.hook_specs():
            parts = target.split(".")
            for i in range(len(parts) - 1, 0, -1):
                try:
                    obj = importlib.import_module(".".join(parts[:i]))
                except ModuleNotFoundError:
                    continue
                for attr in parts[i:]:
                    obj = getattr(obj, attr)
                break
            else:
                self.fail(target)


if __name__ == "__main__":
    unittest.main()
