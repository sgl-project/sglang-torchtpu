"""Host staging, scheduler metadata ownership, and TPU forward padding."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.srt.environ import envs
from sglang.srt.model_executor import forward_batch_info
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import patches
from sglang_tpu.compilation import compiled_forward_shim as shim
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.compilation.compiled_forward_shim import (
    _pad_tensor_on_host,
)
from sglang_tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPositionOnHost(CustomTestCase):
    def test_compute_position_torch_cpu_staged(self):
        cases = (
            ([7], [3], [7, 8, 9], [0]),
            ([2, 10, 5], [2, 0, 3], [2, 3, 5, 6, 7], [0, 2, 2]),
            ([0, 9], [0, 0], [], [0, 0]),
        )
        for dtype in (torch.int32, torch.int64):
            for prefixes, lengths, expected_positions, expected_starts in cases:
                with self.subTest(dtype=dtype, lengths=lengths):
                    device = torch.device("meta")
                    prefix_tensor = torch.empty(
                        len(prefixes), dtype=dtype, device=device
                    )
                    length_tensor = torch.empty(
                        len(lengths), dtype=dtype, device=device
                    )
                    host_values = {
                        id(prefix_tensor): prefixes,
                        id(length_tensor): lengths,
                    }
                    reads, uploads = [], []

                    def read_lengths(tensor):
                        reads.append(id(tensor))
                        return host_values[id(tensor)]

                    class CpuStaging(TorchDispatchMode):
                        def __torch_dispatch__(mode, func, types, args=(), kwargs=None):
                            kwargs = kwargs or {}
                            result = func(*args, **kwargs)
                            if (
                                func == torch.ops.aten._to_copy.default
                                and kwargs.get("device") == device
                            ):
                                self.assertEqual(args[0].device.type, "cpu")
                                uploads.append(args[0])
                            elif isinstance(result, torch.Tensor):
                                self.assertEqual(result.device.type, "cpu", str(func))
                            return result

                    # Simulate only the two input readbacks. Real meta tensors
                    # cannot supply values to any device-side offset calculation.
                    with (
                        patch.object(patches, "_compute_position_torch_patched", False),
                        patch.object(
                            forward_batch_info,
                            "compute_position_torch",
                            forward_batch_info.compute_position_torch,
                        ),
                        patch.object(torch.Tensor, "tolist", read_lengths),
                        CpuStaging(),
                        torch.device("meta"),
                    ):
                        patches._patch_compute_position_torch()
                        positions, starts = forward_batch_info.compute_position_torch(
                            prefix_tensor, length_tensor
                        )

                    self.assertEqual(reads, [id(prefix_tensor), id(length_tensor)])
                    self.assertEqual(
                        (positions.device, starts.device), (device, device)
                    )
                    self.assertEqual(len(uploads), 2)
                    self.assertEqual(uploads[0].dtype, torch.int64)
                    self.assertEqual(uploads[1].dtype, dtype)
                    self.assertEqual(uploads[0].tolist(), expected_positions)
                    self.assertEqual(uploads[1].tolist(), expected_starts)


class TestPadTensorOnHost(CustomTestCase):
    """Runs the host path on CPU tensors. The TPU copies need a real TPU."""

    def test_bucketed_metadata_is_returned_without_copying(self):
        value = torch.arange(6).reshape(2, 3)

        self.assertIs(_pad_tensor_on_host(value, 2), value)
        self.assertIs(_pad_tensor_on_host(value, 3, dim=-1), value)

    def test_dim0_preserves_values_and_zero_fills(self):
        value = torch.arange(6, dtype=torch.int64).reshape(3, 2)

        result = _pad_tensor_on_host(value, 5)

        self.assertEqual(result.shape, (5, 2))
        self.assertTrue(torch.equal(result[:3], value))
        self.assertTrue(torch.all(result[3:] == 0))

    def test_last_dim_preserves_values_and_zero_fills(self):
        value = torch.arange(6, dtype=torch.int64).reshape(2, 3)

        result = _pad_tensor_on_host(value, 5, dim=-1)

        self.assertEqual(result.shape, (2, 5))
        self.assertTrue(torch.equal(result[:, :3], value))
        self.assertTrue(torch.all(result[:, 3:] == 0))


class TestHostPadFieldSelection(CustomTestCase):
    """Host staging is opt-in; device token padding retains its existing policy."""

    def setUp(self):
        super().setUp()
        env_patch = patch.dict(os.environ)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        os.environ.pop(envs.SGLANG_TPU_HOST_PADDING.name, None)
        override = get_context().override_server_args(
            page_size=1, dp_size=1, disaggregation_mode="null", dllm_algorithm=None
        )
        override.install()
        self.addCleanup(override.restore)
        dp_patch = patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
            return_value=False,
        )
        dp_patch.start()
        self.addCleanup(dp_patch.stop)

    def _extend_batch(self):
        return SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=torch.arange(5, dtype=torch.int64),
            positions=torch.tensor([11, 12, 13, 50, 51]),
            out_cache_loc=torch.arange(5, dtype=torch.int32),
            mrope_positions=torch.arange(15).reshape(3, 5),
            seq_lens=torch.tensor([14, 52], dtype=torch.int32),
            extend_seq_lens=torch.tensor([3, 2], dtype=torch.int32),
            extend_prefix_lens=torch.tensor([11, 50], dtype=torch.int32),
            extend_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            extend_seq_lens_cpu=[3, 2],
            extend_prefix_lens_cpu=[11, 50],
            extend_logprob_start_lens_cpu=[0, 0],
            spec_info=None,
            mm_inputs=None,
        )

    def _decode_batch(self):
        return SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            input_ids=torch.arange(3, dtype=torch.int64),
            positions=torch.tensor([13, 51, 7]),
            out_cache_loc=torch.arange(3, dtype=torch.int32),
            seq_lens=torch.tensor([14, 52, 8], dtype=torch.int32),
            spec_info=None,
            mm_inputs=None,
        )

    def _policy(self):
        return DecoderTpuCompilePolicy(
            model_runner=SimpleNamespace(device="cpu"),
            t_bucketer=Bucketer(name="Token", buckets=[8]),
            b_bucketer=Bucketer(name="Bs", buckets=[1, 4]),
        )

    def _run(self, batch):
        """Return the tensors padded on device and those padded on host."""
        with patch.object(shim, "_pad_tensor", wraps=shim._pad_tensor) as pad:
            self._policy().prepare_forward_batch(batch)
        calls = pad.call_args_list
        return (
            [call.args[0] for call in calls if not call.kwargs.get("on_host")],
            [call.args[0] for call in calls if call.kwargs.get("on_host")],
        )

    def _assert_extend_uses_device_padding(self):
        batch = self._extend_batch()
        originals = (
            batch.input_ids,
            batch.positions,
            batch.out_cache_loc,
            batch.mrope_positions,
        )

        device_padded, host_padded = self._run(batch)

        self.assertEqual(host_padded, [])
        for original in originals:
            self.assertTrue(any(t is original for t in device_padded))
        self.assertEqual(batch.input_ids.tolist(), [0, 1, 2, 3, 4, 0, 0, 0])
        self.assertEqual(batch.positions.tolist(), [11, 12, 13, 50, 51, 0, 0, 0])
        self.assertEqual(batch.out_cache_loc.tolist(), [0, 1, 2, 3, 4, 0, 0, 0])
        self.assertEqual(batch.mrope_positions.shape, (3, 8))
        self.assertEqual(batch.mrope_positions[1].tolist(), [5, 6, 7, 8, 9, 0, 0, 0])

    def test_extend_metadata_uses_device_pad_by_default(self):
        self.assertFalse(envs.SGLANG_TPU_HOST_PADDING.is_set())
        self.assertFalse(envs.SGLANG_TPU_HOST_PADDING.get())
        self._assert_extend_uses_device_padding()

    def test_extend_metadata_uses_device_pad_when_disabled(self):
        with envs.SGLANG_TPU_HOST_PADDING.override(False):
            self._assert_extend_uses_device_padding()

    def test_extend_metadata_takes_host_round_trip_when_enabled(self):
        batch = self._extend_batch()
        input_ids, out_cache_loc = batch.input_ids, batch.out_cache_loc
        positions, mrope = batch.positions, batch.mrope_positions

        with envs.SGLANG_TPU_HOST_PADDING.override(True):
            device_padded, host_padded = self._run(batch)

        for original in (input_ids, positions, out_cache_loc, mrope):
            self.assertTrue(any(t is original for t in host_padded))
            self.assertFalse(any(t is original for t in device_padded))
        self.assertEqual(batch.input_ids.tolist(), [0, 1, 2, 3, 4, 0, 0, 0])
        self.assertEqual(batch.out_cache_loc.tolist(), [0, 1, 2, 3, 4, 0, 0, 0])
        self.assertEqual(batch.mrope_positions.shape, (3, 8))
        self.assertEqual(batch.mrope_positions[1].tolist(), [5, 6, 7, 8, 9, 0, 0, 0])
        self.assertEqual(batch.positions.tolist(), [11, 12, 13, 50, 51, 0, 0, 0])

    def test_overridden_positions_are_padded_not_reconstructed(self):
        batch = self._extend_batch()
        batch.positions = torch.tensor([90, 70, 71, 22, 21])
        batch.spec_info = EagleDraftExtendInput(positions=batch.positions)
        positions = batch.positions

        with envs.SGLANG_TPU_HOST_PADDING.override(True):
            device_padded, host_padded = self._run(batch)

        self.assertTrue(any(t is positions for t in host_padded))
        self.assertFalse(any(t is positions for t in device_padded))
        self.assertEqual(batch.positions.tolist(), [90, 70, 71, 22, 21, 0, 0, 0])

    def test_target_verify_keeps_device_pad(self):
        # Three requests of two verify tokens pad from six to eight tokens.
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            input_ids=torch.arange(6, dtype=torch.int64),
            positions=torch.tensor([13, 14, 51, 52, 7, 8]),
            out_cache_loc=torch.arange(6, dtype=torch.int32),
            mrope_positions=torch.arange(18).reshape(3, 6),
            seq_lens=torch.tensor([14, 52, 8], dtype=torch.int32),
            spec_info=SimpleNamespace(
                num_tokens_per_req=2,
                hidden_states=None,
                record_padding=lambda **kwargs: None,
            ),
            mm_inputs=None,
        )

        with envs.SGLANG_TPU_HOST_PADDING.override(True):
            device_padded, host_padded = self._run(batch)

        self.assertEqual(host_padded, [])
        self.assertGreaterEqual(len(device_padded), 3)
        self.assertEqual(batch.input_ids.tolist(), [0, 1, 2, 3, 4, 5, 0, 0])
        self.assertEqual(batch.positions.tolist(), [13, 14, 51, 52, 7, 8, 0, 0])
        self.assertEqual(batch.mrope_positions.shape, (3, 8))

    def test_restore_trims_each_field_on_its_token_dim(self):
        # Fewer real tokens than MROPE rows, so trimming dim 0 would be wrong.
        batch = self._extend_batch()
        batch.input_ids = torch.arange(8)
        batch.mrope_positions = torch.arange(24).reshape(3, 8)

        shim._restore_forward_batch_inputs(batch, real_q=2, real_b=2)

        self.assertEqual(batch.input_ids.tolist(), [0, 1])
        self.assertEqual(batch.mrope_positions.shape, (3, 2))
        self.assertEqual(batch.mrope_positions[:, 0].tolist(), [0, 8, 16])

    def test_decode_restoration_keeps_mrope_but_trims_consumed_metadata(self):
        for live_rows in (4, 7, 8):
            with self.subTest(live_rows=live_rows):
                batch = self._decode_batch()
                batch.input_ids = torch.arange(8)
                batch.positions = torch.arange(8)
                batch.seq_lens = torch.arange(8)
                mrope = torch.empty((3, 8), dtype=torch.int64, device="meta")
                batch.mrope_positions = mrope

                class RejectMropeSlice(TorchDispatchMode):
                    def __torch_dispatch__(mode, func, types, args=(), kwargs=None):
                        if func == torch.ops.aten.slice.Tensor and args[0] is mrope:
                            raise AssertionError("ordinary decode must not slice mRoPE")
                        return func(*args, **(kwargs or {}))

                with RejectMropeSlice():
                    shim._restore_forward_batch_inputs(batch, live_rows, live_rows)
                self.assertIs(batch.mrope_positions, mrope)
                self.assertEqual(batch.positions.shape, (live_rows,))
                self.assertEqual(batch.input_ids.shape, (live_rows,))
                self.assertEqual(batch.seq_lens.shape, (live_rows,))

        # Speculative decode retains its existing restoration contract.
        batch.spec_info = SimpleNamespace(hidden_states=None)
        shim._restore_forward_batch_inputs(batch, 7, 7)
        self.assertEqual(batch.mrope_positions.shape, (3, 7))

    def test_decode_keeps_device_pad(self):
        batch = self._decode_batch()

        with envs.SGLANG_TPU_HOST_PADDING.override(True):
            device_padded, host_padded = self._run(batch)

        self.assertEqual(host_padded, [])
        self.assertGreaterEqual(len(device_padded), 3)
        self.assertEqual(batch.input_ids.tolist(), [0, 1, 2, 0])
        self.assertEqual(batch.positions.tolist(), [13, 51, 7, 0])

    def test_padding_precedes_upload_and_preserves_host_mirrors(self):
        batch = self._extend_batch()
        # Forward staging must also work without sampling metadata.
        batch.sampling_info = None
        batch.req_pool_indices_cpu = torch.tensor([2, 7])
        policy = self._policy()
        policy.model_runner.device = "meta"
        copies = {}

        class ObserveCopies(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func == torch.ops.aten._to_copy.default:
                    copies[id(args[0])] = args[0].clone()
                return func(*args, **(kwargs or {}))

        with envs.SGLANG_TPU_HOST_PADDING.override(True), ObserveCopies():
            policy.prepare_forward_batch(batch)
        self.assertEqual(batch.input_ids.device.type, "meta")
        self.assertEqual(batch.input_ids.shape, (8,))
        self.assertEqual(batch.extend_seq_lens.shape, (4,))
        self.assertEqual(batch.req_pool_indices_cpu.device.type, "cpu")
        self.assertEqual(batch.req_pool_indices_cpu.tolist(), [2, 7, 0, 0])
        self.assertEqual(
            sorted(x.shape[0] for x in copies.values()), [3, 4, 4, 4, 4, 8, 8, 8]
        )
        self.assertIn(
            [11, 12, 13, 50, 51, 0, 0, 0], [x.tolist() for x in copies.values()]
        )

    def test_decode_mrope_builds_on_host_and_uploads_only_bucket_shape(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.hooks import host_padding

        override = get_context().override_server_args(
            enable_torch_compile=True,
            enable_mixed_chunk=False,
        )
        override.install()
        self.addCleanup(override.restore)
        original = forward_batch_info._mrope_decode_device

        def placement(batch, device):
            return host_padding.mrope_decode_device(original, batch, device)

        for size in (4, 7):
            for multimodal in (False, True):
                with self.subTest(size=size, multimodal=multimodal):
                    lengths = torch.arange(10, 10 + size)
                    deltas = [(-3 if i % 2 == 0 else 0) for i in range(size)]
                    mm_inputs = [
                        SimpleNamespace(
                            mrope_positions=None,
                            mrope_position_delta=torch.tensor(delta),
                        )
                        if multimodal and delta
                        else None
                        for delta in deltas
                    ]
                    batch = SimpleNamespace(
                        forward_mode=ForwardMode.DECODE,
                        spec_algorithm=SpeculativeAlgorithm.NONE,
                        multimodal_inputs=mm_inputs,
                    )
                    forward = SimpleNamespace(
                        forward_mode=ForwardMode.DECODE,
                        spec_info=None,
                        seq_lens_cpu=lengths,
                        seq_lens=lengths.to("meta"),
                        input_ids=torch.empty(size, dtype=torch.int64, device="meta"),
                        positions=torch.empty(size, dtype=torch.int64, device="meta"),
                        out_cache_loc=torch.empty(
                            size, dtype=torch.int32, device="meta"
                        ),
                        mm_inputs=None,
                    )
                    runner = SimpleNamespace(device="meta")
                    with (
                        envs.SGLANG_TPU_HOST_PADDING.override(True),
                        patch.object(
                            forward_batch_info, "_mrope_decode_device", placement
                        ),
                    ):
                        forward_batch_info.ForwardBatch._compute_mrope_positions_decode(
                            forward, runner, batch
                        )
                        expected = lengths - 1
                        if multimodal:
                            expected = expected + torch.tensor(deltas)
                        self.assertEqual(forward.mrope_positions.device.type, "cpu")
                        self.assertTrue(
                            torch.equal(
                                forward.mrope_positions,
                                expected.unsqueeze(0).repeat(3, 1),
                            )
                        )
                        uploads = []

                        class RecordUploads(TorchDispatchMode):
                            def __torch_dispatch__(
                                mode, func, types, args=(), kwargs=None
                            ):
                                if func == torch.ops.aten._to_copy.default:
                                    uploads.append(args[0].clone())
                                return func(*args, **(kwargs or {}))

                        policy = self._policy()
                        policy.model_runner = runner
                        policy.b_bucketer = Bucketer(name="Bs", buckets=[8])
                        with RecordUploads():
                            policy.prepare_forward_batch(forward)
                        self.assertEqual(len(uploads), 1)
                        self.assertEqual(uploads[0].shape, (3, 8))
                        self.assertTrue(
                            torch.equal(
                                uploads[0][:, :size], expected.unsqueeze(0).repeat(3, 1)
                            )
                        )
                        self.assertTrue(torch.all(uploads[0][:, size:] == 0))
                        self.assertEqual(forward.mrope_positions.device.type, "meta")
                        self.assertEqual(forward.mrope_positions.dtype, torch.int64)
                    with envs.SGLANG_TPU_HOST_PADDING.override(False):
                        self.assertEqual(placement(batch, "meta"), "meta")
                    with envs.SGLANG_TPU_HOST_PADDING.override(True):
                        batch.spec_algorithm = SpeculativeAlgorithm.EAGLE3
                        self.assertEqual(placement(batch, "meta"), "meta")

    def test_precomputed_decode_mrope_keeps_spatial_axes(self):
        from types import MethodType

        positions = torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
        mm_input = SimpleNamespace(mrope_positions=positions)
        forward = SimpleNamespace(
            seq_lens_cpu=torch.tensor([2, 3]), seq_lens=torch.empty(2, device="meta")
        )
        forward._expand_mrope_from_input = MethodType(
            forward_batch_info.ForwardBatch._expand_mrope_from_input, forward
        )
        # A fallback request must not consult delta-based placement at all.
        with patch.object(
            forward_batch_info,
            "_mrope_decode_device",
            side_effect=AssertionError("precomputed positions must be preserved"),
        ):
            forward_batch_info.ForwardBatch._compute_mrope_positions_decode(
                forward,
                SimpleNamespace(device="cpu"),
                SimpleNamespace(multimodal_inputs=[mm_input, None]),
            )
        self.assertEqual(forward.mrope_positions.tolist(), [[11, 2], [21, 2], [31, 2]])

    def test_decode_metadata_pads_on_host_before_upload(self):
        # Device tokens coexist with independent CPU length/index snapshots.
        batch = self._decode_batch()
        batch.input_ids = batch.input_ids.to("meta")
        batch.out_cache_loc = batch.out_cache_loc.to("meta")
        batch.orig_seq_lens = torch.tensor([90, 70, 80], dtype=torch.int32)
        batch.req_pool_indices = torch.tensor([2, 7, 9])
        batch.seq_lens_cpu = batch.seq_lens
        batch.req_pool_indices_cpu = batch.req_pool_indices
        batch.mamba_track_indices = torch.tensor([3, 4, 5])
        originals = {
            name: getattr(batch, name)
            for name in ("seq_lens", "orig_seq_lens", "req_pool_indices")
        }
        values = {name: value.clone() for name, value in originals.items()}
        policy = self._policy()
        policy.model_runner.device = "meta"
        uploads = []

        class RecordUploads(TorchDispatchMode):
            def __torch_dispatch__(mode, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func == torch.ops.aten._to_copy.default and kwargs.get(
                    "device"
                ) == torch.device("meta"):
                    uploads.append(args[0].clone())
                return func(*args, **kwargs)

        with envs.SGLANG_TPU_HOST_PADDING.override(True), RecordUploads():
            policy.prepare_forward_batch(batch)
        for name, original in originals.items():
            self.assertTrue(torch.equal(original, values[name]))
            self.assertEqual(original.shape[0], 3)
            self.assertEqual(getattr(batch, name).device.type, "meta")
            self.assertEqual(getattr(batch, name).shape[0], 4)
        self.assertTrue(any(x.tolist() == [90, 70, 80, 0] for x in uploads))
        self.assertTrue(all(x.shape[0] == 4 for x in uploads))
        self.assertEqual(batch.orig_seq_lens.dtype, torch.int32)
        self.assertEqual(batch.mamba_track_indices.device.type, "meta")
        self.assertIs(batch.seq_lens_cpu, originals["seq_lens"])
        self.assertEqual(batch.req_pool_indices_cpu.device.type, "cpu")

    def test_host_scheduler_merge_filter_decode_preserves_snapshots(self):
        from unittest.mock import MagicMock

        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        def make_batch(lengths, originals, indices):
            return ScheduleBatch(
                reqs=[
                    SimpleNamespace(
                        beam_group=None,
                        decode_batch_idx=0,
                        return_logprob=False,
                        grammar=None,
                        return_hidden_states_mode=CaptureHiddenMode.NULL,
                    )
                    for _ in lengths
                ],
                device="meta",
                model_config=SimpleNamespace(is_encoder_decoder=False),
                spec_algorithm=SpeculativeAlgorithm.NONE,
                seq_lens=torch.tensor(lengths),
                seq_lens_cpu=torch.tensor(lengths),
                orig_seq_lens=torch.tensor(originals, dtype=torch.int32),
                req_pool_indices=torch.tensor(indices),
                req_pool_indices_cpu=torch.tensor(indices),
                sampling_info=MagicMock(),
            )

        batch = make_batch([5, 6, 7], [50, 60, 70], [1, 2, 3])
        other = make_batch([8, 9], [80, 90], [4, 5])
        names = ("seq_lens", "orig_seq_lens", "req_pool_indices")
        snapshot = {name: getattr(batch, name) for name in names}
        batch.merge_batch(other)
        batch.filter_batch(keep_indices=[4, 0, 2])
        self.assertEqual(batch.seq_lens.tolist(), [9, 5, 7])
        self.assertEqual(batch.orig_seq_lens.tolist(), [90, 50, 70])
        self.assertEqual(batch.req_pool_indices.tolist(), [5, 1, 3])
        forward_snapshot = {name: getattr(batch, name) for name in names}
        batch.sampling_info.penalizer_orchestrator.is_required = False
        with patch(
            "sglang.srt.managers.schedule_batch.alloc_for_decode",
            return_value=torch.empty(3, device="meta"),
        ):
            batch.prepare_for_decode()
        self.assertEqual(batch.seq_lens.tolist(), [10, 6, 8])
        self.assertEqual(batch.orig_seq_lens.tolist(), [91, 51, 71])
        self.assertEqual(forward_snapshot["seq_lens"].tolist(), [9, 5, 7])
        self.assertEqual(forward_snapshot["orig_seq_lens"].tolist(), [90, 50, 70])
        self.assertEqual(snapshot["seq_lens"].tolist(), [5, 6, 7])
        self.assertEqual(snapshot["orig_seq_lens"].tolist(), [50, 60, 70])
        self.assertEqual(snapshot["req_pool_indices"].tolist(), [1, 2, 3])
        self.assertTrue(
            all(getattr(batch, name).device.type == "cpu" for name in names)
        )

    def test_relay_materializes_device_indices_without_replacing_host_metadata(self):
        from sglang.srt.managers import overlap_utils
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        indices = torch.tensor([2, 7, 9])
        batch = ScheduleBatch(reqs=[], req_pool_indices=indices)
        relay = overlap_utils.FutureMap.__new__(overlap_utils.FutureMap)
        relay.spec_algo = SpeculativeAlgorithm.NONE
        relay._use_dynamic_index = True
        relay.output_tokens_buf = torch.empty(16, dtype=torch.int64, device="meta")
        operations = []

        class CheckIndexDevice(TorchDispatchMode):
            def __torch_dispatch__(mode, func, types, args=(), kwargs=None):
                if func == torch.ops.aten.index.Tensor:
                    self.assertEqual(args[1][0].device, args[0].device)
                    operations.append("gather")
                elif func == torch.ops.aten.index_copy_.default:
                    self.assertEqual(args[2].device, args[0].device)
                    operations.append("store")
                return func(*args, **(kwargs or {}))

        with patch.object(overlap_utils, "_DEBUG_ASSERT", False), CheckIndexDevice():
            overlap_utils.resolve_forward_inputs(batch, relay)
            relay.stash_bonus_tokens(
                indices, torch.empty(3, dtype=torch.int64, device="meta")
            )
        self.assertEqual(operations, ["gather", "store"])
        self.assertIs(batch.req_pool_indices, indices)
        self.assertEqual(indices.tolist(), [2, 7, 9])
        self.assertEqual(batch.input_ids.device.type, "meta")

    def test_prebuilt_metadata_uses_same_host_placement(self):
        from unittest.mock import MagicMock

        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.hooks import host_padding

        req = SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=1),
            prefix_indices=[0, 1],
            extend_range=SimpleNamespace(length=3),
            origin_input_ids=[10, 11, 12, 13, 14],
            output_ids=[],
            retracted_stain=True,
            is_retracted=True,
            multimodal_inputs=None,
            beam_group=None,
        )
        batch = ScheduleBatch(
            reqs=[req],
            device="meta",
            spec_algorithm=SpeculativeAlgorithm.NONE,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(10).reshape(2, 5)
            ),
            model_config=SimpleNamespace(
                vocab_size=32,
                is_generation=True,
                is_multimodal=False,
                is_encoder_decoder=False,
                use_ngram_embedding=False,
                use_engram=False,
            ),
        )
        override = get_context().override_server_args(
            enable_torch_compile=True,
            sampling_backend="pytorch",
            enable_mixed_chunk=False,
        )
        override.install()
        self.addCleanup(override.restore)
        original = ScheduleBatch.metadata_device

        def placement(batch):
            return host_padding.metadata_device(original, batch)

        with (
            envs.SGLANG_TPU_HOST_PADDING.override(True),
            patch.object(ScheduleBatch, "metadata_device", placement),
            patch(
                "sglang.srt.disaggregation.decode_schedule_batch_mixin.SamplingBatchInfo.from_schedule_batch",
                return_value=MagicMock(),
            ),
        ):
            # Multimodal models used to inherit the text-token staging gate,
            # leaving scheduler metadata on device even for text-only requests.
            for multimodal in (False, True):
                with self.subTest(multimodal=multimodal):
                    batch.model_config.is_multimodal = multimodal
                    batch.prepare_for_prebuilt()
                    self.assertEqual(batch.seq_lens.device.type, "cpu")
                    self.assertEqual(batch.seq_lens.tolist(), [5])
                    self.assertEqual(batch.orig_seq_lens.tolist(), [5])
                    self.assertEqual(batch.req_pool_indices.tolist(), [1])
                    self.assertEqual(batch.orig_seq_lens.dtype, torch.int32)
                    self.assertIsNone(batch.input_ids)
                    batch.prefill_input_ids_cpu = torch.tensor([10, 11, 12])
                    tokens = host_padding.stage_prefill_tokens(
                        lambda b: b.prefill_input_ids_cpu.to("meta"), batch
                    )
                    self.assertEqual(
                        tokens.device.type, "meta" if multimodal else "cpu"
                    )
        # The placement hook defaults to the existing backend contract.
        with envs.SGLANG_TPU_HOST_PADDING.override(False):
            self.assertEqual(placement(batch), "meta")

    def test_staging_fallbacks_preserve_device_inputs(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.hooks import host_padding

        override = get_context().override_server_args(
            enable_torch_compile=True,
            sampling_backend="pytorch",
            enable_mixed_chunk=False,
        )
        override.install()
        self.addCleanup(override.restore)
        batch = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.NONE,
            mix_running_indices=None,
            dllm_config=None,
            model_config=SimpleNamespace(
                is_generation=True,
                is_multimodal=False,
                is_encoder_decoder=False,
                use_ngram_embedding=False,
                use_engram=False,
            ),
            prefill_input_ids_cpu=torch.tensor([1, 2, 3]),
        )

        def upload(batch):
            return batch.prefill_input_ids_cpu.to("meta")

        with envs.SGLANG_TPU_HOST_PADDING.override(True):
            staged = host_padding.stage_prefill_tokens(upload, batch)
            self.assertIs(staged, batch.prefill_input_ids_cpu)
            for field, value in (
                ("mix_running_indices", [0]),
                ("spec_algorithm", SpeculativeAlgorithm.EAGLE3),
            ):
                old = getattr(batch, field)
                setattr(batch, field, value)
                self.assertEqual(
                    host_padding.stage_prefill_tokens(upload, batch).device.type, "meta"
                )
                setattr(batch, field, old)
        with envs.SGLANG_TPU_HOST_PADDING.override(False):
            self.assertEqual(
                host_padding.stage_prefill_tokens(upload, batch).device.type, "meta"
            )


if __name__ == "__main__":
    unittest.main()
