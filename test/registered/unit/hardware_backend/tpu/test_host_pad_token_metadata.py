"""Host staging, scheduler metadata ownership, and TPU forward padding."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

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
    _pad_tensor,
)
from sglang_tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang_tpu.envs import envs

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPaddedSamplingCompileGuards(CustomTestCase):
    def test_ineligible_paths_do_not_guard_sampling_metadata(self):
        policy = object.__new__(DecoderTpuCompilePolicy)

        def forward(value, batch):
            if policy.keep_outputs_padded(batch):
                return value + 2
            return value + 1

        for enabled, speculative in ((False, False), (False, True), (True, True)):
            with (
                self.subTest(enabled=enabled, speculative=speculative),
                envs.SGLANG_TPU_HOST_PADDING.override(enabled),
            ):
                torch._dynamo.reset()
                self.addCleanup(torch._dynamo.reset)
                batch = SimpleNamespace(
                    spec_info=SimpleNamespace() if speculative else None,
                    sampling_info=None,
                )
                compiled = torch.compile(forward, backend="eager", fullgraph=True)
                value = torch.ones(2)
                torch.testing.assert_close(compiled(value, batch), value + 1)
                batch.sampling_info = SimpleNamespace(sampling_params_on_cpu=True)
                with torch.compiler.set_stance("fail_on_recompile"):
                    torch.testing.assert_close(compiled(value, batch), value + 1)


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


class TestPadTensor(CustomTestCase):
    """Padding preserves values on the input's device and handles token axes."""

    def test_bucketed_metadata_is_returned_without_copying(self):
        value = torch.arange(6).reshape(2, 3)

        self.assertIs(_pad_tensor(value, 2), value)
        self.assertIs(_pad_tensor(value, 3, dim=-1), value)

    def test_dim0_preserves_values_and_zero_fills(self):
        value = torch.arange(6, dtype=torch.int64).reshape(3, 2)

        result = _pad_tensor(value, 5)

        self.assertEqual(result.shape, (5, 2))
        self.assertTrue(torch.equal(result[:3], value))
        self.assertTrue(torch.all(result[3:] == 0))

    def test_last_dim_preserves_values_and_zero_fills(self):
        value = torch.arange(6, dtype=torch.int64).reshape(2, 3)

        result = _pad_tensor(value, 5, dim=-1)

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
            sampling_info=None,
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
            sampling_info=None,
            mm_inputs=None,
        )

    def _policy(self):
        return DecoderTpuCompilePolicy(
            model_runner=SimpleNamespace(device="cpu"),
            t_bucketer=Bucketer(name="Token", buckets=[8]),
            b_bucketer=Bucketer(name="Bs", buckets=[1, 4]),
        )

    def _run(self, batch):
        self._policy().prepare_forward_batch(batch)

    def _assert_extend_uses_device_padding(self):
        batch = self._extend_batch()
        self._run(batch)
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

    def test_device_token_padding_never_reads_back(self):
        batch = self._extend_batch()
        fields = (
            "input_ids",
            "positions",
            "out_cache_loc",
            "mrope_positions",
            "seq_lens",
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_start_loc",
        )
        for name in fields:
            setattr(batch, name, getattr(batch, name).to("meta"))
        policy = self._policy()
        policy.model_runner.device = "meta"

        class RejectReadbacks(TorchDispatchMode):
            def __torch_dispatch__(mode, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if (
                    func == torch.ops.aten._to_copy.default
                    and args[0].device.type == "meta"
                    and kwargs.get("device") == torch.device("cpu")
                ):
                    raise AssertionError("padding must not read device inputs back")
                return func(*args, **kwargs)

        with envs.SGLANG_TPU_HOST_PADDING.override(True), RejectReadbacks():
            policy.prepare_forward_batch(batch)
        for name in fields:
            self.assertEqual(getattr(batch, name).device.type, "meta")
        self.assertEqual(batch.input_ids.shape, (8,))
        self.assertEqual(batch.mrope_positions.shape, (3, 8))

    def test_device_prefill_tokens_warn_but_device_decode_does_not(self):
        shim.print_warning_once.cache_clear()
        self.addCleanup(shim.print_warning_once.cache_clear)
        policy = self._policy()
        policy.model_runner.device = "meta"
        decode = self._decode_batch()
        decode.input_ids = decode.input_ids.to("meta")
        with envs.SGLANG_TPU_HOST_PADDING.override(True):
            with self.assertNoLogs("sglang.srt.utils.common", level="WARNING"):
                policy.prepare_forward_batch(decode)
            prefill = self._extend_batch()
            prefill.input_ids = prefill.input_ids.to("meta")
            with self.assertLogs("sglang.srt.utils.common", level="WARNING"):
                policy.prepare_forward_batch(prefill)

    def test_extend_mrope_follows_the_extend_metadata_placement(self):
        """Extend mRoPE positions must land where the extend lengths do.

        A backend hook that keeps extend metadata on the host (the spec worker
        does) would otherwise see mRoPE positions padded on the device at the
        live length, one XLA program per prompt length.
        """
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        fb = object.__new__(ForwardBatch)
        fb.seq_lens_cpu = torch.tensor([5, 3])
        batch = SimpleNamespace(
            multimodal_inputs=[None, None],
            extend_lens=[5, 3],
            prefix_lens=[0, 2],
            reqs=[SimpleNamespace(session=None)] * 2,
            input_tensor_device=None,
        )
        runner = SimpleNamespace(device="meta")
        with get_context().override_server_args():
            fb._compute_mrope_positions_extend(runner, batch)
            self.assertEqual(fb.mrope_positions.device.type, "meta")
            with patch.object(
                forward_batch_info, "_extend_metadata_device", lambda b, d: "cpu"
            ):
                fb._compute_mrope_positions_extend(runner, batch)
        self.assertEqual(fb.mrope_positions.device.type, "cpu")
        self.assertEqual(fb.mrope_positions.shape, (3, 8))
        self.assertEqual(fb.mrope_positions[0].tolist(), [0, 1, 2, 3, 4, 2, 3, 4])

    def test_startup_check_rejects_configs_that_can_never_stage(self):
        """A set env with no host-return KV path fails at startup, not per batch."""
        from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
        from sglang_tpu.host_padding import check_host_padding_config

        def runner(allocator):
            return SimpleNamespace(
                token_to_kv_pool_allocator=allocator,
                req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros((2, 8))),
            )

        allocator = TpuPagedTokenToKVPoolAllocator(
            size=8,
            page_size=4,
            dtype=torch.bfloat16,
            device="meta",
            kvcache=None,
            need_sort=False,
        )
        with get_context().override_server_args(enable_torch_compile=True):
            check_host_padding_config(runner(allocator))
            with self.assertRaisesRegex(ValueError, "returns device KV indices"):
                check_host_padding_config(runner(SimpleNamespace(page_size=4)))
            allocator.page_size = 1
            with self.assertRaisesRegex(ValueError, "page size 1"):
                check_host_padding_config(runner(allocator))
        with get_context().override_server_args(enable_torch_compile=False):
            allocator.page_size = 4
            with self.assertRaisesRegex(ValueError, "torch.compile is off"):
                check_host_padding_config(runner(allocator))

    def test_overridden_positions_are_padded_not_reconstructed(self):
        batch = self._extend_batch()
        batch.positions = torch.tensor([90, 70, 71, 22, 21])
        batch.spec_info = EagleDraftExtendInput(positions=batch.positions)
        with envs.SGLANG_TPU_HOST_PADDING.override(True):
            self._run(batch)

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
            sampling_info=None,
            mm_inputs=None,
        )

        with envs.SGLANG_TPU_HOST_PADDING.override(True):
            self._run(batch)

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
            with (
                self.subTest(live_rows=live_rows),
                envs.SGLANG_TPU_HOST_PADDING.override(True),
            ):
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

        # Without host padding and for speculative decode, mRoPE is trimmed as before.
        for flag, spec_info in (
            (False, None),
            (True, SimpleNamespace(hidden_states=None)),
        ):
            with (
                self.subTest(flag=flag, spec=spec_info is not None),
                envs.SGLANG_TPU_HOST_PADDING.override(flag),
            ):
                batch = self._decode_batch()
                batch.spec_info = spec_info
                batch.mrope_positions = torch.zeros((3, 8), dtype=torch.int64)
                shim._restore_forward_batch_inputs(batch, 7, 7)
                self.assertEqual(batch.mrope_positions.shape, (3, 7))

    def test_decode_keeps_device_pad(self):
        batch = self._decode_batch()

        with envs.SGLANG_TPU_HOST_PADDING.override(True):
            self._run(batch)

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

    def test_host_extend_metadata_uploads_when_tokens_are_on_device(self):
        """Host extend lengths upload even when tokens and seq_lens are on device."""
        batch = self._extend_batch()
        for name in ("input_ids", "positions", "out_cache_loc", "seq_lens"):
            setattr(batch, name, getattr(batch, name).to("meta"))
        batch.mrope_positions = None
        policy = object.__new__(DecoderTpuCompilePolicy)
        policy.model_runner = SimpleNamespace(device="meta")
        with envs.SGLANG_TPU_HOST_PADDING.override(True):
            policy._upload_host_inputs(batch)
        for name in ("extend_seq_lens", "extend_prefix_lens", "extend_start_loc"):
            self.assertEqual(getattr(batch, name).device.type, "meta", name)

    def test_experimental_spec_keeps_non_overlap_metadata_on_host(self):
        """An experimental-worker PD-prefill batch filters and pads its lengths on
        host; an overlap batch, or one off the worker, keeps them on the device."""
        from sglang_tpu.hooks import host_padding

        def batch(*, overlap):
            return SimpleNamespace(
                host_pad_bs=8,
                enable_overlap=overlap,
                spec_algorithm=SimpleNamespace(is_none=lambda: False),
                req_to_token_pool=SimpleNamespace(
                    req_to_token=torch.zeros(1, device="cpu")
                ),
                hisparse_coordinator=None,
                reqs=[],
            )

        original = lambda b: "tpu"  # noqa: E731
        self.assertEqual(
            host_padding.metadata_device(original, batch(overlap=False)), "cpu"
        )
        self.assertEqual(
            host_padding.metadata_device(original, batch(overlap=True)), "tpu"
        )
        off_worker = batch(overlap=False)
        off_worker.host_pad_bs = None
        self.assertEqual(host_padding.metadata_device(original, off_worker), "tpu")

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
                        sampling_info=None,
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
        # The placement hook defaults to the existing backend contract.
        with envs.SGLANG_TPU_HOST_PADDING.override(False):
            self.assertEqual(placement(batch), "meta")

    def test_vl_kv_locations_upload_only_after_padding(self):
        """VL KV slots stay on CPU until forward-input padding and upload."""
        from sglang.srt.mem_cache.allocation import alloc_paged_token_slots_extend
        from sglang.srt.model_executor.forward_batch_info import (
            _extend_metadata_device,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        override = get_context().override_server_args(
            enable_torch_compile=True, enable_mixed_chunk=False
        )
        override.install()
        self.addCleanup(override.restore)
        batch = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.NONE,
            mix_running_indices=None,
            dllm_config=None,
            input_tensor_device=None,
            model_config=SimpleNamespace(
                is_generation=True,
                is_multimodal=True,
                is_encoder_decoder=False,
                use_ngram_embedding=False,
                use_engram=False,
            ),
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros((2, 16))),
        )
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                envs.SGLANG_TPU_HOST_PADDING.override(enabled),
            ):
                allocator = TpuPagedTokenToKVPoolAllocator(
                    size=24,
                    page_size=4,
                    dtype=torch.bfloat16,
                    device="meta",
                    kvcache=None,
                    need_sort=False,
                )
                tree = SimpleNamespace(
                    token_to_kv_pool_allocator=allocator,
                    supports_prefix_sharing=lambda: False,
                )
                prefix = torch.tensor([0, 0])
                lengths = torch.tensor([3, 2])
                batch.token_to_kv_pool_allocator = allocator
                batch.input_tensor_device = TpuSRTPlatform().get_batch_input_device(
                    batch
                )
                slots = alloc_paged_token_slots_extend(
                    tree,
                    prefix,
                    prefix,
                    lengths,
                    lengths,
                    prefix - 1,
                    5,
                    batch=batch,
                )
                self.assertEqual(slots.device.type, "cpu" if enabled else "meta")
                if not enabled:
                    continue
                self.assertEqual(slots.tolist(), [4, 5, 6, 8, 9])
                forward = self._extend_batch()
                forward.input_ids = forward.input_ids.to("meta")
                forward.out_cache_loc = slots
                batch.input_ids = forward.input_ids
                metadata_device = _extend_metadata_device(batch, "meta")
                forward.extend_seq_lens = lengths.to(metadata_device)
                forward.extend_prefix_lens = prefix.to(metadata_device)
                self.assertEqual(forward.extend_seq_lens.device.type, "cpu")
                positions, forward.extend_start_loc = (
                    forward_batch_info.compute_position_torch(
                        forward.extend_prefix_lens, forward.extend_seq_lens
                    )
                )
                self.assertEqual(positions.tolist(), [0, 1, 2, 0, 1])
                shim._pad_batch_fields(
                    forward, pad_t=8, pad_b=8, real_b=2, is_draft_extend_v2=False
                )
                self.assertEqual(
                    forward.extend_seq_lens.tolist(), [3, 2, 0, 0, 0, 0, 0, 0]
                )
                self.assertEqual(forward.out_cache_loc.device.type, "cpu")
                self.assertEqual(
                    forward.out_cache_loc.tolist(), [4, 5, 6, 8, 9, 0, 0, 0]
                )
                policy = object.__new__(DecoderTpuCompilePolicy)
                policy.model_runner = SimpleNamespace(device="meta")
                policy._upload_host_inputs(forward)
                self.assertEqual(forward.out_cache_loc.device.type, "meta")
                self.assertEqual(forward.out_cache_loc.shape, (8,))
                allocator.free(slots)
                allocator.merge_and_sort_free()
                self.assertEqual(allocator.available_size(), 24)

    def test_prebuilt_tokens_upload_only_after_relay_padding(self):
        """PD handoff must not upload live-size tokens before the bucketed relay."""
        from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
            ScheduleBatchDisaggregationDecodeMixin,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.host_padding_relay import HostPaddingFutureMap

        pool = SimpleNamespace(
            req_to_token=torch.zeros((9, 16), dtype=torch.int32),
            _tpu_relay_bucketer=Bucketer("Bs", [1, 2, 4, 8]),
        )
        relay = HostPaddingFutureMap("meta", SpeculativeAlgorithm.NONE, pool)
        batch = SimpleNamespace(
            device="meta",
            reqs=[
                SimpleNamespace(
                    output_ids=[token],
                    swa_branching_seqlen=None,
                    grammar=None,
                    finished=lambda: False,
                    skip_radix_cache_insert=True,
                )
                for token in (11, 22, 33)
            ],
            req_pool_indices=torch.tensor([1, 2, 3]),
            spec_algorithm=SpeculativeAlgorithm.NONE,
            tree_cache=None,
        )
        uploads = []

        class ObserveUploads(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func == torch.ops.aten._to_copy.default and kwargs.get(
                    "device"
                ) == torch.device("meta"):
                    uploads.append(args[0].clone())
                return func(*args, **kwargs)

        # The platform decides the placement; None keeps the live-size upload.
        for placement in ("cpu", None):
            uploads.clear()
            platform = SimpleNamespace(get_batch_input_device=lambda b: placement)
            with (
                self.subTest(placement=placement),
                patch(
                    "sglang.srt.disaggregation.decode_schedule_batch_mixin."
                    "current_platform",
                    platform,
                ),
                ObserveUploads(),
            ):
                ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(batch, relay)
                self.assertIsNone(batch.input_ids)
                if placement is None:
                    self.assertEqual(uploads[0].shape, (3,))
                    continue
                self.assertEqual([value.shape for value in uploads], [(4,), (4,)])
                self.assertEqual(uploads[0].tolist(), [11, 22, 33, 0])
                self.assertEqual(uploads[1].tolist(), [1, 2, 3, 12])

    def test_bucketed_relay_preserves_live_slots_and_only_uses_bucket_shapes(self):
        from sglang.srt.managers.overlap_utils import RelayPayload
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.host_padding_relay import HostPaddingFutureMap

        buckets = Bucketer(name="Bs", buckets=[1, 2, 4, 8])
        table = torch.full((9, 16), 123, dtype=torch.int32)
        pool = SimpleNamespace(req_to_token=table, _tpu_relay_bucketer=buckets)
        relay = HostPaddingFutureMap("cpu", SpeculativeAlgorithm.NONE, pool)
        observed = set()

        class ObserveRelay(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func == torch.ops.aten.index_copy_.default:
                    observed.add(args[2].numel())
                    # Includes ghost writes: no duplicate destination indices.
                    assert args[2].unique().numel() == args[2].numel()
                if func == torch.ops.aten.index.Tensor:
                    observed.add(args[1][0].numel())
                return func(*args, **(kwargs or {}))

        for live in range(1, 9):
            indices = torch.arange(8, 8 - live, -1)
            width = buckets.next_bucket(live)
            values = torch.arange(width) + 100 * live
            before = relay.output_tokens_buf.clone()
            with ObserveRelay():
                relay.stash(indices, RelayPayload(bonus_tokens=values))
                result = relay.gather(indices)
            self.assertEqual(len(result), width)
            torch.testing.assert_close(result[:live], values[:live])
            # Inactive request slots, including slot0, cannot be overwritten.
            torch.testing.assert_close(
                relay.output_tokens_buf[: 9 - live], before[: 9 - live]
            )
        self.assertEqual(observed, {1, 2, 4, 8})
        self.assertTrue(torch.all(table == 123).item())

    def test_prefill_cap_relay_can_gather_a_smaller_decode_bucket(self):
        from sglang.srt.managers.overlap_utils import RelayPayload
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.host_padding_relay import HostPaddingFutureMap

        pool = SimpleNamespace(
            req_to_token=torch.zeros((9, 16), dtype=torch.int32),
            _tpu_relay_bucketer=Bucketer("Bs", [1, 2, 4, 8]),
        )
        relay = HostPaddingFutureMap("cpu", SpeculativeAlgorithm.NONE, pool)
        untouched_indices = torch.tensor([0, 2, 3, 5, 6, 8])
        untouched_before = relay.output_tokens_buf[untouched_indices].clone()
        relay.stash(
            torch.tensor([7, 1, 4]), RelayPayload(bonus_tokens=torch.arange(8) + 10)
        )
        self.assertEqual(relay.gather(torch.tensor([4, 7])).tolist(), [12, 10])
        torch.testing.assert_close(
            relay.output_tokens_buf[untouched_indices], untouched_before
        )

    def test_bucketed_relay_preserves_ci_consume_once_checks(self):
        from sglang.srt.managers import overlap_utils
        from sglang.srt.managers.overlap_utils import RelayPayload
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.host_padding_relay import HostPaddingFutureMap

        pool = SimpleNamespace(
            req_to_token=torch.zeros((9, 16), dtype=torch.int32),
            _tpu_relay_bucketer=Bucketer("Bs", [1, 2, 4, 8]),
        )
        check = overlap_utils._assert_nonneg_and_invalidate
        check = getattr(check, "_torchdynamo_orig_callable", check)
        with (
            patch.object(overlap_utils, "_DEBUG_ASSERT", True),
            patch.object(overlap_utils, "_assert_nonneg_and_invalidate", check),
        ):
            relay = HostPaddingFutureMap("cpu", SpeculativeAlgorithm.NONE, pool)
            indices = torch.tensor([7, 1, 4])
            relay.stash(indices, RelayPayload(bonus_tokens=torch.arange(8) + 10))
            self.assertEqual(relay.gather(indices).tolist(), [10, 11, 12, 0])
            with self.assertRaises(RuntimeError):
                relay.gather(indices)

    def test_decode_allocator_can_retain_host_locations_until_padding(self):
        from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator

        allocator = TpuPagedTokenToKVPoolAllocator(
            size=24,
            page_size=4,
            dtype=torch.bfloat16,
            device="meta",
            kvcache=None,
            need_sort=False,
        )
        allocator.defer_decode_upload = True
        slots = allocator.alloc_decode(
            torch.tensor([2, 3, 4]), torch.tensor([2, 3, 4]), torch.tensor([4, 9, 14])
        )
        self.assertEqual(slots.device.type, "cpu")
        self.assertEqual(slots.tolist(), [5, 10, 15])

    def test_result_trims_only_after_copy_to_cpu(self):
        from sglang_tpu.hooks.host_padding import copy_result_to_cpu

        result = SimpleNamespace(next_token_ids=torch.arange(8), _tpu_live_token_rows=3)

        def copy(result):
            self.assertEqual(result.next_token_ids.shape, (8,))
            result.next_token_ids = result.next_token_ids.cpu()

        copy_result_to_cpu(copy, result)
        self.assertEqual(result.next_token_ids.tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
