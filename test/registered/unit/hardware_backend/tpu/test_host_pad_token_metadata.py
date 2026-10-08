"""Tests for padding small per-token tensors on the host during TPU prefill."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.compilation import compiled_forward_shim as shim
from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
    _pad_tensor_on_host,
)
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
    """Host padding defaults on for prefill, never for decode or verify."""

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
            model_runner=SimpleNamespace(),
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

    def test_extend_metadata_uses_host_pad_by_default(self):
        self.assertFalse(envs.SGLANG_TPU_HOST_PADDING.is_set())
        self.assertTrue(envs.SGLANG_TPU_HOST_PADDING.get())
        batch = self._extend_batch()
        originals = [getattr(batch, name) for name in shim._HOST_PAD_TOKEN_FIELDS]
        _, host_padded = self._run(batch)
        for original in originals:
            self.assertTrue(any(value is original for value in host_padded))

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

    def test_restore_reuses_metadata_without_retaining_large_embeddings(self):
        batch = self._extend_batch()
        batch.input_embeds = torch.randn(5, 4)
        batch.mm_input_embeds = torch.randn(5, 4)
        originals = {name: getattr(batch, name) for name in shim._TOKEN_FIELDS}
        # Repeat on the same batch to catch stale snapshots from earlier calls.
        for _ in range(2):
            self._run(batch)
            self.assertNotIn("input_embeds", batch._tpu_token_padding)
            self.assertNotIn("mm_input_embeds", batch._tpu_token_padding)
            shim._restore_forward_batch_inputs(batch, real_q=5, real_b=2)
            for name, original in originals.items():
                if name in shim._HOST_PAD_TOKEN_FIELDS:
                    self.assertIs(getattr(batch, name), original, name)
                else:
                    torch.testing.assert_close(getattr(batch, name), original)
            self.assertIsNone(batch._tpu_token_padding)

    def test_restore_preserves_a_consumer_replacement(self):
        batch = self._extend_batch()
        self._run(batch)
        # An eager embedding/position producer can replace a padded tensor.
        batch.positions = torch.arange(100, 108)
        shim._restore_forward_batch_inputs(batch, real_q=5, real_b=2)
        self.assertEqual(batch.positions.tolist(), list(range(100, 105)))

    def test_restore_trims_inputs_already_padded_by_dp(self):
        batch = self._decode_batch()
        self._run(batch)
        # DP can pass three physical rows for only two local live requests.
        shim._restore_forward_batch_inputs(batch, real_q=2, real_b=2)
        self.assertEqual(batch.input_ids.tolist(), [0, 1])
        self.assertEqual(batch.positions.tolist(), [13, 51])
        self.assertIsNone(batch._tpu_token_padding)

    def test_decode_keeps_device_pad(self):
        batch = self._decode_batch()

        with envs.SGLANG_TPU_HOST_PADDING.override(True):
            device_padded, host_padded = self._run(batch)

        self.assertEqual(host_padded, [])
        self.assertGreaterEqual(len(device_padded), 3)
        self.assertEqual(batch.input_ids.tolist(), [0, 1, 2, 0])
        self.assertEqual(batch.positions.tolist(), [13, 51, 7, 0])


if __name__ == "__main__":
    unittest.main()
