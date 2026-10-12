"""Unit tests for the TPU compiled-forward shim."""

import inspect
import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import resolve_forward_inputs
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.runtime_context import get_context, get_exec, get_parallel, get_schedule
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import patches
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.compilation.bucketer import (
    Bucketer,
    PowerBucketer,
)
from sglang_tpu.compilation.compiled_forward_shim import (
    _BS_FIELDS,
    _PINNED_LOGPROB_FIELDS,
    _TOKEN_FIELDS,
    DEFAULT_TPU_COMPILE_MAX_BS,
    CompiledForwardShim,
    _attach_input_logprobs,
    _dp_attn_apply_bucket_metadata,
    _dp_attn_select_bucket_shape,
    _extend_batch_cap,
    _matches_default,
    _pad_forward_batch_inputs,
    _pin_fields,
    _restore_fields,
    _restore_forward_batch_inputs,
    _static_default_kwargs,
    _zero_pad_dim0,
    build_bucketers,
    needs_input_logprob,
    reconcile_tpu_batch_limits,
)
from sglang_tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
    _DummyForwardBatchFields,
)
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.spec_decode_compile_policy_eagle3 import (
    Eagle3SpecDecodeCompilePolicyMixin,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.tpu_platform import TpuSRTPlatform

# Pure-logic tests: CPU tensors only, and the shim's import chain is
# torch_tpu-free, so this runs on the CPU CI runner (no TPU required).
register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _t(*vals, dtype=torch.int32):
    return torch.tensor(list(vals), dtype=dtype)


def _bucketer(name, buckets):
    return Bucketer(name=name, buckets=buckets)


class _ForwardModeMock:
    """Minimal forward_mode stand-in for unit tests."""

    def __init__(self, mode):
        self._mode = mode
        self.name = mode.upper()

    def is_decode(self):
        return self._mode == "decode"

    def is_idle(self):
        return self._mode == "idle"

    def is_extend(self):
        # Real ForwardMode.is_extend() is True for TARGET_VERIFY too (it is an
        # extend-family mode), so mirror that here; otherwise the shim guard's
        # `is_extend() and not is_target_verify()` short-circuit isn't tested
        # faithfully.
        return self._mode in ("extend", "mixed", "target_verify")

    def is_mixed(self):
        return self._mode == "mixed"

    def is_target_verify(self):
        return self._mode == "target_verify"

    def is_draft_extend_v2(self):
        return self._mode == "draft_extend_v2"


def _forward_batch(real_t, real_b, mode="extend", return_logprob=False):
    """Synthetic ForwardBatch carrying the fields the padder touches."""
    extend = mode in ("extend", "mixed", "draft_extend_v2")
    return SimpleNamespace(
        input_ids=torch.arange(real_t, dtype=torch.int32),
        positions=torch.arange(real_t, dtype=torch.int32),
        out_cache_loc=torch.arange(real_t, dtype=torch.int32),
        seq_lens=torch.full((real_b,), 100, dtype=torch.int32),
        seq_lens_cpu=torch.full((real_b,), 100, dtype=torch.int32),
        orig_seq_lens=torch.full((real_b,), 100, dtype=torch.int32),
        req_pool_indices=torch.arange(real_b, dtype=torch.int64),
        req_pool_indices_cpu=torch.arange(real_b, dtype=torch.int64),
        mrope_positions=None,
        extend_seq_lens=(
            torch.full((real_b,), real_t // real_b, dtype=torch.int32)
            if extend
            else None
        ),
        extend_prefix_lens=(torch.zeros(real_b, dtype=torch.int32) if extend else None),
        extend_seq_lens_cpu=([real_t // real_b] * real_b if extend else None),
        extend_prefix_lens_cpu=([0] * real_b if extend else None),
        extend_logprob_start_lens_cpu=([0] * real_b if extend else None),
        forward_mode=_ForwardModeMock(mode),
        return_logprob=return_logprob,
        dp_padding_mode=None,
        global_num_tokens_cpu=None,
        # Real ForwardBatch always carries spec_info (None when not spec).
        spec_info=None,
        mm_inputs=None,
    )


class TestDummyFieldPadCoverage(CustomTestCase):
    """Bookkeeping: every device tensor the precompile dummies place on a
    ForwardBatch must be in the shim's pad lists, or live batches reach the
    compiled graphs with shapes the trace never saw (XLA e0102 — happened
    with `extend_start_loc`). Host mirrors (`*_cpu`) are not graph inputs."""

    def test_extend_dummy_device_tensors_are_padded(self):
        dummy = _DummyForwardBatchFields.default(
            num_tokens=8,
            bs=4,
            is_extend=True,
            tokens_per_req=2,
            device="cpu",
            slot=0,
            out_cache_loc_dtype=torch.int32,
        )
        device_tensor_fields = {
            name
            for name in dummy.__struct_fields__
            if isinstance(getattr(dummy, name), torch.Tensor)
            and not name.endswith("_cpu")
        }
        self.assertLessEqual(device_tensor_fields, set(_TOKEN_FIELDS) | set(_BS_FIELDS))


class TestHostStagedPrefillInputs(CustomTestCase):
    """Arbitrary prefill lengths must not reach the device before bucketing."""

    def test_default_platform_keeps_device_input_contract(self):
        batch = ScheduleBatch(
            reqs=[], device="meta", prefill_input_ids_cpu=torch.arange(3)
        )
        with patch(
            "sglang.srt.managers.schedule_batch.current_platform",
            SRTPlatform(),
        ):
            resolve_forward_inputs(batch, None)
        self.assertEqual(batch.input_ids.device.type, "meta")
        self.assertEqual(batch.input_ids.shape, (3,))
        self.assertIsNone(batch.prefill_input_ids_cpu)

    def test_mixed_and_speculative_batches_keep_the_device_contract(self):
        with (
            get_context().override_server_args(enable_torch_compile=True),
            tpu_envs.SGLANG_TPU_HOST_PADDING.override(True),
        ):
            allocator = TpuPagedTokenToKVPoolAllocator(
                size=32,
                page_size=4,
                dtype=torch.bfloat16,
                device="meta",
                kvcache=None,
                need_sort=False,
            )
            batch = ScheduleBatch(
                reqs=[
                    SimpleNamespace(
                        positional_embed_overrides=None,
                        input_embeds=None,
                        beam_group=None,
                    )
                ],
                device="meta",
                spec_algorithm=SpeculativeAlgorithm.NONE,
                model_config=SimpleNamespace(
                    is_generation=True,
                    is_encoder_decoder=False,
                    is_multimodal=False,
                    use_ngram_embedding=False,
                    use_engram=False,
                ),
                token_to_kv_pool_allocator=allocator,
                req_to_token_pool=SimpleNamespace(req_to_token=torch.empty(2, 32)),
            )
            platform = TpuSRTPlatform()
            self.assertEqual(platform.get_batch_input_device(batch), "cpu")
            with tpu_envs.SGLANG_TPU_HOST_PADDING.override(False):
                self.assertIsNone(platform.get_batch_input_device(batch))
            with get_schedule().override(enable_mixed_chunk=True):
                self.assertIsNone(platform.get_batch_input_device(batch))
            batch.spec_algorithm = SpeculativeAlgorithm.EAGLE
            self.assertIsNone(platform.get_batch_input_device(batch))
            batch.spec_algorithm = SpeculativeAlgorithm.NONE
            batch.model_config.is_multimodal = True
            self.assertEqual(platform.get_batch_input_device(batch), "cpu")

    def test_host_inputs_survive_forward_construction_and_upload_after_padding(self):
        class HostInputPlatform(SRTPlatform):
            calls = 0

            def get_batch_input_device(self, batch):
                self.calls += 1
                if self.calls > 1:
                    raise AssertionError("placement must be resolved once per prefill")
                return "cpu"

        runner = SimpleNamespace(
            device="meta",
            is_draft_worker=False,
            kv_index_translator=SimpleNamespace(rebind_write_loc=lambda _: None),
            ngram_embedding_manager=SimpleNamespace(enabled=False),
            model_config=SimpleNamespace(
                model_is_mrope=True, requires_mm_token_modalities=False
            ),
            lora_manager=None,
            prefill_attention_backend_str="torch_native",
        )
        with (
            get_context().override_server_args(),
            get_parallel().override(attn_dcp_size=1),
            tpu_envs.SGLANG_TPU_HOST_PADDING.override(True),
            patch(
                "sglang.srt.managers.schedule_batch.SamplingBatchInfo.from_schedule_batch",
                return_value=None,
            ),
        ):
            for n, has_image in (
                (1, False),
                (3, True),
                (8, False),
                (9, True),
                (17, True),
            ):
                with self.subTest(tokens=n, multimodal=has_image):
                    platform = HostInputPlatform()
                    req = Req(
                        "0", "", array("q", range(n + 3)), SamplingParams(), False
                    )
                    req.refresh_fill_ids()
                    req.prefix_indices = torch.arange(3)
                    req.set_extend_range(3, n + 3)
                    expected_mrope = torch.arange(3, n + 3).repeat(3, 1)
                    if has_image:
                        spatial_offsets = torch.tensor([[0], [10], [20]])
                        req.multimodal_inputs = SimpleNamespace(
                            mrope_positions=torch.arange(n + 3).repeat(3, 1)
                            + spatial_offsets,
                            vision_position_ids=None,
                            visible_frame_counts=None,
                        )
                        expected_mrope = expected_mrope + spatial_offsets
                    batch = ScheduleBatch(
                        reqs=[req],
                        device="meta",
                        model_config=SimpleNamespace(
                            is_encoder_decoder=False, vocab_size=32
                        ),
                        spec_algorithm=SpeculativeAlgorithm.NONE,
                    )

                    def allocate(batch):
                        # Allocation and the later upload must share the decision.
                        self.assertEqual(batch.input_tensor_device, "cpu")
                        return (
                            torch.arange(n, dtype=torch.int32),
                            torch.empty(1, dtype=torch.int64, device="meta"),
                            torch.tensor([0]),
                        )

                    with (
                        patch(
                            "sglang.srt.managers.schedule_batch.current_platform",
                            platform,
                        ),
                        patch(
                            "sglang.srt.managers.schedule_batch.alloc_for_extend",
                            allocate,
                        ),
                    ):
                        batch.prepare_for_extend()
                    resolve_forward_inputs(batch, None)
                    self.assertEqual(batch.input_ids.device.type, "cpu")
                    fb = ForwardBatch.init_new(
                        batch,
                        runner,
                        capture_hidden_mode=CaptureHiddenMode.NULL,
                        return_hidden_states_before_norm=False,
                    )
                    torch.testing.assert_close(fb.positions, torch.arange(3, n + 3))
                    torch.testing.assert_close(fb.mrope_positions, expected_mrope)
                    for field in (
                        "input_ids",
                        "positions",
                        "mrope_positions",
                        "out_cache_loc",
                        "extend_seq_lens",
                        "extend_prefix_lens",
                        "extend_start_loc",
                    ):
                        self.assertEqual(getattr(fb, field).device.type, "cpu")
                    policy = DecoderTpuCompilePolicy()
                    policy.model_runner = runner
                    policy.t_bucketer = _bucketer("Token", [8, 16, 32])
                    policy.b_bucketer = _bucketer("Bs", [1, 4])
                    policy.extend_bs = 4
                    policy.prepare_forward_batch(fb)
                    self.assertEqual(
                        fb.input_ids.shape, (8 if n <= 8 else 16 if n <= 16 else 32,)
                    )
                    self.assertEqual(fb.extend_start_loc.shape, (4,))
                    for field in (
                        "input_ids",
                        "positions",
                        "out_cache_loc",
                        "mrope_positions",
                        "extend_seq_lens",
                        "extend_prefix_lens",
                        "extend_start_loc",
                    ):
                        self.assertEqual(getattr(fb, field).device.type, "meta")
                    # Preparation owns the forward snapshot, not scheduler inputs.
                    self.assertEqual(batch.input_ids.device.type, "cpu")
                    self.assertEqual(batch.input_ids.numel(), n)
                    self.assertEqual(fb.req_pool_indices_cpu.device.type, "cpu")
                    self.assertEqual(platform.calls, 1)

                    batch.sampling_info = SimpleNamespace(
                        penalizer_orchestrator=SimpleNamespace(is_required=False)
                    )
                    with patch(
                        "sglang.srt.managers.schedule_batch.alloc_for_decode",
                        return_value=torch.empty(1, dtype=torch.int32, device="meta"),
                    ):
                        batch.prepare_for_decode()
                    self.assertEqual(batch.out_cache_loc.device.type, "meta")

    def test_extend_allocator_can_retain_host_output_without_changing_default(self):
        with get_context().override_server_args():
            for output_device in (None, "cpu"):
                allocator = TpuPagedTokenToKVPoolAllocator(
                    size=32,
                    page_size=4,
                    dtype=torch.bfloat16,
                    device="meta",
                    kvcache=None,
                    need_sort=False,
                )
                indices = allocator.alloc_extend(
                    torch.tensor([0]),
                    torch.tensor([0]),
                    torch.tensor([5]),
                    torch.tensor([5]),
                    torch.tensor([-1]),
                    5,
                    **({"return_host": True} if output_device else {}),
                )
                self.assertEqual(indices.device.type, output_device or "meta")
                self.assertEqual(allocator.available_size(), 24)
                if output_device == "cpu":
                    self.assertEqual(indices.tolist(), [4, 5, 6, 7, 8])


class TestZeroPadDim0(CustomTestCase):
    def test_pads_zeros_at_tail(self):
        out = _zero_pad_dim0(_t(1, 2, 3), target_size=5)
        self.assertEqual(out.tolist(), [1, 2, 3, 0, 0])

    def test_no_op_when_already_at_size(self):
        x = _t(1, 2, 3)
        self.assertIs(_zero_pad_dim0(x, target_size=3), x)
        self.assertIs(_zero_pad_dim0(x, target_size=2), x)

    def test_pads_2d_tensor_along_dim0(self):
        x = torch.ones((3, 4), dtype=torch.float32)
        out = _zero_pad_dim0(x, target_size=5)
        self.assertEqual(out.shape, (5, 4))
        self.assertTrue(torch.equal(out[:3], x))
        self.assertTrue(torch.equal(out[3:], torch.zeros((2, 4), dtype=torch.float32)))


class TestPadForwardBatchInputs(CustomTestCase):
    """Pads ForwardBatch fields to bucket size before init_forward_metadata."""

    def test_speculative_padding_roundtrip_preserves_rows_and_metadata(self):
        for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2):
            for width in (1, 4):
                with self.subTest(mode=mode, width=width):
                    real_b, real_q = 3, 3 * width
                    # Retain extend lists in both modes to check their restoration.
                    fb = _forward_batch(real_q, real_b, mode="draft_extend_v2")
                    fb.forward_mode = mode
                    fb.input_embeds = torch.arange(real_q * 2).reshape(real_q, 2)
                    fb.mm_input_embeds = fb.input_embeds.clone()
                    fb.mrope_positions = torch.arange(3 * real_q).reshape(3, real_q)
                    fb.mm_inputs = [object(), None, object()]
                    fb.spec_info = SimpleNamespace(
                        num_tokens_per_req=width,
                        hidden_states=torch.arange(real_q * 5).reshape(real_q, 5),
                    )
                    original = (
                        fb.input_ids.clone(),
                        fb.out_cache_loc.clone(),
                        fb.input_embeds.clone(),
                        fb.mrope_positions.clone(),
                        fb.spec_info.hidden_states.clone(),
                    )
                    mm_inputs = list(fb.mm_inputs)
                    # Reuse the same batch: a second pad must not retain ghost rows.
                    for _ in range(2):
                        q, b = _pad_forward_batch_inputs(
                            fb, _bucketer("Token", [8, 16]), _bucketer("Bs", [1, 4])
                        )
                        self.assertEqual((q, b), (real_q, real_b))
                        self.assertEqual(
                            fb.spec_info.hidden_states.shape, (4 * width, 5)
                        )
                        self.assertEqual(fb.mrope_positions.shape, (3, 4 * width))
                        self.assertEqual(
                            fb.out_cache_loc[real_q:].tolist(), [0] * width
                        )
                        self.assertEqual(fb.mm_inputs, mm_inputs + [None])
                        _restore_forward_batch_inputs(fb, q, b)
                        restored = (
                            fb.input_ids,
                            fb.out_cache_loc,
                            fb.input_embeds,
                            fb.mrope_positions,
                            fb.spec_info.hidden_states,
                        )
                        for before, after in zip(original, restored):
                            self.assertTrue(torch.equal(before, after))
                        self.assertEqual(fb.batch_size, real_b)
                        self.assertEqual(fb.mm_inputs, mm_inputs)
                        self.assertEqual(fb.extend_seq_lens_cpu, [width] * real_b)

    def test_extend_single_req(self):
        # tokens=4 -> max(tokens, bs_cap=4) -> bucket 8.
        # Extend bs pads to the fixed cap so precompile only needs one bs.
        fb = _forward_batch(real_t=4, real_b=1, mode="extend")
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [8, 64]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual((real_q, real_b), (4, 1))
        self.assertEqual(fb.input_ids.shape[0], 8)
        self.assertEqual(fb.positions.shape[0], 8)
        self.assertEqual(fb.out_cache_loc.shape[0], 8)
        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.extend_seq_lens.shape[0], 4)

    def test_extend_batched(self):
        # tokens=10 -> 16; bs=3 -> 4; ghost slots zero-filled.
        fb = _forward_batch(real_t=10, real_b=3, mode="extend")
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [16, 64]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual((real_q, real_b), (10, 3))
        self.assertEqual(fb.input_ids.shape[0], 16)
        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.req_pool_indices.shape[0], 4)
        self.assertEqual(fb.req_pool_indices_cpu.tolist(), [0, 1, 2, 0])
        self.assertEqual(fb.extend_seq_lens.shape[0], 4)
        self.assertEqual(fb.seq_lens[:3].tolist(), [100, 100, 100])
        self.assertEqual(fb.seq_lens[3].item(), 0)
        self.assertEqual(fb.extend_seq_lens[3].item(), 0)
        self.assertEqual(fb.extend_seq_lens_cpu, [3, 3, 3, 0])
        self.assertEqual(fb.extend_prefix_lens_cpu, [0, 0, 0, 0])
        self.assertEqual(fb.extend_logprob_start_lens_cpu, [0, 0, 0, 0])
        self.assertEqual(fb.batch_size, 4)

    def test_mamba_tracking_padding_roundtrip(self):
        fb = _forward_batch(real_t=10, real_b=3, mode="extend")
        original = {
            "mamba_track_indices": _t(7, 8, 9),
            "mamba_track_mask": _t(True, False, True, dtype=torch.bool),
            "mamba_track_seqlens": _t(64, -1, 128),
        }
        for name, value in original.items():
            setattr(fb, name, value)
        q, b = _pad_forward_batch_inputs(
            fb, _bucketer("Token", [16, 64]), _bucketer("Bs", [1, 4])
        )
        for name, value in original.items():
            padded = getattr(fb, name)
            torch.testing.assert_close(padded[:3], value)
            self.assertEqual(padded.shape[0], 4)
            self.assertEqual(padded[3].item(), 0)
        _restore_forward_batch_inputs(fb, q, b)
        for name, value in original.items():
            torch.testing.assert_close(getattr(fb, name), value)

    def test_extend_batch_cap_does_not_exceed_token_ceiling(self):
        fb = _forward_batch(real_t=4, real_b=2, mode="extend")
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [4]),
            b_bucketer=_bucketer("Bs", [1, 8]),
        )
        self.assertEqual(fb.input_ids.shape[0], 4)
        self.assertEqual(fb.seq_lens.shape[0], 4)

    def test_extend_cap_is_independent_from_decode_buckets(self):
        fb = _forward_batch(real_t=10, real_b=3, mode="extend")
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [16, 64]),
            b_bucketer=_bucketer("Bs", [1, 4, 16, 64, 256]),
            extend_bs=4,
        )
        self.assertEqual(fb.input_ids.shape[0], 16)
        self.assertEqual(fb.seq_lens.shape[0], 4)

    def test_extend_rejects_batch_above_compile_cap(self):
        fb = _forward_batch(real_t=10, real_b=5, mode="extend")
        with self.assertRaisesRegex(ValueError, "exceeds its compile cap 4"):
            _pad_forward_batch_inputs(
                fb,
                t_bucketer=_bucketer("Token", [16, 64]),
                b_bucketer=_bucketer("Bs", [1, 4, 16, 64, 256]),
                extend_bs=4,
            )

    def test_mixed_batch_uses_decode_bucket_above_prefill_cap(self):
        fb = _forward_batch(real_t=80, real_b=56, mode="mixed")
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [64, 128]),
            b_bucketer=_bucketer("Bs", [16, 32, 64]),
            extend_bs=16,
        )

        self.assertEqual((real_q, real_b), (80, 56))
        self.assertEqual(fb.input_ids.shape[0], 128)
        self.assertEqual(fb.seq_lens.shape[0], 64)
        self.assertEqual(fb.extend_seq_lens.shape[0], 64)
        self.assertEqual(fb.batch_size, 64)

    def test_mixed_batch_shapes_are_all_precompiled(self):
        t_bucketer = _bucketer("Token", [16, 64, 128])
        b_bucketer = _bucketer("Bs", [16, 32, 64])
        policy = DecoderTpuCompilePolicy(
            t_bucketer=t_bucketer,
            b_bucketer=b_bucketer,
        )
        policy.extend_bs = 16
        # _extend_compile_cases reads enable_mixed_chunk off the schedule bag.
        with get_context().override_server_args(enable_mixed_chunk=True):
            precompiled_shapes = set(policy._extend_compile_shapes())

        for real_q in (1, 16, 17, 64, 65, 128):
            for real_b in (1, 16, 17, 32, 33, 64):
                with self.subTest(real_q=real_q, real_b=real_b):
                    fb = _forward_batch(real_t=real_q, real_b=real_b, mode="mixed")
                    _pad_forward_batch_inputs(
                        fb,
                        t_bucketer=t_bucketer,
                        b_bucketer=b_bucketer,
                        extend_bs=policy.extend_bs,
                    )
                    padded_shape = (
                        fb.input_ids.shape[0],
                        fb.seq_lens.shape[0],
                    )
                    self.assertIn(padded_shape, precompiled_shapes)

    def test_target_verify_pins_token_dim_to_spec_width(self):
        # Verify query dim is cap * num_tokens_per_req, not a token bucket.
        # The extend branch would otherwise over-pad to the next token bucket.
        fb = _forward_batch(real_t=8, real_b=2, mode="target_verify")
        fb.spec_info = SimpleNamespace(draft_token_num=3, num_tokens_per_req=4)
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [64, 128]),  # must NOT be used for verify
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual((real_q, real_b), (8, 2))
        # pad_b = 4 and width = 4, so pad_t = 16 rather than token bucket 64.
        self.assertEqual(fb.input_ids.shape[0], 16)
        self.assertEqual(fb.positions.shape[0], 16)
        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.batch_size, 4)

    def test_draft_extend_v2_uses_forward_mode_api(self):
        fb = _forward_batch(real_t=8, real_b=2, mode="draft_extend_v2")
        fb.spec_info = SimpleNamespace(
            num_tokens_per_req=4,
            hidden_states=torch.zeros(8, 16),
        )
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [64]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual(fb.input_ids.shape[0], 16)
        self.assertEqual(fb.out_cache_loc.shape[0], 16)
        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.extend_seq_lens.tolist(), [4, 4, 0, 0])
        self.assertEqual(fb.extend_seq_lens_cpu, [4, 4, 0, 0])
        self.assertEqual(fb.batch_size, 4)

    def test_draft_extend_v2_rejects_unset_num_tokens_per_req(self):
        fb = _forward_batch(real_t=8, real_b=2, mode="draft_extend_v2")
        fb.spec_info = SimpleNamespace(num_tokens_per_req=-1)

        with self.assertRaisesRegex(AssertionError, "num_tokens_per_req > 0"):
            _pad_forward_batch_inputs(
                fb,
                t_bucketer=_bucketer("Token", [64]),
                b_bucketer=_bucketer("Bs", [1, 4]),
            )

    def test_decode_uses_bs_bucketer_for_tokens(self):
        # In DECODE, token-axis tensors share the bs bucketer
        # (one query token per running seq).
        fb = _forward_batch(real_t=3, real_b=3, mode="decode")
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [16, 64]),  # would over-pad if (wrongly) used
            b_bucketer=_bucketer("Bs", [1, 4, 8]),
        )
        self.assertEqual(fb.input_ids.shape[0], 4)
        self.assertEqual(fb.positions.shape[0], 4)
        self.assertEqual(fb.seq_lens.shape[0], 4)

    def test_no_op_when_at_bucket(self):
        # Shapes already at bucket: tensors unchanged (same identity).
        fb = _forward_batch(real_t=4, real_b=4, mode="extend")
        orig_input_ids = fb.input_ids
        orig_seq_lens = fb.seq_lens
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [4]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertIs(fb.input_ids, orig_input_ids)
        self.assertIs(fb.seq_lens, orig_seq_lens)

    def test_mm_inputs_none_becomes_a_padded_list(self):
        fb = _forward_batch(real_t=4, real_b=1, mode="extend")
        fb.mm_inputs = None
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [8, 64]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual(fb.mm_inputs, [None] * 4)

    def test_decode_cpu_list_fields_stay_none(self):
        # Non-extend batches keep the upstream None for the extend_*_cpu
        # lists. Consumers gate on forward_mode.is_extend(), and the decode
        # precompile dummy also carries None.
        fb = _forward_batch(real_t=2, real_b=2, mode="decode")
        fb.extend_seq_lens_cpu = None
        fb.extend_prefix_lens_cpu = None
        fb.extend_logprob_start_lens_cpu = None
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [16]),
            b_bucketer=_bucketer("Bs", [4]),
        )
        self.assertIsNone(fb.extend_seq_lens_cpu)
        self.assertIsNone(fb.extend_prefix_lens_cpu)
        self.assertIsNone(fb.extend_logprob_start_lens_cpu)

    def test_optional_fields_left_alone_when_none(self):
        # extend_* are None in DECODE; padder must not crash.
        fb = _forward_batch(real_t=2, real_b=2, mode="decode")
        self.assertIsNone(fb.extend_seq_lens)
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [16]),
            b_bucketer=_bucketer("Bs", [4]),
        )
        self.assertIsNone(fb.extend_seq_lens)

    def test_input_logprob_pads_like_any_other_extend(self):
        # Input/prompt (echo) logprobs used to be rejected here. They are now
        # served by capturing every prefill row and gathering eagerly after the
        # forward, which leaves padding itself unaffected: this batch must
        # bucket exactly like the output-logprob case below.
        fb = _forward_batch(real_t=4, real_b=1, return_logprob=True)
        fb.extend_logprob_start_lens_cpu = [0]  # extend_len(4) - start_len(0) > 0
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [8, 64]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual((real_q, real_b), (4, 1))
        self.assertEqual(fb.input_ids.shape[0], 8)

    def test_output_logprob_passes(self):
        # OUTPUT (next-token) logprobs only: start_len == extend_len, so there
        # are no prompt positions to score inside the compiled forward. The
        # guard must let it through (the eager Sampler produces them afterwards).
        fb = _forward_batch(real_t=4, real_b=1, return_logprob=True)
        fb.extend_logprob_start_lens_cpu = [4]  # extend_len - start_len == 0
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [8, 64]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual((real_q, real_b), (4, 1))
        self.assertEqual(fb.input_ids.shape[0], 8)

    def test_decode_output_logprob_passes(self):
        # Decode-mode logprob requests are output-only (not an extend batch), so
        # the input-logprob path never triggers and the request must pass.
        fb = _forward_batch(real_t=2, real_b=2, mode="decode", return_logprob=True)
        _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [16]),
            b_bucketer=_bucketer("Bs", [4]),
        )
        self.assertEqual(fb.input_ids.shape[0], 4)

    def test_target_verify_output_logprob_passes(self):
        # Spec-decode target-verify is an extend-family mode (is_extend() is
        # True), but it never produces INPUT/prompt logprobs: LogitsProcessor
        # treats it like decode (_get_pruned_states -> pruned_states = all hidden
        # states, input_logprob_indices = None; from_forward_batch excludes it
        # from extend_return_logprob). So there are no dynamically-pruned states
        # and the guard must let it through even with return_logprob set. The
        # batch pads to the static verify shape (pad_b * num_tokens_per_req).
        fb = _forward_batch(
            real_t=6, real_b=3, mode="target_verify", return_logprob=True
        )
        fb.spec_info = SimpleNamespace(
            draft_token_num=1, num_tokens_per_req=2, hidden_states=None
        )
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [8, 64]),
            b_bucketer=_bucketer("Bs", [4]),
        )
        self.assertEqual((real_q, real_b), (6, 3))
        self.assertEqual(fb.input_ids.shape[0], 8)  # pad_b(4) * width(2)

    def test_pads_and_restores_input_embeds(self):
        fb = _forward_batch(real_t=4, real_b=1, mode="extend")
        fb.input_embeds = torch.ones((4, 16), dtype=torch.bfloat16)
        real_q, real_b = _pad_forward_batch_inputs(
            fb,
            t_bucketer=_bucketer("Token", [8, 64]),
            b_bucketer=_bucketer("Bs", [1, 4]),
        )
        self.assertEqual((real_q, real_b), (4, 1))
        self.assertEqual(fb.input_embeds.shape, (8, 16))
        self.assertTrue(
            torch.equal(fb.input_embeds[:4], torch.ones((4, 16), dtype=torch.bfloat16))
        )
        self.assertTrue(
            torch.equal(fb.input_embeds[4:], torch.zeros((4, 16), dtype=torch.bfloat16))
        )

        _restore_forward_batch_inputs(fb, real_q, real_b)
        self.assertEqual(fb.input_embeds.shape, (4, 16))


class TestEagle3SpecDecodeCompilePolicyPadding(CustomTestCase):
    def test_draft_decode_pads_req_pool_indices_cpu_to_bucket(self):
        policy = object.__new__(Eagle3SpecDecodeCompilePolicyMixin)
        policy.b_bucketer = _bucketer("Bs", [1, 2, 4])
        fb = _forward_batch(real_t=3, real_b=3, mode="decode")

        policy._pad_draft_decode(fb)

        self.assertEqual(fb.seq_lens.shape[0], 4)
        self.assertEqual(fb.req_pool_indices.tolist(), [0, 1, 2, 0])
        self.assertEqual(fb.req_pool_indices_cpu.tolist(), [0, 1, 2, 0])
        self.assertEqual(fb.batch_size, 4)


class TestDefaultKwargHelpers(CustomTestCase):
    def test_static_default_kwargs_filters_to_simple_defaults(self):
        def forward(a, b=None, c=False, d=3, e=1.5, f="x", g=torch.zeros(1)):
            pass

        compiled = MagicMock()
        compiled.__signature__ = inspect.signature(forward)
        self.assertEqual(
            _static_default_kwargs(compiled),
            {"b": None, "c": False, "d": 3, "e": 1.5, "f": "x"},
        )

    def test_static_default_kwargs_missing_signature_returns_empty(self):
        self.assertEqual(_static_default_kwargs(MagicMock()), {})

    def test_matches_default_is_type_strict(self):
        self.assertTrue(_matches_default(None, None))
        self.assertTrue(_matches_default(3, 3))
        self.assertFalse(_matches_default(False, 0))
        self.assertFalse(_matches_default(3.0, 3))


class TestCompiledForwardShim(CustomTestCase):
    """Adapts compiled forward calls and outputs."""

    @staticmethod
    def _forward_batch(**kwargs):
        defaults = {
            "is_prefill_only": False,
            "mm_input_embeds": None,
            "input_embeds": None,
            "seq_lens": torch.zeros(4, dtype=torch.int32),
            "spec_info": None,
            # The logprob pin reads these on every call, and a real
            # ForwardBatch always carries them. Default to decode so the
            # existing cases stay on the path that touches nothing.
            "forward_mode": _ForwardModeMock("decode"),
            "return_logprob": False,
            "top_logprobs_nums": None,
            "token_ids_logprobs": None,
            "extend_input_logprob_token_ids_gpu": None,
            "extend_seq_lens_cpu": None,
            "extend_logprob_start_lens_cpu": None,
            "capture_hidden_mode": CaptureHiddenMode.NULL,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_hidden_state_boundary_keeps_mm_embeds_on_the_batch(self):
        """A boundary that returns hidden states (Gemma4, Qwen3.5 wrappers) has
        nowhere to re-attach the multimodal embeds; they stay on the batch for
        the wrapper's eager tail instead of raising on the tensor."""
        hidden = torch.zeros(8, 4)
        seen_by_graph = []

        def compiled(input_ids, positions, forward_batch, *args, **kwargs):
            seen_by_graph.append(forward_batch.mm_input_embeds)
            return hidden

        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        embeds = torch.ones(3, 4)
        fb = self._forward_batch(
            input_ids=torch.arange(8, dtype=torch.int32), mm_input_embeds=embeds
        )
        out = shim(None, None, fb)
        self.assertIs(out, hidden)
        self.assertIs(fb.mm_input_embeds, embeds)
        # The graph itself never saw them.
        self.assertEqual(seen_by_graph, [None])

    def test_logits_boundary_gets_mm_embeds_re_attached(self):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        result = LogitsProcessorOutput(next_token_logits=torch.zeros(8, 4))
        compiled = MagicMock(return_value=result)
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        embeds = torch.ones(3, 4)
        fb = self._forward_batch(
            input_ids=torch.arange(8, dtype=torch.int32), mm_input_embeds=embeds
        )
        out = shim(None, None, fb)
        self.assertIs(out.mm_input_embeds, embeds)
        self.assertIs(fb.mm_input_embeds, embeds)

    def test_draft_worker_graph_keeps_the_mm_embeds(self):
        # Draft models read forward_batch.mm_input_embeds inside their forward.
        seen_by_graph = []

        def compiled(input_ids, positions, forward_batch, *args, **kwargs):
            seen_by_graph.append(forward_batch.mm_input_embeds)
            return torch.zeros(8, 4)

        policy = DecoderTpuCompilePolicy()
        policy.model_runner = SimpleNamespace(
            model=SimpleNamespace(),
            token_to_kv_pool=SimpleNamespace(layer_transfer_counter=None),
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            is_draft_worker=True,
        )
        shim = CompiledForwardShim(compiled, policy=policy)
        embeds = torch.ones(3, 4)
        fb = self._forward_batch(
            input_ids=torch.arange(8, dtype=torch.int32), mm_input_embeds=embeds
        )
        shim(None, None, fb)
        self.assertEqual(len(seen_by_graph), 1)
        torch.testing.assert_close(seen_by_graph[0][:3], embeds)
        torch.testing.assert_close(fb.mm_input_embeds[:3], embeds)

    def test_recompilation_check_runs_when_forward_raises(self):
        """A compile that raises mid-forward must still be reported."""
        error = torch._dynamo.exc.RecompileError("fail_on_recompile")
        compiled = MagicMock(side_effect=error)
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        checker = MagicMock()
        shim.set_recompilation_checker(checker, model_label="M")
        fb = self._forward_batch(input_ids=torch.arange(8, dtype=torch.int32))
        with self.assertRaises(torch._dynamo.exc.RecompileError) as raised:
            shim(None, None, fb)
        self.assertIs(raised.exception, error)
        checker.check.assert_called_once()

    def test_tracing_never_reaches_the_recompilation_checker(self):
        """The fused draft loop traces this call, so the checker must stay out
        of that graph's guards."""
        compiled = MagicMock(return_value="out")
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        checker = MagicMock()
        shim.set_recompilation_checker(checker, model_label="M")
        fb = self._forward_batch(input_ids=torch.arange(8, dtype=torch.int32))
        with patch(
            "sglang_tpu.compilation.compiled_forward_shim.torch.compiler.is_compiling",
            return_value=True,
        ):
            shim(None, None, fb)
        checker.check.assert_not_called()
        compiled.assert_called_once()

    def test_recompilation_context_reports_real_and_padded_shapes(self):
        """The padded shapes are read at entry, before output trimming."""
        compiled = MagicMock(
            return_value=LogitsProcessorOutput(
                next_token_logits=torch.zeros(64, 10),
                hidden_states=torch.zeros(1024, 5),
            )
        )
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        checker = MagicMock()
        shim.set_recompilation_checker(checker, model_label="M")
        fb = self._forward_batch(
            forward_mode=_ForwardModeMock("extend"),
            input_ids=torch.arange(1024, dtype=torch.int32),
            seq_lens=torch.zeros(64, dtype=torch.int32),
            extend_seq_lens_cpu=[952],
            extend_logprob_start_lens_cpu=[0],
            _tpu_real_q=952,
            _tpu_real_b=1,
        )
        shim(None, None, fb)
        context_fn = checker.check.call_args.kwargs["context_fn"]
        self.assertEqual(context_fn(), "M EXTEND real=q952,b1 pad=q1024,b64")

    def test_slices_logits_and_hidden_states(self):
        padded_logits = torch.arange(4 * 10, dtype=torch.float32).reshape(4, 10)
        padded_hidden = torch.arange(8 * 5, dtype=torch.float32).reshape(8, 5)
        compiled = MagicMock(
            return_value=LogitsProcessorOutput(
                next_token_logits=padded_logits,
                hidden_states=padded_hidden,
            )
        )
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        fb = self._forward_batch(_tpu_real_q=4, _tpu_real_b=2)
        out = shim(None, None, fb)
        self.assertEqual(out.next_token_logits.shape, (2, 10))
        self.assertEqual(out.hidden_states.shape, (4, 5))
        self.assertTrue(torch.equal(out.next_token_logits, padded_logits[:2]))
        self.assertTrue(torch.equal(out.hidden_states, padded_hidden[:4]))

        # When real_b == pad_b and real_q == pad_q, trim_next_token_outputs
        # must not materialize a no-op slice view.
        exact_logits = torch.zeros(4, 10)
        exact_hidden = torch.zeros(8, 5)
        shim_exact = CompiledForwardShim(
            MagicMock(
                return_value=LogitsProcessorOutput(
                    next_token_logits=exact_logits,
                    hidden_states=exact_hidden,
                )
            ),
            policy=DecoderTpuCompilePolicy(),
        )
        out_exact = shim_exact(
            None, None, self._forward_batch(_tpu_real_q=8, _tpu_real_b=4)
        )
        self.assertIs(out_exact.next_token_logits, exact_logits)
        self.assertIs(out_exact.hidden_states, exact_hidden)

    def test_defers_output_unpadding_for_chained_forwards(self):
        padded = LogitsProcessorOutput(
            next_token_logits=torch.zeros(4, 10),
            hidden_states=torch.zeros(8, 5),
        )
        shim = CompiledForwardShim(
            MagicMock(return_value=padded), policy=DecoderTpuCompilePolicy()
        )
        fb = self._forward_batch(
            _tpu_real_q=4,
            _tpu_real_b=2,
            batch_size=4,
            spec_info=SimpleNamespace(keep_outputs_padded=True),
        )

        out = shim(None, None, fb)

        self.assertIs(out, padded)
        self.assertEqual(out.next_token_logits.shape, (4, 10))
        self.assertEqual(out.hidden_states.shape, (8, 5))
        self.assertEqual(fb.batch_size, 4)

    def test_passes_args_through(self):
        compiled = MagicMock(
            return_value=LogitsProcessorOutput(next_token_logits=torch.empty(0))
        )
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        fb = self._forward_batch(_tpu_real_q=4, _tpu_real_b=2)
        shim("ids", "pos", fb, "extra1", kw="extra2")
        compiled.assert_called_once_with("ids", "pos", fb, "extra1", kw="extra2")

    def test_non_spec_sampling_keeps_physical_outputs_and_metadata(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        from sglang_tpu.compilation.decoder_compile_policy import (
            Qwen3VLDecoderTpuCompilePolicy,
        )

        # A full decoder returns logits; Qwen-VL returns hidden states to an
        # outer logits processor. Neither may unpad the forward metadata.
        for hidden_boundary in (False, True):
            with self.subTest(hidden_boundary=hidden_boundary):
                tensor = torch.arange(80).reshape(8, 10).float()
                padded = (
                    tensor
                    if hidden_boundary
                    else LogitsProcessorOutput(next_token_logits=tensor)
                )
                policy_cls = (
                    Qwen3VLDecoderTpuCompilePolicy
                    if hidden_boundary
                    else DecoderTpuCompilePolicy
                )
                policy = policy_cls()
                policy.keep_outputs_padded = MagicMock(return_value=True)
                shim = CompiledForwardShim(lambda *args: padded, policy=policy)
                fb = self._forward_batch(
                    _tpu_real_q=3,
                    _tpu_real_b=3,
                    batch_size=8,
                    input_ids=torch.arange(8),
                    positions=torch.arange(8),
                    out_cache_loc=torch.arange(8),
                    seq_lens=torch.ones(8),
                    orig_seq_lens=torch.ones(8),
                    mm_input_embeds=torch.ones(8, 4),
                )
                original = {
                    name: getattr(fb, name)
                    for name in (
                        "input_ids",
                        "positions",
                        "out_cache_loc",
                        "seq_lens",
                        "orig_seq_lens",
                        "mm_input_embeds",
                    )
                }
                slices = []

                class ObserveSlices(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        if func == torch.ops.aten.slice.Tensor:
                            slices.append(func)
                        return func(*args, **(kwargs or {}))

                with ObserveSlices():
                    out = shim(None, None, fb)
                self.assertIs(out, padded)
                self.assertEqual(fb.batch_size, 8)
                for name, value in original.items():
                    self.assertIs(getattr(fb, name), value)
                self.assertEqual(slices, [])

    def test_preserves_compiled_signature_for_later_inspection(self):
        def forward(input_ids, positions, forward_batch, pp_proxy_tensors=None):
            pass

        compiled = MagicMock(return_value=SimpleNamespace())
        compiled.__signature__ = inspect.signature(forward)
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        self.assertIn("pp_proxy_tensors", inspect.signature(shim).parameters)

    def test_drops_default_kwargs_before_compiled_call(self):
        def forward(input_ids, positions, forward_batch, pp_proxy_tensors=None):
            pass

        compiled = MagicMock(return_value=SimpleNamespace())
        compiled.__signature__ = inspect.signature(forward)
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        fb = self._forward_batch(_tpu_real_q=4, _tpu_real_b=2)
        shim("ids", "pos", fb, pp_proxy_tensors=None)
        compiled.assert_called_once_with("ids", "pos", fb)

    def test_keeps_non_default_kwargs_before_compiled_call(self):
        def forward(input_ids, positions, forward_batch, get_embedding=False):
            pass

        compiled = MagicMock(return_value=SimpleNamespace())
        compiled.__signature__ = inspect.signature(forward)
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        fb = self._forward_batch(_tpu_real_q=4, _tpu_real_b=2)
        shim("ids", "pos", fb, get_embedding=True)
        compiled.assert_called_once_with("ids", "pos", fb, get_embedding=True)

    def test_syncs_input_embeds_kwarg_with_padded_forward_batch(self):
        def forward(input_ids, positions, forward_batch, input_embeds=None):
            pass

        compiled = MagicMock(return_value=SimpleNamespace())
        compiled.__signature__ = inspect.signature(forward)
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        fb = self._forward_batch(_tpu_real_q=4, _tpu_real_b=2)
        padded_embeds = torch.zeros((8, 16), dtype=torch.bfloat16)
        fb.input_embeds = padded_embeds
        stale_unpadded_embeds = torch.zeros((4, 16), dtype=torch.bfloat16)

        shim("ids", "pos", fb, input_embeds=stale_unpadded_embeds)
        compiled.assert_called_once_with("ids", "pos", fb, input_embeds=padded_embeds)

    def test_no_op_when_real_size_keys_missing(self):
        # Padding hook didn't run (e.g. forward_idle): output passes through.
        out_obj = SimpleNamespace(
            next_token_logits=torch.zeros(4, 10),
            hidden_states=torch.zeros(8, 5),
        )
        shim = CompiledForwardShim(
            MagicMock(return_value=out_obj), policy=DecoderTpuCompilePolicy()
        )
        out = shim(None, None, self._forward_batch())
        self.assertIs(out, out_obj)

    def test_no_op_when_already_at_real_shape(self):
        # No padding to slice off when shapes already match.
        out_obj = SimpleNamespace(
            next_token_logits=torch.zeros(2, 10),
            hidden_states=torch.zeros(4, 5),
        )
        shim = CompiledForwardShim(
            MagicMock(return_value=out_obj), policy=DecoderTpuCompilePolicy()
        )
        fb = self._forward_batch(_tpu_real_q=4, _tpu_real_b=2)
        out = shim(None, None, fb)
        self.assertEqual(out.next_token_logits.shape, (2, 10))
        self.assertEqual(out.hidden_states.shape, (4, 5))

    def test_restores_real_batch_size_on_forward_batch(self):
        # Compiled-forward padding set batch_size to pad_b; the
        # shim must restore real_b so downstream consumers (sampler,
        # logprob, scheduler hooks) don't iterate over ghost slots.
        compiled = MagicMock(
            return_value=SimpleNamespace(next_token_logits=torch.zeros(4, 10))
        )
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        fb = self._forward_batch(_tpu_real_q=8, _tpu_real_b=2, batch_size=4)
        shim(None, None, fb)
        self.assertEqual(fb.batch_size, 2)

    def test_does_not_touch_batch_size_when_real_keys_missing(self):
        # forward_idle / split_prefill skip the pad hook, so the slicer
        # must not mutate batch_size in that case.
        fb = self._forward_batch(batch_size=4)
        compiled = MagicMock(return_value=SimpleNamespace())
        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        shim(None, None, fb)
        self.assertEqual(fb.batch_size, 4)

    def test_pins_is_prefill_only_for_compiled_call_and_restores(self):
        # The compiled graph must see a batch-invariant is_prefill_only
        # (pinned False); the worker's sampling gate reads the true value
        # after the forward, so it is restored.
        seen = {}

        def compiled(input_ids, positions, forward_batch):
            seen["during"] = forward_batch.is_prefill_only
            return SimpleNamespace()

        shim = CompiledForwardShim(compiled, policy=DecoderTpuCompilePolicy())
        fb = self._forward_batch(batch_size=0, is_prefill_only=True)
        shim(None, None, fb)
        self.assertIs(seen["during"], False)
        self.assertIs(fb.is_prefill_only, True)


class TestContextParallelInputs(CustomTestCase):
    """Under prefill context parallelism the graph takes the rank's zigzag rows
    and one extend_seq_lens_cpu per program family."""

    @staticmethod
    def _call(cp_layout=False, input_embeds=None):
        seen = {}

        def compiled(input_ids, positions, forward_batch, *args, **kwargs):
            seen.update(ids=input_ids, kwargs=kwargs, batch_ids=forward_batch.input_ids)
            seen["lens"] = list(forward_batch.extend_seq_lens_cpu)
            return SimpleNamespace()

        fb = TestCompiledForwardShim._forward_batch(
            input_ids=torch.zeros(48, dtype=torch.int64),
            positions=torch.arange(48),
            forward_mode=_ForwardModeMock("extend"),
            extend_seq_lens_cpu=[40, 8],
            input_embeds=input_embeds,
        )
        if cp_layout:
            fb._tpu_cp_layout = SimpleNamespace(
                host=SimpleNamespace(cp_size=2, local_rows_padded=32),
                local_rows=torch.arange(32),
            )
        shim = CompiledForwardShim(compiled, policy=_policy(SimpleNamespace()))
        shim._attn_cp_size = 2
        shim(fb.input_ids, fb.positions, fb)
        return fb, seen

    def test_cp_batch_embeds_the_global_ids_and_keeps_the_rows(self):
        fb, seen = self._call(cp_layout=True)
        self.assertEqual(seen["ids"].shape[0], 64)  # global ids padded to cp x rows
        self.assertEqual(seen["kwargs"]["cp_local_rows"].shape[0], 32)
        self.assertEqual(seen["batch_ids"].shape[0], 32)
        self.assertEqual(seen["lens"], [4, 4])
        self.assertEqual(fb.input_ids.shape[0], 48)
        self.assertEqual(fb.extend_seq_lens_cpu, [40, 8])

    def test_plain_extend_presents_the_plain_stand_in(self):
        """A plain extend on a context-parallel server must not trace the CP
        program, whatever its live lengths."""
        fb, seen = self._call()
        self.assertEqual(seen["lens"], [0, 0])
        self.assertNotIn("cp_local_rows", seen["kwargs"])
        self.assertEqual(fb.extend_seq_lens_cpu, [40, 8])

    def test_supplied_embeddings_skip_the_lookup(self):
        """Caller embeddings are global; they narrow to the rank's rows and the
        graph takes local ids, not the global ids plus the row selector."""
        _, seen = self._call(cp_layout=True, input_embeds=torch.ones(48, 3))
        self.assertEqual(seen["ids"].shape[0], 32)
        self.assertNotIn("cp_local_rows", seen["kwargs"])
        self.assertEqual(seen["kwargs"]["input_embeds"].shape, (32, 3))


class TestBuildBucketers(CustomTestCase):
    """tpu_compile_bs is read off exec.graph, so the case publishes its config."""

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    @staticmethod
    def _server_args(tpu_compile_tokens=None):
        # Only what production still reads off the handed record.
        return SimpleNamespace(tpu_compile_tokens=tpu_compile_tokens)

    def test_dynamic_chunk_ladder_uses_resolved_page_size(self):
        with (
            patch.dict("os.environ", {"SGLANG_TPU_PP_DYNAMIC_CHUNKS": "1"}),
            get_context().override_server_args(
                device="tpu",
                pp_size=8,
                disaggregation_mode="null",
                chunked_prefill_size=16384,
                page_size=256,
                tpu_compile_bs=[1, 8],
            ) as raw,
        ):
            self.assertIsNone(raw.page_size)
            self.assertIsNone(raw.device)
            self.assertIsNone(raw.chunked_prefill_size)
            token, batch = build_bucketers(raw, max_t=16384, max_b=8)
            self.assertTrue(set(range(2048, 16385, 2048)).issubset(token.buckets))
            self.assertEqual(batch.buckets, [1, 8])

    def test_clipping_keeps_smallest_configured_bucket_covering_scheduler_limit(self):
        # Keep the next configured shape even when the real token limit is
        # unaligned; synthesizing a bucket at that limit loses its alignment.
        cases = (
            ([512], 256, [512]),
            ([128, 512, 1024], 256, [128, 512]),
            ([128, 256, 512, 1024], 257, [128, 256, 512]),
            ([512, 1024], 257, [512]),
            ([1024, 512, 128, 512, 256], 257, [128, 256, 512]),
        )
        for buckets, max_t, expected in cases:
            with self.subTest(buckets=buckets, max_t=max_t):
                args = self._server_args(tpu_compile_tokens=buckets)
                with get_context().override_server_args(tpu_compile_bs=[1, 4]):
                    token_buckets, batch_buckets = build_bucketers(
                        args, max_t=max_t, max_b=4
                    )
                self.assertEqual(token_buckets.buckets, expected)
                fb = _forward_batch(real_t=max_t, real_b=1)
                _pad_forward_batch_inputs(fb, token_buckets, batch_buckets)
                self.assertEqual(fb.input_ids.shape[0], expected[-1])

    def test_uses_user_buckets_as_plain_bucketer(self):
        # User list yields a plain Bucketer (not the PowerBucketer fallback).
        sa = self._server_args(tpu_compile_tokens=[64, 512, 2048])
        with get_exec().graph.override(tpu_compile_bs=[1, 4, 16]):
            t_b, b_b = build_bucketers(sa, max_t=2048, max_b=16)
        self.assertEqual(t_b.buckets, [64, 512, 2048])
        self.assertEqual(b_b.buckets, [1, 4, 16])
        self.assertIsInstance(t_b, Bucketer)
        self.assertNotIsInstance(t_b, PowerBucketer)

    def test_explicit_buckets_used_verbatim(self):
        # Explicit lists are trusted as-is; out-of-range runtime shapes raise.
        sa = self._server_args(tpu_compile_tokens=[64, 512])
        with get_exec().graph.override(tpu_compile_bs=[1, 2]):
            t_b, b_b = build_bucketers(sa, max_t=2048, max_b=16)
        self.assertEqual(t_b.buckets, [64, 512])
        self.assertEqual(b_b.buckets, [1, 2])

    def test_token_buckets_above_exact_max_t_dropped(self):
        # A terminal bucket already at max_t is retained without duplication.
        sa = self._server_args(tpu_compile_tokens=[64, 512, 2048, 4096])
        t_b, _ = build_bucketers(sa, max_t=2048, max_b=8)
        self.assertEqual(t_b.buckets, [64, 512, 2048])  # 4096 dropped

    def test_falls_back_to_power_bucketer_when_unset(self):
        # min_val hardcoded: 64 (T), 1 (B).
        sa = self._server_args()
        t_b, b_b = build_bucketers(sa, max_t=2048, max_b=8)
        self.assertIsInstance(t_b, PowerBucketer)
        self.assertIsInstance(b_b, PowerBucketer)
        self.assertEqual(t_b.min_val, 64)
        self.assertEqual(t_b.max_val, 2048)
        self.assertEqual(b_b.min_val, 1)
        self.assertEqual(b_b.max_val, 8)

    def test_user_t_list_with_power_bs_fallback(self):
        # Mixed: user supplied T but not B.
        sa = self._server_args(tpu_compile_tokens=[64, 2048])
        t_b, b_b = build_bucketers(sa, max_t=2048, max_b=8)
        self.assertNotIsInstance(t_b, PowerBucketer)
        self.assertIsInstance(b_b, PowerBucketer)
        self.assertEqual(t_b.buckets, [64, 2048])
        self.assertEqual(b_b.buckets, [1, 2, 4, 8])

    def test_extend_batch_cap_uses_reachable_minimum(self):
        self.assertEqual(
            _extend_batch_cap(
                _bucketer("Token", [64, 4096]),
                _bucketer("Bs", [1, 8192]),
            ),
            4096,
        )


class TestReconcileTpuBatchLimits(CustomTestCase):
    """The compiled grid covers scheduler concurrency without compiling past
    explicit limits when no larger serving batch is reachable (Task #40)."""

    def _r(
        self,
        torch_compile_max_bs=None,
        tpu_compile_bs=None,
        max_running_requests=None,
        attn_dp_size=1,
        attn_tp_size=1,
    ):
        return reconcile_tpu_batch_limits(
            torch_compile_max_bs,
            tpu_compile_bs,
            max_running_requests,
            attn_dp_size,
            attn_tp_size=attn_tp_size,
        )

    def test_default_neither_set(self):
        # Nothing requested -> ceiling and concurrency default to 64 (NOT the
        # ~2048 memory estimate), no explicit bucket list.
        self.assertEqual(
            self._r(),
            (DEFAULT_TPU_COMPILE_MAX_BS, None, DEFAULT_TPU_COMPILE_MAX_BS),
        )

    def test_explicit_torch_compile_max_bs_above_default_honored(self):
        tcmb, _, mrr = self._r(torch_compile_max_bs=128)
        self.assertEqual((tcmb, mrr), (128, 128))

    def test_explicit_torch_compile_max_bs_below_default_avoids_dead_shapes(self):
        self.assertEqual(self._r(torch_compile_max_bs=8), (8, None, 8))

    def test_max_running_requests_grows_ceiling(self):
        # --max-running-requests 128 -> grid ceiling 128, served (not clamped 64).
        self.assertEqual(self._r(max_running_requests=128), (128, None, 128))

    def test_max_running_requests_below_default_caps_grid(self):
        # --max-running-requests 4 with no explicit compile limit -> the grid
        # stops at 4 instead of compiling dead shapes up to 64.
        self.assertEqual(self._r(max_running_requests=4), (4, None, 4))

    def test_dp_attention_derives_per_worker_cap(self):
        # --max-running-requests 32 across 4 attention-DP workers -> each
        # worker batches at most 8, so the grid stops at 8.
        self.assertEqual(
            self._r(max_running_requests=32, attn_dp_size=4), (8, None, 32)
        )

    def test_dp_attention_grows_ceiling_per_worker_only(self):
        # Explicit ceiling 16 with per-worker concurrency 32 (128/4) -> grown
        # to 32, not to the global 128.
        tcmb, _, mrr = self._r(
            torch_compile_max_bs=16, max_running_requests=128, attn_dp_size=4
        )
        self.assertEqual((tcmb, mrr), (32, 128))

    def test_dp_attention_bucket_list_grows_to_per_worker_cap(self):
        tcmb, bs, mrr = self._r(
            tpu_compile_bs=[1, 8], max_running_requests=64, attn_dp_size=4
        )
        self.assertEqual((tcmb, mrr), (16, 64))
        self.assertEqual(bs, [1, 8, 16])

    def test_dp_attention_default_concurrency_scales_with_workers(self):
        # Nothing requested with 4 workers -> each worker compiles the default
        # grid, so the global concurrency is 4x the per-worker ceiling.
        self.assertEqual(
            self._r(attn_dp_size=4),
            (DEFAULT_TPU_COMPILE_MAX_BS, None, DEFAULT_TPU_COMPILE_MAX_BS * 4),
        )

    def test_tpu_compile_bs_sets_ceiling_and_concurrency(self):
        tcmb, bs, mrr = self._r(tpu_compile_bs=[1, 8])
        self.assertEqual((tcmb, mrr, bs), (8, 8, [1, 8]))

    def test_both_set_and_agree(self):
        tcmb, bs, mrr = self._r(
            tpu_compile_bs=[1, 8, 64, 128], max_running_requests=128
        )
        self.assertEqual((tcmb, mrr, max(bs)), (128, 128, 128))

    def test_conflict_grows_bucket_list(self):
        # Case E: pinned buckets top out at 64 but concurrency is 128 -> the list
        # is grown to include 128 so build_bucketers covers the batch (no crash).
        tcmb, bs, mrr = self._r(tpu_compile_bs=[1, 8, 64], max_running_requests=128)
        self.assertEqual((tcmb, mrr), (128, 128))
        self.assertIn(128, bs)
        self.assertEqual(max(bs), 128)

    def test_bucket_list_only_grows_to_scheduler_concurrency(self):
        tcmb, bs, mrr = self._r(
            torch_compile_max_bs=128,
            tpu_compile_bs=[1, 8],
            max_running_requests=32,
        )
        self.assertEqual((tcmb, bs, mrr), (128, [1, 8, 32], 32))

    def test_invariant_grid_always_covers_concurrency(self):
        cases = [
            (None, None, None),
            (None, None, 200),
            (None, [1, 8, 64], 256),
            (None, [1, 8, 64, 512], None),
            (256, None, 128),
        ]
        for tcmb_in, bs_in, mrr_in in cases:
            tcmb, bs, mrr = self._r(tcmb_in, bs_in, mrr_in)
            self.assertGreaterEqual(tcmb, mrr, f"ceiling < concurrency for {cases}")
            if bs is not None:
                self.assertGreaterEqual(
                    max(bs), mrr, f"buckets don't cover concurrency for {cases}"
                )

    def test_dp_attention_grid_covers_attention_tp_padded_decode_rows(self):
        # Attention DP pads each worker's decode rows up to a multiple of the
        # attention-TP width before the bucket lookup. 184 requests over 4
        # workers at attention TP 8 pad a full worker's 46 rows to 48, which a
        # grid ending at 46 refused ("exceeds the maximum bucket size 46").
        cases = [(46, 4, 8, [16, 32, 46])] + [
            (per_worker, 2, attn_tp_size, bs)
            for per_worker in (1, 3, 64, 94)
            for attn_tp_size in (2, 8, 16)
            for bs in (None, [1, 2, 4])
        ]
        for per_worker, attn_dp_size, attn_tp_size, bs_in in cases:
            with self.subTest(
                per_worker=per_worker, attn_tp_size=attn_tp_size, bs=bs_in
            ):
                tcmb, bs, mrr = self._r(
                    tpu_compile_bs=bs_in,
                    max_running_requests=per_worker * attn_dp_size,
                    attn_dp_size=attn_dp_size,
                    attn_tp_size=attn_tp_size,
                )
                self.assertEqual(mrr, per_worker * attn_dp_size)
                bucketer = (
                    _bucketer("Bs", bs)
                    if bs
                    else PowerBucketer(name="Bs", min_val=1, max_val=tcmb)
                )
                for rows in range(1, per_worker + 1):
                    padded = -(-rows // attn_tp_size) * attn_tp_size
                    self.assertLessEqual(padded, tcmb)
                    bucketer.next_bucket(padded)


class TestDpAttnSelectBucketShape(CustomTestCase):
    """Every DP rank must agree on the same (pad_t, pad_b) bucket."""

    def _fb(self, mode, is_extend_in_batch=False, num_tokens_per_req=4):
        return SimpleNamespace(
            forward_mode=_ForwardModeMock(mode),
            is_extend_in_batch=is_extend_in_batch,
            spec_info=SimpleNamespace(num_tokens_per_req=num_tokens_per_req),
        )

    def test_decode_uses_bs_bucketer_for_both_axes(self):
        # decode BATCH bucket must be identical across ranks -> both axes use the
        # bs bucketer seeded with the cross-rank max_tokens_per_rank.
        t_b = _bucketer("Token", [16, 64])
        b_b = _bucketer("Bs", [4, 8, 16])
        pad_t, pad_b = _dp_attn_select_bucket_shape(
            self._fb("decode"), t_b, b_b, max_tokens_per_rank=8, real_q=3, real_b=3
        )
        self.assertEqual((pad_t, pad_b), (8, 8))

    def test_extend_pads_to_max_bs_bucket(self):
        t_b = _bucketer("Token", [16, 64])
        b_b = _bucketer("Bs", [4, 8, 16])
        pad_t, pad_b = _dp_attn_select_bucket_shape(
            self._fb("extend", is_extend_in_batch=True),
            t_b,
            b_b,
            max_tokens_per_rank=8,
            real_q=10,
            real_b=3,
        )
        # pad_b is forced to the max bs bucket. pad_t is the next token bucket
        # covering max(real_q, max_tokens_per_rank, pad_b).
        self.assertEqual(pad_b, 16)
        self.assertEqual(pad_t, 16)

    def test_idle_without_extend_in_batch_is_decode_group(self):
        t_b = _bucketer("Token", [16, 64])
        b_b = _bucketer("Bs", [4, 8, 16])
        pad_t, pad_b = _dp_attn_select_bucket_shape(
            self._fb("idle", is_extend_in_batch=False),
            t_b,
            b_b,
            max_tokens_per_rank=8,
            real_q=1,
            real_b=1,
        )
        # Treated as decode: both axes from the bs bucketer.
        self.assertEqual((pad_t, pad_b), (8, 8))

    def test_fixed_width_spec_uses_cross_rank_request_count(self):
        t_b = _bucketer("Token", [64, 128])
        b_b = _bucketer("Bs", [1, 2, 4, 8])
        for mode in ("target_verify", "draft_extend_v2"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    _dp_attn_select_bucket_shape(
                        self._fb(mode, num_tokens_per_req=4),
                        t_b,
                        b_b,
                        max_tokens_per_rank=12,
                        real_q=4,
                        real_b=1,
                    ),
                    (16, 4),
                )


class TestDpAttnApplyBucketMetadata(CustomTestCase):
    def _fb(self, dp_size=2, mode="decode"):
        return SimpleNamespace(
            input_ids=torch.zeros(4, dtype=torch.int32),  # CPU -> device for gpu counts
            global_num_tokens_cpu=[3] * dp_size,
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_cpu=[3] * dp_size,
            global_num_tokens_for_logprob_gpu=None,
            dp_local_start_pos=torch.tensor([1]),
            dp_local_num_tokens=torch.tensor([2]),
            global_dp_buffer_len=0,
            is_prefill_only=True,
            # Read to publish is_extend_in_batch to the flags tier; the
            # getattr default is evaluated eagerly, so it must always exist.
            forward_mode=_ForwardModeMock(mode),
        )

    def test_sets_uniform_padded_global_counts(self):
        fb = self._fb(dp_size=2)
        _dp_attn_apply_bucket_metadata(fb, pad_t=8, pad_b=16)
        self.assertEqual(fb.global_num_tokens_cpu, [8, 8])
        self.assertEqual(fb.global_num_tokens_gpu.tolist(), [8, 8])
        self.assertEqual(fb.global_num_tokens_gpu.dtype, torch.int64)
        self.assertEqual(fb.global_num_tokens_for_logprob_cpu, [16, 16])

    def test_does_not_mutate_is_prefill_only(self):
        # The is_prefill_only pin lives in CompiledForwardShim.__call__ (pinned
        # False for the compiled call, restored after); applying bucket
        # metadata must leave the worker-visible flag untouched.
        fb = self._fb()
        _dp_attn_apply_bucket_metadata(fb, pad_t=8, pad_b=8)
        self.assertIs(fb.is_prefill_only, True)

    def test_clears_cached_local_offsets(self):
        fb = self._fb()
        _dp_attn_apply_bucket_metadata(fb, pad_t=8, pad_b=8)
        self.assertIsNone(fb.dp_local_start_pos)
        self.assertIsNone(fb.dp_local_num_tokens)

    def test_global_buffer_len_is_pad_t_times_dp_size(self):
        fb = self._fb(dp_size=4)
        _dp_attn_apply_bucket_metadata(fb, pad_t=8, pad_b=8)
        self.assertEqual(fb.global_dp_buffer_len, 8 * 4)

    def test_target_verify_uses_token_rows_for_logprob_gather(self):
        fb = self._fb(dp_size=2, mode="target_verify")
        _dp_attn_apply_bucket_metadata(fb, pad_t=16, pad_b=4)
        self.assertEqual(fb.global_num_tokens_for_logprob_cpu, [16, 16])
        self.assertEqual(fb.global_num_tokens_for_logprob_gpu.tolist(), [16, 16])


class TestHiCacheReloadGate(CustomTestCase):
    """The shim must wait on the device pool's transfer counter per forward.

    Regression: the gate looked the pool up on the ForwardBatch, which
    carries no pool references, so the defensive lookup returned None and a
    forward consumed hicache-reloaded pages while their host->device DMAs
    were still in flight (garbage output on every reload-hit request; run
    32933877240).
    """

    def _shim(self, counter):
        policy = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool=SimpleNamespace(layer_transfer_counter=counter)
            ),
            finalize_forward_batch=lambda out, fb: out,
            keep_outputs_padded=lambda fb: False,
        )
        return CompiledForwardShim(lambda *a, **k: "out", policy=policy)

    def test_forward_waits_on_reload_counter(self):
        counter = MagicMock()
        shim = self._shim(counter)
        fb = SimpleNamespace(
            is_prefill_only=False,
            mm_input_embeds=None,
            input_embeds=None,
            spec_info=None,
            forward_mode=_ForwardModeMock("decode"),
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_input_logprob_token_ids_gpu=None,
        )
        self.assertEqual(shim(None, None, fb), "out")
        counter.wait.assert_called_once_with()

    def test_forward_runs_without_counter(self):
        # hicache off: the pool's counter slot stays None.
        shim = self._shim(None)
        fb = SimpleNamespace(
            is_prefill_only=False,
            mm_input_embeds=None,
            input_embeds=None,
            spec_info=None,
            forward_mode=_ForwardModeMock("decode"),
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_input_logprob_token_ids_gpu=None,
        )
        self.assertEqual(shim(None, None, fb), "out")

    def test_compiled_caller_does_not_trace_reload_counter(self):
        counter = MagicMock()
        shim = self._shim(counter)
        fb = SimpleNamespace(
            is_prefill_only=False,
            mm_input_embeds=None,
            input_embeds=None,
            spec_info=None,
            forward_mode=_ForwardModeMock("decode"),
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_input_logprob_token_ids_gpu=None,
        )

        with unittest.mock.patch("torch.compiler.is_compiling", return_value=True):
            self.assertEqual(shim(None, None, fb), "out")

        counter.wait.assert_not_called()


def _policy(model):
    """Policy whose model_runner exposes just what the shim reads; the topology
    is published (``_publish_topology``), not injected."""
    policy = DecoderTpuCompilePolicy()
    policy.model_runner = SimpleNamespace(
        model=model,
        # Read by the HiCache reload gate on every eager call.
        token_to_kv_pool=SimpleNamespace(layer_transfer_counter=None),
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    return policy


class TestNeedsInputLogprob(CustomTestCase):
    """Distinguishes input (prompt) logprobs from output logprobs."""

    @staticmethod
    def _fb(mode="extend", return_logprob=True, starts=(0, 0), lens=(4, 4)):
        return SimpleNamespace(
            forward_mode=_ForwardModeMock(mode),
            return_logprob=return_logprob,
            # None is reachable: ForwardBatch leaves both unset off the
            # extend path.
            extend_seq_lens_cpu=None if lens is None else list(lens),
            extend_logprob_start_lens_cpu=None if starts is None else list(starts),
        )

    def test_true_when_prefill_positions_are_requested(self):
        self.assertTrue(needs_input_logprob(self._fb()))

    def test_false_without_return_logprob(self):
        self.assertFalse(needs_input_logprob(self._fb(return_logprob=False)))

    def test_false_on_decode(self):
        # Output-only logprobs: the eager Sampler handles them after the
        # forward, so nothing about the graph changes.
        self.assertFalse(needs_input_logprob(self._fb(mode="decode")))

    def test_false_on_target_verify(self):
        # TARGET_VERIFY is an extend-family mode but carries no prompt to echo.
        self.assertFalse(needs_input_logprob(self._fb(mode="target_verify")))

    def test_false_when_start_equals_length(self):
        # Every prompt position already returned; nothing left to gather.
        self.assertFalse(needs_input_logprob(self._fb(starts=(4, 4), lens=(4, 4))))

    def test_true_when_any_request_wants_positions(self):
        self.assertTrue(needs_input_logprob(self._fb(starts=(4, 1), lens=(4, 4))))

    def test_false_on_empty_metadata(self):
        self.assertFalse(needs_input_logprob(self._fb(starts=(), lens=())))

    def test_false_when_metadata_is_none(self):
        self.assertFalse(needs_input_logprob(self._fb(starts=None, lens=None)))


class TestPromptLogprobRejection(CustomTestCase):
    """One answer on prompt logprobs, shared by the shim (per batch) and intake (per request)."""

    def test_flag_off_names_the_flag(self):
        self.assertIn(
            "SGLANG_TPU_ENABLE_PROMPT_LOGPROB=1",
            CompiledForwardShim._prompt_logprob_rejection(False),
        )

    def test_flag_on_plain_server_serves(self):
        with get_context().override_server_args():
            self.assertIsNone(CompiledForwardShim._prompt_logprob_rejection(True))

    def test_flag_on_dp_attention_rejects(self):
        with get_context().override_server_args(tp_size=2, attn_dp_size=2):
            self.assertIn(
                "DP attention", CompiledForwardShim._prompt_logprob_rejection(True)
            )

    def test_flag_on_pipeline_parallel_rejects(self):
        with get_context().override_server_args(pp_size=2):
            self.assertIn(
                "pipeline parallelism",
                CompiledForwardShim._prompt_logprob_rejection(True),
            )


class TestPromptLogprobIntakeGate(CustomTestCase):
    """The TPU patch finishes an unservable prompt-logprob request with 400 at
    scheduler intake; everything else reaches the queue untouched."""

    def _enqueue(self, *, logprob_start_len, flag, return_logprob=True):
        req = Req("r", "", array("q", [1, 2, 3, 4]), SamplingParams(), return_logprob)
        req.logprob_start_len = logprob_start_len
        queue = MagicMock()
        with (
            patch.object(Scheduler, "_add_request_to_queue", queue),
            envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.override(flag),
            get_context().override_server_args(),
            get_parallel().override(tp_rank=0),
        ):
            patches._patch_scheduler_reject_prompt_logprob()
            Scheduler._add_request_to_queue(SimpleNamespace(), req)
        queue.assert_called_once()
        return req

    def test_prompt_logprobs_refused_with_the_flag_off(self):
        # -2 is not validated upstream and batch prep clamps it to 0: prompt logprobs.
        for start in (0, 2, -2):
            with self.subTest(logprob_start_len=start):
                req = self._enqueue(logprob_start_len=start, flag=False)
                self.assertIsInstance(req.to_finish, FINISH_ABORT)
                self.assertEqual(req.to_finish.status_code, HTTPStatus.BAD_REQUEST)
                self.assertIn(
                    "SGLANG_TPU_ENABLE_PROMPT_LOGPROB=1", req.to_finish.message
                )

    def test_output_only_logprobs_pass(self):
        for start in (-1, 4):
            with self.subTest(logprob_start_len=start):
                self.assertIsNone(
                    self._enqueue(logprob_start_len=start, flag=False).to_finish
                )
        self.assertIsNone(
            self._enqueue(
                logprob_start_len=0, flag=False, return_logprob=False
            ).to_finish
        )

    def test_prompt_logprobs_pass_with_the_flag_on(self):
        self.assertIsNone(self._enqueue(logprob_start_len=0, flag=True).to_finish)


class TestPinFields(CustomTestCase):
    """Freezes logprob metadata for the compiled call and puts it back."""

    def test_records_and_applies_only_changed_fields(self):
        fb = SimpleNamespace(
            return_logprob=True,
            top_logprobs_nums=[5],
            token_ids_logprobs=None,
            extend_input_logprob_token_ids_gpu=torch.zeros(3),
        )
        saved = _pin_fields(fb, _PINNED_LOGPROB_FIELDS)

        self.assertFalse(fb.return_logprob)
        self.assertIsNone(fb.top_logprobs_nums)
        self.assertIsNone(fb.extend_input_logprob_token_ids_gpu)
        # Already at its pinned value, so it is neither written nor recorded.
        self.assertNotIn("token_ids_logprobs", saved)
        self.assertEqual(saved["top_logprobs_nums"], [5])

    def test_no_logprob_batch_is_left_untouched(self):
        fb = SimpleNamespace(
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_input_logprob_token_ids_gpu=None,
        )
        self.assertEqual(_pin_fields(fb, _PINNED_LOGPROB_FIELDS), {})

    def test_restore_round_trips(self):
        original = {
            "return_logprob": True,
            "top_logprobs_nums": [3],
            "token_ids_logprobs": [[7]],
            "extend_input_logprob_token_ids_gpu": None,
        }
        fb = SimpleNamespace(**original)
        _restore_fields(fb, _pin_fields(fb, _PINNED_LOGPROB_FIELDS))
        for field, value in original.items():
            self.assertEqual(getattr(fb, field), value, field)


class TestShimInputLogprobPath(CustomTestCase):
    """Presents FULL capture to the graph and gathers logprobs eagerly."""

    def setUp(self):
        super().setUp()
        # The gate reads the parallel bag per forward; publish a single-rank server.
        self._publish_topology()

    def _publish_topology(self, *, pp_size=1, dp_size=1, enable_dp_attention=False):
        """Publish the topology the prompt-logprob gate reads, for this test."""
        # Attention DP is declared by width now; the flag alone no longer
        # resolves to a width inside an override.
        override = get_context().override_server_args(
            pp_size=pp_size,
            dp_size=1 if enable_dp_attention else dp_size,
            attn_dp_size=dp_size if enable_dp_attention else 1,
        )
        override.install()
        self.addCleanup(override.restore)

    @staticmethod
    def _fb(mode="extend", return_logprob=True, starts=(0, 0)):
        return SimpleNamespace(
            is_prefill_only=False,
            mm_input_embeds=None,
            input_embeds=None,
            spec_info=None,
            seq_lens=torch.zeros(2, dtype=torch.int32),
            forward_mode=_ForwardModeMock(mode),
            return_logprob=return_logprob,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_input_logprob_token_ids_gpu=None,
            extend_seq_lens_cpu=[4, 4],
            extend_logprob_start_lens_cpu=list(starts),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            input_ids=torch.arange(8, dtype=torch.int32),
            _tpu_real_q=8,
            _tpu_real_b=2,
        )

    def _shim(
        self,
        seen,
        *,
        model=None,
        enabled=True,
        pp_size=1,
        dp_size=1,
        enable_dp_attention=False,
    ):
        def compiled(input_ids, positions, forward_batch, *a, **kw):
            # Snapshot what the graph actually sees.
            seen["capture_hidden_mode"] = forward_batch.capture_hidden_mode
            seen["return_logprob"] = forward_batch.return_logprob
            seen["start_lens"] = list(forward_batch.extend_logprob_start_lens_cpu or [])
            return LogitsProcessorOutput(
                next_token_logits=torch.zeros(2, 10),
                hidden_states=torch.zeros(8, 5),
            )

        # The gate reads the topology when the shim is called, so this stays installed.
        self._publish_topology(
            pp_size=pp_size,
            dp_size=dp_size,
            enable_dp_attention=enable_dp_attention,
        )

        # Constructed inside the override on purpose: the flag decides which
        # extend variant warmup compiled, so the shim reads it once and cannot
        # honour a later change.
        with envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.override(enabled):
            return CompiledForwardShim(compiled, policy=_policy(model))

    def test_graph_sees_full_capture_and_no_logprob_request(self):
        seen, fb = {}, self._fb()
        model = _StubModel()
        self._shim(seen, model=model)(None, None, fb)

        self.assertIs(seen["capture_hidden_mode"], CaptureHiddenMode.FULL)
        # Pinned: warmup only ever compiled the no-logprob values.
        self.assertFalse(seen["return_logprob"])
        # One zero per padded request: a stable guard length that asks for
        # every position.
        self.assertEqual(seen["start_lens"], [0, 0])

    def test_caller_state_is_restored_after_the_forward(self):
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                fb = self._fb(starts=(2, 3))
                shim = self._shim({}, model=_StubModel())
                finalized = []

                def finalize(out, batch):
                    finalized.append(batch.capture_hidden_mode)
                    self.assertFalse(batch.return_logprob)
                    return out

                shim.policy.finalize_forward_batch = finalize
                if deferred:

                    def warmup(*args, **kwargs):
                        out = shim.fn(*args, **kwargs)
                        return PendingCall(lambda: out)

                    shim.warmup_async_fn = warmup
                    out = shim.warmup_async(None, None, fb)
                    self.assertEqual(finalized, [])
                    self.assertIs(fb.capture_hidden_mode, CaptureHiddenMode.NULL)
                    self.assertTrue(fb.return_logprob)
                    out = out.wait()
                else:
                    out = shim(None, None, fb)

                self.assertEqual(finalized, [CaptureHiddenMode.FULL])
                self.assertIsNotNone(out.input_token_logprobs)
                self.assertIs(fb.capture_hidden_mode, CaptureHiddenMode.NULL)
                self.assertTrue(fb.return_logprob)
                self.assertEqual(fb.extend_logprob_start_lens_cpu, [2, 3])
                self.assertFalse(fb.is_prefill_only)

    def test_state_is_restored_even_when_the_forward_raises(self):
        with envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.override(True):
            shim = CompiledForwardShim(
                MagicMock(side_effect=RuntimeError("boom")),
                policy=_policy(_StubModel()),
            )
        for prefill_only in (False, True):
            with self.subTest(prefill_only=prefill_only):
                fb = self._fb(starts=(1, 1))
                fb.is_prefill_only = prefill_only
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    shim(None, None, fb)
                self.assertIs(fb.is_prefill_only, prefill_only)
                self.assertIs(fb.capture_hidden_mode, CaptureHiddenMode.NULL)
                self.assertTrue(fb.return_logprob)
                self.assertEqual(fb.extend_logprob_start_lens_cpu, [1, 1])

    def test_input_logprobs_are_attached(self):
        model = _StubModel()
        out = self._shim({}, model=model)(None, None, self._fb())
        self.assertIsNotNone(out.input_token_logprobs)
        self.assertTrue(torch.equal(out.input_token_logprobs, _StubModel.LOGPROBS))

    def test_decode_batch_keeps_its_capture_mode_and_skips_the_gather(self):
        seen, fb = {}, self._fb(mode="decode", return_logprob=True)
        model = _StubModel()
        out = self._shim(seen, model=model)(None, None, fb)

        self.assertIs(seen["capture_hidden_mode"], CaptureHiddenMode.NULL)
        self.assertIs(fb.capture_hidden_mode, CaptureHiddenMode.NULL)
        self.assertEqual(model.calls, 0)
        self.assertIsNone(out.input_token_logprobs)

    def test_output_only_logprob_request_skips_the_gather(self):
        # return_logprob=True but no prompt positions requested: still pinned
        # for the graph, but no eager pass.
        model = _StubModel()
        fb = self._fb(starts=(4, 4))
        self._shim({}, model=model)(None, None, fb)
        self.assertEqual(model.calls, 0)

    # --- Disabled by default. Serving prompt logprobs makes the compiled
    # extend return every prefill row, which costs ~8% of prefill for everyone,
    # so the no-regression guarantee has to be structural rather than measured:
    # with the flag off, none of the above runs. Each case below goes red if
    # the gate is dropped and the cost becomes unconditional again.

    def test_disabled_leaves_the_extend_batch_untouched(self):
        seen = {}
        fb = self._fb(return_logprob=False)
        starts_before = fb.extend_logprob_start_lens_cpu
        self._shim(seen, model=_StubModel(), enabled=False)(None, None, fb)

        # The graph sees the caller's capture mode -- the NULL-capture extend
        # variant warmup has always compiled -- not FULL.
        self.assertIs(seen["capture_hidden_mode"], CaptureHiddenMode.NULL)
        self.assertIs(fb.capture_hidden_mode, CaptureHiddenMode.NULL)
        # Identity, not equality: the list is never even swapped out.
        self.assertIs(fb.extend_logprob_start_lens_cpu, starts_before)

    def test_disabled_never_runs_the_eager_gather(self):
        model = _StubModel()
        self._shim({}, model=model, enabled=False)(
            None, None, self._fb(return_logprob=False)
        )
        self.assertEqual(model.calls, 0)

    def test_disabled_rejects_a_prompt_logprob_request(self):
        # Fail loud and name the flag rather than silently returning nothing.
        with self.assertRaisesRegex(
            NotImplementedError, "SGLANG_TPU_ENABLE_PROMPT_LOGPROB"
        ):
            self._shim({}, model=_StubModel(), enabled=False)(None, None, self._fb())

    def test_dp_attention_rejects_prompt_logprobs(self):
        # The eager gather re-runs the LogitsProcessor, whose logits path
        # enters the attention-TP/DP collectives. Wanting input logprobs is a
        # property of the local batch, so under DP attention one rank would
        # join a collective its peers never reach: the worker group hangs
        # rather than failing. Reject up front instead.
        model = _StubModel()
        shim = self._shim({}, model=model, dp_size=2, enable_dp_attention=True)
        with self.assertRaisesRegex(NotImplementedError, "DP attention"):
            shim(None, None, self._fb())
        self.assertEqual(model.calls, 0)

    def test_pipeline_parallelism_rejects_prompt_logprobs(self):
        # Non-final PP stages return PPProxyTensors, which carries no hidden
        # states, so the gather would fail on an attribute that isn't there.
        model = _StubModel()
        with self.assertRaisesRegex(NotImplementedError, "pipeline"):
            self._shim({}, model=model, pp_size=2)(None, None, self._fb())
        self.assertEqual(model.calls, 0)

    def test_single_rank_config_is_not_rejected(self):
        # The negative side of the two guards above: the ordinary
        # attn_dp_size == 1, pp_size == 1 server must still serve them.
        model = _StubModel()
        out = self._shim({}, model=model)(None, None, self._fb())
        self.assertIsNotNone(out.input_token_logprobs)

    def test_disabled_still_serves_output_logprobs(self):
        # The pin is unconditional and must stay so: an output-logprob request
        # guard-misses the warmed graph without it, which is fatal under
        # set_stance("fail_on_recompile"). This is not part of the opt-in.
        seen = {}
        fb = self._fb(starts=(4, 4))
        self._shim(seen, model=_StubModel(), enabled=False)(None, None, fb)
        self.assertFalse(seen["return_logprob"])
        self.assertTrue(fb.return_logprob)


class TestAttachInputLogprobs(CustomTestCase):
    """Delegates the gather to the model and fails loudly when it cannot."""

    def test_raises_without_captured_hidden_states(self):
        out = LogitsProcessorOutput(next_token_logits=torch.zeros(2, 4))
        with self.assertRaisesRegex(RuntimeError, "CaptureHiddenMode.FULL"):
            _attach_input_logprobs(
                logits_output=out,
                forward_batch=SimpleNamespace(
                    input_ids=torch.zeros(2), top_logprobs_nums=None
                ),
                model=_StubModel(),
            )

    def test_raises_when_the_model_returns_no_input_logprobs(self):
        out = LogitsProcessorOutput(
            next_token_logits=torch.zeros(2, 4), hidden_states=torch.zeros(2, 4)
        )
        with self.assertRaisesRegex(RuntimeError, "input_token_logprobs"):
            _attach_input_logprobs(
                logits_output=out,
                forward_batch=SimpleNamespace(
                    input_ids=torch.zeros(2), top_logprobs_nums=None
                ),
                model=_StubModel(input_token_logprobs=None),
            )

    def test_compute_flag_is_restored_even_on_failure(self):
        model = _StubModel(input_token_logprobs=None)
        out = LogitsProcessorOutput(
            next_token_logits=torch.zeros(2, 4), hidden_states=torch.zeros(2, 4)
        )
        with self.assertRaises(RuntimeError):
            _attach_input_logprobs(
                logits_output=out,
                forward_batch=SimpleNamespace(
                    input_ids=torch.zeros(2), top_logprobs_nums=None
                ),
                model=model,
            )
        self.assertFalse(model.logits_processor.compute_input_logprobs)

    def test_gather_runs_with_the_compute_flag_set(self):
        model = _StubModel()
        out = LogitsProcessorOutput(
            next_token_logits=torch.zeros(2, 4), hidden_states=torch.zeros(2, 4)
        )
        _attach_input_logprobs(
            logits_output=out,
            forward_batch=SimpleNamespace(
                input_ids=torch.zeros(2), top_logprobs_nums=None
            ),
            model=model,
        )
        # The reference gather only emits input logprobs when the flag is on,
        # so the model must observe it set.
        self.assertTrue(model.flag_during_call)
        self.assertFalse(model.logits_processor.compute_input_logprobs)

    def test_all_input_logprob_fields_reach_the_output(self):
        # The gather fills five input logprob fields. An earlier revision kept
        # only input_token_logprobs, so a request asking for top-k (or for named
        # token ids) reached the scheduler with the rest still None and died
        # there indexing it -- see logprob_result_processor's
        # `output.input_top_logprobs_val[i]`.
        model = _StubModel(
            input_top_logprobs_val=[[[-0.1, -0.2]]],
            input_top_logprobs_idx=[[[11, 22]]],
            input_token_ids_logprobs_val=[[[-0.3]]],
            input_token_ids_logprobs_idx=[[[33]]],
        )
        out = LogitsProcessorOutput(
            next_token_logits=torch.zeros(2, 4), hidden_states=torch.zeros(2, 4)
        )
        _attach_input_logprobs(
            logits_output=out,
            forward_batch=SimpleNamespace(
                input_ids=torch.zeros(2), top_logprobs_nums=[2]
            ),
            model=model,
        )
        self.assertIsNotNone(out.input_token_logprobs)
        self.assertEqual(out.input_top_logprobs_val, [[[-0.1, -0.2]]])
        self.assertEqual(out.input_top_logprobs_idx, [[[11, 22]]])
        self.assertEqual(out.input_token_ids_logprobs_val, [[[-0.3]]])
        self.assertEqual(out.input_token_ids_logprobs_idx, [[[33]]])

    def test_raises_when_top_logprobs_were_asked_for_but_not_gathered(self):
        # top_logprobs_nums is pinned to None for the compiled forward and put
        # back before the gather. This case guards that restore: an empty top-k
        # here would otherwise reach the scheduler as a None index.
        out = LogitsProcessorOutput(
            next_token_logits=torch.zeros(2, 4), hidden_states=torch.zeros(2, 4)
        )
        with self.assertRaisesRegex(RuntimeError, "top_logprobs_nums"):
            _attach_input_logprobs(
                logits_output=out,
                forward_batch=SimpleNamespace(
                    input_ids=torch.zeros(2), top_logprobs_nums=[5]
                ),
                model=_StubModel(),  # gathers no top-k
            )

    def test_no_raise_when_top_logprobs_were_not_asked_for(self):
        # The negative branch, and the reason the check reads top_logprobs_nums
        # rather than just testing the field for None: a chosen-token-only
        # request legitimately gathers no top-k and must pass through.
        out = LogitsProcessorOutput(
            next_token_logits=torch.zeros(2, 4), hidden_states=torch.zeros(2, 4)
        )
        _attach_input_logprobs(
            logits_output=out,
            forward_batch=SimpleNamespace(
                input_ids=torch.zeros(2), top_logprobs_nums=[0]
            ),
            model=_StubModel(),
        )
        self.assertIsNone(out.input_top_logprobs_val)


class TestGatherMatchesRealLogitsProcessor(CustomTestCase):
    """Bookkeeping against the real LogitsProcessor, not the stub.

    An earlier revision gathered via `model.process_logit(...)`, which exists on
    no upstream model. Every unit test passed -- a stub is free to implement an
    API that is not there -- and the server died on AttributeError under the
    first real request. These assertions run against the real class, so they go
    red if the contract the shim depends on moves.
    """

    def test_logits_processor_takes_the_arguments_the_shim_passes(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        params = list(inspect.signature(LogitsProcessor.forward).parameters)
        self.assertEqual(
            params[:5],
            ["self", "input_ids", "hidden_states", "lm_head", "logits_metadata"],
        )

    def test_stub_mirrors_the_real_signature(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        real = list(inspect.signature(LogitsProcessor.forward).parameters)[:5]
        stub = list(inspect.signature(_StubLogitsProcessor.__call__).parameters)
        self.assertEqual(stub, real)

    def test_compute_input_logprobs_is_a_real_attribute(self):
        # The shim toggles this around the eager gather; if it is renamed the
        # toggle silently stops working and prompt logprobs come back empty.
        from sglang.srt.layers.logits_processor import LogitsProcessor

        source = inspect.getsource(LogitsProcessor.__init__)
        self.assertIn("self.compute_input_logprobs", source)

    def test_shim_copies_every_input_field_the_output_defines(self):
        # The gather's result is transferred field by field, so a new input_*
        # field on LogitsProcessorOutput is dropped on the floor unless the copy
        # is extended with it. That is exactly how top-k went missing: the
        # request succeeded on GPU and died on TPU indexing a None.
        copied = {
            line.split("=")[0].strip().removeprefix("logits_output.")
            for line in inspect.getsource(_attach_input_logprobs).splitlines()
            if line.strip().startswith("logits_output.input_")
        }
        defined = {
            name
            for name in LogitsProcessorOutput.__annotations__
            if name.startswith("input_")
        }
        self.assertEqual(copied, defined)


class _StubLogitsProcessor:
    """Stands in for LogitsProcessor.

    The signature is copied from `LogitsProcessor.forward` deliberately. An
    earlier version of this stub exposed a `process_logit(...)` method that
    exists only on some models; the unit tests passed against it while the real
    server died on AttributeError, because a stub is free to implement an API
    that isn't there. Keep this in step with the real signature.
    """

    def __init__(self, input_token_logprobs, **input_logprob_fields):
        self.compute_input_logprobs = False
        self._input_token_logprobs = input_token_logprobs
        # The real gather fills top-k and named-token-id rows alongside the
        # chosen-token logprobs; tests that care pass them in.
        self._input_logprob_fields = input_logprob_fields
        self.calls = 0
        self.flag_during_call = None

    def __call__(self, input_ids, hidden_states, lm_head, logits_metadata):
        self.calls += 1
        self.flag_during_call = self.compute_input_logprobs
        return LogitsProcessorOutput(
            next_token_logits=torch.zeros(1, 4),
            input_token_logprobs=self._input_token_logprobs,
            **self._input_logprob_fields,
        )


class _StubModel:
    """A model as the shim sees it: a logits_processor and an lm_head."""

    LOGPROBS = torch.tensor([-0.5, -1.5, -2.5])

    def __init__(self, input_token_logprobs=LOGPROBS, **input_logprob_fields):
        self.logits_processor = _StubLogitsProcessor(
            input_token_logprobs, **input_logprob_fields
        )
        self.lm_head = object()

    @property
    def calls(self):
        return self.logits_processor.calls

    @property
    def flag_during_call(self):
        return self.logits_processor.flag_during_call


if __name__ == "__main__":
    unittest.main()
