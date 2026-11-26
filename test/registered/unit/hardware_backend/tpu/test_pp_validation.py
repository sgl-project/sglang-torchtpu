"""Unit tests for the TPU platform's pipeline-parallel launch validation.

Mocks the slicebuilder env setup, so it runs without TPU hardware.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.arg_groups.overrides import declare_resolution, resolution_result
from sglang.srt.arg_groups.validation_hook import (
    check_pipeline_parallel_compat as core_check_pipeline_parallel_compat,
)
from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
)
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pp_stage_chips, shared_slice
from sglang_tpu.envs import envs
from sglang_tpu.hooks.pipeline_parallel import (
    check_pipeline_parallel_compat,
    skip_spec_op_precompile_without_draft,
)
from sglang_tpu.pp_scheduler import _check_aux_capture_relay
from sglang_tpu.pp_wave import num_relayed_aux_slots
from sglang_tpu.tpu_platform import TpuSRTPlatform


def _args(**overrides):
    # Platform defaults run after handle_deprecated_dp_attention, so they only
    # ever see attn_dp_size=N; the deprecated spelling is a state they never get.
    assert not overrides.get("enable_dp_attention"), "pass attn_dp_size=N"
    defaults = dict(
        dtype=None,
        attention_backend=None,
        sampling_backend=None,
        disable_cuda_graph=False,
        cuda_graph_config=default_cuda_graph_config(),
        enable_torch_compile=False,
        disable_overlap_schedule=False,
        mem_fraction_static=None,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        dcp_size=1,
        ep_size=1,
        enable_dp_attention=False,
        attn_dp_size=1,
        enable_dp_lm_head=False,
        enable_dynamic_chunking=False,
        page_size=None,
        kv_cache_dtype="auto",
        use_ray=False,
        hicache_io_backend=None,
        max_total_tokens=None,
        speculative_algorithm=None,
        speculative_eagle_topk=None,
        speculative_use_rejection_sampling=False,
        pp_async_batch_depth=0,
        disaggregation_mode="null",
        disaggregation_transfer_backend="mooncake",
        optimistic_prefill_attempts=0,
        max_queued_requests=None,
        enable_priority_scheduling=False,
        disaggregation_enable_kv_checksum=False,
        attn_cp_size=1,
        enable_prefill_cp=False,
        cp_strategy=None,
        moe_dp_size=1,
        tpu_compile_tracer="dynamo",
        nnodes=1,
        enable_page_major_kv_layout=False,
        enable_unified_memory=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _cp_stage_args(**overrides):
    """Two-device stages that run prefill context parallelism."""
    fields = dict(
        model_path="openai/gpt-oss-20b",
        trust_remote_code=False,
        revision=None,
        pp_size=4,
        tp_size=2,
        attn_cp_size=2,
        enable_prefill_cp=True,
        cp_strategy="zigzag",
        moe_dp_size=2,
        enable_torch_compile=True,
        sampling_backend="pytorch",
    )
    fields.update(overrides)
    return _args(**fields)


def _apply(args):
    with (
        patch("sglang_tpu.patches._prepare_tpu_env") as prepare_tpu_env,
        patch(
            "sglang.srt.hf_transformers.config.get_config",
            return_value=SimpleNamespace(architectures=["GptOssForCausalLM"]),
        ),
    ):
        TpuSRTPlatform().apply_server_args_defaults(args)
    return prepare_tpu_env


class TestPipelineParallelValidation(CustomTestCase):
    def test_pp8_tp1_is_accepted_and_sizes_the_slice_to_eight_devices(self):
        args = _args(pp_size=8, tp_size=1, enable_torch_compile=True)
        prepare_tpu_env = _apply(args)
        prepare_tpu_env.assert_called_once_with(8, setup_slicebuilder=True)
        self.assertIsNone(args.attention_backend)
        self.assertEqual(
            resolution_result(args, "attention_backend"), "pallas_tpu_rpa_v3"
        )
        self.assertTrue(args.enable_torch_compile)

    def test_pipeline_runs_at_async_batch_depth_one(self):
        """The ICI hand-off needs the last stage's outputs received before
        each forward; depth 0 exchanges them after the forward and deadlocks."""
        args = _args(pp_size=8, tp_size=1, enable_torch_compile=True)
        _apply(args)
        self.assertEqual(args.pp_async_batch_depth, 0)
        self.assertEqual(resolution_result(args, "pp_async_batch_depth"), 1)
        args = _args(
            pp_size=8, tp_size=1, enable_torch_compile=True, pp_async_batch_depth=2
        )
        with self.assertRaisesRegex(NotImplementedError, "pp-async-batch-depth 1"):
            _apply(args)
        args = _args(tp_size=8)
        _apply(args)
        self.assertEqual(args.pp_async_batch_depth, 0)

    def test_pipeline_validation_honors_prior_resolution(self):
        args = _args(enable_torch_compile=True)
        declare_resolution(args, "test.pipeline", pp_size=8)
        prepare_tpu_env = _apply(args)
        prepare_tpu_env.assert_called_once_with(8, setup_slicebuilder=True)
        self.assertEqual(args.pp_size, 1)
        self.assertEqual(args.pp_async_batch_depth, 0)
        self.assertEqual(resolution_result(args, "pp_async_batch_depth"), 1)

    def test_resolved_unsupported_depth_is_rejected_before_environment_setup(self):
        args = _args(pp_size=8)
        declare_resolution(args, "test.pipeline", pp_async_batch_depth=2)
        with patch("sglang_tpu.patches._prepare_tpu_env") as prepare:
            with self.assertRaisesRegex(NotImplementedError, "pp-async-batch-depth 1"):
                TpuSRTPlatform().apply_server_args_defaults(args)
        prepare.assert_not_called()
        self.assertEqual(args.pp_async_batch_depth, 0)

    def test_tp_only_slice_size_is_unchanged(self):
        prepare_tpu_env = _apply(_args(tp_size=8))
        prepare_tpu_env.assert_called_once_with(8, setup_slicebuilder=True)

    def test_accepts_tensor_parallel_pair_stages(self):
        for ep_size in (1, 2):
            with self.subTest(ep_size=ep_size):
                prepare_tpu_env = _apply(
                    _args(
                        pp_size=8,
                        tp_size=2,
                        ep_size=ep_size,
                        enable_torch_compile=True,
                    )
                )
                prepare_tpu_env.assert_called_once_with(16, setup_slicebuilder=True)

    def test_rejects_unsupported_tensor_parallel_stages(self):
        for overrides in (
            dict(tp_size=4),
            dict(tp_size=2, attn_dp_size=2),
            dict(tp_size=2, ep_size=4),
        ):
            with self.subTest(**overrides):
                with self.assertRaisesRegex(
                    NotImplementedError, r"tensor-parallel pair"
                ):
                    _apply(_args(pp_size=2, enable_torch_compile=True, **overrides))

    def test_rejects_stages_wider_than_their_context_parallel_group(self):
        with self.assertRaisesRegex(
            NotImplementedError, r"--attn-cp-size equal to --tp-size"
        ):
            _apply(_cp_stage_args(tp_size=4, attn_cp_size=2))

    def test_rejects_decode_context_parallelism(self):
        with self.assertRaisesRegex(NotImplementedError, r"--dcp-size 2"):
            _apply(_cp_stage_args(dcp_size=2))

    def test_context_parallel_stages_run_on_one_host(self):
        with self.assertRaisesRegex(NotImplementedError, r"one host.*--nnodes 2"):
            _apply(_cp_stage_args(nnodes=2))

    def test_context_parallel_stages_reject_other_kv_cache_dtypes(self):
        """Context-parallel stages serve only a bf16 KV cache, so any other
        dtype fails at launch instead of serving."""
        for dtype in ("fp8_e4m3", "mxfp8"):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(
                    NotImplementedError, r"only a bf16 KV cache"
                ):
                    _apply(_cp_stage_args(kv_cache_dtype=dtype))

    def test_context_parallel_stages_default_to_a_bf16_kv_cache(self):
        """The TPU fp8 default for the RPA backends does not apply to
        context-parallel stages."""
        args = _cp_stage_args()
        _apply(args)
        self.assertEqual(resolution_result(args, "kv_cache_dtype"), "bfloat16")

    def test_rejects_speculative_decoding_outside_prefill_disaggregation(self):
        with self.assertRaisesRegex(
            NotImplementedError, r"Speculative decoding.*pipeline parallelism"
        ):
            _apply(_args(pp_size=2, speculative_algorithm="EAGLE3"))

    def test_accepts_speculative_decoding_on_a_prefill_node(self):
        _apply(
            _args(
                pp_size=2,
                speculative_algorithm="EAGLE3",
                disaggregation_mode="prefill",
                enable_torch_compile=True,
            )
        )

    def test_rejects_dynamic_chunking(self):
        with self.assertRaisesRegex(
            NotImplementedError, r"--enable-dynamic-chunking.*pipeline"
        ):
            _apply(_args(pp_size=2, enable_dynamic_chunking=True))

    def test_rejects_eager_forward(self):
        """An eager forward gets the received activations at the hand-off
        buffer's row count, not the batch's token count."""
        with self.assertRaisesRegex(NotImplementedError, "--enable-torch-compile"):
            _apply(_args(pp_size=2))

    def test_rejection_happens_before_the_slice_env_is_prepared(self):
        with patch("sglang_tpu.patches._prepare_tpu_env") as prepare_tpu_env:
            with self.assertRaises(NotImplementedError):
                TpuSRTPlatform().apply_server_args_defaults(
                    _cp_stage_args(tp_size=4, attn_cp_size=2)
                )
        prepare_tpu_env.assert_not_called()

    def test_single_stage_keeps_every_flag(self):
        args = _args(pp_size=1, tp_size=8, enable_dynamic_chunking=True)
        _apply(args)
        self.assertTrue(args.enable_dynamic_chunking)


def _compat_cfg(**overrides):
    defaults = dict(
        speculative_algorithm="EAGLE3",
        disable_overlap_schedule=True,
        enable_multi_layer_eagle=False,
        scheduler_recv_interval=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestEagle3PipelineParallelCompat(CustomTestCase):
    def test_eagle3_runs_the_core_check_as_eagle(self):
        original = Mock()
        check_pipeline_parallel_compat(original, _compat_cfg())
        original.assert_called_once()
        (cfg,) = original.call_args.args
        self.assertEqual(cfg.speculative_algorithm, "EAGLE")
        self.assertTrue(cfg.disable_overlap_schedule)

    def test_other_algorithms_run_the_core_check(self):
        for algorithm in (None, "EAGLE", "DSPARK"):
            original = Mock()
            cfg = _compat_cfg(speculative_algorithm=algorithm)
            check_pipeline_parallel_compat(original, cfg)
            original.assert_called_once_with(cfg)

    def test_eagle3_rejects_overlap_schedule(self):
        with self.assertRaisesRegex(AssertionError, "overlap schedule"):
            check_pipeline_parallel_compat(
                core_check_pipeline_parallel_compat,
                _compat_cfg(disable_overlap_schedule=False),
            )

    def test_eagle3_rejects_multi_layer_eagle(self):
        with self.assertRaisesRegex(AssertionError, "non-multi-layer"):
            check_pipeline_parallel_compat(
                core_check_pipeline_parallel_compat,
                _compat_cfg(enable_multi_layer_eagle=True),
            )

    def test_experimental_dflash_is_admitted_on_a_prefill_node_only(self):
        """The experimental worker seeds DFlash draft KV on a PP prefill; nothing else runs it."""
        cfg = dict(
            speculative_algorithm="DFLASH",
            min_free_slots_delay=None,
            disaggregation_mode="prefill",
        )
        with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(True):
            check_pipeline_parallel_compat(
                core_check_pipeline_parallel_compat, _compat_cfg(**cfg)
            )
            with self.assertRaisesRegex(AssertionError, "disaggregation-mode=prefill"):
                check_pipeline_parallel_compat(
                    core_check_pipeline_parallel_compat,
                    _compat_cfg(**{**cfg, "disaggregation_mode": "null"}),
                )
        with (
            envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False),
            envs.SGLANG_TPU_HOST_PADDING_NO_D2H.override(False),
        ):
            # Core's DFlash worker builds its draft on every stage.
            with self.assertRaisesRegex(AssertionError, "only supports EAGLE"):
                check_pipeline_parallel_compat(
                    core_check_pipeline_parallel_compat, _compat_cfg(**cfg)
                )

    def test_stage_without_draft_skips_spec_op_precompile(self):
        original = Mock()
        scheduler = SimpleNamespace(draft_worker=SimpleNamespace(draft_worker=None))
        skip_spec_op_precompile_without_draft(original, scheduler)
        original.assert_not_called()

    def test_stage_with_draft_precompiles_spec_ops(self):
        original = Mock()
        scheduler = SimpleNamespace(draft_worker=SimpleNamespace(draft_worker=object()))
        skip_spec_op_precompile_without_draft(original, scheduler)
        original.assert_called_once_with(scheduler)

    def test_model_without_relay_rejects_captures_before_the_last_stage(self):
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers_to_capture=[2, 18], start_layer=10, end_layer=20
            )
        )
        with self.assertRaisesRegex(NotImplementedError, r"layers \[2\]"):
            _check_aux_capture_relay(model)

    def test_model_without_relay_hands_on_no_aux_slots(self):
        config = SimpleNamespace(
            eagle_use_aux_hidden_state=True, eagle_aux_hidden_state_layer_ids=[2, 18]
        )
        self.assertEqual(num_relayed_aux_slots(config, SimpleNamespace()), 0)
        relaying = SimpleNamespace(relays_pp_aux_hidden_states=True)
        self.assertEqual(num_relayed_aux_slots(config, relaying), 2)

    def test_dflash_captures_size_the_aux_slots(self):
        """The wave width must match the DFlash capture count the stages relay."""
        relaying = SimpleNamespace(relays_pp_aux_hidden_states=True)
        config = SpecAuxHiddenStateConfig(
            dflash_use_aux_hidden_state=True, dflash_target_layer_ids=[1, 9, 17, 25, 33]
        )
        self.assertEqual(num_relayed_aux_slots(config, relaying), 5)
        with self.assertRaisesRegex(NotImplementedError, "dflash_target_layer_ids"):
            num_relayed_aux_slots(
                SpecAuxHiddenStateConfig(dflash_use_aux_hidden_state=True), relaying
            )


def _group(*ranks):
    return SimpleNamespace(
        unique_name=f"group{ranks}",
        ranks=list(ranks),
        world_size=len(ranks),
        device_group=SimpleNamespace(ranks=ranks),
    )


def _on(chip, core):
    return pp_stage_chips.CoreLocation(slice_index=0, coords=chip, core_on_chip=core)


class TestPipelineStageChips(CustomTestCase):
    def setUp(self):
        # Two-core chips, like the slices the ring orders are known on.
        patcher = patch.object(pp_stage_chips, "_devices_per_chip", return_value=2)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_pair_must_hold_every_core_of_its_chip(self):
        """A pair split over two-device chips all-reduces over ICI and shares
        each chip with another stage, also when it uses core 0 of both."""
        chip_a, chip_b = (0, 0, 0), (1, 0, 0)
        violations = pp_stage_chips._partial_chip_problems
        self.assertEqual(
            violations([_on(chip_a, 0), _on(chip_a, 1)], devices_per_chip=2), []
        )
        split = [_on(chip_a, 0), _on(chip_b, 0)]
        self.assertEqual(len(violations(split, devices_per_chip=2)), 2)
        self.assertEqual(violations(split, devices_per_chip=1), [])

    def test_an_unrecognized_tpu_fails_at_startup(self):
        # Context-parallel stages (attn_cp_size 2) are checked like TP pairs.
        for attn_cp_size in (1, 2):
            parallel = SimpleNamespace(
                pp_size=4,
                tp_size=2,
                attn_cp_size=attn_cp_size,
                pp_rank=1,
                tp_group=_group(2, 3),
            )
            with (
                self.subTest(attn_cp_size=attn_cp_size),
                patch.object(pp_stage_chips, "_devices_per_chip", return_value=None),
                self.assertRaisesRegex(RuntimeError, "does not recognize"),
            ):
                pp_stage_chips.validate_chips_for_pp(parallel, ring_expected=True)

    # One 2x2x2 slice over two hosts, chips in the order a stage-per-chip
    # launch fills them: each host's four chips snake from the same corner, so
    # 3->4 and the 7->0 relay each cross two hops.
    SNAKE = [
        (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1),
        (0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0),
    ]  # fmt: skip

    def _pipeline(self, *, pp_rank=0, tp_size=2):
        return SimpleNamespace(
            pp_size=8,
            tp_size=tp_size,
            pp_rank=pp_rank,
            pp_group=_group(*range(0, 8 * tp_size, tp_size)),
            tp_group=_group(*range(pp_rank * tp_size, (pp_rank + 1) * tp_size)),
        )

    def test_hops_count_chips_crossed_per_hand_off(self):
        locations = [_on(chip, 0) for chip in self.SNAKE]
        self.assertEqual(
            [h for _, _, h in pp_stage_chips._boundary_hops(locations)],
            [1, 1, 1, 2, 1, 1, 1, 2],
        )
        # Stages sharing a chip (one device per stage on two-device chips) hand
        # off without ICI; another slice is not counted in hops.
        self.assertEqual(pp_stage_chips._hops(_on((1, 0, 0), 0), _on((1, 0, 0), 1)), 0)
        other_slice = pp_stage_chips.CoreLocation(
            slice_index=1, coords=(1, 0, 0), core_on_chip=0
        )
        self.assertIsNone(pp_stage_chips._hops(_on((1, 0, 0), 0), other_slice))

    def _check(self, parallel, column, *, ring_expected=True):
        """Run the startup check with ``column`` as the cores of this rank's
        pipeline column and each stage's TP group on one chip of SNAKE."""

        def locate(ranks):
            if list(ranks) == parallel.pp_group.ranks:
                return column
            return [_on(self.SNAKE[parallel.pp_rank % 8], c) for c in range(len(ranks))]

        with (
            patch.object(pp_stage_chips, "_stage_core_locations", side_effect=locate),
        ):
            pp_stage_chips.validate_chips_for_pp(parallel, ring_expected=ring_expected)

    def test_single_chip_stages_refuse_a_far_hand_off_on_every_rank(self):
        column = [_on(chip, 0) for chip in self.SNAKE]
        for pp_rank in (0, 5):
            with (
                self.subTest(pp_rank=pp_rank),
                self.assertRaisesRegex(
                    RuntimeError,
                    r"3->4: 2 \(chips \(1, 0, 1\) to \(0, 0, 0\)\), "
                    r"7->0 \(relay\): 2 .*SGLANG_TPU_PP_ALLOW_DISTANT_STAGES=1",
                ),
            ):
                self._check(self._pipeline(pp_rank=pp_rank), column)

    def test_the_opt_out_downgrades_the_refusal_to_a_warning(self):
        column = [_on(chip, 0) for chip in self.SNAKE]
        with (
            envs.SGLANG_TPU_PP_ALLOW_DISTANT_STAGES.override(True),
            self.assertLogs(pp_stage_chips.logger, "INFO") as logs,
        ):
            self._check(self._pipeline(), column)
        warnings = [r.getMessage() for r in logs.records if r.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("7->0 (relay): 2", warnings[0])
        self.assertIn("allowed by SGLANG_TPU_PP_ALLOW_DISTANT_STAGES", warnings[0])

    def test_multi_chip_stages_warn_instead_of_refusing(self):
        # TP4 on two-device chips: no rank order makes every column one hop.
        parallel = SimpleNamespace(
            pp_size=4,
            tp_size=4,
            pp_rank=0,
            pp_group=_group(0, 4, 8, 12),
            tp_group=_group(0, 1, 2, 3),
        )
        column = [_on(self.SNAKE[k], 0) for k in (0, 2, 4, 6)]

        def locate(ranks):
            if list(ranks) == parallel.pp_group.ranks:
                return column
            return [_on(self.SNAKE[c // 2], c % 2) for c in range(4)]

        with (
            patch.object(pp_stage_chips, "_stage_core_locations", side_effect=locate),
            self.assertLogs(pp_stage_chips.logger, "INFO") as logs,
        ):
            pp_stage_chips.validate_chips_for_pp(parallel, ring_expected=True)
        warnings = [r.getMessage() for r in logs.records if r.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("only when each fits on one chip", warnings[0])

    def test_layouts_the_launch_did_not_order_warn_instead_of_refusing(self):
        column = [_on(chip, 0) for chip in self.SNAKE]
        with self.assertLogs(pp_stage_chips.logger, "WARNING") as logs:
            self._check(self._pipeline(), column, ring_expected=False)
        (warning,) = [r.getMessage() for r in logs.records]
        self.assertIn("3->4: 2", warning)
        # Nor on chips of another core count, whose native order is not known.
        with (
            patch.object(pp_stage_chips, "_devices_per_chip", return_value=1),
            patch.object(pp_stage_chips, "_stage_core_locations", return_value=column),
            self.assertLogs(pp_stage_chips.logger, "WARNING"),
        ):
            pp_stage_chips.validate_chips_for_pp(
                self._pipeline(tp_size=1), ring_expected=True
            )

    def test_hand_offs_to_another_slice_are_logged_not_warned(self):
        def on(slice_index, x):
            return pp_stage_chips.CoreLocation(
                slice_index=slice_index, coords=(x, 0, 0), core_on_chip=0
            )

        # Two slices of four chips in a row: 3->4 and the relay change slice.
        locations = [on(k // 4, k % 4 if k < 4 else 3 - k % 4) for k in range(8)]
        with (
            patch.object(
                pp_stage_chips, "_stage_core_locations", return_value=locations
            ),
            self.assertLogs(pp_stage_chips.logger, "INFO") as logs,
        ):
            pp_stage_chips.validate_chips_for_pp(
                self._pipeline(tp_size=1), ring_expected=True
            )
        self.assertEqual([r.levelname for r in logs.records], ["INFO"])
        self.assertIn("3->4: other slice", logs.records[0].getMessage())

    def test_only_the_first_stage_of_a_column_logs_the_hops(self):
        ring = [_on(chip, 0) for chip in self.SNAKE[:4] + self.SNAKE[4:][::-1]]
        with self.assertLogs(pp_stage_chips.logger, "INFO") as logs:
            self._check(self._pipeline(pp_rank=3), ring)
        self.assertNotIn("ICI hops", " ".join(r.getMessage() for r in logs.records))

    def test_single_device_stages_are_reported_but_not_pair_checked(self):
        locations = [_on(chip, core) for chip in self.SNAKE[:4] for core in (0, 1)]
        with (
            patch.object(pp_stage_chips, "_devices_per_chip", return_value=None),
            patch.object(
                pp_stage_chips, "_stage_core_locations", return_value=locations
            ),
            self.assertLogs(pp_stage_chips.logger, "INFO") as logs,
        ):
            pp_stage_chips.validate_chips_for_pp(
                self._pipeline(tp_size=1), ring_expected=True
            )
        (line,) = [r.getMessage() for r in logs.records if r.levelname == "INFO"]
        self.assertIn("0->1: 0, 1->2: 1", line)

    def test_a_runtime_without_coordinates_skips_the_report(self):
        with (
            patch.object(
                pp_stage_chips,
                "_stage_core_locations",
                # Any lookup error, not only the runtime's own RuntimeError.
                side_effect=ValueError("no chip coordinates"),
            ),
            self.assertLogs(pp_stage_chips.logger, "WARNING") as logs,
        ):
            pp_stage_chips.validate_chips_for_pp(
                self._pipeline(tp_size=1), ring_expected=True
            )
        (warning,) = [r.getMessage() for r in logs.records]
        self.assertIn("not checked", warning)


class TestPipelinePlacement(CustomTestCase):
    # A 2x2x2 slice of two hosts, two cores per chip, ranks filling host 0
    # first. Each host's native device order walks its four chips as a ring,
    # as the startup log of such a slice shows (TestPipelineStageChips.SNAKE).
    HOST_RING = [(0, 0), (0, 1), (1, 1), (1, 0)]

    def _placement(self, rank, *, pp_size=8, tp_size=2, nnodes=2, env=None):
        cfg = SimpleNamespace(
            device="tpu",
            tp_size=tp_size,
            pp_size=pp_size,
            nnodes=nnodes,
            torch_tpu_topology=None,
        )
        # Only what the repo's launchers leave set: TPU_CHIPS_PER_HOST_BOUNDS
        # and TORCH_TPU_TOPOLOGY are unset when a worker is prepared.
        if env is None:
            env = {"TPU_TOPOLOGY": "2x2x2"}
        with patch.dict(os.environ, env, clear=True):
            return pp_stage_chips.pipeline_placement(
                cfg, tp_rank=rank % tp_size, pp_rank=rank // tp_size
            )

    def _core(self, rank, *, tp_size, mirror):
        node, local = divmod(rank, 8)
        placement = self._placement(rank, pp_size=16 // tp_size, tp_size=tp_size)
        self.assertTrue(placement.ring_expected)
        device = placement.local_device if mirror else None
        chip, core = divmod(local if device is None else device, 2)
        return _on((*self.HOST_RING[chip], 1 - node), core)

    def _hops(self, *, tp_size, mirror):
        column = [
            self._core(rank, tp_size=tp_size, mirror=mirror)
            for rank in range(0, 16, tp_size)
        ]
        return [h for _, _, h in pp_stage_chips._boundary_hops(column)]

    def test_the_second_host_binds_its_chips_in_reverse_for_a_one_hop_ring(self):
        self.assertIn(2, self._hops(tp_size=2, mirror=False))
        self.assertEqual(self._hops(tp_size=2, mirror=True), [1] * 8)
        # One device per stage: stage pairs share a chip, then step to the next.
        self.assertEqual(self._hops(tp_size=1, mirror=True), [0, 1] * 8)

    def test_each_tensor_parallel_pair_stays_on_one_chip(self):
        for stage in range(8):
            pair = [self._core(2 * stage + i, tp_size=2, mirror=True) for i in (0, 1)]
            self.assertEqual(pair[0].coords, pair[1].coords)
            self.assertEqual({c.core_on_chip for c in pair}, {0, 1})

    def test_other_layouts_keep_the_native_order_and_are_not_refused(self):
        native = pp_stage_chips.PipelinePlacement(
            local_device=None, ring_expected=False
        )
        slice_env = {"TPU_TOPOLOGY": "2x2x2"}
        cases = {
            "no pipeline": dict(pp_size=1),
            "multi-chip stages": dict(pp_size=4, tp_size=4),
            "one core per chip": dict(pp_size=8, tp_size=1),
            "four hosts": dict(nnodes=4, env={"TPU_TOPOLOGY": "2x2x4"}),
            "two hosts side by side": dict(env={"TPU_TOPOLOGY": "4x2x1"}),
            "unknown topology": dict(env={}),
            "co-located P/D": dict(env={**slice_env, "SGLANG_TPU_PD_COLOCATED": "1"}),
            "shared slice": dict(
                env={**slice_env, shared_slice.ENV_VAR: "0,16,host:1234"}
            ),
        }
        for name, args in cases.items():
            with self.subTest(name):
                self.assertEqual(self._placement(9, **args), native)
        # One host's native order is already the ring, but only stages filling
        # all four chips close it.
        for pp_size, tp_size, ring in ((8, 1, True), (4, 2, True), (3, 2, False)):
            with self.subTest(one_host=(pp_size, tp_size)):
                self.assertEqual(
                    self._placement(1, pp_size=pp_size, tp_size=tp_size, nnodes=1),
                    pp_stage_chips.PipelinePlacement(
                        local_device=None, ring_expected=ring
                    ),
                )


if __name__ == "__main__":
    unittest.main()
