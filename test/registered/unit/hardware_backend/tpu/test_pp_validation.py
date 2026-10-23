"""Unit tests for the TPU platform's pipeline-parallel launch validation.

Mocks the slicebuilder env setup, so it runs without TPU hardware.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.arg_groups.overrides import declare_resolution, resolution_result
from sglang.srt.arg_groups.validation_hook import (
    check_pipeline_parallel_compat as core_check_pipeline_parallel_compat,
)
from sglang.srt.distributed import parallel_state
from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
)
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pp_stage_chips, pp_stage_groups
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
        enable_unified_memory=False,
        enable_page_major_kv_layout=False,
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
            "sglang.srt.utils.hf_transformers.config.get_config",
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
        prepare_tpu_env = _apply(_args(pp_size=8, tp_size=2, enable_torch_compile=True))
        prepare_tpu_env.assert_called_once_with(16, setup_slicebuilder=True)

    def test_rejects_tensor_parallel_stages_other_than_a_plain_pair(self):
        """Only two-device stages of plain tensor parallelism were validated."""
        for overrides in (
            dict(tp_size=4),
            dict(tp_size=2, attn_dp_size=2),
            dict(tp_size=2, ep_size=2),
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
        with envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.override(False):
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


def _parallel(*, pp_size, tp_size, stage):
    return SimpleNamespace(pp_size=pp_size, tp_size=tp_size, pp_rank=1, tp_group=stage)


class TestStandaloneStageGroups(CustomTestCase):
    def _made_standalone(self, parallel):
        make_standalone = Mock()
        with patch.object(
            pp_stage_groups, "_load_make_standalone", return_value=make_standalone
        ):
            pp_stage_groups.make_stage_groups_standalone(parallel)
        return [call.args[0] for call in make_standalone.call_args_list]

    def test_every_group_inside_the_stage_is_made_standalone_in_creation_order(self):
        """A stage collective over a group left out waits for the other stages,
        and groups that span stages carry the hand-off, so they stay as built."""
        world, pp, one_device = _group(*range(8)), _group(0, 2, 4, 6), _group(2)
        # The stage's tp group, its PD-multiplexing copy and a decode CP group.
        stage, tp_copy, dcp = _group(2, 3), _group(2, 3), _group(2, 3)
        built = (world, stage, tp_copy, dcp, one_device, pp)
        registry = {f"g{i}": (lambda g=group: g) for i, group in enumerate(built)}
        with patch.object(parallel_state, "_groups", registry):
            made = self._made_standalone(_parallel(pp_size=4, tp_size=2, stage=stage))
        self.assertEqual(
            made, [stage.device_group, tp_copy.device_group, dcp.device_group]
        )

    def test_one_device_stages_and_one_stage_do_not_need_the_api(self):
        load = Mock(side_effect=AssertionError("loaded make_standalone"))
        layouts = (
            _parallel(pp_size=4, tp_size=1, stage=_group(1)),
            _parallel(pp_size=1, tp_size=2, stage=_group(0, 1)),
            _parallel(pp_size=1, tp_size=1, stage=_group(0)),
        )
        with patch.object(pp_stage_groups, "_load_make_standalone", load):
            for parallel in layouts:
                pp_stage_groups.make_stage_groups_standalone(parallel)
        load.assert_not_called()

    def test_a_torch_tpu_without_standalone_subgroups_fails_at_startup(self):
        """A context-parallel pipeline on a torch_tpu without standalone
        subgroups stalls at its first stage collective, so it must not start."""
        parallel = _parallel(pp_size=4, tp_size=2, stage=_group(2, 3))
        without_api = types.ModuleType("torch_tpu._internal.distributed.spmd_util")
        with patch.dict(
            sys.modules, {"torch_tpu._internal.distributed.spmd_util": without_api}
        ):
            with self.assertRaisesRegex(
                RuntimeError, r"torch_tpu with standalone process subgroups"
            ):
                pp_stage_groups.make_stage_groups_standalone(parallel)


def _on(chip, core):
    return pp_stage_chips.CoreLocation(slice_index=0, coords=chip, core_on_chip=core)


class TestPipelineStageChips(CustomTestCase):
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
                pp_stage_chips.validate_chips_for_pp(parallel)


if __name__ == "__main__":
    unittest.main()
