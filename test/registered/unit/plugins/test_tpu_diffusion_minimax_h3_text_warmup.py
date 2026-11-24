# SPDX-License-Identifier: Apache-2.0
"""CPU tests for sglang_tpu's MiniMax H3 text-length warmup (minimax_h3_text_warmup)."""

import contextlib
import gc
import os
import sys
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_plan_from_batch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
# The CPU gate installs python/ only; an installed sglang_tpu still wins.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from sglang_tpu.envs import envs as tpu_envs  # noqa: E402
from sglang_tpu.multimodal_gen import minimax_h3_text_warmup as tw  # noqa: E402


class _Tokenizer:
    """Whitespace tokenizer; ``wide`` words take two tokens each, from the
    ``wide_from``-th word of a text on."""

    def __init__(self, wide=(), wide_from=0):
        self.wide = set(wide)
        self.wide_from = wide_from

    def __call__(self, text, add_special_tokens=False):
        ids = []
        words = text.split()
        for word in words:
            wide = word in self.wide and len(words) > self.wide_from
            ids += [len(word)] * (2 if wide else 1)
        return {"input_ids": ids}


def _server_args(**overrides):
    values = dict(
        pipeline_class_name="MiniMaxH3Pipeline",
        pipeline_config=MiniMaxH3PipelineConfig(),
        warmup_sampling_params=None,
        warmup_resolutions=None,
        warmup_num_frames=None,
        model_variant="fl2va",
        model_subfolder=None,
        model_path="/models/MiniMax-H3",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _stage(name):
    return type(name, (), {})()


_STAGES = (
    "InputValidationStage",
    "MiniMaxH3PartitionAdmissionStage",
    "MiniMaxH3TextEncodingStage",
    "MiniMaxH3VisualEncodingStage",
    "MiniMaxH3AudioEncodingStage",
    "MiniMaxH3LatentPreparationStage",
    "MiniMaxH3TimestepPreparationStage",
    "MiniMaxH3DenoisingStage",
    "MiniMaxH3DecodingStage",
)


class _WaitTensor(torch.Tensor):
    """Counts the host reads ``_wait`` makes of a request's outputs."""

    reads = 0

    def cpu(self, *args, **kwargs):
        type(self).reads += 1
        return super().cpu(*args, **kwargs)


class _Executor:
    def __init__(self):
        self.calls = []
        self.conditioning_cache = SimpleNamespace(cleared=0)
        self.conditioning_cache.clear = self._clear
        self.requests = []

    def _clear(self):
        self.conditioning_cache.cleared += 1

    def execute_with_profiling(self, stages, req, server_args):
        self.calls.append([type(stage).__name__ for stage in stages])
        self.requests.append(weakref.ref(req))
        names = self.calls[-1]
        if "MiniMaxH3TextEncodingStage" in names:
            req.prompt_embeds = [torch.zeros(len(req.prompt.split()), 4)]
        if "MiniMaxH3DenoisingStage" in names:
            model = req.extra.get("_test_model")
            if model is not None:
                model.refine_prompt_embeds(torch.ones(2))
            req.latents = torch.zeros(1, 2, 3).as_subclass(_WaitTensor)
            req.audio_latents = torch.zeros(2, 3).as_subclass(_WaitTensor)
        return req


class _Pipeline:
    def __init__(self, stages=_STAGES, tasks=("t2va", "fl2va"), tokenizer=None):
        self.stages = [_stage(name) for name in stages]
        self.release_metadata = SimpleNamespace(tasks=tasks)
        self.executor = _Executor()
        self.modules = {"tokenizer": tokenizer or _Tokenizer()}
        self.residency_installs = 0

    def get_module(self, name, default=None):
        return self.modules.get(name, default)

    def _install_component_residency_manager(self, server_args):
        self.residency_installs += 1


@contextlib.contextmanager
def _env(**values):
    with contextlib.ExitStack() as stack:
        for name, value in values.items():
            stack.enter_context(getattr(tpu_envs, name).override(value))
        yield


class TestLengths(CustomTestCase):
    def test_parse(self):
        self.assertEqual(tw.parse_lengths("36-38, 40,38 ,64"), [36, 37, 38, 40, 64])
        self.assertEqual(tw.parse_lengths("5"), [5])
        self.assertEqual(tw.parse_lengths("1 - 3"), [1, 2, 3])
        for bad in ("", " , ", "0", "5-3", "a-b", "3-", "-3", "1.5"):
            with self.assertRaises(ValueError, msg=bad):
                tw.parse_lengths(bad)
        with self.assertRaises(ValueError):
            tw.parse_lengths(f"1-{tw.MAX_LENGTHS + 1}")

    def test_format(self):
        self.assertEqual(tw.format_lengths([40, 36, 37, 38, 64, 63]), "36-38,40,63-64")
        self.assertEqual(tw.format_lengths([7]), "7")

    def test_requested(self):
        with _env(SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS=None):
            self.assertFalse(tw.requested())
        with _env(SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS="  "):
            self.assertFalse(tw.requested())
        with _env(SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS="36-47"):
            self.assertTrue(tw.requested())


class TestFillerPrompt(CustomTestCase):
    def test_exact_token_counts(self):
        tokenizer = _Tokenizer()
        for length in (1, 2, 39, 47, 300):
            prompt = tw.filler_prompt(tokenizer, length)
            self.assertEqual(len(tokenizer(prompt)["input_ids"]), length)

    def test_falls_back_to_another_word(self):
        tokenizer = _Tokenizer(wide={"the"})
        prompt = tw.filler_prompt(tokenizer, 7)
        self.assertNotIn("the", prompt.split())
        self.assertEqual(len(tokenizer(prompt)["input_ids"]), 7)

    def test_raises_when_no_word_fits(self):
        with self.assertRaises(ValueError):
            tw.filler_prompt(_Tokenizer(wide=tw.FILLER_WORDS), 3)


class TestWarmupRequests(CustomTestCase):
    def test_benchmark_canvas_one_step_warmup(self):
        (req,) = tw.warmup_requests(_server_args(), "the the the")
        self.assertTrue(req.is_warmup)
        self.assertFalse(req.save_output)
        self.assertEqual(req.num_inference_steps, 1)
        plan = minimax_h3_plan_from_batch(req)
        self.assertEqual(plan.task, "t2va")
        self.assertEqual(plan.prompt, "the the the")
        shape = plan.shape
        # The 1344x768x124 request (short edge 768, 16:9, 5 s) resolves to this.
        self.assertEqual(
            (shape["width"], shape["height"], shape["frame_count"]), (1344, 768, 124)
        )

    def test_warmup_flags_pick_the_canvas(self):
        reqs = tw.warmup_requests(
            _server_args(
                warmup_resolutions=["768x1344", "1344x768"], warmup_num_frames=97
            ),
            "the",
        )
        targets = [req.extra["minimax_h3_canonical_request"]["target"] for req in reqs]
        self.assertEqual(
            targets,
            [
                {"short_edge": 768, "aspect_ratio": ratio, "duration_seconds": 97 / 24}
                for ratio in ("9:16", "16:9")
            ],
        )
        shapes = [minimax_h3_plan_from_batch(req).shape for req in reqs]
        self.assertEqual(
            [(s["width"], s["height"]) for s in shapes], [(768, 1344), (1344, 768)]
        )


class TestWarm(CustomTestCase):
    def setUp(self):
        _WaitTensor.reads = 0

    def test_one_request_a_length_before_decoding(self):
        pipeline = _Pipeline()
        with self.assertLogs(level="INFO") as logs:
            result = tw.warm(pipeline, _server_args(), [3, 5])
        self.assertEqual(sorted(result["seconds"]), [3, 5])
        (summary,) = [line for line in logs.output if "prompt lengths" in line]
        self.assertIn("2 prompt lengths (3,5)", summary)
        self.assertIn(f"3 tokens {result['seconds'][3]:.2f} s", summary)
        self.assertIn(f"the other 1 {result['seconds'][5]:.2f} s a length", summary)
        before_decode = list(_STAGES[: _STAGES.index("MiniMaxH3DecodingStage")])
        self.assertEqual(pipeline.executor.calls, [before_decode, before_decode])
        self.assertEqual(pipeline.residency_installs, 1)
        self.assertEqual(pipeline.executor.conditioning_cache.cleared, 1)
        # Each request's latents and audio latents are read once (the wait).
        self.assertEqual(_WaitTensor.reads, 4)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in pipeline.executor.requests))

    def test_drains_like_a_request_where_the_platform_asks(self):
        from sglang.multimodal_gen.runtime.platforms import current_platform

        pipeline = _Pipeline()
        with (
            patch.object(
                current_platform, "drain_device_queue_after_request", return_value=True
            ),
            patch.object(torch.accelerator, "synchronize") as synchronize,
        ):
            tw.warm(pipeline, _server_args(), [3, 5])
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(_WaitTensor.reads, 0)

    def test_requests_are_one_step_warmups_with_the_listed_token_counts(self):
        seen = []
        pipeline = _Pipeline()
        execute = pipeline.executor.execute_with_profiling

        def record(stages, req, server_args):
            plan = minimax_h3_plan_from_batch(req)
            seen.append(
                (
                    len(pipeline.get_module("tokenizer")(plan.prompt)["input_ids"]),
                    req.is_warmup,
                    req.save_output,
                    req.num_inference_steps,
                )
            )
            return execute(stages, req, server_args)

        pipeline.executor.execute_with_profiling = record
        tw.warm(pipeline, _server_args(), [1, 40, 47])
        self.assertEqual(
            seen, [(1, True, False, 1), (40, True, False, 1), (47, True, False, 1)]
        )

    def test_each_warmup_resolution(self):
        pipeline = _Pipeline()
        tw.warm(
            pipeline,
            _server_args(warmup_resolutions=["1344x768", "768x1344"]),
            [2, 4],
        )
        self.assertEqual(len(pipeline.executor.calls), 4)

    def test_skips_without_the_stages_or_t2va(self):
        for pipeline in (
            _Pipeline(stages=_STAGES[:-1]),
            _Pipeline(stages=("MiniMaxH3ComfyUIStepStage",)),
            _Pipeline(tasks=("ref2va",)),
        ):
            with self.assertLogs(level="WARNING"):
                self.assertIsNone(tw.warm(pipeline, _server_args(), [3]))
            self.assertEqual(pipeline.executor.calls, [])

    def test_skips_with_layerwise_offload(self):
        pipeline = _Pipeline()
        args = _server_args(has_layerwise_offload_components=lambda: True)
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(tw.warm(pipeline, args, [3]))
        self.assertEqual(pipeline.executor.calls, [])
        args = _server_args(has_layerwise_offload_components=lambda: False)
        self.assertIsNotNone(tw.warm(pipeline, args, [3]))

    def test_skips_on_every_rank_alike_before_any_device_work(self):
        # No filler fits 4 tokens: nothing runs, not even the 3-token length.
        tokenizer = _Tokenizer(wide=tw.FILLER_WORDS, wide_from=3)
        pipeline = _Pipeline(tokenizer=tokenizer)
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(tw.warm(pipeline, _server_args(), [3, 4]))
        self.assertEqual(pipeline.executor.calls, [])
        self.assertEqual(pipeline.residency_installs, 0)

    def test_device_errors_propagate(self):
        pipeline = _Pipeline()

        def fail(stages, req, server_args):
            raise RuntimeError("device")

        pipeline.executor.execute_with_profiling = fail
        with self.assertRaises(RuntimeError):
            tw.warm(pipeline, _server_args(), [3])

    def test_debug_timing_splits_at_the_text_stage(self):
        refined = []

        class Model:
            def refine_prompt_embeds(self, x):
                refined.append(x)
                return x

        model = Model()
        pipeline = _Pipeline()
        pipeline.modules["transformer"] = model
        execute = pipeline.executor.execute_with_profiling

        def with_model(stages, req, server_args):
            req.extra["_test_model"] = model
            return execute(stages, req, server_args)

        pipeline.executor.execute_with_profiling = with_model
        with _env(SGLANG_TPU_DEBUG_H3_TEXT_WARMUP_TIMING=True):
            with self.assertLogs(level="INFO") as logs:
                tw.warm(pipeline, _server_args(), [3])
        text_end = _STAGES.index("MiniMaxH3TextEncodingStage") + 1
        decode = _STAGES.index("MiniMaxH3DecodingStage")
        self.assertEqual(
            pipeline.executor.calls,
            [list(_STAGES[:text_end]), list(_STAGES[text_end:decode])],
        )
        self.assertEqual(len(refined), 1)
        # The timing wrapper is gone: the class method serves requests again.
        self.assertNotIn("refine_prompt_embeds", vars(model))
        self.assertTrue(any("token refiner" in line for line in logs.output))


class TestInstall(CustomTestCase):
    def setUp(self):
        self._init = MiniMaxH3Pipeline.__init__
        self._installed = tw._installed
        tw._installed = False

    def tearDown(self):
        MiniMaxH3Pipeline.__init__ = self._init
        tw._installed = self._installed

    def test_warms_after_the_pipeline_is_built(self):
        order = []

        def init(self, *args, **kwargs):
            order.append("init")
            self.server_args = "args"

        MiniMaxH3Pipeline.__init__ = init
        with (
            _env(SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS="40,36-37"),
            patch.object(tw, "warm", side_effect=lambda *a: order.append(a)) as warm,
        ):
            tw.install()
            tw.install()
            pipeline = MiniMaxH3Pipeline()
        self.assertEqual(order, ["init", (pipeline, "args", [36, 37, 40])])
        warm.assert_called_once()

    def test_bad_lengths_fail_the_install(self):
        with _env(SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS="36-x"):
            with self.assertRaises(ValueError):
                tw.install()
        self.assertIs(MiniMaxH3Pipeline.__init__, self._init)

    def test_worker_installs_when_requested(self):
        from sglang_tpu.multimodal_gen import worker

        with (
            patch.object(tw, "install") as install,
            patch("sglang_tpu.multimodal_gen.rope.install"),
            patch("sglang_tpu.multimodal_gen.minimax_h3.install"),
            patch("sglang_tpu.patches._register_aten_fallbacks"),
            patch.object(torch, "set_float32_matmul_precision"),
        ):
            with _env(SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS=None):
                worker.install_runtime_patches()
            install.assert_not_called()
            with _env(SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS="36-47"):
                worker.install_runtime_patches()
            install.assert_called_once()


if __name__ == "__main__":
    unittest.main()
