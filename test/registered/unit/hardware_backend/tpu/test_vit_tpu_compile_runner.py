import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.attention.vision import (
    VisionTpuAttention,
    _build_tpu_segment_ids,
)
from sglang.srt.observability.compile_timing import CompileTimingRecorder
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.compilation import direct_compile
from sglang_tpu.graph_runner import vit_tpu_compile_runner as vit_runner
from sglang_tpu.graph_runner.vit_tpu_compile_runner import (
    Qwen3VLVisionTpuCompilePolicy,
    VisionPaddingContext,
    VisionTpuCompilePolicy,
    ViTTpuCompileRunner,
    _cu_seqlens_to_segment_ids,
    _Qwen3VLRunContext,
)

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)


@contextlib.contextmanager
def _unset(field):
    """Run with ``field`` unset, restoring whatever was there before.

    ``override`` snapshots and restores on exit, so clearing inside it is safe;
    the Envs API has no override-to-unset of its own.
    """
    with field.override(""):
        field.clear()
        yield


class _SegmentBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SimpleNamespace(
            qkv_backend=SimpleNamespace(supports_segment_ids=True)
        )
        self.graph_cu_seqlens = None
        self.segment_ids = None

    def forward(
        self,
        x,
        cu_seqlens,
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
        segment_ids=None,
    ):
        self.graph_cu_seqlens = cu_seqlens
        self.segment_ids = segment_ids
        return x


class _LegacyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SimpleNamespace(
            qkv_backend=SimpleNamespace(supports_segment_ids=False)
        )
        self.cu_seqlens = None

    def forward(
        self,
        x,
        cu_seqlens,
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
    ):
        self.cu_seqlens = cu_seqlens
        return x


def _make_vit(block):
    return SimpleNamespace(
        blocks=[block],
        merger=nn.Identity(),
        deepstack_visual_indexes=[],
        deepstack_merger_list=None,
        spatial_merge_size=1,
    )


class _Qwen3VLTowerStub(nn.Module):
    """Qwen3-VL vision tower surface the compile policy touches."""

    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.spatial_merge_size = 2
        self.dtype = torch.float32
        # 3 channels * 2 temporal * 1x1 patch == 6 features per patch.
        self.patch_embed = SimpleNamespace(
            in_channels=3, temporal_patch_size=2, patch_size=1
        )
        self.forward_with_tpu_graph = MagicMock()
        self.prologue_inputs = []

    def prepare_tpu_graph_inputs(self, x, grid_thw):
        self.prologue_inputs.append(x)
        bucket = x.shape[0]
        return (
            torch.zeros(bucket, self.hidden),
            torch.tensor([0, bucket], dtype=torch.int32),
            torch.zeros(bucket, 2),
            torch.zeros(bucket, 2),
        )


class _FakeVisionPolicy(VisionTpuCompilePolicy):
    """Smallest subclass that exercises the base ``run`` seam."""

    def compute_patch_buckets(self):
        return [4]

    def make_warmup_inputs(self, patch_count):
        return (torch.zeros(patch_count),)

    def prepare_call(self, x):
        return (x, torch.tensor([1])), {"real_rows": 2}

    def unpad_outputs(self, result, ctx):
        return result[: ctx["real_rows"]]


class _DirectVisionPolicy(_FakeVisionPolicy):
    def make_warmup_inputs(self, patch_count):
        return self.prepare_call(torch.zeros(patch_count, dtype=torch.int64))[0]

    def get_compile_model(self):
        return self.vit


class TestViTTpuCompileRunner(unittest.TestCase):
    def test_direct_warmup_queues_buckets_before_execution(self):
        owner = SimpleNamespace(forward=lambda x, extra=None: x * 2)
        recorder = CompileTimingRecorder()
        policy = _DirectVisionPolicy(
            nn.Identity(),
            owner=owner,
            compile_timing_recorder=recorder,
            model_runner=SimpleNamespace(
                server_args=SimpleNamespace(
                    tpu_compile_tracer="direct",
                    unsafe_save_torch_compile_bundle=None,
                    unsafe_load_torch_compile_bundle=None,
                )
            ),
        )
        events = []
        clock = SimpleNamespace(now=0.0)

        def compile_artifact(args, kwargs, **options):
            events.append(("compile", len(args[0]), options["async_compile"]))
            clock.now += 1

            def execute(call):
                args, kwargs = call
                events.append(("execute", len(args[0])))
                clock.now += 2
                return policy.compiled_fn.fn(*args, **kwargs)

            return execute

        def synchronize(*args, **kwargs):
            clock.now += 3

        with (
            envs.SGLANG_DIRECT_COMPILE_ASYNC.override(True),
            patch.object(vit_runner, "_precompiled_max_patch_count", None),
            patch.object(
                vit_runner, "time", SimpleNamespace(perf_counter=lambda: clock.now)
            ),
            patch.object(vit_runner, "tpu_sync", side_effect=synchronize) as sync,
            patch.object(
                direct_compile.CompiledFunction,
                "_compile",
                side_effect=compile_artifact,
            ),
        ):
            self.assertTrue(policy.install())
            self.assertIs(owner.forward, policy.compiled_fn)
            pending = policy.precompile([4, 8])
            self.assertEqual(events, [("compile", 4, True), ("compile", 8, True)])
            sync.assert_not_called()
            self.assertEqual(recorder.snapshot(), [])

            clock.now += 100  # Other warmups must not inflate this shape's timing.
            for call in pending:
                call.wait()
            self.assertEqual(events[-2:], [("execute", 4), ("execute", 8)])
            self.assertEqual(sync.call_count, 2)
            self.assertTrue(all(c.kwargs["wait"] for c in sync.call_args_list))
            steps = recorder.snapshot()
            self.assertEqual([step["num_tokens"] for step in steps], [4, 8])
            self.assertEqual([step["duration_seconds"] for step in steps], [6, 6])

            self.assertEqual(policy.run(torch.arange(4)).tolist(), [0, 2])
            self.assertEqual(len(policy.compiled_fn.cache), 2)

    def test_default_policy_keeps_dynamo_with_async_enabled(self):
        policy = _FakeVisionPolicy(
            model_runner=SimpleNamespace(
                server_args=SimpleNamespace(tpu_compile_tracer="direct")
            )
        )
        with envs.SGLANG_DIRECT_COMPILE_ASYNC.override(True):
            compiled = policy._compile_method(lambda x: x, use_bundle=False)
        self.assertNotIsInstance(compiled, direct_compile.CompiledFunction)

    def test_policy_declares_tpu_platform(self):
        self.assertEqual(VisionTpuCompilePolicy.platform, "tpu")

    def test_qwen3_vl_policy_extends_base_and_implements_every_hook(self):
        # A base hook left unimplemented here would not surface until the policy
        # is instantiated at install time, i.e. in a serving run.
        self.assertEqual(Qwen3VLVisionTpuCompilePolicy.platform, "tpu")
        self.assertTrue(
            issubclass(Qwen3VLVisionTpuCompilePolicy, VisionTpuCompilePolicy)
        )
        self.assertEqual(
            getattr(Qwen3VLVisionTpuCompilePolicy, "__abstractmethods__", frozenset()),
            frozenset(),
        )
        # The base gate must be permissive: an encoder that does not override it
        # should compile, not silently fall back to eager.
        self.assertTrue(
            VisionTpuCompilePolicy._runner_ready_for_compile(None, object())
        )

    def test_base_run_drives_prepare_then_compiled_fn_then_unpad(self):
        policy = _FakeVisionPolicy()
        with self.assertRaisesRegex(RuntimeError, "before install"):
            policy.run(torch.zeros(4))

        policy.compiled_fn = MagicMock(side_effect=lambda x, extra: x * 2)
        result = policy.run(torch.arange(5).float())

        # prepare_call's tuple is splatted into the compiled fn ...
        self.assertEqual(policy.compiled_fn.call_args.args[1].tolist(), [1])
        # ... and its ctx drives unpad_outputs on the way back.
        self.assertEqual(result.tolist(), [0.0, 2.0])

    def test_base_run_sends_inputs_past_the_ladder_to_eager_fn(self):
        """Any policy reporting a patch count gets the fallback from the base."""

        class _SizedPolicy(_FakeVisionPolicy):
            def input_patch_count(self, x):
                return x.shape[0]

        policy = _SizedPolicy()
        policy.compiled_fn = MagicMock(side_effect=lambda x, extra: x * 2)
        policy.eager_fn = MagicMock(side_effect=lambda x, extra: x * 3)

        # 4 patches fit the [4] ladder; 5 do not.
        self.assertEqual(policy.run(torch.arange(4).float()).tolist(), [0.0, 2.0])
        self.assertEqual(policy.run(torch.arange(5).float()).tolist(), [0.0, 3.0])
        policy.compiled_fn.assert_called_once()
        # The eager fn gets prepare_call's arguments, like the compiled one.
        self.assertEqual(policy.eager_fn.call_args.args[1].tolist(), [1])

    def test_base_run_without_a_patch_count_keeps_the_compiled_fn(self):
        """A policy that reports no count is unchanged, even past the ladder."""
        policy = _FakeVisionPolicy()
        policy.compiled_fn = MagicMock(side_effect=lambda x, extra: x * 2)
        policy.eager_fn = MagicMock()

        policy.run(torch.arange(5).float())

        policy.eager_fn.assert_not_called()
        policy.compiled_fn.assert_called_once()

    def test_qwen3_vl_patch_buckets_honor_the_env_override(self):
        policy = Qwen3VLVisionTpuCompilePolicy(SimpleNamespace(spatial_merge_size=2))
        sizes = envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES

        with _unset(sizes), _unset(envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES):
            ladder = policy.compute_patch_buckets()
        self.assertEqual((ladder[0], ladder[-1]), (196, 4900))

        # Sorted and de-duplicated, and validated against the merge window.
        with sizes.override("324, 196 ,196"):
            self.assertEqual(policy.compute_patch_buckets(), [196, 324])
        # Set but empty disables precompile.
        with sizes.override(""):
            self.assertEqual(policy.compute_patch_buckets(), [])
        with (
            sizes.override("200"),
            self.assertRaisesRegex(ValueError, "perfect squares"),
        ):
            policy.compute_patch_buckets()

        # resolve_patch_buckets caches, so a later env change is not picked up.
        with sizes.override("196"):
            self.assertEqual(policy.resolve_patch_buckets(), [196])
        with sizes.override("324"):
            self.assertEqual(policy.resolve_patch_buckets(), [196])

    def test_env_cap_bounds_the_ladder_but_is_not_its_top_bucket(self):
        policy = Qwen3VLVisionTpuCompilePolicy(SimpleNamespace(spatial_merge_size=2))
        with (
            _unset(envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES),
            envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.override(5000),
        ):
            ladder = policy.compute_patch_buckets()
        # 72^2 = 5184 overshoots the cap, so the real ceiling stays 70^2. Callers
        # sizing a batch must read the warmed value rather than the cap.
        self.assertEqual(ladder[-1], 4900)

    def test_precompile_publishes_the_top_warmed_bucket(self):
        recorder = CompileTimingRecorder()
        policy = _FakeVisionPolicy(compile_timing_recorder=recorder)
        policy.owner = SimpleNamespace()
        policy.compiled_fn = MagicMock(return_value=torch.zeros(1))
        with (
            patch.object(vit_runner, "_precompiled_max_patch_count", None),
            patch.object(vit_runner, "tpu_sync"),
            patch.object(
                vit_runner,
                "time",
                SimpleNamespace(
                    perf_counter=MagicMock(side_effect=[0, 1, 10, 12, 20, 23])
                ),
            ),
        ):
            policy.precompile(patch_counts=[196, 324, 4900])
            self.assertEqual(vit_runner.precompiled_max_patch_count(), 4900)
        steps = recorder.snapshot()
        self.assertEqual([step["num_tokens"] for step in steps], [196, 324, 4900])
        self.assertEqual([step["duration_seconds"] for step in steps], [1, 2, 3])

    def test_warmup_syncs_every_tensor_of_a_tuple_result(self):
        # A (features, mask) boundary hands warmup a tuple; sync each tensor in it.
        policy = _FakeVisionPolicy()
        policy.owner = SimpleNamespace()
        pooled, mask = torch.zeros(2), torch.ones(2)
        policy.compiled_fn = MagicMock(return_value=(pooled, mask))

        with patch.object(vit_runner, "tpu_sync") as sync:
            policy.warmup_once(4)

        (tensors,), kwargs = sync.call_args
        self.assertEqual(len(tensors), 2)
        self.assertIs(tensors[0], pooled)
        self.assertIs(tensors[1], mask)
        self.assertTrue(kwargs["wait"])

    def test_qwen3_vl_warmup_covers_the_whole_pass_not_just_the_compiled_region(self):
        """Warmup must enter at the encoder, ahead of patch_embed.

        patch_embed and the host-staged prologue sit outside the compiled
        region but are still shape-static per bucket. Warming only compiled_fn
        (the base class default) leaves them to compile on the first real
        request of each bucket, which is exactly what this override prevents.
        """
        patch_embed = SimpleNamespace(
            in_channels=3, temporal_patch_size=2, patch_size=16
        )
        vit = SimpleNamespace(
            spatial_merge_size=2,
            patch_embed=patch_embed,
            dtype=torch.float32,
            forward_with_tpu_graph=MagicMock(return_value=torch.zeros(1)),
        )
        policy = Qwen3VLVisionTpuCompilePolicy(vit)

        policy.warmup_once(324)

        (x, grid_thw), _ = vit.forward_with_tpu_graph.call_args
        # 3 channels * 2 temporal * 16 * 16 == 1536 features per patch.
        self.assertEqual(tuple(x.shape), (324, 1536))
        self.assertEqual(x.device.type, "cpu")
        self.assertEqual(grid_thw.device.type, "cpu")
        # 324 == 18^2, so a square grid lands exactly on the bucket.
        self.assertEqual(grid_thw.tolist(), [[1, 18, 18]])

    def test_qwen3_vl_supplies_the_vision_tower_for_direct_compile(self):
        tower = nn.Linear(2, 2)
        self.assertIs(Qwen3VLVisionTpuCompilePolicy(tower).get_compile_model(), tower)
        # A non-module stand-in keeps the Dynamo path rather than tracing
        # something direct_compile cannot enumerate state for.
        self.assertIsNone(
            Qwen3VLVisionTpuCompilePolicy(
                SimpleNamespace(spatial_merge_size=2)
            ).get_compile_model()
        )

    def test_qwen3_vl_direct_warmup_queues_the_serve_time_boundary(self):
        """Under direct compile, buckets are submitted, not compiled inline.

        The warmup must hand warmup_async the same argument tuple run() passes
        at serve time; otherwise the first real image of each bucket misses the
        specialization cache and compiles synchronously.
        """
        hidden = 4
        vit = _Qwen3VLTowerStub(hidden)
        runner = SimpleNamespace(
            _supports_segment_ids=True, forward=lambda x, cu, cos, sin, seg: x
        )
        vit.graph_runners = runner
        policy = Qwen3VLVisionTpuCompilePolicy(
            vit,
            owner=vit,
            model_runner=SimpleNamespace(
                server_args=SimpleNamespace(
                    tpu_compile_tracer="direct",
                    unsafe_save_torch_compile_bundle=None,
                    unsafe_load_torch_compile_bundle=None,
                )
            ),
        )
        events = []

        def compile_artifact(args, kwargs, **options):
            events.append(("compile", args[0].shape[0], options["async_compile"]))

            def execute(call):
                args, kwargs = call
                events.append(("execute", args[0].shape[0]))
                return policy.compiled_fn.fn(*args, **kwargs)

            return execute

        with (
            envs.SGLANG_DIRECT_COMPILE_ASYNC.override(True),
            envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(False),
            patch.object(vit_runner, "_precompiled_max_patch_count", None),
            patch.object(vit_runner, "tpu_sync") as sync,
            patch.object(
                direct_compile.CompiledFunction,
                "_compile",
                side_effect=compile_artifact,
            ),
        ):
            self.assertTrue(policy.install())
            self.assertIsInstance(policy.compiled_fn, direct_compile.CompiledFunction)
            pending = policy.precompile([4, 16])

            self.assertEqual(events, [("compile", 4, True), ("compile", 16, True)])
            self.assertEqual(
                [tuple(x.shape) for x in vit.prologue_inputs], [(4, 6), (16, 6)]
            )
            vit.forward_with_tpu_graph.assert_not_called()
            sync.assert_not_called()

            for call in pending:
                call.wait()
            self.assertEqual(events[-2:], [("execute", 4), ("execute", 16)])
            self.assertEqual(sync.call_count, 2)

            # A real 3x4 image padded into the 16 bucket reuses the warmed graph.
            out = runner.run(
                torch.zeros(16, hidden),
                torch.tensor([0, 12], dtype=torch.int32),
                rotary_pos_emb_cos=torch.zeros(16, 2),
                rotary_pos_emb_sin=torch.zeros(16, 2),
            )
            self.assertEqual(len(policy.compiled_fn.cache), 2)
            self.assertEqual(events[-1], ("execute", 16))
            # 12 real patches with a 2x2 merge leave 3 merged rows.
            self.assertEqual(out.shape[0], 3)

    def test_precompile_publishes_nothing_when_warming_is_disabled(self):
        policy = _FakeVisionPolicy()
        policy.owner = SimpleNamespace()
        policy.compiled_fn = MagicMock(return_value=torch.zeros(1))
        with patch.object(vit_runner, "_precompiled_max_patch_count", None):
            policy.precompile(patch_counts=[])
            self.assertIsNone(vit_runner._precompiled_max_patch_count)

    def test_batching_budget_prefers_the_warmed_ladder_over_the_env_cap(self):
        with patch.object(vit_runner, "_precompiled_max_patch_count", 2304):
            self.assertEqual(vit_runner.precompiled_max_patch_count(), 2304)

    def test_batching_budget_falls_back_to_env_before_precompile_runs(self):
        unwarmed = patch.object(vit_runner, "_precompiled_max_patch_count", None)
        with (
            unwarmed,
            _unset(envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES),
            envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.override(1600),
        ):
            self.assertEqual(vit_runner.precompiled_max_patch_count(), 1600)
        with unwarmed, envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.override("196, 1024"):
            self.assertEqual(vit_runner.precompiled_max_patch_count(), 1024)

    def test_qwen3_vl_unpad_trims_bucket_padding_and_selects_rows(self):
        policy = Qwen3VLVisionTpuCompilePolicy(SimpleNamespace(spatial_merge_size=2))
        padded = torch.arange(8).float()
        # 12 real patches in a 16 bucket, 2x2 merge => 3 real output rows.
        padding = VisionPaddingContext(
            real_seq_len=12, bucket_len=16, padded=True, merge_sq=4
        )

        trimmed = policy.unpad_outputs(
            padded, _Qwen3VLRunContext(padding=padding, output_indices=None)
        )
        self.assertEqual(trimmed.tolist(), [0.0, 1.0, 2.0])

        selected = policy.unpad_outputs(
            padded,
            _Qwen3VLRunContext(padding=padding, output_indices=torch.tensor([2, 0])),
        )
        self.assertEqual(selected.tolist(), [2.0, 0.0])

    def test_qwen3_vl_unpad_moves_to_host_before_trimming(self):
        """The D2H must happen while the shape is still the static bucket.

        Everything downstream of unpad (per-image split, embedding cache, chunk
        assembly) is variable-length; leaving it on device recompiles once per
        length. Asserting the device of the result cannot catch a regression
        here because unit-test tensors are already on CPU, so spy on the shape
        at which .to() is called instead: it must be the bucket, not the trim.
        """

        class _HostMoveSpy:
            """Stands in for the compiled ViT output; records .to() calls."""

            def __init__(self, tensor):
                self._tensor = tensor
                self.to_calls = []

            def to(self, device):
                self.to_calls.append((device, tuple(self._tensor.shape)))
                return self._tensor

        policy = Qwen3VLVisionTpuCompilePolicy(SimpleNamespace(spatial_merge_size=2))
        spy = _HostMoveSpy(torch.arange(8).float())
        padding = VisionPaddingContext(
            real_seq_len=12, bucket_len=16, padded=True, merge_sq=4
        )

        trimmed = policy.unpad_outputs(
            spy, _Qwen3VLRunContext(padding=padding, output_indices=None)
        )

        self.assertEqual(spy.to_calls, [("cpu", (8,))])
        self.assertEqual(trimmed.tolist(), [0.0, 1.0, 2.0])

    def test_qwen3_vl_prepare_call_consumes_host_padded_inputs(self):
        """prepare_call reads the real length off cu_seqlens, not off x_3d.

        Padding moved to the host, so x_3d already spans the bucket. Deriving
        real_seq_len from its leading dim would make unpad_outputs stop
        trimming and leak padded merger rows downstream.
        """
        policy = Qwen3VLVisionTpuCompilePolicy(SimpleNamespace(spatial_merge_size=2))
        bucket_len = 12
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        x_3d = torch.zeros(bucket_len, 1, 4)
        cos = torch.zeros(bucket_len, 8)
        sin = torch.zeros(bucket_len, 8)

        args, ctx = policy.prepare_call(x_3d, cu_seqlens, cos, sin)
        out_x, graph_cu_seqlens, out_cos, out_sin, segment_ids = args

        self.assertIs(out_x, x_3d)
        self.assertIs(out_cos, cos)
        self.assertIs(out_sin, sin)
        self.assertEqual(graph_cu_seqlens.tolist(), [0, bucket_len])
        self.assertEqual(segment_ids.tolist(), [[0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]])
        self.assertEqual(ctx.padding.real_seq_len, 5)
        self.assertEqual(ctx.padding.bucket_len, bucket_len)
        self.assertTrue(ctx.padding.padded)

    @staticmethod
    def _installed_qwen3_vl_policy(buckets):
        """A Qwen3-VL policy wired onto a runner, with both fns as spies."""
        policy = Qwen3VLVisionTpuCompilePolicy(SimpleNamespace(spatial_merge_size=1))
        policy._patch_buckets = list(buckets)

        def boundary(x, cu_seqlens, cos, sin, segment_ids):
            return x[:, 0] + 1

        # Stand in for install(): eager is the runner's forward, compiled is its
        # torch.compile wrapper; both compute the same thing here.
        policy.eager_fn = MagicMock(side_effect=boundary)
        policy.compiled_fn = MagicMock(side_effect=boundary)
        runner = SimpleNamespace()
        policy._install_run_adapter(runner)
        return policy, runner

    def test_qwen3_vl_images_past_the_ladder_run_eagerly(self):
        """Past the top bucket, recompiling would raise under fail_on_recompile
        and take down every scheduler rank, so these images run uncompiled."""
        policy, runner = self._installed_qwen3_vl_policy([16])
        # The caller found no bucket, so it left x at its real 20 patches.
        x = torch.arange(20 * 4, dtype=torch.float32).reshape(20, 4)
        rope = torch.zeros(20, 2)
        cu_seqlens = torch.tensor([0, 8, 20], dtype=torch.int32)

        out = runner.run(
            x, cu_seqlens, rotary_pos_emb_cos=rope, rotary_pos_emb_sin=rope
        )

        policy.compiled_fn.assert_not_called()
        policy.eager_fn.assert_called_once()
        x_3d, graph_cu_seqlens, _, _, segment_ids = policy.eager_fn.call_args.args
        self.assertEqual(tuple(x_3d.shape), (20, 1, 4))
        # Same boundary arguments the compiled fn takes: the segmentation rides
        # on segment_ids, and with no padding there is no trailing segment.
        self.assertEqual(graph_cu_seqlens.tolist(), [0, 20])
        self.assertEqual(segment_ids.tolist(), [[0] * 8 + [1] * 12])
        torch.testing.assert_close(out, x + 1)

    def test_qwen3_vl_bucketed_images_stay_compiled(self):
        policy, runner = self._installed_qwen3_vl_policy([16])
        # 12 real patches, padded on the host to the 16 bucket.
        x = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)
        rope = torch.zeros(16, 2)
        cu_seqlens = torch.tensor([0, 12], dtype=torch.int32)

        out = runner.run(
            x, cu_seqlens, rotary_pos_emb_cos=rope, rotary_pos_emb_sin=rope
        )

        policy.eager_fn.assert_not_called()
        policy.compiled_fn.assert_called_once()
        torch.testing.assert_close(out, x[:12] + 1)

    def test_segment_ids_are_built_from_host_cumulative_lengths(self):
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

        segment_ids = _cu_seqlens_to_segment_ids(
            cu_seqlens, total_patches=5, device=torch.device("cpu")
        )

        self.assertEqual(segment_ids.dtype, torch.int32)
        self.assertEqual(segment_ids.tolist(), [[0, 0, 1, 1, 1]])

    def test_shared_tpu_attention_builds_packed_segment_ids(self):
        segment_ids = _build_tpu_segment_ids(
            torch.tensor([0, 2, 5], dtype=torch.int32),
            bsz=1,
            seq_len=5,
            flatten_batch=True,
            device=torch.device("cpu"),
        )

        self.assertEqual(segment_ids.tolist(), [[0, 0, 1, 1, 1]])

        segment_ids = _build_tpu_segment_ids(
            torch.tensor([0, 0, 2, 2, 5], dtype=torch.int32),
            bsz=1,
            seq_len=5,
            flatten_batch=True,
            device=torch.device("cpu"),
        )

        self.assertEqual(segment_ids.tolist(), [[1, 1, 3, 3, 3]])

    def test_shared_tpu_attention_separates_dense_padding(self):
        segment_ids = _build_tpu_segment_ids(
            torch.tensor([0, 3, 5], dtype=torch.int32),
            bsz=2,
            seq_len=3,
            flatten_batch=False,
            device=torch.device("cpu"),
        )

        self.assertEqual(segment_ids.tolist(), [[0, 0, 0], [0, 0, 1]])

    def test_shared_tpu_attention_supports_batched_flattened_callers(self):
        flash_op = MagicMock(side_effect=lambda q, k, v, ids: torch.zeros_like(q))
        with (
            patch("sglang.srt.layers.attention.vision._is_tpu", True),
            patch(
                "sglang.srt.layers.attention.vision._get_vision_flash_op",
                return_value=flash_op,
            ),
        ):
            attention = VisionTpuAttention(
                head_dim=4,
                num_heads=1,
                num_kv_heads=1,
                flatten_batch=True,
            )
        q = torch.zeros(6, 1, 4)

        result = attention(q, q, q, bsz=2)

        self.assertEqual(tuple(result.shape), (6, 1, 4))
        segment_ids = flash_op.call_args.args[-1]
        self.assertEqual(tuple(segment_ids.shape), (2, 128))
        self.assertEqual(segment_ids[:, :3].tolist(), [[0, 0, 0], [0, 0, 0]])
        self.assertTrue(torch.all(segment_ids[:, 3:] == 1))

    def test_shared_tpu_attention_delegates_explicit_masks_to_sdpa(self):
        with (
            patch("sglang.srt.layers.attention.vision._is_tpu", True),
            patch(
                "sglang.srt.layers.attention.vision._get_vision_flash_op",
                return_value=MagicMock(),
            ),
        ):
            attention = VisionTpuAttention(
                head_dim=4,
                num_heads=2,
                num_kv_heads=1,
                flatten_batch=True,
            )
        expected = torch.zeros(6, 2, 4)
        attention._sdpa_fallback.forward = MagicMock(return_value=expected)
        q = torch.zeros(6, 2, 4)
        kv = torch.zeros(6, 1, 4)
        mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)

        result = attention(q, kv, kv, bsz=2, attention_mask=mask)

        self.assertIs(result, expected)
        attention._sdpa_fallback.forward.assert_called_once()
        _, expanded_k, expanded_v, _ = attention._sdpa_fallback.forward.call_args.args
        self.assertEqual(expanded_k.shape[1], 2)
        self.assertEqual(expanded_v.shape[1], 2)

        with self.assertRaisesRegex(ValueError, "cannot combine"):
            attention(
                q,
                kv,
                kv,
                bsz=2,
                attention_mask=mask,
                segment_ids=torch.zeros(2, 3, dtype=torch.int32),
            )

    def test_shared_tpu_attention_uses_pallas_for_fp32_softmax(self):
        flash_op = MagicMock(side_effect=lambda q, k, v, ids: torch.zeros_like(q))
        with (
            patch("sglang.srt.layers.attention.vision._is_tpu", True),
            patch(
                "sglang.srt.layers.attention.vision._get_vision_flash_op",
                return_value=flash_op,
            ),
        ):
            attention = VisionTpuAttention(
                head_dim=4,
                num_heads=1,
                num_kv_heads=1,
                flatten_batch=True,
                softmax_in_single_precision=True,
            )
        attention._sdpa_fallback.forward = MagicMock()
        q = torch.zeros(3, 1, 4)

        result = attention(q, q, q, bsz=1)

        self.assertEqual(tuple(result.shape), (3, 1, 4))
        flash_op.assert_called_once()
        attention._sdpa_fallback.forward.assert_not_called()

    def test_shared_tpu_attention_expands_kv_heads(self):
        flash_op = MagicMock(side_effect=lambda q, k, v, ids: torch.zeros_like(q))
        with (
            patch("sglang.srt.layers.attention.vision._is_tpu", True),
            patch(
                "sglang.srt.layers.attention.vision._get_vision_flash_op",
                return_value=flash_op,
            ),
        ):
            attention = VisionTpuAttention(
                head_dim=4,
                num_heads=2,
                num_kv_heads=1,
                flatten_batch=True,
            )
        q = torch.zeros(3, 2, 4)
        kv = torch.zeros(3, 1, 4)

        attention(q, kv, kv, bsz=1)

        _, k, v, _ = flash_op.call_args.args
        self.assertEqual(k.shape[1], 2)
        self.assertEqual(v.shape[1], 2)

    def test_eager_segment_backend_receives_segment_ids(self):
        block = _SegmentBlock()
        runner = ViTTpuCompileRunner(_make_vit(block))
        x = torch.zeros(5, 4)
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        rotary = torch.zeros(5, 2)

        result = runner.run(
            x,
            cu_seqlens,
            rotary_pos_emb_cos=rotary,
            rotary_pos_emb_sin=rotary,
        )

        self.assertEqual(tuple(result.shape), (5, 1, 4))
        self.assertEqual(block.graph_cu_seqlens.tolist(), [0, 5])
        self.assertEqual(block.segment_ids.tolist(), [[0, 0, 1, 1, 1]])

    def test_eager_legacy_backend_keeps_original_cumulative_lengths(self):
        block = _LegacyBlock()
        runner = ViTTpuCompileRunner(_make_vit(block))
        x = torch.zeros(5, 4)
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        rotary = torch.zeros(5, 2)

        runner.run(
            x,
            cu_seqlens,
            rotary_pos_emb_cos=rotary,
            rotary_pos_emb_sin=rotary,
        )

        self.assertIs(block.cu_seqlens, cu_seqlens)


class TestRunnerHostState(unittest.TestCase):
    """State the compile policy publishes onto the runner, and its readers."""

    @staticmethod
    def _runner(**vit_attrs):
        """Bare ViTTpuCompileRunner: __init__ inspects a real ViT we do not need."""
        runner = ViTTpuCompileRunner.__new__(ViTTpuCompileRunner)
        runner.vit = SimpleNamespace(**vit_attrs)
        runner.padded_patch_buckets = ()
        runner._warned_oversize_patches = set()
        runner._rotary_cos_sin_cache_cpu = None
        return runner

    def test_bucket_for_patches_is_identity_without_a_ladder(self):
        """No published ladder means no padding.

        The ladder only exists once the compile policy installs, so an
        uninstalled runner must keep the exact patch count.
        """
        self.assertEqual(self._runner().bucket_for_patches(1234), 1234)

    def test_bucket_for_patches_rounds_up_and_degrades_past_the_top(self):
        """Oversize inputs fall back to their own length instead of raising."""
        runner = self._runner()
        runner.padded_patch_buckets = (196, 256, 324)
        self.assertEqual(runner.bucket_for_patches(196), 196)
        self.assertEqual(runner.bucket_for_patches(197), 256)
        self.assertEqual(runner.bucket_for_patches(999), 999)

    def test_host_rotary_cache_is_taken_once_and_detached(self):
        """The mirror is lazy and must not alias the live device cache.

        Loaders can build the ViT under a meta device, so the copy cannot be
        taken at construction; aliasing it would let a later device-side write
        change what the host prologue gathers.
        """
        cache = torch.randn(16, 8)
        runner = self._runner(rotary_pos_emb=SimpleNamespace(cos_sin_cache=cache))

        first = runner._host_rotary_cache()
        self.assertIsNot(first, cache)
        self.assertTrue(torch.equal(first, cache))

        cache.zero_()
        self.assertIs(runner._host_rotary_cache(), first)
        self.assertFalse(torch.equal(runner._host_rotary_cache(), cache))

    def test_rot_pos_emb_cpu_matches_eager_and_pads(self):
        """Host rot_pos_emb must match the eager path and zero-pad the tail.

        _rot_pos_emb_cpu mirrors rot_pos_emb against the host cache; a
        mis-sized max_grid_size slice or a lost pad would feed the encoder
        wrong rotary factors.
        """
        from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
        from sglang.srt.models.utils import RotaryPosMixin

        merge, rotary_dim = 2, 8
        cache = torch.randn(256, 2 * rotary_dim)
        vit = SimpleNamespace(
            device=torch.device("cpu"),
            spatial_merge_size=merge,
            rot_pos_ids=RotaryPosMixin.rot_pos_ids,
            rotary_pos_emb=SimpleNamespace(
                cos_sin_cache=cache,
                get_cos_sin=lambda seqlen: cache[:seqlen].chunk(2, dim=-1),
            ),
        )
        runner = self._runner(**vars(vit))

        for grid in ([[1, 16, 16]], [[4, 16, 20]], [[1, 16, 24], [1, 32, 12]]):
            ref_cos, ref_sin = Qwen3VLMoeVisionModel.rot_pos_emb(vit, grid)
            real_len = ref_cos.shape[0]
            bucket_len = real_len + 3 * merge**2

            cos, sin = runner._rot_pos_emb_cpu(grid, bucket_len=bucket_len)

            self.assertEqual(cos.shape, (bucket_len, ref_cos.shape[1]), str(grid))
            self.assertTrue(torch.equal(cos[:real_len], ref_cos), str(grid))
            self.assertTrue(torch.equal(sin[:real_len], ref_sin), str(grid))


if __name__ == "__main__":
    unittest.main()
