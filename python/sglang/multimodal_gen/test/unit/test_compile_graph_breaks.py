"""Code that runs inside a torch.compile'd DiT forward must trace as one graph
and must not recompile from one denoise step to the next."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2Config
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.layers.layernorm import FP32LayerNorm
from sglang.multimodal_gen.runtime.layers.rotary_embedding.mrope import (
    OneDRotaryEmbedding,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    build_torch_compile_kwargs,
)


def _compile(fn):
    torch._dynamo.reset()
    return torch.compile(fn, backend="eager", fullgraph=True)


class TestCompileGraphBreaks(unittest.TestCase):
    def test_fp32_layernorm_traces_without_a_break(self):
        norm = FP32LayerNorm(16, eps=1e-5).to(torch.bfloat16)
        inputs = torch.randn(4, 16, dtype=torch.bfloat16)

        with torch.no_grad():
            # An eager call first fills the fp32 parameter cache.
            expected = norm(inputs)
            torch.testing.assert_close(_compile(norm)(inputs), expected)

    def test_req_delegates_sampling_params_without_a_break(self):
        req = Req(sampling_params=SamplingParams(num_frames=5))

        def frames(x):
            return x * req.num_frames

        self.assertEqual(_compile(frames)(torch.ones(2)).tolist(), [5.0, 5.0])

    def test_model_config_extra_attrs_resolve_without_a_break(self):
        config = LTX2Config()
        config.arch_config.ltx_variant = "ltx_2"

        def scale(x):
            if getattr(config, "ltx_variant", "ltx_2") == "ltx_2_3":
                return x * 2
            return x + getattr(config, "missing_attr", 1)

        self.assertEqual(_compile(scale)(torch.ones(2)).tolist(), [2.0, 2.0])

    def test_1d_rope_traces_without_a_break(self):
        rope = OneDRotaryEmbedding(dim=8)
        pos = torch.arange(6)
        want = rope(pos)
        got = _compile(rope)(pos)
        for g, w in zip(got, want):
            torch.testing.assert_close(g, w)

    def test_platform_amp_query_traces_without_a_break(self):
        def cast(x):
            return x if current_platform.is_amp_supported() else x.float()

        self.assertEqual(_compile(cast)(torch.ones(2)).tolist(), [1.0, 1.0])


class TestDenoisingStepTimestep(unittest.TestCase):
    def test_step_timestep_is_not_a_view_of_the_schedule(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
            DenoisingStage,
        )

        stage = DenoisingStage.__new__(DenoisingStage)
        stage._select_and_manage_model = MagicMock(return_value=(MagicMock(), 1.0))
        stage._prepare_step_attn_metadata = MagicMock(return_value=None)
        timesteps = torch.linspace(999.0, 0.0, 4)
        ctx = SimpleNamespace(timesteps=timesteps, boundary_timestep=None)

        for step_index, t_host in enumerate(timesteps):
            state = stage._prepare_step_state(
                ctx, None, None, step_index, t_host, timesteps
            )
            # torch_tpu keys compiled executables on the storage offset.
            self.assertEqual(state.t_device.storage_offset(), 0)
            self.assertNotEqual(
                state.t_device.untyped_storage().data_ptr(),
                timesteps.untyped_storage().data_ptr(),
            )
            self.assertEqual(state.t_device.item(), t_host.item())


class _ScaleShift(torch.nn.Module):
    def forward(self, x, shift, scale):
        return x * (1 + scale) + shift


class TestWanStepCounter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.multimodal_gen.runtime.distributed import parallel_state

        cls._rendezvous = None
        if not parallel_state.model_parallel_is_initialized():
            cls._rendezvous = tempfile.TemporaryDirectory()
            parallel_state.maybe_init_distributed_environment_and_model_parallel(
                tp_size=1,
                sp_size=1,
                distributed_init_method=f"file://{cls._rendezvous.name}/store",
            )

    @classmethod
    def tearDownClass(cls):
        from sglang.multimodal_gen.runtime.distributed import parallel_state

        if cls._rendezvous is not None:
            parallel_state.cleanup_dist_env_and_memory()
            cls._rendezvous.cleanup()

    def _tiny_wan(self):
        from sglang.multimodal_gen.configs.models.dits.wanvideo import (
            WanVideoArchConfig,
            WanVideoConfig,
        )
        from sglang.multimodal_gen.configs.pipeline_configs.base import (
            PipelineConfig,
        )
        from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
            WanTransformer3DModel,
        )
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        with patch.object(PipelineConfig, "from_kwargs", return_value=PipelineConfig()):
            server_args = ServerArgs.from_kwargs(model_path="/fake")
        arch_config = WanVideoArchConfig(
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=4,
            out_channels=4,
            text_dim=16,
            freq_dim=16,
            ffn_dim=32,
            num_layers=0,
        )
        with patch(
            "sglang.multimodal_gen.runtime.server_args.server_args._global_server_args",
            server_args,
        ):
            model = WanTransformer3DModel(
                WanVideoConfig(arch_config=arch_config), hf_config={}
            )
        # The counter lives outside the blocks; the CPU fused output norm needs
        # sgl_kernel's CPU build.
        model.norm_out = _ScaleShift()
        return model.eval()

    def test_counter_stays_put_without_teacache_and_forward_compiles_once(self):
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        model = self._tiny_wan()
        graphs = []

        def count_graphs(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        torch._dynamo.reset()
        compiled = torch.compile(model, backend=count_graphs)
        req = Req(sampling_params=SamplingParams(enable_teacache=False))
        hidden_states = torch.randn(1, 4, 1, 4, 4)
        encoder_hidden_states = torch.randn(1, 8, 16)
        graphs_after_first_step = None
        with torch.no_grad():
            for step, t in enumerate((999, 666, 333, 0)):
                with set_forward_context(
                    current_timestep=step, attn_metadata=None, forward_batch=req
                ):
                    compiled(hidden_states, encoder_hidden_states, torch.tensor([t]))
                if graphs_after_first_step is None:
                    graphs_after_first_step = len(graphs)

        self.assertEqual(model.cnt, 0)
        self.assertEqual(len(graphs), graphs_after_first_step)

        # A second request with the same parameters reuses the graph: dynamo
        # does not guard on the identity of the per-request Req.
        other_req = Req(sampling_params=SamplingParams(enable_teacache=False))
        with (
            torch.no_grad(),
            set_forward_context(
                current_timestep=0, attn_metadata=None, forward_batch=other_req
            ),
        ):
            compiled(hidden_states, encoder_hidden_states, torch.tensor([999]))
        self.assertEqual(len(graphs), graphs_after_first_step)

    def test_lingbot_world_counts_steps_only_for_teacache(self):
        from sglang.multimodal_gen.configs.models.dits.lingbot_world import (
            LingBotWorldArchConfig,
            LingBotWorldVideoConfig,
        )
        from sglang.multimodal_gen.configs.pipeline_configs.base import (
            PipelineConfig,
        )
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )
        from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
            LingBotWorldTransformer3DModel,
        )
        from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
            WanTransformer3DModel,
        )
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        # Wan's TeaCache step check, which reads the counter, is borrowed.
        self.assertIs(
            LingBotWorldTransformer3DModel.should_skip_forward_for_cached_states,
            WanTransformer3DModel.should_skip_forward_for_cached_states,
        )
        with patch.object(PipelineConfig, "from_kwargs", return_value=PipelineConfig()):
            server_args = ServerArgs.from_kwargs(model_path="/fake")
        arch_config = LingBotWorldArchConfig(
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=4,
            out_channels=4,
            text_dim=16,
            freq_dim=16,
            ffn_dim=32,
            num_layers=0,
        )
        with patch(
            "sglang.multimodal_gen.runtime.server_args.server_args._global_server_args",
            server_args,
        ):
            model = LingBotWorldTransformer3DModel(
                LingBotWorldVideoConfig(arch_config=arch_config), hf_config={}
            ).eval()
        model.norm_out = _ScaleShift()
        for enable_teacache, expected in ((False, 0), (True, 3)):
            model.cnt = 0
            req = Req(sampling_params=SamplingParams(enable_teacache=enable_teacache))
            with (
                self.subTest(enable_teacache=enable_teacache),
                # The cache itself is out of scope: never skip, never store.
                patch.object(
                    LingBotWorldTransformer3DModel,
                    "should_skip_forward_for_cached_states",
                    return_value=False,
                ),
                patch.object(LingBotWorldTransformer3DModel, "maybe_cache_states"),
                torch.no_grad(),
            ):
                for step, t in enumerate((999, 500, 0)):
                    with set_forward_context(
                        current_timestep=step, attn_metadata=None, forward_batch=req
                    ):
                        model(
                            torch.randn(1, 4, 1, 4, 4),
                            torch.randn(1, 8, 16),
                            torch.tensor([t]),
                        )
                self.assertEqual(model.cnt, expected)


class TestTpuCompileKwargs(unittest.TestCase):
    """What the TPU platform compiles, and with which arguments."""

    def test_tpu_compiles_static_shapes_with_platform_backend(self):
        compile_path = "sglang.multimodal_gen.runtime.utils.torch_compile"
        with (
            patch(
                f"{compile_path}.current_platform.is_out_of_tree",
                return_value=False,
            ),
            patch(f"{compile_path}.current_platform.is_tpu", return_value=True),
            patch(
                f"{compile_path}.current_platform.get_compile_backend",
                return_value="tpu",
            ) as get_compile_backend,
        ):
            compile_kwargs = build_torch_compile_kwargs(
                mode="max-autotune-no-cudagraphs",
                module=torch.nn.Module(),
                enable_inductor_compute_comm_overlap=True,
            )

        self.assertEqual(
            compile_kwargs, {"backend": "tpu", "dynamic": False, "fullgraph": False}
        )
        get_compile_backend.assert_called_once_with("max-autotune-no-cudagraphs")

    def test_vae_decode_stays_eager_where_the_platform_cannot_compile_it(self):
        stage = DecodingStage.__new__(DecodingStage)
        decode = MagicMock()
        compiled_callable = MagicMock(target_id=None, compiled_module=None)
        module_path = "sglang.multimodal_gen.runtime.pipelines_core.stages.decoding"

        with patch(
            f"{module_path}.current_platform.supports_vae_torch_compile",
            return_value=False,
        ):
            result = stage._get_vae_decode_fn(
                torch.nn.Module(),
                SimpleNamespace(enable_torch_compile=True),
                decode_fn=decode,
                compiled_callable=compiled_callable,
            )

        self.assertIs(result, decode)
        compiled_callable.get_or_compile.assert_not_called()


if __name__ == "__main__":
    unittest.main()
