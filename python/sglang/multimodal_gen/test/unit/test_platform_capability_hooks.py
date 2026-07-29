import contextlib
import unittest
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime.platforms.interface import Platform


class TestCapabilityHookDefaults(unittest.TestCase):
    """The base-class defaults must preserve pre-hook behavior on GPU."""

    def test_boolean_capability_defaults(self):
        self.assertTrue(Platform.supports_op_level_torch_compile())
        self.assertTrue(Platform.supports_p2p_communication())
        self.assertTrue(Platform.supports_vae_decode_torch_compile())
        self.assertFalse(Platform.use_functional_seq_all_to_all())

    def test_context_hooks_default_to_noop(self):
        with Platform.vae_decode_context(pipeline=None):
            pass
        with Platform.denoise_loop_context():
            pass
        self.assertIsNone(Platform.on_denoise_step_end())

    def test_wrap_forward_for_compile_is_identity(self):
        def forward():
            return 7

        self.assertIs(Platform.wrap_forward_for_compile(forward), forward)

    def test_aux_profiler_and_spawn_env_defaults(self):
        self.assertIsNone(Platform.create_aux_profiler("/tmp/logs"))
        self.assertIsNone(Platform.prepare_server_environment(world_size=8))
        self.assertEqual(
            Platform.worker_spawn_env(rank=0, world_size=8, master_port=1234), {}
        )


class TestCapabilityHookConsumers(unittest.TestCase):
    def test_spatial_shard_decode_disabled_without_p2p(self):
        from sglang.multimodal_gen.runtime import platforms
        from sglang.multimodal_gen.runtime.models.vaes import common

        platform = MagicMock()
        platform.supports_p2p_communication.return_value = False
        with patch.object(platforms, "current_platform", platform):
            self.assertFalse(
                common.should_run_spatial_shard_parallel_decode(
                    config=MagicMock(), z=MagicMock()
                )
            )

    def test_vae_decode_fn_not_compiled_without_capability(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages import decoding

        stage = object.__new__(decoding.DecodingStage)
        vae = MagicMock()
        server_args = MagicMock()
        server_args.enable_torch_compile = True
        platform = MagicMock()
        platform.supports_vae_decode_torch_compile.return_value = False
        with patch.object(decoding, "current_platform", platform):
            # nn.Module isinstance check passes for MagicMock spec-less objects
            # only via the capability early-return; assert we got the raw fn.
            self.assertIs(
                stage._get_vae_decode_fn(vae, server_args), vae.decode
            )

    def test_distributed_attention_uses_functional_path_when_selected(self):
        from sglang.multimodal_gen.runtime.layers.attention import turbo_layer

        attention = object.__new__(turbo_layer.DistributedAttention)
        attention.pg = MagicMock()
        attention.stream = None
        attention.local_attn = MagicMock(return_value="attn_out")

        platform = MagicMock()
        platform.use_functional_seq_all_to_all.return_value = True
        functional = MagicMock(side_effect=lambda tensor, *_args: tensor)
        with (
            patch.object(turbo_layer, "current_platform", platform),
            patch.object(turbo_layer, "_functional_seq_all_to_all", functional),
            patch.object(turbo_layer.dist, "get_world_size", return_value=2),
        ):
            result = turbo_layer.DistributedAttention.forward(
                attention, "q", "k", "v", ctx_attn_metadata=None
            )
        self.assertEqual(result, "attn_out")
        # q, k, v scattered + one gather of the attention output.
        self.assertEqual(functional.call_count, 4)


if __name__ == "__main__":
    unittest.main()
