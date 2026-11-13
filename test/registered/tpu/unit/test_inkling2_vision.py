"""The bucketed TPU Inkling2 vision encoder agrees with the reference encoder.

sglang.tml.vision.Inkling2VisionEncoder (the CUDA reference, run here on CPU) is
the oracle: same random weights, two images of different grids in one bucket,
checked eagerly and through the TPU compile backend. Image isolation is checked
by encoding each image alone and comparing with the batched rows.

Needs a TPU with Pallas; skipped otherwise.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

DEVICE = "tpu"
VISION_CONFIG = {
    "temporal_patch_size": 2,
    "native_patch_size": 20,
    "adapter_merge_size": 2,
    "projection_hidden_dim": 4096,
    "projector_bias": False,
    "hidden_size": 1024,
    "num_hidden_layers": 3,  # the checkpoint has 23; the layer math is identical
    "num_attention_heads": 16,
    "mlp_ratio": 4.0,
    "posemb_grid_size": 32,
    "layer_norm_eps": 1e-05,
}
TEXT_HIDDEN = 4608
GRID = [(2, 3), (1, 1)]  # merged (h, w) per image


def _skip_if_no_tpu() -> bool:
    return importlib.util.find_spec("torch_tpu") is None


def _copy_reference_weights(reference, encoder) -> None:
    """Checkpoint-named reference tensors into the TPU module (fused qkv, proj)."""
    params = dict(encoder.named_parameters())
    for name, tensor in reference.state_dict().items():
        for shard in ("q", "k", "v"):
            if f".self_attn.{shard}_proj." in name:
                target = name.replace(f".{shard}_proj.", ".qkv_proj.")
                params[target].weight_loader(params[target], tensor.to(DEVICE), shard)
                break
        else:
            name = name.replace(".self_attn.out_proj.", ".self_attn.proj.")
            param = params[name]
            loader = getattr(param, "weight_loader", None)
            if loader is None:
                param.data.copy_(tensor.to(param.dtype))
            else:
                loader(param, tensor.to(DEVICE))


@unittest.skipIf(_skip_if_no_tpu(), "requires torch_tpu")
class TestInkling2Vision(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import os
        import socket

        import sglang.tml.vision as reference_vision
        from sglang.srt.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.test.test_utils import publish_build_topology
        from sglang_tpu.models.inkling2_vision import Inkling2VisionEncoderTpu

        # One-rank gloo world so the row-parallel out projection can run.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
        publish_build_topology(tp_size=1)
        initialize_model_parallel(backend="gloo")
        torch.manual_seed(0)
        # The reference compiles its renormalization with Inductor; keep the CPU
        # oracle on eager ops.
        reference_vision._renormalize = lambda x, mean, std: (
            2.0 * (x * std.to(x.dtype) + mean.to(x.dtype)) - 1.0
        )
        cls.reference = reference_vision.Inkling2VisionEncoder(
            VISION_CONFIG, TEXT_HIDDEN
        )
        for param in cls.reference.parameters():
            param.data.normal_(0, 0.02)
        for name, param in cls.reference.named_parameters():
            if name.endswith(("ls_1.gamma", "ls_2.gamma")):
                param.data.fill_(0.5)
            elif name.endswith(("ln_1.weight", "ln_2.weight", "ln_pre.weight")):
                param.data.fill_(1.0)
            elif (
                ".self_attn.q_proj.weight" in name or ".self_attn.k_proj.weight" in name
            ):
                # Peaked attention, so a wrong RoPE convention shows up in the output.
                param.data.mul_(10.0)
        # Build on device in bf16 as the model loader does; Module.to() would
        # replace the Parameters and lose their weight_loader.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(DEVICE):
                cls.encoder = Inkling2VisionEncoderTpu(VISION_CONFIG, TEXT_HIDDEN)
        finally:
            torch.set_default_dtype(default_dtype)
        _copy_reference_weights(cls.reference, cls.encoder)
        num_patches = sum(h * w for h, w in GRID)
        cls.patches = (
            (
                torch.rand((num_patches, 1, 40, 40, 3))
                .mul(2.5)
                .sub(1.8)
                .to(torch.bfloat16)
            )
            .expand(-1, 2, -1, -1, -1)
            .contiguous()
        )
        cls.expected = cls.reference(
            cls.patches, torch.tensor([[1, h, w] for h, w in GRID], dtype=torch.int64)
        ).float()

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()

    def _encode(self, grid, patches, forward):
        from sglang_tpu.models.inkling2_vision import prepare_vision_input

        enc = self.encoder
        inputs, count = prepare_vision_input(
            patches,
            grid,
            patch_size=enc.patch_size,
            merge=enc.merge_size,
            head_dim=enc.head_dim,
            posemb_grid=enc.posemb_grid_size,
        )
        with torch.no_grad():
            out = forward(*(t.to(DEVICE) for t in inputs))[:count]
        return out.to("cpu", torch.float32)

    def _assert_matches(self, got, want):
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        # bf16 rounding with peaked attention; a wrong RoPE or segment map is ~1.2 off.
        torch.testing.assert_close(got, want, atol=0.15, rtol=5e-2)
        cosine = torch.nn.functional.cosine_similarity(got, want, dim=-1)
        self.assertGreater(float(cosine.min()), 0.99)

    def test_batched_images_match_reference_eagerly(self):
        self._assert_matches(
            self._encode(GRID, self.patches, self.encoder.forward), self.expected
        )

    def test_each_image_alone_matches_its_batched_rows(self):
        """Attention segments keep images apart: an image's rows do not depend on
        its neighbours in the bucket."""
        offset = 0
        for h, w in GRID:
            n = h * w
            alone = self._encode(
                [(h, w)], self.patches[offset : offset + n], self.encoder.forward
            )
            self._assert_matches(alone, self.expected[offset : offset + n])
            offset += n

    def test_compiled_bucket_matches_reference(self):
        from sglang_tpu.patches import _register_tpu_compile_backend

        _register_tpu_compile_backend()
        compiled = torch.compile(
            torch.no_grad()(self.encoder.forward),
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        self._assert_matches(self._encode(GRID, self.patches, compiled), self.expected)


if __name__ == "__main__":
    unittest.main()
