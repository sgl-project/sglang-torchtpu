import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.graph_runner.moonvit_tpu_compile_runner import (
    MoonVitTpuCompilePolicy,
    MoonVitTpuCompileRunner,
)

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)


class _Block(nn.Module):
    """MoonViT block signature; records what the compiled boundary hands it."""

    def __init__(self):
        super().__init__()
        self.attn = SimpleNamespace(
            qkv_backend=SimpleNamespace(supports_segment_ids=True)
        )
        self.hidden_dim = 4
        self.segment_ids = []

    def forward(self, x, cu_seqlens, max_seqlen, rope_freqs_cis=None, segment_ids=None):
        self.segment_ids.append(segment_ids.clone())
        out = x + 1
        return out[None] if out.ndim == 2 else out


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block()])
        self.final_layernorm = nn.Identity()


def _policy(buckets):
    encoder = _Encoder()
    runner = MoonVitTpuCompileRunner(encoder)
    encoder.graph_runners = runner
    policy = MoonVitTpuCompilePolicy(encoder, owner=encoder)
    policy._patch_buckets = list(buckets)
    # Stand in for install(): the compiled fn is the runner's forward with the
    # call shapes recorded, and the eager fn is the same bound method.
    eager = runner.forward
    policy.eager_fn = MagicMock(side_effect=eager)
    policy.compiled_fn = MagicMock(side_effect=eager)
    policy._install_run_adapter(runner)
    return policy, runner, encoder.blocks[0]


class TestMoonVitTpuCompileRunner(unittest.TestCase):
    def test_env_override_must_be_perfect_squares(self):
        policy, _, _ = _policy([])
        policy._patch_buckets = None
        with envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.override("64,16"):
            self.assertEqual(policy.compute_patch_buckets(), [16, 64])
        policy._patch_buckets = None
        with envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.override("15"):
            with self.assertRaises(ValueError):
                policy.compute_patch_buckets()

    def test_single_image_pads_to_its_bucket(self):
        policy, runner, block = _policy([16, 64])
        x = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
        rope = torch.zeros(3, 2)
        out = runner.run(x, rope, rope, torch.tensor([0, 3], dtype=torch.int32))

        self.assertEqual(policy.compiled_fn.call_count, 1)
        self.assertEqual(policy.compiled_fn.call_args.args[0].shape[0], 16)
        # padding is a distinct trailing segment
        self.assertEqual(block.segment_ids[0].tolist(), [[0] * 3 + [1] * 13])
        self.assertEqual(tuple(out.shape), (1, 3, 4))
        torch.testing.assert_close(out[0], x + 1)

    def test_packed_images_run_one_compiled_call_each(self):
        policy, runner, block = _policy([16, 64])
        x = torch.arange(23 * 4, dtype=torch.float32).reshape(23, 4)
        rope = torch.zeros(23, 2)
        out = runner.run(x, rope, rope, torch.tensor([0, 3, 23], dtype=torch.int32))

        # 3 patches -> bucket 16, 20 patches -> bucket 64: never the packed 23.
        self.assertEqual(
            [c.args[0].shape[0] for c in policy.compiled_fn.call_args_list], [16, 64]
        )
        self.assertEqual(block.segment_ids[1].tolist(), [[0] * 20 + [1] * 44])
        self.assertEqual(tuple(out.shape), (1, 23, 4))
        torch.testing.assert_close(out[0], x + 1)
        policy.eager_fn.assert_not_called()

    def test_packed_slices_never_alias_the_packed_buffer(self):
        # An exact bucket hit needs no pad copy; the slice must still be a fresh
        # buffer, since torch_tpu feeds a view's whole backing buffer to the
        # compiled executable.
        policy, runner, _ = _policy([16, 64])
        x = torch.arange(19 * 4, dtype=torch.float32).reshape(19, 4)
        rope = torch.zeros(19, 2)
        out = runner.run(x, rope, rope, torch.tensor([0, 3, 19], dtype=torch.int32))

        second = policy.compiled_fn.call_args_list[1].args
        self.assertEqual(second[0].shape[0], 16)  # exact hit, unpadded
        for arg, src in zip(second[:3], (x, rope, rope)):
            self.assertNotEqual(
                arg.untyped_storage().data_ptr(), src.untyped_storage().data_ptr()
            )
        torch.testing.assert_close(out[0], x + 1)

    def test_image_past_the_ladder_runs_eagerly(self):
        policy, runner, block = _policy([16])
        x = torch.arange(20 * 4, dtype=torch.float32).reshape(20, 4)
        rope = torch.zeros(20, 2)
        out = runner.run(x, rope, rope, torch.tensor([0, 20], dtype=torch.int32))

        policy.compiled_fn.assert_not_called()
        self.assertEqual(policy.eager_fn.call_count, 1)
        # eager gets the real length, one segment, no padding
        self.assertEqual(block.segment_ids[0].tolist(), [[0] * 20])
        self.assertEqual(tuple(out.shape), (1, 20, 4))
        torch.testing.assert_close(out[0], x + 1)

    def test_packed_mixes_compiled_and_eager_per_image(self):
        policy, runner, _ = _policy([16])
        x = torch.arange(25 * 4, dtype=torch.float32).reshape(25, 4)
        rope = torch.zeros(25, 2)
        out = runner.run(x, rope, rope, torch.tensor([0, 5, 25], dtype=torch.int32))

        self.assertEqual(policy.compiled_fn.call_count, 1)  # 5 -> bucket 16
        self.assertEqual(policy.eager_fn.call_count, 1)  # 20 > 16 -> eager
        torch.testing.assert_close(out[0], x + 1)


if __name__ == "__main__":
    unittest.main()
