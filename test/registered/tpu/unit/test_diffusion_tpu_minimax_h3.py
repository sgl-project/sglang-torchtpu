# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's MiniMax H3 patches (sglang_tpu.multimodal_gen.minimax_h3)."""

import contextlib
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers import usp
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    packed_sequence,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
    comfyui_step,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import minimax_h3

# The TPU lane installs the diffusion_tpu extras (cache-dit) the H3 DiT imports.
register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

_BUILDERS = ("minimax_h3_packed_sequence", "minimax_h3_packed_sequence_ref2va_blocks")


@contextlib.contextmanager
def _installed():
    """Run install() with every patched name restored afterwards."""
    with contextlib.ExitStack() as stack:
        for module, names in (
            (packed_sequence, _BUILDERS),
            (comfyui_step, _BUILDERS + ("comfyui_layout_to_packed",)),
            (dit, ("_minimax_h3_attention_core_bcg",)),
            (usp, ("_usp_all_to_all_single",)),
        ):
            stack.enter_context(
                patch.multiple(module, **{n: getattr(module, n) for n in names})
            )
        stack.enter_context(
            patch.object(
                dit.MiniMaxH3DiTModel,
                "_compile_conditions",
                dit.MiniMaxH3DiTModel._compile_conditions,
            )
        )
        stack.enter_context(
            patch.object(
                dit.MiniMaxH3DiTBlock, "forward", dit.MiniMaxH3DiTBlock.forward
            )
        )
        minimax_h3.install()
        minimax_h3.install()  # idempotent
        yield


def _packed(text_len):
    return packed_sequence.minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=3,
        latent_h=8,
        latent_w=8,
        audio_t=10,
        include_keyframe_cond=False,
    )


class _DeviceLiveLengthAttention:
    """Reads the live length from cu_seqlens on device, like the TPU kernel:
    padding keys are masked and padding rows output zeros."""

    trailing_padding_on_device = True

    def __init__(self, scale):
        self.scale = scale

    def forward_varlen(self, q, k, v, *, cu_seqlens, max_seqlen, cu_seqlens_host):
        assert cu_seqlens_host is None
        live = torch.arange(q.shape[0], device=q.device) < cu_seqlens[1]
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * self.scale
        scores = scores.masked_fill(~live[None, None, :], float("-inf"))
        out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
        return (out * live[:, None, None]).to(q.dtype)


class TestMiniMaxH3PackedSequence(CustomTestCase):
    def test_pads_to_512_and_keeps_layout(self):
        reference = {text_len: _packed(text_len) for text_len in (60, 130, 250)}
        with _installed():
            layouts = {text_len: _packed(text_len) for text_len in reference}
            self.assertIs(
                comfyui_step.minimax_h3_packed_sequence,
                packed_sequence.minimax_h3_packed_sequence,
            )
        self.assertEqual({p["seq_len"] for p in layouts.values()}, {512})
        for text_len, packed in layouts.items():
            ref = reference[text_len]
            used = int(ref["cu_seqlens"][1])
            self.assertEqual(packed["cu_seqlens"].tolist(), [0, used, 512])
            torch.testing.assert_close(
                packed["img_position_ids"][: ref["seq_len"]], ref["img_position_ids"]
            )
            self.assertTrue((packed["img_position_ids"][ref["seq_len"] :] == 0).all())
            self.assertTrue((packed["token_tags"][used:] == -1).all())
            for key in ("img_pos", "audio_pos", "text_pos", "update_mask"):
                torch.testing.assert_close(packed[key], ref[key])
        # Restored after the test.
        self.assertEqual(_packed(60)["seq_len"], reference[60]["seq_len"])


class TestMiniMaxH3CompiledBlock(CustomTestCase):
    def test_block_compiles_once_across_live_lengths(self):
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
        torch.manual_seed(0)
        hidden, seq_len = 256, 128
        arch = MiniMaxH3DiTArchConfig(
            hidden_size=hidden,
            num_attention_heads=2,
            attention_head_dim=128,
            ffn_hidden_size=512,
            time_embed_dim=64,
            adaln_out_features=18 * hidden,
        )
        block = dit.MiniMaxH3DiTBlock(arch, None, prefix="blocks.0")
        for param in block.parameters():
            param.data.normal_(0, 0.02)
        attn = block.attn
        attn._attention_impl = _DeviceLiveLengthAttention(attn.softmax_scale)
        attn._attention_backend_enum = AttentionBackendEnum.TORCH_SDPA

        num_timesteps = 2
        adaln_input = torch.randn(num_timesteps, arch.time_embed_dim).bfloat16()
        combined = torch.randint(
            0, num_timesteps * MINIMAX_H3_ADALN_MODALITY_NUM, (seq_len,)
        )
        freqs = torch.randn(seq_len, arch.rope_inv_freq_len * 6)
        rope_cache = (
            dit._rope_cos_sin_cache(freqs, dtype=torch.bfloat16),
            torch.arange(seq_len),
        )
        x = torch.randn(seq_len, hidden).bfloat16()

        def run(module, used):
            return module(
                x.clone(),
                adaln_input=adaln_input,
                combined_indices=combined,
                rope_cache=rope_cache,
                cu_seqlens=torch.tensor([0, used, seq_len], dtype=torch.int32),
                cu_seqlens_host=None,
                max_seqlen=seq_len,
            )

        graphs = []

        def counting_backend(graph_module, example_inputs):
            graphs.append(graph_module)
            return graph_module.forward

        with _installed(), torch.no_grad():
            compiled = torch.compile(
                block, backend=counting_backend, fullgraph=True, dynamic=False
            )
            for used in (100, 77, 128):
                torch.testing.assert_close(run(compiled, used), run(block, used))
        self.assertEqual(len(graphs), 1)

    def test_refiner_blocks_stay_eager(self):
        with _installed():
            (condition,) = dit.MiniMaxH3DiTModel._compile_conditions
            block = torch.nn.Linear(1, 1)
            self.assertTrue(condition("blocks.7", block))
            self.assertFalse(condition("token_refiner.blocks.0", block))


def _all_to_all_worker(rank, world, init_file, results):
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.layers import usp

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    try:
        group = type("SpGroup", (), {"ulysses_group": dist.group.WORLD})()
        with patch.object(usp, "get_sp_group", return_value=group):
            lane_dense = minimax_h3._lane_dense_all_to_all_single(
                usp._usp_all_to_all_single
            )
            # The H3 packed-QKV and output exchange shapes, scaled down.
            for shape in ((world, 16, 3, 384), (world, 16, 3, 128), (6, 5)):
                torch.manual_seed(rank)
                x = torch.randn(shape).bfloat16()
                flat = usp._usp_all_to_all_single(x.clone())
                dense = lane_dense(x.clone())
                staged = lane_dense(x.clone(), role="usp_test")
                results[(rank, shape)] = torch.equal(flat, dense) and torch.equal(
                    flat, staged
                )
    finally:
        dist.destroy_process_group()


class TestLaneDenseAllToAll(CustomTestCase):
    def test_matches_the_flat_exchange(self):
        import torch.multiprocessing as mp

        world = 2
        with tempfile.TemporaryDirectory() as tmp, mp.Manager() as manager:
            results = manager.dict()
            mp.spawn(
                _all_to_all_worker,
                args=(world, os.path.join(tmp, "init"), results),
                nprocs=world,
            )
            self.assertEqual(len(results), 3 * world)
            self.assertTrue(all(results.values()), dict(results))


if __name__ == "__main__":
    unittest.main()
