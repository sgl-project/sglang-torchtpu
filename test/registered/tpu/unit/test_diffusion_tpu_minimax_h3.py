# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's MiniMax H3 patches (sglang_tpu.multimodal_gen.minimax_h3)."""

import contextlib
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    packed_sequence,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_ROW_WIDTH,
    MINIMAX_H3_VIDEO_ROW_WIDTH,
    MiniMaxH3DenoiseBranch,
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
from sglang_tpu.multimodal_gen import minimax_h3, worker

# The TPU lane installs the diffusion_tpu extras (cache-dit) the H3 DiT imports.
register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)


def _init_parallel_state():
    """Distributed init without the TPU worker's process-wide runtime patches."""
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    with patch.object(worker, "install_runtime_patches"):
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


@contextlib.contextmanager
def _installed():
    """Run install() with every patched name restored afterwards."""
    with contextlib.ExitStack() as stack:
        for target, name in (
            (packed_sequence, "MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT"),
            (comfyui_step, "MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT"),
            (dit, "_minimax_h3_attention_core_bcg"),
            (dit.MiniMaxH3DiTModel, "_compile_conditions"),
            (dit.MiniMaxH3DiTBlock, "forward"),
        ):
            stack.enter_context(patch.object(target, name, getattr(target, name)))
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


def _tiny_arch():
    hidden = 256
    return MiniMaxH3DiTArchConfig(
        num_layers=1,
        token_refiner_num_layers=1,
        hidden_size=hidden,
        num_attention_heads=2,
        attention_head_dim=128,
        ffn_hidden_size=512,
        text_dim=64,
        time_embed_hidden_size=64,
        time_embed_dim=64,
        adaln_out_features=18 * hidden,
        final_adaln_out_features=2 * hidden,
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
    def test_pads_to_a_multiple_of_512_and_keeps_layout(self):
        # 68 latent rows: used 128, 198 and 318 share the first bucket, 512 is
        # aligned already, 668 takes the second.
        buckets = {60: 512, 130: 512, 250: 512, 444: 512, 600: 1024}
        reference = {t: _packed(t) for t in buckets}
        # The core builders' own padding: the buckets below come from install().
        self.assertNotEqual(reference[600]["seq_len"], 1024)
        ref2va_args = dict(
            text_len=60,
            latent_t=3,
            latent_h=8,
            latent_w=8,
            audio_t=10,
            ref_blocks=[{"kind": "image", "latent_h": 8, "latent_w": 8}],
        )
        ref2va = packed_sequence.minimax_h3_packed_sequence_ref2va_blocks(**ref2va_args)
        empty = torch.zeros(0, dtype=torch.long)
        comfyui_layout = {
            "seq_len": 600,
            "position_ids": torch.ones(600, 3),
            "segments": [(0, 600, "text")],
            "img_pos": empty,
            "img_update": empty.bool(),
            "audio_pos": empty,
            "audio_update": empty.bool(),
        }
        with _installed():
            layouts = {text_len: _packed(text_len) for text_len in reference}
            other_layouts = (
                (
                    packed_sequence.minimax_h3_packed_sequence_ref2va_blocks(
                        **ref2va_args
                    ),
                    int(ref2va["cu_seqlens"][1]),
                    512,
                ),
                (comfyui_step.comfyui_layout_to_packed(comfyui_layout), 600, 1024),
            )
        for packed, used, seq_len in other_layouts:
            self.assertEqual(packed["seq_len"], seq_len)
            self.assertEqual(packed["cu_seqlens"].tolist(), [0, used, seq_len])
            self.assertTrue((packed["img_position_ids"][used:] == 0).all())
            self.assertTrue((packed["token_tags"][used:] == -1).all())
        for text_len, packed in layouts.items():
            ref = reference[text_len]
            used = int(ref["cu_seqlens"][1])
            self.assertEqual(packed["seq_len"], buckets[text_len])
            self.assertEqual(
                packed["cu_seqlens"].tolist(), [0, used, buckets[text_len]]
            )
            torch.testing.assert_close(
                packed["img_position_ids"][:used], ref["img_position_ids"][:used]
            )
            self.assertTrue((packed["img_position_ids"][used:] == 0).all())
            self.assertTrue((packed["token_tags"][used:] == -1).all())
            for key in ("img_pos", "audio_pos", "text_pos", "update_mask"):
                torch.testing.assert_close(packed[key], ref[key])


class TestMiniMaxH3LiveLength(CustomTestCase):
    def test_blocks_read_the_live_length_only_from_the_attention(self):
        _init_parallel_state()
        with torch.device("meta"):
            model = dit.MiniMaxH3DiTModel(
                config=MiniMaxH3DiTConfig(), hf_config={}, quant_config=None
            )
        self.assertFalse(model._attention_reads_live_length())
        attn = model.blocks[0].attn
        attn._attention_impl = _DeviceLiveLengthAttention(attn.softmax_scale)
        self.assertTrue(model._attention_reads_live_length())
        # Cache-DiT swaps the block list for its wrapper while it runs.
        model.blocks = torch.nn.ModuleList([torch.nn.Identity()])
        self.assertFalse(model._attention_reads_live_length())

    def test_forward_gives_blocks_no_host_bounds_for_a_live_length_attention(self):
        # Host bounds or a live max_seqlen would key each compiled block on
        # the prompt length; attention without the live length keeps both.
        _init_parallel_state()
        torch.manual_seed(0)
        arch = _tiny_arch()
        model = dit.MiniMaxH3DiTModel(
            config=MiniMaxH3DiTConfig(arch_config=arch), hf_config={}, quant_config=None
        )
        for param in model.parameters():
            param.data.normal_(0, 0.02)
        attn = model.blocks[0].attn
        host_bounds_impl = SDPAImpl(
            num_heads=attn.num_heads,
            head_size=attn.head_dim,
            causal=False,
            softmax_scale=attn.softmax_scale,
        )
        # Resolved, as forward would: every attention (the refiner's too) has
        # an impl, so forward keeps the ones installed here.
        for module in model.modules():
            if isinstance(module, dit.MiniMaxH3Attention):
                module._attention_impl = host_bounds_impl
        model._resolved_attention_backend = AttentionBackendEnum.TORCH_SDPA
        packed = packed_sequence.minimax_h3_packed_sequence(
            text_len=20,
            latent_t=1,
            latent_h=4,
            latent_w=4,
            audio_t=2,
            include_keyframe_cond=False,
        )
        seq_len, cu = packed["seq_len"], packed["cu_seqlens"]
        kwargs = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=torch.randn(20, arch.text_dim).bfloat16(),
            token_tags=packed["token_tags"],
            device=torch.device("cpu"),
        ).forward_kwargs(
            video_rows=torch.randn(
                packed["img_pos"].numel(), MINIMAX_H3_VIDEO_ROW_WIDTH
            ),
            audio_rows=torch.randn(
                packed["audio_pos"].numel(), MINIMAX_H3_AUDIO_ROW_WIDTH
            ),
            step_timesteps=(torch.tensor([0.5]), torch.zeros(seq_len).long(), None),
        )
        for impl, cu_seqlens_host, max_seqlen in (
            (_DeviceLiveLengthAttention(attn.softmax_scale), None, seq_len),
            (host_bounds_impl, tuple(cu.tolist()), int(cu[1])),
        ):
            attn._attention_impl = impl
            with (
                patch.object(
                    dit.MiniMaxH3DiTBlock,
                    "forward",
                    autospec=True,
                    side_effect=dit.MiniMaxH3DiTBlock.forward,
                ) as block_forward,
                torch.no_grad(),
            ):
                model(**kwargs)
            self.assertEqual(block_forward.call_count, arch.num_layers)
            self.assertEqual(
                block_forward.call_args.kwargs["cu_seqlens_host"], cu_seqlens_host
            )
            self.assertEqual(block_forward.call_args.kwargs["max_seqlen"], max_seqlen)


class TestMiniMaxH3CompiledBlock(CustomTestCase):
    def test_block_compiles_once_across_live_lengths(self):
        _init_parallel_state()
        torch.manual_seed(0)
        seq_len = 128
        arch = _tiny_arch()
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
        x = torch.randn(seq_len, arch.hidden_size).bfloat16()

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


if __name__ == "__main__":
    unittest.main()
