# SPDX-License-Identifier: Apache-2.0
"""CPU tests for sglang_tpu's MiniMax H3 patches (sglang_tpu.multimodal_gen.minimax_h3)."""

import os
import sys
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
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    packed_sequence,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)

# The CPU gate installs python/ only; an installed sglang_tpu still wins.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from sglang_tpu.multimodal_gen import minimax_h3  # noqa: E402


def _installed():
    """Run install() with every patched name restored afterwards."""
    return patch.multiple(
        packed_sequence,
        minimax_h3_packed_sequence=packed_sequence.minimax_h3_packed_sequence,
        minimax_h3_packed_sequence_ref2va_blocks=(
            packed_sequence.minimax_h3_packed_sequence_ref2va_blocks
        ),
    ), patch.object(
        dit, "_minimax_h3_attention_core_bcg", dit._minimax_h3_attention_core_bcg
    )


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
        a, b = _installed()
        with a, b:
            minimax_h3.install()
            minimax_h3.install()  # idempotent
            layouts = {text_len: _packed(text_len) for text_len in reference}
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
            maybe_init_distributed_environment_and_model_parallel(
                tp_size=1, sp_size=1
            )
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

        a, b = _installed()
        with a, b, torch.no_grad():
            minimax_h3.install()
            compiled = torch.compile(
                block, backend=counting_backend, fullgraph=True, dynamic=False
            )
            for used in (100, 77, 128):
                torch.testing.assert_close(run(compiled, used), run(block, used))
        self.assertEqual(len(graphs), 1)


if __name__ == "__main__":
    unittest.main()
