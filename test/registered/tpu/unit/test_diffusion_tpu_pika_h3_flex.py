# SPDX-License-Identifier: Apache-2.0
"""Pika-H3-Flex on TPU workers: MiniMax-H3 DiT blocks with Pika's unmerged LoRA adapters."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.compiler.config as compiler_config

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    BaseLayerWithLoRA,
    replace_submodule,
    wrap_with_lora_layer,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_vision import (
    Qwen3VLVisionTransformer,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    denoise_loop,
    packed_sequence,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils import torch_compile
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import minimax_h3, worker
from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform

register_tpu_ci(
    est_time=60,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "diffusion"],
)

# The layers Pika's adapters target in each DiT block (PEFT to_q/to_k/to_v,
# to_out.0, ff.net.0.proj, ff.net.2 on the native fused layers).
ADAPTED = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
RANK = 8
ARCH = MiniMaxH3DiTArchConfig(
    num_layers=1,
    token_refiner_num_layers=1,
    hidden_size=256,
    num_attention_heads=2,
    attention_head_dim=128,
    ffn_hidden_size=512,
    text_dim=64,
    time_embed_hidden_size=64,
    time_embed_dim=64,
    adaln_out_features=18 * 256,
    final_adaln_out_features=2 * 256,
)


class _LiveLengthAttention:
    """Reads the live length on device like the TPU kernel; padding rows are 0."""

    trailing_padding_on_device = True

    def __init__(self, scale):
        self.scale = scale

    def forward_varlen(self, q, k, v, *, cu_seqlens, max_seqlen, cu_seqlens_host):
        live = torch.arange(q.shape[0], device=q.device) < cu_seqlens[1]
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * self.scale
        scores = scores.masked_fill(~live[None, None, :], float("-inf"))
        out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
        return (out * live[:, None, None]).to(q.dtype)


class _OneBlockDiT(torch.nn.Module):
    """One block under the DiT's compile conditions."""

    def __init__(self, block):
        super().__init__()
        self.blocks = torch.nn.ModuleList([block])
        self._compile_conditions = dit.MiniMaxH3DiTModel._compile_conditions
        self._compile_regions_spmd_safe = (
            dit.MiniMaxH3DiTModel._compile_regions_spmd_safe
        )

    def forward(self, *args, **kwargs):
        return self.blocks[0](*args, **kwargs)


def _block(adapted=True, seq_len=128):
    """A tiny DiT block (with Pika-shaped unmerged BF16 adapters) and its runner."""
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        with patch.object(worker, "install_runtime_patches"):
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    torch.manual_seed(0)
    block = dit.MiniMaxH3DiTBlock(ARCH, None, prefix="blocks.0")
    for param in block.parameters():
        param.data.normal_(0, 0.02)
    block.attn._attention_impl = _LiveLengthAttention(block.attn.softmax_scale)
    block.attn._attention_backend_enum = AttentionBackendEnum.TORCH_SDPA
    g = torch.Generator().manual_seed(1)
    for name in ADAPTED if adapted else ():
        layer = block.get_submodule(name)
        out_f, in_f = layer.weight.shape
        # Q/K/V stack like the PEFT mapping does.
        groups = 3 if name == "attn.qkv_proj" else 1
        a = torch.randn(groups, RANK, in_f, generator=g) * in_f**-0.5
        b = torch.randn(groups, out_f // groups, RANK, generator=g) * 0.05
        if groups == 1:
            a, b = a[0], b[0]
        wrapped = wrap_with_lora_layer(layer, lora_rank=RANK, lora_alpha=RANK)
        wrapped.set_lora_weights(a.bfloat16(), b.bfloat16(), merge_weights=False)
        replace_submodule(block, name, wrapped)

    adaln_input = torch.randn(2, ARCH.time_embed_dim, generator=g).bfloat16()
    combined = torch.randint(
        0, 2 * MINIMAX_H3_ADALN_MODALITY_NUM, (seq_len,), generator=g
    )
    freqs = torch.randn(seq_len, 6 * ARCH.rope_inv_freq_len, generator=g)
    cos_sin = dit._rope_cos_sin_cache(freqs, dtype=torch.bfloat16)
    x = torch.randn(seq_len, ARCH.hidden_size, generator=g).bfloat16()

    def run(module, used, device="cpu"):
        # Each request brings its own Req; the adapters read their scale there.
        request = SimpleNamespace(
            runtime_lora_scale=1.0, sampling_params=SimpleNamespace(quality="lossless")
        )
        cu = torch.tensor([0, used, seq_len], dtype=torch.int32)
        with torch.no_grad(), set_forward_context(0, None, forward_batch=request):
            return module(
                x.to(device),
                adaln_input=adaln_input.to(device),
                combined_indices=combined.to(device),
                rope_cache=(cos_sin.to(device), torch.arange(seq_len).to(device)),
                cu_seqlens=cu.to(device),
                cu_seqlens_host=None,
                max_seqlen=seq_len,
            )

    return block, run


@contextlib.contextmanager
def _tpu_compile_path():
    """The TPU worker's H3 patches and compile path, restored afterwards."""
    with contextlib.ExitStack() as stack:
        for target, name in (
            # The ComfyUI layout pads through packed_sequence's function.
            (packed_sequence, "MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT"),
            (dit, "_minimax_h3_attention_core_bcg"),
            (dit.MiniMaxH3DiTModel, "_compile_conditions"),
            (dit.MiniMaxH3DiTModel, "_embed"),
        ):
            stack.enter_context(patch.object(target, name, getattr(target, name)))
        minimax_h3.install()
        for module in (torch_compile, dit):
            stack.enter_context(
                patch.object(module, "current_platform", TpuOOTPlatform())
            )
        stack.enter_context(compiler_config.patch(cache_key_tag=""))
        yield


def _has_tpu():
    accelerator = torch.accelerator.current_accelerator()
    return accelerator is not None and accelerator.type == "tpu"


class TestPikaAdaptedBlock(CustomTestCase):
    def test_adapted_block_compiles_once_as_one_spmd_safe_graph(self):
        # The adapters' matmuls stay in the block's program: one graph for
        # every live length and request, marked spmd-safe like a stock block.
        base, _ = _block(adapted=False)
        block, run = _block()
        graphs = []

        def counting_backend(graph_module, example_inputs):
            graphs.append(graph_module)
            return graph_module.forward

        registry = torch_compile.CompiledModuleRegistry()
        compile_kwargs = dict(backend=counting_backend, fullgraph=True, dynamic=False)
        with _tpu_compile_path():
            eager = {used: run(block, used) for used in (100, 77)}
            self.assertFalse(torch.equal(eager[100], run(base, 100)))
            model = _OneBlockDiT(block)
            registry.compile_regions_once(model, compile_kwargs=compile_kwargs)
            for used in (100, 77, 100):
                torch.testing.assert_close(run(model, used), eager[used])
        self.assertEqual(len(graphs), 1)
        ops = [n for n in graphs[0].graph.nodes if n.op == "call_function"]
        self.assertTrue(ops)
        for node in ops:
            self.assertEqual(node.meta.get("custom"), {"spmd_safe": True}, node)
        adapted = [m for m in block.modules() if isinstance(m, BaseLayerWithLoRA)]
        self.assertEqual(len(adapted), len(ADAPTED))
        self.assertTrue(all(m.lora_active and not m.merged for m in adapted))

    @unittest.skipUnless(_has_tpu(), "needs a TPU")
    def test_compiled_adapted_block_on_tpu_matches_the_cpu_path(self):
        block, run = _block(seq_len=512)
        ref = run(block, 400)
        device = torch.device("tpu", 0)
        compiled = torch.compile(
            block.to(device), backend="tpu", dynamic=False, fullgraph=True
        )
        out = run(compiled, 400, device).cpu()
        self.assertTrue(torch.equal(out, run(compiled, 400, device).cpu()))
        err = (out.float() - ref.float()).norm() / ref.float().norm()
        self.assertLess(err.item(), 1e-2)


class TestPromptLengthShapes(CustomTestCase):
    def test_embed_shapes_do_not_follow_the_prompt_and_shards_match(self):
        # Each SP rank's share of video rows moves with the prompt length; the
        # TPU embedding projects all of them, so only the text copy moves.
        _block(adapted=False)  # parallel state
        torch.manual_seed(0)
        model = dit.MiniMaxH3DiTModel(
            config=MiniMaxH3DiTConfig(arch_config=ARCH), hf_config={}, quant_config=None
        )
        for param in model.parameters():
            param.data.normal_(0, 0.02)
        stock = dit.MiniMaxH3DiTModel._embed
        rows_projected = []
        model.video_patch_proj.register_forward_pre_hook(
            lambda module, args: rows_projected.append(args[0].shape[0])
        )
        world = 2
        with (
            _tpu_compile_path(),
            patch.object(minimax_h3, "_EMBED_CHUNK_ROWS", 40),  # several chunks
            torch.no_grad(),
        ):
            for text_len in (20, 37):
                packed = packed_sequence.minimax_h3_packed_sequence(
                    text_len=text_len,
                    latent_t=3,
                    latent_h=8,
                    latent_w=8,
                    audio_t=10,
                    include_keyframe_cond=False,
                )
                seq_len = int(packed["seq_len"])
                x = torch.randn(1, seq_len, 96)
                audio_x = torch.randn(1, seq_len, 32)
                text = torch.randn(text_len, ARCH.hidden_size).bfloat16()
                for rank in range(world):
                    pos = {k: packed[k] for k in ("text_pos", "img_pos", "audio_pos")}
                    kwargs = dict(
                        x=x,
                        audio_x=audio_x,
                        text_embeddings_selected=text,
                        unique_timesteps=torch.tensor([0.25, 0.5]),
                        refiner_cu_seqlens=torch.tensor([0, text_len]),
                        refiner_max_seqlen=text_len,
                        row_start=rank * seq_len // world,
                        row_stop=(rank + 1) * seq_len // world,
                        device=torch.device("cpu"),
                        refined_prompt_embeds_length=text_len,
                        local_embedding_layout=denoise_loop._build_local_embedding_layout(
                            seq_len=seq_len,
                            world_size=world,
                            rank=rank,
                            device=torch.device("cpu"),
                            **pos,
                        ),
                        **pos,
                    )
                    rows_projected.clear()
                    out, t_emb = model._embed(**kwargs)
                    self.assertEqual(sum(rows_projected), packed["img_pos"].numel())
                    self.assertEqual(rows_projected[0], 40)
                    self.assertEqual(out.storage_offset(), 0)
                    ref, ref_t_emb = stock(model, **kwargs)
                    torch.testing.assert_close(out, ref, rtol=0, atol=0)
                    torch.testing.assert_close(t_emb, ref_t_emb, rtol=0, atol=0)


class TestRef2vaVisionGrids(CustomTestCase):
    @unittest.skipUnless(_has_tpu(), "needs a TPU")
    def test_vision_tower_takes_image_and_video_grids_on_tpu(self):
        # ref2va encodes two image references and a video in one call: three
        # grid rows, one of them temporal. The packed bounds come from the host.
        config = SimpleNamespace(
            hidden_size=16,
            intermediate_size=24,
            hidden_act="gelu_pytorch_tanh",
            num_heads=2,
            depth=0,
            patch_size=2,
            temporal_patch_size=1,
            in_channels=3,
            num_position_embeddings=16,
            spatial_merge_size=2,
            out_hidden_size=12,
            deepstack_visual_indexes=[],
        )
        torch.manual_seed(0)
        with get_parallel().override(tp_size=1, tp_rank=0):
            model = Qwen3VLVisionTransformer(config)
        model.merger = torch.nn.Identity()  # srt's TP linear; not under test
        for param in model.parameters():
            param.data.normal_(0, 0.1)
        grid = torch.tensor([[1, 4, 4], [1, 4, 8], [3, 4, 4]])
        x = torch.randn(int(grid.prod(-1).sum()), 12)
        with torch.no_grad():
            ref = model(x, grid_thw=grid).last_hidden_state
            out = model.to("tpu:0")(x, grid_thw=grid).last_hidden_state.cpu()
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
