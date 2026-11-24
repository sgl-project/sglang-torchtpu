# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the MiniMax-H3 megakernel integration (sglang_tpu.multimodal_gen.h3_megakernel).

The hook, its refusals, the padding seam and the per-step input builders run
in torch everywhere. Tests that need JAX, torch_tpu's Pallas adapter, the FL2VA
checkpoint (a directory in ``MINIMAX_H3_CHECKPOINT``) or a libtpu compile-only
topology skip without them:

- the packer against ``reference.pack_weights`` on two real blocks, bitwise;
- the AdaLN tables against each real block's ``adaln_proj``, bitwise;
- the RoPE tables: applied as the kernel does, sglang's eager RoPE bitwise;
- the host copy of the proven fixed-max layers against
  ``kernel.fixed_max_layers`` (FL2VA: 41 of 50);
- the op's JAX function in Pallas interpret mode on 4 CPU devices with two real
  blocks against sglang's eager blocks on the same inputs, within the bf16
  floor (two processes: ``--real-stage1 DIR`` builds sglang's side and the
  hook's operands, ``--real-stage2 DIR`` runs the kernel);
- one program for every prompt length of a 21-prompt benchmark set at
  1344x768x124.
"""

import contextlib
import importlib.util
import json
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Host-CPU tests: in the TPU image the diffusion platform and JAX would
# otherwise select the TPU.
os.environ.setdefault("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch


def _real_kernel_side(directory: str, devices: int = 4) -> dict:
    """Stage 2 of the real-block test: the op's JAX function on stage 1's
    operands, in Pallas interpret mode on ``devices`` CPU devices, against
    stage 1's sglang outputs. It imports only JAX and sglang_tpu's kernel
    package, so it also runs where sglang does not import."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
    if root not in sys.path:
        sys.path.append(root)
    import dataclasses

    import jax
    import ml_dtypes
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.multimodal_gen.h3_megakernel import op
    from sglang_tpu.multimodal_gen.h3_megakernel import pack as kernel_pack
    from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref

    with open(os.path.join(directory, "meta.json")) as f:
        meta = json.load(f)

    def load(name):
        value = np.load(os.path.join(directory, f"{name}.npy"), mmap_mode="r")
        return value.view(ml_dtypes.bfloat16) if value.dtype == np.uint16 else value

    mesh = Mesh(np.asarray(jax.devices()[:devices]), ("sp",))
    names = ("x", "onehot", "cos", "sin", "tables", "used", *kernel_pack.WEIGHT_ORDER)
    args = [
        jax.device_put(load(name), NamedSharding(mesh, P("sp") if i < 4 else P()))
        for i, name in enumerate(names)
    ]
    interpret = pltpu.InterpretParams(detect_races=False, dma_execution_mode="on_wait")
    # The serving tiling with 128-row phase A and C tiles: two a device here
    # (the kernel does not trace a device with a single phase A row tile, a
    # case no 1344x768x124 row count reaches).
    tiling = dataclasses.replace(op.STACK_TILING, qkv_rows=128, mlp_rows=128)
    fn = op.sharded_stack(
        mesh, ref.Config(layers=len(meta["layers"])), tiling, interpret
    )
    got = np.asarray(jax.jit(fn)(*args), np.float32)[: meta["used"]]
    truth = np.load(os.path.join(directory, "truth.npy"))
    served = np.load(os.path.join(directory, "served.npy"))

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    return {
        "finite": bool(np.isfinite(got).all()),
        "kernel": rel(got, truth),
        "served": rel(served, truth),
        "kernel_vs_served": rel(got, served),
        "max": float(np.abs(got - truth).max() / np.abs(truth).max()),
        "jax": jax.__version__,
        "devices": devices,
    }


if __name__ == "__main__" and sys.argv[1:2] == ["--real-stage2"]:
    print(json.dumps(_real_kernel_side(sys.argv[2], *map(int, sys.argv[3:4]))))
    sys.exit(0)

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (  # noqa: E402
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    SDPABackend,
)
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

# The real-block tests need MINIMAX_H3_CHECKPOINT and skip without it.
register_tpu_ci(
    est_time=150,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "diffusion"],
)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)

# sglang_tpu from the checkout where it is not installed.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from sglang_tpu.envs import envs as tpu_envs  # noqa: E402
from sglang_tpu.multimodal_gen.h3_megakernel import hook, pack  # noqa: E402

_HAS_JAX = importlib.util.find_spec("jax") is not None
_HAS_TORCH_TPU = importlib.util.find_spec("torch_tpu") is not None
_CKPT = os.environ.get("MINIMAX_H3_CHECKPOINT", "")
_TRANSFORMER = os.path.join(_CKPT, "transformer")
_HAS_CKPT = bool(_CKPT) and os.path.isfile(
    os.path.join(_TRANSFORMER, "model.safetensors.index.json")
)
_BF16 = torch.bfloat16
# Two real blocks: 35's softmax runs at a fixed max (its norms bound it), 36's
# online (FL2VA proves every layer but 36, 37 and 43-49).
_REAL_LAYERS = (35, 36)
_FL2VA_ONLINE = [36, 37, 43, 44, 45, 46, 47, 48, 49]
# Text tokens of a 21-prompt benchmark set at 1344x768x124: its 20 measured
# prompts and its warmup (FL2VA tokenizer, no special tokens): 36-47 tokens,
# 9 distinct counts.
_SPEC_TEXT_TOKENS = {
    "prompt-01": 47,
    "prompt-02": 40,
    "prompt-03": 40,
    "prompt-04": 39,
    "prompt-05": 43,
    "prompt-06": 40,
    "prompt-07": 45,
    "prompt-08": 41,
    "prompt-09": 39,
    "prompt-10": 37,
    "prompt-11": 39,
    "prompt-12": 39,
    "prompt-13": 38,
    "prompt-14": 40,
    "prompt-15": 36,
    "prompt-16": 38,
    "prompt-17": 40,
    "prompt-18": 40,
    "prompt-19": 39,
    "prompt-20": 38,
    "warmup": 39,
}
# 1344x768, 124 frames at 24 fps: 37 latent frames of 48x84, 207 audio frames.
_SPEC_SHAPE = dict(latent_t=37, latent_h=48, latent_w=84, audio_t=207)
# The static-structure option's row counts: (devices, rows a device,
# SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS), as op.STRUCTURED_SHAPES.
_STRUCTURED = (
    (8, 4992, None),
    (8, 4864, 4864),
    (8, 4800, 4800),
    (4, 9600, None),
    (2, 19456, None),
    (2, 19200, 19200),
)


def _init_parallel():
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _arch(heads: int = 2) -> MiniMaxH3DiTArchConfig:
    hidden = 256
    return MiniMaxH3DiTArchConfig(
        num_layers=2,
        token_refiner_num_layers=1,
        hidden_size=hidden,
        num_attention_heads=heads,
        attention_head_dim=128,
        ffn_hidden_size=512,
        text_dim=64,
        time_embed_hidden_size=64,
        time_embed_dim=64,
        adaln_out_features=18 * hidden,
        final_adaln_out_features=2 * hidden,
    )


def _randomize_block(block, arch, norm_scale: float = 1.0) -> None:
    """Projections ``1/sqrt(fan_in)``, AdaLN params O(0.1-1), q/k/v and
    gate/up at distinct scales and norm weights varying by dim, so a block
    changes its input by O(1) and a swap or a lane mixup shows."""
    for name, param in block.named_parameters():
        if name in ("attn.q_norm.weight", "attn.k_norm.weight"):
            param.data.uniform_(0.5 * norm_scale, 1.5 * norm_scale)
        elif name.startswith("norm"):
            param.data.uniform_(0.5, 1.5)
        elif name == "adaln_proj.linear.weight":
            param.data.normal_(0, 0.08)
        elif name == "adaln_proj.linear.bias":
            param.data.normal_(0, 0.3)
        elif param.dim() == 2:
            param.data.normal_(0, param.shape[1] ** -0.5)
        else:
            param.data.normal_(0, 0.1)
    heads, dim = arch.num_attention_heads, arch.attention_head_dim
    qkv = block.attn.qkv_proj.weight.data.view(3, heads, dim, -1)
    qkv[1] *= 1.3
    qkv[2] *= 0.7
    block.mlp.fc1.weight.data[: arch.ffn_hidden_size] *= 1.5


def _blocks(arch, layers: int, norm_scales=None, seed: int = 0):
    """Random sglang blocks; ``norm_scales[i]`` scales block i's q/k norms."""
    _init_parallel()
    torch.manual_seed(seed)
    blocks = []
    for index in range(layers):
        block = dit.MiniMaxH3DiTBlock(arch, None, prefix=f"blocks.{index}")
        block.requires_grad_(False)
        scale = 1.0 if norm_scales is None else norm_scales[index]
        _randomize_block(block, arch, scale)
        blocks.append(block)
    return blocks


def _request(arch, tokens: int, used: int, timesteps: int = 2, seed: int = 1):
    """One forward's block inputs, as served (bf16 AdaLN input and RoPE
    cache) and unrounded (``*_fp32``, for the fp32 floor)."""
    generator = torch.Generator().manual_seed(seed)
    rows = timesteps * MINIMAX_H3_ADALN_MODALITY_NUM
    freqs = torch.rand(tokens, arch.rope_inv_freq_len * 6, generator=generator) * 30
    adaln_input = torch.nn.functional.silu(
        torch.randn(timesteps, arch.time_embed_dim, generator=generator)
    )
    return {
        "x": torch.randn(tokens, arch.hidden_size, generator=generator).to(_BF16),
        "adaln_input": adaln_input.to(_BF16),
        "adaln_input_fp32": adaln_input,
        "combined": torch.randint(0, rows, (tokens,), generator=generator),
        "cos_sin_cache": dit._rope_cos_sin_cache(freqs, dtype=_BF16),
        "cos_sin_cache_fp32": dit._rope_cos_sin_cache(freqs, dtype=torch.float32),
        "cu_seqlens": torch.tensor([0, used, tokens], dtype=torch.int32),
    }


def _clone_block(block, arch, dtype=None):
    """A fresh copy of an sglang block (blocks hold process groups, which
    ``copy.deepcopy`` cannot copy)."""
    prefix = block.attn.prefix.rsplit(".attn", 1)[0]
    clone = dit.MiniMaxH3DiTBlock(arch, None, prefix=prefix).requires_grad_(False)
    clone.load_state_dict(block.state_dict())
    clone.attn._set_attention_backend(SDPABackend)
    return clone if dtype is None else clone.to(dtype)


def _eager_stack(blocks, request, arch, *, fp32: bool) -> torch.Tensor:
    """sglang's blocks run eagerly on one rank with SDPA attention: as served
    (bf16), or with every bf16 cast of H3's code and every weight in fp32
    (``_BF16_DTYPE`` patched, the blocks cast: the rounding floor's truth)."""
    tokens = request["x"].shape[0]
    used = int(request["cu_seqlens"][1])
    suffix = "_fp32" if fp32 else ""
    kwargs = dict(
        adaln_input=request["adaln_input" + suffix],
        combined_indices=request["combined"],
        rope_cache=(request["cos_sin_cache" + suffix], torch.arange(tokens)),
        cu_seqlens=request["cu_seqlens"],
        cu_seqlens_host=(0, used, tokens),
        max_seqlen=used,
    )
    x = request["x"].float() if fp32 else request["x"].clone()
    with contextlib.ExitStack() as stack, torch.no_grad():
        if fp32:
            stack.enter_context(patch.object(dit, "_BF16_DTYPE", torch.float32))
        for block in blocks:
            if fp32:
                block = _clone_block(block, arch, torch.float32)
            for module in block.modules():
                if isinstance(module, dit.MiniMaxH3Attention):
                    module._set_attention_backend(SDPABackend)
            x = block(x, **kwargs)
            del block
    return x.float()


def _rel(got: np.ndarray, want: np.ndarray) -> float:
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _kernel_config(arch, layers: int):
    """``pack.config_from_arch`` without JAX (a stand-in ``Config``)."""
    if _HAS_JAX:
        return pack.config_from_arch(arch, layers)
    return SimpleNamespace(
        hidden=arch.hidden_size,
        heads=arch.num_attention_heads,
        head_dim=arch.attention_head_dim,
        ffn=arch.ffn_hidden_size,
        layers=layers,
        rope_dims=6 * arch.rope_inv_freq_len,
        width=arch.num_attention_heads * arch.attention_head_dim,
    )


@contextlib.contextmanager
def _padding_restored():
    """Restore the packed builders' padding function after a test."""
    with patch.object(
        packed_sequence,
        "minimax_h3_padded_seq_len",
        packed_sequence.minimax_h3_padded_seq_len,
    ):
        yield


def _spec_layout(text_len: int):
    return packed_sequence.minimax_h3_packed_sequence(
        text_len=text_len, include_keyframe_cond=False, **_SPEC_SHAPE
    )


class TestMegakernelPadding(CustomTestCase):
    """The padding seam: the kernel's row count for every packed builder."""

    @unittest.skipUnless(_HAS_JAX, "needs JAX (tiling.padded_rows)")
    def test_1344x768x124_rows_a_device(self):
        # All 21 prompts (37,746-37,757 live rows) share one padded row
        # count per device count: the kernel's padded_rows rule (8: 4,992 a device,
        # not 512-alignment's 4,736 = 37 x 128, whose query tiles would be
        # 128 rows), or the override when a request fits it.
        config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)
        default = {text: _spec_layout(text) for text in (36, 47)}
        for devices, rows, override in (
            (8, 4992, None),
            (4, 9600, None),
            (2, 18944, None),
            (2, 19456, 19456),
            (8, 4864, 4864),
        ):
            with (
                _padding_restored(),
                tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS.override(override),
            ):
                hook._use_kernel_padding(devices, config)
                layouts = {
                    name: _spec_layout(text) for name, text in _SPEC_TEXT_TOKENS.items()
                }
            for name, packed in layouts.items():
                used = 37_296 + 414 + _SPEC_TEXT_TOKENS[name]
                self.assertEqual(packed["seq_len"], rows * devices, (devices, name))
                self.assertEqual(
                    packed["cu_seqlens"].tolist(), [0, used, rows * devices]
                )
                self.assertTrue((packed["img_position_ids"][used:] == 0).all())
                self.assertTrue((packed["token_tags"][used:] == -1).all())
            for text, ref in default.items():
                packed = layouts["prompt-15" if text == 36 else "prompt-01"]
                used = int(ref["cu_seqlens"][1])
                torch.testing.assert_close(
                    packed["img_position_ids"][:used], ref["img_position_ids"][:used]
                )
                torch.testing.assert_close(
                    packed["token_tags"][:used], ref["token_tags"][:used]
                )
        # A request the override cannot hold pads by the rule instead.
        with (
            _padding_restored(),
            tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS.override(4096),
            self.assertLogs(hook.logger, logging.WARNING),
        ):
            hook._use_kernel_padding(8, config)
            self.assertEqual(_spec_layout(40)["seq_len"], 4992 * 8)

    @unittest.skipUnless(_HAS_JAX, "needs JAX (tiling.padded_rows)")
    def test_static_structure_option_rows_a_device(self):
        # The option pads every benchmark prompt to 19,456 rows a device at 2
        # devices (padded_rows: 18,944); the other device counts keep
        # padded_rows' or the override's rows. A 2-device request padded_rows
        # gives other rows keeps them.
        config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)
        for devices, rows, override in _STRUCTURED:
            with (
                _padding_restored(),
                tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_STATIC_STRUCTURE.override(True),
                tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS.override(override),
            ):
                hook._use_kernel_padding(devices, config)
                padded = {
                    _spec_layout(text)["seq_len"] for text in _SPEC_TEXT_TOKENS.values()
                }
                self.assertEqual(padded, {rows * devices}, (devices, override))
                if devices == 2:
                    # 38,000 live rows: padded_rows gives 19,200 a device.
                    self.assertEqual(
                        packed_sequence.minimax_h3_padded_seq_len(38_000), 38_400
                    )

    @unittest.skipUnless(_HAS_JAX, "needs JAX (tiling.padded_rows)")
    def test_comfyui_layout_pads_to_the_kernel_rows(self):
        config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)
        empty = torch.zeros(0, dtype=torch.long)
        layout = {
            "seq_len": 37_749,
            "position_ids": torch.ones(37_749, 3),
            "segments": [(0, 37_749, "text")],
            "signature": (37_749, 1, 2, 2, 0),
            "img_pos": empty,
            "img_update": empty.bool(),
            "audio_pos": empty,
            "audio_update": empty.bool(),
        }
        with _padding_restored():
            hook._use_kernel_padding(8, config)
            packed = comfyui_step.comfyui_layout_to_packed(layout)
        self.assertEqual(packed["cu_seqlens"].tolist(), [0, 37_749, 39_936])

    def test_padding_unchanged_unless_packed(self):
        # Off, refused or dump-only, the builders keep their own padding.
        original = packed_sequence.minimax_h3_padded_seq_len
        with _padding_restored():
            model = _bare_model()
            with patch.object(hook, "_ranks_agree", lambda reason: reason):
                self.assertFalse(hook.enable(model))
            self.assertIs(packed_sequence.minimax_h3_padded_seq_len, original)
            self.assertEqual(_packed_t2va(60)["seq_len"], 128)


def _bare_model(cls=dit.MiniMaxH3DiTModel, packed: bool = False):
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.blocks = torch.nn.ModuleList()
    model.adaln_cache = None
    model.adaln_t_table = None
    model._adaln_precomputed = False
    if packed:
        model.__dict__[hook._STATE] = SimpleNamespace(
            calls=0, begin_request=lambda request_id: None, end_request=lambda: None
        )
    return model


def _packed_t2va(text_len: int):
    return packed_sequence.minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=3,
        latent_h=8,
        latent_w=8,
        audio_t=10,
        include_keyframe_cond=False,
    )


class TestMegakernelInputs(CustomTestCase):
    def test_rope_tables_reproduce_sglang_rope(self):
        # sglang's per-op eager RoPE (bf16, rounding each op, as the CUDA
        # kernels do) equals the kernel's whole-head rotation over these
        # tables, rounding where it rounds. The RotaryEmbedding path computes
        # in fp32 and rounds once: three roundings of terms bounded by the
        # head's largest input put the two within 3 bf16 eps of it.
        arch = _arch(heads=2)
        request = _request(arch, 64, 64)
        cos, sin = pack.rope_tables(request["cos_sin_cache"], 128)
        self.assertEqual((cos.shape, cos.dtype), ((64, 128), _BF16))
        self.assertTrue((cos[:, 96:] == 1).all() and (sin[:, 96:] == 0).all())
        q = torch.randn(64, 2, 128).to(_BF16)
        k = torch.randn(64, 2, 128).to(_BF16)
        once_q, once_k = dit._apply_rope_qk(
            q.clone(),
            k.clone(),
            request["cos_sin_cache"],
            torch.arange(64),
            rotary_emb=_blocks(arch, 1)[0].attn.rotary_emb,
        )
        for x, once in ((q, once_q), (k, once_k)):
            got = _kernel_rope(x, cos, sin, 96)
            want = dit._apply_rope(x, request["cos_sin_cache"])
            torch.testing.assert_close(got, want, rtol=0, atol=0)
            bound = 3 * torch.finfo(_BF16).eps * x.float().abs().amax(-1, True)
            self.assertTrue(((got.float() - once.float()).abs() <= bound).all())

    def test_onehot_selects_sglang_rows(self):
        combined = torch.tensor([0, 5, 3, 3, 1])
        table = torch.randn(16, 32).to(_BF16)
        onehot = pack.table_onehot(combined)
        self.assertEqual((onehot.shape, onehot.dtype), ((5, 128), _BF16))
        got = onehot[:, :16].float() @ table.float()
        torch.testing.assert_close(got.to(_BF16), table[combined], rtol=0, atol=0)

    def test_adaln_tables_hold_each_blocks_params(self):
        arch = _arch(heads=2)
        blocks = _blocks(arch, 2)
        adaln_input = torch.randn(3, arch.time_embed_dim).to(_BF16)
        with torch.no_grad():
            tables = pack.adaln_tables([b.adaln_proj for b in blocks], adaln_input)
            self.assertEqual(tables.shape, (2, 6, 16, arch.hidden_size))
            for layer, block in enumerate(blocks):
                for k, param in enumerate(block.adaln_proj(adaln_input)):
                    torch.testing.assert_close(
                        tables[layer, k, :9], param, rtol=0, atol=0
                    )
            self.assertFalse(tables[:, :, 9:].any())
            # Each call projects its own input: no table survives a call.
            other = pack.adaln_tables([b.adaln_proj for b in blocks], adaln_input * 2)
            self.assertFalse(torch.equal(other, tables))
            with self.assertRaisesRegex(ValueError, "AdaLN rows"):
                pack.adaln_tables(
                    [b.adaln_proj for b in blocks], adaln_input.repeat(15, 1)
                )

    def test_operands_read_the_live_length_on_device(self):
        arch = _arch(heads=2)
        blocks = _blocks(arch, 1)
        request = _request(arch, 128, 77)
        stack = hook.H3MegakernelStack(
            _bare_model(),
            [b.adaln_proj for b in blocks],
            {name: torch.zeros(1) for name in pack.WEIGHT_ORDER},
            None,
            _kernel_config(arch, 1),
        )
        with torch.no_grad():
            *_, used = stack.operands(
                request["adaln_input"],
                request["combined"],
                request["cos_sin_cache"],
                request["cu_seqlens"],
            )
        self.assertEqual((used.tolist(), used.dtype), ([77], torch.int32))
        self.assertEqual(used.device, request["cu_seqlens"].device)


def _kernel_rope(x, cos, sin, rope_dims):
    """The kernel's whole-head rotate-half: ``x cos + rotate(x) sin`` with
    ``rotate(x) = (-x2, x1, 0)``, each product rounded to bf16 as sglang's
    eager path rounds."""
    half = rope_dims // 2
    lane = torch.arange(x.shape[-1])
    rotated = torch.where(
        lane < half,
        -torch.roll(x, -half, -1),
        torch.where(lane < 2 * half, torch.roll(x, half, -1), 0),
    )
    return (x * cos[:, None]).to(_BF16) + (rotated * sin[:, None]).to(_BF16)


@contextlib.contextmanager
def _installed(run_full_loop=None):
    """``hook.install`` with every patched class attribute restored after;
    ``run_full_loop`` stands in for the stage's denoise loop."""
    cls = dit.MiniMaxH3DiTModel
    names = (
        "post_load_weights",
        "validate_weight_update_source",
        "validate_lora_layers",
    )
    loop = run_full_loop or MiniMaxH3DenoisingStage._run_full_loop
    with (
        patch.multiple(cls, **{name: getattr(cls, name) for name in names}),
        patch.object(MiniMaxH3DenoisingStage, "_run_full_loop", loop),
        patch.object(
            LoRAPipeline,
            "_reject_lora_on_packed_weights",
            LoRAPipeline._reject_lora_on_packed_weights,
        ),
    ):
        hook.install()
        hook.install()  # idempotent
        try:
            yield cls
        finally:
            del cls._tpu_h3_megakernel_installed


def _tiny_model(arch=None, device=None):
    """A two-block H3 DiT with random weights and SDPA attention."""
    _init_parallel()
    torch.manual_seed(0)
    arch = arch or _arch(heads=2)
    with torch.device(device or "cpu"):
        model = dit.MiniMaxH3DiTModel(
            config=MiniMaxH3DiTConfig(arch_config=arch), hf_config={}, quant_config=None
        )
    if device is None:
        model.requires_grad_(False)
        for param in model.parameters():
            param.data.normal_(0, 0.02)
        half = arch.rope_inv_freq_len
        model.rope.inv_freq.copy_(
            10000.0 ** (-torch.arange(0, 2 * half, 2, dtype=torch.float32) / (2 * half))
        )
        for block in model.blocks:
            _randomize_block(block, arch)
        for module in model.modules():
            if isinstance(module, dit.MiniMaxH3Attention):
                module._set_attention_backend(SDPABackend)
        model._resolved_attention_backend = AttentionBackendEnum.TORCH_SDPA
    return model


def _forward_kwargs(arch, text_len: int = 52):
    """A t2va forward's DiT kwargs: 228 live rows (text 52, audio 48, video
    128) in 256 packed rows, a multiple of 128 for the stack on one rank."""
    packed = packed_sequence.minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=2,
        latent_h=16,
        latent_w=16,
        audio_t=24,
        include_keyframe_cond=False,
    )
    seq_len = packed["seq_len"]
    assert seq_len % 128 == 0, seq_len
    generator = torch.Generator().manual_seed(3)
    return MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.randn(text_len, arch.text_dim, generator=generator).to(
            _BF16
        ),
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
    ).forward_kwargs(
        video_rows=torch.randn(
            packed["img_pos"].numel(), MINIMAX_H3_VIDEO_ROW_WIDTH, generator=generator
        ),
        audio_rows=torch.randn(
            packed["audio_pos"].numel(), MINIMAX_H3_AUDIO_ROW_WIDTH, generator=generator
        ),
        step_timesteps=(
            torch.tensor([0.25, 0.5]),
            torch.zeros(seq_len).long(),
            None,
        ),
    )


class _Platform:
    def __init__(self, tpu: bool):
        self.tpu = tpu

    def is_tpu(self):
        return self.tpu


def _server_args(**overrides):
    values = dict(
        minimax_h3_adaln_cache_path=None,
        minimax_h3_adaln_online=False,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        enable_torch_compile=False,
        lora_path=None,
    )
    resident = overrides.pop("resident", True)
    subblock = overrides.pop("subblock", False)
    backend = overrides.pop("backend", None)
    values.update(overrides)
    return SimpleNamespace(
        residency_mode=lambda name: "resident" if resident else "offload",
        pipeline_config=SimpleNamespace(
            uses_subblock_attention=lambda server_args: subblock,
            resolve_transformer_attention_backend=lambda server_args: backend,
        ),
        **values,
    )


@contextlib.contextmanager
def _support(
    tpu=True,
    ulysses=(2, 0),
    world=(2, 0),
    ring=1,
    tp=1,
    server_args=None,
    cache_dit=False,
):
    """``unsupported_reason``'s view of the world, patched."""
    state = "sglang.multimodal_gen.runtime.distributed.parallel_state"
    with (
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform", _Platform(tpu)
        ),
        patch(f"{state}.get_ulysses_ctx", return_value=ulysses),
        patch(f"{state}.get_ring_ctx", return_value=(ring, 0)),
        patch(f"{state}.get_tp_world_size", return_value=tp),
        patch(f"{state}.get_world_size", return_value=world[0]),
        patch(f"{state}.get_world_rank", return_value=world[1]),
        patch("sglang.multimodal_gen.envs.SGLANG_CACHE_DIT_ENABLED", cache_dit),
        patch.object(hook, "_server_args", return_value=server_args or _server_args()),
    ):
        yield


class TestMegakernelRefusals(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # A meta-device model passes the "on device" check without memory.
        cls.model = _tiny_model(device="meta")

    @unittest.skipUnless(_HAS_JAX, "needs JAX (tiling.fit)")
    def test_supported_layout_passes_and_heads_must_split(self):
        with _support():
            self.assertIsNone(hook.unsupported_reason(self.model))
        # Ulysses must split the heads (2 heads here).
        with _support(ulysses=(4, 0), world=(4, 0)):
            self.assertIn(
                "heads that split 4 ways", hook.unsupported_reason(self.model)
            )

    def test_each_refusal(self):
        from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod

        model = self.model
        block = model.blocks[1]
        cases = [
            ("not a TPU platform", dict(tpu=False), None),
            ("ring degree 1 and TP 1", dict(ring=2), None),
            ("ring degree 1 and TP 1", dict(tp=2), None),
            ("Ulysses over the whole world", dict(ulysses=(1, 0), world=(1, 0)), None),
            ("Ulysses over the whole world", dict(ulysses=(2, 0), world=(4, 0)), None),
            ("Ulysses rank differs", dict(ulysses=(2, 1), world=(2, 0)), None),
            (
                "AdaLN cache (--minimax-h3",
                dict(server_args=_server_args(minimax_h3_adaln_online=True)),
                None,
            ),
            (
                "AdaLN cache (--minimax-h3",
                dict(server_args=_server_args(minimax_h3_adaln_cache_path="/x")),
                None,
            ),
            (
                "--use-fsdp-inference",
                dict(server_args=_server_args(use_fsdp_inference=True)),
                None,
            ),
            (
                "--dit-cpu-offload",
                dict(server_args=_server_args(dit_cpu_offload=True)),
                None,
            ),
            (
                "--dit-layerwise-offload",
                dict(server_args=_server_args(dit_layerwise_offload=True)),
                None,
            ),
            (
                "--enable-torch-compile",
                dict(server_args=_server_args(enable_torch_compile=True)),
                None,
            ),
            ("LoRA", dict(server_args=_server_args(lora_path="/lora")), None),
            ("not resident", dict(server_args=_server_args(resident=False)), None),
            ("sub-block sparse", dict(server_args=_server_args(subblock=True)), None),
            (
                "sparse attention backend video_sparse_attn_h3",
                dict(
                    server_args=_server_args(
                        backend=AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3
                    )
                ),
                None,
            ),
            ("Cache-DiT", dict(cache_dit=True), None),
            ("AdaLN cache or curve AdaLN", {}, (model, "adaln_cache", object())),
            ("unsupported block", {}, (model.blocks, "1", torch.nn.Identity())),
            (
                "hybrid or sparse",
                {},
                (block.attn, "to_gate_compress", torch.nn.Identity()),
            ),
            (
                "not plain bf16 device tensors",
                {},
                (block.mlp.fc2, "quant_method", MagicMock(spec=object)),
            ),
            (
                "not plain bf16 device tensors",
                {},
                (
                    block.adaln_proj,
                    "linear",
                    torch.nn.Linear(4, 4, dtype=torch.float32),
                ),
            ),
        ]
        for want, support, swap in cases:
            with self.subTest(want=want, support=support, swap=swap and swap[1]):
                with contextlib.ExitStack() as stack:
                    stack.enter_context(_support(**support))
                    if swap is not None:
                        owner, name, value = swap
                        if isinstance(owner, torch.nn.ModuleList):
                            original = owner[int(name)]
                            owner[int(name)] = value
                            stack.callback(owner.__setitem__, int(name), original)
                        else:
                            stack.enter_context(patch.object(owner, name, value))
                    reason = hook.unsupported_reason(model)
                self.assertIsNotNone(reason)
                self.assertIn(want, reason)
        self.assertIsInstance(block.mlp.fc2.quant_method, UnquantizedLinearMethod)

    @unittest.skipUnless(_HAS_JAX, "needs JAX (tiling.fit)")
    def test_rows_override_must_tile(self):
        with _support(), tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS.override(4000):
            self.assertIn(
                "SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS=4000",
                hook.unsupported_reason(self.model),
            )
        with _support(), tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS.override(4096):
            self.assertIsNone(hook.unsupported_reason(self.model))

    @unittest.skipUnless(_HAS_JAX, "needs JAX (op.parse_tiling)")
    def test_tiling_bounds_must_parse(self):
        knob = tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_TILING
        structured = tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_STATIC_STRUCTURE
        with _support(), structured.override(True), knob.override("block_q=x"):
            self.assertIn(
                "SGLANG_TPU_OPT_H3_MEGAKERNEL_TILING",
                hook.unsupported_reason(self.model),
            )
        # Read only with the static-structure option.
        with _support(), knob.override("block_q=x"):
            self.assertIsNone(hook.unsupported_reason(self.model))
        with _support(), structured.override(True), knob.override("block_q=4096"):
            self.assertIsNone(hook.unsupported_reason(self.model))

    def test_refusal_or_failing_check_leaves_the_blocks(self):
        # A rank whose check raises still joins the vote, as a refusal.
        model = _bare_model()
        votes = []
        with (
            patch.object(hook, "unsupported_reason", side_effect=KeyError("attn")),
            patch.object(hook, "_ranks_agree", lambda why: votes.append(why) or why),
        ):
            self.assertFalse(hook.enable(model))
            self.assertFalse(hook.enable(model))  # decided once
        self.assertEqual(len(votes), 1)
        self.assertIn("KeyError", votes[0])
        self.assertFalse(hook.is_packed(model))

    def test_request_refusals(self):
        stage = SimpleNamespace(
            _cache_dit_requested_for_batch=lambda batch: bool(
                batch.sampling_params.enable_cache_dit
            )
        )

        def batch(**fields):
            params = dict(
                enable_spectrum=False,
                enable_teacache=False,
                enable_cache_dit=None,
                quality="lossless",
            )
            params.update(fields)
            return SimpleNamespace(sampling_params=SimpleNamespace(**params))

        self.assertIsNone(hook.request_refusal(stage, batch()))
        self.assertIn(
            "Spectrum", hook.request_refusal(stage, batch(enable_spectrum=True))
        )
        self.assertIn(
            "TeaCache", hook.request_refusal(stage, batch(enable_teacache=True))
        )
        self.assertIn(
            "Cache-DiT", hook.request_refusal(stage, batch(enable_cache_dit=True))
        )
        self.assertIn("Cache-DiT", hook.request_refusal(stage, batch(quality="high")))
        # The per-request kill switch wins over quality="high", as in H3's stage.
        self.assertIsNone(
            hook.request_refusal(stage, batch(quality="high", enable_cache_dit=False))
        )


def _fake_op(blocks, request, packed_reference, record):
    """An op that checks the operands it gets against sglang's own values and
    runs sglang's (unpacked copies of the) blocks on them."""

    def op(x, onehot, cos, sin, tables, used, *weights):
        record.append(x)
        adaln_input = request["adaln_input"]
        want_tables = pack.adaln_tables([b.adaln_proj for b in blocks], adaln_input)
        torch.testing.assert_close(tables, want_tables, rtol=0, atol=0)
        torch.testing.assert_close(
            onehot, pack.table_onehot(request["combined_indices"]), rtol=0, atol=0
        )
        want_cos, want_sin = pack.rope_tables(request["rope_cache"][0], 128)
        torch.testing.assert_close(cos, want_cos, rtol=0, atol=0)
        torch.testing.assert_close(sin, want_sin, rtol=0, atol=0)
        assert (
            used.tolist() == [int(request["cu_seqlens"][1])]
            and used.dtype == torch.int32
        )
        for name, weight in zip(pack.WEIGHT_ORDER, weights):
            torch.testing.assert_close(weight, packed_reference[name], rtol=0, atol=0)
        hidden = x.clone()
        for block in blocks:
            hidden = block(hidden, **{k: v for k, v in request.items()})
        return hidden

    return op


class TestMegakernelHook(CustomTestCase):
    def _enabled(self, model, op_factory):
        """``hook.enable`` as on a 2-rank TPU world, with ``op_factory``'s op."""
        with (
            patch.object(hook, "unsupported_reason", return_value=None),
            patch.object(hook, "_ranks_agree", lambda reason: reason),
            patch.object(hook, "_world_size", return_value=2),
            patch.object(hook, "_op_for", side_effect=op_factory),
            patch.object(
                hook,
                "_pack_on_every_rank",
                lambda blocks, config: pack.pack_blocks(
                    blocks, config, stage_on_host=True
                ),
            ),
            self.assertLogs(hook.logger, logging.INFO) as logs,
        ):
            self.assertTrue(hook.enable(model))
        return logs.output

    @unittest.skipUnless(_HAS_JAX, "needs JAX (reference.Config, tiling)")
    def test_enable_swaps_the_blocks_and_the_forward_reaches_the_op(self):
        arch = _arch(heads=2)
        model = _tiny_model(arch)
        kwargs = _forward_kwargs(arch)
        # sglang's own run: the rows entering block 0, its kwargs, the output.
        seen = {}

        def pre_hook(module, args, block_kwargs):
            seen["hidden"] = args[0].clone()
            seen["kwargs"] = dict(block_kwargs)

        handle = model.blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
        with torch.no_grad():
            want_video, want_audio = model(**kwargs)
        handle.remove()
        originals = [_clone_block(block, arch) for block in model.blocks]
        adaln = [block.adaln_proj for block in model.blocks]
        packed_reference = pack.pack_blocks(
            [_clone_block(block, arch) for block in originals],
            pack.config_from_arch(arch, 2),
            free_originals=False,
        )
        calls = []
        request = dict(seen["kwargs"])
        with _padding_restored():
            logs = self._enabled(
                model,
                lambda config, devices, structured: _fake_op(
                    originals, request, packed_reference, calls
                ),
            )
            padding = packed_sequence.minimax_h3_padded_seq_len
        self.assertEqual(padding.func, hook.padded_seq_len)
        self.assertEqual(padding.keywords["devices"], 2)
        (stack,) = model.blocks
        self.assertIsInstance(stack, hook.H3MegakernelStack)
        self.assertTrue(hook.is_packed(model))
        self.assertEqual([id(p) for p in stack.adaln], [id(p) for p in adaln])
        for name in pack.WEIGHT_ORDER:
            torch.testing.assert_close(
                getattr(stack, name), packed_reference[name], rtol=0, atol=0
            )
        self.assertNotIn("blocks.0.wqkv", model.state_dict())  # not persistent
        # The batched-AdaLN and live-length probes see the wrapper.
        self.assertFalse(model._can_batch_block_adaln())
        self.assertFalse(model._attention_reads_live_length())
        (line,) = [line for line in logs if "megakernel on" in line]
        self.assertIn("2 blocks, Ulysses 2", line)
        self.assertIn("packed bf16 weights a device", line)
        self.assertIn("(2/2): 0-1; online softmax: none", line)
        # sglang's forward calls the stack once, with block 0's arguments,
        # and the op once; with an exact op the DiT's output is unchanged.
        real_forward = hook.H3MegakernelStack.forward

        def forward(stack_self, hidden, **block_kwargs):
            # One process here: Ulysses is what the 2-rank world would set.
            return real_forward(
                stack_self, hidden, **{**block_kwargs, "ulysses_active": True}
            )

        with (
            patch.object(
                hook.H3MegakernelStack, "forward", autospec=True, side_effect=forward
            ) as stack_forward,
            torch.no_grad(),
        ):
            got_video, got_audio = model(**kwargs)
        self.assertEqual(stack_forward.call_count, 1)
        call_kwargs = stack_forward.call_args.kwargs
        self.assertEqual(sorted(call_kwargs), sorted(seen["kwargs"]))
        torch.testing.assert_close(
            stack_forward.call_args.args[1], seen["hidden"], rtol=0, atol=0
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(stack.calls, 1)
        torch.testing.assert_close(got_video, want_video, rtol=0, atol=0)
        torch.testing.assert_close(got_audio, want_audio, rtol=0, atol=0)

    def test_stack_refuses_calls_it_cannot_run(self):
        arch = _arch(heads=2)
        model = _bare_model()
        stack = hook.H3MegakernelStack(
            model,
            [b.adaln_proj for b in _blocks(arch, 1)],
            {name: torch.zeros(1) for name in pack.WEIGHT_ORDER},
            MagicMock(),
            _kernel_config(arch, 1),
        )
        model.blocks = torch.nn.ModuleList([stack])
        request = _request(arch, 128, 100)
        good = dict(
            adaln_input=request["adaln_input"],
            combined_indices=request["combined"],
            rope_cache=(request["cos_sin_cache"], torch.arange(128)),
            cu_seqlens=request["cu_seqlens"],
            ulysses_active=True,
        )
        with torch.no_grad():
            stack(request["x"], **good)
        self.assertEqual(stack.op.call_count, 1)
        for bad in (
            dict(ring_active=True),
            dict(ulysses_active=False),
            dict(adaln_params=(torch.zeros(1),) * 6),
            dict(subblock_sparse_query_block_mask=torch.zeros(1)),
            dict(cu_seqlens=torch.tensor([0, 50, 100, 128], dtype=torch.int32)),
        ):
            with self.subTest(bad=sorted(bad)), self.assertRaises(RuntimeError):
                stack(request["x"], **{**good, **bad})
        with self.assertRaisesRegex(RuntimeError, "multiple of 128"):
            stack(request["x"][:100], **good)
        with self.assertRaisesRegex(RuntimeError, "bf16"):
            stack(request["x"].float(), **good)
        # Cache-DiT puts its own wrapper in place of the block list.
        model.blocks = torch.nn.ModuleList([torch.nn.Identity()])
        with self.assertRaisesRegex(RuntimeError, "changed after packing"):
            stack(request["x"], **good)
        model.blocks = torch.nn.ModuleList([stack])
        model._resolved_attention_backend = AttentionBackendEnum.CUBE_SPARSE_ATTN
        with self.assertRaisesRegex(RuntimeError, "dense attention"):
            stack(request["x"], **good)
        self.assertEqual(stack.op.call_count, 1)

    def test_packed_model_refuses_lora_and_weight_updates(self):
        with _installed() as cls:
            packed, plain = _bare_model(cls, packed=True), _bare_model(cls)
            with self.assertRaisesRegex(ValueError, "LoRA"):
                packed.validate_lora_layers(["blocks.0.attn.qkv_proj"])
            with self.assertRaisesRegex(ValueError, "weight updates"):
                packed.validate_weight_update_source(weights_path="/x")
            plain.validate_lora_layers(["blocks.0.attn.qkv_proj"])
            # set_lora and convert_to_lora_layers check here before any layer
            # is replaced.
            pipeline = SimpleNamespace(
                modules={"transformer": packed}, is_target_layer=lambda name: True
            )
            with self.assertRaisesRegex(ValueError, "LoRA"):
                LoRAPipeline._reject_lora_on_packed_weights(pipeline)
            pipeline.modules["transformer"] = plain
            LoRAPipeline._reject_lora_on_packed_weights(pipeline)

    def test_each_request_logs_its_op_calls_and_refuses_step_caching(self):
        model = _bare_model(packed=True)
        stack = model.__dict__[hook._STATE]

        def loop(stage, batch, server_args):
            stack.calls += 49  # one op call a denoise step

        loop_mock = MagicMock(side_effect=loop)
        stage = SimpleNamespace(
            transformer=model, _cache_dit_requested_for_batch=lambda batch: False
        )

        def batch(rid, **fields):
            params = dict(
                enable_spectrum=False,
                enable_teacache=False,
                enable_cache_dit=None,
                quality="lossless",
            )
            params.update(fields)
            return SimpleNamespace(
                request_id=rid,
                is_warmup=False,
                sampling_params=SimpleNamespace(**params),
            )

        with _installed(run_full_loop=loop_mock):
            run = MiniMaxH3DenoisingStage._run_full_loop
            with self.assertLogs(hook.logger, logging.INFO) as logs:
                run(stage, batch("r1"), None)
                run(stage, batch("r2"), None)
            self.assertEqual(len(logs.output), 2)
            for rid, line in zip(("r1", "r2"), logs.output):
                self.assertTrue(
                    line.endswith(
                        f"MiniMax H3 megakernel: request {rid} ran the block-stack "
                        "op 49 times"
                    ),
                    line,
                )
            for fields in (dict(enable_spectrum=True), dict(quality="high")):
                with self.assertRaisesRegex(ValueError, "Spectrum|Cache-DiT"):
                    run(stage, batch("r3", **fields), None)
            self.assertEqual(loop_mock.call_count, 2)
            # An unpacked DiT goes straight to sglang's loop.
            stage.transformer = _bare_model()
            run(stage, batch("r4", enable_spectrum=True), None)
            self.assertEqual(loop_mock.call_count, 3)

    def test_host_waves_fit_free_memory(self):
        # 8 ranks on one host, 65 GB each, 400 GB free: 4 at a time.
        infos = [("host", 65 << 30, 400 << 30)] * 8
        group = (
            "sglang.multimodal_gen.runtime.distributed.parallel_state.get_world_group"
        )
        for rank, wave in ((0, 0), (3, 0), (4, 1), (7, 1)):
            with (
                patch.object(hook, "_gather", return_value=infos),
                patch(group, return_value=SimpleNamespace(rank_in_group=rank)),
            ):
                self.assertEqual(hook._host_waves(65 << 30), (wave, 2))

    def test_host_memory_respects_the_cgroup_limit(self):
        import builtins
        import io

        files = {
            "/sys/fs/cgroup/memory.max": str(100 << 30),
            "/sys/fs/cgroup/memory.current": str(40 << 30),
        }
        real_open = builtins.open

        def fake_open(path, *args, **kwargs):
            if path in files:
                return io.StringIO(files[path])
            return real_open(path, *args, **kwargs)

        with (
            patch(
                "psutil.virtual_memory", return_value=SimpleNamespace(available=1 << 40)
            ),
            patch("builtins.open", fake_open),
        ):
            self.assertEqual(hook._host_available(), 60 << 30)
            files["/sys/fs/cgroup/memory.max"] = "max"
            self.assertEqual(hook._host_available(), 1 << 40)

    def test_worker_installs_the_h3_hook_when_enabled(self):
        from sglang_tpu import patches
        from sglang_tpu.multimodal_gen import minimax_h3, rope, worker

        for h3_on in (False, True):
            with (
                patch.object(patches, "_register_aten_fallbacks"),
                patch.object(rope, "install"),
                patch.object(minimax_h3, "install") as minimax_h3_install,
                patch.object(hook, "install") as h3_install,
                tpu_envs.SGLANG_TPU_ENABLE_H3_MEGAKERNEL.override(h3_on),
                patch.object(torch, "set_float32_matmul_precision"),
            ):
                worker.install_runtime_patches()
            minimax_h3_install.assert_called_once_with()
            self.assertEqual(h3_install.call_count, int(h3_on))


# ------------------------------------------------------------ vouched heads


def _vouched_fake_op(record: list, layers: int, heads: int):
    """The vouched op's stand-in: checks its operands, records each call's
    tried heads, tally and outputs, and returns the rows unchanged, every
    head vouched (``call + 1``) and the tally plus (1 on odd calls, the
    heads it was given)."""

    def op(x, onehot, cos, sin, tables, used, vouch, tally, *weights):
        assert (vouch.shape, vouch.dtype) == ((layers, heads), torch.int32)
        assert (tally.shape, tally.dtype) == ((2,), torch.int32)
        assert used.dtype == torch.int32 and len(weights) == len(pack.WEIGHT_ORDER)
        call = len(record)
        step = torch.tensor([call % 2, int(vouch.count_nonzero())], dtype=torch.int32)
        out = (
            x.clone(),
            torch.full((layers, heads), call + 1, dtype=torch.int32),
            tally + step,
        )
        record.append(SimpleNamespace(vouch=vouch, tally=tally, out=out))
        return out

    return op


# A log reader's pattern for the per-request line (a benchmark runner counts
# the op calls of each request with it).
_OP_CALLS = (
    r"MiniMax H3 megakernel: request (?P<rid>\S+) ran the block-stack op "
    r"(?P<calls>\d+) times(?P<warmup> \(warmup\))?"
)


class TestVouchedHeads(CustomTestCase):
    """SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH: the heads a step tries at a fixed
    max come from its own request's previous step only."""

    layers, heads = 2, 2

    def setUp(self):
        self.arch = _arch(heads=self.heads)
        request = _request(self.arch, 128, 100)
        self.x = request["x"]
        self.kwargs = dict(
            adaln_input=request["adaln_input"],
            combined_indices=request["combined"],
            rope_cache=(request["cos_sin_cache"], torch.arange(128)),
            cu_seqlens=request["cu_seqlens"],
            ulysses_active=True,
        )

    def _stack(self, op, margin=1.1, model=None):
        model = model or _bare_model()
        stack = hook.H3MegakernelStack(
            model,
            [b.adaln_proj for b in _blocks(self.arch, self.layers)],
            {name: torch.zeros(1) for name in pack.WEIGHT_ORDER},
            op,
            _kernel_config(self.arch, self.layers),
            vouch_margin=margin,
        )
        model.blocks = torch.nn.ModuleList([stack])
        return stack

    def _steps(self, stack, steps: int, batch=None):
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            set_forward_context,
        )

        with torch.no_grad():
            for step in range(steps):
                with set_forward_context(step, None, forward_batch=batch):
                    stack(self.x, **self.kwargs)

    def assert_starts_proven_only(self, call):
        # No heads to try and a zero tally: the proven layers only.
        self.assertEqual(call.vouch.shape, (self.layers, self.heads))
        self.assertFalse(call.vouch.any())
        self.assertFalse(call.tally.any())

    def assert_chained(self, calls):
        # Each step tries exactly the heads its request's previous step
        # returned, as the same device tensors (no host round trip).
        for before, after in zip(calls, calls[1:]):
            self.assertIs(after.vouch, before.out[1])
            self.assertIs(after.tally, before.out[2])

    @unittest.skipUnless(_HAS_JAX, "needs JAX (reference.Config, tiling)")
    def test_enable_builds_the_vouched_op_and_logs_its_margin(self):
        op = op_module()
        config = pack.config_from_arch(self.arch, self.layers)
        vouch = tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH
        with (
            patch.dict(hook._OPS, clear=True),
            patch.object(op, "build_stack_op", side_effect=lambda *a, **k: k) as build,
        ):
            self.assertEqual(
                hook._op_for(config, 2), {"structured": False, "vouch_margin": None}
            )
            with vouch.override(True):
                self.assertEqual(
                    hook._op_for(config, 2), {"structured": False, "vouch_margin": 1.1}
                )
                hook._op_for(config, 2)  # cached per margin
        self.assertEqual(build.call_count, 2)
        model = _tiny_model(self.arch)
        with vouch.override(True), _padding_restored():
            logs = TestMegakernelHook._enabled(
                self,
                model,
                lambda config, devices, structured: MagicMock(name="vouched op"),
            )
        (stack,) = model.blocks
        self.assertEqual(stack.vouch_margin, 1.1)
        (line,) = [line for line in logs if "megakernel on" in line]
        self.assertIn("online softmax: none; from each request's second step", line)
        self.assertIn("by margin 1.1, checked on each step's own peaks", line)

    def test_off_by_default(self):
        self.assertFalse(tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH.get())
        self.assertIsNone(hook._vouch_margin())
        with tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH.override(True):
            self.assertEqual(hook._vouch_margin(), hook.VOUCH_MARGIN)
        # Without a margin the stack calls the plain op and keeps nothing.
        op = MagicMock(return_value="rows")
        stack = self._stack(op, margin=None)
        stack.begin_request("r1")
        self.assertIsNone(stack._vouching)
        with torch.no_grad():
            self.assertEqual(stack(self.x, **self.kwargs), "rows")
        self.assertEqual(len(op.call_args.args), 6 + len(pack.WEIGHT_ORDER))
        self.assertIsNone(stack.end_request())
        self.assertEqual(hook._vouch_summary(None, 49), "")

    def test_each_request_starts_proven_only_and_keeps_only_its_own_heads(self):
        record = []
        stack = self._stack(_vouched_fake_op(record, self.layers, self.heads))
        tallies = []
        for request_id, steps in (("a", 3), ("b", 2)):
            stack.begin_request(request_id)
            self._steps(stack, steps, SimpleNamespace(request_id=request_id))
            tallies.append(stack.end_request())
            self.assertIsNone(stack._vouching)
        a, b = record[:3], record[3:]
        for calls in (a, b):
            self.assert_starts_proven_only(calls[0])
            self.assert_chained(calls)
        # b's first step saw none of a's heads: a new zero tensor, not a's
        # last output (which vouched every head).
        self.assertIsNot(b[0].vouch, a[-1].out[1])
        self.assertTrue(a[-1].out[1].all())
        # The tally is each request's own: (odd calls, heads given).
        self.assertEqual(tallies, [(1, 8), (1, 4)])
        self.assertEqual(
            hook._vouch_summary(tallies[0], 3),
            "; 2.7 heads a step at a fixed max beyond the proven layers, "
            "1 step(s) recomputed with the proven layers only",
        )

    def test_a_forward_outside_a_request_tries_no_heads_and_keeps_none(self):
        record = []
        stack = self._stack(_vouched_fake_op(record, self.layers, self.heads))
        self._steps(stack, 2)
        for call in record:
            self.assert_starts_proven_only(call)
        self.assertIsNot(record[1].vouch, record[0].out[1])
        self.assertIsNone(stack._vouching)
        self.assertIsNone(stack.end_request())

    def test_a_forward_of_another_request_is_refused(self):
        record = []
        stack = self._stack(_vouched_fake_op(record, self.layers, self.heads))
        stack.begin_request("a")
        self._steps(stack, 1, SimpleNamespace(request_id="a"))
        with self.assertRaisesRegex(RuntimeError, "request b while request a"):
            self._steps(stack, 1, SimpleNamespace(request_id="b"))
        self.assertEqual(len(record), 1)
        # A new request drops whatever the last one left, even without its
        # end_request.
        stack.begin_request("c")
        self.assertEqual(stack._vouching.request_id, "c")
        self._steps(stack, 1, SimpleNamespace(request_id="c"))
        self.assert_starts_proven_only(record[1])
        self.assertIsNot(record[1].vouch, record[0].out[1])

    def test_requests_through_the_denoise_loop(self):
        # The hook's request scope around sglang's loop: a warmup request, a
        # request that fails mid-loop, then a measured one. Each starts at
        # the proven layers, nothing outlives its request, and each request's
        # log line reports its own tally.
        record = []
        model = _bare_model()
        stack = self._stack(
            _vouched_fake_op(record, self.layers, self.heads), model=model
        )
        model.__dict__[hook._STATE] = stack
        stage = SimpleNamespace(
            transformer=model, _cache_dit_requested_for_batch=lambda batch: False
        )

        def loop(stage, batch, server_args):
            self.assertIsNotNone(stack._vouching)
            self.assertEqual(stack._vouching.request_id, batch.request_id)
            self._steps(stack, 2 if batch.request_id == "fails" else 3, batch)
            if batch.request_id == "fails":
                raise RuntimeError("lost a step")

        def batch(rid, warmup=False):
            return SimpleNamespace(
                request_id=rid,
                is_warmup=warmup,
                sampling_params=SimpleNamespace(
                    enable_spectrum=False,
                    enable_teacache=False,
                    enable_cache_dit=None,
                    quality="lossless",
                ),
            )

        with _installed(run_full_loop=MagicMock(side_effect=loop)):
            run = MiniMaxH3DenoisingStage._run_full_loop
            with self.assertLogs(hook.logger, logging.INFO) as logs:
                run(stage, batch("warm", warmup=True), None)
                with self.assertRaisesRegex(RuntimeError, "lost a step"):
                    run(stage, batch("fails"), None)
                run(stage, batch("measured"), None)
        self.assertIsNone(stack._vouching)
        requests = (record[0:3], record[3:5], record[5:8])
        for calls in requests:
            self.assert_starts_proven_only(calls[0])
            self.assert_chained(calls)
        self.assertEqual(len(logs.output), 3)
        for line, (rid, calls, tally, suffix) in zip(
            logs.output,
            (
                ("warm", 3, (1, 8), " (warmup)"),
                ("fails", 2, (1, 4), ""),
                ("measured", 3, (2, 8), ""),
            ),
        ):
            want = (
                f"request {rid} ran the block-stack op {calls} times"
                + suffix
                + hook._vouch_summary(tally, calls)
            )
            self.assertTrue(line.endswith(want), line)
            # The runner's log contract still reads the request, its op
            # calls and the warmup mark.
            match = re.search(_OP_CALLS, line)
            self.assertEqual(
                (match["rid"], match["calls"], bool(match["warmup"])),
                (rid, str(calls), bool(suffix)),
            )


# ---------------------------------------------------------------- real blocks


def _spec_used(text_len: int) -> int:
    """A 1344x768x124 prompt's live rows: 37,296 video, 414 audio, its text."""
    return 37_296 + 414 + text_len


def _structured_stack(
    model, config, devices: int, structured: bool = True, vouch_margin=None
):
    """A packed stack over ``config``'s loop structures with one tiny block
    (operands) and a recording op; ``vouch_margin``: the vouched variant."""
    arch = _arch(heads=2)
    stack = hook.H3MegakernelStack(
        model,
        [block.adaln_proj for block in _blocks(arch, 1)],
        {name: torch.zeros(1) for name in pack.WEIGHT_ORDER},
        MagicMock(name="op"),
        config,
        devices=devices,
        structured=structured,
        vouch_margin=vouch_margin,
    )
    model.blocks = torch.nn.ModuleList([stack])
    return stack


def _structured_call(stack, rows: int, host, used: int = 37_749):
    """One call of ``stack`` on ``rows`` rows with host bounds ``host``;
    returns the op's arguments."""
    request = _request(_arch(heads=2), rows, used)
    stack.op.reset_mock()
    with torch.no_grad():
        stack(
            request["x"],
            adaln_input=request["adaln_input"],
            combined_indices=request["combined"],
            rope_cache=(request["cos_sin_cache"], torch.arange(rows)),
            cu_seqlens=request["cu_seqlens"],
            cu_seqlens_host=host,
            ulysses_active=True,
        )
    return stack.op.call_args.args


@unittest.skipUnless(_HAS_JAX, "needs JAX (tiling, kernel.loop_structure)")
class TestStaticStructureOption(CustomTestCase):
    """SGLANG_TPU_OPT_H3_MEGAKERNEL_STATIC_STRUCTURE: each call's static
    structure from the host live length, the op that takes it, and the
    generic walk where it does not apply."""

    @classmethod
    def setUpClass(cls):
        cls.config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)

    def test_every_spec_prompt_shares_one_structure_a_shape(self):
        # At each row count every benchmark prompt passes the same static
        # structure: the first live length of their common loop structure.
        from sglang_tpu.multimodal_gen.h3_megakernel import kernel, op

        self.assertEqual(
            op.STRUCTURED_SHAPES, {(devices, rows) for devices, rows, _ in _STRUCTURED}
        )
        useds = sorted({_spec_used(text) for text in _SPEC_TEXT_TOKENS.values()})
        self.assertEqual((useds[0], useds[-1], len(useds)), (37_746, 37_757, 9))
        for devices, rows, _ in _STRUCTURED:
            with self.subTest(devices=devices, rows=rows):

                def structure(used):
                    return kernel.loop_structure(
                        used,
                        rows,
                        devices,
                        self.config,
                        op.structured_tiling(devices, rows),
                    )

                shapes = {
                    hook.static_structure(used, rows, devices, self.config)
                    for used in useds
                }
                self.assertEqual(len(shapes), 1, shapes)
                (shape,) = shapes
                self.assertLessEqual(shape, useds[0])
                self.assertTrue(
                    all(structure(used) == structure(shape) for used in useds)
                )
                self.assertNotEqual(structure(shape - 1), structure(shape))
                self.assertEqual(
                    hook.static_structure(shape, rows, devices, self.config), shape
                )

    def test_static_structure_is_the_first_live_length_of_its_structure(self):
        # Against a linear scan, around the benchmark's live lengths and across
        # query and key tile edges, at each row count; and at 18,944 rows on 2
        # devices, whose 2,368-row key tiles start structures inside a 128-row
        # step (no static structure runs there).
        from sglang_tpu.multimodal_gen.h3_megakernel import kernel, op

        for devices, rows in [(d, r) for d, r, _ in _STRUCTURED] + [(2, 18944)]:
            with self.subTest(devices=devices, rows=rows):
                tiling = op.structured_tiling(devices, rows)

                def structure(used):
                    return kernel.loop_structure(
                        used, rows, devices, self.config, tiling
                    )

                fitted = tiling.fit(self.config, rows, devices)
                edges = (37_633, 2 * fitted.block_q, 3 * fitted.block_kv)
                for used in sorted(
                    {u for edge in edges for u in range(edge - 140, edge + 140)}
                ):
                    first, want = used, structure(used)
                    while first > 1 and structure(first - 1) == want:
                        first -= 1
                    self.assertEqual(
                        hook.static_structure(used, rows, devices, self.config),
                        first,
                        used,
                    )

    def test_the_stack_passes_each_calls_structure(self):
        model = _bare_model()
        stack = _structured_stack(model, self.config, 8)
        with self.assertLogs(hook.logger, logging.INFO) as logs:
            args = _structured_call(stack, 4864, (0, 37_749, 38_912))
            again = _structured_call(stack, 4864, (0, 37_757, 38_912))
        self.assertEqual(len(args), 15)
        self.assertEqual(args[-1], hook.static_structure(37_749, 4864, 8, self.config))
        self.assertEqual(again[-1], args[-1])
        self.assertEqual(args[5].tolist(), [37_749])  # the live length, traced
        # One line a program.
        (line,) = logs.output
        self.assertIn(f"runs the loops of static live length {args[-1]}", line)
        # Without the option the op takes the 14 tensors only.
        plain = _structured_stack(model, self.config, 8, structured=False)
        self.assertEqual(len(_structured_call(plain, 4864, (0, 37_749, 38_912))), 14)

    def test_falls_back_to_the_generic_walk(self):
        model = _bare_model()
        stack = _structured_stack(model, self.config, 8)
        # A row count without a static structure (sglang's 512-row alignment),
        # host bounds that do not match the rows, or none: structure 0.
        for rows, host in (
            (4736, (0, 37_749, 37_888)),
            (4864, (0, 37_749, 39_936)),
            (4864, None),
            (4864, (0, 0, 38_912)),
        ):
            with self.subTest(rows=rows, host=host):
                self.assertEqual(_structured_call(stack, rows, host)[-1], 0)
        # Structure 0 is the generic walk: no static structure in the call.
        from sglang_tpu.multimodal_gen.h3_megakernel import op

        self.assertIsNone(op.stack_kwargs(8, 0)["static_structure"])

    def test_the_structured_call_takes_the_structure_and_narrow_tail(self):
        from sglang_tpu.multimodal_gen.h3_megakernel import op

        # Off: the pinned call, unchanged.
        self.assertIs(op.stack_kwargs(8), op.STACK_KWARGS)
        self.assertIs(op.stack_kwargs(4), op.STACK_KWARGS)
        for devices, structure in ((8, 37_633), (4, 37_633), (2, 0)):
            want = dict(
                op.STACK_KWARGS, narrow_tail=True, static_structure=structure or None
            )
            if devices == 4:  # VMEM headroom at 9,600 rows a device
                want["prefetch_down"] = False
            self.assertEqual(op.stack_kwargs(devices, structure), want)
        names = ("x", "onehot", "cos", "sin", "tables", "used", *pack.WEIGHT_ORDER)
        args = [SimpleNamespace(name=name) for name in names]
        args[5] = MagicMock(name="used")
        with patch.object(op.kernel, "stack", return_value="out") as stack:
            op.local_stack(
                *args,
                config=self.config,
                tiling=op.STACK_TILING,
                axis="sp",
                options=op.stack_kwargs(4, 37_633),
            )
        kwargs = stack.call_args.kwargs
        self.assertEqual(kwargs["static_structure"], 37_633)
        self.assertTrue(kwargs["narrow_tail"])
        self.assertFalse(kwargs["prefetch_down"])
        self.assertEqual(
            op.op_name(self.config, 8, structured=True),
            "sglang_tpu_pallas::minimax_h3_stack_sp8_L50_structured",
        )

    def test_enable_passes_the_option_through(self):
        # The option reaches the op (structured), the stack and the padding.
        arch = _arch(heads=2)
        model = _tiny_model(arch)
        with (
            tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_STATIC_STRUCTURE.override(True),
            _padding_restored(),
            patch.object(hook, "unsupported_reason", return_value=None),
            patch.object(hook, "_ranks_agree", lambda reason: reason),
            patch.object(hook, "_world_size", return_value=2),
            patch.object(hook, "_op_for", return_value=MagicMock()) as op_for,
            patch.object(
                hook,
                "_pack_on_every_rank",
                lambda blocks, config: pack.pack_blocks(
                    blocks, config, stage_on_host=True
                ),
            ),
            self.assertLogs(hook.logger, logging.INFO) as logs,
        ):
            self.assertTrue(hook.enable(model))
            padding = packed_sequence.minimax_h3_padded_seq_len
        self.assertEqual(op_for.call_args.args[1:], (2, True))
        (stack,) = model.blocks
        self.assertTrue(stack.structured)
        self.assertEqual(stack.devices, 2)
        self.assertTrue(padding.keywords["structured"])
        (line,) = [line for line in logs.output if "static loop structure" in line]
        self.assertIn("at 19200, 19456 rows a device", line)
        self.assertIn("19456 rows a device where padded_rows gives 18944", line)

    def test_the_stack_passes_the_structure_to_the_vouched_op(self):
        # Both options: each call hands the vouched op its request's heads to
        # try and tally after the live length, and the static structure last.
        model = _bare_model()
        stack = _structured_stack(model, self.config, 8, vouch_margin=1.1)
        heads = torch.ones((50, 56), dtype=torch.int32)
        tally = torch.tensor([0, 3], dtype=torch.int32)
        stack.op.return_value = ("rows", heads, tally)
        stack.begin_request("r1")
        with self.assertLogs(hook.logger, logging.INFO):
            first = _structured_call(stack, 4864, (0, 37_749, 38_912))
            second = _structured_call(stack, 4864, (0, 37_757, 38_912), used=37_757)
        structure = hook.static_structure(37_749, 4864, 8, self.config)
        self.assertEqual([len(first), len(second)], [17, 17])
        self.assertEqual([first[-1], second[-1]], [structure, structure])
        self.assertEqual([first[5].tolist(), second[5].tolist()], [[37_749], [37_757]])
        # The request's first step tries no heads; the next takes its output.
        self.assertEqual((first[6].shape, first[7].shape), ((50, 56), (2,)))
        self.assertFalse(first[6].any())
        self.assertFalse(first[7].any())
        self.assertIs(second[6], heads)
        self.assertIs(second[7], tally)
        self.assertEqual(stack.end_request(), (0, 3))

    def test_enable_builds_the_vouched_structured_op(self):
        # Both options reach the op (structured and vouched), the stack and
        # the startup lines; the op is kept per option.
        op = op_module()
        vouch = tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH
        with (
            patch.dict(hook._OPS, clear=True),
            patch.object(op, "build_stack_op", side_effect=lambda *a, **k: k) as build,
            vouch.override(True),
        ):
            for structured in (True, True, False):
                self.assertEqual(
                    hook._op_for(self.config, 8, structured),
                    {"structured": structured, "vouch_margin": 1.1},
                )
        self.assertEqual(build.call_count, 2)
        arch = _arch(heads=2)
        model = _tiny_model(arch)
        built = []
        with (
            tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_STATIC_STRUCTURE.override(True),
            vouch.override(True),
            _padding_restored(),
        ):
            logs = TestMegakernelHook._enabled(
                self,
                model,
                lambda config, devices, structured: (
                    built.append(structured) or MagicMock(name="vouched structured op")
                ),
            )
        self.assertEqual(built, [True])
        (stack,) = model.blocks
        self.assertEqual((stack.structured, stack.vouch_margin), (True, 1.1))
        (line,) = [line for line in logs if "megakernel on" in line]
        self.assertIn("by margin 1.1, checked on each step's own peaks", line)
        (line,) = [line for line in logs if "static loop structure" in line]
        self.assertIn("19456 rows a device where padded_rows gives 18944", line)


@unittest.skipUnless(_HAS_JAX, "needs JAX (tiling, kernel.loop_structure)")
class TestKernelLevers(CustomTestCase):
    """The kernel lever flags (SGLANG_TPU_OPT_H3_MEGAKERNEL_LEAN_CUT,
    _SPLIT_ITEMS, _EPILOGUE_CHUNKS, _WIDE_KV, _WIDE_KV_ONLINE,
    _MODULATE_PIECES, _PREFETCH_DOWN, _UNIFORM_ADALN, _ZERO_AHEAD,
    _MASK_SCORES): the structured calls at op.LEVER_SHAPES' row counts take
    them, every other call is unchanged."""

    _ENVS = {
        "lean_cut": tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_LEAN_CUT,
        "split_items": tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_SPLIT_ITEMS,
        "epilogue_chunks": tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_EPILOGUE_CHUNKS,
        "wide_kv": tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_WIDE_KV,
        "wide_kv_online": tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_WIDE_KV_ONLINE,
        "modulate_pieces": tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_MODULATE_PIECES,
        "prefetch_down": tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_PREFETCH_DOWN,
        "uniform_adaln": tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_UNIFORM_ADALN,
        "zero_ahead": tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ZERO_AHEAD,
        "mask_scores": tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_MASK_SCORES,
    }

    @classmethod
    def setUpClass(cls):
        cls.config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)

    def test_each_lever_takes_only_its_shapes(self):
        # 9,600 rows on 4 devices take split_items and lean_cut (their cut
        # tile's second 1,280-key sub-block runs 640 keys); their 640-row
        # phase A tiles already take 64-row epilogue chunks; wide_kv_online
        # (not wide_kv) puts eight 640-key online sub-blocks in a key step.
        # modulate_pieces: one a chunk of the phase A tile (608 rows: 19 of
        # 32; 640: 8 of 80; 480: 6 of 80).
        import dataclasses

        op = op_module()
        every = frozenset(op.LEVER_SHAPES)
        wants = {
            (8, 4992): {},
            (8, 4864): {
                "cut_kv_compute": 256,
                "split_items": True,
                "epilogue_chunk": 64,
                "kv_group": 2,
                "modulate_pieces": 19,
                "uniform_adaln": True,
                "zero_ahead": True,
                "mask_scores": True,
            },
            # prefetch_down: back on at 4 devices (stack_kwargs' default off);
            # wide_kv_online groups the online key steps only.
            (4, 9600): {
                "split_items": True,
                "kv_group": (1, 2),
                "cut_kv_compute": 256,
                "modulate_pieces": 8,
                "prefetch_down": True,
                "uniform_adaln": True,
                "zero_ahead": True,
                "mask_scores": True,
            },
            (2, 19456): {
                "cut_kv_compute": 256,
                "split_items": True,
                "epilogue_chunk": 64,
                "kv_group": 2,
                "modulate_pieces": 19,
                "uniform_adaln": True,
                "zero_ahead": True,
                "mask_scores": True,
            },
            # 640-row phase tiles: no epilogue chunks.
            (2, 19200): {
                "cut_kv_compute": 256,
                "split_items": True,
                "kv_group": 2,
                "modulate_pieces": 8,
                "uniform_adaln": True,
                "zero_ahead": True,
                "mask_scores": True,
            },
            # 4 x 9,600's attention tiles take its levers; 480-row phase A
            # tiles take 64-row epilogue chunks.
            (8, 4800): {
                "split_items": True,
                "cut_kv_compute": 256,
                "epilogue_chunk": 64,
                "modulate_pieces": 6,
                "uniform_adaln": True,
                "zero_ahead": True,
                "mask_scores": True,
            },
        }
        for devices, rows, _ in _STRUCTURED:
            with self.subTest(devices=devices, rows=rows):
                plain = op.stack_kwargs(devices, 37_633)
                got = op.stack_kwargs(devices, 37_633, rows=rows, levers=every)
                extra = {k: v for k, v in got.items() if plain.get(k, "-") != v}
                self.assertEqual(extra, wants[(devices, rows)])
                self.assertEqual(op.stack_kwargs(devices, 37_633, rows=rows), plain)
                self.assertEqual(
                    op.cut_kv_compute(devices, rows, every),
                    wants[(devices, rows)].get("cut_kv_compute", 0),
                )
        # modulate_pieces follows the call's own tiling (the TILING knob's).
        self.assertEqual(
            op.stack_kwargs(
                4,
                37_633,
                rows=9600,
                levers={"modulate_pieces"},
                tiling=dataclasses.replace(op.STACK_TILING, chunk_rows=160),
            )["modulate_pieces"],
            4,
        )
        # wide_kv_online alone groups the online softmax's key tiles only.
        online = op.stack_kwargs(2, 37_633, rows=19200, levers={"wide_kv_online"})
        self.assertEqual(online["kv_group"], (1, 2))
        self.assertNotIn(
            "kv_group", op.stack_kwargs(8, 37_633, rows=4800, levers=every)
        )
        # The generic walk takes none, inside the structured op too.
        self.assertIs(op.stack_kwargs(8, rows=4864, levers=every), op.STACK_KWARGS)
        self.assertEqual(
            op.stack_kwargs(8, 0, rows=4864, levers=every), op.stack_kwargs(8, 0)
        )
        self.assertEqual(
            op.op_name(
                self.config, 8, structured=True, levers={"split_items", "lean_cut"}
            ),
            "sglang_tpu_pallas::minimax_h3_stack_sp8_L50_structured_lean_cut_split_items",
        )

    def test_19200_rows_take_the_9600_tiles_and_the_19456_key_tiles(self):
        # 2 x 19,200 rows (ROWS=19200): the structured calls cut 4 x 9,600's
        # query and phase tiles and 2 x 19,456's key tiles and sub-blocks;
        # the generic walk keeps the op's tiling (2,560-key tiles dividing
        # the rows), as does every other shape.
        op = op_module()
        tiling = op.structured_tiling(2, 19200)
        got = tiling.fit(self.config, 19200, 2)
        quarter = op.STACK_TILING.fit(self.config, 9600, 4)
        half = op.STACK_TILING.fit(self.config, 19456, 2)
        for name in ("block_q", "qkv_rows", "mlp_rows", "chunk_rows", "epilogue_rows"):
            self.assertEqual(getattr(got, name), getattr(quarter, name), name)
        for name in ("block_kv", "block_kv_compute", "block_kv_compute_online"):
            self.assertEqual(getattr(got, name), getattr(half, name), name)
        self.assertEqual((got.block_q, got.block_kv, got.qkv_rows), (3200, 2048, 640))
        self.assertEqual(op.STACK_TILING.fit(self.config, 19200, 2).block_kv, 2560)
        for devices, rows, _ in _STRUCTURED:
            if (devices, rows) not in ((2, 19200), (8, 4800)):
                self.assertIs(op.structured_tiling(devices, rows), op.STACK_TILING)

    def test_4800_rows_take_the_9600_attention_tiles(self):
        # 8 x 4,800 rows (ROWS=4800): every call, the generic walk too,
        # cuts 4 x 9,600's query and key tiles, the query tiles across device
        # boundaries (no 128-row tile divides 4,800), and 480-row phase
        # tiles; the spec prompts' structure is 4 x 9,600's.
        from sglang_tpu.multimodal_gen.h3_megakernel import kernel

        op = op_module()
        tiling = op.shape_tiling(8, 4800)
        self.assertIs(op.structured_tiling(8, 4800), tiling)
        got = tiling.fit(self.config, 4800, 8)
        quarter = op.STACK_TILING.fit(self.config, 9600, 4)
        for name in (
            "block_q",
            "block_kv",
            "block_kv_compute",
            "block_kv_compute_online",
        ):
            self.assertEqual(getattr(got, name), getattr(quarter, name), name)
        self.assertEqual((got.qkv_rows, got.mlp_rows), (480, 480))
        with self.assertRaisesRegex(ValueError, "multiple of 128 rows"):
            op.STACK_TILING.fit(self.config, 4800, 8)
        self.assertEqual(
            kernel.loop_structure(37_633, 4800, 8, self.config, tiling),
            kernel.loop_structure(37_633, 9600, 4, self.config, op.STACK_TILING),
        )

    @unittest.skipUnless(_HAS_TORCH_TPU, "needs torch_tpu's Pallas adapter")
    def test_every_op_cuts_4800_rows_with_their_tiling(self):
        # The plain and the vouched op pick each call's tiling by its rows:
        # op.SHAPE_TILINGS' at 4,800 rows a device, the op's elsewhere.
        from jax.sharding import AbstractMesh

        op = op_module()
        for margin in (None, 1.1):
            with self.subTest(margin=margin):
                captured = _capture_op(
                    self.config, 8, AbstractMesh((8,), ("sp",)), vouch_margin=margin
                )
                for rows, want in (
                    (4800, op.shape_tiling(8, 4800)),
                    (4864, op.STACK_TILING),
                ):
                    (kwargs,) = _kernel_calls(captured, self.config, 8, rows=rows)
                    self.assertEqual(kwargs["tiling"], want)

    def test_tiling_bounds_reach_the_structured_op_and_its_structure(self):
        # SGLANG_TPU_OPT_H3_MEGAKERNEL_TILING: parsed bounds over the
        # structured calls' own tiling, in the op (a "_tiled" name) and in
        # each call's static structure; a bad value refuses the kernel.
        op = op_module()
        self.assertEqual(
            op.parse_tiling("block_q=4864+block_kv_compute=1024, split_q_tiles=1"),
            (("block_kv_compute", 1024), ("block_q", 4864), ("split_q_tiles", True)),
        )
        self.assertEqual(op.parse_tiling(None), ())
        for bad in ("block_q", "vmem_limit_bytes=1", "block_q=1.5"):
            with self.assertRaisesRegex(ValueError, "tiling bound"):
                op.parse_tiling(bad)
        bounds = (("block_kv_compute", 1024), ("block_q", 4864))
        tiling = op.structured_tiling(8, 4864, overrides=bounds)
        self.assertEqual((tiling.block_q, tiling.block_kv_compute), (4864, 1024))
        knob = tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_TILING
        with (
            patch.dict(hook._OPS, clear=True),
            patch.object(op, "build_stack_op", side_effect=lambda *a, **k: k),
            knob.override("block_q=4864+block_kv_compute=1024"),
        ):
            self.assertEqual(
                hook._op_for(self.config, 8, True),
                {"structured": True, "vouch_margin": None, "tiling_overrides": bounds},
            )
            self.assertEqual(
                hook._op_for(self.config, 8, False),
                {"structured": False, "vouch_margin": None},
            )
            stack = _structured_stack(_bare_model(), self.config, 8)
            self.assertEqual(stack.tiling_overrides, bounds)
            with (
                patch.object(hook, "static_structure", return_value=37_633) as shape,
                self.assertLogs(hook.logger, logging.INFO),
            ):
                _structured_call(stack, 4864, (0, 37_749, 38_912))
        self.assertEqual(shape.call_args.kwargs["tiling"], tiling)
        self.assertEqual(
            op.op_name(self.config, 8, structured=True, tiled=True),
            "sglang_tpu_pallas::minimax_h3_stack_sp8_L50_structured_tiled",
        )

    def test_lean_cut_keeps_every_spec_prompt_on_one_structure(self):
        # lean_cut's halving boundary (1,024 keys into a 2,048-key tile, or
        # 1,920 into a 2,560-key one of 1,280-key sub-blocks) is a multiple of
        # 128 rows, so the structure classes are the plain ones: every
        # benchmark prompt still passes one static structure, whose cut tile's last
        # real sub-block runs half its keys.
        from sglang_tpu.multimodal_gen.h3_megakernel import kernel
        from sglang_tpu.multimodal_gen.h3_megakernel import tiling as tl

        op = op_module()
        useds = sorted({_spec_used(text) for text in _SPEC_TEXT_TOKENS.values()})
        for devices, rows in sorted(op.LEVER_SHAPES["lean_cut"]):
            with self.subTest(devices=devices, rows=rows):
                cut = op.cut_kv_compute(devices, rows, {"lean_cut"})
                plain = {
                    hook.static_structure(u, rows, devices, self.config) for u in useds
                }
                leaned = {
                    hook.static_structure(
                        u, rows, devices, self.config, cut_kv_compute=cut
                    )
                    for u in useds
                }
                self.assertEqual(leaned, plain)
                (shape,) = leaned
                tiling = op.structured_tiling(devices, rows)
                structure = kernel.loop_structure(
                    shape, rows, devices, self.config, tiling, cut_kv_compute=cut
                )
                fitted = tiling.fit(self.config, rows, devices)
                sub = tl.divisor(fitted.block_kv, fitted.block_kv_compute, 16)
                self.assertEqual(structure[-1], sub // 2)
                # A live length past the boundary is another structure.
                start = (shape - 1) // fitted.block_kv * fitted.block_kv
                edge = start + (shape - start - 1) // sub * sub + sub // 2
                self.assertNotEqual(
                    kernel.loop_structure(
                        edge + 1,
                        rows,
                        devices,
                        self.config,
                        tiling,
                        cut_kv_compute=cut,
                    )[-1],
                    sub // 2,
                )

    def test_the_flags_reach_the_structured_op_and_its_structure(self):
        op = op_module()
        with (
            patch.dict(hook._OPS, clear=True),
            patch.object(op, "build_stack_op", side_effect=lambda *a, **k: k),
            self._ENVS["lean_cut"].override(True),
            self._ENVS["split_items"].override(True),
        ):
            self.assertEqual(
                hook._op_for(self.config, 8, True),
                {
                    "structured": True,
                    "vouch_margin": None,
                    "levers": frozenset({"lean_cut", "split_items"}),
                },
            )
            # The generic walk takes no lever.
            self.assertEqual(
                hook._op_for(self.config, 8, False),
                {"structured": False, "vouch_margin": None},
            )
            model = _bare_model()
            stack = _structured_stack(model, self.config, 8)
            self.assertEqual(stack.levers, {"lean_cut", "split_items"})
            with (
                patch.object(hook, "static_structure", return_value=37_633) as shape,
                self.assertLogs(hook.logger, logging.INFO),
            ):
                _structured_call(stack, 4864, (0, 37_749, 38_912))
                _structured_call(stack, 4992, (0, 37_749, 39_936))
        cuts = [call.kwargs["cut_kv_compute"] for call in shape.call_args_list]
        self.assertEqual(cuts, [256, 0])
        # Without the flags: no lever.
        self.assertEqual(_structured_stack(_bare_model(), self.config, 8).levers, set())

    @unittest.skipUnless(_HAS_TORCH_TPU, "needs torch_tpu's Pallas adapter")
    def test_each_lever_reaches_the_kernel_call_at_its_shape(self):
        # The structured and the vouched structured op, traced with the
        # kernel stubbed at each row count.
        from jax.sharding import AbstractMesh

        op = op_module()
        every = frozenset(op.LEVER_SHAPES)
        for devices, rows in ((8, 4864), (4, 9600), (2, 19456)):
            for margin in (None, 1.1):
                with self.subTest(devices=devices, rows=rows, margin=margin):
                    captured = _capture_op(
                        self.config,
                        devices,
                        AbstractMesh((devices,), ("sp",)),
                        structured=True,
                        vouch_margin=margin,
                        levers=every,
                    )
                    self.assertTrue(
                        captured["name"].endswith(
                            "_epilogue_chunks_lean_cut_mask_scores"
                            "_modulate_pieces_prefetch_down_split_items"
                            "_uniform_adaln_wide_kv_wide_kv_online_zero_ahead"
                        )
                    )
                    (kwargs,) = _kernel_calls(
                        captured, self.config, devices, 37_633, rows=rows
                    )
                    want = op.stack_kwargs(devices, 37_633, rows=rows, levers=every)
                    for name in (
                        "cut_kv_compute",
                        "split_items",
                        "epilogue_chunk",
                        "kv_group",
                        "modulate_pieces",
                        "prefetch_down",
                        "uniform_adaln",
                        "zero_ahead",
                        "mask_scores",
                    ):
                        self.assertEqual(kwargs.get(name), want.get(name), name)
                    self.assertTrue(kwargs["split_items"])


def _checkpoint_tensors(names) -> dict:
    """Checkpoint name -> tensor, opening only the files that hold them."""
    from safetensors import safe_open

    with open(os.path.join(_TRANSFORMER, "model.safetensors.index.json")) as f:
        where = json.load(f)["weight_map"]
    by_file: dict = {}
    for name in names:
        by_file.setdefault(where[name], []).append(name)
    tensors = {}
    for file, file_names in sorted(by_file.items()):
        with safe_open(os.path.join(_TRANSFORMER, file), "pt") as f:
            for name in file_names:
                tensors[name] = f.get_tensor(name)
    return tensors


def _block_names(layer: int) -> list[str]:
    return [
        f"blocks.{layer}.{suffix}"
        for suffix in (
            "norm1.weight",
            "norm2.weight",
            "attn.qkv_proj.weight",
            "attn.out_proj.weight",
            "attn.q_norm.weight",
            "attn.k_norm.weight",
            "mlp.fc1.weight",
            "mlp.fc2.weight",
            "adaln_proj.linear.weight",
            "adaln_proj.linear.bias",
        )
    ]


def _real_blocks(layers=_REAL_LAYERS, keep_raw: bool = True):
    """sglang's ``MiniMaxH3DiTBlock``s loaded through sglang's own weight
    loaders (the checkpoint's per-head q|k|v rows become [Q | K | V]), and
    (``keep_raw``) the raw checkpoint tensors by block-relative name, without
    the AdaLN projections."""
    _init_parallel()
    arch = MiniMaxH3DiTArchConfig()
    blocks, checkpoints = [], []
    for layer in layers:
        raw = _checkpoint_tensors(_block_names(layer))
        block = dit.MiniMaxH3DiTBlock(arch, None, prefix=f"blocks.{layer}")
        block.requires_grad_(False)
        params = dict(block.named_parameters())
        prefix = f"blocks.{layer}."
        checkpoint = {}
        for name in _block_names(layer):
            local = name[len(prefix) :]
            tensor = raw.pop(name)
            if keep_raw and not local.startswith("adaln_proj"):
                checkpoint[local] = tensor
            loader = getattr(params[local], "weight_loader", None)
            if loader is None:
                params[local].data.copy_(tensor)
            else:
                loader(params[local], tensor)
            del tensor
        blocks.append(block)
        checkpoints.append(checkpoint)
    return arch, blocks, checkpoints


def _bf16_numpy(tensor: torch.Tensor):
    """A bf16 tensor's values as an ml_dtypes bfloat16 array (no copy)."""
    import ml_dtypes

    return tensor.contiguous().view(torch.int16).numpy().view(ml_dtypes.bfloat16)


def _real_time_embedder(arch):
    embedder = dit.MiniMaxH3TimeEmbedder(arch, prefix="time_embedder")
    names = {
        f"time_embedder.{local}": local for local, _ in embedder.named_parameters()
    }
    tensors = _checkpoint_tensors(list(names))
    for name, local in names.items():
        embedder.get_parameter(local).data.copy_(tensors[name])
    return embedder.requires_grad_(False)


def _real_adaln_input(arch, timesteps=(0.27, 0.6)) -> torch.Tensor:
    """``[M, Dt]`` bf16 AdaLN input as the DiT forms it (SiLU of the real time
    embedding, cast to bf16) for these unique timesteps."""
    with torch.no_grad():
        t_emb = _real_time_embedder(arch)(torch.tensor(timesteps))
    return torch.nn.functional.silu(t_emb).to(_BF16)


def _real_positions(text_len=41, latent_t=4, latent_h=24, latent_w=32, audio_t=40):
    """A t2va packed layout's ``img_position_ids`` and its live length: 889
    rows (text 41, audio 80, video 768), padded to 1024 by the caller."""
    packed = packed_sequence.minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        include_keyframe_cond=False,
    )
    return packed


def _real_inv_freq() -> torch.Tensor:
    return _checkpoint_tensors(["rope.inv_freq"])["rope.inv_freq"].float()


def _bf16_bits(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().contiguous().cpu().view(torch.int16).numpy().view(np.uint16)


@unittest.skipUnless(_HAS_JAX, "needs JAX")
class TestProvenLayers(CustomTestCase):
    def test_constants_and_random_norms_match_the_kernel(self):
        import jax.numpy as jnp

        from sglang_tpu.multimodal_gen.h3_megakernel import kernel
        from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref

        self.assertEqual(hook._FIXED_BOUND, kernel._FIXED_BOUND)
        self.assertEqual(hook._BOUND_SLACK, kernel._BOUND_SLACK)
        self.assertEqual(op_module().STACK_KWARGS["fixed_max"], kernel._FIXED_M)
        config = ref.Config(layers=64)
        generator = torch.Generator().manual_seed(5)
        # Scales that put the bounds on both sides of the fixed-max limit,
        # with one large pair (q on dim d, k on dim d + 48) or one large
        # unrotated dim in some layers.
        q = torch.rand(64, 1, 128, generator=generator) * 1.6
        k = torch.rand(64, 1, 128, generator=generator) * 1.6
        scale = torch.linspace(0.5, 1.8, 64).view(64, 1, 1)
        q, k = q * scale, k * scale
        q[5, 0, 3], k[5, 0, 51] = 9.0, 9.0
        q[6, 0, 100], k[6, 0, 100] = 7.0, 7.0
        q[7, 0, 3], k[7, 0, 3 + 96 - 48] = -9.0, 9.0
        # A bound of 107.2: under 109 alone, over it with the rounding slack.
        q[8], k[8] = 2.5625, 2.5625
        q, k = q.to(_BF16), k.to(_BF16)
        want = np.asarray(
            kernel.fixed_max_layers(
                {
                    "q_norm": jnp.asarray(q.float().numpy(), jnp.bfloat16),
                    "k_norm": jnp.asarray(k.float().numpy(), jnp.bfloat16),
                },
                config,
            )
        )
        got = hook.proven_layers(q, k, config)
        self.assertEqual(got, np.flatnonzero(want).tolist())
        self.assertTrue(0 < len(got) < 64)
        for layer in (5, 6, 7, 8):
            self.assertNotIn(layer, got)

    @unittest.skipUnless(_HAS_CKPT, "needs the FL2VA checkpoint")
    def test_fl2va_norms_prove_41_layers(self):
        import jax.numpy as jnp

        from sglang_tpu.multimodal_gen.h3_megakernel import kernel

        names = [
            f"blocks.{layer}.attn.{norm}.weight"
            for layer in range(50)
            for norm in ("q_norm", "k_norm")
        ]
        tensors = _checkpoint_tensors(names)
        q, k = (
            torch.stack(
                [tensors[f"blocks.{layer}.attn.{norm}.weight"] for layer in range(50)]
            ).view(50, 1, 128)
            for norm in ("q_norm", "k_norm")
        )
        config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)
        got = hook.proven_layers(q, k, config)
        want = kernel.fixed_max_layers(
            {
                "q_norm": jnp.asarray(q.float().numpy(), jnp.bfloat16),
                "k_norm": jnp.asarray(k.float().numpy(), jnp.bfloat16),
            },
            config,
        )
        self.assertEqual(got, np.flatnonzero(np.asarray(want)).tolist())
        self.assertEqual(
            sorted(set(range(50)) - set(got)), _FL2VA_ONLINE, "online layers"
        )


def op_module():
    from sglang_tpu.multimodal_gen.h3_megakernel import op

    return op


@unittest.skipUnless(_HAS_JAX, "needs JAX")
class TestOpCall(CustomTestCase):
    def test_the_kernel_call_is_pinned(self):
        # Proven-only fixed max (the kernel's own norm bound; nothing vouched,
        # no flags or calibration), bf16, no sparse/windowed/far attention,
        # no peaks, the default fixed max; the live length as data.
        op = op_module()
        config = pack.config_from_arch(_arch(heads=2), 2)
        names = ("x", "onehot", "cos", "sin", "tables", "used", *pack.WEIGHT_ORDER)
        args = [SimpleNamespace(name=name) for name in names]
        args[5] = MagicMock(name="used")
        with patch.object(op.kernel, "stack", return_value="out") as stack:
            out = op.local_stack(
                *args, config=config, tiling=op.STACK_TILING, axis="sp"
            )
        self.assertEqual(out, "out")
        (x, weights, tables, onehot, cos, sin, used), kwargs = stack.call_args
        self.assertEqual(
            [x.name, tables.name, onehot.name, cos.name, sin.name],
            ["x", "tables", "onehot", "cos", "sin"],
        )
        self.assertEqual(list(weights), list(pack.WEIGHT_ORDER))
        self.assertEqual([w.name for w in weights.values()], list(pack.WEIGHT_ORDER))
        self.assertIs(used, args[5].__getitem__.return_value)
        args[5].__getitem__.assert_called_once_with(0)
        self.assertEqual(
            kwargs,
            dict(
                config=config,
                tiling=op.STACK_TILING,
                axis="sp",
                interpret=False,
                fixed=None,
                fixed_heads=None,
                head_perm=None,
                return_peaks=False,
                fp8=False,
                sparse=None,
                window=None,
                far=None,
                far_mode=None,
                tile_masks=None,
                listed=None,
                narrow_tail=False,
                scaled_v=False,
                fast_phase_c=False,
                fixed_max=17.0,
            ),
        )
        # The tiling: the padding bounds plus the tuned MLP tiling (the
        # call), the weights in the kernel's order.
        self.assertEqual(
            (op.ROWS_TILING.sub_rows, op.ROWS_TILING.stage_slots),
            (640, 24),
        )
        self.assertEqual(op.ROWS_TILING.block_kv_compute_online, 1024)
        self.assertEqual(
            (op.STACK_TILING.mlp_rows, op.STACK_TILING.chunk_rows), (640, 80)
        )
        self.assertEqual(op.STACK_TILING.qkv_rows, 704)

    def test_1344x768x124_tiles_match_the_tuned_tiling(self):
        # At every 1344x768x124 row count the call's tiles are those of the
        # tuned tiling (padding bounds + MLP tiling); the 704-row QKV
        # bound only keeps 768-row tiles (over VMEM with 24 staging slots)
        # off other shapes.
        import dataclasses

        op = op_module()
        config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)
        tuned = dataclasses.replace(op.ROWS_TILING, mlp_rows=640, chunk_rows=80)
        for rows, devices in ((4992, 8), (4864, 8), (9600, 4), (18944, 2), (19456, 2)):
            self.assertEqual(
                op.STACK_TILING.fit(config, rows, devices),
                tuned.fit(config, rows, devices),
                (rows, devices),
            )
        self.assertEqual(tuned.fit(config, 2304, 8).qkv_rows, 768)
        self.assertEqual(op.STACK_TILING.fit(config, 2304, 8).qkv_rows, 576)
        self.assertEqual(tuple(op.kernel._WEIGHT_ORDER), pack.WEIGHT_ORDER)
        self.assertEqual(pack.TABLE_LANES, op.ref.TABLE_LANES)
        self.assertEqual(pack.TABLE_ROW_TILE, op.ref._TABLE_ROW_TILE)


@unittest.skipUnless(_HAS_JAX and _HAS_CKPT, "needs JAX and the FL2VA checkpoint")
class TestRealBlocks(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.arch, cls.blocks, cls.checkpoints = _real_blocks()
        cls.config = pack.config_from_arch(cls.arch, len(cls.blocks))

    def test_tables_equal_each_blocks_adaln_proj_bitwise(self):
        adaln_input = _real_adaln_input(self.arch)
        stack = hook.H3MegakernelStack(
            _bare_model(),
            [block.adaln_proj for block in self.blocks],
            {name: torch.zeros(1) for name in pack.WEIGHT_ORDER},
            None,
            self.config,
        )
        combined = torch.zeros(128, dtype=torch.long)
        cache = torch.zeros(128, 96, dtype=_BF16)
        cu = torch.tensor([0, 100, 128], dtype=torch.int32)
        with torch.no_grad():
            _, _, _, tables, _ = stack.operands(adaln_input, combined, cache, cu)
            self.assertEqual(tables.shape, (2, 6, 16, 5376))
            rows = 3 * adaln_input.shape[0]
            for layer, block in enumerate(self.blocks):
                for k, chunk in enumerate(block.adaln_proj(adaln_input)):
                    self.assertTrue(
                        np.array_equal(
                            _bf16_bits(tables[layer, k, :rows]), _bf16_bits(chunk)
                        ),
                        (layer, k),
                    )
            self.assertFalse(tables[:, :, rows:].any())
            # A real projection, not zeros: each table row has signal.
            self.assertTrue((tables[:, :, :rows].float().abs().amax(-1) > 0).all())

    def test_rope_tables_reproduce_sglang_rope_on_real_positions(self):
        from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref

        packed = _real_positions()
        rope = dit.MiniMaxH3Rope(self.arch.rope_inv_freq_len)
        rope.inv_freq.copy_(_real_inv_freq())
        positions = packed["img_position_ids"][None].to(torch.float32)
        cache = dit._rope_cos_sin_cache(rope(positions), dtype=_BF16)
        cos, sin = pack.rope_tables(cache, 128)
        generator = torch.Generator().manual_seed(2)
        tokens = cache.shape[0]
        q = (torch.randn(tokens, 4, 128, generator=generator) * 3).to(_BF16)
        k = (torch.randn(tokens, 4, 128, generator=generator) * 3).to(_BF16)
        want_q, want_k = dit._apply_rope_qk(q, k, cache, torch.arange(tokens))
        for x, want in ((q, want_q), (k, want_k)):
            torch.testing.assert_close(
                _kernel_rope(x, cos, sin, 96), want, rtol=0, atol=0
            )
        # The reference's own tables (cos/sin in XLA) agree to a bf16 ulp;
        # the kernel gets sglang's values.
        ref_cos, ref_sin = ref.rope_tables(
            positions[0].numpy(), _real_inv_freq().numpy(), self.config
        )
        for got, theirs in ((cos, ref_cos), (sin, ref_sin)):
            diff = np.abs(got.float().numpy() - np.asarray(theirs, np.float32))
            self.assertLessEqual(float(diff.max()), 2.0**-8)

    def test_packer_matches_the_checkpoint_layout_bitwise(self):
        from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref

        # The class's blocks themselves (the other tests need only their
        # AdaLN projections, which packing keeps).
        blocks = self.blocks
        layer_config = pack.config_from_arch(self.arch, 1)
        # Staged through host memory as on TPU (a no-op move on CPU).
        got = pack.pack_blocks(blocks, self.config, stage_on_host=True)
        self.assertEqual(sorted(got), sorted(pack.WEIGHT_ORDER))
        self.assertEqual(
            pack.tile_widths(self.config),
            {"qkv_heads": 2, "out_cols": 768, "ffn_cols": 256, "down_cols": 256},
        )
        for layer, checkpoint in enumerate(self.checkpoints):
            raw = {name: _bf16_numpy(tensor) for name, tensor in checkpoint.items()}
            want = ref.pack_weights([raw], layer_config)
            for name in pack.WEIGHT_ORDER:
                self.assertEqual(
                    tuple(got[name].shape[1:]), tuple(want[name].shape[1:]), name
                )
                theirs = np.asarray(want[name][0]).view(np.uint16)
                self.assertTrue(
                    np.array_equal(_bf16_bits(got[name][layer]), theirs),
                    (layer, name),
                )
            del want, raw
        # The projections are freed; the norms and AdaLN projections stay.
        for block in blocks:
            for module in hook._projections(block):
                self.assertEqual(module.weight.numel(), 0)
            self.assertEqual(block.norm1.weight.numel(), self.arch.hidden_size)
            self.assertEqual(tuple(block.adaln_proj.linear.weight.shape), (96768, 2688))


# ------------------------------------------------- the op in interpret mode

# Small-model kernel tests: rows a device and tile bounds that give two key
# tiles a query tile and two compute steps a key tile.
_SMALL_ROWS = 128
_SMALL_TILING = dict(
    qkv_rows=64,
    mlp_rows=64,
    block_q=128,
    block_kv=64,
    block_kv_compute=32,
    sub_rows=32,
    n_chunk=128,
    chunk_rows=16,
    epilogue_rows=32,
    stage_slots=3,
)
# The bf16 floor: the kernel's error against sglang's fp32 blocks may be at
# most this much over sglang's own bf16 blocks' error on the same inputs.
_FLOOR_RATIO = 1.25
# A planted layout bug (RoPE turned the other way, gate and up swapped) is
# far over the floor.
_BUG_RATIO = 10.0


def _to_jax(tensor: torch.Tensor):
    import jax.numpy as jnp

    if tensor.dtype == _BF16:
        return jnp.asarray(_bf16_numpy(tensor))
    return jnp.asarray(tensor.numpy())


def _run_op(mesh, config, tiling, args, races: bool):
    """The op's JAX function (op.sharded_stack) on ``args``, interpreted."""
    import jax
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.multimodal_gen.h3_megakernel import op

    interpret = pltpu.InterpretParams(
        detect_races=races, dma_execution_mode="eager" if races else "on_wait"
    )
    fn = op.sharded_stack(mesh, config, tiling, interpret)
    return np.asarray(jax.jit(fn)(*args), np.float32)


def _small_stack_errors(devices: int, races: bool, bugs: list[str]) -> dict:
    """Kernel vs sglang's blocks on a small model, one head a device: layer 0
    at a fixed max, layer 1 online; the live length cuts a key tile and
    leaves a padding tile. ``bugs`` adds the error of each planted bug."""
    import jax
    from jax.sharding import Mesh

    from sglang_tpu.multimodal_gen.h3_megakernel import kernel
    from sglang_tpu.multimodal_gen.h3_megakernel import tiling as tl

    arch = _arch(heads=devices)
    blocks = _blocks(arch, 2, norm_scales=(1.0, 3.0))
    config = pack.config_from_arch(arch, 2)
    tokens = _SMALL_ROWS * devices
    used = tokens - _SMALL_ROWS - 37
    request = _request(arch, tokens, used)
    truth = _eager_stack(blocks, request, arch, fp32=True)[:used].numpy()
    served = _eager_stack(blocks, request, arch, fp32=False)[:used].numpy()
    stack = hook.H3MegakernelStack(
        _bare_model(),
        [block.adaln_proj for block in blocks],
        pack.pack_blocks(blocks, config, free_originals=False),
        None,
        config,
    )
    with torch.no_grad():
        onehot, cos, sin, tables, used_t = stack.operands(
            request["adaln_input"],
            request["combined"],
            request["cos_sin_cache"],
            request["cu_seqlens"],
        )
    weights = {name: getattr(stack, name) for name in pack.WEIGHT_ORDER}
    mesh = Mesh(np.asarray(jax.devices()[:devices]), ("sp",))
    tiling = tl.Tiling(**_SMALL_TILING)

    def errors(bug=None):
        w, s = dict(weights), sin
        if bug == "rope":
            s = -sin
        elif bug == "gate_up":
            half = w["w13"].shape[-1] // 2
            w["w13"] = w["w13"].roll(half, -1)
        args = (request["x"], onehot, cos, s, tables, used_t)
        args += tuple(w[name] for name in pack.WEIGHT_ORDER)
        got = _run_op(mesh, config, tiling, tuple(map(_to_jax, args)), races)[:used]
        return got

    got = errors()
    fixed = kernel.fixed_max_layers(
        {name: _to_jax(weights[name]) for name in ("q_norm", "k_norm")}, config
    )
    return {
        "fixed": [bool(f) for f in np.asarray(fixed)],
        "proven": hook.proven_layers(weights["q_norm"], weights["k_norm"], config),
        "finite": bool(np.isfinite(got).all()),
        "update": _rel(request["x"][:used].float().numpy(), truth),
        "floor": _rel(served, truth),
        "kernel": _rel(got, truth),
        "bugs": {bug: _rel(errors(bug), truth) for bug in bugs},
    }


def _run_errors(args: list[str], devices: int, timeout: int = 3600) -> dict:
    """This file's ``args`` mode in a fresh process with ``devices`` CPU
    devices; its last stdout line, as JSON."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = dict(
        os.environ,
        XLA_FLAGS=f"--xla_force_host_platform_device_count={devices}",
        JAX_PLATFORMS="cpu",
        # This process may already hold the default rendezvous port.
        MASTER_PORT=str(port),
    )
    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr[-4000:])
    if "RACE DETECTED" in result.stdout:
        raise AssertionError("race detected:\n" + result.stdout[-4000:])
    return json.loads(result.stdout.splitlines()[-1])


@unittest.skipUnless(_HAS_JAX, "needs JAX")
class TestMegakernelStackSmall(CustomTestCase):
    def test_matches_sglang_blocks_across_four_devices(self):
        # Four devices, one head each: every two-head QKV tile goes to two
        # owners, the padding starts on device 2, device 3 is all padding.
        # The race detector watches every DMA.
        errors = _run_errors(["--small-errors", "4", "1", ""], devices=4)
        self.assertEqual(errors["fixed"], [True, False])
        self.assertEqual(errors["proven"], [0])
        self.assertTrue(errors["finite"])
        self.assertGreater(errors["update"], 0.3)
        self.assertLess(errors["kernel"], _FLOOR_RATIO * errors["floor"])

    def test_layout_bugs_fail(self):
        # Negative controls: the comparison above sees each planted bug.
        errors = _run_errors(["--small-errors", "2", "0", "rope,gate_up"], devices=2)
        self.assertLess(errors["kernel"], _FLOOR_RATIO * errors["floor"])
        self.assertEqual(sorted(errors["bugs"]), ["gate_up", "rope"])
        for bug, error in errors["bugs"].items():
            self.assertGreater(error, _BUG_RATIO * errors["floor"], bug)


# Two real blocks on 4 devices: 889 live rows (text 41, audio 80, video 768),
# padded by the hook's rule to 1,024 (256 a device), timesteps 0.27 (audio)
# and 0.6 (video).
_REAL_DEVICES = 4
_REAL_TIMESTEPS = (0.27, 0.6)


def _real_sglang_side(directory: str) -> dict:
    """Stage 1: two real blocks through sglang's eager path (bf16 as served,
    and fp32), and the hook's operands for the same inputs, saved for the
    kernel side."""
    arch, blocks, _ = _real_blocks(keep_raw=False)
    config = pack.config_from_arch(arch, len(blocks))
    with _padding_restored():
        hook._use_kernel_padding(_REAL_DEVICES, config)
        packed = _real_positions()
    tokens, used = int(packed["seq_len"]), int(packed["cu_seqlens"][1])
    assert (tokens, used) == (1024, 889), (tokens, used)
    rope = dit.MiniMaxH3Rope(arch.rope_inv_freq_len)
    rope.inv_freq.copy_(_real_inv_freq())
    freqs = rope(packed["img_position_ids"][None].to(torch.float32))
    with torch.no_grad():
        t_emb = _real_time_embedder(arch)(torch.tensor(_REAL_TIMESTEPS))
    adaln_input = torch.nn.functional.silu(t_emb)
    # Rows' AdaLN index as the denoise loop forms it: tag (video 0, text 1,
    # audio 2; padding 0) + 3 x its timestep's index (audio 0, others 1).
    tags = packed["token_tags"]
    inverse = torch.ones(tokens, dtype=torch.long)
    inverse[packed["audio_pos"]] = 0
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(tokens, arch.hidden_size, generator=generator).to(_BF16)
    x[used:] = 0
    request = {
        "x": x,
        "adaln_input": adaln_input.to(_BF16),
        "adaln_input_fp32": adaln_input,
        "combined": tags.clamp(min=0) + MINIMAX_H3_ADALN_MODALITY_NUM * inverse,
        "cos_sin_cache": dit._rope_cos_sin_cache(freqs, dtype=_BF16),
        "cos_sin_cache_fp32": dit._rope_cos_sin_cache(freqs, dtype=torch.float32),
        "cu_seqlens": packed["cu_seqlens"].to(torch.int32),
    }
    truth = _eager_stack(blocks, request, arch, fp32=True)[:used].numpy()
    served = _eager_stack(blocks, request, arch, fp32=False)[:used].numpy()
    weights = pack.pack_blocks(blocks, config, stage_on_host=True)
    stack = hook.H3MegakernelStack(
        _bare_model(), [block.adaln_proj for block in blocks], weights, None, config
    )
    with torch.no_grad():
        operands = dict(
            zip(
                ("onehot", "cos", "sin", "tables", "used"),
                stack.operands(
                    request["adaln_input"],
                    request["combined"],
                    request["cos_sin_cache"],
                    request["cu_seqlens"],
                ),
            )
        )
    os.makedirs(directory, exist_ok=True)
    arrays = {"x": x, **operands, **weights}
    for name, tensor in arrays.items():
        value = _bf16_bits(tensor) if tensor.dtype == _BF16 else tensor.numpy()
        np.save(os.path.join(directory, f"{name}.npy"), value)
    np.save(os.path.join(directory, "truth.npy"), truth)
    np.save(os.path.join(directory, "served.npy"), served)
    meta = {
        "used": used,
        "tokens": tokens,
        "layers": list(_REAL_LAYERS),
        "proven": hook.proven_layers(weights["q_norm"], weights["k_norm"], config),
        "floor": _rel(served, truth),
        "update": _rel(x[:used].float().numpy(), truth),
    }
    with open(os.path.join(directory, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


@unittest.skipUnless(
    _HAS_JAX and _HAS_CKPT and os.environ.get("H3_MEGAKERNEL_INTERPRET_TESTS") == "1",
    "needs JAX, the FL2VA checkpoint and H3_MEGAKERNEL_INTERPRET_TESTS=1 "
    "(about 14 GB of host memory and ten minutes on 4 devices)",
)
class TestMegakernelStackReal(CustomTestCase):
    def test_matches_sglang_blocks_within_the_bf16_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            meta = _run_errors(["--real-stage1", directory], devices=1)
            errors = _run_errors(
                ["--real-stage2", directory], devices=_REAL_DEVICES, timeout=7200
            )
        _check_real_errors(self, meta, errors)


def _check_real_errors(test, meta: dict, errors: dict) -> None:
    test.assertEqual(meta["proven"], [0])  # block 35 fixed-max, 36 online
    test.assertTrue(errors["finite"])
    test.assertGreater(meta["update"], 0.1)
    test.assertLess(errors["kernel"], _FLOOR_RATIO * meta["floor"])


# ------------------------------------------- one program for every prompt


def _capture_op(config, devices: int, mesh, **options):
    """``op.build_stack_op``'s exported function and its export arguments."""
    from sglang_tpu import pallas_jax_op
    from sglang_tpu.multimodal_gen.h3_megakernel import op

    captured = {}

    def export(name, function, **kwargs):
        captured.update(name=name, function=function, **kwargs)
        return MagicMock(name=name)

    with patch.object(pallas_jax_op, "cpu_export_jax_op", export):
        op.build_stack_op(config, devices, mesh=mesh, **options)
    return captured


def _meta_operands(config, rows: int, text_len: int):
    """Rank 0's op arguments for a 1344x768x124 prompt (meta tensors, the hook's
    own builders): shapes and dtypes only."""
    _init_parallel()
    arch = MiniMaxH3DiTArchConfig()
    with torch.device("meta"):
        projs = [
            dit.MiniMaxH3AdalnProj(
                arch,
                arch.adaln_out_features,
                None,
                prefix=f"blocks.{index}.adaln_proj",
                expand_ratio=6,
                modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
            )
            for index in range(config.layers)
        ]
        stack = hook.H3MegakernelStack(
            _bare_model(), projs, _meta_packed_weights(config), None, config
        )
        onehot, cos, sin, tables, used = stack.operands(
            torch.empty(2, arch.time_embed_dim, dtype=_BF16),
            torch.empty(rows, dtype=torch.long),
            torch.empty(rows, 6 * arch.rope_inv_freq_len, dtype=_BF16),
            torch.empty(3, dtype=torch.int32),
        )
        x = torch.empty(rows, config.hidden, dtype=_BF16)
    weights = [getattr(stack, name) for name in pack.WEIGHT_ORDER]
    return [x, onehot, cos, sin, tables, used, *weights]


def _meta_packed_weights(config):
    tiles = pack.tile_widths(config)
    layers, hidden, ffn = config.layers, config.hidden, config.ffn
    group = tiles["qkv_heads"]
    shapes = {
        "norm1": (layers, 1, hidden),
        "norm2": (layers, 1, hidden),
        "q_norm": (layers, 1, config.head_dim),
        "k_norm": (layers, 1, config.head_dim),
        "wqkv": (layers, config.heads // group, hidden, 3 * group * config.head_dim),
        "wo": (layers, hidden // tiles["out_cols"], config.width, tiles["out_cols"]),
        "w13": (layers, ffn // tiles["ffn_cols"], hidden, 2 * tiles["ffn_cols"]),
        "w2": (layers, hidden // tiles["down_cols"], ffn, tiles["down_cols"]),
    }
    return {
        name: torch.empty(shape, dtype=_BF16, device="meta")
        for name, shape in shapes.items()
    }


def _kernel_calls(captured, config, devices: int, structure=None, rows=128) -> list:
    """The keyword arguments of each ``kernel.stack`` call in ``_capture_op``'s
    exported function, traced (``kernel.stack`` stubbed) over its abstract
    ``"sp"`` mesh of ``devices`` with ``rows`` a device; ``structure``: the
    structured op's static argument."""
    import inspect

    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal.pallas import pallas as pallas_impl

    from sglang_tpu.multimodal_gen.h3_megakernel import op

    calls = []

    def stack(x, weights, tables, onehot, cos, sin, used, **kwargs):
        calls.append(kwargs)
        peaks = jnp.zeros((config.layers, 2, 8, op.kernel.TABLE_LANES), jnp.float32)
        return (x, peaks) if kwargs["return_peaks"] else x

    signature = inspect.signature(captured["function"], follow_wrapped=False)
    static = pallas_impl._infer_static_argnums(signature)
    tensors = _meta_operands(config, rows, 40)
    if "vouch" in signature.parameters:
        tensors[6:6] = [
            torch.empty((config.layers, config.heads), dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
        ]
    mesh = captured["mesh"]
    placeholders = []
    for tensor, spec in zip(tensors, captured["input_partition_specs"]):
        shape = list(tensor.shape)
        if spec == P("sp"):
            shape[0] *= devices
        dtype = {_BF16: jnp.bfloat16, torch.int32: jnp.int32}[tensor.dtype]
        placeholders.append(
            jax.ShapeDtypeStruct(
                tuple(shape), dtype, sharding=NamedSharding(mesh, spec)
            )
        )
    with patch.object(op.kernel, "stack", stack):
        jax.jit(captured["function"], static_argnums=static).trace(
            *placeholders, *(() if structure is None else (structure,))
        )
    return calls


def _export_for_tpu7x(config, devices: int, rows: int, structure=None):
    """The export torch_tpu runs on the worker (pallas_jax_op) of the op at
    ``rows`` a device on ``devices`` tpu7x devices; ``structure``: of the
    structured op, with that static structure (its static argument)."""
    import jax
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from sglang_tpu import pallas_export
    from sglang_tpu.multimodal_gen.h3_megakernel import op

    topology = {8: "tpu7x:2x2x1", 4: "tpu7x:2x1x1", 2: "tpu7x:1x1x1"}[devices]
    mesh = Mesh(
        np.asarray(topologies.get_topology_desc(topology, "tpu").devices).reshape(
            devices
        ),
        ("sp",),
    )
    structured = structure is not None
    captured = _capture_op(config, devices, mesh, structured=structured)
    in_specs, out_spec = op.partition_specs()
    args = _meta_operands(config, rows, 40)
    placeholders = []
    for tensor, spec in zip(args, in_specs):
        shape = list(tensor.shape)
        if spec == P("sp"):
            shape[0] *= devices
        dtype = {_BF16: jax.numpy.bfloat16, torch.int32: jax.numpy.int32}[tensor.dtype]
        placeholders.append(
            jax.ShapeDtypeStruct(
                tuple(shape), dtype, sharding=NamedSharding(mesh, spec)
            )
        )
    jitted = jax.jit(
        captured["function"],
        static_argnums=(len(in_specs),) if structured else (),
        keep_unused=True,
        out_shardings=NamedSharding(mesh, out_spec),
    )
    with (
        patch.object(pallas_export, "get_torch_tpu_device_kind", return_value="TPU v7"),
        pallas_export.tpu_info_for_cpu_export(),
        jax._src.config.export_ignore_forward_compatibility(True),
    ):
        return jax.export.export(
            jitted,
            platforms=["tpu"],
            disabled_checks=[jax.export.DisabledSafetyCheck.custom_call("ALL")],
        )(*placeholders, *((structure,) if structured else ()))


@unittest.skipUnless(_HAS_JAX and _HAS_TORCH_TPU, "needs JAX and torch_tpu")
class TestOneProgramForEveryPrompt(CustomTestCase):
    def test_every_h3_v2_prompt_has_one_kernel_key(self):
        # torch_tpu keys a registered kernel by its tensor arguments' shapes
        # and dtypes and its static arguments' values: the live length is a
        # [1] int32 tensor, never static, and every prompt pads to the same
        # rows, so all 21 prompts share one program.
        import inspect

        from jax.sharding import AbstractMesh
        from torch_tpu._internal.pallas import pallas as pallas_impl

        config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)
        captured = _capture_op(config, 8, AbstractMesh((8,), ("sp",)))
        self.assertEqual(
            captured["name"], "sglang_tpu_pallas::minimax_h3_stack_sp8_L50"
        )
        signature = inspect.signature(captured["function"], follow_wrapped=False)
        static = pallas_impl._infer_static_argnums(signature)
        self.assertEqual(static, ())
        self.assertEqual(list(signature.parameters)[5], "used")
        keys = set()
        with _padding_restored():
            hook._use_kernel_padding(8, config)
            for name, text_len in _SPEC_TEXT_TOKENS.items():
                seq_len = _spec_layout(text_len)["seq_len"]
                args = _meta_operands(config, seq_len // 8, text_len)
                keys.add(
                    pallas_impl._get_kernel_invocation_key(
                        captured["name"], args, {}, static
                    )
                )
        self.assertEqual(len(keys), 1, keys)
        (key,) = keys
        self.assertIn("torch.Size([4992, 5376])xtorch.bfloat16", key)
        self.assertIn("torch.Size([1])xtorch.int32", key)

    def test_every_h3_v2_prompt_has_one_structured_kernel_key_a_shape(self):
        # The static-structure option: the structured op's last argument, the
        # static structure, is static (a key part); all 21 prompts pad to one
        # row count and pass one structure, so they share one program at each
        # of its row counts.
        import inspect

        from jax.sharding import AbstractMesh
        from torch_tpu._internal.pallas import pallas as pallas_impl

        config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)
        for devices, rows, override in _STRUCTURED:
            with self.subTest(devices=devices, rows=rows):
                captured = _capture_op(
                    config,
                    devices,
                    AbstractMesh((devices,), ("sp",)),
                    structured=True,
                )
                self.assertTrue(captured["name"].endswith("_structured"))
                self.assertEqual(len(captured["input_partition_specs"]), 15)
                signature = inspect.signature(
                    captured["function"], follow_wrapped=False
                )
                static = pallas_impl._infer_static_argnums(signature)
                self.assertEqual(static, (14,))
                self.assertEqual(list(signature.parameters)[14], "structure")
                keys, structures = set(), set()
                with (
                    _padding_restored(),
                    tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_STATIC_STRUCTURE.override(
                        True
                    ),
                    tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS.override(override),
                ):
                    hook._use_kernel_padding(devices, config)
                    stack = hook.H3MegakernelStack(
                        _bare_model(),
                        [],
                        _meta_packed_weights(config),
                        None,
                        config,
                        devices=devices,
                        structured=True,
                    )
                    for text_len in _SPEC_TEXT_TOKENS.values():
                        packed = _spec_layout(text_len)
                        local = packed["seq_len"] // devices
                        structure = stack.structure(
                            tuple(packed["cu_seqlens"].tolist()), local
                        )
                        structures.add(structure)
                        args = _meta_operands(config, local, text_len) + [structure]
                        keys.add(
                            pallas_impl._get_kernel_invocation_key(
                                captured["name"], args, {}, static
                            )
                        )
                self.assertEqual(len(keys), 1, keys)
                (key,) = keys
                (structure,) = structures
                self.assertGreater(structure, 0)
                self.assertIn(f"torch.Size([{rows}, 5376])xtorch.bfloat16", key)
                # The last argument's part of the key: "<inputs>;<kwargs>".
                self.assertTrue(key.endswith(f",static_{structure};"), key[-80:])

    @unittest.skipUnless(
        os.environ.get("H3_MEGAKERNEL_LOWERING_TESTS") == "1",
        "needs libtpu's compile-only tpu7x topology and "
        "H3_MEGAKERNEL_LOWERING_TESTS=1 (about a minute)",
    )
    def test_structured_op_lowers_for_tpu7x_at_4864_rows(self):
        # The structured op's export at 4,864 rows a device on 8 tpu7x
        # devices with the benchmark prompts' static structure: the structure is
        # no operand, the live length still is.
        config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)
        structure = hook.static_structure(37_749, 4864, 8, config)
        exported = _export_for_tpu7x(config, 8, 4864, structure)
        text = exported.mlir_module()
        self.assertIn("minimax_h3_stack", text)
        start = text.index("func.func public @main(")
        signature = text[start : text.index("->", start)]
        self.assertIn("%arg5: tensor<1xi32>", signature)
        self.assertEqual(len(exported.in_avals), 14)

    def test_the_vouched_op_takes_its_heads_and_tally_as_tensors(self):
        # The vouched variant: the heads to try ([L, heads] int32) and the
        # tally ([2] int32) follow the live length as replicated tensors, so
        # every step and every set of heads runs one program; it returns the
        # rows, the vouched heads and the tally.
        import inspect

        import jax
        from jax.sharding import AbstractMesh
        from jax.sharding import PartitionSpec as P
        from torch_tpu._internal.pallas import pallas as pallas_impl

        config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)
        captured = _capture_op(config, 8, AbstractMesh((8,), ("sp",)), vouch_margin=1.1)
        self.assertEqual(
            captured["name"], "sglang_tpu_pallas::minimax_h3_stack_sp8_L50_vouch"
        )
        signature = inspect.signature(captured["function"], follow_wrapped=False)
        self.assertEqual(pallas_impl._infer_static_argnums(signature), ())
        self.assertEqual(
            list(signature.parameters),
            ["x", "onehot", "cos", "sin", "tables", "used", "vouch", "tally"]
            + list(pack.WEIGHT_ORDER),
        )
        self.assertEqual(
            signature.return_annotation, tuple[jax.Array, jax.Array, jax.Array]
        )
        self.assertEqual(
            captured["input_partition_specs"][:8], (P("sp"),) * 4 + (P(),) * 4
        )
        self.assertEqual(captured["output_partition_specs"], (P("sp"), P(), P()))

    def test_the_vouched_structured_op_takes_the_structure_last(self):
        # Both options: the vouched op's tensors, then the static structure,
        # its one static argument (no partition); the same three outputs.
        import inspect

        import jax
        from jax.sharding import AbstractMesh
        from jax.sharding import PartitionSpec as P
        from torch_tpu._internal.pallas import pallas as pallas_impl

        op = op_module()
        config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)
        captured = _capture_op(
            config,
            8,
            AbstractMesh((8,), ("sp",)),
            structured=True,
            vouch_margin=1.1,
        )
        base = "sglang_tpu_pallas::minimax_h3_stack_sp8_L50"
        self.assertEqual(captured["name"], base + "_vouch_structured")
        signature = inspect.signature(captured["function"], follow_wrapped=False)
        self.assertEqual(pallas_impl._infer_static_argnums(signature), (16,))
        self.assertEqual(
            list(signature.parameters),
            ["x", "onehot", "cos", "sin", "tables", "used", "vouch", "tally"]
            + list(pack.WEIGHT_ORDER)
            + ["structure"],
        )
        self.assertEqual(
            signature.return_annotation, tuple[jax.Array, jax.Array, jax.Array]
        )
        in_specs, out_specs = op.vouched_partition_specs()
        self.assertEqual(captured["input_partition_specs"], (*in_specs, None))
        self.assertEqual(captured["output_partition_specs"], (P("sp"), P(), P()))
        # Each option alone keeps its own op.
        self.assertEqual(op.op_name(config, 8), base)
        self.assertEqual(op.op_name(config, 8, structured=True), base + "_structured")
        self.assertEqual(op.op_name(config, 8, vouched=True), base + "_vouch")
        # Trace regions (SGLANG_TPU_MK_TRACE_REGIONS, profiling only): their
        # own op; level 0 leaves the kernel options as they are.
        traced = _capture_op(
            config,
            8,
            AbstractMesh((8,), ("sp",)),
            structured=True,
            vouch_margin=1.1,
            trace=1,
        )
        self.assertEqual(traced["name"], base + "_vouch_structured_trace1")
        options = op.stack_kwargs(8, 37_633, rows=4864)
        self.assertIs(op.traced(options, 0), options)
        self.assertEqual(op.traced(options, 2), dict(options, trace=2))

    def test_every_h3_v2_prompt_has_one_vouched_structured_kernel_key_a_shape(
        self,
    ):
        # Both options: all 21 prompts pad to one row count and pass one
        # static structure, and the heads to try and the tally are tensors of
        # one shape, so they share one program at each of its row counts.
        import inspect

        from jax.sharding import AbstractMesh
        from torch_tpu._internal.pallas import pallas as pallas_impl

        config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)
        for devices, rows, override in _STRUCTURED:
            with self.subTest(devices=devices, rows=rows):
                captured = _capture_op(
                    config,
                    devices,
                    AbstractMesh((devices,), ("sp",)),
                    structured=True,
                    vouch_margin=1.1,
                )
                signature = inspect.signature(
                    captured["function"], follow_wrapped=False
                )
                static = pallas_impl._infer_static_argnums(signature)
                keys, structures = set(), set()
                with (
                    _padding_restored(),
                    tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_STATIC_STRUCTURE.override(
                        True
                    ),
                    tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_VOUCH.override(True),
                    tpu_envs.SGLANG_TPU_OPT_H3_MEGAKERNEL_ROWS.override(override),
                ):
                    hook._use_kernel_padding(devices, config)
                    stack = hook.H3MegakernelStack(
                        _bare_model(),
                        [],
                        _meta_packed_weights(config),
                        None,
                        config,
                        devices=devices,
                        structured=True,
                        vouch_margin=1.1,
                    )
                    for text_len in _SPEC_TEXT_TOKENS.values():
                        packed = _spec_layout(text_len)
                        local = packed["seq_len"] // devices
                        structure = stack.structure(
                            tuple(packed["cu_seqlens"].tolist()), local
                        )
                        structures.add(structure)
                        vouching = stack._fresh_vouching(None)
                        operands = _meta_operands(config, local, text_len)
                        args = [
                            *operands[:6],
                            vouching.heads,
                            vouching.tally,
                            *operands[6:],
                            structure,
                        ]
                        keys.add(
                            pallas_impl._get_kernel_invocation_key(
                                captured["name"], args, {}, static
                            )
                        )
                self.assertEqual(len(keys), 1, keys)
                (key,) = keys
                (structure,) = structures
                self.assertGreater(structure, 0)
                self.assertIn(f"torch.Size([{rows}, 5376])xtorch.bfloat16", key)
                self.assertIn("torch.Size([50, 56])xtorch.int32", key)
                self.assertIn("torch.Size([2])xtorch.int32", key)
                self.assertTrue(key.endswith(f",static_{structure};"), key[-80:])

    def test_each_option_reaches_the_kernel_call(self):
        # The exported function of each op, traced with the kernel stubbed:
        # one kernel call site. The plain op passes the pinned options, the
        # structured op the static structure's (op.stack_kwargs), the
        # vouched op the plain op's plus the tried heads (a traced [L, heads]
        # bool) and the peaks; with both options the vouched op's call is the
        # structured op's plus those two.
        import jax.numpy as jnp
        from jax.sharding import AbstractMesh

        op = op_module()
        config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)
        structure = 37_633
        for devices in (8, 4, 2):
            mesh = AbstractMesh((devices,), ("sp",))
            for structured, margin in (
                (False, None),
                (True, None),
                (False, 1.1),
                (True, 1.1),
            ):
                with self.subTest(
                    devices=devices, structured=structured, margin=margin
                ):
                    captured = _capture_op(
                        config,
                        devices,
                        mesh,
                        structured=structured,
                        vouch_margin=margin,
                    )
                    (kwargs,) = _kernel_calls(
                        captured, config, devices, structure if structured else None
                    )
                    want = dict(
                        op.stack_kwargs(devices, structure if structured else None),
                        config=config,
                        tiling=op.STACK_TILING,
                        axis="sp",
                        interpret=False,
                    )
                    if margin is not None:
                        fixed = kwargs.pop("fixed_heads")
                        self.assertEqual(fixed.shape, (50, 56))
                        self.assertEqual(fixed.dtype, jnp.bool_)
                        del want["fixed_heads"]
                        want["return_peaks"] = True
                    self.assertEqual(kwargs, want)
                    self.assertEqual(
                        kwargs.get("static_structure"),
                        structure if structured else None,
                    )
                    self.assertEqual(
                        kwargs.get("prefetch_down", True),
                        not (structured and devices == 4),
                    )

    @unittest.skipUnless(
        os.environ.get("H3_MEGAKERNEL_LOWERING_TESTS") == "1",
        "needs libtpu's compile-only tpu7x topology and "
        "H3_MEGAKERNEL_LOWERING_TESTS=1 (about a minute)",
    )
    def test_lowers_for_tpu7x_at_1344x768x124(self):
        # The export torch_tpu runs on the worker (pallas_jax_op), at 4,992
        # rows a device on 8 tpu7x devices: the live length is the program's
        # [1] int32 argument, not a constant.
        config = pack.config_from_arch(MiniMaxH3DiTArchConfig(), 50)
        exported = _export_for_tpu7x(config, 8, 4992)
        text = exported.mlir_module()
        self.assertIn("tpu_custom_call", text)
        self.assertIn("minimax_h3_stack", text)
        # Argument 5 of the program is the live length.
        start = text.index("func.func public @main(")
        signature = text[start : text.index("->", start)]
        self.assertIn("%arg5: tensor<1xi32>", signature)
        self.assertEqual(len(exported.in_avals), 14)
        self.assertEqual(exported.in_avals[5].shape, (1,))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--small-errors"]:
        bugs = [bug for bug in sys.argv[4].split(",") if bug]
        print(
            json.dumps(_small_stack_errors(int(sys.argv[2]), sys.argv[3] == "1", bugs))
        )
    elif sys.argv[1:2] == ["--real-stage1"]:
        print(json.dumps(_real_sglang_side(sys.argv[2])))
    else:
        unittest.main()
