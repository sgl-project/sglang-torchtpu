# SPDX-License-Identifier: Apache-2.0
"""VDN-H3's TPU hybrid attention (sglang_tpu.multimodal_gen.minimax_h3_vdn) against
the core's reference: the window mask on a dense softmax and the eager linear branch."""

import os
import socket
import unittest
from pkgutil import resolve_name
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.compiler.config as compiler_config
import torch.multiprocessing as mp

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    VDNHybridAttentionArchConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.hybrid_window_attn_h3 import (
    window_mask_reference,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
from sglang.multimodal_gen.runtime.models.dits import minimax_h3_vdn as branch_ops
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn import (
    VDNH3Layout,
    cholesky_spd_inverse,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn_attention import (
    prepare_hybrid_attention_metadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    packed_sequence,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import minimax_h3_vdn as vdn
from sglang_tpu.multimodal_gen import worker
from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform

register_tpu_ci(
    est_time=180,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "diffusion"],
)

HEADS, HEAD_DIM, HIDDEN = 4, 128, 256
# 27 frames of 5-frame chunks: two interior chunks share one batched call.
FRAMES, FRAME_H, FRAME_W = 27, 2, 4
TPF = FRAME_H * FRAME_W
AUDIO_ROWS, SEQ_LEN = 20, 320  # one bucket for both text lengths, with padding
ROPE_FREQS = 96  # 6 * the arch's rope_inv_freq_len


def _init_parallel_state(sp_size=1):
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    with patch.object(worker, "install_runtime_patches"):
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=1, sp_size=sp_size, ulysses_degree=sp_size
        )


def _layout(text_len):
    video_start = text_len + AUDIO_ROWS
    return VDNH3Layout(
        seq_len=SEQ_LEN,
        used=video_start + FRAMES * TPF,
        text_len=text_len,
        video_start=video_start,
        num_frames=FRAMES,
        tokens_per_frame=TPF,
        frame_height=FRAME_H,
        frame_width=FRAME_W,
    )


def _attention():
    """One VDN DiT-block attention on the TPU hybrid backend, seeded."""
    torch.manual_seed(0)
    hybrid = VDNHybridAttentionArchConfig(chunk=5, radius=1, anchor_frames="both")
    arch = MiniMaxH3DiTArchConfig(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        hybrid_attention=hybrid,
    )
    attn = dit.MiniMaxH3Attention(arch, None, prefix="blocks.0")
    with torch.no_grad():
        for name, p in attn.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(1, 16)))
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape) - 3)
            elif "norm" in name:
                p.copy_(1 + 0.1 * torch.randn(p.shape))
            else:
                p.copy_((torch.randn(p.shape) * 0.05).to(p.dtype))
    attn._set_attention_backend(vdn.TpuHybridWindowAttentionH3Backend)
    return attn


def _inputs(attn, text_len, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(SEQ_LEN, HIDDEN, generator=g).bfloat16()
    freqs = torch.randn(SEQ_LEN, ROPE_FREQS, generator=g)
    rope = (dit._rope_cos_sin_cache(freqs, dtype=torch.bfloat16), torch.arange(SEQ_LEN))
    layout = _layout(text_len)
    cu = torch.tensor([0, layout.used, SEQ_LEN], dtype=torch.int32)
    return x, rope, layout, cu


def _metadata(attn, layout, rope_cache_full=None, inverse="newton_schulz"):
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=attn)])
    args = SimpleNamespace(attention_backend_config={"vdn_spd_inverse": inverse})
    return vdn.TpuHybridWindowAttentionH3MetadataBuilder().build(
        layout=layout,
        hybrid=attn.hybrid.linear_attention.hybrid,
        device=torch.device("cpu"),
        rope_cache_full=rope_cache_full,
        model=model,
        server_args=args,
    )


def _core_inputs(attn, x):
    hybrid, total = attn.hybrid, x.shape[0]
    q, k, v = attn.qkv_proj(x)[0].split(attn.local_inner_dim, dim=-1)
    q, k, v = (t.view(total, attn.num_heads, attn.head_dim) for t in (q, k, v))
    gate_hidden, _ = hybrid.linear_attention.output_gate.down(x)
    return q, k, v, hybrid.softmax_gate(x), hybrid.linear_attention.beta(x), gate_hidden


def _reference(attn, x, rope, layout):
    """(gated softmax [S, H, d], linear [S, H * d]) as the core computes them on
    CUDA: the window mask on a float32 softmax, the eager branch with Cholesky."""
    q, k, v, softmax_gate, beta, gate_hidden = _core_inputs(attn, x)
    branch, used = attn.hybrid.linear_attention, layout.used
    q_sm, k_sm = dit._apply_rope_qk(
        attn.q_norm(q), attn.k_norm(k), *rope, rotary_emb=attn.rotary_emb
    )
    mask = window_mask_reference(branch.hybrid, layout, x.device)
    scores = torch.einsum("qhd,khd->hqk", q_sm[:used].float(), k_sm[:used].float())
    scores = (scores * attn.softmax_scale).masked_fill(~mask, float("-inf"))
    window = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v[:used].float())
    softmax = torch.zeros_like(q)
    softmax[:used] = window.to(q.dtype) * softmax_gate[:used, :, None]
    video, text = slice(layout.video_start, used), slice(0, layout.text_len)
    inverse, branch.spd_inverse = branch.spd_inverse, cholesky_spd_inverse
    linear = torch.zeros(x.shape[0], HEADS * HEAD_DIM, dtype=x.dtype)
    linear[video] = branch(
        q_raw=q[video],
        k_raw=k[video],
        v_raw=v[video],
        beta=beta[video],
        gate=branch.output_gate.up_gate(gate_hidden)[video],
        frame_mean=x[video].view(FRAMES, TPF, -1).mean(1, dtype=torch.float32),
        layout=layout,
        text_k_raw=k[text],
        text_v_raw=v[text],
        text_beta=beta[text],
    )
    branch.spd_inverse = inverse
    return softmax, linear


def _core(attn, x, rope, cu, meta, rows=slice(None), ulysses=False):
    q, k, v, softmax_gate, beta, gate_hidden = (t[rows] for t in _core_inputs(attn, x))
    with set_forward_context(current_timestep=0, attn_metadata=meta):
        return vdn.tpu_hybrid_attention_core(
            attn, x[rows], q, k, v, softmax_gate, beta, gate_hidden,
            rope_cache=rope, cu_seqlens=cu, ulysses_active=ulysses,
        )  # fmt: skip


def _assert_close(test, got, want, rows=slice(None)):
    # bf16 outputs of float32 math in another order, Newton-Schulz for Cholesky
    got, want = got.float(), want[rows].float()
    test.assertLess(((got - want).norm() / want.norm()).item(), 1e-3)
    torch.testing.assert_close(got, want, atol=3e-2, rtol=1e-2)


class TestTpuHybridCore(CustomTestCase):
    def test_platform_and_denoise_stage_reach_the_tpu_backend(self):
        _init_parallel_state()
        enum = AttentionBackendEnum.HYBRID_WINDOW_ATTN_H3
        backend = resolve_name(
            TpuOOTPlatform.get_attn_backend_cls_str(enum, HEAD_DIM, torch.bfloat16)
        )
        self.assertIs(backend, vdn.TpuHybridWindowAttentionH3Backend)
        attn = _attention()
        model = SimpleNamespace(
            blocks=[SimpleNamespace(attn=attn)],
            arch=SimpleNamespace(hybrid_attention=attn.hybrid.linear_attention.hybrid),
            _resolve_attention_backend_once=lambda: None,
            _resolved_attention_backend=enum,
        )
        packed = packed_sequence.minimax_h3_packed_sequence(
            text_len=30, latent_t=FRAMES, latent_h=2 * FRAME_H, latent_w=2 * FRAME_W,
            audio_t=10, include_keyframe_cond=False,
        )  # fmt: skip
        build = prepare_hybrid_attention_metadata(
            model=model,
            packed=packed,
            latent_shape=(FRAMES, 2 * FRAME_H, 2 * FRAME_W),
            server_args=SimpleNamespace(attention_backend_config=None),
            device=torch.device("cpu"),
        )
        self.assertIsInstance(build(0), vdn.TpuHybridWindowAttentionH3Metadata)
        self.assertIs(
            attn.hybrid.linear_attention.spd_inverse, vdn.newton_schulz_spd_inverse
        )

    def test_core_matches_the_reference_for_two_text_lengths(self):
        _init_parallel_state()
        attn = _attention()
        for text_len in (30, 50):
            x, rope, layout, cu = _inputs(attn, text_len)
            with torch.no_grad():
                meta = _metadata(attn, layout)
                softmax, linear = _core(attn, x, rope, cu, meta)
                ref_softmax, ref_linear = _reference(attn, x, rope, layout)
            _assert_close(self, softmax, ref_softmax)
            _assert_close(self, linear, ref_linear)
            # padding rows are zero in both branches
            self.assertEqual(softmax[layout.used :].abs().max().item(), 0)
            self.assertEqual(linear[layout.used :].abs().max().item(), 0)

    def test_newton_schulz_inverse_matches_cholesky(self):
        # (I + A) of a 1344x768 frame: 1008 unit SiLU keys sharing a direction,
        # beta in (0, 1); eigenvalues 1.2 to 456 (8 steps leave 20% error)
        g = torch.Generator().manual_seed(0)
        keys = torch.randn(4, 1008, 128, generator=g) + 3
        keys = torch.nn.functional.normalize(torch.nn.functional.silu(keys), dim=-1)
        beta = torch.rand(4, 1008, 1, generator=g)
        m = torch.eye(128) + (keys * beta).transpose(-1, -2) @ keys
        got = vdn.newton_schulz_spd_inverse(m)
        torch.testing.assert_close(got, cholesky_spd_inverse(m), atol=1e-6, rtol=1e-4)


class TestTpuHybridCompile(CustomTestCase):
    def test_compiled_attention_stages_the_branch_and_compiles_once(self):
        # torch_tpu miscompiles the branch whole: it runs between compiled
        # graphs, a graph per stage. The prompt length is device data only, so
        # no graph recompiles for it.
        _init_parallel_state()
        attn = _attention()
        with compiler_config.patch(cache_key_tag=""):
            region = TpuOOTPlatform().prepare_spmd_region_wrapper()
        attn._attention_impl.spmd_parts = (
            region(vdn._rows_to_window),
            region(vdn._window_to_rows),
        )
        graphs, stage_graphs = [], []

        def stage_backend(gm, example_inputs):
            trace = [n.meta.get("stack_trace") or "" for n in gm.graph.nodes]
            stage_graphs.append(" ".join(trace))
            return gm.forward

        def backend(gm, example_inputs):
            tensors = [t for t in example_inputs if isinstance(t, torch.Tensor)]
            graphs.append(
                (gm, {(t.storage_offset(), t.is_contiguous()) for t in tensors})
            )
            return gm.forward

        def run(module, text_len):
            x, rope, layout, cu = _inputs(attn, text_len)
            with set_forward_context(
                current_timestep=0, attn_metadata=_metadata(attn, layout)
            ):
                return module(x, rope_cache=rope, cu_seqlens=cu, max_seqlen=SEQ_LEN)

        torch._dynamo.reset()
        vdn._compiled_branch_stages.cache_clear()
        compiled = torch.compile(attn, backend=backend, dynamic=False)
        eager_stages = {n: getattr(branch_ops, n) for n in vdn.BRANCH_STAGES}
        counts = []
        kwargs = {"backend": stage_backend, "dynamic": False}
        with (
            torch.no_grad(),
            patch.object(vdn, "build_torch_compile_kwargs", lambda **_: kwargs),
        ):
            for text_len in (30, 50, 30):
                torch.testing.assert_close(run(compiled, text_len), run(attn, text_len))
                counts.append((len(graphs), len(stage_graphs)))
        vdn._compiled_branch_stages.cache_clear()
        self.assertEqual(counts, [counts[0]] * 3)
        self.assertEqual(
            {n: getattr(branch_ops, n) for n in eager_stages}, eager_stages
        )
        # frame statistics (text and video), then one each; the features stay eager
        self.assertEqual(len(stage_graphs), 6)
        self.assertFalse([g for g in stage_graphs if "in linear_features" in g])
        self.assertTrue(all(g.strip() for g in stage_graphs))
        ops = [
            [n for n in gm.graph.nodes if n.op == "call_function"] for gm, _ in graphs
        ]
        # the branch is not in the block's graphs
        self.assertFalse(any("conv" in str(n.target) for g in ops for n in g))
        marked = [
            i
            for i, g in enumerate(ops)
            if g and all(n.meta.get("custom") == {"spmd_safe": True} for n in g)
        ]
        self.assertEqual(len(marked), 2, "the core's parts around the branch")
        # from the second part on, every graph input is offset 0 and contiguous
        for _, layouts in graphs[marked[1] :]:
            self.assertEqual(layouts, {(0, True)})


def _ulysses_worker(rank, port):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2"
    )  # fmt: skip
    _init_parallel_state(sp_size=2)
    attn = _attention()
    x, rope, layout, cu = _inputs(attn, 50)
    rows = slice(rank * SEQ_LEN // 2, (rank + 1) * SEQ_LEN // 2)
    with torch.no_grad():
        ref_softmax, ref_linear = _reference(attn, x, rope, layout)
        meta = _metadata(attn, layout, rope_cache_full=rope)
        # this rank's heads, cut and cloned once
        branch = attn.hybrid.linear_attention
        assert branch.alpha.up.weight.shape[0] == HEADS // 2 * HEAD_DIM
        for p in branch.parameters():
            assert p.storage_offset() == 0
        softmax, linear = _core(attn, x, None, cu, meta, rows=rows, ulysses=True)
    test = unittest.TestCase()
    _assert_close(test, softmax, ref_softmax, rows)
    _assert_close(test, linear, ref_linear, rows)


class TestTpuHybridUlysses(CustomTestCase):
    def test_two_rank_ulysses_matches_one_rank(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with patch.dict(os.environ, {"SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "cpu"}):
            mp.spawn(_ulysses_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
