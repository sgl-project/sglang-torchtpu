"""CPU checks of the V4.1 DSpark drafter port (sglang_tpu/dsv41/dspark.py).

The drafter's TPU leaves are swapped for CPU stand-ins: the shifted-mHC
boundary ops for a torch port (itself checked against the JAX kernels' XLA
path), the fused EP MoE for a dense reference MoE. The drafter (context
rings, floors, injection, block attention, stage chaining, Markov head) is
compared with an independent per-request reference of core DSpark semantics
(models/deepseek_v4_dspark.py: block [r, noise...] at positions n..n+K-1,
bidirectional inside the block, the last sliding_window committed positions as
context, Markov-corrected greedy chain).
"""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import sglang.srt.layers.linear as linear
import sglang.srt.models.deepseek_v4 as deepseek_v4
import sglang_tpu.dsv41.mhc as mhc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.dsv41 import dspark
from sglang_tpu.dsv41.loader import plan_expert_reads
from sglang_tpu.experimental.speculative.dspark.dsv41_dspark_worker import (
    commit_rows,
    extend_rows,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

HIDDEN, HEADS, HEAD_DIM, VOCAB, WINDOW = 64, 8, 128, 61, 32
NOISE, GAMMA, STAGES = 60, 5, 3


def tiny_config(**overrides):
    values = dict(
        model_type="deepseek_v41",
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        qk_rope_head_dim=64,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        rms_norm_eps=1e-6,
        compress_ratios=[0] * 6,
        q_head_norm=False,
        rope_theta=10_000,
        compress_rope_theta=160_000,
        rope_scaling=None,
        max_position_embeddings=1024,
        sliding_window=WINDOW,
        vocab_size=VOCAB,
        hc_mult=4,
        hc_eps=1e-6,
        hc_sinkhorn_iters=20,
        hc_pre_from_prev_sublayer=True,
        num_hidden_layers=6,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        num_nextn_predict_layers=STAGES,
        dspark_block_size=GAMMA,
        dspark_noise_token_id=NOISE,
        dspark_target_layer_ids=[3, 4, 5],
        dspark_markov_rank=8,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# ---- shifted mHC: torch port of sglang_tpu/kernels/deepseek_v41/mhc/delayed.py (XLA path) ----


def _gates(
    flat, fn, scale, base, rms_eps, pre_eps, sinkhorn_eps, post_mult, repeats, mult
):
    x = flat.float()
    mixes = x @ fn.float().T
    squares = x.square().mean(-1, keepdim=True)
    mixes = mixes * torch.rsqrt(squares + rms_eps)
    pre = torch.sigmoid(mixes[:, :mult] * scale[0] + base[:mult]) + pre_eps
    post = torch.sigmoid(mixes[:, mult : 2 * mult] * scale[1] + base[mult : 2 * mult])
    post = post * post_mult
    comb = mixes[:, 2 * mult :].view(-1, mult, mult) * scale[2] + base[2 * mult :].view(
        1, mult, mult
    )
    comb = comb.softmax(-1) + sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    for _ in range(repeats - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    return pre, post, comb


def _collapse(flat, previous_pre):
    mult = previous_pre.shape[1]
    streams = flat.view(flat.shape[0], mult, -1).float()
    return (previous_pre[..., None] * streams).sum(1).to(flat.dtype)


def torch_pre(flat, fn, scale, base, previous_pre, *constants):
    pre, post, comb = _gates(flat, fn, scale, base, *constants, previous_pre.shape[1])
    return post[..., None], comb, _collapse(flat, previous_pre), pre


def torch_post(x, flat, post, comb):
    mult = comb.shape[-1]
    streams = flat.view(flat.shape[0], mult, -1).float()
    x = x.to(flat.dtype).float()
    post = post.reshape(-1, mult)
    out = [
        (post[:, j : j + 1] * x + (comb[:, :, j, None] * streams).sum(1)).to(flat.dtype)
        for j in range(mult)
    ]
    return torch.cat(out, dim=1)


def torch_seam(x, flat, post, comb, fn, scale, base, previous_pre, *constants):
    new_flat = torch_post(x, flat, post, comb)
    post2, comb2, _, pre = torch_pre(
        new_flat, fn, scale, base, previous_pre, *constants
    )
    return new_flat, post2, comb2, _collapse(new_flat, previous_pre), pre


def torch_mhc_ops():
    return torch_pre, torch_seam, torch_post


# ---- dense reference MoE with the FusedEPExperts / SharedExperts names --------


class ReferenceMoE(nn.Module):
    def __init__(self, config, layer_id, quant_config, prefix):
        super().__init__()
        del layer_id, quant_config, prefix
        experts, inter = config.n_routed_experts, config.moe_intermediate_size
        self.topk = config.num_experts_per_tok
        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(torch.randn(experts, config.hidden_size) * 0.1)
        self.gate.e_score_correction_bias = nn.Parameter(torch.randn(experts) * 0.01)
        self.experts = nn.Module()
        for name, shape in (
            ("w13_weight", (experts, 2 * inter, config.hidden_size)),
            ("w2_weight", (experts, config.hidden_size, inter)),
        ):
            param = nn.Parameter(torch.randn(*shape) * 0.05)
            param.weight_loader = self._expert_loader
            self.experts.register_parameter(name, param)
        self.shared_experts = nn.Module()
        self.shared_experts.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * inter, bias=False
        )
        self.shared_experts.gate_up_proj.weight.weight_loader = self._stacked_loader
        self.shared_experts.down_proj = nn.Linear(inter, config.hidden_size, bias=False)

    @staticmethod
    def _expert_loader(param, loaded, name, *, shard_id, expert_id):
        target = param.data[expert_id]
        if shard_id in ("w1", "w3"):
            half = target.shape[0] // 2
            target = target.narrow(0, 0 if shard_id == "w1" else half, half)
        target.copy_(loaded)

    @staticmethod
    def _stacked_loader(param, loaded, shard):
        half = param.shape[0] // 2
        param.data.narrow(0, shard * half, half).copy_(loaded)

    @staticmethod
    def _ffn(x, w13, w2):
        gate, up = (x @ w13.T).chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate) * up) @ w2.T

    def forward(self, x, input_ids, valid):
        del input_ids
        scores = (x.float() @ self.gate.weight.T).sigmoid()
        ids = torch.topk(scores + self.gate.e_score_correction_bias, self.topk).indices
        weights = scores.gather(1, ids)
        weights = weights / weights.sum(-1, keepdim=True)
        # Dense over experts (no data-dependent Python, so it traces fullgraph).
        out = torch.zeros_like(x)
        for e in range(self.experts.w13_weight.shape[0]):
            weight = (weights * (ids == e)).sum(-1, keepdim=True)
            ffn = self._ffn(x, self.experts.w13_weight[e], self.experts.w2_weight[e])
            out = out + weight.to(x.dtype) * ffn
        shared = self.shared_experts
        out = out + self._ffn(x, shared.gate_up_proj.weight, shared.down_proj.weight)
        return out.masked_fill(~valid[:, None], 0)


class LMHead(nn.Module):
    quant_method = None

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(VOCAB + 3, HIDDEN) * 0.3)


@contextlib.contextmanager
def cpu_parallel():
    """A single-rank parallel context for the core linear layers' forward."""
    parallel = SimpleNamespace(tp_group=None, attn_tp_group=None)
    with (
        # Plain functions, not mocks: the compile test traces these.
        patch.object(linear, "get_parallel", lambda: parallel),
        patch.object(
            linear, "use_symmetric_memory", lambda *a, **k: contextlib.nullcontext()
        ),
    ):
        yield


def build_drafter(seed=0):
    torch.manual_seed(seed)
    parallel = SimpleNamespace(attn_tp_rank=0, attn_tp_size=1, tp_size=1)
    with (
        patch.object(deepseek_v4, "get_parallel", return_value=parallel),
        patch.object(mhc, "register_ops", torch_mhc_ops),
    ):
        drafter = dspark.DSV41DSparkDrafter(
            tiny_config(), None, moe_factory=ReferenceMoE
        )
    with torch.no_grad():
        for name, param in drafter.named_parameters():
            if name.endswith("_scale"):
                param.copy_(torch.rand_like(param) * 0.5 + 0.5)
            elif name.endswith("_base"):
                param.copy_(torch.randn_like(param) * 0.3)
            elif "norm" in name:
                param.copy_(1 + 0.1 * torch.randn_like(param))
            elif "attn_sink" in name:
                param.copy_(torch.randn_like(param))
            elif "experts" in name or "gate" in name:
                continue
            else:
                param.copy_(torch.randn_like(param) / param.shape[-1] ** 0.5)
    embed = nn.Embedding(VOCAB, HIDDEN)
    drafter.attach(embed_tokens=embed, lm_head=LMHead())
    drafter.allocate_state(4, "cpu")
    return drafter


# ---- independent per-request reference --------------------------------------


def jax_mhc():
    """The JAX kernels' XLA path (TPU-free), on numpy round trips."""
    import jax
    import jax.numpy as jnp

    from sglang_tpu.kernels.deepseek_v41.mhc.delayed import _post_jax, delayed_pre_jax

    jax.config.update("jax_default_matmul_precision", "highest")

    def to_np(t):
        return jnp.asarray(t.detach().float().numpy())

    def pre(flat, fn, scale, base, previous_pre, constants):
        tokens, width = flat.shape
        mult = previous_pre.shape[1]
        out = delayed_pre_jax(
            to_np(flat).reshape(tokens, mult, width // mult),
            to_np(flat),
            to_np(fn),
            to_np(scale),
            to_np(base),
            to_np(previous_pre),
            *constants,
        )
        return [torch.from_numpy(np.asarray(o)) for o in out]

    def post(x, flat, post_mix, comb):
        out = _post_jax(to_np(x), to_np(flat), to_np(post_mix), to_np(comb))
        return torch.from_numpy(np.asarray(out))

    return pre, post


def reference_rotate(x, position, freqs, inverse=False):
    table = freqs[position]
    if inverse:
        table = table.conj()
    rope = 2 * table.shape[-1]
    tail = torch.view_as_complex(x[..., -rope:].float().reshape(*x.shape[:-1], -1, 2))
    tail = torch.view_as_real(tail * table).reshape(*x.shape[:-1], rope)
    return torch.cat([x[..., :-rope], tail.to(x.dtype)], dim=-1)


def reference_context(drafter, features, position):
    """Ring-rounded context KV of one committed position, every stage."""
    main = drafter.main_norm(drafter.main_proj(features[None])[0])
    out = []
    for stage in drafter.stages:
        attn = stage.self_attn
        kv = attn.kv_norm(attn.wkv(main)[0])[0]
        out.append(reference_rotate(kv, position, attn.freqs_cis).bfloat16().float())
    return out


def reference_draft(drafter, committed, floor, bonus, n):
    """core DSpark draft step for one request; committed: position -> features."""
    pre_fn, post_fn = jax_mhc()
    width = drafter.gamma
    positions = list(range(n, n + width))
    ids = torch.tensor([bonus] + [drafter.noise_token_id] * (width - 1))
    ctx_positions = [p for p in range(n - drafter.window, n) if p >= max(floor, 0)]
    contexts = [reference_context(drafter, committed[p], p) for p in ctx_positions]
    x = drafter._embed(ids)
    residual = x[:, None, :].expand(-1, drafter.hc_mult, -1).reshape(width, -1)
    first = torch.zeros(width, drafter.hc_mult)
    first[:, 0] = 1
    carry = None  # (output, post, comb, pre)
    valid = torch.ones(width, dtype=torch.bool)

    def boundary(layer, prefix, residual, carry):
        if carry is not None:
            residual = post_fn(carry[0], residual, carry[1], carry[2])
        previous = first if carry is None else carry[3]
        constants = (
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            2.0,
            layer.hc_sinkhorn_iters,
        )
        post, comb, value, pre = pre_fn(
            residual,
            getattr(layer, f"hc_{prefix}_fn"),
            getattr(layer, f"hc_{prefix}_scale"),
            getattr(layer, f"hc_{prefix}_base"),
            previous,
            constants,
        )
        return residual, value, (post, comb, pre)

    for index, stage in enumerate(drafter.stages):
        attn = stage.self_attn
        residual, value, mix = boundary(stage, "attn", residual, carry)
        h = stage.input_layernorm(value)
        block_kv = [
            reference_rotate(
                attn.kv_norm(attn.wkv(h[i : i + 1])[0])[0], positions[i], attn.freqs_cis
            )
            for i in range(width)
        ]
        keys = torch.stack([c[index] for c in contexts] + block_kv).float()
        out_rows = []
        for i in range(width):
            q = attn.wq_b(attn.q_norm(attn.wq_a(h[i : i + 1])[0]))[0]
            q = reference_rotate(q.view(HEADS, HEAD_DIM), positions[i], attn.freqs_cis)
            scores = (q.float() @ keys.T) * attn.softmax_scale
            logits = torch.cat([scores, attn.attn_sink.float()[:, None]], dim=1)
            probs = logits.softmax(-1)[:, :-1]
            o = reference_rotate(
                (probs @ keys).to(h.dtype), positions[i], attn.freqs_cis, True
            )
            grouped = o.reshape(attn.n_local_groups, -1)
            weight = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
            projected = torch.einsum("gd,grd->gr", grouped.float(), weight.float())
            out_rows.append(attn.wo_b(projected.reshape(1, -1).to(h.dtype))[0][0])
        carry = (torch.stack(out_rows), *mix[:2], mix[2])
        residual, value, mix = boundary(stage, "ffn", residual, carry)
        ffn = stage.mlp(stage.post_attention_layernorm(value), ids, valid)
        carry = (ffn, *mix[:2], mix[2])
    residual = post_fn(carry[0], residual, carry[1], carry[2])
    hidden = drafter.norm(_collapse(residual, carry[3]))
    base = (hidden @ drafter._lm_head.weight.T)[:, :VOCAB].float()
    previous, drafts = bonus, []
    for step in range(width):
        bias = (drafter.markov_w1[previous] @ drafter.markov_w2.T).float()
        previous = int(torch.argmax(base[step] + bias))
        drafts.append(previous)
    return hidden, drafts


@torch.library.custom_op("dspark_test::rope", mutates_args=())
def _test_rope_op(
    x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor, inverse: bool
) -> torch.Tensor:
    with patch.object(dspark, "_DEVICE_OPS", None):
        return dspark.rotate(x, positions, freqs, inverse=inverse)


_test_rope_op.register_fake(lambda x, *_: torch.empty_like(x))


@torch.library.custom_op("dspark_test::project", mutates_args=())
def _test_project_op(
    o: torch.Tensor, weight: torch.Tensor, groups: int
) -> torch.Tensor:
    grouped = o.reshape(o.shape[0], groups, -1)
    weight = weight.view(groups, weight.shape[0] // groups, -1)
    out = torch.einsum("tgd,grd->tgr", grouped.float(), weight.float())
    return out.reshape(o.shape[0], -1).to(o.dtype)


_test_project_op.register_fake(
    lambda o, weight, groups: o.new_empty((o.shape[0], weight.shape[0]))
)

# Pre-registered custom ops in place of the TPU leaves (what prepare_tpu_ops does).
TEST_DEVICE_OPS = dspark.DeviceOps(
    rope=lambda x, positions, freqs, inverse: _test_rope_op(
        x, positions, freqs, inverse
    ),
    project=lambda o, layer: _test_project_op(
        o, layer.wo_a.weight, layer.n_local_groups
    ),
)


def _lazy_project_ops():
    """A project leaf that registers its custom op on first call, while tracing."""
    import functools

    @functools.cache
    def register():
        @torch.library.custom_op("dspark_test::lazy_project", mutates_args=())
        def lazy(o: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
            return _test_project_op(o, weight, groups)

        lazy.register_fake(
            lambda o, weight, groups: o.new_empty((o.shape[0], weight.shape[0]))
        )
        return lazy

    return TEST_DEVICE_OPS._replace(
        project=lambda o, layer: register()(o, layer.wo_a.weight, layer.n_local_groups)
    )


class TestDsv41DsparkDrafter(CustomTestCase):
    def _compile_draft_and_inject(self, ops):
        """torch.compile(fullgraph=True) of the worker's draft and inject programs."""
        import torch._dynamo

        drafter = build_drafter()
        self._seed_contexts(drafter)
        args = (
            torch.tensor([11, 23, 0]),
            torch.tensor([309, 52, 0]),
            torch.tensor([1, 2, 0]),
            torch.tensor([True, True, False]),
        )
        inject_args = (
            torch.randn(6, HIDDEN * STAGES),
            torch.arange(309, 315),
            torch.ones(6, dtype=torch.int64),
            torch.tensor([True] * 3 + [False] * 3),
        )
        torch._dynamo.reset()
        with torch.no_grad(), cpu_parallel(), patch.object(dspark, "_DEVICE_OPS", None):
            eager = drafter.draft(*args)
            dspark.set_device_ops(ops)
            draft = torch.compile(drafter.draft, backend="eager", fullgraph=True)
            inject = torch.compile(drafter.inject, backend="eager", fullgraph=True)
            compiled = draft(*args)
            inject(*inject_args)
        return eager, compiled

    def test_draft_and_inject_compile_as_full_graphs(self):
        eager, compiled = self._compile_draft_and_inject(TEST_DEVICE_OPS)
        self.assertEqual(compiled[0].tolist(), eager[0].tolist())
        torch.testing.assert_close(compiled[1], eager[1], atol=1e-4, rtol=1e-4)

    def test_lazily_registered_leaf_breaks_the_graph(self):
        # Positive control: registering an op while tracing is a graph break,
        # which fullgraph turns into an error.
        import torch._dynamo.exc

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile_draft_and_inject(_lazy_project_ops())

    def test_tpu_tensors_need_prepared_ops(self):
        fake = SimpleNamespace(device=SimpleNamespace(type="tpu"))
        with patch.object(dspark, "_DEVICE_OPS", None), self.assertRaises(RuntimeError):
            dspark._device_ops(fake)

    def test_torch_mhc_port_matches_jax_kernels(self):
        torch.manual_seed(1)
        tokens, mult, dim = 5, 4, 16
        flat = torch.randn(tokens, mult * dim)
        fn = torch.randn(mult * (mult + 2), mult * dim) * 0.1
        scale, base = torch.rand(3) + 0.5, torch.randn(mult * (mult + 2)) * 0.3
        previous = torch.rand(tokens, mult)
        constants = (1e-6, 1e-6, 1e-6, 2.0, 20)
        pre_fn, post_fn = jax_mhc()
        expected = pre_fn(flat, fn, scale, base, previous, constants)
        got = torch_pre(flat, fn, scale, base, previous, *constants)
        for e, g in zip(expected, got):
            torch.testing.assert_close(g.float(), e.float(), atol=2e-5, rtol=2e-5)
        x = torch.randn(tokens, dim)
        torch.testing.assert_close(
            torch_post(x, flat, got[0], got[1]),
            post_fn(x, flat, got[0], got[1]),
            atol=2e-5,
            rtol=2e-5,
        )

    def _seed_contexts(self, drafter):
        """Two live requests (one past the ring wrap, one with a radix floor)."""
        torch.manual_seed(7)
        width = HIDDEN * STAGES
        committed = {1: {}, 2: {}}
        bad = {}
        # Request A (slot 1): extend [0, 180) + [180, 300), then two verify steps.
        # Request B (slot 2): radix prefix up to 40, extend [40, 52).
        chunks = [((1,), [0], [180], [0]), ((1, 2), [180, 40], [120, 12], [0, 40])]
        for slots, prefix_lens, extend_lens, floors in chunks:
            host = extend_rows(
                prefix_lens=prefix_lens,
                extend_lens=extend_lens,
                slots=list(slots),
                floors=floors,
                tails=[[0, 0, 0]] * len(slots),
                rows=512,
            )
            feats = torch.randn(512, width)
            for row in range(sum(extend_lens)):
                slot, pos = int(host["slots"][row]), int(host["positions"][row])
                committed[slot][pos] = feats[row]
            drafter.set_floor(host["slots"], host["floors"])
            drafter.inject(feats, host["positions"], host["slots"], host["write"])
        # Verify steps for A: 6 rows at 300.. accept 3, then 6 rows at 303.. accept 6.
        for start, accept in ((300, 3), (303, 6)):
            feats = torch.randn(6, width)
            slots = torch.tensor([1])
            row_slots, write = commit_rows(slots, torch.tensor([accept]), 6)
            positions = torch.arange(start, start + 6)
            for row in range(6):
                (committed[1] if row < accept else bad)[start + row] = feats[row]
            drafter.inject(feats, positions, row_slots, write)
        return committed, bad

    def test_block_draft_matches_reference(self):
        drafter = build_drafter()
        committed, bad = self._seed_contexts(drafter)
        # A's rejected rows 303..305 of the first step were rewritten by the second.
        self.assertTrue(all(p in committed[1] for p in bad))
        bonus = torch.tensor([11, 23, 0])
        seq_lens = torch.tensor([309, 52, 0])
        slots = torch.tensor([1, 2, 0])
        live = torch.tensor([True, True, False])
        with torch.no_grad(), cpu_parallel():
            drafts, hidden = drafter.draft(bonus, seq_lens, slots, live)
            hidden = hidden.view(3, GAMMA, -1)
            for row, (slot, floor) in enumerate(((1, 0), (2, 40))):
                ref_hidden, ref_drafts = reference_draft(
                    drafter, committed[slot], floor, int(bonus[row]), int(seq_lens[row])
                )
                torch.testing.assert_close(
                    hidden[row], ref_hidden, atol=2e-4, rtol=2e-4
                )
                self.assertEqual(drafts[row].tolist(), ref_drafts)

    def test_extend_draft_verify_commit_flow_real_batch(self):
        """The worker's host + device steps on real scheduler objects.

        extend (ExtendContext on a real ScheduleBatch/Req -> set_floor, tails,
        inject) -> draft -> verify Engram rows from the tails table -> commit
        (commit_rows / inject / advance_tails) -> draft again; each draft equals
        the reference over the committed features, each Engram window equals
        hashing the committed tokens.
        """
        from array import array

        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang_tpu.engram import capture_inputs, verify_inputs
        from sglang_tpu.experimental.speculative.dspark.dsv41_dspark_worker import (
            ExtendContext,
            advance_tails,
        )

        torch.manual_seed(3)
        drafter = build_drafter()
        prompt = torch.randint(0, VOCAB - 1, (70,)).tolist()
        req = Req("r", "", array("q", prompt), SamplingParams())
        batch = ScheduleBatch(reqs=[req])
        batch.prefix_lens, batch.extend_lens = [0], [70]
        batch.req_pool_indices_cpu = torch.tensor([2])
        req.kv.req_pool_idx = 2
        tails = torch.zeros((4, 3), dtype=torch.int64)
        host = ExtendContext().rows(batch, tail=3, rows=128)
        feats = torch.randn(128, HIDDEN * STAGES)
        committed = {p: feats[p] for p in range(70)}
        drafter.set_floor(host["slots"], host["floors"])
        tails.index_copy_(0, host["slots"], host["tails"])
        drafter.inject(feats, host["positions"], host["slots"], host["write"])

        slots, live = torch.tensor([2]), torch.tensor([True])
        history = list(prompt)
        n, bonus = 70, 17
        with torch.no_grad(), cpu_parallel():
            for accept in (3, 6):
                drafts, _ = drafter.draft(
                    torch.tensor([bonus]), torch.tensor([n]), slots, live
                )
                _, ref = reference_draft(drafter, committed, 0, bonus, n)
                self.assertEqual(drafts[0].tolist(), ref)
                verify = torch.cat([torch.tensor([bonus]), drafts[0]])
                # Engram rows: tails + verify tokens vs the same tokens committed.
                got = verify_inputs(
                    tails[slots].numpy(), verify[None].numpy(), np.array([n]), 4
                )
                req.output_ids.extend(verify.tolist())
                want = capture_inputs([req], [n], [len(verify)], 4)
                del req.output_ids[len(req.output_ids) - len(verify) :]
                np.testing.assert_array_equal(got.tokens, want.tokens)
                # Commit accept tokens (r, d_1..d_{accept-1}); new root is arbitrary.
                step = torch.randn(6, HIDDEN * STAGES)
                row_slots, write = commit_rows(slots, torch.tensor([accept]), 6)
                drafter.inject(step, torch.arange(n, n + 6), row_slots, write)
                advance_tails(tails, slots, torch.tensor([accept]), verify)
                for j in range(accept):
                    committed[n + j] = step[j]
                history += verify[:accept].tolist()
                req.output_ids.extend(verify[:accept].tolist())
                n, bonus = n + accept, 23 + accept
                self.assertEqual(tails[2].tolist(), history[-3:])

    def test_context_validity(self):
        drafter = build_drafter()
        self._seed_contexts(drafter)
        seq_lens = torch.tensor([309, 52, 5])
        contexts, valid = drafter.gather_context(
            seq_lens, torch.tensor([1, 2, 3]), torch.tensor([True, True, True])
        )
        self.assertTrue(valid[0].all())  # 277..308, across the ring wrap
        self.assertEqual(valid[1].tolist(), [p >= 40 for p in range(20, 52)])
        self.assertFalse(valid[2].any())  # slot 3 was never seeded
        self.assertEqual(len(contexts), STAGES)

    def test_dead_rows_do_not_write_or_read(self):
        drafter = build_drafter()
        before = [ring.clone() for ring in drafter.ring]
        feats = torch.randn(4, HIDDEN * STAGES)
        drafter.inject(
            feats,
            torch.arange(4),
            torch.tensor([1, 1, 2, 2]),
            torch.zeros(4, dtype=torch.bool),
        )
        for old, new in zip(before, drafter.ring):
            torch.testing.assert_close(old[:-1], new[:-1])  # only the sink row moves

    def test_weight_names_cover_every_parameter(self):
        drafter = build_drafter()
        params = dict(drafter.named_parameters())
        names = []
        for stage in range(STAGES):
            for leaf in (
                "attn.wq_a.weight",
                "attn.wq_b.weight",
                "attn.wkv.weight",
                "attn.wo_a.weight",
                "attn.wo_b.weight",
                "attn.q_norm.weight",
                "attn.kv_norm.weight",
                "attn.attn_sink",
                "attn_norm.weight",
                "ffn_norm.weight",
                "ffn.gate.weight",
                "ffn.gate.bias",
                "ffn.gate.bias_vl",
                "ffn.shared_experts.w1.weight",
                "ffn.shared_experts.w2.weight",
                "ffn.shared_experts.w3.weight",
                "hc_attn_fn",
                "hc_attn_base",
                "hc_attn_scale",
                "hc_ffn_fn",
                "hc_ffn_base",
                "hc_ffn_scale",
            ):
                names.append(f"mtp.{stage}.{leaf}")
            for expert in range(4):
                for proj in ("w1", "w2", "w3"):
                    names.append(f"mtp.{stage}.ffn.experts.{expert}.{proj}.weight")
        names += [
            "mtp.0.main_proj.weight",
            "mtp.0.main_norm.weight",
            "mtp.2.norm.weight",
            "mtp.2.markov_head.embed.weight",
            "mtp.2.markov_head.head.weight",
            "mtp.2.confidence_head.proj.weight",
        ]

        def tensor_for(name):
            mapped = drafter.remap(name)
            if mapped is None:
                return torch.zeros(1)
            if ".experts." in mapped:
                head, tail = mapped.split(".mlp.experts.")
                proj = tail.split(".")[1]
                family = "w2" if proj == "down_proj" else "w13"
                shape = params[f"{head}.mlp.experts.{family}_weight"].shape[1:]
                return torch.randn(shape[0] // (1 if family == "w2" else 2), shape[1])
            for proj in (".gate_proj.", ".up_proj."):
                if proj in mapped:
                    full = params[mapped.replace(proj, ".gate_up_proj.")]
                    return torch.randn(full.shape[0] // 2, full.shape[1])
            return torch.randn(params[mapped].shape)

        loaded = drafter.load_weights((n, tensor_for(n)) for n in names)
        self.assertEqual(loaded, set(params))
        self.assertEqual(
            drafter.remap("mtp.1.ffn.experts.7.w2.scale"),
            "stages.1.mlp.experts.7.down_proj.weight_scale_inv",
        )
        self.assertEqual(
            drafter.remap("mtp.0.attn.wo_a.scale"),
            "stages.0.self_attn.wo_a.weight_scale_inv",
        )
        self.assertEqual(
            drafter.remap("mtp.0.main_proj.scale"), "main_proj.weight_scale_inv"
        )
        self.assertIsNone(drafter.remap("layers.3.attn.wq_a.weight"))

    def test_loader_keeps_local_draft_experts_only_when_enabled(self):
        weight_map = {
            f"layers.0.ffn.experts.{e}.w{w}.{k}": "a"
            for e in range(8)
            for w in (1, 2, 3)
            for k in ("weight", "scale")
        }
        weight_map.update(
            {
                f"mtp.{s}.ffn.experts.{e}.w{w}.weight": "m"
                for s in range(3)
                for e in range(4)
                for w in (1, 2, 3)
            }
        )
        weight_map["mtp.0.main_proj.weight"] = "m"
        off = plan_expert_reads(
            weight_map, num_layers=1, num_experts=8, ep_size=2, ep_rank=1
        )
        self.assertFalse(any(n.startswith("mtp.") for n in off.tensor_names))
        on = plan_expert_reads(
            weight_map,
            num_layers=1,
            num_experts=8,
            ep_size=2,
            ep_rank=1,
            draft_experts=4,
        )
        drafts = {n for n in on.tensor_names if n.startswith("mtp.")}
        self.assertIn("mtp.0.main_proj.weight", drafts)
        experts = {int(n.split(".")[4]) for n in drafts if ".experts." in n}
        self.assertEqual(experts, {2, 3})


if __name__ == "__main__":
    unittest.main()
