"""CPU unit tests for Qwen3.5 / Qwen3.8 GDN Q/K head replication (attn_tp > num_k_heads).

Qwen3.8-2.4T-A95B-FP8 has ``linear_num_key_heads = 16`` and ``linear_num_value_heads = 128``
with ``num_key_value_heads = 4`` on full-attention layers. Serving across a 4-host
tpu7x 2x2x4 slice (``tp_size = 32, dcp_size = 8``) replicates each GDN Q/K head
``qk_rep = 32 // 16 = 2`` times while sharding the 128 V/Z heads to 4 per rank.

Verifies:
1. ``Mamba2StateShape.create`` state sizing at ``tp = 1, 16, 32, 64`` and rejection
   of non-divisible ``tp = 24``.
2. ``mamba_v2_sharded_weight_loader`` replication for both ``n_groups == 1`` and
   ``n_groups > 1``.
3. ``Qwen3_5GatedDeltaNet`` weight loading across split ``in_proj_qkv`` + ``in_proj_z``
   (Qwen3.8 checkpoint layout), 4-way split, fused ``in_proj_qkvz``, and block-FP8
   scale tensors.
4. End-to-end numerical equivalence of the full GDN layer output (summed across TP
   ranks) across ``tp = 1, 4, 8 (2x Q/K rep), 16 (4x Q/K rep)``.
5. ``Qwen3_5Attention`` head geometry under ``tp = 32, dcp = 8`` for Qwen3.8.
"""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt import runtime_context as rc
from sglang.srt.configs.mamba_utils import Mamba2StateShape
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.parameter import BlockQuantScaleParameter
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _gdn_config(
    *,
    hidden_size: int = 64,
    num_k_heads: int = 4,
    num_v_heads: int = 16,
    head_k_dim: int = 16,
    head_v_dim: int = 16,
    conv_kernel: int = 4,
    dtype: torch.dtype = torch.float32,
):
    return SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=hidden_size,
        linear_num_key_heads=num_k_heads,
        linear_num_key_heads_cpu=num_k_heads,
        linear_num_value_heads=num_v_heads,
        linear_num_value_heads_cpu=num_v_heads,
        linear_key_head_dim=head_k_dim,
        linear_value_head_dim=head_v_dim,
        linear_conv_kernel_dim=conv_kernel,
        hidden_act="silu",
        output_gate_type="swish",
        rms_norm_eps=1e-6,
        torch_dtype=dtype,
    )


def _build_gdn_layer(cfg, tp_size: int, tp_rank: int):
    from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

    with (
        get_context().override_server_args(tp_size=tp_size),
        rc.get_parallel().override(tp_rank=tp_rank, attn_tp_rank=tp_rank),
        torch.no_grad(),
    ):
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(cfg.torch_dtype)
        try:
            return Qwen3_5GatedDeltaNet(config=cfg, layer_id=0, quant_config=None)
        finally:
            torch.set_default_dtype(old_dtype)


def _load_gdn_weights(layer, weights: dict, tp_size: int, tp_rank: int, mode: str):
    """Load checkpoint weights into one TP rank of ``Qwen3_5GatedDeltaNet``."""
    with (
        get_context().override_server_args(tp_size=tp_size),
        rc.get_parallel().override(tp_rank=tp_rank, attn_tp_rank=tp_rank),
        torch.no_grad(),
    ):
        qkvz_loader = layer.in_proj_qkvz.weight.weight_loader
        ba_loader = layer.in_proj_ba.weight.weight_loader
        q, k, v, z = weights["q"], weights["k"], weights["v"], weights["z"]
        if mode == "split_qkv_z":
            qkvz_loader(
                layer.in_proj_qkvz.weight, torch.cat([q, k, v], dim=0), (0, 1, 2)
            )
            qkvz_loader(layer.in_proj_qkvz.weight, z, 3)
        elif mode == "four_way":
            for idx, part in enumerate((q, k, v, z)):
                qkvz_loader(layer.in_proj_qkvz.weight, part, idx)
        elif mode == "fused":
            qkvz_loader(layer.in_proj_qkvz.weight, torch.cat([q, k, v, z], dim=0), None)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        ba_loader(layer.in_proj_ba.weight, weights["b"], 0)
        ba_loader(layer.in_proj_ba.weight, weights["a"], 1)
        layer.conv1d.weight.weight_loader(layer.conv1d.weight, weights["conv1d"])
        layer.A_log.weight_loader(layer.A_log, weights["A_log"])
        layer.dt_bias.weight_loader(layer.dt_bias, weights["dt_bias"])
        layer.norm.weight.copy_(weights["norm"])
        layer.out_proj.weight.weight_loader(layer.out_proj.weight, weights["out_proj"])


def _run_local_gdn_forward(layer, x: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch execution of one TP rank's GDN shard (exact GDN recurrence)."""
    with torch.no_grad():
        proj_qkvz = F.linear(x, layer.in_proj_qkvz.weight)
        proj_ba = F.linear(x, layer.in_proj_ba.weight)
        query, key, value, z, b, a = layer.fix_query_key_value_ordering(
            proj_qkvz, proj_ba
        )
        # Causal depthwise conv1d + SiLU on concatenated [query, key, value]
        T = x.shape[0]
        mixed_qkv = torch.cat([query, key, value.reshape(T, -1)], dim=-1)
        K = layer.conv_kernel_size
        padded = F.pad(mixed_qkv.transpose(0, 1).unsqueeze(0), (K - 1, 0))
        conv_out = F.conv1d(
            padded, layer.conv1d.weight, bias=None, groups=mixed_qkv.shape[-1]
        )
        conv_out = F.silu(conv_out.squeeze(0).transpose(0, 1))

        k_tp = layer.local_key_dim
        v_tp = layer.local_num_v_heads * layer.head_v_dim
        q_c, k_c, v_c = conv_out.split([k_tp, k_tp, v_tp], dim=-1)
        hk = layer.local_num_k_heads
        hv = layer.local_num_v_heads
        dk = layer.head_k_dim
        dv = layer.head_v_dim
        group_ratio = hv // hk

        q_h = F.normalize(q_c.view(T, hk, dk), p=2.0, dim=-1).repeat_interleave(
            group_ratio, dim=1
        )
        k_h = F.normalize(k_c.view(T, hk, dk), p=2.0, dim=-1).repeat_interleave(
            group_ratio, dim=1
        )
        v_h = v_c.view(T, hv, dv)

        beta = torch.sigmoid(b.float())
        g = -torch.exp(layer.A_log.float()) * F.softplus(
            a.float() + layer.dt_bias.float()
        )
        scale = dk**-0.5

        state = torch.zeros(hv, dv, dk, dtype=torch.float32)
        outs = []
        for t in range(T):
            state = state * torch.exp(g[t]).view(hv, 1, 1)
            k_t = k_h[t].float()
            v_t = v_h[t].float()
            q_t = q_h[t].float() * scale
            kv_mem = torch.einsum("hvk,hk->hv", state, k_t)
            delta = (v_t - kv_mem) * beta[t].unsqueeze(-1)
            state = state + torch.einsum("hv,hk->hvk", delta, k_t)
            out_t = torch.einsum("hvk,hk->hv", state, q_t)
            outs.append(out_t)
        core_out = torch.stack(outs, dim=0).to(x.dtype)
        normed = layer.norm(core_out.reshape(-1, dv), z.reshape(-1, dv)).reshape(
            T, hv * dv
        )
        return F.linear(normed, layer.out_proj.weight)


class TestQwen35GdnQkReplication(CustomTestCase):
    def test_mamba2_state_shape_qwen38_geometry(self):
        # Qwen3.8-2.4T: n_groups=16, num_heads=128, d=128, K=4
        for tp, expected_conv_dim, expected_hv in (
            (1, 2048 * 2 + 16384, 128),
            (16, 128 * 2 + 1024, 8),
            (32, 128 * 2 + 512, 4),
            (64, 128 * 2 + 256, 2),
        ):
            with self.subTest(tp=tp):
                shape = Mamba2StateShape.create(
                    tp_world_size=tp,
                    intermediate_size=16384,
                    n_groups=16,
                    num_heads=128,
                    head_dim=128,
                    state_size=128,
                    conv_kernel=4,
                    conv_shard_groups=[2048, 2048, 16384],
                )
                self.assertEqual(shape.conv, [(expected_conv_dim, 3)])
                self.assertEqual(shape.temporal, (expected_hv, 128, 128))
                self.assertEqual(shape.num_k_heads_per_tp, max(1, 16 // tp))

        with self.assertRaises(AssertionError):
            Mamba2StateShape.create(
                tp_world_size=24,
                intermediate_size=16384,
                n_groups=16,
                num_heads=128,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            )

    def test_mamba_v2_sharded_weight_loader_replicates_groups(self):
        g = torch.Generator().manual_seed(7)
        # 4 K groups of dim 8 -> key_dim=32; value_dim=64; TP=8 (2x Q/K rep)
        q_full = torch.randn(32, 1, 4, generator=g)
        k_full = torch.randn(32, 1, 4, generator=g)
        v_full = torch.randn(64, 1, 4, generator=g)
        loaded = torch.cat([q_full, k_full, v_full], dim=0)

        spec = [(64, 32, True), (64, 32, True), (64, 0, False)]
        for tp_rank in range(8):
            param = torch.empty(24, 1, 4)
            mamba_v2_sharded_weight_loader(spec, tp_size=8, tp_rank=tp_rank)(
                param, loaded
            )
            qk_group = tp_rank // 2
            self.assertTrue(
                torch.equal(param[:8], q_full[qk_group * 8 : (qk_group + 1) * 8])
            )
            self.assertTrue(
                torch.equal(param[8:16], k_full[qk_group * 8 : (qk_group + 1) * 8])
            )
            self.assertTrue(
                torch.equal(param[16:24], v_full[tp_rank * 8 : (tp_rank + 1) * 8])
            )

        # Legacy n_groups == 1 replication across TP=4 stays identical.
        b_full = torch.randn(8, 1, 4, generator=g)
        c_full = torch.randn(8, 1, 4, generator=g)
        x_full = torch.randn(32, 1, 4, generator=g)
        loaded_1g = torch.cat([x_full, b_full, c_full], dim=0)
        spec_1g = [(32, 0, False), (32, 24, True), (32, 24, True)]
        for tp_rank in range(4):
            param = torch.empty(24, 1, 4)
            mamba_v2_sharded_weight_loader(spec_1g, tp_size=4, tp_rank=tp_rank)(
                param, loaded_1g
            )
            self.assertTrue(
                torch.equal(param[:8], x_full[tp_rank * 8 : (tp_rank + 1) * 8])
            )
            self.assertTrue(torch.equal(param[8:16], b_full))
            self.assertTrue(torch.equal(param[16:24], c_full))

    def test_checkpoint_formats_and_forward_equivalence_across_tp(self):
        cfg = _gdn_config()
        g = torch.Generator().manual_seed(42)
        key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim  # 64
        val_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim  # 256
        weights = {
            "q": torch.randn(key_dim, cfg.hidden_size, generator=g) * 0.1,
            "k": torch.randn(key_dim, cfg.hidden_size, generator=g) * 0.1,
            "v": torch.randn(val_dim, cfg.hidden_size, generator=g) * 0.1,
            "z": torch.randn(val_dim, cfg.hidden_size, generator=g) * 0.1,
            "b": torch.randn(cfg.linear_num_value_heads, cfg.hidden_size, generator=g)
            * 0.1,
            "a": torch.randn(cfg.linear_num_value_heads, cfg.hidden_size, generator=g)
            * 0.1,
            "conv1d": torch.randn(
                key_dim * 2 + val_dim, 1, cfg.linear_conv_kernel_dim, generator=g
            )
            * 0.1,
            "A_log": torch.randn(cfg.linear_num_value_heads, generator=g) * 0.1,
            "dt_bias": torch.randn(cfg.linear_num_value_heads, generator=g) * 0.1,
            "norm": torch.ones(cfg.linear_value_head_dim),
            "out_proj": torch.randn(cfg.hidden_size, val_dim, generator=g) * 0.1,
        }
        x = torch.randn(9, cfg.hidden_size, generator=g)

        ref_layer = _build_gdn_layer(cfg, tp_size=1, tp_rank=0)
        _load_gdn_weights(ref_layer, weights, tp_size=1, tp_rank=0, mode="split_qkv_z")
        ref_out = _run_local_gdn_forward(ref_layer, x)

        for tp_size in (4, 8, 16):
            rank_outputs = []
            for tp_rank in range(tp_size):
                by_mode = {}
                for mode in ("split_qkv_z", "four_way", "fused"):
                    layer = _build_gdn_layer(cfg, tp_size=tp_size, tp_rank=tp_rank)
                    _load_gdn_weights(layer, weights, tp_size, tp_rank, mode=mode)
                    by_mode[mode] = layer
                # All three checkpoint formats produce identical sharded weights.
                self.assertTrue(
                    torch.equal(
                        by_mode["split_qkv_z"].in_proj_qkvz.weight,
                        by_mode["four_way"].in_proj_qkvz.weight,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        by_mode["split_qkv_z"].in_proj_qkvz.weight,
                        by_mode["fused"].in_proj_qkvz.weight,
                    )
                )
                rank_outputs.append(_run_local_gdn_forward(by_mode["split_qkv_z"], x))
            tp_sum = torch.stack(rank_outputs, dim=0).sum(dim=0)
            torch.testing.assert_close(
                tp_sum,
                ref_out,
                rtol=1e-5,
                atol=1e-5,
                msg=f"GDN forward mismatch at tp_size={tp_size}",
            )

    def test_block_quant_scale_split_and_qk_replication(self):
        from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet

        # 16 K heads (dim 128 -> 2048), 128 V heads (dim 128 -> 16384), TP=32 (qk_rep=2),
        # block_size=[128, 128] -> per-rank scale blocks: [1, 1, 4, 4] = 10 rows.
        calls = []

        def fake_orig_loader(p, chunk, shard_id):
            calls.append((module.tp_group.rank_in_group, shard_id, chunk.clone()))

        orig_tp_group = SimpleNamespace(rank_in_group=5, world_size=32)
        module = SimpleNamespace(
            output_sizes=[4096, 4096, 16384, 16384],
            _checkpoint_output_sizes=[2048, 2048, 16384, 16384],
            _shard_tp_reps=(2, 2, 1, 1),
            tp_group=orig_tp_group,
            use_presharded_weights=False,
            quant_method=SimpleNamespace(
                quant_config=SimpleNamespace(weight_block_size=[128, 128])
            ),
        )
        scale_param = BlockQuantScaleParameter(
            data=torch.zeros(10, 64),
            output_dim=0,
            input_dim=1,
            weight_loader=fake_orig_loader,
        )
        wrapped = Qwen3_5GatedDeltaNet._make_packed_weight_loader(
            module, fake_orig_loader
        )
        # Checkpoint in_proj_qkv.weight_scale_inv has 16 + 16 + 128 = 160 rows.
        loaded_qkv_scale = torch.arange(160 * 64, dtype=torch.float32).reshape(160, 64)
        loaded_z_scale = torch.arange(128 * 64, dtype=torch.float32).reshape(128, 64)
        wrapped(scale_param, loaded_qkv_scale, (0, 1, 2))
        wrapped(scale_param, loaded_z_scale, 3)

        self.assertIs(module.tp_group, orig_tp_group)
        self.assertEqual(module.tp_group.rank_in_group, 5)
        self.assertEqual([c[:2] for c in calls], [(2, 0), (2, 1), (5, 2), (5, 3)])
        self.assertEqual([c[2].shape[0] for c in calls], [16, 16, 128, 128])

    def test_qwen38_full_attention_dcp8_tp32_geometry(self):
        from sglang.srt.distributed.utils import get_group_rank_size
        from sglang.srt.models.qwen3_5 import Qwen3_5AttentionDecoderLayer

        cfg = SimpleNamespace(
            model_type="qwen3_5_text",
            num_hidden_layers=4,
            full_attention_interval=4,
            hidden_size=8192,
            intermediate_size=8192,
            hidden_act="silu",
            num_attention_heads=64,
            num_attention_heads_cpu=64,
            num_key_value_heads=4,
            head_dim=256,
            rms_norm_eps=1e-6,
            attention_bias=False,
            attn_output_gate=True,
            max_position_embeddings=4096,
            partial_rotary_factor=0.25,
            rope_parameters={
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
        )
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = SimpleNamespace(model_type="qwen3_5_moe_text")
        model_config.hf_text_config = SimpleNamespace(num_key_value_heads=4)
        model_config.is_draft_model = False
        pool_kv_heads = model_config.get_num_kv_heads(32, dcp_size=8)
        self.assertEqual(pool_kv_heads, 1)

        with (
            get_context().override_server_args(
                tp_size=32,
                dcp_size=8,
                dcp_comm_backend="ag_rs",
                page_size=256,
                chunked_prefill_size=4096,
            ),
            rc.get_parallel().override(
                tp_rank=13,
                attn_tp_rank=13,
                dcp_rank=13 % 8,
                attn_dcp_rank=13 % 8,
                pp_rank=0,
            ),
            torch.device("meta"),
        ):
            attn = Qwen3_5AttentionDecoderLayer(
                config=cfg, layer_id=3, quant_config=None, build_stages=False
            )

        self.assertEqual(attn.num_heads, 2)
        self.assertEqual(attn.num_kv_heads, 1)
        self.assertEqual(attn.kv_tp_size, 4)
        self.assertEqual(attn.kv_tp_rank, 13 // 8)
        self.assertEqual(get_group_rank_size(attn.qkv_proj.kv_tp_group), (13 // 8, 4))
        self.assertEqual(attn.attn.tp_k_head_num, pool_kv_heads)
        self.assertEqual(attn.attn.tp_q_head_num % attn.attn.tp_k_head_num, 0)

    def test_dynamo_guard_stability_input_proj_and_text_wrapper(self):
        """Regression test for Dynamo guard stability under fail_on_recompile:
        1. _forward_input_proj on non-CPU must not evaluate LazyValue
           _fused_input_proj_cpu_enabled inside the first compiled trace
           (which previously mutated _creator=None and invalidated graph 0/0).
        2. Qwen3_5ForCausalLM (qwen3_5_text) and Qwen3_5TextForCausalLM (qwen3_5)
           must pass input_embeds by keyword rather than as a 4th positional None.
        """
        from unittest.mock import patch

        from sglang.srt.models import qwen3_5 as qwen3_5_mod
        from sglang.srt.models import qwen3_5_text as qwen3_5_text_mod

        gdn = _build_gdn_layer(_gdn_config(dtype=torch.bfloat16), tp_size=4, tp_rank=0)
        gdn.in_proj_qkvz.weight.data.normal_(0, 0.02)
        gdn.in_proj_ba.weight.data.normal_(0, 0.02)
        x = torch.randn(4, 64, dtype=torch.bfloat16)

        with patch.object(qwen3_5_mod, "_is_cpu", False):
            self.assertIsNotNone(gdn._fused_input_proj_cpu_enabled._creator)
            compiled_proj = torch.compile(
                gdn._forward_input_proj, backend="eager", fullgraph=False
            )
            qkvz_0, ba_0 = compiled_proj(x)
            self.assertIsNotNone(gdn._fused_input_proj_cpu_enabled._creator)
            with torch.compiler.set_stance("fail_on_recompile"):
                qkvz_1, ba_1 = compiled_proj(x)
            torch.testing.assert_close(qkvz_0, qkvz_1)
            torch.testing.assert_close(ba_0, ba_1)

        recorded_calls = []

        def spy_backbone(*args, **kwargs):
            recorded_calls.append((args, kwargs))
            return args[0].to(torch.bfloat16).unsqueeze(-1)

        fb = SimpleNamespace(mrope_positions=None)
        ids = torch.tensor([3, 5], dtype=torch.int64)
        pos = torch.tensor([1, 2], dtype=torch.int64)

        outer = qwen3_5_text_mod.Qwen3_5ForCausalLM.__new__(
            qwen3_5_text_mod.Qwen3_5ForCausalLM
        )
        torch.nn.Module.__init__(outer)
        outer.is_mrope_enabled = False
        outer.pp_group = SimpleNamespace(is_last_rank=True)
        outer.model = spy_backbone
        outer.lm_head = None
        outer.logits_processor = lambda input_ids, hs, lm_head, batch, aux: hs
        outer.forward(ids, pos, fb)

        text_outer = qwen3_5_mod.Qwen3_5TextForCausalLM.__new__(
            qwen3_5_mod.Qwen3_5TextForCausalLM
        )
        torch.nn.Module.__init__(text_outer)
        text_outer.model = spy_backbone
        text_outer.lm_head = None
        text_outer.logits_processor = lambda input_ids, hs, lm_head, batch: hs
        text_outer.forward(ids, pos, fb)

        self.assertEqual(len(recorded_calls), 2)
        for call_args, call_kwargs in recorded_calls:
            self.assertEqual(len(call_args), 3)
            self.assertIn("input_embeds", call_kwargs)
            self.assertIsNone(call_kwargs["input_embeds"])


if __name__ == "__main__":
    unittest.main()
