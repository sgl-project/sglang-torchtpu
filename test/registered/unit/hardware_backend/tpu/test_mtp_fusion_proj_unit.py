# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for column-parallel MTP fusion projection (eh_proj)."""

from __future__ import annotations

import pkgutil
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.plugins.hook_registry import HookType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import mtp_fusion_proj
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks.mtp_fusion_proj import (
    shard_deepseek_nextn_eh_proj as hook_shard_eh_proj,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_stub_model(
    in_features: int = 64,
    out_features: int = 32,
    *,
    bias: bool = False,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> nn.Module:
    model = nn.Module()
    model.eh_proj = nn.Linear(
        in_features, out_features, bias=bias, dtype=dtype, device=device
    )
    return model


def _parallel(
    *,
    tp_rank: int = 0,
    tp_size: int = 8,
    attn_tp_rank: int = 0,
    attn_tp_size: int = 1,
    attn_tp_group: object = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        tp_rank=tp_rank,
        tp_size=tp_size,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_tp_group=attn_tp_group,
    )


class TestMtpFusionProjHookRegistration(CustomTestCase):
    def test_hook_is_registered_and_target_resolves(self):
        target = "sglang.srt.models.deepseek_nextn.DeepseekModelNextN.__init__"
        self.assertIn((target, hook_shard_eh_proj, HookType.AFTER), HOOKS)
        self.assertTrue(callable(pkgutil.resolve_name(target)))


class TestColumnParallelMtpFusionProj(CustomTestCase):
    def _build_sharded_ranks(
        self,
        full: nn.Linear,
        tp: int,
        *,
        dp_attn: bool = False,
        attn_tp_group: object = None,
    ) -> list[mtp_fusion_proj.ColumnParallelMtpFusionProj]:
        ranks = []
        with (
            mock.patch.object(
                mtp_fusion_proj, "is_dp_attention_enabled", return_value=dp_attn
            ),
            mock.patch(
                "sglang.srt.layers.linear.get_forward",
                return_value=SimpleNamespace(sp_active=False),
            ),
        ):
            for r in range(tp):
                par = _parallel(
                    tp_rank=r,
                    tp_size=tp,
                    attn_tp_rank=r,
                    attn_tp_size=tp,
                    attn_tp_group=attn_tp_group,
                )
                model = _make_stub_model(
                    full.in_features,
                    full.out_features,
                    dtype=full.weight.dtype,
                )
                with mock.patch.object(
                    mtp_fusion_proj, "get_parallel", return_value=par
                ):
                    hook_shard_eh_proj(None, model, prefix="model")
                proj = model.eh_proj
                self.assertIsInstance(
                    proj, mtp_fusion_proj.ColumnParallelMtpFusionProj
                )
                param = proj.weight
                param.weight_loader(param, full.weight.detach())
                ranks.append(proj)
        return ranks

    def test_matches_replicated_linear_exactly(self):
        for tp, shape, dtype in (
            (8, (5, 64), torch.bfloat16),
            (8, (32, 64), torch.bfloat16),
            (4, (2, 6, 32), torch.float32),
        ):
            with self.subTest(tp=tp, shape=shape, dtype=dtype):
                hidden = shape[-1] // 2
                gen = torch.Generator().manual_seed(tp * 13 + len(shape))
                full = nn.Linear(2 * hidden, hidden, bias=False, dtype=dtype)
                with torch.no_grad():
                    full.weight.copy_(
                        torch.randn((hidden, 2 * hidden), generator=gen, dtype=dtype)
                    )
                x = torch.randn(shape, generator=gen, dtype=dtype)
                expected = full(x)

                ranks = self._build_sharded_ranks(full, tp)
                part = hidden // tp
                for r, proj in enumerate(ranks):
                    self.assertEqual(proj.prefix, "model.eh_proj")
                    self.assertEqual(proj.weight.shape, (part, 2 * hidden))
                    self.assertEqual(proj.weight.dtype, dtype)
                    self.assertTrue(
                        torch.equal(proj.weight, full.weight[r * part : (r + 1) * part])
                    )

                partials = [
                    torch.nn.functional.linear(x, proj.weight) for proj in ranks
                ]
                gathered = torch.cat(partials, dim=-1)
                captured_locals: list[torch.Tensor] = []

                def _fake_gather(local: torch.Tensor) -> torch.Tensor:
                    captured_locals.append(local)
                    return gathered

                with (
                    mock.patch(
                        "sglang.srt.layers.linear.get_forward",
                        return_value=SimpleNamespace(sp_active=False),
                    ),
                    mock.patch(
                        "sglang.srt.layers.linear.tensor_model_parallel_all_gather",
                        side_effect=_fake_gather,
                    ) as tp_gather,
                ):
                    for r, proj in enumerate(ranks):
                        out = proj(x)
                        self.assertIsInstance(out, torch.Tensor)
                        self.assertEqual(out.shape, expected.shape)
                        self.assertTrue(
                            torch.equal(captured_locals[r], partials[r]), f"rank {r}"
                        )
                        if dtype == torch.bfloat16:
                            self.assertTrue(torch.equal(out, expected), f"rank {r}")
                        else:
                            torch.testing.assert_close(
                                out, expected, atol=1e-6, rtol=1e-6
                            )
                    self.assertEqual(tp_gather.call_count, tp)

    def test_dp_attention_uses_attn_tp_group_and_skips_when_attn_tp_is_one(self):
        # DP8 + TP8 -> attn_tp_size == 1: distinct token batches, keep nn.Linear.
        model_dp8 = _make_stub_model(64, 32, dtype=torch.bfloat16)
        orig_linear = model_dp8.eh_proj
        with (
            mock.patch.object(
                mtp_fusion_proj, "is_dp_attention_enabled", return_value=True
            ),
            mock.patch.object(
                mtp_fusion_proj,
                "get_parallel",
                return_value=_parallel(tp_size=8, attn_tp_size=1),
            ),
        ):
            hook_shard_eh_proj(None, model_dp8, prefix="model")
        self.assertIs(model_dp8.eh_proj, orig_linear)

        # PCP (attn_cp_size > 1) with attn_tp_size == 1: distinct zigzag rows, skip.
        model_cp8 = _make_stub_model(64, 32, dtype=torch.bfloat16)
        orig_cp_linear = model_cp8.eh_proj
        cp_par = _parallel(tp_size=8, attn_tp_size=1)
        cp_par.attn_cp_size = 8
        with (
            mock.patch.object(
                mtp_fusion_proj, "is_dp_attention_enabled", return_value=False
            ),
            mock.patch.object(
                mtp_fusion_proj, "get_parallel", return_value=cp_par
            ),
        ):
            hook_shard_eh_proj(None, model_cp8, prefix="model")
        self.assertIs(model_cp8.eh_proj, orig_cp_linear)

        # DP2 x AttnTP4 -> shard 4-way within attn_tp_group, gather on attn_tp_group.
        hidden = 16
        full = nn.Linear(2 * hidden, hidden, bias=False, dtype=torch.float32)
        attn_group = mock.Mock()
        ranks = self._build_sharded_ranks(
            full, tp=4, dp_attn=True, attn_tp_group=attn_group
        )
        x = torch.randn(3, 2 * hidden)
        expected = full(x)
        partials = [torch.nn.functional.linear(x, proj.weight) for proj in ranks]
        attn_group.all_gather.side_effect = lambda local, dim=-1: torch.cat(
            partials, dim=dim
        )
        par = _parallel(tp_size=8, attn_tp_size=4, attn_tp_group=attn_group)
        with (
            mock.patch.object(mtp_fusion_proj, "get_parallel", return_value=par),
            mock.patch(
                "sglang.srt.layers.linear.get_forward",
                return_value=SimpleNamespace(sp_active=False),
            ),
            mock.patch(
                "sglang.srt.layers.linear.tensor_model_parallel_all_gather"
            ) as global_tp_gather,
        ):
            out = ranks[0](x)
        global_tp_gather.assert_not_called()
        attn_group.all_gather.assert_called_once()
        self.assertTrue(
            torch.equal(attn_group.all_gather.call_args.args[0], partials[0])
        )
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_guards_and_idempotence(self):
        # tp_size == 1: untouched
        m_tp1 = _make_stub_model(64, 32)
        orig = m_tp1.eh_proj
        with (
            mock.patch.object(
                mtp_fusion_proj, "is_dp_attention_enabled", return_value=False
            ),
            mock.patch.object(
                mtp_fusion_proj, "get_parallel", return_value=_parallel(tp_size=1)
            ),
        ):
            hook_shard_eh_proj(None, m_tp1)
        self.assertIs(m_tp1.eh_proj, orig)

        # out_features not divisible by tp_size: untouched
        m_indiv = _make_stub_model(60, 30)
        orig_indiv = m_indiv.eh_proj
        with (
            mock.patch.object(
                mtp_fusion_proj, "is_dp_attention_enabled", return_value=False
            ),
            mock.patch.object(
                mtp_fusion_proj, "get_parallel", return_value=_parallel(tp_size=8)
            ),
        ):
            hook_shard_eh_proj(None, m_indiv)
        self.assertIs(m_indiv.eh_proj, orig_indiv)

        # bias=True nn.Linear: untouched
        m_bias = _make_stub_model(64, 32, bias=True)
        orig_bias = m_bias.eh_proj
        with (
            mock.patch.object(
                mtp_fusion_proj, "is_dp_attention_enabled", return_value=False
            ),
            mock.patch.object(
                mtp_fusion_proj, "get_parallel", return_value=_parallel(tp_size=8)
            ),
        ):
            hook_shard_eh_proj(None, m_bias)
        self.assertIs(m_bias.eh_proj, orig_bias)

        # Non-nn.Linear (e.g. Quark ReplicatedLinear) & 2nd call: idempotent
        m_meta = _make_stub_model(64, 32, dtype=torch.bfloat16, device="meta")
        with (
            mock.patch.object(
                mtp_fusion_proj, "is_dp_attention_enabled", return_value=False
            ),
            mock.patch.object(
                mtp_fusion_proj, "get_parallel", return_value=_parallel(tp_size=8)
            ),
        ):
            hook_shard_eh_proj(None, m_meta, prefix="model")
            first = m_meta.eh_proj
            self.assertIsInstance(
                first, mtp_fusion_proj.ColumnParallelMtpFusionProj
            )
            self.assertEqual(first.weight.device.type, "meta")
            self.assertEqual(first.weight.dtype, torch.bfloat16)
            hook_shard_eh_proj(None, m_meta, prefix="model")
            self.assertIs(m_meta.eh_proj, first)


if __name__ == "__main__":
    unittest.main()
