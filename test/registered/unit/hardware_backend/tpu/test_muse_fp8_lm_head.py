# SPDX-License-Identifier: Apache-2.0
"""Unit and TPU7x Mosaic export tests for Muse Spark FP8 LM head & fused Top-K."""

from __future__ import annotations

import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kernels.fp8_lm_head_fused import quantize_lm_head_fp8_k_major


def _make_fake_lm_head(
    *,
    local_width: int = 25256,
    hidden_size: int = 128,
    org_vocab_size: int = 202048,
    global_vocab_size: int = 202048,
    org_vocab_padded_count: int = 25256,
    rank: int = 0,
    tp_size: int = 8,
    with_fp8: bool = True,
    with_bias: bool = False,
):
    head = nn.Module()
    head.enable_tp = True
    head.tp_size = tp_size
    head.num_embeddings = global_vocab_size
    head.num_embeddings_per_partition = local_width
    head.org_vocab_size = org_vocab_size
    added_w = local_width - org_vocab_padded_count
    org_start = min(rank * org_vocab_padded_count, org_vocab_size)
    org_end = min((rank + 1) * org_vocab_padded_count, org_vocab_size)
    added_start = min(org_vocab_size + rank * added_w, global_vocab_size)
    added_end = min(org_vocab_size + (rank + 1) * added_w, global_vocab_size)
    head.shard_indices = SimpleNamespace(
        org_vocab_start_index=org_start,
        num_org_elements=org_end - org_start,
        num_org_elements_padded=org_vocab_padded_count,
        added_vocab_start_index=added_start,
        num_added_elements=added_end - added_start,
    )
    head.weight = nn.Parameter(
        torch.randn(local_width, hidden_size, dtype=torch.bfloat16) * 0.025,
        requires_grad=False,
    )
    if with_bias:
        head.bias = nn.Parameter(torch.zeros(local_width, dtype=torch.float32))
    if with_fp8:
        w8k, scale = quantize_lm_head_fp8_k_major(head.weight, tn=1024)
        head.register_buffer("fp8_w8k", w8k, persistent=False)
        head.register_buffer("fp8_scale", scale, persistent=False)
    head.quant_method = SimpleNamespace(apply=mock.MagicMock())
    return head


class TestMuseFp8LmHead(unittest.TestCase):
    def test_default_off_flags(self):
        self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_FP8_LM_HEAD.default)
        self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_FUSED_FP8_LM_HEAD_TOPK.default)
        self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_FAST_DISTRIBUTED_TOPK.default)

    def test_quantize_lm_head_fp8_k_major_accuracy_and_padding(self):
        torch.manual_seed(0)
        v, h = 1300, 128
        w = (torch.randn(v, h, dtype=torch.bfloat16) * 0.03).contiguous()
        for tn in (512, 1024):
            w8k, scale = quantize_lm_head_fp8_k_major(w, tn=tn)
            vp = -(-v // tn) * tn
            self.assertEqual(tuple(w8k.shape), (h, vp))
            self.assertEqual(tuple(scale.shape), (1, vp))
            self.assertEqual(w8k.dtype, torch.float8_e4m3fn)
            self.assertEqual(scale.dtype, torch.float32)
            self.assertTrue(w8k.is_contiguous())
            self.assertTrue(scale.is_contiguous())
            self.assertIsNone(w8k._base)
            # Padded columns must be strictly zero in both weight and scale.
            self.assertEqual(float(w8k[:, v:].float().abs().max()), 0.0)
            self.assertEqual(float(scale[:, v:].abs().max()), 0.0)
            # Reconstructed logits must closely match BF16->FP32 reference.
            x = torch.randn(16, h, dtype=torch.bfloat16)
            ref = x.float() @ w.float().t()
            got = x.float() @ (w8k[:, :v].float() * scale[:, :v])
            cos = float(
                torch.nn.functional.cosine_similarity(
                    ref.reshape(-1), got.reshape(-1), dim=0
                )
            )
            self.assertGreater(cos, 0.999)

    def test_fused_fp8_distributed_topk_dispatch_and_fallbacks(self):
        from sglang_tpu.distributed_topk_pallas import (
            fused_fp8_lm_head_distributed_top_k,
        )

        group = SimpleNamespace(world_size=8, rank_in_group=0, ranks=tuple(range(8)))
        head = _make_fake_lm_head(local_width=256, hidden_size=64, org_vocab_size=2048, global_vocab_size=2048, org_vocab_padded_count=256)

        # Fake TPU tensor wrapper via mock device.type == "tpu".
        fake_x = mock.MagicMock(spec=torch.Tensor)
        fake_x.device = SimpleNamespace(type="tpu")
        fake_x.ndim = 2
        fake_x.shape = (64, 64)

        mock_op = mock.MagicMock(
            return_value=(
                torch.zeros(64, 64, dtype=torch.float32),
                torch.zeros(64, 64, dtype=torch.int32),
            )
        )
        with (
            mock.patch("sglang_tpu.top_k._group_spans_world", return_value=True),
            mock.patch(
                "sglang_tpu.kernels.fp8_lm_head_fused.get_fp8_lm_head_distributed_topk_op",
                return_value=mock_op,
            ),
            mock.patch("torch.compiler.is_compiling", return_value=True),
        ):
            res = fused_fp8_lm_head_distributed_top_k(
                fake_x, head, top_k=64, logit_scale=0.1767767, group=group
            )
            self.assertIsNotNone(res)
            mock_op.assert_called_once()
            args = mock_op.call_args.args
            self.assertEqual(args[3:9], (256, 2048, 2048, 256, 64, 0.1767767))

            # Fallback: rows > 256
            fake_big = mock.MagicMock(spec=torch.Tensor)
            fake_big.device = SimpleNamespace(type="tpu")
            fake_big.ndim = 2
            fake_big.shape = (384, 64)
            self.assertIsNone(
                fused_fp8_lm_head_distributed_top_k(
                    fake_big, head, top_k=64, logit_scale=1.0, group=group
                )
            )
            # Fallback: negative/zero logit_scale
            self.assertIsNone(
                fused_fp8_lm_head_distributed_top_k(
                    fake_x, head, top_k=64, logit_scale=-1.0, group=group
                )
            )
            # Fallback: CPU tensor
            cpu_x = torch.randn(16, 64, dtype=torch.bfloat16)
            self.assertIsNone(
                fused_fp8_lm_head_distributed_top_k(
                    cpu_x, head, top_k=64, logit_scale=1.0, group=group
                )
            )
            # Fallback: head with bias
            head_bias = _make_fake_lm_head(
                local_width=256, hidden_size=64, org_vocab_size=2048, global_vocab_size=2048, org_vocab_padded_count=256, with_bias=True
            )
            self.assertIsNone(
                fused_fp8_lm_head_distributed_top_k(
                    fake_x, head_bias, top_k=64, logit_scale=1.0, group=group
                )
            )
            # Fallback: non-canonical rank
            bad_group = SimpleNamespace(world_size=8, rank_in_group=3, ranks=tuple(range(8)))
            self.assertIsNone(
                fused_fp8_lm_head_distributed_top_k(
                    fake_x, head, top_k=64, logit_scale=1.0, group=bad_group
                )
            )

    def test_dflash_draft_argmax_and_lossless_topk_dispatches(self):
        from sglang_tpu.dflash_utils_tpu import argmax_draft_tokens_tpu
        from sglang_tpu.distributed_topk_pallas import (
            pallas_distributed_top_k,
            register_pallas_distributed_top_k_op,
        )

        head = _make_fake_lm_head(
            local_width=256, hidden_size=64, org_vocab_size=2048, global_vocab_size=2048, org_vocab_padded_count=256
        )
        hs = torch.randn(16 * 4, 64, dtype=torch.bfloat16)
        fake_group = SimpleNamespace(world_size=8, rank_in_group=0, ranks=tuple(range(8)))
        fake_parallel = SimpleNamespace(tp_group=fake_group)
        fake_ids = torch.arange(16 * 3, dtype=torch.int32).unsqueeze(-1)
        fake_vals = torch.ones(16 * 3, 1, dtype=torch.float32)

        with (
            mock.patch("sglang_tpu.dflash_utils_tpu._tp_size", return_value=8),
            mock.patch("sglang_tpu.dflash_utils_tpu.get_parallel", return_value=fake_parallel),
            mock.patch.object(
                tpu_envs.SGLANG_TPU_MUSE_FUSED_FP8_LM_HEAD_TOPK, "get", return_value=True
            ),
            mock.patch(
                "sglang_tpu.distributed_topk_pallas.fused_fp8_lm_head_distributed_top_k",
                return_value=(fake_vals, fake_ids),
            ) as mock_fused,
        ):
            out = argmax_draft_tokens_tpu(lm_head=head, hidden_states=hs, block_size=4)
            self.assertEqual(tuple(out.shape), (16, 3))
            self.assertEqual(out.dtype, torch.int64)
            mock_fused.assert_called_once()
            # Must NOT call quant_method.apply when fused path succeeded!
            head.quant_method.apply.assert_not_called()

        # Lossless SGLANG_TPU_MUSE_FAST_DISTRIBUTED_TOPK dispatch + registration.
        mock_lo_op = mock.MagicMock(
            return_value=(torch.zeros(128, 64), torch.zeros(128, 64, dtype=torch.int32))
        )
        mock_orig_op = mock.MagicMock(
            return_value=(torch.zeros(32, 64), torch.zeros(32, 64, dtype=torch.int32))
        )
        with (
            mock.patch.object(
                tpu_envs.SGLANG_TPU_MUSE_FAST_DISTRIBUTED_TOPK, "get", return_value=True
            ),
            mock.patch.object(
                tpu_envs.SGLANG_TPU_MUSE_FUSED_FP8_LM_HEAD_TOPK, "get", return_value=True
            ),
            mock.patch(
                "sglang_tpu.distributed_topk_pallas.get_pallas_distributed_top_k_op",
                return_value=mock_orig_op,
            ) as mock_get_orig,
            mock.patch(
                "sglang_tpu.kernels.fp8_lm_head_fused.get_lossless_distributed_topk_vmem_ag_op",
                return_value=mock_lo_op,
            ) as mock_get_lo,
            mock.patch(
                "sglang_tpu.kernels.fp8_lm_head_fused.get_fp8_lm_head_distributed_topk_op",
            ) as mock_get_fused,
        ):
            register_pallas_distributed_top_k_op(8)
            mock_get_orig.assert_called_once_with(8)
            mock_get_lo.assert_called_once_with(8)
            mock_get_fused.assert_called_once_with(8)

            # TP=8 B=128 routes to lossless VMEM AG op (-49..-95 us win).
            vals, ids = pallas_distributed_top_k(
                torch.zeros(128, 256),
                org_vocab_size=2048,
                global_vocab_size=2048,
                org_vocab_padded_count=256,
                k=64,
                world_size=8,
            )
            self.assertEqual(tuple(vals.shape), (128, 64))
            self.assertEqual(tuple(ids.shape), (128, 64))
            mock_lo_op.assert_called_once()
            mock_orig_op.assert_not_called()

            # TP=8 B=32 (< 128) cleanly falls back to orig Pallas TopK (zero regression!).
            vals_sm, ids_sm = pallas_distributed_top_k(
                torch.zeros(32, 256),
                org_vocab_size=2048,
                global_vocab_size=2048,
                org_vocab_padded_count=256,
                k=64,
                world_size=8,
            )
            self.assertEqual(tuple(vals_sm.shape), (32, 64))
            self.assertEqual(tuple(ids_sm.shape), (32, 64))
            mock_orig_op.assert_called_once()

    def test_logits_processor_distributed_topk_fused_and_softcap(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        lp = object.__new__(LogitsProcessor)
        lp.use_attn_tp_group = False
        lp.logit_scale = 0.1767767
        lp.final_logit_softcapping = 30.0
        head = _make_fake_lm_head(
            local_width=256, hidden_size=64, org_vocab_size=2048, global_vocab_size=2048, org_vocab_padded_count=256
        )
        fake_group = SimpleNamespace(world_size=8, rank_in_group=0, ranks=tuple(range(8)))
        raw_vals = torch.full((16, 64), 15.0, dtype=torch.float32)
        raw_ids = torch.arange(64, dtype=torch.int32).unsqueeze(0).expand(16, 64)

        with (
            mock.patch(
                "sglang.srt.layers.logits_processor.get_parallel",
                return_value=SimpleNamespace(tp_group=fake_group, attn_tp_group=fake_group),
            ),
            mock.patch.object(
                tpu_envs.SGLANG_TPU_MUSE_FUSED_FP8_LM_HEAD_TOPK, "get", return_value=True
            ),
            mock.patch(
                "sglang_tpu.distributed_topk_pallas.fused_fp8_lm_head_distributed_top_k",
                return_value=(raw_vals, raw_ids),
            ) as mock_fused,
        ):
            vals, ids = lp._get_distributed_top_k(torch.randn(16, 64), head, top_k=64)
            mock_fused.assert_called_once()
            expected = 30.0 * torch.tanh(raw_vals / 30.0)
            self.assertTrue(torch.allclose(vals, expected))
            self.assertTrue(torch.equal(ids, raw_ids))

    def test_tpu7x_mosaic_export_all_lm_head_kernels(self):
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh, PartitionSpec as P

        from sglang_tpu.kernels.fp8_lm_head_fused import (
            distributed_topk_vmem_ag_jax,
            fp8_lm_head_argmax_jax,
            fp8_lm_head_fused_topk64_jax,
            fp8_lm_head_zsp_jax,
        )

        all_devs = list(jax.devices())
        for tp in (8, 16):
            if len(all_devs) < tp:
                continue
            mesh = Mesh(np.array(all_devs[:tp]), ("model",))
            v_loc = 202048 // tp
            vp = -(-v_loc // 1024) * 1024
            for b in (16, 32, 64, 128, 256):
                with self.subTest(tp=tp, b=b):
                    xs = jax.ShapeDtypeStruct((b, 8192), jnp.bfloat16)
                    ws = jax.ShapeDtypeStruct((8192, vp * tp), jnp.float8_e4m3fn)
                    ss = jax.ShapeDtypeStruct((1, vp * tp), jnp.float32)
                    ls = jax.ShapeDtypeStruct((b, v_loc * tp), jnp.float32)
                    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
                        for fn, args, out_shapes in [
                            (
                                jax.jit(
                                    jax.shard_map(
                                        lambda x, w, s: fp8_lm_head_zsp_jax(
                                            x, w, s, logit_scale=0.1767767, tn=512
                                        ),
                                        mesh=mesh,
                                        in_specs=(P(), P(None, "model"), P(None, "model")),
                                        out_specs=P(None, "model"),
                                        check_vma=False,
                                    )
                                ),
                                (xs, ws, ss),
                                ((b, vp * tp),),
                            ),
                            (
                                jax.jit(
                                    jax.shard_map(
                                        lambda x, w, s: fp8_lm_head_argmax_jax(
                                            x,
                                            w,
                                            s,
                                            world_size=tp,
                                            local_width=v_loc,
                                            org_vocab_size=202048,
                                            global_vocab_size=202048,
                                            org_vocab_padded_count=v_loc,
                                            logit_scale=0.1767767,
                                            tn=1024,
                                        ),
                                        mesh=mesh,
                                        in_specs=(P(), P(None, "model"), P(None, "model")),
                                        out_specs=(P(), P()),
                                        check_vma=False,
                                    )
                                ),
                                (xs, ws, ss),
                                ((b, 1), (b, 1)),
                            ),
                            (
                                jax.jit(
                                    jax.shard_map(
                                        lambda x, w, s: fp8_lm_head_fused_topk64_jax(
                                            x,
                                            w,
                                            s,
                                            world_size=tp,
                                            local_width=v_loc,
                                            org_vocab_size=202048,
                                            global_vocab_size=202048,
                                            org_vocab_padded_count=v_loc,
                                            k=64,
                                            logit_scale=0.1767767,
                                            tn=1024,
                                        ),
                                        mesh=mesh,
                                        in_specs=(P(), P(None, "model"), P(None, "model")),
                                        out_specs=(P(), P()),
                                        check_vma=False,
                                    )
                                ),
                                (xs, ws, ss),
                                ((b, 64), (b, 64)),
                            ),
                            (
                                jax.jit(
                                    jax.shard_map(
                                        lambda l: distributed_topk_vmem_ag_jax(
                                            l,
                                            world_size=tp,
                                            local_width=v_loc,
                                            org_vocab_size=202048,
                                            global_vocab_size=202048,
                                            org_vocab_padded_count=v_loc,
                                            k=64,
                                        ),
                                        mesh=mesh,
                                        in_specs=(P(None, "model"),),
                                        out_specs=(P(), P()),
                                        check_vma=False,
                                    )
                                ),
                                (ls,),
                                ((b, 64), (b, 64)),
                            ),
                        ]:
                            exp = jax.export.export(fn, platforms=("tpu",))(*args)
                            self.assertEqual(
                                tuple(a.shape for a in exp.out_avals), out_shapes
                            )


if __name__ == "__main__":
    unittest.main()
