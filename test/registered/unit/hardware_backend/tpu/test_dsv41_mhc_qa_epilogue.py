"""G-B-lite's try_boundary_qa: hands results only to the exact normed input it
produced, declines unless its flag, the owner and the shape all fit, and its
exported op has a schema torch can infer."""

import inspect
import typing
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.models.deepseek_v41 import qa_chain_op
from sglang_tpu.models.deepseek_v41.ops import mhc as shifted

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _seam(rows=6, owner_name="attn_hc", packed=True):
    """A fake MoE -> attention seam wired the way #2411 binds it."""
    layer = torch.nn.Module()
    attn = torch.nn.Module()
    attn.fuse_wqa_wkv, attn.q_lora_rank, attn.rope_head_dim, attn.eps = (
        True,
        1280,
        64,
        1e-20,
    )
    attn.wqkv_a = torch.nn.Module()
    attn.wqkv_a.weight = torch.empty(5120, 1792, dtype=torch.float8_e4m3fn)
    attn.wqkv_a.weight_scale = torch.empty(1792)
    attn.q_norm = SimpleNamespace(weight=torch.ones(1280))
    attn.kv_norm = SimpleNamespace(weight=torch.ones(512))
    layer.self_attn = attn
    norm = torch.nn.Module()
    norm.weight = torch.nn.Parameter(torch.ones(5120))
    norm.variance_epsilon = 1e-6
    if packed:
        norm.register_buffer(
            shifted._PACKED, torch.empty(20480, 128, dtype=torch.bfloat16)
        )
    cfg = SimpleNamespace(
        mult=4, hidden=5120, rms_eps=1e-20, eps=1e-6, sinkhorn_iters=20
    )
    layer.attn_hc = SimpleNamespace(
        cfg=cfg, norm=norm, scale=torch.ones(3), base=torch.zeros(24)
    )
    setattr(norm, shifted._OWNER, (weakref.ref(layer), owner_name))
    boundary = SimpleNamespace(norm=norm, norm_fusable=True)
    hc = SimpleNamespace(cfg=cfg)
    y = torch.zeros(rows, 5120, dtype=torch.bfloat16)
    residual = torch.zeros(rows, 4, 5120, dtype=torch.bfloat16)
    coeff = tuple(torch.zeros(rows, *s) for s in ((4,), (4,), (4, 4)))
    return layer, hc, y, residual, coeff, boundary


class TryBoundaryQaTest(unittest.TestCase):
    def test_off_by_default(self):
        _, hc, y, residual, coeff, boundary = _seam()
        with tpu_envs.SGLANG_TPU_DSV41_MHC_QA_EPILOGUE.override(False):
            self.assertIsNone(
                qa_chain_op.try_boundary_qa(hc, y, residual, coeff, boundary)
            )

    def test_declines_without_attention_owner_or_packed_weight(self):
        with tpu_envs.SGLANG_TPU_DSV41_MHC_QA_EPILOGUE.override(True):
            for kwargs in ({"owner_name": "ffn_hc"}, {"packed": False}):
                layer, hc, y, residual, coeff, boundary = _seam(**kwargs)
                self.assertIsNone(
                    qa_chain_op.try_boundary_qa(hc, y, residual, coeff, boundary),
                    kwargs,
                )

    def test_row_gate(self):
        with tpu_envs.SGLANG_TPU_DSV41_MHC_QA_EPILOGUE.override(True):
            for rows, takes in (
                (6, True),
                (48, True),
                (128, True),
                (64, False),
                (192, False),
            ):
                layer, hc, y, residual, coeff, boundary = _seam(rows)
                fake = mock.MagicMock(
                    return_value=(
                        residual.flatten(1),
                        coeff[1],
                        coeff[2],
                        y,
                        coeff[0],
                        None,
                        None,
                    )
                )
                with (
                    mock.patch.object(qa_chain_op, "_op", return_value=fake),
                    mock.patch.object(
                        torch.Tensor,
                        "device",
                        new_callable=mock.PropertyMock,
                        return_value=torch.device("tpu", 0)
                        if hasattr(torch, "tpu")
                        else SimpleNamespace(type="tpu"),
                    ),
                ):
                    got = qa_chain_op.try_boundary_qa(hc, y, residual, coeff, boundary)
                self.assertEqual(got is not None, takes, rows)
                self.assertEqual(fake.called, takes, rows)


class StashTest(unittest.TestCase):
    def test_taken_once_by_the_same_tensor(self):
        x = torch.zeros(16, 8)
        q, kv = torch.ones(16, 4), torch.ones(16, 2)
        qa_chain_op.stash(x, q, kv)
        self.assertIsNone(qa_chain_op.take_precomputed(torch.zeros(16, 8)))
        got = qa_chain_op.take_precomputed(x)
        self.assertIs(got[0], q)
        self.assertIs(got[1], kv)
        self.assertIsNone(qa_chain_op.take_precomputed(x))


class SchemaTest(unittest.TestCase):
    """torch.library must infer the op's schema (jax.Array read as Tensor); a
    variadic tuple return fails only at server start."""

    def test_boundary_qa_schema(self):
        import jax

        def conv(a):
            if a is jax.Array:
                return torch.Tensor
            if typing.get_origin(a) in (tuple, typing.Tuple):
                return typing.Tuple[tuple(conv(x) for x in typing.get_args(a))]
            return a

        sig = inspect.signature(qa_chain_op._boundary_qa)
        params = [
            inspect.Parameter(p.name, p.kind, annotation=conv(p.annotation))
            for p in sig.parameters.values()
        ]

        def stub(*args):
            pass

        stub.__signature__ = inspect.Signature(
            params, return_annotation=conv(sig.return_annotation)
        )
        self.assertIn(
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
            torch.library.infer_schema(stub, mutates_args=()),
        )


if __name__ == "__main__":
    unittest.main()
