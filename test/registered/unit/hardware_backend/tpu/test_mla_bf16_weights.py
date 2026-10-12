"""CPU test of the actual model post-loader's CUDA and TPU MLA branches.

Extracting the method avoids initializing serving backends just to test a
checkpoint transformation. The executed method body is unchanged repo code.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

root = Path(__file__).resolve().parents[5]
loader = root / "python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py"
utils = root / "python/sglang/srt/layers/quantization/fp8_utils.py"


def extract(path, name):
    node = next(
        x
        for x in ast.walk(ast.parse(path.read_text()))
        if isinstance(x, ast.FunctionDef) and x.name == name
    )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    return compile(ast.fix_missing_locations(tree), str(path), "exec")


def tensor_quant(x, dtype):
    scale = x.abs().max().clamp_min(1e-12) / torch.finfo(dtype).max
    return (x / scale).clamp(-448, 448).to(dtype), scale


class TestMlaBf16Weights(unittest.TestCase):
    def test_checkpoint_reconstruction(self):
        g = dict(
            torch=torch,
            _is_cuda=False,
            _is_tpu=True,
            _is_cpu=False,
            _is_musa=False,
            _is_xpu=False,
            _is_npu=False,
            _is_hip=False,
            _is_fp8_fnuz=False,
            _use_aiter_gfx95=False,
            deep_gemm_wrapper=NS(ENABLE_JIT_DEEPGEMM=False, DEEPGEMM_BLACKWELL=False),
            should_deepgemm_weight_requant_ue8m0=lambda **kw: False,
            bind_or_assign=lambda old, new: new,
            input_to_float8=tensor_quant,
        )
        for name in ["block_quant_dequant", "block_quant_to_tensor_quant"]:
            exec(extract(utils, name), g)
        for name in ["mla_absorbed_weight_layout", "post_load_weights"]:
            exec(extract(loader, name), g)
        torch.manual_seed(91011)
        w = (torch.randn(1024, 512) * 25).to(torch.float8_e4m3fn)
        scales = torch.logspace(-4, -1, 32).reshape(8, 4)
        results = {}
        for mode, cuda in [
            ("mla_bf16", False),
            ("cuda_reference", True),
        ]:
            g["_is_cuda"], g["_is_tpu"] = cuda, not cuda
            attention = NS(
                kv_b_proj=NS(weight=w.clone(), weight_scale_inv=scales.clone()),
                attn_mha=NS(kv_b_proj=None),
                qk_nope_head_dim=128,
                v_head_dim=128,
                w_kc=None,
                w_vc=None,
                w_scale=1.0,
            )
            obj = NS(
                config=NS(num_hidden_layers=1),
                quant_config=NS(weight_block_size=[128, 128]),
                model=NS(start_layer=0, end_layer=1, layers=[NS(self_attn=attention)]),
            )
            g["post_load_weights"](obj)
            results[mode] = attention
        ref = (
            w.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
        ).bfloat16()
        k, v = ref.reshape(4, 256, 512).split([128, 128], dim=1)
        treatment = results["mla_bf16"]
        assert (
            treatment.w_kc.dtype == torch.bfloat16
            and treatment.w_vc.dtype == torch.bfloat16
        )
        assert torch.equal(treatment.w_kc, k)
        assert torch.equal(treatment.w_vc, v.transpose(1, 2))
        # TPU: owned contiguous storage, no strided view graph inputs.
        for t in (treatment.w_kc, treatment.w_vc):
            assert t._base is None and t.is_contiguous()
        assert torch.equal(treatment.w_kc, results["cuda_reference"].w_kc)
        assert torch.equal(treatment.w_vc, results["cuda_reference"].w_vc)


if __name__ == "__main__":
    unittest.main()
