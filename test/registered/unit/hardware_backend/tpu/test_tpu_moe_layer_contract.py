"""Every TPU MoE quant method must build a FusedMoE whose derived contract keeps
the TPU path: no CUDA runner flags, no deferred finalize, in-place combine, a
standard dispatcher with global expert ids and standard top-k output.

FusedMoE derives these from quant_method.get_moe_runner_backend(); a TPU method
that skips its parent's create_moe_runner must still answer it.
"""

import contextlib
import importlib
import sys
import types
import unittest
from typing import Callable, NamedTuple
from unittest import mock

from sglang.srt.layers import quantization
from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    TpuCompressedTensorsWNA16MoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.modelopt_quant import (
    HybridFp8NvFp4Config,
    TpuModelOptCompatibleFp4Config,
    TpuModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config, Mxfp4MoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.srt.utils import device as device_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models import muse_spark
from sglang_tpu.platform import TpuOOTPlatform
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

AUTO = MoeRunnerBackend.AUTO
PALLAS = MoeRunnerBackend.PALLAS_TPU
FUSED_RS = MoeRunnerBackend.PALLAS_TPU_FUSED_RS

# Modules whose import-time `_is_tpu` gates FusedMoE construction.
_IS_TPU_MODULES = (
    "sglang.srt.layers.moe.fused_moe_triton.layer",
    "sglang.srt.layers.moe.token_dispatcher.standard",
    "sglang.srt.layers.moe.topk",
    "sglang.srt.layers.moe.utils",
    "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors",
    "sglang.srt.layers.quantization.fp8",
    "sglang.srt.layers.quantization.fp8_utils",
    "sglang.srt.layers.quantization.unquant",
)
# Imports torch_tpu; FusedMoE only registers its custom op here.
_FUSED_MOE_OP = "sglang_tpu.moe.pallas.fused_moe_op"

_NVFP4_CHECKPOINT = {
    "quant_method": "modelopt",
    "quantization": {
        "quant_algo": "NVFP4",
        "group_size": 512,
        "exclude_modules": ["self_attn", "lm_head"],
    },
}
_CT_W4A16_CHECKPOINT = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["re:.*mlp\\.experts.*"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 128,
            },
            "input_activations": None,
        }
    },
    "ignore": ["lm_head"],
}


@contextlib.contextmanager
def _stub_fused_moe_op():
    # Swap only this entry: patch.dict(sys.modules) would also drop every module
    # imported meanwhile, splitting later imports from the package attributes.
    stub = types.ModuleType(_FUSED_MOE_OP)
    stub._build_fused_moe_op = lambda: None
    previous = sys.modules.get(_FUSED_MOE_OP)
    sys.modules[_FUSED_MOE_OP] = stub
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(_FUSED_MOE_OP, None)
        else:
            sys.modules[_FUSED_MOE_OP] = previous


@contextlib.contextmanager
def _tpu_platform(platform):
    """Run as on a TPU host whose platform identity is `platform` (mode A or B)."""
    with contextlib.ExitStack() as stack:
        for name in _IS_TPU_MODULES:
            stack.enter_context(
                mock.patch.object(importlib.import_module(name), "_is_tpu", True)
            )
        stack.enter_context(_stub_fused_moe_op())
        stack.enter_context(
            mock.patch("sglang.srt.platforms.current_platform", platform)
        )
        stack.enter_context(
            mock.patch.object(device_utils, "current_platform", platform)
        )
        stack.enter_context(
            mock.patch.object(quantization, "current_platform", platform)
        )
        stack.enter_context(mock.patch.object(quantization, "is_cpu", lambda: False))
        stack.callback(device_utils.is_tpu.cache_clear)
        device_utils.is_tpu.cache_clear()
        yield


def _nvfp4_config():
    config_cls = quantization.get_quantization_config("modelopt_fp4")
    assert config_cls is TpuModelOptCompatibleFp4Config, config_cls
    return config_cls.from_config(_NVFP4_CHECKPOINT)


class _Case(NamedTuple):
    quant_kwargs: Callable[[], dict]
    # The class whose apply() runs the experts: the layer's scheme if it has one.
    method_cls: type
    backends: tuple
    fuses_routed_scaling_in_topk: bool = False


_ALL_TPU_BACKENDS = (AUTO, PALLAS, FUSED_RS)

_CASES = {
    "modelopt_fp4": _Case(
        quant_kwargs=lambda: {"quant_config": _nvfp4_config()},
        method_cls=TpuModelOptNvFp4FusedMoEMethod,
        backends=_ALL_TPU_BACKENDS,
        fuses_routed_scaling_in_topk=True,
    ),
    "hybrid_fp8_nvfp4": _Case(
        quant_kwargs=lambda: {
            "quant_config": HybridFp8NvFp4Config(
                fp8_config=Fp8Config(is_checkpoint_fp8_serialized=True),
                nvfp4_config=_nvfp4_config(),
            )
        },
        method_cls=TpuModelOptNvFp4FusedMoEMethod,
        backends=_ALL_TPU_BACKENDS,
        fuses_routed_scaling_in_topk=True,
    ),
    "unquantized": _Case(
        quant_kwargs=lambda: {},
        method_cls=UnquantizedFusedMoEMethod,
        backends=_ALL_TPU_BACKENDS,
    ),
    "fp8_block": _Case(
        quant_kwargs=lambda: {
            "quant_config": Fp8Config(
                is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
            )
        },
        method_cls=Fp8MoEMethod,
        backends=_ALL_TPU_BACKENDS,
    ),
    "fp8_per_tensor": _Case(
        quant_kwargs=lambda: {
            "quant_config": Fp8Config(is_checkpoint_fp8_serialized=True)
        },
        method_cls=Fp8MoEMethod,
        backends=_ALL_TPU_BACKENDS,
    ),
    "dsv4_mxfp4_experts": _Case(
        quant_kwargs=lambda: {
            "quant_config": Fp8Config(
                is_checkpoint_fp8_serialized=True,
                weight_block_size=[128, 128],
                is_fp4_experts=True,
            )
        },
        method_cls=Fp8MoEMethod,
        backends=_ALL_TPU_BACKENDS,
    ),
    "mxfp4": _Case(
        quant_kwargs=lambda: {
            "quant_config": Mxfp4Config(is_checkpoint_mxfp4_serialized=True)
        },
        method_cls=Mxfp4MoEMethod,
        # The MXFP4 method has no fused-RS path.
        backends=(AUTO, PALLAS),
    ),
    "compressed_tensors_w4a16": _Case(
        quant_kwargs=lambda: {
            "quant_config": CompressedTensorsConfig.from_config(_CT_W4A16_CHECKPOINT)
        },
        method_cls=TpuCompressedTensorsWNA16MoE,
        backends=_ALL_TPU_BACKENDS,
    ),
    "muse_bf16": _Case(
        quant_kwargs=lambda: {"quant_method": muse_spark.MuseTpuBf16MoEMethod()},
        method_cls=muse_spark.MuseTpuBf16MoEMethod,
        backends=_ALL_TPU_BACKENDS,
    ),
    "muse_host_fp8": _Case(
        quant_kwargs=lambda: {"quant_config": muse_spark._HostQuantizedFp8Config()},
        method_cls=muse_spark._HostQuantizedFp8MoEMethod,
        backends=_ALL_TPU_BACKENDS,
    ),
}


def _build_layer(backend, quant_kwargs):
    with (
        get_context().override_server_args(model_path="dummy", device="cpu"),
        get_flags().moe.override(
            runner_backend=backend, a2a_backend=MoeA2ABackend.NONE
        ),
        get_parallel().override(
            tp_size=1,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
        ),
    ):
        layer = FusedMoE(
            num_experts=4,
            hidden_size=512,
            intermediate_size=512,
            layer_id=0,
            top_k=2,
            prefix="model.layers.1.mlp.experts",
            **quant_kwargs(),
        )
        topk = TopK(top_k=2, layer_id=0)
        topk.bind_runner_backend(
            layer.runner_backend, routing_method_type=layer.routing_method_type
        )
    return layer, topk


class TestTpuMoeLayerContract(CustomTestCase):
    def test_every_tpu_moe_method_keeps_the_tpu_layer_contract(self):
        for platform_cls in (TpuSRTPlatform, TpuOOTPlatform):
            for name, case in _CASES.items():
                for backend in case.backends:
                    with (
                        self.subTest(
                            platform=platform_cls.__name__,
                            method=name,
                            backend=backend.value,
                        ),
                        _tpu_platform(platform_cls()),
                    ):
                        layer, topk = _build_layer(backend, case.quant_kwargs)
                        method = (
                            layer.scheme
                            if layer.scheme is not None
                            else layer.quant_method
                        )
                        self.assertIs(type(method), case.method_cls)
                        self._assert_tpu_contract(
                            layer, topk, case.fuses_routed_scaling_in_topk
                        )

    def _assert_tpu_contract(self, layer, topk, fuses_routed_scaling_in_topk):
        self.assertFalse(layer.enable_flashinfer_cutlass_moe)
        self.assertFalse(layer.use_flashinfer_mxfp4_moe)
        self.assertFalse(layer.use_deep_gemm)
        self.assertFalse(layer.use_flashinfer_trtllm_moe)
        self.assertFalse(layer.supports_deferred_finalize)
        self.assertFalse(layer.supports_routed_deferred_finalize)
        self.assertTrue(layer.moe_runner_config.inplace)
        self.assertIsNone(layer.routing_method_type)
        self.assertEqual(
            layer.should_fuse_routed_scaling_factor_in_topk,
            fuses_routed_scaling_in_topk,
        )

        dispatcher = layer.dispatcher
        self.assertIs(type(dispatcher), StandardDispatcher)
        self.assertTrue(dispatcher.skip_local_expert_mapping)
        self.assertFalse(dispatcher.enable_flashinfer_cutlass_moe)
        self.assertFalse(dispatcher.enable_flashinfer_mxfp4_moe)
        self.assertFalse(dispatcher.enable_flashinfer_trtllm_routed_moe)
        self.assertFalse(dispatcher.use_aiter_moe_runner)

        self.assertIs(topk.topk_config.output_format, TopKOutputFormat.STANDARD)


if __name__ == "__main__":
    unittest.main()
