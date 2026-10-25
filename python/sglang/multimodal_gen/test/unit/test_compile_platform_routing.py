import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.nn as nn

from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform
from sglang.multimodal_gen.runtime.utils import torch_compile


def test_builtin_tpu_uses_declared_compiler_without_inductor_mode(monkeypatch):
    monkeypatch.setattr(torch_compile, "current_platform", TpuPlatform())
    assert torch_compile.build_torch_compile_kwargs(mode="max-autotune") == {
        "fullgraph": False,
        "dynamic": False,
        "backend": "tpu",
    }


@pytest.mark.parametrize("out_of_tree", [False, True])
@pytest.mark.parametrize(
    "backend,options",
    [
        ("inductor", None),
        ("inductor", {"max_autotune": True}),
        ("custom", None),
        ("custom", {"custom_option": True}),
    ],
)
def test_platform_backend_options_and_overlap(
    monkeypatch, out_of_tree, backend, options
):
    platform = SimpleNamespace(
        is_npu=lambda: False,
        is_out_of_tree=lambda: out_of_tree,
        get_compile_dynamic=lambda: None,
        get_compile_backend=Mock(return_value=backend),
        get_compile_options=Mock(return_value=options),
    )
    overlap = Mock()
    monkeypatch.setattr(torch_compile, "current_platform", platform)
    monkeypatch.setattr(
        torch_compile, "maybe_enable_inductor_compute_comm_overlap", overlap
    )
    module = nn.Linear(2, 2)
    actual = torch_compile.build_torch_compile_kwargs(
        mode="max-autotune",
        module=module,
        enable_inductor_compute_comm_overlap=True,
    )
    expected = {"fullgraph": False, "dynamic": None, "backend": backend}
    if options is not None:
        expected["options"] = options
    elif backend == "inductor":
        expected["mode"] = "max-autotune"
    assert actual == expected
    assert overlap.call_count == int(backend == "inductor")
    platform.get_compile_backend.assert_called_once_with("max-autotune")
    platform.get_compile_options.assert_called_once_with(module)


def test_native_npu_preserves_special_backend_and_static_shapes(monkeypatch):
    platform = SimpleNamespace(
        is_npu=lambda: True,
        is_out_of_tree=lambda: False,
        get_compile_dynamic=lambda: None,
    )
    monkeypatch.setattr(torch_compile, "current_platform", platform)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.utils.common",
        SimpleNamespace(
            get_compiler_backend=lambda: "npu_backend",
        ),
    )
    assert torch_compile.build_torch_compile_kwargs(mode="max-autotune") == {
        "fullgraph": False,
        "dynamic": False,
        "backend": "npu_backend",
    }


def test_default_platform_keeps_inductor_and_automatic_shapes(monkeypatch):
    from sglang.multimodal_gen.runtime.platforms.interface import Platform

    monkeypatch.setattr(torch_compile, "current_platform", Platform())
    assert torch_compile.build_torch_compile_kwargs(mode="default") == {
        "fullgraph": False,
        "dynamic": None,
        "backend": "inductor",
        "mode": "default",
    }
