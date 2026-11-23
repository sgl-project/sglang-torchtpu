# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the fused EP MoE selection envs (sglang_tpu.moe.fused_ep).

PYTHONPATH=<tree>/python:<tree> python -m pytest -q <this file>
"""

import ast
import logging
import os

import pytest

from sglang.srt.environ import EnvBool, EnvInt
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.envs import TpuEnvs
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.moe import fused_ep

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PUSH_PKG = os.path.join(
    os.path.dirname(fused_ep.__file__), "pallas", "fused_moe_v2", "push_fused_moe"
)


def _kernel_push_options() -> tuple[str, ...]:
    """push_fused_moe.PUSH_OPTIONS, read from source (the package needs JAX)."""
    with open(os.path.join(_PUSH_PKG, "__init__.py")) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "PUSH_OPTIONS":
            return ast.literal_eval(node.value)
    raise AssertionError("PUSH_OPTIONS not found")


def _kernel_push_defaults() -> dict:
    """fused_ep_moe_push's defaults for the push options, read from source."""
    with open(os.path.join(_PUSH_PKG, "layer.py")) as f:
        tree = ast.parse(f.read())
    (fn,) = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "fused_ep_moe_push"
    ]
    return {
        arg.arg: ast.literal_eval(default)
        for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults)
        if arg.arg in fused_ep.PUSH_OPTIONS
    }


@pytest.fixture(autouse=True)
def _unset(monkeypatch):
    monkeypatch.delenv(fused_ep.IMPL_ENV, raising=False)
    for option in fused_ep.PUSH_OPTIONS:
        monkeypatch.delenv(fused_ep.push_option_env(option), raising=False)


def test_push_options_mirror_the_kernel():
    assert fused_ep.PUSH_OPTIONS == _kernel_push_options()


def test_every_push_option_has_a_typed_env_that_keeps_the_kernel_default():
    defaults = _kernel_push_defaults()
    expected = {fused_ep.push_option_env(o) for o in fused_ep.PUSH_OPTIONS}
    declared = {
        n for n in TpuEnvs.names() if n.startswith("SGLANG_TPU_MOE_FUSED_EP_PUSH_")
    }
    assert declared == expected
    for option in fused_ep.PUSH_OPTIONS:
        field = vars(TpuEnvs)[fused_ep.push_option_env(option)]
        # Unset forwards nothing, so the kernel default applies.
        assert field.default is None, option
        kind = EnvBool if isinstance(defaults[option], bool) else EnvInt
        assert type(field) is kind, option


def test_unset_or_empty_selects_nothing():
    assert fused_ep.resolve_fused_ep_impl() is None
    assert not fused_ep.is_fused_ep_requested()
    with tpu_envs.SGLANG_TPU_MOE_FUSED_EP_KERNEL_IMPL.override(""):
        assert fused_ep.resolve_fused_ep_impl() is None
        assert not fused_ep.is_fused_ep_requested()


def test_v2_uses_all_gather_and_forwards_no_push_options():
    with (
        tpu_envs.SGLANG_TPU_MOE_FUSED_EP_KERNEL_IMPL.override("v2"),
        tpu_envs.SGLANG_TPU_MOE_FUSED_EP_PUSH_SC_PREPUSH.override(1),
    ):
        assert fused_ep.resolve_fused_ep_impl() == fused_ep.FusedEpImpl(
            name="v2", dispatch="all_gather"
        )


def test_push_forwards_only_the_set_options_sorted():
    with tpu_envs.SGLANG_TPU_MOE_FUSED_EP_KERNEL_IMPL.override("push"):
        assert fused_ep.resolve_fused_ep_impl() == fused_ep.FusedEpImpl(
            name="push", dispatch="push"
        )
        with (
            tpu_envs.SGLANG_TPU_MOE_FUSED_EP_PUSH_TC_EXCHANGE.override(True),
            tpu_envs.SGLANG_TPU_MOE_FUSED_EP_PUSH_SC_PREPUSH.override(1),
        ):
            impl = fused_ep.resolve_fused_ep_impl()
    # Sorted, so the same knobs always build (and name) the same cached op.
    assert impl.push_options == (("sc_prepush", 1), ("tc_exchange", True))


def test_unknown_impl_is_refused():
    with tpu_envs.SGLANG_TPU_MOE_FUSED_EP_KERNEL_IMPL.override("adaptive"):
        assert fused_ep.is_fused_ep_requested()
        with pytest.raises(ValueError, match="expected one of"):
            fused_ep.resolve_fused_ep_impl()


def test_push_knobs_under_v2_warn(caplog):
    from sglang.srt.environ import envs

    with (
        tpu_envs.SGLANG_TPU_MOE_FUSED_EP_KERNEL_IMPL.override("v2"),
        tpu_envs.SGLANG_TPU_MOE_FUSED_EP_PUSH_SC_PREPUSH.override(1),
        envs.SGLANG_SHARED_EXPERT_TP1.override(True),
        caplog.at_level(logging.WARNING, logger=fused_ep.logger.name),
    ):
        fused_ep.check_fused_ep_env()
    assert "SGLANG_TPU_MOE_FUSED_EP_PUSH_SC_PREPUSH" in caplog.text
