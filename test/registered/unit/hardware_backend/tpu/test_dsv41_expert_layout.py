# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the static fused EP expert layout (expert_layout.py): file
validation, the slot table, the checkpoint read plan and the weight-loader
placement.

    PYTHONPATH=<tree>/python:<tree> python -m pytest -q <this file>
"""

import json
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LAYOUT = {3: [[5, 0, 7, 2], [1, 6, 3, 4]]}


def test_read_validate_and_slot_table(tmp_path, monkeypatch):
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.models.deepseek_v41 import expert_layout as el

    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps({"experts": 8, "shards": 2, "slots": 4, "layers": {"3": LAYOUT[3]}})
    )
    with (
        tpu_envs.SGLANG_TPU_DSV41_FUSED_EP.override(True),
        tpu_envs.SGLANG_TPU_DSV41_FUSED_EP_LAYOUT.override(str(path)),
    ):
        layout = el.active_layout()
    assert layout == LAYOUT
    el.validate(layout, 8, 2)
    assert el.expert_slot_table(layout, 3, 8)[:, 0].tolist() == [1, 4, 3, 6, 7, 0, 5, 2]
    with pytest.raises(ValueError, match="not a permutation"):
        el.validate({3: [[0, 0, 7, 2], [1, 6, 3, 4]]}, 8, 2)
    with pytest.raises(ValueError, match="2 shards of 4"):
        el.validate({3: [[0, 1, 2], [3, 4, 5, 6, 7]]}, 8, 2)


@pytest.mark.parametrize(
    "name",
    ["dsv41_flash_dp8_agentx_c32_visit.json", "dsv41_flash_dp8_agentx_c128_visit.json"],
)
def test_packaged_layouts_are_valid(name):
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.models.deepseek_v41 import expert_layout as el

    with (
        tpu_envs.SGLANG_TPU_DSV41_FUSED_EP.override(True),
        tpu_envs.SGLANG_TPU_DSV41_FUSED_EP_LAYOUT.override(name),
    ):
        assert el.layout_path() == str(el.PACKAGED_DIR / name)
        layout = el.active_layout()
    assert sorted(layout) == list(range(40))
    el.validate(layout, 384, 8)


def test_layout_ignored_without_fused_ep(tmp_path):
    """Without fused EP nothing places experts by slot, so the loader must not
    fetch the layout's permuted experts either (it would load wrong weights)."""
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.models.deepseek_v41 import expert_layout as el
    from sglang_tpu.models.deepseek_v41.hooks import fused_ep_hook_specs

    path = tmp_path / "layout.json"
    path.write_text(
        json.dumps({"experts": 8, "shards": 2, "slots": 4, "layers": {"3": LAYOUT[3]}})
    )
    with (
        tpu_envs.SGLANG_TPU_DSV41_FUSED_EP.override(False),
        tpu_envs.SGLANG_TPU_DSV41_FUSED_EP_LAYOUT.override(str(path)),
    ):
        assert el.active_layout() is None
        assert fused_ep_hook_specs() == []


def test_keep_expert_reads_the_layout_experts():
    from sglang_tpu.models.deepseek_v41 import expert_layout as el

    keep = el.keep_expert
    # Listed layer: this rank's layout experts, whatever the contiguous range.
    assert keep("layers.3.ffn.experts.5.w1.weight", LAYOUT, 0, (0, 4)) is True
    assert keep("layers.3.ffn.experts.1.w1.weight", LAYOUT, 0, (0, 4)) is False
    assert keep("model.layers.3.mlp.experts.6.down_proj.weight", LAYOUT, 1, (4, 8))
    # Unlisted layer: the contiguous range.
    assert keep("layers.4.ffn.experts.1.w1.weight", LAYOUT, 0, (0, 4)) is True
    assert keep("layers.4.ffn.experts.5.w1.weight", LAYOUT, 0, (0, 4)) is False
    # MTP tensors and non-expert tensors.
    assert keep("mtp.layers.3.ffn.experts.5.w1.weight", LAYOUT, 1, (4, 8)) is True
    assert keep("layers.3.ffn.gate.weight", LAYOUT, 0, (0, 4)) is None


def test_loader_plan_uses_the_layout(monkeypatch):
    from sglang_tpu.models.deepseek_v41 import expert_layout as el
    from sglang_tpu.models.deepseek_v41.loader import DSV41RunaiModelStreamerLoader

    monkeypatch.setattr(el, "active_layout", lambda: LAYOUT)
    loader = object.__new__(DSV41RunaiModelStreamerLoader)
    loader._expert_range = (4, 8)
    names = [f"layers.3.ffn.experts.{e}.w1.weight" for e in range(8)] + [
        "layers.3.attn.wq.weight"
    ]
    loader._layout_rank = None
    stock = loader._plan_expert_fetch({n: "s" for n in names})
    assert sorted(stock.local_experts) == sorted(names[4:8])
    loader._layout_rank = 1
    plan = loader._plan_expert_fetch({n: "s" for n in names})
    assert sorted(plan.local_experts) == sorted(names[e] for e in (1, 6, 3, 4))
    assert "layers.3.attn.wq.weight" in plan.kept
    assert sorted(plan.skipped) == sorted(names[e] for e in (0, 2, 5, 7))


def test_place_expert_ids_maps_to_layout_slots(monkeypatch):
    from sglang_tpu.models.deepseek_v41 import expert_layout as el

    monkeypatch.setattr(el, "active_layout", lambda: LAYOUT)
    calls = []

    def original(moe, param, expert_id):
        calls.append(expert_id)
        yield expert_id - 4

    moe = SimpleNamespace(layer_id=3)
    param = SimpleNamespace()
    # Untagged layer: the stock mapping.
    assert list(el.place_expert_ids(original, moe, param, 6)) == [2]
    setattr(moe, el.LAYOUT_RANK_ATTR, 1)
    setattr(moe, el.FILLED_ATTR, set())
    assert list(el.place_expert_ids(original, moe, param, 6)) == [1]
    assert list(el.place_expert_ids(original, moe, param, 4)) == [3]
    assert list(el.place_expert_ids(original, moe, param, 5)) == []
    assert getattr(moe, el.FILLED_ATTR) == {1, 3}
    # A layer outside the layout and parameters that need every expert.
    moe.layer_id = 9
    assert list(el.place_expert_ids(original, moe, param, 6)) == [2]
    moe.layer_id = 3
    param._sglang_require_global_experts = True
    assert list(el.place_expert_ids(original, moe, param, 6)) == [2]
    assert calls == [6, 6, 6]


def test_solve_lpt_balances():
    from sglang_tpu.models.deepseek_v41 import expert_layout as el

    counts = [100, 90, 10, 10, 50, 50, 5, 5]
    shards = el.solve_lpt(counts, 2)
    el.validate({0: shards}, 8, 2)
    loads = [sum(counts[e] for e in s) for s in shards]
    assert max(loads) - min(loads) <= 10


def test_hook_registered_only_with_layout(monkeypatch):
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.models.deepseek_v41.hooks import fused_ep_hook_specs

    target = (
        "sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE."
        "_weight_loader_expert_ids"
    )
    with tpu_envs.SGLANG_TPU_DSV41_FUSED_EP.override(True):
        assert target not in [s[0] for s in fused_ep_hook_specs()]
        with tpu_envs.SGLANG_TPU_DSV41_FUSED_EP_LAYOUT.override("/x.json"):
            assert target in [s[0] for s in fused_ep_hook_specs()]
