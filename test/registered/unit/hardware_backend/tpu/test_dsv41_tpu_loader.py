# SPDX-License-Identifier: Apache-2.0
"""EP-local routed-expert read plan for the DSV4.1 RunAI loader."""

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.models.deepseek_v41.loader import plan_expert_reads

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _weight_map(layers=2, experts=8):
    names = {"embed.weight": "s0", "layers.0.attn.wq_a.weight": "s0"}
    for layer in range(layers):
        for expert in range(experts):
            for w in ("w1", "w2", "w3"):
                for kind in ("weight", "scale"):
                    names[f"layers.{layer}.ffn.experts.{expert}.{w}.{kind}"] = (
                        f"e{expert}"
                    )
    names["mtp.0.ffn.experts.0.w1.weight"] = "mtp"
    return names


@pytest.mark.parametrize("ep_rank", range(4))
def test_keeps_local_experts_and_all_replicated_tensors(ep_rank):
    plan = plan_expert_reads(
        _weight_map(), num_layers=2, num_experts=8, ep_size=4, ep_rank=ep_rank
    )
    local = {2 * ep_rank, 2 * ep_rank + 1}
    assert plan.local_expert_tensors == 2 * 2 * 6
    assert "embed.weight" in plan.tensor_names
    assert "layers.0.attn.wq_a.weight" in plan.tensor_names
    assert not any(n.startswith("mtp.") for n in plan.tensor_names)
    assert plan.shard_names == {"s0"} | {f"e{e}" for e in local}


def test_fails_closed_on_missing_expert_tensors():
    names = _weight_map()
    del names["layers.1.ffn.experts.7.w3.scale"]
    with pytest.raises(ValueError, match="routed expert tensors"):
        plan_expert_reads(names, num_layers=2, num_experts=8, ep_size=4, ep_rank=0)


def test_rejects_indivisible_ep():
    with pytest.raises(ValueError):
        plan_expert_reads(
            _weight_map(), num_layers=2, num_experts=8, ep_size=3, ep_rank=0
        )


def test_mapped_safetensors_matches_safetensors_reader(tmp_path):
    """The Engram tables bypass RunAI through a hand-written header parser;
    a wrong offset or dtype map would load shifted or reinterpreted rows."""
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from sglang_tpu.models.deepseek_v41.loader import mapped_safetensors

    tensors = {
        "a": torch.arange(24, dtype=torch.float32).reshape(4, 6),
        "layers.1.engram.embed.weight": torch.randn(8, 16).to(torch.float8_e4m3fn),
        "layers.1.engram.embed.scale": torch.ones(8, 1).to(torch.float8_e8m0fnu),
    }
    path = tmp_path / "shard.safetensors"
    safetensors_torch.save_file(tensors, str(path))
    got = dict(mapped_safetensors(str(path), set(tensors)))
    for name, ref in tensors.items():
        assert got[name].dtype == ref.dtype and got[name].shape == ref.shape
        assert torch.equal(got[name].view(torch.uint8), ref.view(torch.uint8))
