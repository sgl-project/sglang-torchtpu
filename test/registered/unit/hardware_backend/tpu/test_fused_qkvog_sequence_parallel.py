# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import patch

import pytest
import torch

from sglang.srt.hardware_backend.tpu import fused_qkvog_sequence_parallel as qkvog
from sglang.srt.hardware_backend.tpu.kernels.sequence_parallel.fused_rmsnorm_qkvog_mm_a2a._schedule import (
    AllToAllPlan,
    HybridGatherAllToAllPlan,
    make_schedule_plan,
    supports_token_count,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")

_PERMUTED_LOGICAL_TO_PHYSICAL = (
    8,
    9,
    12,
    13,
    14,
    15,
    10,
    11,
    0,
    1,
    4,
    5,
    6,
    7,
    2,
    3,
)
_PERMUTED_PHYSICAL_TO_LOGICAL = tuple(
    _PERMUTED_LOGICAL_TO_PHYSICAL.index(rank) for rank in range(16)
)


def test_nonidentity_rank_map_builds_local_metadata():
    mapping = qkvog.validate_physical_to_logical(_PERMUTED_PHYSICAL_TO_LOGICAL)
    assert mapping == (8, 9, 14, 15, 10, 11, 12, 13, 0, 1, 6, 7, 2, 3, 4, 5)

    metadata = qkvog.make_local_rank_metadata(
        physical_id=12,
        logical_id=2,
        physical_to_logical=mapping,
    )
    assert metadata.dtype == torch.int32
    assert metadata.shape == (18,)
    assert metadata.tolist() == [12, 2, *mapping]


def test_rank_map_rejects_duplicate_logical_rank():
    mapping = list(range(16))
    mapping[2] = mapping[0]
    with pytest.raises(ValueError, match="TP16 permutation"):
        qkvog.validate_physical_to_logical(mapping)


def test_rank_map_rejects_noncontiguous_core_pairs():
    mapping = list(range(16))
    mapping[0], mapping[2] = mapping[2], mapping[0]
    with pytest.raises(ValueError, match="ordered two-core pairs"):
        qkvog.validate_physical_to_logical(mapping)


def test_device_gate_is_tpu7x_only():
    target = "sglang.srt.hardware_backend.tpu.pallas_export.get_pallas_tpu_device_kind"
    with patch(target, return_value="TPU7x"):
        assert qkvog.supports_tpu_sp_qkvog_device()
        assert qkvog.tpu_sp_qkvog_device_unsupported_reason() is None
    with patch(target, return_value="TPU v6 lite"):
        assert not qkvog.supports_tpu_sp_qkvog_device()
        reason = qkvog.tpu_sp_qkvog_device_unsupported_reason()
        assert "require TPU7x" in reason
        assert "TPU v6 lite" in reason


def test_model_config_contract_is_owned_by_the_kernel_adapter():
    config = dict(
        tp_size=16,
        hidden_size=8192,
        head_dim=64,
        query_heads_per_rank=8,
        kv_heads_per_rank=1,
        output_gate_heads_per_rank=8,
    )
    assert qkvog.tpu_sp_qkvog_config_unsupported_reason(**config) is None

    reason = qkvog.tpu_sp_qkvog_config_unsupported_reason(
        **config,
        world_size=32,
    )
    assert reason is not None
    assert "DP1/TP16" in reason
    assert "WORLD_SIZE=32" in reason

    config["head_dim"] = 128
    reason = qkvog.tpu_sp_qkvog_config_unsupported_reason(**config)
    assert reason is not None
    assert "head size 64" in reason
    assert "head size 128" in reason


def test_weight_blocks_are_packed_in_physical_destination_order():
    mapping = _PERMUTED_PHYSICAL_TO_LOGICAL
    with (
        patch.object(qkvog, "COLUMNS_PER_DESTINATION", 2),
        patch.object(qkvog, "HIDDEN_SIZE", 3),
    ):
        logical = (
            torch.arange(16 * 2 * 3, dtype=torch.float32)
            .to(torch.bfloat16)
            .reshape(32, 3)
        )
        packed = qkvog.pack_local_qkvog_weights_by_physical_destination(
            logical,
            mapping,
        )

    expected = logical.reshape(16, 2, 3)[list(mapping)].reshape(32, 3).t()
    torch.testing.assert_close(packed, expected)
    assert packed.is_contiguous()


@pytest.mark.parametrize(
    ("tokens", "plan_type", "name", "collective_id"),
    (
        (1024, HybridGatherAllToAllPlan, "gather-8-sources", 100),
        (2048, HybridGatherAllToAllPlan, "gather-2-sources", 101),
        (4096, AllToAllPlan, "all-to-all", 102),
        (8192, AllToAllPlan, "all-to-all", 102),
        (16384, AllToAllPlan, "all-to-all-grid", 103),
    ),
)
def test_tuned_token_schedule(tokens, plan_type, name, collective_id):
    assert supports_token_count(tokens)
    plan = make_schedule_plan(tokens // 16)
    assert isinstance(plan, plan_type)
    assert plan.name == name
    assert plan.collective_id == collective_id


def test_schedule_owned_collective_ids_have_finite_cardinality():
    ids = {
        make_schedule_plan(tokens // 16).collective_id
        for tokens in range(4096, 16385, 1024)
    }
    ids.update(
        {
            make_schedule_plan(1024 // 16).collective_id,
            make_schedule_plan(2048 // 16).collective_id,
        }
    )
    assert ids == {100, 101, 102, 103}


def test_inlined_ici_schedule_keeps_the_tuned_hypercube_routes():
    plan = make_schedule_plan(64)
    assert isinstance(plan, HybridGatherAllToAllPlan)
    assert plan.routing_steps == (
        ((2, 0), (4, 0), (8, 0)),
        ((2, 4), (4, 8), (8, 2)),
        ((2, 12), (4, 10), (8, 6)),
    )


@pytest.mark.parametrize("tokens", (0, 512, 1000, 17408))
def test_unsupported_token_schedule_falls_back(tokens):
    assert not supports_token_count(tokens)


def test_runtime_contract_accepts_only_tuned_bf16_shapes():
    hidden = torch.empty((64, 8192), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((8192, 16 * 1152), dtype=torch.bfloat16, device="meta")
    gain = torch.empty((8192,), dtype=torch.bfloat16, device="meta")
    metadata = torch.empty((18,), dtype=torch.int32, device="meta")

    assert qkvog.can_use_tpu_sp_fused_qkvog(hidden, weight, gain, metadata)
    assert qkvog.can_use_tpu_sp_fused_qkvog(
        hidden.to(torch.float32), weight, gain, metadata
    )
    assert not qkvog.can_use_tpu_sp_fused_qkvog(
        hidden.to(torch.float16), weight, gain, metadata
    )
    assert not qkvog.can_use_tpu_sp_fused_qkvog(hidden[:32], weight, gain, metadata)
    assert not qkvog.can_use_tpu_sp_fused_qkvog(
        torch.empty((), dtype=torch.bfloat16, device="meta"),
        weight,
        gain,
        metadata,
    )


def test_layer_collective_id_is_not_forwarded_to_exported_op(monkeypatch):
    hidden = torch.empty((64, 8192), dtype=torch.bfloat16, device="meta")
    weight = torch.empty((8192, 16 * 1152), dtype=torch.bfloat16, device="meta")
    gain = torch.empty((8192,), dtype=torch.bfloat16, device="meta")
    cos = torch.empty((64, 64), dtype=torch.float32, device="meta")
    sin = torch.empty((64, 64), dtype=torch.float32, device="meta")
    metadata = torch.empty((18,), dtype=torch.int32, device="meta")
    calls = []
    sentinel = (object(), object(), object(), object())

    def exported_op(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(qkvog, "_FUSED_QKVOG_OP", exported_op)
    kwargs = {
        "eps": 1e-5,
        "eps_qk": 1e-5,
        "scale_query_by": 1.0,
        "use_qk_norm": True,
        "use_rope": True,
    }
    for layer_collective_id in (100, 161):
        assert (
            qkvog.tpu_sp_fused_qkvog(
                hidden,
                weight,
                gain,
                cos,
                sin,
                metadata,
                collective_id=layer_collective_id,
                **kwargs,
            )
            is sentinel
        )

    assert len(calls) == 2
    for args in calls:
        assert len(args) == 11
        assert all(
            actual is expected
            for actual, expected in zip(
                args[:6],
                (
                    hidden,
                    weight,
                    gain,
                    cos,
                    sin,
                    metadata,
                ),
                strict=True,
            )
        )
        assert args[6:] == (1e-5, 1e-5, 1.0, True, True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
