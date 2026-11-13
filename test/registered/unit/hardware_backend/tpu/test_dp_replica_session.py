# SPDX-License-Identifier: Apache-2.0
"""CPU checks for replica isolation and failed-startup rank restoration."""

import os
import subprocess
import sys
import textwrap
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from sglang.srt.distributed import parallel_state
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import publish_build_topology
from sglang_tpu import dp_replicas
from sglang_tpu.replica_group import ReplicaGroupCoordinator
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def reset_session(monkeypatch):
    monkeypatch.setattr(dp_replicas, "_active_session", None)
    monkeypatch.setenv("SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "0")
    monkeypatch.delenv("TPU_VISIBLE_DEVICES", raising=False)


@pytest.mark.parametrize("first_module", ["sglang_tpu.envs", "sglang.srt.platforms"])
def test_platform_discovery_without_portpicker(first_module):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import importlib
                import importlib.metadata
                import sys
                from types import SimpleNamespace
                from unittest.mock import patch

                import torch

                sys.modules["portpicker"] = None
                entry_points = importlib.metadata.entry_points
                plugin = SimpleNamespace(
                    name="replica_test",
                    load=lambda: lambda: "sglang_tpu.tpu_platform:TpuSRTPlatform",
                )

                def discover(*args, **kwargs):
                    if kwargs.get("group") == "sglang.srt.platforms":
                        return [plugin]
                    return entry_points(*args, **kwargs)

                with patch("importlib.metadata.entry_points", discover):
                    importlib.import_module(sys.argv[1])
                    from sglang.srt.platforms import current_platform
                    from sglang_tpu.dp_replicas import get_free_port
                    from sglang_tpu.envs import envs

                    assert current_platform.is_tpu()
                    assert get_free_port() > 0
                    assert envs.SGLANG_TPU_DP_SESSION_PORT.get() == 12349
                """
            ),
            first_module,
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(sys.path),
            "SGLANG_PLATFORM": "replica_test",
            "SGLANG_TPU_DP_SESSION_PORT": "12349",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tp4_ep4_groups_never_cross_engines():
    groups = dp_replicas.device_group_ranks(2, 4, 4)
    assert [group for group in groups if len(group) > 1] == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert {group[0] for group in groups if len(group) == 1} == set(range(8))


def test_ep_and_expert_tp_groups_stay_inside_engine():
    groups = dp_replicas.device_group_ranks(2, 4, 2)
    assert {(0, 2), (1, 3), (4, 6), (5, 7)} <= set(groups)
    assert {(0, 1), (2, 3), (4, 5), (6, 7)} <= set(groups)
    assert all(len({rank // 4 for rank in group}) == 1 for group in groups)
    assert len(groups) == len(set(groups))


def test_session_restores_engine_rank_even_when_backend_fails(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    session = dp_replicas.ReplicaSession(1, 2, 4, 4, device=6)
    with pytest.raises(RuntimeError, match="backend failed"):
        with session.initialize():
            assert os.environ["RANK"] == os.environ["LOCAL_RANK"] == "6"
            assert os.environ["WORLD_SIZE"] == os.environ["LOCAL_WORLD_SIZE"] == "8"
            raise RuntimeError("backend failed")
    assert os.environ["RANK"] == "2" and os.environ["WORLD_SIZE"] == "4"
    assert "LOCAL_RANK" not in os.environ and "LOCAL_WORLD_SIZE" not in os.environ


@pytest.mark.parametrize("tp_size", [1, 4])
def test_parent_preserves_ports_and_libtpu_args(monkeypatch, tp_size):
    monkeypatch.delenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", raising=False)
    monkeypatch.delenv("TORCH_TPU_INTERNAL_HANDSHAKE_PORT", raising=False)
    monkeypatch.delenv("SGLANG_TPU_DP_SESSION_PORT", raising=False)
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--unrelated=false")
    slicebuilder = Mock(
        return_value={
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(
                f"localhost:{port}" for port in range(12340, 12348)
            )
        }
    )
    monkeypatch.setattr(dp_replicas, "compute_single_host_tpu_env", slicebuilder)
    pick = Mock(side_effect=[12348, 12349])
    monkeypatch.setattr(dp_replicas, "get_free_port", pick)
    dp_replicas.prepare_host_session(8 // tp_size, tp_size)
    dp_replicas.prepare_host_session(8 // tp_size, tp_size)
    slicebuilder.assert_called_once_with(8)
    assert pick.call_count == 2
    assert len(os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")) == 8
    assert os.environ["SGLANG_TPU_DP_SESSION_PORT"] == "12349"
    assert os.environ["LIBTPU_INIT_ARGS"] == "--unrelated=false"


def test_session_applies_timeout_to_borrowed_groups(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("SGLANG_TPU_DP_SESSION_PORT", "12341")
    monkeypatch.delenv("TORCH_TPU_TOPOLOGY", raising=False)
    monkeypatch.setattr(
        dp_replicas, "_resolve_single_host_tpu_topology", lambda size: "2,2,1,2"
    )
    monkeypatch.setattr(dp_replicas.torch, "ones", Mock())
    monkeypatch.setattr(dp_replicas.dist, "is_initialized", lambda: False)
    initialize = Mock()
    create_group = Mock(side_effect=lambda ranks, **kwargs: tuple(ranks))
    monkeypatch.setattr(dp_replicas.dist, "init_process_group", initialize)
    monkeypatch.setattr(dp_replicas.dist, "new_group", create_group)

    session = dp_replicas.ReplicaSession(1, 2, 4, 4, 6, timeout_seconds=17)
    with session.initialize():
        assert os.environ["RANK"] == "6"
    assert os.environ["RANK"] == "2" and os.environ["WORLD_SIZE"] == "4"
    assert initialize.call_args.kwargs["timeout"] == timedelta(seconds=17)
    assert all(
        call.kwargs["timeout"] == timedelta(seconds=17)
        for call in create_group.call_args_list
    )
    assert dp_replicas.get_replica_session() is session
    assert session.device_groups == {(4, 5, 6, 7): (4, 5, 6, 7), (6,): (6,)}


@pytest.mark.parametrize("rank", range(8))
def test_tp4_ep2_initialization_stays_inside_replica(monkeypatch, rank):
    session = dp_replicas.ReplicaSession(rank // 4, 2, 4, 2, device=rank)
    session.device_groups = {
        ranks: object()
        for ranks in dp_replicas.device_group_ranks(2, 4, 2)
        if rank in ranks
    }
    monkeypatch.setattr(dp_replicas, "_active_session", session)
    monkeypatch.setattr(parallel_state.platforms, "current_platform", TpuSRTPlatform())
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: rank)
    monkeypatch.setattr(dist, "get_world_size", lambda: 8)
    create_group = Mock(side_effect=lambda *args, **kwargs: object())
    destroy_group = Mock()
    monkeypatch.setattr(dist, "new_group", create_group)
    monkeypatch.setattr(dist, "destroy_process_group", destroy_group)
    publish_build_topology(tp_size=4, ep_size=2, dp_size=2, world_rank=rank % 4)
    try:
        parallel_state.init_distributed_environment(
            world_size=4, rank=rank % 4, local_rank=rank, backend="tpu_dist"
        )
        parallel_state.initialize_model_parallel(backend="tpu_dist")
        world = parallel_state.get_world_group()
        assert world.world_size == 4
        assert world.rank_in_group == rank % 4
        for group in (
            world,
            parallel_state.get_tp_group(),
            parallel_state.get_moe_ep_group(),
            parallel_state.get_moe_tp_group(),
        ):
            assert group.device_group is session.device_groups[tuple(group.ranks)]
        assert parallel_state.get_moe_ep_group().world_size == 2
        assert parallel_state.get_moe_tp_group().world_size == 2
        for call in create_group.call_args_list:
            assert call.kwargs["backend"] == "gloo"
            assert call.kwargs["use_local_synchronization"] is True
            assert all(member // 4 == rank // 4 for member in call.args[0])
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()
    assert dp_replicas.get_replica_session() is None
    assert all(
        call.args[0] not in session.device_groups.values()
        for call in destroy_group.call_args_list
        if call.args
    )


def test_borrowed_group_survives_cleanup_and_next_world_is_ordinary(monkeypatch):
    monkeypatch.setattr(parallel_state.platforms, "current_platform", TpuSRTPlatform())
    dist.init_process_group(
        backend="gloo", store=dist.HashStore(), rank=0, world_size=1
    )
    try:
        borrowed = dist.new_group([0], backend="gloo")
        session = dp_replicas.ReplicaSession(0, 1, 1, 1, device=0)
        session.device_groups = {(0,): borrowed}
        monkeypatch.setattr(dp_replicas, "_active_session", session)
        publish_build_topology(tp_size=1, pp_size=1)
        parallel_state.init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend="tpu_dist"
        )
        world = parallel_state.get_world_group()
        assert isinstance(world, ReplicaGroupCoordinator)
        parallel_state.initialize_model_parallel(backend="tpu_dist")
        parallel_state.destroy_model_parallel()
        assert dp_replicas.get_replica_session() is session
        value = torch.tensor([7])
        dist.all_reduce(value, group=borrowed)
        assert value.item() == 7
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()
    assert dp_replicas.get_replica_session() is None

    dist.init_process_group(
        backend="gloo", store=dist.HashStore(), rank=0, world_size=1
    )
    try:
        parallel_state.init_distributed_environment(
            world_size=1, rank=0, local_rank=0, backend="gloo"
        )
        world = parallel_state.get_world_group()
        assert type(world) is parallel_state.GroupCoordinator
        assert world.device_group is not borrowed
    finally:
        parallel_state.destroy_distributed_environment()


@pytest.mark.parametrize("tp_size", [1, 4])
def test_parent_rejects_inherited_addresses_for_another_session(monkeypatch, tp_size):
    monkeypatch.setenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", "localhost:1,localhost:2")
    with pytest.raises(ValueError, match="session of 8 workers"):
        dp_replicas.prepare_host_session(8 // tp_size, tp_size)


def test_single_engine_needs_no_replica_session(monkeypatch):
    monkeypatch.delenv("TORCH_TPU_SLICEBUILDER_ADDRESSES", raising=False)
    pick = Mock()
    monkeypatch.setattr(dp_replicas, "get_free_port", pick)
    dp_replicas.prepare_host_session(1, 1)
    pick.assert_not_called()
    assert "TORCH_TPU_SLICEBUILDER_ADDRESSES" not in os.environ


def test_tp1_session_keeps_distributed_initialization_engine_local(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("TORCH_TPU_TOPOLOGY", raising=False)
    monkeypatch.setattr(
        dp_replicas, "_resolve_single_host_tpu_topology", lambda size: "2,2,1,2"
    )
    seen = []
    tensor = Mock()
    tensor.item.side_effect = lambda: seen.append(dict(os.environ))
    monkeypatch.setattr(dp_replicas.torch, "ones", lambda *args, **kwargs: tensor)
    initialize = Mock()
    create_group = Mock()
    monkeypatch.setattr(dp_replicas.dist, "init_process_group", initialize)
    monkeypatch.setattr(dp_replicas.dist, "new_group", create_group)
    session = dp_replicas.ReplicaSession(3, 8, 1, 1, device=3)
    with session.initialize():
        assert os.environ["RANK"] == "3" and os.environ["WORLD_SIZE"] == "8"
    assert seen[0]["TPU_VISIBLE_DEVICES"] == "3"
    assert seen[0]["RANK"] == "3" and seen[0]["WORLD_SIZE"] == "8"
    assert os.environ["RANK"] == "0" and os.environ["WORLD_SIZE"] == "1"
    initialize.assert_not_called()
    create_group.assert_not_called()
    assert dp_replicas.get_replica_session() is None


@pytest.mark.parametrize("tp_size", [1, 4])
def test_bad_session_topology_is_rejected_before_device_work(monkeypatch, tp_size):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", str(tp_size))
    monkeypatch.setenv("TORCH_TPU_TOPOLOGY", "1,1,1,2")
    monkeypatch.setattr(
        dp_replicas, "_resolve_single_host_tpu_topology", lambda size: "2,2,1,2"
    )
    materialize = Mock()
    monkeypatch.setattr(dp_replicas.torch, "ones", materialize)
    session = dp_replicas.ReplicaSession(0, 8 // tp_size, tp_size, 1, device=0)
    with pytest.raises(ValueError, match="is not the topology"):
        with session.initialize():
            pass
    materialize.assert_not_called()
    assert os.environ["WORLD_SIZE"] == str(tp_size)


@pytest.mark.parametrize("tp_size", [1, 4])
@pytest.mark.parametrize("disaggregation_mode", ["null", "prefill", "decode"])
@pytest.mark.parametrize("speculative_algorithm", [None, "EAGLE"])
def test_worker_prepares_replicas_at_every_tp_size(
    monkeypatch, tp_size, disaggregation_mode, speculative_algorithm
):
    cfg = SimpleNamespace(
        dp_size=8 // tp_size,
        tp_size=tp_size,
        ep_size=tp_size,
        pp_size=1,
        nnodes=1,
        use_ray=False,
        base_gpu_id=0,
        gpu_id_step=1,
        attn_dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        moe_dp_size=1,
        shared_experts_tp_size=None,
        # A draft worker builds its groups on the engine's precreated device groups.
        speculative_algorithm=speculative_algorithm,
        disaggregation_mode=disaggregation_mode,
        dist_timeout=17,
    )
    monkeypatch.delenv("SGLANG_PLATFORM", raising=False)
    monkeypatch.setenv("TPU_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SGLANG_TPU_PD_COLOCATED", "1")
    monkeypatch.setattr("sglang_tpu.tpu_platform._resolving_view", lambda args: cfg)
    platform = TpuSRTPlatform()
    platform.prepare_worker(cfg, tp_rank=0, pp_rank=0, dp_rank=1)
    session = platform._dp_replica_session
    assert (session.dp_rank, session.tp_size, session.device) == (1, tp_size, tp_size)
    assert session.timeout_seconds == 17
    assert "TPU_VISIBLE_DEVICES" not in os.environ
    assert "SGLANG_TPU_PD_COLOCATED" not in os.environ


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
