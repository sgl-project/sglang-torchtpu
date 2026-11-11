# SPDX-License-Identifier: Apache-2.0
# Adapted for SGLang-TorchTPU from vllm-project/vllm-torchtpu (Apache-2.0).
# Source: tests/kernels/experimental/push_fused_moe/test_routing.py
# Revision: ef2599d3d4342f41d2ad223b266e126299862b53
"""CPU checks for exact routing dispatch eligibility."""

import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from sglang_tpu.moe.pallas.fused_moe_v2.push import fusion, routing


def supported(tokens=288, **changes):
    i32 = jnp.zeros((1,), jnp.int32)
    gathered = SimpleNamespace(
        topk=6, n_tokens=tokens * 16, local_blocks=i32, block_hist=i32
    )
    values = dict(
        fuse_metadata=True,
        t_local=tokens,
        gathered=gathered,
        e_total=384,
        ep=16,
        tile_m=128,
        block=routing.host.routing_block(tokens, 6),
        shard_stride=routing.host.ragged_stride_bound(tokens * 16, 6, 384, 128),
        runs=(i32,) * 4,
        me=i32,
    )
    for key in ("topk", "n_tokens", "local_blocks", "block_hist"):
        if key in changes:
            setattr(gathered, key, changes.pop(key))
    values.update(changes)
    try:
        routing.prepare_routing(**values)
    except DispatchReached as exc:
        return str(exc) == "fused"
    raise AssertionError("routing did not reach either dispatch path")


class DispatchReached(Exception):
    pass


def stop(path):
    def reached(*args, **kwargs):
        raise DispatchReached(path)

    return reached


@pytest.fixture(autouse=True)
def tpu_v7(monkeypatch):
    monkeypatch.setattr(routing, "build_routing_tables_sharded", lambda *a, **k: None)
    monkeypatch.setattr(routing.host, "local_slab_rows", lambda *a, **k: None)
    monkeypatch.setattr(
        routing.host, "shard_transport_tables_in_blocks", stop("fallback")
    )
    monkeypatch.setattr(routing, "fuse_function", stop("fused"))
    monkeypatch.setattr(fusion, "get_tpu_generation", lambda: 7)


@pytest.mark.parametrize("tokens", [96, 192, 256, 288, 384])
def test_exact_approved_tokens(tokens):
    assert supported(tokens)
    stride = routing.host.ragged_stride_bound(tokens * 16, 6, 384, 128)
    assert not supported(tokens, shard_stride=stride + 128)
    assert not supported(tokens, block=routing.host.routing_block(tokens, 6) + 128)


@pytest.mark.parametrize(
    "tokens", [1, 95, 97, 128, 191, 193, 255, 257, 287, 289, 383, 385, 512]
)
def test_unapproved_tokens(tokens):
    assert not supported(tokens)


@pytest.mark.parametrize(
    "changes",
    [
        {"fuse_metadata": False},
        {"topk": 4},
        {"e_total": 383},
        {"ep": 8},
        {"tile_m": 256},
        {"n_tokens": 288 * 16 + 1},
    ],
)
def test_contract_guards(changes):
    assert not supported(**changes)


@pytest.mark.parametrize("field", ["me", "local_blocks", "block_hist", "runs"])
def test_dtype_guards(field):
    bad = jax.ShapeDtypeStruct((1,), jnp.int16)
    assert not supported(**{field: (bad,) * 4 if field == "runs" else bad})


@pytest.mark.parametrize("version", [-1, 6])
def test_backend_guards(monkeypatch, version):
    monkeypatch.setattr(fusion, "get_tpu_generation", lambda: version)
    assert not supported()


def test_routing_fuses_during_cpu_export(monkeypatch):
    from sglang_tpu import pallas_export

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(fusion, "get_tpu_generation", pallas_export.get_tpu_generation)
    monkeypatch.setattr(pallas_export, "get_pallas_tpu_device_kind", lambda: "TPU7x")
    with pallas_export.tpu_info_for_cpu_export():
        assert supported()
