"""Host contracts for V4.1 FP8 storage and flat custom-op fake outputs.

These tests isolate definitions from TPU runtime imports; they do not assert
TPU lowering, DMA correctness or model quality. Hardware tests cover kernels.
"""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[4]


def definition(relative, name, namespace, *, nested=False):
    """Execute actual source definitions without importing the TPU runtime."""
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    nodes = ast.walk(tree) if nested else tree.body
    node = next(n for n in nodes if getattr(n, "name", None) == name)
    # Only checked-in source definitions are executed, never external input.
    exec(  # noqa: S102
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


def layout_type():
    from dataclasses import dataclass

    return definition(
        "sglang_tpu/kernels/index_selection/packed_fp8_cache.py",
        "PackedFP8Layout",
        {"dataclass": dataclass, "PAGE_ROWS": 33, "PAGE_RECORDS": 128},
    )


@pytest.mark.parametrize("storage", [64, 128])
@pytest.mark.parametrize("scheduler", [128, 256])
def test_allocation_and_scheduler_page_mapping(storage, scheduler):
    pool_type = definition(
        "sglang_tpu/dsv41/memory_pool.py", "PackedFP8IndexPool", {"torch": torch}
    )
    pool = pool_type(4096, storage, 128, 2, "cpu", record_alignment=256)
    rows = (33 * storage + 127) // 128
    assert pool.k_buffer[0].shape == (pool.num_pages * rows, 128)
    assert pool.get_kv_size_bytes() == (2 * pool.num_pages * rows * 512, 0)
    assert pool.contiguous_page_row_buffers()[0].shape == (pool.num_pages, rows * 128)
    geometry = definition(
        "sglang_tpu/dsv41/custom_ops.py",
        "index_geometry",
        {"jnp": jnp, "PackedFP8Layout": layout_type()},
    )
    physical, layout = geometry(
        pool.k_buffer[0], jnp.array([[1, 3, -1]], jnp.int32), scheduler, storage
    )
    parts = scheduler // storage
    np.testing.assert_array_equal(
        physical,
        [[*range(parts, 2 * parts), *range(3 * parts, 4 * parts), *([-1] * parts)]],
    )
    assert layout.page_size == storage
    layout.validate((pool.k_buffer[0].shape[0],))


def test_query_is_fp8_and_matches_writer_quantization():
    quantize = definition(
        "sglang_tpu/dsv41/attention_backend.py",
        "quantize_index_query",
        {"torch": torch},
    )
    writer_quantize = definition(
        "sglang_tpu/kernels/index_selection/packed_fp8_cache.py",
        "quantize_index_rows",
        {"jax": jax, "jnp": jnp},
    )
    generator = torch.Generator().manual_seed(41)
    values = torch.randn(17, 32, 128, generator=generator, dtype=torch.bfloat16)
    values[0] = 0
    query, scale = quantize(values)
    expected, expected_scale, _ = writer_quantize(
        jnp.asarray(values.float().numpy(), dtype=jnp.bfloat16)
    )
    assert query.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    np.testing.assert_array_equal(
        query.float().numpy(), np.asarray(expected.astype(jnp.float32))
    )
    np.testing.assert_array_equal(scale.numpy(), np.asarray(expected_scale))


@pytest.mark.parametrize("tokens,max_queries", [(8, 1), (128, 128)])
def test_prepare_fake_shapes_follow_union_capacity(tokens, max_queries):
    fake = definition(
        "sglang_tpu/dsv41/custom_ops.py", "fake_prepare", {"torch": torch}, nested=True
    )
    groups, columns, count, block, topk = 8, 128, 2048, 8, 512
    words = torch.empty((17 * 1024, 128), dtype=torch.int32)
    pages = torch.empty((groups, columns), dtype=torch.int32)
    requests = torch.empty(tokens, dtype=torch.int32)
    candidates = torch.empty((tokens, count), dtype=torch.int32)
    result = fake(
        words, pages, requests, requests, candidates, 128, 64, topk, block, max_queries
    )
    width = columns * 128
    capacity = min(width // block, min(tokens, max_queries) * count + topk // block)
    capacity = (max(1, capacity) + 15) // 16 * 16
    shapes = [
        (groups * capacity * block // 128 * 33, 128),
        (groups, capacity),
        (groups, capacity * block // 128),
        (tokens,),
        (tokens, count),
        (tokens,),
        (),
    ]
    assert [tuple(x.shape) for x in result] == shapes
    assert [x.dtype for x in result] == [torch.int32] * 5 + [torch.bool] * 2


def test_pool_factory_uses_fp8_even_for_core_force_fp4_selector():
    factory = definition(
        "sglang_tpu/dsv41/memory_pool.py",
        "_make_indexer_pool",
        {"PackedFP8IndexPool": lambda *args, **kw: (args, kw)},
        nested=True,
    )
    for force in (False, True):
        args, kwargs = factory(
            SimpleNamespace(page_size=256),
            4096,
            64,
            torch.bfloat16,
            128,
            2,
            "cpu",
            False,
            force_fp4=force,
        )
        assert args == (4096, 64, 128, 2, "cpu")
        assert kwargs == {"record_alignment": 256}


@pytest.mark.parametrize("storage,per_record", [(64, 652), (128, 648)])
def test_plugin_budget_matches_physical_packed_pages(storage, per_record):
    budget = definition("sglang_tpu/hooks/dsv41.py", "cache_bytes", {})
    config = SimpleNamespace(
        _is_tpu_dsv4=True, attn_head_dim=512, stage_owner_layers={1: [20], 2: [2]}
    )
    module = SimpleNamespace(DSV41_INDEX_PAGE_SIZE=storage)
    with patch.dict(
        sys.modules, {"sglang.srt.mem_cache.deepseek_v4_memory_pool": module}
    ):
        assert budget(100, config) == 100 + per_record + per_record / 2
        config.stage_owner_layers = {1: [20]}
        assert budget(0, config) == per_record
        config.stage_owner_layers = {2: [2]}
        assert budget(0, config) == per_record / 2
        config._is_tpu_dsv4 = False
        assert budget(100, config) == 100


@pytest.mark.parametrize("tokens,groups", [(0, 8), (8, 0), (0, 0)])
def test_prepare_fake_empty_owner_matches_kernel(tokens, groups):
    fake = definition(
        "sglang_tpu/dsv41/custom_ops.py", "fake_prepare", {"torch": torch}, nested=True
    )
    words = torch.empty((17, 128), dtype=torch.int32)
    pages = torch.empty((groups, 8), dtype=torch.int32)
    requests = torch.empty(tokens, dtype=torch.int32)
    candidates = torch.empty((tokens, 128), dtype=torch.int32)
    result = fake(words, pages, requests, requests, candidates, 128, 64, 512, 8, 1)
    assert result[0].shape == (33, 128)
    assert result[1].shape[0] == groups
    assert result[3].shape == (tokens,)


def test_publish_preserves_pool_identity_and_supplies_physical_page_size():
    publish = definition("sglang_tpu/dsv41/custom_ops.py", "publish_source", {})
    latent = torch.zeros((4, 128), dtype=torch.int32)
    index = torch.zeros((17, 128), dtype=torch.int32)
    scales = torch.zeros(4)
    seen = []

    def write_index(words, records, key, after, page_records):
        seen.append(page_records)
        return words + 3

    pool = SimpleNamespace(
        low_ratio_cache=lambda layer, **kw: index if kw.get("index") else latent,
        low_ratio_scales=lambda layer: scales,
        index_page_records=lambda layer: 64,
    )
    ops = SimpleNamespace(
        compressed=lambda *args: (latent + 1, scales + 2), index=write_index
    )
    identities = [x.data_ptr() for x in (latent, scales, index)]
    publish(ops, pool, 20, None, None, None, None)
    assert [x.data_ptr() for x in (latent, scales, index)] == identities
    assert torch.all(latent == 1) and torch.all(scales == 2) and torch.all(index == 3)
    assert seen == [64]


def test_candidate_reuse_is_scoped_by_source_and_publication():
    quantize = definition(
        "sglang_tpu/dsv41/attention_backend.py",
        "quantize_index_query",
        {"torch": torch},
    )
    forward = definition(
        "sglang_tpu/dsv41/attention_backend.py",
        "forward_low_ratio_sources",
        {
            "torch": torch,
            "quantize_index_query": quantize,
            "apply_tpu_dsv4_rope": lambda q, *args: q,
        },
        nested=True,
    )
    calls = []
    metadata = SimpleNamespace(
        valid=torch.ones(2, dtype=torch.bool),
        candidates=torch.zeros((2, 128), dtype=torch.int32),
        index_requests=torch.arange(2, dtype=torch.int32),
        full_pages=torch.ones((2, 8), dtype=torch.int32),
        index_versions={},
        candidate_version=1,
        prepared_index={},
        index_max_queries=1,
        low_indices={},
    )
    words = torch.zeros((17, 128), dtype=torch.int32)
    pool = SimpleNamespace(
        index_page_records=lambda layer: 64,
        source_layer_of=lambda layer: 20,
        low_ratio_cache=lambda *args, **kw: words,
    )

    def prepare(*args):
        calls.append("gather")
        return (torch.zeros(1),) * 7

    ops = SimpleNamespace(
        prepare_index=prepare,
        select_prepared=lambda *args: torch.zeros((2, 512), dtype=torch.int32),
    )
    backend = SimpleNamespace(
        forward_metadata=metadata,
        token_to_kv_pool=pool,
        ops=ops,
        page_size=256,
        index_topk=512,
    )
    indexer = SimpleNamespace(
        n_heads=32,
        index_head_dim=128,
        wq_b=lambda x: (torch.ones((2, 4096), dtype=torch.bfloat16), None),
        head_weights_raw=lambda x: torch.ones((2, 32)),
        head_weight_scale=1.0,
        is_candidate_source=False,
        uses_candidates=True,
        candidate_topk_blocks=128,
        candidate_block_size=8,
    )
    layer = SimpleNamespace(
        compress_ratio=1, compressor=None, indexer=indexer, layer_id=24, freqs_cis=None
    )

    def run():
        forward(
            backend,
            layer=layer,
            x=torch.ones((2, 1)),
            q_lora=None,
            positions=torch.tensor([1023, 1023]),
            forward_batch=None,
        )

    run()
    run()
    assert calls == ["gather"]
    metadata.candidate_version += 1
    run()
    assert len(calls) == 2
    metadata.index_versions[20] = 1
    run()
    assert len(calls) == 3
