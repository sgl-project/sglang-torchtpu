# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash attention metadata as a direct-tracer graph input.

The direct tracer passes traced_attention_metadata(backend) through JAX
tree_util: for GLM-5.3-Flash's HybridLinearAttnBackend that is the sparse MLA
backend's PallasForwardMetadata and the KDA backend's GdnTracedMetadata. A
deferred call fetches every leaf back through its key path. Guards that every
key path resolves the way direct_compile resolves it, that the pooled
indexer's raw-ring bases are a graph input rather than a constant, and that
the only non-tensor leaves are the static scalars direct_compile bakes in.
Runs on the host; JAX comes with the TPU stack.
"""

import unittest

import torch
from jax import tree_util

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_gdn_pallas import GdnTracedMetadata, TpuGDNAttnBackend
from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend
from sglang_tpu.attention_pallas_rpa_base import PallasForwardMetadata
from sglang_tpu.compilation.direct_compile import _STATIC_INPUT_TYPES
from sglang_tpu.compilation.hybrid_linear_direct_compile import (
    traced_attention_metadata,
)

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

BS = 4
_KEYS = (tree_util.GetAttrKey, tree_util.DictKey, tree_util.SequenceKey)


def _i32(n: int, start: int = 0) -> torch.Tensor:
    return torch.arange(start, start + n, dtype=torch.int32)


def _sparse_mla_metadata(raw_ring: bool) -> PallasForwardMetadata:
    return PallasForwardMetadata(
        kv_lens=_i32(BS),
        cu_q_lens=_i32(BS + 1),
        cu_kv_lens=_i32(BS + 1),
        page_indices=_i32(BS * 8),
        distribution=_i32(3),
        active_num_tokens=torch.tensor(BS, dtype=torch.int32),
        raw_page_base=_i32(BS, start=100) if raw_ring else None,
    )


def _kda_backend(verify: bool) -> TpuGDNAttnBackend:
    backend = object.__new__(TpuGDNAttnBackend)
    backend._set_traced_metadata(
        GdnTracedMetadata(
            distribution=_i32(3),
            seq_lens=_i32(BS),
            state_indices=_i32(BS),
            query_start_loc=_i32(BS + 1),
            track_indices=_i32(BS) if verify else None,
            track_end=_i32(BS) if verify else None,
            read_offsets=_i32(8) if verify else None,
            req_pool_indices=_i32(BS) if verify else None,
        )
    )
    return backend


def _hybrid_backend(*, raw_ring: bool, verify: bool) -> HybridLinearAttnBackend:
    backend = object.__new__(HybridLinearAttnBackend)
    full = object.__new__(TpuPallasSparseMlaAttnBackend)
    full.forward_metadata = _sparse_mla_metadata(raw_ring)
    backend.full_attn_backend = full
    backend.linear_attn_backend = _kda_backend(verify)
    return backend


def _resolve(root, path):
    value = root
    for key in path:
        if isinstance(key, tree_util.GetAttrKey):
            value = getattr(value, key.name)
        elif isinstance(key, tree_util.DictKey):
            value = value[key.key]
        else:
            value = value[key.idx]
    return value


class TestGlm5NextMetadataPytree(CustomTestCase):
    def test_key_paths_resolve_like_direct_compile(self):
        """direct_compile._get_path accepts only GetAttr / Dict / Sequence keys."""
        for raw_ring in (False, True):
            for verify in (False, True):
                with self.subTest(raw_ring=raw_ring, verify=verify):
                    metadata = traced_attention_metadata(
                        _hybrid_backend(raw_ring=raw_ring, verify=verify)
                    )
                    paths = tree_util.tree_flatten_with_path(metadata)[0]
                    self.assertEqual(len(paths), len(tree_util.tree_leaves(metadata)))
                    for path, leaf in paths:
                        self.assertTrue(
                            all(isinstance(key, _KEYS) for key in path), str(path)
                        )
                        self.assertIs(_resolve(metadata, path), leaf)

    def test_raw_ring_bases_are_a_graph_input(self):
        """Each row's first ring page depends on its request slot: a constant
        would bake one batch's slots into the graph."""
        metadata = traced_attention_metadata(
            _hybrid_backend(raw_ring=True, verify=False)
        )
        full, _ = metadata
        leaves = tree_util.tree_leaves(metadata)
        self.assertTrue(any(leaf is full.raw_page_base for leaf in leaves))

    def test_only_static_scalars_besides_tensors(self):
        """Scalars are baked into the trace; anything else must be a tensor."""
        metadata = traced_attention_metadata(
            _hybrid_backend(raw_ring=True, verify=True)
        )
        for leaf in tree_util.tree_leaves(metadata):
            self.assertIsInstance(leaf, (torch.Tensor, *_STATIC_INPUT_TYPES))

    def test_same_mode_same_structure(self):
        """Warmup and serving build the metadata alike, so one batch's values
        never change the flattened structure the executable binds."""
        a = traced_attention_metadata(_hybrid_backend(raw_ring=True, verify=True))
        b = traced_attention_metadata(_hybrid_backend(raw_ring=True, verify=True))
        b[0].kv_lens.add_(7)
        self.assertEqual(tree_util.tree_structure(a), tree_util.tree_structure(b))


if __name__ == "__main__":
    unittest.main()
