# SPDX-License-Identifier: Apache-2.0
"""Main-KV gather groups of the DeepSeek-V4.1 CSA2 backend.

A layer may reuse an earlier layer's gathered main KV only while neither the
main KV source nor the selection has changed since: a new kv_source layer
publishes new main KV, a new index_source layer selects new records. The
backend keeps a group's gather on the per-forward metadata from the group's
first layer to its last. Runs on the host; JAX comes with the TPU stack.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41.attention_backend import (
    DSV41TPUAttnBackend,
    main_gather_groups,
)

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

# DeepSeek-V4.1-Flash text config: 40 layers + 3 MTP (ratio 0).
V41_RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
V41_KV_SOURCES = [2, 8, 14, 20]
V41_INDEX_SOURCES = [2, 8, 14, 20, 24, 28, 32, 36]


def _group_map(ratios, kv_sources, index_sources):
    def source_layer_of(layer_id):
        return max(
            s for s in kv_sources if s <= layer_id and ratios[s] == ratios[layer_id]
        )

    return main_gather_groups(
        compress_ratios=ratios,
        index_source_layer_ids=index_sources,
        source_layer_of=source_layer_of,
        layer_ids=range(len(ratios)),
    )


def _groups(ratios, kv_sources, index_sources):
    return sorted(set(_group_map(ratios, kv_sources, index_sources).values()))


class _RecordingOps:
    """csa2_ops stand-in. Tests pass the layer id as ``q``; a fresh gather keeps
    ("kv", layer id) and every reuse records the gather it attended."""

    def __init__(self):
        self.gathers = []  # (layer id, keep_gathered)
        self.reads = {}  # layer id -> gather it attended

    def attention(self, q, *_, main, keep_gathered):
        self.gathers.append((q, keep_gathered))
        return (q, ("kv", q)) if keep_gathered else q

    def attention_gathered(self, q, gathered, **_):
        self.reads[q] = gathered
        return q


class TestMainGatherGroups(CustomTestCase):
    def test_deepseek_v41_flash_layout(self):
        groups = _groups(V41_RATIOS, V41_KV_SOURCES, V41_INDEX_SOURCES)
        self.assertEqual(
            groups,
            [(2, 7), (8, 13), (14, 19), (20, 23), (24, 27)]
            + [(28, 31), (32, 35), (36, 39)],
        )

    def test_new_kv_source_splits_a_group_without_a_new_selection(self):
        # Layer 4 publishes new main KV under layer 1's selection.
        ratios = [0, 1, 1, 1, 1, 1]
        self.assertEqual(_groups(ratios, [1, 4], [1]), [(1, 3), (4, 5)])

    def test_forward_frees_each_kept_gather_after_its_last_layer(self):
        """Later layers attend their own group's first gather, at most one gather
        is kept at a time, and none outlives the forward: a gather left on the
        metadata holds [T, 512, 128] words into the next program."""
        group_map = _group_map(V41_RATIOS, V41_KV_SOURCES, V41_INDEX_SOURCES)
        backend = object.__new__(DSV41TPUAttnBackend)
        backend.gather_groups = group_map
        backend.csa2_ops = ops = _RecordingOps()
        backend.token_to_kv_pool = SimpleNamespace(
            get_main_kv=lambda layer_id: (None, None),
            main_page_records=lambda layer_id: 0,
        )
        metadata = SimpleNamespace(
            selected={1: "ratio-1 selection", 2: "ratio-2 selection"},
            full_pages=None,
            swa_meta=None,
            query_start_loc=None,
            positions=None,
            gathered_main={},
        )

        def attend(layer_id, ratio):
            backend._attend_main(
                q=layer_id,
                swa=None,
                sink=None,
                cos_sin=None,
                layer=SimpleNamespace(layer_id=layer_id, scaling=1.0),
                compress_ratio=ratio,
                metadata=metadata,
            )

        for layer_id, ratio in enumerate(V41_RATIOS):
            if ratio:
                attend(layer_id, ratio)
                self.assertLessEqual(len(metadata.gathered_main), 1)
        # A layer outside every group gathers without keeping it.
        attend(99, 1)

        self.assertEqual(
            ops.gathers, [(first, True) for first in V41_INDEX_SOURCES] + [(99, False)]
        )
        self.assertEqual(
            ops.reads,
            {
                layer_id: ("kv", first)
                for layer_id, (first, _) in group_map.items()
                if layer_id != first
            },
        )
        self.assertEqual(metadata.gathered_main, {})


if __name__ == "__main__":
    unittest.main()
