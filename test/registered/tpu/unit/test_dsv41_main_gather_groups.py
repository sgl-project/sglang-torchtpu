# SPDX-License-Identifier: Apache-2.0
"""Main-KV gather groups of the DeepSeek-V4.1 CSA2 backend.

A layer may reuse an earlier layer's gathered main KV only while neither the
main KV source nor the selection has changed since: a new kv_source layer
publishes new main KV, a new index_source layer selects new records. Runs on
the host; JAX comes with the TPU stack.
"""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41.attention_backend import main_gather_groups

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)


def _groups(ratios, kv_sources, index_sources):
    def source_layer_of(layer_id):
        return max(
            s for s in kv_sources if s <= layer_id and ratios[s] == ratios[layer_id]
        )

    groups = main_gather_groups(
        compress_ratios=ratios,
        index_source_layer_ids=index_sources,
        source_layer_of=source_layer_of,
        layer_ids=range(len(ratios)),
    )
    return sorted(set(groups.values()))


class TestMainGatherGroups(CustomTestCase):
    def test_deepseek_v41_flash_layout(self):
        # DeepSeek-V4.1-Flash text config: 40 layers + 3 MTP (ratio 0).
        ratios = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
        groups = _groups(ratios, [2, 8, 14, 20], [2, 8, 14, 20, 24, 28, 32, 36])
        self.assertEqual(
            groups,
            [(2, 7), (8, 13), (14, 19), (20, 23), (24, 27)]
            + [(28, 31), (32, 35), (36, 39)],
        )

    def test_new_kv_source_splits_a_group_without_a_new_selection(self):
        # Layer 4 publishes new main KV under layer 1's selection.
        ratios = [0, 1, 1, 1, 1, 1]
        self.assertEqual(_groups(ratios, [1, 4], [1]), [(1, 3), (4, 5)])


if __name__ == "__main__":
    unittest.main()
