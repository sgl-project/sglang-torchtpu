"""CPU contracts of the hybrid draft KV write: which layers go to which sub-pool,
and how full-pool slots translate to the SWA pool before the compiled writer."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.dflash_draft_kv_write import (
    draft_layer_groups,
    translate_draft_swa_locs,
)


def _hybrid_pool(mapping: torch.Tensor) -> SWAKVPool:
    pool = MagicMock(spec=SWAKVPool)
    # Inkling2 MTP: local layers 0, 1, 3, 4 around one global layer.
    pool.layers_mapping = {0: (0, True), 1: (1, True), 2: (0, False), 3: (2, True), 4: (3, True)}
    pool.translate_loc_from_full_to_swa = lambda loc: mapping[loc.to(torch.long)]
    return pool


class TestDraftSwaLocs(CustomTestCase):
    def test_layer_groups_follow_the_pool_mapping(self):
        pool = _hybrid_pool(torch.arange(8))
        self.assertEqual(draft_layer_groups(pool, 5), ([2], [0, 1, 3, 4]))
        plain = MagicMock()
        self.assertEqual(draft_layer_groups(plain, 3), ([0, 1, 2], []))

    def test_padded_and_unmapped_slots_stay_on_the_dummy_page(self):
        """Padded tokens sit on full slot 0 and the mapping's sentinel is -1; a
        gather at -1 would write the last SWA slot, so both land on slot 0."""
        mapping = torch.tensor([5, 6, 7, -1, 9], dtype=torch.int64)
        pool = _hybrid_pool(mapping)
        loc = torch.tensor([0, 1, 3, 4, 0], dtype=torch.int64)
        loc_2d = torch.tensor([[1, 2], [3, 0]], dtype=torch.int64)
        swa, swa_2d = translate_draft_swa_locs(pool, loc, loc_2d)
        self.assertEqual(swa.tolist(), [0, 6, 0, 9, 0])
        self.assertEqual(swa_2d.tolist(), [[6, 7], [0, 0]])
        self.assertEqual(swa.dtype, torch.int32)

    def test_plain_pool_translates_nothing(self):
        self.assertEqual(
            translate_draft_swa_locs(MagicMock(), torch.zeros(2, dtype=torch.int64), None),
            (None, None),
        )


if __name__ == "__main__":
    unittest.main()
