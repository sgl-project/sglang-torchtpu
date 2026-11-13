import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2 import (
    denoising_av,
)

MAIN_ONLY = StageParallelismType.MAIN_RANK_ONLY
MAIN_AND_SEND = StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS
REPLICATED = StageParallelismType.REPLICATED


def _parallelism_type(*, cfg_parallel, rank_subset_collectives, decode_every_rank):
    stage = object.__new__(denoising_av.LTX2RefinementStage)
    stage.server_args = SimpleNamespace(enable_cfg_parallel=cfg_parallel)
    stage.vae = None
    platform = SimpleNamespace(
        supports_rank_subset_collectives=lambda: rank_subset_collectives
    )
    with (
        patch.object(denoising_av, "current_platform", platform),
        patch.object(
            denoising_av, "decode_uses_every_rank", return_value=decode_every_rank
        ),
    ):
        return stage.parallelism_type


class TestLtx2RefinementStageParallelism(unittest.TestCase):
    def test_stage2_placement(self):
        cases = [
            # cfg_parallel, rank_subset_collectives, decode_every_rank, expected
            (True, True, False, MAIN_ONLY),
            (True, True, True, MAIN_AND_SEND),
            (True, False, False, REPLICATED),
            (True, False, True, REPLICATED),
            (False, True, False, REPLICATED),
            (False, False, False, REPLICATED),
        ]
        for cfg_parallel, rank_subset, every_rank, expected in cases:
            with self.subTest(
                cfg_parallel=cfg_parallel,
                rank_subset_collectives=rank_subset,
                decode_every_rank=every_rank,
            ):
                self.assertEqual(
                    _parallelism_type(
                        cfg_parallel=cfg_parallel,
                        rank_subset_collectives=rank_subset,
                        decode_every_rank=every_rank,
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
