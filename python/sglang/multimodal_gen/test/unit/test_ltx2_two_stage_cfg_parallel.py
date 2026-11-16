import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

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


class TestLtx2RefinementStageSharesMainRankLatents(unittest.TestCase):
    def _share(self, *, rank_subset_collectives, decode_every_rank):
        stage = object.__new__(denoising_av.LTX2RefinementStage)
        stage.server_args = SimpleNamespace(enable_cfg_parallel=True)
        stage.vae = None
        batch = SimpleNamespace(latents=torch.ones(2, 3), audio_latents=torch.ones(4))
        world = MagicMock()
        world.broadcast.side_effect = lambda tensor, src: torch.zeros_like(tensor)
        platform = SimpleNamespace(
            supports_rank_subset_collectives=lambda: rank_subset_collectives
        )
        with (
            patch.object(denoising_av, "current_platform", platform),
            patch.object(
                denoising_av, "decode_uses_every_rank", return_value=decode_every_rank
            ),
            patch.object(denoising_av, "get_world_group", return_value=world),
        ):
            stage._share_main_rank_latents(batch)
        return batch, world

    def test_every_rank_decodes_rank_0_latents(self):
        # Stage 2 ran on every CFG rank and the decode spreads across ranks.
        batch, world = self._share(
            rank_subset_collectives=False, decode_every_rank=True
        )
        self.assertEqual(
            [c.kwargs["src"] for c in world.broadcast.call_args_list], [0, 0]
        )
        self.assertEqual(batch.latents.sum().item(), 0)
        self.assertEqual(batch.audio_latents.sum().item(), 0)

    def test_no_broadcast_otherwise(self):
        for rank_subset, every_rank in ((True, True), (False, False)):
            with self.subTest(
                rank_subset_collectives=rank_subset, decode_every_rank=every_rank
            ):
                batch, world = self._share(
                    rank_subset_collectives=rank_subset, decode_every_rank=every_rank
                )
                world.broadcast.assert_not_called()
                self.assertEqual(batch.latents.sum().item(), 6)


if __name__ == "__main__":
    unittest.main()
