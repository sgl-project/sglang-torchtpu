import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy


class TestCFGPolicyCombine(unittest.TestCase):
    def test_cfg_parallel_uses_parallel_arithmetic_order(self):
        policy = CFGPolicy()
        req = MagicMock()
        req.cfg_normalization = 0
        req.guidance_rescale = 0

        pipeline_config = MagicMock()
        pipeline_config.postprocess_cfg_noise.side_effect = lambda _, noise, __: noise

        pos = torch.tensor([1.0], dtype=torch.bfloat16)
        neg = torch.tensor([0.1], dtype=torch.bfloat16)

        serial = policy.combine([pos, neg], req, 7.0, pipeline_config)
        parallel = policy.combine(
            [pos, neg], req, 7.0, pipeline_config, cfg_parallel=True
        )

        self.assertTrue(torch.equal(serial, neg + 7.0 * (pos - neg)))
        self.assertTrue(torch.equal(parallel, 7.0 * pos + (1 - 7.0) * neg))
        self.assertFalse(torch.equal(serial, parallel))


class _ThreadSpGroup:
    """The SP group across one thread per rank: all_reduce sums the ranks'
    tensors."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size, timeout=60)
        self.inputs = [None] * world_size
        self.local = threading.local()

    @property
    def rank_in_group(self):
        return self.local.rank

    def all_reduce(self, x):
        self.inputs[self.local.rank] = x
        self.barrier.wait()
        total = torch.stack(self.inputs).sum(0)
        self.barrier.wait()
        return total


def _no_host_read(tensor):
    raise AssertionError("a tensor was read on the host")


class TestCfgNormalization(unittest.TestCase):
    """Z-Image's CFG normalization (on by default) caps the norm of the
    guided prediction at the conditional one's, over the whole latents."""

    def _predictions(self, shape):
        generator = torch.Generator().manual_seed(0)
        pos = torch.randn(shape, generator=generator)
        # Rows of growing magnitude give each SP shard its own norm ratio.
        ramp = torch.linspace(0.5, 4.0, shape[3]).view(1, 1, 1, -1, 1)
        neg = torch.randn(shape, generator=generator) * ramp
        return pos, neg

    def test_sp_shards_scale_by_the_whole_latents_norms(self):
        world_size, cfg_scale = 3, 5.0
        config = ZImagePipelineConfig()
        config.vae_config.post_init()
        # 1216x832: 76 token rows split 26/25/25.
        pos, neg = self._predictions((1, 16, 1, 152, 104))

        def request():
            return SimpleNamespace(
                height=1216,
                width=832,
                raw_latent_shape=None,
                did_sp_shard_latents=True,
                cfg_normalization=1.0,
                guidance_rescale=0.0,
            )

        group = _ThreadSpGroup(world_size)
        requests = [request() for _ in range(world_size)]
        outputs, errors = [None] * world_size, []

        def run(rank):
            group.local.rank = rank
            try:
                batch = requests[rank]
                pos_shard, _ = config.shard_latents_for_sp(batch, pos)
                neg_shard, _ = config.shard_latents_for_sp(batch, neg)
                outputs[rank] = CFGPolicy().combine(
                    [pos_shard, neg_shard], batch, cfg_scale, config
                )
            except Exception as error:  # surfaced below
                errors.append(error)
                group.barrier.abort()

        with patch.object(parallel_state, "_SP", group):
            workers = [
                threading.Thread(target=run, args=(r,)) for r in range(world_size)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
        if errors:
            raise errors[0]

        plan = requests[0]._zimage_sp_plan
        self.assertEqual(plan["shard_axis"], "h")
        self.assertEqual(plan["shard_sizes_tok"], [26, 25, 25])
        one_rank = request()
        one_rank.did_sp_shard_latents = False
        expected = CFGPolicy().combine([pos, neg], one_rank, cfg_scale, config)
        # The normalization scales this prediction.
        self.assertFalse(torch.equal(expected, neg + cfg_scale * (pos - neg)))
        torch.testing.assert_close(torch.cat(outputs, dim=3), expected)

    def test_one_rank_matches_the_reference_without_reading_the_norms(self):
        config = ZImagePipelineConfig()
        pos, neg = self._predictions((1, 16, 1, 8, 8))
        guided = neg + 5.0 * (pos - neg)
        ori_norm = torch.linalg.vector_norm(pos)
        new_norm = torch.linalg.vector_norm(guided)
        # Capped at the conditional norm, and left as is under a high cap.
        for cfg_normalization in (1.0, 100.0):
            with self.subTest(cfg_normalization=cfg_normalization):
                max_norm = ori_norm * cfg_normalization
                if new_norm > max_norm:
                    expected = guided * (max_norm / new_norm)
                else:
                    expected = guided
                req = SimpleNamespace(
                    cfg_normalization=cfg_normalization, guidance_rescale=0.0
                )
                with patch.object(torch.Tensor, "__bool__", _no_host_read):
                    out = CFGPolicy().combine([pos, neg], req, 5.0, config)
                self.assertTrue(torch.equal(out, expected))


if __name__ == "__main__":
    unittest.main()
