"""TPU CI entry for the all-gather ring attention tests: the dispatch and
reference logic, and the ring with the Pallas kernel on device."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers import usp
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.server_args.server_args import (
    TPU_RING_CAPABLE_ATTENTION_BACKENDS,
)
from sglang.multimodal_gen.test.unit.test_usp_allgather_ring import (  # noqa: F401
    TestAllGatherRingAttention,
)
from sglang.multimodal_gen.test.unit.test_usp_ring_tail_pad import (  # noqa: F401
    TestRingTailPadDispatch,
)
from sglang_tpu.multimodal_gen import tpu_splash_attn
from sglang_tpu.multimodal_gen.tpu_splash_attn import (
    TpuSplashAttentionBackend,
    TpuSplashAttentionImpl,
)


class TestTpuSplashRingCapability(unittest.TestCase):
    def test_declares_ring_support(self):
        self.assertTrue(TpuSplashAttentionBackend.supports_ring_rotation())
        self.assertEqual(
            TPU_RING_CAPABLE_ATTENTION_BACKENDS,
            (TpuSplashAttentionBackend.get_enum().name.lower(),),
        )

    def test_only_the_tpu_impl_resumes_in_kernel(self):
        self.assertTrue(TpuSplashAttentionImpl.supports_ring_resume())
        self.assertFalse(SDPAImpl.supports_ring_resume())
        self.assertFalse(FlashAttentionBackend.get_impl_cls().supports_ring_resume())

    def test_auto_selected_ring_backend_serves_sdpa_layers(self):
        # Without --attention-backend a TPU ring launch selects tpu_splash_attn
        # implicitly; a DiT layer that lists TORCH_SDPA must still take it.
        from sglang.multimodal_gen.configs.pipeline_configs.base import (
            PipelineConfig,
        )
        from sglang.multimodal_gen.runtime.layers.attention import selector
        from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
        from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform
        from sglang.multimodal_gen.runtime.server_args import server_args
        from sglang.multimodal_gen.runtime.server_args.server_args import ServerArgs

        with patch.object(PipelineConfig, "from_kwargs", return_value=PipelineConfig()):
            args = ServerArgs.from_dict({"model_path": "/fake"})
        args.num_gpus, args.ring_degree = 8, 2
        platforms = "sglang.multimodal_gen.runtime.platforms"
        with patch(f"{platforms}.current_platform.is_tpu", return_value=True):
            args._adjust_attention_backend()
        self.assertEqual(args.attention_backend, "tpu_splash_attn")
        self.assertFalse(args.is_arg_explicitly_set("attention_backend"))

        selector._cached_get_attn_backend.cache_clear()
        try:
            with (
                patch(f"{platforms}.current_platform", TpuPlatform()),
                patch.object(server_args, "_global_server_args", args),
            ):
                backend = selector.get_attn_backend(
                    128,
                    torch.bfloat16,
                    supported_attention_backends=set(
                        BaseDiT._supported_attention_backends
                    ),
                )
        finally:
            selector._cached_get_attn_backend.cache_clear()
        self.assertIs(backend, TpuSplashAttentionBackend)


def _skip_if_no_tpu() -> bool:
    try:
        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


def _float32_attention(q, k, v):
    """float32 attention on CPU; [B, S, H, D] in and out."""
    q, k, v = (t.float().transpose(1, 2) for t in (q, k, v))
    scores = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
    return (torch.softmax(scores, dim=-1) @ v).transpose(1, 2)


def _rel_l2(got, want):
    return ((got.float() - want).norm() / want.norm()).item()


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestAllGatherRingOnDevice(unittest.TestCase):
    """usp.allgather_kv_ring_attn with the Pallas impl on device, where the
    stacked gather layout, the device kv_info and the float32 carry meet the
    kernel. Ranks run one after another; the two collectives return what they
    would on a ring of that size."""

    # bf16 inputs; an all-gather ring matches unsharded attention to bf16.
    REL_L2_BOUND = 1e-2

    def setUp(self):
        if not tpu_splash_attn._check_splash_available():
            self.skipTest("Pallas kernels unavailable (needs TPU v7x and jax)")
        tpu_splash_attn._register_ops()
        self.impl = TpuSplashAttentionImpl(
            num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5
        )

    def _ring(self, world_size, q, k, v, kv_real_len):
        """Each rank's [B, S_shard, H, D] output."""
        shards = [t.chunk(world_size, dim=1) for t in (q, k, v)]
        # all_gather_tensor's result on every rank: the heads-major K (then
        # V) shards stacked along dim 0.
        stacks = [
            torch.cat([s.transpose(1, 2).contiguous() for s in t_shards]).to("tpu")
            for t_shards in shards[1:]
        ]
        outputs = []
        for rank in range(world_size):
            gathered = iter(stacks)

            def all_gather_tensor(t, gather_dim, group, gathered=gathered):
                return next(gathered)

            def all_to_all_single(
                t, output_split_sizes, input_split_sizes, group, rank=rank
            ):
                # Rank r receives element r of every rank's [0, .., R-1].
                return torch.full_like(t, float(rank))

            with (
                patch.object(
                    usp,
                    "get_sp_group",
                    lambda: SimpleNamespace(ring_group=object()),
                ),
                patch.object(usp.dist, "get_world_size", lambda group=None: world_size),
                patch.object(usp.ft_c, "all_gather_tensor", all_gather_tensor),
                patch.object(usp.ft_c, "all_to_all_single", all_to_all_single),
                patch.object(
                    tpu_splash_attn,
                    "_FLASH_RING_OP",
                    wraps=tpu_splash_attn._FLASH_RING_OP,
                ) as ring,
            ):
                out = usp.allgather_kv_ring_attn(
                    *(t_shards[rank].to("tpu") for t_shards in shards),
                    attn_impl=self.impl,
                    kv_real_len=kv_real_len,
                )
            # The local shard, then one resumed step per peer, all on the
            # Pallas ring kernel.
            self.assertEqual(ring.call_count, world_size, f"rank {rank}")
            outputs.append(out.to("cpu"))
        return outputs

    def _check(self, world_size, shard_len, kv_real_len=None):
        gen = torch.Generator().manual_seed(world_size * shard_len)
        shape = (1, world_size * shard_len, 2, 128)
        q, k, v = (torch.randn(shape, generator=gen).bfloat16() for _ in range(3))
        real = kv_real_len or world_size * shard_len
        # Keys past the real count are padding and may hold anything.
        k[:, real:], v[:, real:] = float("nan"), float("nan")

        out = torch.cat(self._ring(world_size, q, k, v, kv_real_len), dim=1)
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(out.isnan().any().item())
        # Query rows past the real count are padding: zeroed.
        self.assertTrue(torch.all(out[:, real:] == 0).item())
        out = out[:, :real]
        want = _float32_attention(q, k[:, :real], v[:, :real])[:, :real]
        # The same impl over the whole sequence, as without ring parallelism.
        unsharded = self.impl.forward(
            q.to("tpu"), k[:, :real].to("tpu"), v[:, :real].to("tpu"), None
        ).to("cpu")[:, :real]
        for name, reference in (
            ("float32 attention", want),
            ("unsharded impl", unsharded.float()),
        ):
            rel = _rel_l2(out, reference)
            print(f"{self._testMethodName} R={world_size}: rel-L2 vs {name} {rel:.3e}")
            self.assertLess(rel, self.REL_L2_BOUND, name)

    def test_ring_matches_attention_over_all_keys(self):
        for world_size in (2, 4):
            with self.subTest(world_size=world_size):
                self._check(world_size, shard_len=1024)

    def test_tail_pad_leaves_fewer_than_128_keys_in_the_last_shard(self):
        for world_size in (2, 4):
            with self.subTest(world_size=world_size):
                # The last shard holds 100 real keys and 924 padding keys.
                self._check(
                    world_size,
                    shard_len=1024,
                    kv_real_len=(world_size - 1) * 1024 + 100,
                )


if __name__ == "__main__":
    unittest.main()
