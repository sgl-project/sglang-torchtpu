"""The fused DFlash draft KV write agrees with the per-layer reference.

dflash_write_draft_kv folds neox rotation, K/V head interleave, padding, the
store-dtype cast and an L-layer scatter into one span-addressed Pallas launch.
dflash_write_draft_kv_ref reaches the same buffer state the long way, per layer,
so a divergence localizes to the fused write.

Both extend geometries are covered because they decompose into different spans:
page-aligned (whole pages plus a trailing empty span) and resuming mid-page
(short first span). Verify is covered separately -- there the two paths
deliberately disagree on the masked slot.

Needs a TPU with Pallas; skipped otherwise. Set DFLASH_KV_WRITE_BENCH=1 to also
time the two paths:

    DFLASH_KV_WRITE_BENCH=1 python3 -m pytest -s \
        test/registered/tpu/unit/test_dflash_draft_kv_write.py
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import os
import time
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from sglang.test.test_utils import CustomTestCase

# Geometry of /models/z-lab/gpt-oss-120b-DFlash served at TP8: 8 draft layers,
# hidden 2880, head_dim 64, and 8 KV heads sharded to 1 per rank.
DEVICE = "tpu"
LAYERS = 8
KV_HEADS = 1
HEAD_DIM = 64
HIDDEN = 2880
BLOCK_SIZE = 10
# The reference's index_copy_ lowers to a dynamic-update-slice over the whole
# buffer, so its cost tracks pool size rather than token count -- benching a
# small pool understates the fused speedup badly. Production is 262144 slots
# over page_size 128 (~1 GB per rank across 8 layers); override both to compare.
NUM_PAGES = int(os.environ.get("DFLASH_KV_WRITE_PAGES", "2048"))
PAGE_SIZE = int(os.environ.get("DFLASH_KV_WRITE_PAGE_SIZE", "128"))
# Bounds the extend span table, so it sets the fused kernel's grid. Production
# passes req_to_token_pool.size.
MAX_NUM_REQS = int(os.environ.get("DFLASH_KV_WRITE_MAX_REQS", "256"))
DTYPE = torch.bfloat16
# gpt-oss DFlash draft rope. yarn matters: it folds mscale into cos_sin_cache,
# so the gather-based fused path stays correct only because the scaling never
# appears outside the cache.
ROPE_THETA = 150000
MAX_POSITION = 131072
ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 32.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "original_max_position_embeddings": 4096,
    "truncate": False,
}
BENCH = bool(os.environ.get("DFLASH_KV_WRITE_BENCH"))


def _skip_if_no_tpu() -> bool:
    return importlib.util.find_spec("torch_tpu") is None or not hasattr(torch, "tpu")


class _StubAttention(nn.Module):
    """The slice of DFlashAttention that the draft KV write touches.

    Building a real DFlashAttention needs a distributed init for its parallel
    linears; the write path only calls kv_proj_only / apply_k_norm /
    apply_k_rope and reads num_kv_heads / head_dim / rotary_emb, so a stub with
    real weights and the real rope exercises the same math.
    """

    def __init__(self, rotary_emb: nn.Module) -> None:
        super().__init__()
        self.num_kv_heads = KV_HEADS
        self.head_dim = HEAD_DIM
        self.kv_size = KV_HEADS * HEAD_DIM
        self.rotary_emb = rotary_emb
        self.kv_weight = nn.Parameter(
            torch.randn(2 * self.kv_size, HIDDEN, dtype=DTYPE) / HIDDEN**0.5
        )
        self.k_norm_weight = nn.Parameter(torch.rand(HEAD_DIM, dtype=DTYPE) + 0.5)
        self.eps = 1e-6

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv = F.linear(hidden_states, self.kv_weight)
        return kv.split([self.kv_size, self.kv_size], dim=-1)

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        by_head = k.reshape(-1, self.head_dim).float()
        normed = by_head * torch.rsqrt(by_head.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(k.dtype) * self.k_norm_weight).view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        _, k = self.rotary_emb(positions, k.new_empty(k.shape), k)
        return k


class _StubDraftModel(nn.Module):
    """A DFlash draft model reduced to what the KV write reads."""

    def __init__(self, rotary_emb: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Module()
            for _ in range(LAYERS)  # containers for self_attn
        )
        for layer in self.layers:
            layer.self_attn = _StubAttention(rotary_emb)

    def prepare_context_hidden_for_kv(
        self, layer: nn.Module, ctx_hidden: torch.Tensor
    ) -> torch.Tensor:
        return ctx_hidden


def _extend_cache_loc(*, head_offset: int, num_tokens: int) -> torch.Tensor:
    """Slots an allocator hands out: consecutive within a page, pages scattered.

    Args:
      head_offset: Offset into the first page, so the write resumes mid-page.
      num_tokens: Total slots to hand out.

    Returns:
      Flat slot index per token [num_tokens].
    """
    slots: list[int] = []
    offset, page = head_offset, 0
    while len(slots) < num_tokens:
        take = min(PAGE_SIZE - offset, num_tokens - len(slots))
        # 37 is coprime with NUM_PAGES, so this walks a permutation of the
        # pages -- consecutive tokens stay in one page, pages do not.
        base = ((page * 37 + 5) % NUM_PAGES) * PAGE_SIZE + offset
        slots.extend(range(base, base + take))
        offset, page = 0, page + 1
    return torch.tensor(slots, dtype=torch.int64, device=DEVICE)


def _verify_inputs(bs: int) -> dict[str, torch.Tensor]:
    """One decode step's draft KV write: bs requests of BLOCK_SIZE draft slots.

    Accept counts vary across the batch, including a request that accepted
    nothing, which is what makes the two paths differ on slot 0.

    Args:
      bs: Physical batch size.

    Returns:
      The tensor kwargs of a verify-path write.
    """
    offsets = torch.arange(BLOCK_SIZE)
    starts = torch.tensor(
        [(i * 293 + 11) % (NUM_PAGES * PAGE_SIZE // 2) for i in range(bs)]
    )
    commits = torch.tensor([i % (BLOCK_SIZE + 1) for i in range(bs)], dtype=torch.int32)
    loc2d = starts[:, None] + offsets[None, :]
    valid = offsets[None, :] < commits[:, None]
    num_tokens = bs * BLOCK_SIZE
    return dict(
        ctx_positions=torch.arange(num_tokens, dtype=torch.int64, device=DEVICE),
        ctx_cache_loc=torch.where(valid, loc2d, torch.zeros_like(loc2d))
        .reshape(-1)
        .to(DEVICE),
        cache_loc_2d=loc2d.to(DEVICE),
        commit_lens=commits.to(DEVICE),
    )


def _shape_key(**kwargs: torch.Tensor | None) -> tuple[object, ...]:
    """Key direct-compile artifacts on input shapes, mirroring production's."""
    return tuple(
        (name, None if t is None else (tuple(t.shape), t.dtype))
        for name, t in sorted(kwargs.items())
    )


def _compiled(fn, *, draft_model: nn.Module, kv_pool: object, **static_args: int):
    """Compile a write path the way DFlashSpecDecodeCompilePolicyMixin does.

    draft_model, kv_pool and the static ints are bound into a closure rather
    than passed at call time: direct_compile lifts the model's tensors itself,
    a module or pool cannot be a graph input, and _shape_key only keys tensors.

    Args:
      fn: dflash_write_draft_kv or dflash_write_draft_kv_ref.
      draft_model: Draft model whose params and buffers get lifted.
      kv_pool: Fused TPU KV pool; the one mutable input, copied back per call.
      **static_args: Non-tensor kwargs of fn, e.g. max_num_reqs on the fused path.

    Returns:
      The compiled callable, taking fn's remaining tensor kwargs.
    """
    from sglang_tpu.compilation.direct_compile import (
        direct_compile,
    )

    def bound(**kwargs: torch.Tensor | None) -> None:
        fn(draft_model=draft_model, kv_pool=kv_pool, **static_args, **kwargs)

    return direct_compile(
        torch.no_grad()(bound),
        specialization_key=_shape_key,
        model=draft_model,
        kv_pool=kv_pool,
    )


@unittest.skipIf(_skip_if_no_tpu(), "requires torch_tpu")
class TestDflashDraftKvWrite(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.rotary_embedding import get_rope
        from sglang.srt.runtime_context import publish
        from sglang.srt.server_args import ServerArgs

        # get_rope reads the exec config bag, which fails closed until published.
        publish(ServerArgs(model_path="dummy"), role="test")
        torch.manual_seed(0)
        rope = get_rope(
            HEAD_DIM,
            rotary_dim=HEAD_DIM,
            max_position=MAX_POSITION,
            base=ROPE_THETA,
            rope_scaling=ROPE_SCALING,
            is_neox_style=True,
            dtype=DTYPE,
        )
        rope.cos_sin_cache = rope.cos_sin_cache.to(DEVICE)
        cls.draft_model = _StubDraftModel(rope).to(DEVICE)

    def _make_pool(self):
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

        return TpuFusedMHATokenToKVPool(
            size=NUM_PAGES * PAGE_SIZE,
            page_size=PAGE_SIZE,
            dtype=DTYPE,
            head_num=KV_HEADS,
            head_dim=HEAD_DIM,
            layer_num=LAYERS,
            device=DEVICE,
            enable_memory_saver=False,
        )

    def _run_both(self, *, cache_loc, positions, cache_loc_2d=None, commit_lens=None):
        """Write the same inputs through the fused path and the reference.

        Returns:
          Tuple of (fused buffers, reference buffers), each a list of per-layer
          CPU float32 copies.
        """
        from sglang_tpu.dflash_draft_kv_write import (
            dflash_write_draft_kv,
            dflash_write_draft_kv_ref,
        )

        num_tokens = int(cache_loc.shape[0])
        ctx_hidden = torch.randn((num_tokens, HIDDEN), dtype=DTYPE, device=DEVICE)
        fused_pool, ref_pool = self._make_pool(), self._make_pool()

        _compiled(
            dflash_write_draft_kv,
            draft_model=self.draft_model,
            kv_pool=fused_pool,
            max_num_reqs=MAX_NUM_REQS,
        )(
            ctx_hidden=ctx_hidden,
            ctx_positions=positions,
            ctx_cache_loc=cache_loc,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
        )
        _compiled(
            dflash_write_draft_kv_ref, draft_model=self.draft_model, kv_pool=ref_pool
        )(
            ctx_hidden=ctx_hidden,
            ctx_positions=positions,
            ctx_cache_loc=cache_loc,
        )
        torch.accelerator.synchronize()
        to_cpu = lambda pool: [  # noqa: E731
            pool.get_fused_kv_buffer(i).to("cpu", torch.float32) for i in range(LAYERS)
        ]
        return to_cpu(fused_pool), to_cpu(ref_pool)

    def _assert_layers_match(self, fused, ref, *, ignore_slot_0=False):
        for layer in range(LAYERS):
            got, want = fused[layer], ref[layer]
            if ignore_slot_0:
                got, want = got.clone(), want.clone()
                got[0, 0], want[0, 0] = 0, 0
            torch.testing.assert_close(got, want, atol=3e-2, rtol=3e-2, msg=f"L{layer}")

    def test_extend_page_aligned(self):
        num_tokens = 2 * PAGE_SIZE
        cache_loc = _extend_cache_loc(head_offset=0, num_tokens=num_tokens)
        positions = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
        self._assert_layers_match(
            *self._run_both(cache_loc=cache_loc, positions=positions)
        )

    def test_extend_resumes_mid_page(self):
        num_tokens = 300
        cache_loc = _extend_cache_loc(head_offset=91, num_tokens=num_tokens)
        # Start past the yarn original_max_position_embeddings so the cache rows
        # come from the interpolated regime, not the identity one.
        positions = torch.arange(
            4000, 4000 + num_tokens, dtype=torch.int64, device=DEVICE
        )
        self._assert_layers_match(
            *self._run_both(cache_loc=cache_loc, positions=positions)
        )

    def test_verify_leaves_uncommitted_slots_untouched(self):
        """The fused path skips uncommitted tokens; the reference dumps them on slot 0.

        This is the one place the two deliberately differ, and it is the whole
        point of the span table on the verify path: a request that accepted
        nothing must not write, and slot 0 must not be clobbered.
        """
        block, starts, commits = 10, [1000, 1500, 2100, 300], [10, 0, 3, 7]
        offsets = torch.arange(block)
        loc2d = torch.tensor(starts)[:, None] + offsets[None, :]
        valid = offsets[None, :] < torch.tensor(commits)[:, None]
        cache_loc = (
            torch.where(valid, loc2d, torch.zeros_like(loc2d)).reshape(-1).to(DEVICE)
        )
        num_tokens = len(starts) * block
        positions = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)

        fused, ref = self._run_both(
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=loc2d.to(DEVICE),
            commit_lens=torch.tensor(commits, dtype=torch.int32, device=DEVICE),
        )
        self._assert_layers_match(fused, ref, ignore_slot_0=True)
        for layer in range(LAYERS):
            self.assertEqual(
                float(fused[layer][0, 0].abs().sum()),
                0.0,
                f"L{layer}: fused write touched slot 0 for an uncommitted token",
            )

    @unittest.skipUnless(BENCH, "set DFLASH_KV_WRITE_BENCH=1 to time both paths")
    def test_bench_fused_vs_reference(self):
        """Time both paths under direct_compile, across extend and decode shapes.

        Eager timings are meaningless here -- they measure dispatch, not the
        write -- so both paths go through the same direct_compile wrapper
        production uses. One pool per path is reused across shapes; the
        specialization key gives each shape its own artifact.
        """
        from sglang_tpu.dflash_draft_kv_write import (
            dflash_write_draft_kv,
            dflash_write_draft_kv_ref,
        )

        fused = _compiled(
            dflash_write_draft_kv,
            draft_model=self.draft_model,
            kv_pool=self._make_pool(),
            max_num_reqs=MAX_NUM_REQS,
        )
        ref = _compiled(
            dflash_write_draft_kv_ref,
            draft_model=self.draft_model,
            kv_pool=self._make_pool(),
        )

        cases = []
        for num_tokens in (512, 1024, 2048, 4096, 8192):
            cache_loc = _extend_cache_loc(head_offset=0, num_tokens=num_tokens)
            cases.append(
                (
                    f"extend {num_tokens}",
                    num_tokens,
                    dict(
                        ctx_positions=torch.arange(
                            num_tokens, dtype=torch.int64, device=DEVICE
                        ),
                        ctx_cache_loc=cache_loc,
                        cache_loc_2d=None,
                        commit_lens=None,
                    ),
                )
            )
        for bs in (1, 8, 32, 64):
            cases.append((f"decode bs={bs}", bs * BLOCK_SIZE, _verify_inputs(bs)))

        slots = NUM_PAGES * PAGE_SIZE
        buf = self._make_pool().get_fused_kv_buffer(0)
        mb = buf.numel() * buf.element_size() * LAYERS / 2**20
        print(
            f"\n== draft KV write, {LAYERS} layers, hidden {HIDDEN}, page "
            f"{PAGE_SIZE} x {NUM_PAGES} pages = {slots} slots, {mb:.0f} MB pool, "
            f"max_num_reqs {MAX_NUM_REQS} =="
        )
        print(f"{'case':>14} {'tokens':>7} {'fused':>9} {'ref':>9} {'speedup':>8}")
        for name, num_tokens, kwargs in cases:
            ctx_hidden = torch.randn((num_tokens, HIDDEN), dtype=DTYPE, device=DEVICE)
            ref_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in ("cache_loc_2d", "commit_lens")
            }
            timings = {}
            for label, run in (
                ("fused", lambda: fused(ctx_hidden=ctx_hidden, **kwargs)),
                ("ref", lambda: ref(ctx_hidden=ctx_hidden, **ref_kwargs)),
            ):
                for _ in range(4):  # compile the artifact, then warm it
                    run()
                torch.accelerator.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    run()
                torch.accelerator.synchronize()
                timings[label] = (time.perf_counter() - start) / 20 * 1e3
            print(
                f"{name:>14} {num_tokens:>7} {timings['fused']:>8.3f}m"
                f" {timings['ref']:>8.3f}m {timings['ref'] / timings['fused']:>7.2f}x"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
