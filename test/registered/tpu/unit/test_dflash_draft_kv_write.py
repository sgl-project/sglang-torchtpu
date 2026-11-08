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

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)

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

    def __init__(
        self,
        rotary_emb: nn.Module,
        *,
        kv_heads: int = KV_HEADS,
        head_dim: int = HEAD_DIM,
        hidden: int = HIDDEN,
    ) -> None:
        super().__init__()
        self.num_kv_heads = kv_heads
        self.head_dim = head_dim
        self.kv_size = kv_heads * head_dim
        self.rotary_emb = rotary_emb
        self.kv_weight = nn.Parameter(
            torch.randn(2 * self.kv_size, hidden, dtype=DTYPE) / hidden**0.5
        )
        self.k_norm_weight = nn.Parameter(torch.rand(head_dim, dtype=DTYPE) + 0.5)
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

    def __init__(
        self, rotary_emb: nn.Module, *, num_layers: int = LAYERS, **attn_kwargs: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Module()
            for _ in range(num_layers)  # containers for self_attn
        )
        for layer in self.layers:
            layer.self_attn = _StubAttention(rotary_emb, **attn_kwargs)

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


def _new_pool(*, head_num: int, head_dim: int, layer_num: int, v_head_dim=None):
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

    return TpuFusedMHATokenToKVPool(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=DTYPE,
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        layer_num=layer_num,
        device=DEVICE,
        enable_memory_saver=False,
    )


def _write_both(
    *,
    draft_model: nn.Module,
    ref_fn,
    make_pool,
    hidden: int,
    cache_loc: torch.Tensor,
    positions: torch.Tensor,
    cache_loc_2d: torch.Tensor | None = None,
    commit_lens: torch.Tensor | None = None,
):
    """Write the same inputs through the fused path and ref_fn.

    Args:
      draft_model: Stub draft model both paths read.
      ref_fn: Per-layer reference; takes only the flat per-token slots.
      make_pool: Builds an empty pool for draft_model.
      hidden: Context hidden size.

    Returns:
      Tuple of (fused buffers, reference buffers), each a list of per-layer
      CPU float32 copies.
    """
    from sglang_tpu.dflash_draft_kv_write import dflash_write_draft_kv

    ctx_hidden = torch.randn(
        (int(cache_loc.shape[0]), hidden), dtype=DTYPE, device=DEVICE
    )
    fused_pool, ref_pool = make_pool(), make_pool()
    _compiled(
        dflash_write_draft_kv,
        draft_model=draft_model,
        kv_pool=fused_pool,
        max_num_reqs=MAX_NUM_REQS,
    )(
        ctx_hidden=ctx_hidden,
        ctx_positions=positions,
        ctx_cache_loc=cache_loc,
        cache_loc_2d=cache_loc_2d,
        commit_lens=commit_lens,
    )
    _compiled(ref_fn, draft_model=draft_model, kv_pool=ref_pool)(
        ctx_hidden=ctx_hidden, ctx_positions=positions, ctx_cache_loc=cache_loc
    )
    torch.accelerator.synchronize()
    to_cpu = lambda pool: [  # noqa: E731
        pool.get_fused_kv_buffer(i).to("cpu", torch.float32)
        for i in range(len(draft_model.layers))
    ]
    return to_cpu(fused_pool), to_cpu(ref_pool)


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
        return _new_pool(head_num=KV_HEADS, head_dim=HEAD_DIM, layer_num=LAYERS)

    def _run_both(self, **kwargs):
        from sglang_tpu.dflash_draft_kv_write import dflash_write_draft_kv_ref

        return _write_both(
            draft_model=self.draft_model,
            ref_fn=dflash_write_draft_kv_ref,
            make_pool=self._make_pool,
            hidden=HIDDEN,
            **kwargs,
        )

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


# Kimi-K3.1 DFlash draft at TP32: MLA draft layers served as MHA over the
# expanded latent (sglang_tpu.models.dflash_mla), 96 heads -> 3 per rank,
# K = [k_nope 128 | k_pe 64], V 128, deepseek yarn rope on the 64 rope dims.
# Two layers and a small hidden: only the per-rank head geometry shapes the write.
MLA_LAYERS = 2
MLA_HEADS = 3
MLA_NOPE, MLA_ROPE, MLA_V = 128, 64, 128
MLA_HIDDEN = 256
MLA_ROPE_SCALING = {
    "rope_type": "deepseek_yarn",
    "factor": 32.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 32768,
}


def _stub_mla_attention(rotary_emb: nn.Module) -> nn.Module:
    """A DFlashMLAAttention with plain linear K/V projections (no parallel linears)."""
    from sglang_tpu.models.dflash_mla import DFlashMLAAttention

    class _StubMlaAttention(DFlashMLAAttention):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.num_heads = self.num_kv_heads = MLA_HEADS
            self.qk_nope_head_dim, self.qk_rope_head_dim = MLA_NOPE, MLA_ROPE
            self.head_dim, self.v_head_dim = MLA_NOPE + MLA_ROPE, MLA_V
            self.rotary_emb = rotary_emb
            self.kv_weight = nn.Parameter(
                torch.randn(MLA_HEADS * (MLA_NOPE + MLA_V), MLA_HIDDEN, dtype=DTYPE)
                / MLA_HIDDEN**0.5
            )
            self.pe_weight = nn.Parameter(
                torch.randn(MLA_ROPE, MLA_HIDDEN, dtype=DTYPE) / MLA_HIDDEN**0.5
            )

        def _project_kv(self, hidden_states: torch.Tensor):
            kv = F.linear(hidden_states, self.kv_weight).view(
                -1, MLA_HEADS, MLA_NOPE + MLA_V
            )
            k_nope, v = kv.split([MLA_NOPE, MLA_V], dim=-1)
            return k_nope, F.linear(hidden_states, self.pe_weight).unsqueeze(1), v

    return _StubMlaAttention()


def _mla_write_ref(*, draft_model, kv_pool, ctx_hidden, ctx_positions, ctx_cache_loc):
    """Per-layer MLA reference: rotate k_pe in the projection, scatter K/V per layer."""
    from sglang_tpu.dflash_draft_kv_write import _scatter_layer_kv

    for layer_idx, layer in enumerate(draft_model.layers):
        attn = layer.self_attn
        k, v = attn.project_context_kv_rotated(ctx_positions, ctx_hidden)
        _scatter_layer_kv(
            kv_pool=kv_pool, layer_idx=layer_idx, loc=ctx_cache_loc, k=k, v=v
        )


@unittest.skipIf(_skip_if_no_tpu(), "requires torch_tpu")
class TestDflashMlaDraftKvWrite(CustomTestCase):
    """The fused write of an MLA draft (pre-rotated K, identity rotation in the
    kernel, V padded to K's width) matches the per-layer reference."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.rotary_embedding import get_rope
        from sglang.srt.runtime_context import publish
        from sglang.srt.server_args import ServerArgs

        publish(ServerArgs(model_path="dummy"), role="test")
        torch.manual_seed(1)
        rope = get_rope(
            MLA_ROPE,
            rotary_dim=MLA_ROPE,
            max_position=MAX_POSITION,
            base=50000,
            rope_scaling=MLA_ROPE_SCALING,
            is_neox_style=False,
            dtype=DTYPE,
        )
        rope.cos_sin_cache = rope.cos_sin_cache.to(DEVICE)
        model = nn.Module()
        model.layers = nn.ModuleList(nn.Module() for _ in range(MLA_LAYERS))
        for layer in model.layers:
            layer.self_attn = _stub_mla_attention(rope)
        model.prepare_context_hidden_for_kv = lambda layer, ctx_hidden: ctx_hidden
        cls.draft_model = model.to(DEVICE)

    def test_extend_resumes_mid_page(self):
        num_tokens = 300
        fused, ref = _write_both(
            draft_model=self.draft_model,
            ref_fn=_mla_write_ref,
            make_pool=lambda: _new_pool(
                head_num=MLA_HEADS,
                head_dim=MLA_NOPE + MLA_ROPE,
                v_head_dim=MLA_V,
                layer_num=MLA_LAYERS,
            ),
            hidden=MLA_HIDDEN,
            cache_loc=_extend_cache_loc(head_offset=91, num_tokens=num_tokens),
            # Past original_max_position_embeddings: k_pe comes from the
            # interpolated yarn regime.
            positions=torch.arange(
                40000, 40000 + num_tokens, dtype=torch.int64, device=DEVICE
            ),
        )
        for layer in range(MLA_LAYERS):
            # Same K/V values and the same bf16 cast: the identity rotation is exact.
            torch.testing.assert_close(
                fused[layer], ref[layer], atol=0, rtol=0, msg=f"L{layer}"
            )


# The windowed Muse Spark DFlash2 draft: 5 layers, head_dim 128, page 256, 1 KV
# head per rank on the TP8 decode and all 8 on the TP1 (PP8) prefill.
SAL_LAYERS = 5
SAL_HEAD_DIM = 128
SAL_HIDDEN = 512
SAL_PAGE = 256
SAL_PAGES = 64


def _sal_slots(*, head_offset: int, num_tokens: int, pad_rows: int) -> torch.Tensor:
    """Allocator-like slots over SAL_PAGE pages, then pad_rows slot-0 padding rows."""
    slots: list[int] = []
    offset, page = head_offset, 0
    while len(slots) < num_tokens:
        take = min(SAL_PAGE - offset, num_tokens - len(slots))
        # Page 0 is the dummy page; 13 is coprime with SAL_PAGES - 1.
        base = (1 + (page * 13 + 3) % (SAL_PAGES - 1)) * SAL_PAGE + offset
        slots.extend(range(base, base + take))
        offset, page = 0, page + 1
    return torch.tensor(slots + [0] * pad_rows, dtype=torch.int64, device=DEVICE)


def _sal_verify_slots(starts: list[int], commits: list[int], block: int):
    """Verify slots [bs, block] (a block may cross a page) and commit counts."""
    loc2d = torch.tensor(starts)[:, None] + torch.arange(block)[None, :]
    return loc2d.to(DEVICE), torch.tensor(commits, dtype=torch.int32, device=DEVICE)


def _new_sal_pool(*, kv_heads: int, dtype: torch.dtype):
    from sglang_tpu.kv_pool import TpuSeqAlongLaneKVPool

    return TpuSeqAlongLaneKVPool(
        size=SAL_PAGES * SAL_PAGE,
        page_size=SAL_PAGE,
        dtype=dtype,
        head_num=kv_heads,
        head_dim=SAL_HEAD_DIM,
        layer_num=SAL_LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
    )


def _new_fused_pool(*, kv_heads: int, dtype: torch.dtype):
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

    return TpuFusedMHATokenToKVPool(
        size=SAL_PAGES * SAL_PAGE,
        page_size=SAL_PAGE,
        dtype=dtype,
        head_num=kv_heads,
        head_dim=SAL_HEAD_DIM,
        layer_num=SAL_LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
    )


def _sal_logical(buf: torch.Tensor, heads_x2: int) -> torch.Tensor:
    """[pages, 2 * kv, hd_words, packing, page] on CPU as [slots, 2 * kv, head_dim]."""
    pages, _, words, packing, page = buf.shape
    flat = buf.cpu().permute(0, 4, 1, 2, 3).reshape(pages * page, heads_x2, -1)
    return flat[..., :SAL_HEAD_DIM]


def _fused_logical(buf: torch.Tensor, heads_x2: int) -> torch.Tensor:
    """[pages, page, kv_words, packing, padded_hd] on CPU as [slots, 2 * kv, head_dim]."""
    pages, page = buf.shape[:2]
    flat = buf.cpu().reshape(pages * page, -1, buf.shape[-1])
    return flat[:, :heads_x2, :SAL_HEAD_DIM]


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@unittest.skipIf(_skip_if_no_tpu(), "requires torch_tpu")
class TestSalSpanWrite(CustomTestCase):
    """The vendored SEQ_ALONG_LANE span write equals a reference scatter, bytes exact.

    The pool starts from random bytes, so every slot outside the spans must come
    back unchanged: partial 128-token tiles are read-modify-written.
    """

    def _check(self, *, kv_heads, dtype, cache_loc, cache_loc_2d=None, commits=None):
        from sglang_tpu.dflash_draft_kv_write import get_contiguous_slot_spans
        from sglang_tpu.kv_sal_span_write import sal_span_write_kv

        torch.manual_seed(0)
        pool = _new_sal_pool(kv_heads=kv_heads, dtype=dtype)
        for buf in pool.kv_buffer:
            buf.copy_(torch.randn(buf.shape, device=DEVICE).to(dtype))
        before = [_sal_logical(buf, 2 * kv_heads) for buf in pool.kv_buffer]
        num_tokens = int(cache_loc.shape[0])
        shape = (SAL_LAYERS, num_tokens, kv_heads, SAL_HEAD_DIM)
        k = torch.randn(shape, dtype=DTYPE, device=DEVICE)
        v = torch.randn(shape, dtype=DTYPE, device=DEVICE)

        def write(*, draft_model, kv_pool, k, v, cache_loc, cache_loc_2d, commits):
            del draft_model
            dst, src, lens, _ = get_contiguous_slot_spans(
                page_size=SAL_PAGE,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commits,
                max_num_reqs=8,
            )
            sal_span_write_kv(
                dst_starts=dst,
                src_starts=src,
                span_lens=lens,
                k=k,
                v=v,
                kv_bufs=kv_pool.kv_buffer,
            )

        _compiled(write, draft_model=nn.Linear(1, 1).to(DEVICE), kv_pool=pool)(
            k=k, v=v, cache_loc=cache_loc, cache_loc_2d=cache_loc_2d, commits=commits
        )
        torch.accelerator.synchronize()

        if cache_loc_2d is None:
            rows = (cache_loc != 0).cpu()
            slots = cache_loc.cpu()
        else:
            block = cache_loc_2d.shape[1]
            rows = (torch.arange(block)[None, :] < commits.cpu()[:, None]).reshape(-1)
            slots = cache_loc_2d.cpu().reshape(-1)
        kv = torch.stack([k, v], dim=3).reshape(
            SAL_LAYERS, num_tokens, 2 * kv_heads, SAL_HEAD_DIM
        )
        kv = kv.to(dtype).cpu()
        for layer, buf in enumerate(pool.kv_buffer):
            want = before[layer].clone()
            want[slots[rows]] = kv[layer][rows]
            got = _sal_logical(buf, 2 * kv_heads)
            self.assertTrue(
                torch.equal(_as_bytes(got), _as_bytes(want)), f"layer {layer}"
            )

    def test_extend_spans(self):
        """Mid-page start, page crossings, a partial last tile and padding rows;
        a page-aligned 256-token run also takes the aligned DMA path."""
        cases = (
            ("mid_page", dict(head_offset=91, num_tokens=700, pad_rows=13)),
            ("aligned", dict(head_offset=0, num_tokens=512, pad_rows=0)),
            ("short", dict(head_offset=250, num_tokens=9, pad_rows=7)),
        )
        for kv_heads in (1, 2, 8):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                for name, slots in cases:
                    with self.subTest(case=name, kv_heads=kv_heads, dtype=dtype):
                        self._check(
                            kv_heads=kv_heads,
                            dtype=dtype,
                            cache_loc=_sal_slots(**slots),
                        )

    def test_verify_spans(self):
        """Commit spans: whole, partial, none, and a block crossing a page."""
        block = 8
        loc2d, commits = _sal_verify_slots(
            [SAL_PAGE * 3 + 5, SAL_PAGE * 7 + 252, SAL_PAGE * 9, SAL_PAGE * 11 + 100],
            [8, 6, 0, 3],
            block,
        )
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                self._check(
                    kv_heads=1,
                    dtype=dtype,
                    cache_loc=loc2d.reshape(-1),
                    cache_loc_2d=loc2d,
                    commits=commits,
                )


@unittest.skipIf(_skip_if_no_tpu(), "requires torch_tpu")
class TestDflashSalDraftKvWrite(CustomTestCase):
    """dflash_write_draft_kv on a SEQ_ALONG_LANE pool matches the fused pool.

    Both run the same projection; the SAL path rotates K in XLA while the fused
    kernel rotates in Pallas, so bf16 K may differ by an ulp and fp8 by one fp8
    step after the cast. V, padding rows and untouched slots must match exactly.
    """

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.rotary_embedding import get_rope
        from sglang.srt.runtime_context import publish
        from sglang.srt.server_args import ServerArgs

        publish(ServerArgs(model_path="dummy"), role="test")
        torch.manual_seed(2)
        rope = get_rope(
            SAL_HEAD_DIM,
            rotary_dim=SAL_HEAD_DIM,
            max_position=MAX_POSITION,
            base=ROPE_THETA,
            is_neox_style=True,
            dtype=DTYPE,
        )
        rope.cos_sin_cache = rope.cos_sin_cache.to(DEVICE)
        cls.models = {
            kv_heads: _StubDraftModel(
                rope,
                num_layers=SAL_LAYERS,
                kv_heads=kv_heads,
                head_dim=SAL_HEAD_DIM,
                hidden=SAL_HIDDEN,
            ).to(DEVICE)
            for kv_heads in (1, 8)
        }

    def _check(self, *, kv_heads, dtype, cache_loc, positions, **verify):
        from sglang_tpu.dflash_draft_kv_write import dflash_write_draft_kv

        model = self.models[kv_heads]
        ctx_hidden = torch.randn(
            (int(cache_loc.shape[0]), SAL_HIDDEN), dtype=DTYPE, device=DEVICE
        )
        kwargs = dict(
            ctx_hidden=ctx_hidden,
            ctx_positions=positions,
            ctx_cache_loc=cache_loc,
            cache_loc_2d=verify.get("cache_loc_2d"),
            commit_lens=verify.get("commit_lens"),
        )
        pools = {}
        for name, make in (("sal", _new_sal_pool), ("fused", _new_fused_pool)):
            pools[name] = make(kv_heads=kv_heads, dtype=dtype)
            _compiled(
                dflash_write_draft_kv,
                draft_model=model,
                kv_pool=pools[name],
                max_num_reqs=8,
            )(**kwargs)
        torch.accelerator.synchronize()
        for layer in range(SAL_LAYERS):
            got = _sal_logical(pools["sal"].kv_buffer[layer], 2 * kv_heads)
            want = _fused_logical(pools["fused"].kv_buffer[layer], 2 * kv_heads)
            msg = f"layer {layer}"
            # V rows (odd heads) and every slot the write skipped are bit equal.
            self.assertTrue(
                torch.equal(_as_bytes(got[:, 1::2]), _as_bytes(want[:, 1::2])), msg
            )
            got_k, want_k = got[:, 0::2].float(), want[:, 0::2].float()
            if dtype == torch.bfloat16:
                torch.testing.assert_close(got_k, want_k, atol=1e-2, rtol=1e-2, msg=msg)
                continue
            # One e4m3 step is 2^-3 of the value; flips need a bf16 ulp at a tie.
            torch.testing.assert_close(
                got_k, want_k, atol=2.0**-9, rtol=2.0**-3, msg=msg
            )
            flipped = (got_k != want_k).float().mean().item()
            self.assertLess(flipped, 0.01, msg)

    def test_extend(self):
        for kv_heads in (1, 8):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                with self.subTest(kv_heads=kv_heads, dtype=dtype):
                    cache_loc = _sal_slots(head_offset=91, num_tokens=700, pad_rows=13)
                    positions = torch.arange(
                        3000, 3000 + int(cache_loc.shape[0]), device=DEVICE
                    )
                    self._check(
                        kv_heads=kv_heads,
                        dtype=dtype,
                        cache_loc=cache_loc,
                        positions=positions,
                    )

    def test_verify(self):
        block = 8
        loc2d, commits = _sal_verify_slots(
            [SAL_PAGE * 3 + 5, SAL_PAGE * 7 + 252, SAL_PAGE * 9, SAL_PAGE * 11 + 100],
            [8, 6, 0, 3],
            block,
        )
        valid = torch.arange(block, device=DEVICE)[None, :] < commits[:, None]
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                self._check(
                    kv_heads=1,
                    dtype=dtype,
                    cache_loc=torch.where(valid, loc2d, 0).reshape(-1),
                    positions=torch.arange(loc2d.numel(), device=DEVICE) + 5000,
                    cache_loc_2d=loc2d,
                    commit_lens=commits,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
