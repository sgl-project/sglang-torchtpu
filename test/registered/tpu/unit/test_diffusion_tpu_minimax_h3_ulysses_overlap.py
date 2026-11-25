"""MiniMax-H3 Ulysses head-group pipeline on TPU (minimax_h3_ulysses_overlap).

The pipeline is meant to change only the schedule and data movement of the
attention core, so its kernels are checked bit for bit against the paths they
replace: the pack kernel against the QK RMSNorm + RoPE op plus a torch pack,
and the exchange-buffer attention kernel against the live-length kernel, head
by head. The gates that keep the serial path (Ulysses degree, local heads,
dtype, missing async all-to-all flags) and the pack kernel's VMEM row blocks
for H3's Ulysses 8, 4 and 2 are checked without a device.
"""

import importlib.util
import math
import os
import threading
import types
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as dit
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import minimax_h3_qk_norm_rope as qk_rope
from sglang_tpu.multimodal_gen import minimax_h3_ulysses_overlap as overlap
from sglang_tpu.multimodal_gen import minimax_h3_ulysses_pack as pack
from sglang_tpu.multimodal_gen import tpu_splash_attn, worker

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=1)

EPS = 1e-5  # MiniMaxH3DiTArchConfig.qk_norm_eps
HEADS = 56  # MiniMax-H3 attention heads
FLAGS_ON = " ".join(overlap.LIBTPU_FLAGS)


def _skip_unless_tpu7x(test):
    if importlib.util.find_spec("torch_tpu") is None:
        test.skipTest("torch_tpu not available")
    from torch_tpu._internal.utils.hardware import get_tpu_device_name

    chip = get_tpu_device_name()
    if "7" not in chip:
        test.skipTest(f"Pallas kernels are TPU7x-only, this chip is {chip!r}")


def _gen(seed):
    return torch.Generator().manual_seed(seed)


class TestUlyssesPackKernel(CustomTestCase):
    def test_pack_matches_qk_norm_rope_op_and_torch_pack(self):
        # Ulysses 8, 4 and 2 (7, 14 and 28 local heads); 520 rows leave a
        # ragged last row block. The norm weights reach H3's ~37.
        _skip_unless_tpu7x(self)
        rows = 520
        for ranks in (8, 4, 2):
            with self.subTest(ranks=ranks):
                self._check_pack(ranks, HEADS // ranks, rows)

    def _check_pack(self, ranks, local_heads, rows):
        gen = _gen(ranks)
        width = ranks * local_heads * 128
        qkv = (torch.randn(rows, 3 * width, generator=gen) * 4).bfloat16()
        q_w, k_w = ((torch.randn(128, generator=gen) * 12).bfloat16() for _ in "qk")
        half = torch.rand(rows, 48, generator=gen) * 200
        cache = dit._rope_cos_sin_cache(
            torch.cat((half, half), dim=-1), dtype=torch.bfloat16
        )
        # The replaced path: the QK-norm + RoPE op on [rows, heads, 128] q/k.
        q, k, v = (
            t.reshape(rows, -1, 128).contiguous() for t in qkv.split(width, dim=-1)
        )
        qkv, q_w, k_w, cache = (t.to("tpu") for t in (qkv, q_w, k_w, cache))
        q_n, k_n = qk_rope._ensure_op()(q.to("tpu"), k.to("tpu"), q_w, k_w, cache, EPS)
        parts = [q_n.to("cpu"), k_n.to("cpu"), v]

        sizes, leads = pack.group_layout(
            local_heads, overlap.HEADS_PER_GROUP, overlap.LEAD
        )
        got = []
        for first_group, count in zip((0, 1), overlap._pack_op_counts(local_heads)):
            op = pack.ensure_op(count)
            got += overlap._tensors(
                op(
                    qkv,
                    q_w,
                    k_w,
                    cache,
                    EPS,
                    ranks,
                    overlap.HEADS_PER_GROUP,
                    overlap.LEAD,
                    first_group,
                )
            )
        self.assertEqual(len(got), len(sizes))
        head = 0
        for g, (size, lead) in enumerate(zip(sizes, leads)):
            buffer = got[g].to("cpu")
            self.assertEqual(buffer.shape, (ranks, lead + 3 * size, rows, 128))
            for j in range(size):
                for w in range(ranks):
                    for p in range(3):
                        want = parts[p][:, w * local_heads + head + j]
                        self.assertTrue(
                            torch.equal(buffer[w, lead + 3 * j + p], want),
                            (ranks, g, j, w, p),
                        )
            head += size

    def test_pack_row_blocks_fit_the_vmem_budget(self):
        # Ulysses 8 (7 local heads) and 4 (14) keep 512-row blocks; Ulysses 2's
        # 27-group call takes 256 (512 rows would need ~48 MiB of blocks).
        budget = pack._VMEM_BLOCK_BUDGET_BYTES
        expected = {7: (512, 512), 14: (512, 512), 28: (512, 256)}
        for local_heads, blocks in expected.items():
            counts = overlap._pack_op_counts(local_heads)
            self.assertEqual(counts, (1, local_heads - 1))
            for first_group, count, want in zip((0, 1), counts, blocks):
                tiles = pack.pack_row_blocks(
                    local_heads,
                    overlap.HEADS_PER_GROUP,
                    overlap.LEAD,
                    first_group,
                    count,
                )
                block = pack.block_rows_for(16384, tiles)
                self.assertEqual(block, want, (local_heads, first_group))
                self.assertLessEqual(pack._vmem_bytes(block, tiles), budget)
                if block < 512:
                    self.assertGreater(pack._vmem_bytes(2 * block, tiles), budget)
        # A shard shorter than a block is one block; unaligned rows are refused.
        self.assertEqual(pack.block_rows_for(264, 20), 264)
        self.assertIsNone(pack.block_rows_for(260, 20))


class TestExchangeBufferAttention(CustomTestCase):
    def test_parts_kernel_matches_the_live_length_kernel_per_head(self):
        # q|k|v of 2 heads read in place from a [ranks, parts, rows, 128]
        # buffer, one leading part of garbage before them, against the
        # live-length kernel over the gathered [S, 2 * 128] rows.
        _skip_unless_tpu7x(self)
        tsa = tpu_splash_attn
        self.assertTrue(tsa._check_splash_available())
        tsa._register_ops()
        heads = 2
        for ranks, shard_rows, live in ((8, 2048, 16000), (2, 8192, 16384)):
            seq = ranks * shard_rows
            with self.subTest(ranks=ranks, live=live):
                blocks, _, rows = tsa._live_length_kernel(seq, seq)
                self.assertEqual(rows, seq)
                self.assertEqual(shard_rows % blocks[0], 0)
                self.assertEqual(shard_rows % blocks[1], 0)
                gen = _gen(seq + live)
                buffer = torch.randn(
                    ranks, 1 + 3 * heads, shard_rows, 128, generator=gen
                )
                buffer = buffer.bfloat16()
                # [S, heads * 128] rows of q, k and v.
                q, k, v = (
                    buffer[:, 1 + p :: 3].permute(0, 2, 1, 3).reshape(seq, -1)
                    for p in range(3)
                )
                live_t = torch.tensor([live], dtype=torch.int32).to("tpu")
                scale = 128**-0.5
                full = tsa._FLASH_LIVE_LENGTH_OP(
                    q.to("tpu"), k.to("tpu"), v.to("tpu"), live_t, scale, heads
                ).to("cpu")
                got = tsa._FLASH_LIVE_LENGTH_PARTS_OP(
                    buffer.to("tpu"), live_t, scale, heads, 1
                ).to("cpu")
                self.assertEqual(got.shape, (ranks, heads, shard_rows, 128))
                want = full.view(ranks, shard_rows, heads, 128).permute(0, 2, 1, 3)
                self.assertTrue(torch.equal(got, want))
                self.assertTrue((full[live:] == 0).all())


class TestCopyPathPadding(CustomTestCase):
    def test_pads_only_when_the_kernel_keeps_its_blocks(self):
        # The copy path pads a ragged last 2048-row block above 8192 rows when
        # the padded call keeps the kernel's blocks, so each head's result is
        # unchanged (one-head calls with a ragged block ran 1.8x slower next
        # to the in-flight exchanges).
        cases = {
            33280: 1536,  # 720P: 2048-row blocks either way
            35328: 1536,
            109056: 1536,
            16896: 0,  # query blocks would change, 1920 -> 2048 rows
            8704: 0,  # 1792 -> 2048
            10752: 0,  # 1792 -> 2048
            12800: 0,  # 1920 -> 2048
            37888: 0,  # the kernel already pads it (S % 2048 = 1024)
            15872: 0,  # likewise (S % 2048 = 1536)
            32768: 0,  # whole blocks
            4608: 0,  # up to 8192 rows: no padding
        }
        for seq, pad in cases.items():
            with self.subTest(seq=seq):
                self.assertEqual(overlap._copy_path_pad(seq), pad)
                if pad:
                    blocks = tpu_splash_attn._live_length_kernel(seq, seq)[0]
                    padded = tpu_splash_attn._live_length_kernel(seq + pad, seq + pad)
                    self.assertEqual(padded, (blocks, True, seq + pad))

    def test_padded_one_head_call_equals_the_seven_head_call(self):
        # 720P's copy path: each head padded to 34816 rows and run alone gives
        # the bits of the serial seven-head call over 33280 rows.
        _skip_unless_tpu7x(self)
        tsa = tpu_splash_attn
        self.assertTrue(tsa._check_splash_available())
        tsa._register_ops()
        seq, heads = 33280, 7
        pad = overlap._copy_path_pad(seq)
        for large_logits in (False, True):
            gen = _gen(seq + large_logits)
            q, k, v = (torch.randn(seq, heads, 128, generator=gen) for _ in range(3))
            if large_logits:
                gain = torch.empty(seq, 1, 1).uniform_(
                    math.log(1.25), math.log(10.0), generator=gen
                )
                q = q * gain.exp()
                k = k * torch.linspace(4.0, 8.0, seq)[:, None, None]
            q, k, v = (t.bfloat16() for t in (q, k, v))
            for live in (seq - 180, seq):
                with self.subTest(large_logits=large_logits, live=live):
                    live_t = torch.tensor([live], dtype=torch.int32).to("tpu")
                    full = tsa._FLASH_LIVE_LENGTH_OP(
                        *(t.reshape(seq, -1).to("tpu") for t in (q, k, v)),
                        live_t,
                        128**-0.5,
                        heads,
                    )
                    full = full.to("cpu").view(seq, heads, 128)
                    for h in range(heads):
                        one = tsa._FLASH_LIVE_LENGTH_OP(
                            *(
                                torch.nn.functional.pad(
                                    t[:, h].contiguous().to("tpu"), (0, 0, 0, pad)
                                )
                                for t in (q, k, v)
                            ),
                            live_t,
                            128**-0.5,
                            1,
                        )[:seq]
                        self.assertTrue(torch.equal(one.to("cpu"), full[:, h]), h)


class _Ranks:
    """``world`` ranks run as threads; ``exchange`` is their all-to-all over
    dim 0 (chunk w of every rank's input goes to rank w, in rank order)."""

    def __init__(self, world):
        self.world = world
        self._barrier = threading.Barrier(world)
        self._slots = [None] * world
        self._local = threading.local()

    def exchange(self, x):
        rank = self._local.rank
        self._slots[rank] = x
        self._barrier.wait()
        out = torch.cat([s.chunk(self.world, 0)[rank] for s in self._slots])
        self._barrier.wait()
        return out

    def run(self, fn):
        results, errors = [None] * self.world, []

        def body(rank):
            self._local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as exc:  # surfaced below
                errors.append(exc)
                self._barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.world)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return results


def _head_attention(q, k, v, live):
    """One head's live-length attention, ``[S, 128]`` bf16 -> bf16; float64
    on the CPU, so a head's result does not depend on which call it is in."""
    out = torch.zeros(q.shape, dtype=torch.float64)
    scores = q[:live].double() @ k[:live].double().T * 128**-0.5
    out[:live] = scores.softmax(-1) @ v[:live].double()
    return out.to(q.dtype)


def _rows_kernel(q, k, v, live, scale, heads):
    """The live-length op's contract on ``[S, heads * 128]`` rows."""
    cols = [slice(h * 128, (h + 1) * 128) for h in range(heads)]
    live = int(live[0])
    return torch.cat(
        [_head_attention(q[:, c], k[:, c], v[:, c], live) for c in cols], 1
    )


def _parts_kernel(buffer, live, scale, heads, col):
    """The parts op's contract: ``[W, P, R, 128]`` -> ``[W, heads, R, 128]``."""
    shards, _, rows, d = buffer.shape
    seq = buffer.permute(1, 0, 2, 3).reshape(buffer.shape[1], shards * rows, d)
    out = [
        _head_attention(
            seq[col + 3 * h], seq[col + 3 * h + 1], seq[col + 3 * h + 2], int(live[0])
        )
        for h in range(heads)
    ]
    return torch.stack(out).view(heads, shards, rows, d).transpose(0, 1)


def _pack_op(count):
    """The pack op's contract without the norm and rotation: group buffers
    ``[W, lead + 3 * size, R, 128]`` with q, k, v per head after ``lead``
    empty parts."""

    def op(qkv, q_w, k_w, cache, eps, ranks, per_group, lead, first_group):
        rows = qkv.shape[0]
        local = qkv.shape[1] // (3 * ranks * 128)
        sizes, leads = pack.group_layout(local, per_group, lead)
        q, k, v = (t.view(rows, ranks, local, 128) for t in qkv.chunk(3, dim=1))
        out = []
        for g in range(first_group, first_group + count):
            start = sum(sizes[:g])
            parts = [
                t[:, :, h].transpose(0, 1)
                for h in range(start, start + sizes[g])
                for t in (q, k, v)
            ]
            empty = qkv.new_zeros((ranks, leads[g], rows, 128))
            out.append(torch.cat([empty, torch.stack(parts, dim=1)], dim=1))
        return tuple(out)

    return op


class TestPipelineMatchesSerial(CustomTestCase):
    """The head-group pipeline against serial attention over the gathered
    sequence, with ``world`` ranks simulated on the CPU: the collectives and
    kernels are replaced by their contracts, so this checks the buffer
    layout, group bookkeeping and output reassembly."""

    def _check(self, world, local_heads, shard_rows, live, fused):
        heads, seq = world * local_heads, world * shard_rows
        gen = _gen(seq + heads)
        q, k, v = (
            torch.randn(seq, heads, 128, generator=gen).bfloat16() for _ in range(3)
        )
        want = torch.stack(
            [_head_attention(q[:, h], k[:, h], v[:, h], live) for h in range(heads)],
            dim=1,
        )
        ranks = _Ranks(world)
        cu = torch.tensor([0, live, seq], dtype=torch.int32)
        impl = types.SimpleNamespace(softmax_scale=128**-0.5)
        norm = types.SimpleNamespace(weight=torch.ones(128), eps=EPS)
        attention = types.SimpleNamespace(
            _attention_impl=impl, num_heads=heads, q_norm=norm, k_norm=norm
        )

        def rank_fn(rank):
            shard = slice(rank * shard_rows, (rank + 1) * shard_rows)
            if fused:
                qkv = torch.cat([t[shard].flatten(1) for t in (q, k, v)], dim=1)
                cache = torch.zeros(shard_rows, pack.ROTARY_DIM, dtype=torch.bfloat16)
                return overlap._fused_attend(attention, qkv, cache, cu)
            return overlap._pipelined_core(
                attention, q[shard], k[shard], v[shard], cu_seqlens=cu
            )

        with (
            patch.object(overlap, "_exchange", ranks.exchange),
            patch.object(
                parallel_state, "get_ulysses_parallel_world_size", return_value=world
            ),
            patch.object(tpu_splash_attn, "_FLASH_LIVE_LENGTH_OP", _rows_kernel),
            patch.object(tpu_splash_attn, "_FLASH_LIVE_LENGTH_PARTS_OP", _parts_kernel),
            patch.object(pack, "registered_op", _pack_op),
        ):
            got = torch.cat(ranks.run(rank_fn))
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(torch.equal(got, want))

    def test_pipeline_matches_serial_attention(self):
        # 2 x 2048 rows: the kernel reads the exchange buffers in place;
        # 4 x 1024 rows: its blocks straddle ranks, so each group is copied
        # out. Three local heads: groups 0 and 1 go first, group 2's input
        # rides with group 0's output.
        for world, shard_rows in ((2, 2048), (4, 1024)):
            self.assertEqual(
                overlap._blocks_divide(shard_rows, world), world == 2, world
            )
            for fused in (False, True):
                with self.subTest(world=world, fused=fused):
                    self._check(
                        world, 3, shard_rows, live=world * shard_rows - 300, fused=fused
                    )


def _attention(heads=HEADS, causal=False):
    impl = Mock(spec=tpu_splash_attn.TpuSplashAttentionImpl)
    impl.trailing_padding_on_device = True
    impl.causal = causal
    return types.SimpleNamespace(_attention_impl=impl, num_heads=heads)


def _core_kwargs(**overrides):
    kwargs = dict(
        cu_seqlens=torch.tensor([0, 30000, 32768], dtype=torch.int32),
        cu_seqlens_host=None,
        max_seqlen=32768,
        subblock_sparse_query_block_mask=None,
        ulysses_active=True,
        ring_active=False,
        gate_compress=None,
    )
    kwargs.update(overrides)
    return kwargs


class TestPipelineGating(CustomTestCase):
    def setUp(self):
        _skip_unless_tpu7x(self)
        patches = [
            patch.object(overlap, "_ASYNC_ALL_TO_ALL", True),
            patch.object(tpu_splash_attn, "_FLASH_LIVE_LENGTH_OP", Mock()),
            patch.object(tpu_splash_attn, "_FLASH_LIVE_LENGTH_PARTS_OP", Mock()),
            patch.object(parallel_state, "get_ulysses_parallel_world_size"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        parallel_state.get_ulysses_parallel_world_size.return_value = 8

    def _eligible(self, attention=None, *, rows=4096, heads=HEADS, dtype=None, **kw):
        qkv = torch.empty(
            (rows, heads, 128), dtype=dtype or torch.bfloat16, device="tpu"
        )
        return overlap._eligible(
            attention or _attention(heads), qkv, qkv, qkv, _core_kwargs(**kw)
        )

    def test_eligible_calls_and_every_gate(self):
        world = parallel_state.get_ulysses_parallel_world_size
        self.assertTrue(self._eligible())
        # Ulysses 2: 28 local heads.
        world.return_value = 2
        self.assertTrue(self._eligible())
        # Ulysses 1, and one local head per rank (nothing to pipeline).
        for size in (1, 56):
            world.return_value = size
            self.assertFalse(self._eligible(), size)
        world.return_value = 8
        self.assertFalse(self._eligible(heads=60))  # 60 % 8 != 0
        self.assertFalse(self._eligible(dtype=torch.float32))
        self.assertFalse(self._eligible(rows=4100))  # rows % 16 != 0
        self.assertFalse(self._eligible(ring_active=True))
        self.assertFalse(self._eligible(ulysses_active=False))
        self.assertFalse(self._eligible(cu_seqlens_host=(0, 30000, 32768)))
        self.assertFalse(self._eligible(_attention(causal=True)))
        self.assertFalse(self._eligible(cu_seqlens=torch.tensor([0, 9, 30000, 32768])))
        with patch.object(tpu_splash_attn, "_FLASH_LIVE_LENGTH_PARTS_OP", None):
            self.assertFalse(self._eligible())
        with patch.object(overlap, "_ASYNC_ALL_TO_ALL", False):
            self.assertFalse(self._eligible())

    def test_without_async_flags_the_core_stays_serial(self):
        # LIBTPU_INIT_ARGS without the async all-to-all flags: install() keeps
        # every call on the original core, and layers register no pack op.
        original = Mock(return_value="serial")
        with (
            patch.dict(os.environ, {"LIBTPU_INIT_ARGS": ""}),
            patch.object(overlap, "_INSTALLED", False),
            patch.object(dit, "_minimax_h3_attention_core_impl", original),
            patch.object(dit, "_minimax_h3_bind_attention_backend"),
            patch.object(dit.MiniMaxH3Attention, "__init__"),
            patch.object(dit.MiniMaxH3Attention, "forward"),
        ):
            overlap.install()
            self.assertFalse(overlap._ASYNC_ALL_TO_ALL)
            q = torch.empty((4096, HEADS, 128), dtype=torch.bfloat16, device="tpu")
            attention = _attention()
            out = dit._minimax_h3_attention_core_impl(
                attention, q, q, q, **_core_kwargs()
            )
            self.assertEqual(out, "serial")
            original.assert_called_once()
            with (
                patch.object(parallel_state, "model_parallel_is_initialized") as init,
                patch.object(pack, "ensure_op") as ensure_op,
            ):
                init.return_value = True
                overlap._register_pack_ops(attention)
                ensure_op.assert_not_called()

    def test_async_flag_parsing(self):
        cases = {
            "": False,
            overlap.LIBTPU_FLAGS[0]: False,
            FLAGS_ON: True,
            f"--xla_foo=1 {FLAGS_ON}": True,
            # The last occurrence wins, as libtpu parses them.
            f"{FLAGS_ON} --xla_tpu_force_async_all_to_all=false": False,
            f"--xla_tpu_force_async_all_to_all=false {FLAGS_ON}": True,
        }
        for value, want in cases.items():
            with patch.dict(os.environ, {"LIBTPU_INIT_ARGS": value}):
                self.assertEqual(overlap.async_all_to_all_enabled(), want, value)


class TestWorkerLibtpuFlags(CustomTestCase):
    def _args(self, config_name="MiniMaxH3PipelineConfig", ulysses=8, ring=1):
        config = type(config_name, (), {})
        subclass = type("PikaH3FlexConfig", (config,), {})
        return types.SimpleNamespace(
            pipeline_config=subclass(), ulysses_degree=ulysses, ring_degree=ring
        )

    def test_flags_only_for_an_h3_ulysses_worker(self):
        cases = (
            (self._args(), True),
            (self._args(ulysses=1), False),
            (self._args(ring=2), False),
            (self._args(config_name="WanPipelineConfig"), False),
        )
        target = "sglang.multimodal_gen.runtime.server_args.get_global_server_args"
        for args, want in cases:
            with (
                patch(target, return_value=args),
                patch.dict(os.environ, {"LIBTPU_INIT_ARGS": "--xla_foo=1"}),
            ):
                worker.apply_libtpu_flags()
                value = os.environ["LIBTPU_INIT_ARGS"]
                self.assertEqual(FLAGS_ON in value, want, args)
                self.assertTrue(value.startswith("--xla_foo=1"))
        # A flag the launcher set keeps its value.
        user = "--xla_tpu_enable_async_all_to_all=false"
        with (
            patch(target, return_value=self._args()),
            patch.dict(os.environ, {"LIBTPU_INIT_ARGS": user}),
        ):
            worker.apply_libtpu_flags()
            value = os.environ["LIBTPU_INIT_ARGS"].split()
            self.assertEqual(value, [user, overlap.LIBTPU_FLAGS[1]])
        with patch(target, side_effect=ValueError("not set")):
            self.assertFalse(worker._wants_ulysses_overlap_flags())


if __name__ == "__main__":
    unittest.main()
