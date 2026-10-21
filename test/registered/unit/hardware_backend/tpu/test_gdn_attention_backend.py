"""Unit tests for the TPU GDN attention backend's metadata (CPU-runnable).

These cover the pure-Python/torch part of
``sglang_tpu.attention_gdn_pallas.TpuGDNAttnBackend`` that
needs no JAX / torch_tpu / TPU device: the ``distribution`` dispatch vector the
fused kernel keys on.

``distribution = [decode_end, prefill_end, mixed_end]`` decides how the kernel
splits a ragged batch into decode / prefill regions. A wrong vector silently
routes tokens through the wrong phase (plausible-but-wrong output, not a crash),
so it is worth pinning directly: ``[bs, bs, bs]`` for a pure-decode batch,
``[0, bs, bs]`` for extend/prefill (matching the RPA v3 backend's convention),
int32, and cached per ``(is_decode, bs, device)``.

Kernel/numerics (including mixed prefill+decode batches) are gated on TPU by
``test/registered/tpu/unit/test_gdn_attention_v3.py`` (kernel vs. eager reference)
and ``test_gdn_attention_op.py`` (bridge op + state write-back). This file only
pins the CPU-side metadata contract.

The backend's ``__init__`` builds the JAX op eagerly and needs a ``ModelRunner``;
we instantiate a thin subclass that skips both — ``_get_distribution`` only
touches ``self._distribution_cache``.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_gdn_pallas import TpuGDNAttnBackend


class _BackendNoInit(TpuGDNAttnBackend):
    """TpuGDNAttnBackend without the JAX-op build / ModelRunner dependency.

    ``_get_distribution`` reads only ``self._distribution_cache``, so a bare
    instance with that dict is enough to exercise the metadata logic on CPU.
    """

    def __init__(self):  # noqa: D107  (deliberately skips super().__init__)
        self._distribution_cache = {}
        self._query_start_loc_cache = {}
        self._slot_memo = None
        self._spec_window = 1
        self._read_offsets_pool = None
        self._read_offsets = None
        self._req_pool_indices = None
        self._num_spec_tokens = 0


class TestGDNDistribution(CustomTestCase):
    def setUp(self):
        self.backend = _BackendNoInit()
        self.device = torch.device("cpu")

    def test_decode_distribution(self):
        # Pure-decode batch: every sequence is one step -> [bs, bs, bs].
        for bs in (1, 4, 37):
            dist = self.backend._get_distribution(True, bs, self.device)
            self.assertEqual(dist.tolist(), [bs, bs, bs])
            self.assertEqual(dist.dtype, torch.int32)
            self.assertEqual(dist.device.type, "cpu")

    def test_extend_distribution(self):
        # Extend/prefill batch: no decode seqs -> [0, 0, bs]. The middle element
        # is inert (the GDN wrapper reads only distribution[0] and [-1]); it is
        # written as num_decode to match tpu-inference's tpu_runner.py.
        for bs in (1, 4, 37):
            dist = self.backend._get_distribution(False, bs, self.device)
            self.assertEqual(dist.tolist(), [0, 0, bs])
            self.assertEqual(dist.dtype, torch.int32)

    def test_distribution_last_element_is_the_real_seq_count(self):
        # What the kernel actually depends on: seqs[distribution[0]:distribution[-1]]
        # bounds the non-decode range, so the last element must be the request
        # count in both modes.
        for is_decode in (True, False):
            for bs in (1, 8):
                dist = self.backend._get_distribution(is_decode, bs, self.device)
                self.assertEqual(dist.tolist()[-1], bs)

    def test_decode_and_extend_differ(self):
        bs = 8
        decode = self.backend._get_distribution(True, bs, self.device)
        extend = self.backend._get_distribution(False, bs, self.device)
        self.assertNotEqual(decode.tolist(), extend.tolist())

    def test_cached_identity_per_key(self):
        # Same (is_decode, bs, device) returns the cached tensor (not rebuilt);
        # a different key builds a distinct one.
        a = self.backend._get_distribution(True, 4, self.device)
        b = self.backend._get_distribution(True, 4, self.device)
        self.assertIs(a, b)
        c = self.backend._get_distribution(True, 5, self.device)
        self.assertIsNot(a, c)
        d = self.backend._get_distribution(False, 4, self.device)
        self.assertIsNot(a, d)


class TestGDNSinglePassFlags(CustomTestCase):
    """init_forward_metadata derives the kernel's single-pass flags from the
    same forward mode as the distribution: decode runs batched_only over
    [bs, bs, bs], extend runs prefill_only over [0, 0, bs]. A mismatch would run
    one pass over an empty range: zero output, no state update, no error."""

    def _run(self, forward_mode, bs=3):
        n_kq, n_v, d, k = 1, 2, 4, 4
        dim = 2 * n_kq * d + n_v * d
        backend = _BackendNoInit()
        backend._op = Mock(return_value=torch.zeros(bs, n_v, d))
        cache = SimpleNamespace(
            conv=[torch.zeros(bs + 1, k - 1, 1, dim)],
            temporal=torch.zeros(bs + 1, n_v, d, d),
        )
        backend._extra_buffer_enabled = False
        backend.req_to_token_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(
                checkpoint_width=1, checkpoint_read_offsets=None
            ),
            mamba2_layer_cache=lambda _: cache,
            get_mamba_indices=lambda indices: indices,
            translate_mamba_indices=lambda indices: indices,
            # Read by the slot-id memo (TpuGDNAttnBackend._slot_indices).
            req_index_to_mamba_index_mapping=torch.zeros(1, dtype=torch.int32),
            mamba_v2p_table=None,
        )
        batch = SimpleNamespace(
            seq_lens=torch.full((bs,), 5),
            forward_mode=forward_mode,
            req_pool_indices=torch.arange(1, bs + 1),
            extend_seq_lens=torch.ones(bs, dtype=torch.int64),
            mamba_track_mask=None,
        )
        backend.init_forward_metadata(batch)
        layer = SimpleNamespace(
            layer_id=0,
            conv_weights=torch.zeros(dim, k),
            bias=None,
            A_log=torch.zeros(n_v),
            dt_bias=torch.zeros(n_v),
            num_k_heads=n_kq,
            num_v_heads=n_v,
            head_k_dim=d,
            head_v_dim=d,
        )
        backend._run(
            layer,
            mixed_qkv=torch.zeros(bs, dim),
            a=torch.zeros(bs, n_v),
            b=torch.zeros(bs, n_v),
        )
        kwargs = backend._op.call_args.kwargs
        distribution = backend._op.call_args.args[11].tolist()
        return distribution, kwargs["batched_only"], kwargs["prefill_only"]

    def test_flags_follow_the_distribution(self):
        bs = 3
        for mode, decode in (
            (ForwardMode.DECODE, True),
            (ForwardMode.IDLE, True),
            (ForwardMode.EXTEND, False),
            (ForwardMode.MIXED, False),
        ):
            with self.subTest(mode=mode.name):
                distribution, batched_only, prefill_only = self._run(mode, bs)
                self.assertEqual(distribution, [bs, bs, bs] if decode else [0, 0, bs])
                self.assertEqual(batched_only, decode)
                self.assertEqual(prefill_only, not decode)


class TestGDNVerifyMetadata(CustomTestCase):
    """A verify writes its checkpoints into the request's window (rows from 10,
    width 3 here) and reads the committed state: the slot row for AT_SLOT (-1),
    else a window row. Extend reads and writes the slot row."""

    def _backend(self, offsets):
        backend = _BackendNoInit()
        backend._spec_window = 3
        backend._read_offsets_pool = offsets
        backend._extra_buffer_enabled = False
        backend.req_to_token_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(
                select_checkpoints=Mock(),
                snapshot_checkpoints=Mock(),
                window_bases=lambda req: 10 + req.clamp(min=0) * 3,
            ),
            get_mamba_indices=lambda indices: indices,
            translate_mamba_indices=lambda indices: indices,
            # Read by the slot-id memo (TpuGDNAttnBackend._slot_indices).
            req_index_to_mamba_index_mapping=torch.zeros(1, dtype=torch.int32),
            mamba_v2p_table=None,
        )
        return backend

    def _batch(self, mode):
        return SimpleNamespace(
            req_pool_indices=torch.tensor([2, 1, -1]),
            mamba_track_mask=None,
            _tpu_real_b=2,
            seq_lens=torch.tensor([17, 29, 0]),
            extend_seq_lens=torch.tensor([3, 3, 3]),
            forward_mode=mode,
            spec_info=SimpleNamespace(num_tokens_per_req=3),
        )

    def test_verify_passes_slots_and_the_offset_table(self):
        offsets = torch.tensor([0, 2, 1], dtype=torch.int32)
        backend = self._backend(offsets)
        backend.init_forward_metadata(self._batch(ForwardMode.TARGET_VERIFY))
        self.assertEqual(backend._distribution.tolist(), [2, 2, 2])
        self.assertEqual(backend._query_start_loc.tolist(), [0, 3, 6, 9])
        self.assertEqual(backend._seq_lens.tolist(), [20, 32, 3])
        self.assertEqual(backend._state_indices.tolist(), [2, 1, -1])
        self.assertIs(backend._read_offsets, offsets)
        self.assertEqual(backend._req_pool_indices.tolist(), [2, 1, -1])
        self.assertEqual(backend._num_spec_tokens, 2)
        self.assertEqual(offsets.tolist(), [0, 2, 1])

    def test_verify_rows_resolve_slot_and_window_states(self):
        # Slot 2 is committed at its own row, slot 1 at window row 2, and the
        # DP pad (-1) reads the null slot row 0.
        offsets = torch.tensor([-1, 2, -1], dtype=torch.int32)
        backend = self._backend(offsets)
        backend.init_forward_metadata(self._batch(ForwardMode.TARGET_VERIFY))
        bases, read_offsets = backend._state_rows()
        self.assertEqual(bases.tolist(), [16, 13, 10])
        self.assertEqual((bases + read_offsets).tolist(), [2, 15, 0])

    def test_extend_resets_the_batch_offsets(self):
        offsets = torch.tensor([0, 2, 1], dtype=torch.int32)
        backend = self._backend(offsets)
        backend.init_forward_metadata(self._batch(ForwardMode.EXTEND))
        self.assertIsNone(backend._read_offsets)
        self.assertEqual(backend._num_spec_tokens, 0)
        self.assertEqual(offsets.tolist(), [-1, -1, -1])
        self.assertIs(backend._state_rows()[0], backend._state_indices)

    def test_commit_selects_checkpoints_for_the_verified_slots(self):
        backend = self._backend(torch.zeros(3, dtype=torch.int32))
        steps = torch.tensor([1, -1])
        # A DP pad slot (-1) maps to the null slot, which the pool keeps inert.
        backend.commit_verified_states(
            None, torch.tensor([2, -1]), steps, None, None, None
        )
        slots, got_steps = (
            backend.req_to_token_pool.mamba_pool.select_checkpoints.call_args.args
        )
        self.assertEqual(slots.tolist(), [2, 0])
        self.assertIs(got_steps, steps)

    def test_commit_snapshots_the_tracked_checkpoints(self):
        backend = self._backend(torch.zeros(3, dtype=torch.int32))
        backend._req_pool_indices = torch.tensor([2, 1])
        steps_to_track = torch.tensor([0, -1])
        # One live request: its track slot is padded to the bucket on the host.
        backend.commit_verified_states(
            None,
            torch.tensor([2, 1]),
            torch.tensor([1, -1]),
            torch.tensor([3]),
            steps_to_track,
            None,
        )
        track_slots, req_pool_indices, steps = (
            backend.req_to_token_pool.mamba_pool.snapshot_checkpoints.call_args.args
        )
        self.assertEqual(track_slots.tolist(), [3, -1])
        self.assertIs(req_pool_indices, backend._req_pool_indices)
        self.assertIs(steps, steps_to_track)


class _RowReadRecorder:
    """Mapping stand-in that records each row read by a Python int."""

    def __init__(self, rows: int):
        self.shape = (rows,)
        self.device = torch.device("cpu")
        self.int_rows: list[int] = []

    def __getitem__(self, index):
        if isinstance(index, int):
            self.int_rows.append(index)
        return torch.zeros(1, dtype=torch.int32)

    def __setitem__(self, index, value):
        pass


class TestGDNSlotBookkeepingWarmup(CustomTestCase):
    """The P/D state payload reads mapping[req_pool_idx] with a Python int, which
    compiles one program per row; warmup must cover every row a request can hold."""

    def _warm_rows(self, disaggregation_mode: str) -> list[int]:
        backend = _BackendNoInit()
        mapping = _RowReadRecorder(rows=5)
        backend.req_to_token_pool = SimpleNamespace(
            size=4,
            req_index_to_mamba_index_mapping=mapping,
            mamba_pool=SimpleNamespace(clear_slots=Mock()),
        )
        with patch(
            "sglang_tpu.attention_gdn_pallas.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode=disaggregation_mode),
        ):
            backend._warmup_slot_bookkeeping()
        return mapping.int_rows

    def test_pd_servers_warm_every_mapping_row(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                self.assertEqual(sorted(self._warm_rows(mode)), list(range(5)))

    def test_aggregated_server_skips_row_reads(self):
        self.assertEqual(self._warm_rows("null"), [])


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX")
class TestGDNRequestTilePadding(CustomTestCase):
    def test_partial_verify_tiles_have_inert_metadata_rows(self):
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas.gdn.metadata import (
            compute_batched_seq_metadata,
        )

        for requests in (1, 3, 5):
            with self.subTest(requests=requests):
                md = compute_batched_seq_metadata(
                    cfg=SimpleNamespace(seq_tile_size=4, tile_size=4),
                    seq_lens=jnp.full((requests,), 19, dtype=jnp.int32),
                    query_start_loc=jnp.arange(requests + 1, dtype=jnp.int32) * 3,
                    state_indices=1 + jnp.arange(requests, dtype=jnp.int32) * 3,
                    read_offsets=jnp.zeros(requests, dtype=jnp.int32),
                    end_seq=jnp.asarray(requests, dtype=jnp.int32),
                )
                padded = ((requests + 3) // 4) * 4
                self.assertEqual(
                    md.p_id_to_r_size.data.tolist(),
                    [3] * requests + [0] * (padded - requests),
                )
                self.assertEqual(
                    md.p_id_is_first_tile.data.tolist(),
                    [True] * requests + [False] * (padded - requests),
                )
                self.assertEqual(
                    md.p_id_is_last_tile.data.tolist(),
                    [True] * requests + [False] * (padded - requests),
                )
                self.assertEqual(
                    md.p_id_to_s_idx.data.tolist(),
                    list(range(requests)) + [0] * (padded - requests),
                )


if __name__ == "__main__":
    unittest.main()
