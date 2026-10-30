"""Unit tests for the TPU-native recurrent (GDN) state pool.

``TpuMambaPool`` only allocates plain torch tensors, so these tests run on
CPU (``device="cpu"``) and need no TPU — they pin the kernel-native buffer
layout and slot bookkeeping that the fused Pallas GDN kernel depends on:

* **one tensor per layer, never a stacked ``[num_layers, ...]`` buffer**, and
  ``at_layer_idx`` hands back a tensor that *owns its storage*. This is a
  performance invariant with no functional symptom, so nothing else catches a
  regression: torch_tpu's Pallas bridge writes donated state back with
  ``tensor.copy_(result)``, which XLA elides against a whole buffer but must
  lower to slice + dynamic-update-slice against a view of a stacked pool. On
  Qwen3.5-397B that view cost ~0.70 ms per decode step — 1.4x the GDN kernel
  itself — while producing byte-identical output;
* conv buffer is ``[size+1, K-1, 1, conv_dim]`` per layer (the kernel's
  rank-4 form; any other shape is re-tiled on every call), for both GDN
  ``(conv_dim, K-1)`` and Kimi ``(K-1, conv_dim)`` cache shapes;
* recurrent buffer is ``[size+1, n_v, d_k, d_v]`` per layer (swapped vs.
  the GPU ``[..., n_v, d_v, d_k]``);
* slot 0 is reserved; ``alloc``/``free`` cycle real slots ``1..size``;
* ``clear_slots`` points speculative offsets back at the slot row;
  ``copy_from`` clones slot rows for radix CoW, with or without speculation;
* with speculation, slots keep one row each and verify windows sit after them,
  one per request pool row, so a radix-sized pool fits core's mamba budget.
* ReplaySSM is disabled here, but its flags must still be *present* --
  the shared req-slot alloc path reads them without a hasattr guard.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.mamba_pool import AT_SLOT, TpuMambaPool, verify_track_steps


def _make_cache_params(n_kq, n_v, d_k, d_v, K, layers):
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=n_v * d_v,
        n_groups=n_kq,
        num_heads=n_v,
        head_dim=d_v,
        state_size=d_k,
        conv_kernel=K,
    )
    return Mamba2CacheParams(
        dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
        layers=layers,
        shape=shape,
    )


class TestTpuMambaPool(CustomTestCase):
    N_KQ, N_V, D_K, D_V, K = 2, 8, 128, 128, 4
    LAYERS = [1, 3, 5]
    SIZE = 16

    def _pool(self):
        cp = _make_cache_params(
            self.N_KQ, self.N_V, self.D_K, self.D_V, self.K, self.LAYERS
        )
        return TpuMambaPool(
            size=self.SIZE,
            spec_state_size=0,
            cache_params=cp,
            mamba_layer_ids=self.LAYERS,
            device="cpu",
        )

    def test_speculative_checkpoint_storage_and_commit(self):
        cp = _make_cache_params(
            self.N_KQ, self.N_V, self.D_K, self.D_V, self.K, self.LAYERS
        )
        pool = TpuMambaPool(
            size=3,
            spec_state_size=3,
            cache_params=cp,
            mamba_layer_ids=self.LAYERS,
            device="cpu",
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=1,
        )
        self.assertEqual(pool.checkpoint_width, 4)
        # 4 slot rows, then 4 request windows of 4 rows.
        self.assertEqual(pool.mamba_cache.temporal[0].shape[0], 20)
        pool.select_checkpoints(torch.tensor([1, 3]), torch.tensor([2, 0]))
        self.assertEqual(pool.checkpoint_read_offsets.tolist(), [-1, 2, -1, 0])
        # Reordered batch, null padding, and inactive rows do not change another request.
        pool.select_checkpoints(torch.tensor([3, 1, 0]), torch.tensor([3, -1, 2]))
        self.assertEqual(pool.checkpoint_read_offsets.tolist(), [-1, 2, -1, 3])
        pool.clear_slots(torch.tensor([1]))
        self.assertEqual(pool.checkpoint_read_offsets.tolist(), [-1, -1, -1, 3])
        # fp32 accounting covers all allocated slot and window rows.
        self.assertEqual(
            pool.get_kv_size_bytes(), 20 * pool.cache_bytes_per_request(cp)
        )

    def test_pd_state_entries_follow_per_layer_layout(self):
        """P/D registers one entry per per-layer tensor, in get_state order, with
        the TP slice axis of the kernel-native layout; the transfer row of a slot
        is the slot row, whatever the checkpoint width."""
        cp = _make_cache_params(
            self.N_KQ, self.N_V, self.D_K, self.D_V, self.K, self.LAYERS
        )
        pool = TpuMambaPool(
            size=3,
            spec_state_size=3,
            cache_params=cp,
            mamba_layer_ids=self.LAYERS,
            device="cpu",
            speculative_num_draft_tokens=4,
            speculative_eagle_topk=1,
        )
        state = pool.get_state()
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        self.assertEqual(ptrs, [t.data_ptr() for t in state])
        self.assertEqual(lens, [t.nbytes for t in state])
        self.assertEqual(item_lens, [t[0].nbytes for t in state])
        num_conv = len(pool.mamba_cache.conv) * len(self.LAYERS)
        self.assertEqual(pool.get_state_layer_ids(), self.LAYERS * (len(state) // 3))
        conv_dim = pool.mamba_cache.conv[0][0].shape[-1]
        self.assertEqual(
            pool.get_state_dim_per_tensor(),
            [conv_dim] * num_conv + [self.N_V] * len(self.LAYERS),
        )
        self.assertEqual(pool.committed_row(3), 3)
        self.assertEqual(self._pool().committed_row(3), 3)

    def test_disabled_replayssm_surface_covers_base(self):
        """Disabling ReplaySSM must mean "present and inert", not "absent".

        ``HybridReqToTokenPool.alloc`` resets the ReplaySSM ring on every
        freshly assigned slot and reads the pool's flags with bare attribute
        access -- ``self.mamba_pool.replayssm_cache_base is not None`` -- with
        no hasattr guard. A pool that simply omits an attribute it does not
        use therefore raises AttributeError on the first prefill of any hybrid
        model, long after startup has reported success.

        Comparing against a live ``MambaPool`` rather than a hardcoded name
        list is deliberate: the failure mode this guards is "the base grew
        another disabled-state flag and the TPU pool was not updated", which a
        fixed list would not catch.
        """
        cp = _make_cache_params(
            self.N_KQ, self.N_V, self.D_K, self.D_V, self.K, self.LAYERS
        )
        common = dict(
            size=self.SIZE,
            spec_state_size=0,
            cache_params=cp,
            mamba_layer_ids=self.LAYERS,
            device="cpu",
        )
        base_attrs = {n for n in vars(MambaPool(**common)) if "replayssm" in n}
        tpu_attrs = {n for n in vars(TpuMambaPool(**common)) if "replayssm" in n}
        self.assertTrue(
            base_attrs,
            "no ReplaySSM attributes found on MambaPool -- this test has "
            "drifted from the base pool and no longer guards anything.",
        )
        self.assertEqual(
            set(),
            base_attrs - tpu_attrs,
            "TpuMambaPool is missing ReplaySSM attributes that MambaPool "
            "defines; unconditional readers such as "
            "HybridReqToTokenPool.alloc will raise AttributeError.",
        )

    def test_kernel_native_buffer_layout(self):
        pool = self._pool()
        num_layers = len(self.LAYERS)
        conv_dim = self.N_KQ * self.D_K * 2 + self.N_V * self.D_V
        # One tensor per layer: the layer axis is the list index, not a
        # tensor dimension.
        self.assertEqual(len(pool.mamba_cache.conv[0]), num_layers)
        self.assertEqual(len(pool.mamba_cache.temporal), num_layers)
        for layer in range(num_layers):
            self.assertEqual(
                list(pool.mamba_cache.conv[0][layer].shape),
                [self.SIZE + 1, self.K - 1, 1, conv_dim],
            )
            self.assertEqual(
                list(pool.mamba_cache.temporal[layer].shape),
                [self.SIZE + 1, self.N_V, self.D_K, self.D_V],
            )

    def test_kimi_conv_layout(self):
        # KimiLinearStateShape stores (K-1, conv_dim), not (conv_dim, K-1).
        shape = KimiLinearStateShape.create(
            tp_world_size=1,
            num_heads=self.N_V,
            head_dim=self.D_V,
            conv_kernel_size=self.K,
        )
        cp = KimiLinearCacheParams(
            dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
            layers=self.LAYERS,
            shape=shape,
        )
        pool = TpuMambaPool(
            size=self.SIZE,
            spec_state_size=0,
            cache_params=cp,
            mamba_layer_ids=self.LAYERS,
            device="cpu",
        )
        self.assertEqual(
            list(pool.mamba_cache.conv[0][0].shape),
            [self.SIZE + 1, self.K - 1, 1, 3 * self.N_V * self.D_V],
        )

    def test_layers_are_independent_allocations(self):
        """Every layer's state must own its buffer.

        A stacked ``[num_layers, ...]`` pool passes every *functional* test in
        this file while silently costing ~0.70 ms/decode step, so the allocation
        shape has to be asserted directly. ``_base`` is what discriminates:
        distinct ``data_ptr()`` does not, because iterating a stacked tensor
        also yields one view per layer, each at its own offset.
        """
        pool = self._pool()
        for layer, temporal in enumerate(pool.mamba_cache.temporal):
            self.assertIsNone(temporal._base, f"recurrent layer {layer} is a view")
        for kind, conv_kind in enumerate(pool.mamba_cache.conv):
            for layer, conv in enumerate(conv_kind):
                self.assertIsNone(conv._base, f"conv[{kind}] layer {layer} is a view")

    def test_per_layer_cache_owns_its_storage(self):
        pool = self._pool()
        conv_dim = self.N_KQ * self.D_K * 2 + self.N_V * self.D_V
        # dense layer index 0 corresponds to global layer id LAYERS[0].
        per = pool.mamba2_layer_cache(0)
        self.assertEqual(
            list(per.conv[0].shape), [self.SIZE + 1, self.K - 1, 1, conv_dim]
        )
        self.assertEqual(
            list(per.temporal.shape), [self.SIZE + 1, self.N_V, self.D_K, self.D_V]
        )
        # The invariant the Pallas bridge's in-place write-back depends on:
        # a whole buffer, not a view into a stacked pool. `_base is None` is
        # exactly what distinguishes the two.
        self.assertIsNone(per.temporal._base)
        self.assertIsNone(per.conv[0]._base)
        # ...and it must still be the pool's own tensor, not a copy, or the
        # kernel's state write-back would land nowhere.
        self.assertIs(per.temporal, pool.mamba_cache.temporal[0])
        per.temporal.fill_(4.0)
        self.assertTrue(torch.all(pool.mamba_cache.temporal[0] == 4.0))
        self.assertTrue(torch.all(pool.mamba_cache.temporal[1] == 0.0))

    def test_state_swap_routes_each_layer_to_its_tensor(self):
        """direct_compile swaps placeholders in through set_state while tracing.

        A slicing mistake hands one layer another layer's (or the conv's) state,
        which still traces and runs but silently mixes recurrent states.
        """
        pool = self._pool()
        num_layers = len(self.LAYERS)
        state = pool.get_state()
        self.assertEqual(len(state), 2 * num_layers)
        replacement = [torch.empty_like(tensor) for tensor in state]
        pool.set_state(replacement)
        for layer in range(num_layers):
            per = pool.mamba2_layer_cache(layer)
            self.assertIs(per.conv[0], replacement[layer])
            self.assertIs(per.temporal, replacement[num_layers + layer])
        self.assertEqual(
            [id(t) for t in pool.get_state()], [id(t) for t in replacement]
        )

    def test_total_bytes_match_stacked_layout(self):
        """Per-layer allocation must not change the memory footprint."""
        pool = self._pool()
        num_layers = len(self.LAYERS)
        conv_dim = self.N_KQ * self.D_K * 2 + self.N_V * self.D_V
        expected = (
            num_layers
            * (self.SIZE + 1)
            * 4
            * ((self.K - 1) * conv_dim + self.N_V * self.D_K * self.D_V)
        )
        self.assertEqual(int(pool.mamba_cache.mem_usage_bytes()), expected)

    def test_clear_slots_is_noop(self):
        pool = self._pool()
        for temporal in pool.mamba_cache.temporal:
            temporal.fill_(7.0)
        pool.clear_slots(torch.tensor([1, 2]))
        # Kernel handles fresh-slot masking; pool must not touch the buffer.
        self.assertTrue(all(torch.all(t == 7.0) for t in pool.mamba_cache.temporal))

    def test_copy_from_preserves_sources_and_other_slots(self):
        pool = self._pool()
        states = [*pool.mamba_cache.conv[0], *pool.mamba_cache.temporal]
        for layer, state in enumerate(states):
            state[1].fill_(layer + 1)
            state[2].fill_(layer + 10)
        pool.copy_from(torch.tensor([1, 1, 2]), torch.tensor([3, 4, 5]))
        for layer, state in enumerate(states):
            for slot in (1, 3, 4):
                self.assertTrue(torch.all(state[slot] == layer + 1))
            for slot in (2, 5):
                self.assertTrue(torch.all(state[slot] == layer + 10))
            self.assertTrue(torch.all(state[0] == 0))
            self.assertTrue(torch.all(state[6:] == 0))

    def test_tracking_ids_snapshot_request_state_before_overlap(self):
        from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs

        buffers = [torch.tensor([7, 8]), torch.tensor([9, 10])]
        reqs = [
            SimpleNamespace(
                kv=SimpleNamespace(
                    mamba_ping_pong_track_buffer=buf,
                    mamba_next_track_idx=i,
                )
            )
            for i, buf in enumerate(buffers)
        ]
        for allocator_device in ("cpu", "cuda"):
            with self.subTest(allocator_device=allocator_device):
                batch = SimpleNamespace(
                    reqs=reqs,
                    req_pool_indices=torch.tensor([0, 1]),
                    req_to_token_pool=SimpleNamespace(
                        mamba_pool=SimpleNamespace(
                            slot_allocator_device=allocator_device
                        ),
                        req_index_to_mamba_ping_pong_track_buffer_mapping=torch.stack(
                            buffers
                        ),
                    ),
                )
                # Exercise both the host-ID path and the existing device-gather
                # path with CPU tensors; hardware lowering is tested separately.
                tensor = torch.tensor

                def unpinned_tensor(*args, **kwargs):
                    kwargs.pop("pin_memory", None)
                    return tensor(*args, **kwargs)

                with patch("torch.tensor", side_effect=unpinned_tensor):
                    set_mamba_track_indices_from_reqs(batch)
                    self.assertEqual(batch.mamba_track_indices.tolist(), [7, 10])
                    set_mamba_track_indices_from_reqs(batch, track_positions=[1, 0])
                self.assertEqual(batch.mamba_track_buffer_indices, [1, 0])
                self.assertEqual(batch.mamba_track_indices.tolist(), [8, 9])
                saved = batch.mamba_track_indices.clone()
                buffers[0][1] = 99
                torch.testing.assert_close(batch.mamba_track_indices, saved)
                buffers[0][1] = 8
                reqs[0].kv.mamba_next_track_idx = None
                reqs[0].kv.mamba_ping_pong_track_buffer = None
                with patch("torch.tensor", side_effect=unpinned_tensor):
                    for positions in (None, [1, 1]):
                        set_mamba_track_indices_from_reqs(batch, positions)
                        self.assertEqual(batch.mamba_track_indices.tolist(), [-1, 10])
                reqs[0].kv.mamba_next_track_idx = 0
                reqs[0].kv.mamba_ping_pong_track_buffer = buffers[0]

    def _spec_pool(self, *, size, spec_state_size, width):
        cp = _make_cache_params(
            self.N_KQ, self.N_V, self.D_K, self.D_V, self.K, self.LAYERS
        )
        return TpuMambaPool(
            size=size,
            spec_state_size=spec_state_size,
            cache_params=cp,
            mamba_layer_ids=self.LAYERS,
            device="cpu",
            speculative_num_draft_tokens=width,
            speculative_eagle_topk=1,
        )

    def test_radix_sized_spec_pool_addresses_slots_and_windows(self):
        """More slots than requests (radix snapshots): slots 0..16 keep one row,
        and request r verifies in rows 17 + 2r .. 17 + 2r + 1."""
        pool = self._spec_pool(size=self.SIZE, spec_state_size=4, width=2)
        self.assertEqual(pool.mamba_cache.temporal[0].shape[0], 17 + 5 * 2)
        req = torch.tensor([3, 0])
        self.assertEqual(pool.window_bases(req).tolist(), [23, 17])
        self.assertEqual(pool.checkpoint_rows(req).tolist(), [[23, 24], [17, 18]])
        slots = torch.tensor([9, 4])
        self.assertEqual(pool.committed_state_rows(slots, req).tolist(), [9, 4])
        pool.select_checkpoints(slots, torch.tensor([1, -1]))
        self.assertEqual(pool.committed_state_rows(slots, req).tolist(), [24, 4])
        pool.clear_slots(torch.tensor([9]))
        self.assertEqual(pool.checkpoint_read_offsets[9].item(), AT_SLOT)

    def test_spec_pool_copies_slot_rows(self):
        pool = self._spec_pool(size=self.SIZE, spec_state_size=4, width=2)
        states = pool.get_state()
        for layer, state in enumerate(states):
            state[2].fill_(layer + 1)
        pool.copy_from(torch.tensor([2]), torch.tensor([6]))
        for layer, state in enumerate(states):
            self.assertTrue(torch.all(state[6] == layer + 1))
            self.assertTrue(torch.all(state[17:] == 0))

    def test_snapshot_copies_window_checkpoints_into_track_slots(self):
        pool = self._spec_pool(size=self.SIZE, spec_state_size=4, width=2)
        states = pool.get_state()
        for layer, state in enumerate(states):
            # Request 3's window holds checkpoints at rows 23 and 24.
            state[23].fill_(layer + 1)
            state[24].fill_(layer + 100)
        # Step -1 and track slot <= 0 (no buffer, or padding) skip their rows.
        pool.snapshot_checkpoints(
            torch.tensor([5, 6, -1, 0], dtype=torch.int32),
            torch.tensor([3, 3, 3, 3]),
            torch.tensor([1, -1, 0, 0]),
        )
        for layer, state in enumerate(states):
            self.assertTrue(torch.all(state[5] == layer + 100))
            self.assertTrue(torch.all(state[6] == 0))
            self.assertTrue(torch.all(state[0] == 0))

    def test_verify_track_steps_match_the_boundary_crossing(self):
        """From 62, accepting 3 reaches 65: the 64-token boundary is window step
        1. From 60, accepting 2 stays short; an inactive row (-1) commits nothing;
        from 64 the boundary already passed."""
        steps = verify_track_steps(
            torch.tensor([62, 60, 66, 64]), torch.tensor([2, 1, -1, 3]), 64
        )
        self.assertEqual(steps.tolist(), [1, -1, -1, -1])


if __name__ == "__main__":
    unittest.main()
