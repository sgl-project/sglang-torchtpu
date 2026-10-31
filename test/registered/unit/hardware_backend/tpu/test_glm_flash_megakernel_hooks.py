"""CPU checks of the GLM-5.3-Flash megakernel's serving hooks."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import glm_flash_megakernel as hook

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Kernel:
    def __init__(self, req):
        self.req = req

    def invalidate(self):
        self.req = None


class TestGlmFlashMegakernelHooks(CustomTestCase):
    def test_freeing_the_kernels_request_drops_it(self):
        mk = _Kernel(req=3)
        with mock.patch.object(hook, "_KERNELS", [mk]):
            hook.drop_freed_glm_flash_megakernel_request(object(), [5, 3])
        self.assertIsNone(mk.req)

    def test_freeing_another_request_keeps_it(self):
        mk = _Kernel(req=3)
        with mock.patch.object(hook, "_KERNELS", [mk]):
            hook.drop_freed_glm_flash_megakernel_request(object(), [5])
        self.assertEqual(mk.req, 3)

    def test_checkpoint_flushes_only_the_kernel_holding_the_request(self):
        """The tree reads a finished request's pages and state slot, which lag the
        kernel until it flushes."""
        holder, other = mock.MagicMock(), mock.MagicMock()
        holder.holds.side_effect = lambda req: req == 3
        other.holds.return_value = False
        for mk in (holder, other):
            mk.dsa_caches = [torch.zeros(1)]
        req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=3))
        with mock.patch.object(hook, "_KERNELS", [holder, other]):
            hook.flush_glm_flash_megakernel_request(
                SimpleNamespace(disable=False), req, up_to=8
            )
        holder.flush.assert_called_once()
        other.flush.assert_not_called()

    def test_disabled_cache_never_flushes(self):
        """With the radix cache off the tree reads nothing, so a finished request
        must not pay a flush and device sync."""
        holder = mock.MagicMock()
        holder.holds.return_value = True
        req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=3))
        with mock.patch.object(hook, "_KERNELS", [holder]):
            hook.flush_glm_flash_megakernel_request(
                SimpleNamespace(disable=True), req, up_to=8
            )
        holder.flush.assert_not_called()


class TestGlmFlashMegakernelSnapshot(CustomTestCase):
    """The kernel's KDA state reaches the radix track slot exactly where serving
    would have snapshotted it; a skipped row writes only the null slot 0."""

    def _kernel(self, q: int):
        from sglang_tpu.glm_flash_megakernel.ops import GlmFlashMegakernel

        mk = object.__new__(GlmFlashMegakernel)
        mk.Q, mk.spec = q, SimpleNamespace(kda_layers=[0])
        mk.radix_snapshots = True
        shape = (q,) if q > 1 else ()
        mk.kda_caches = (
            [torch.arange(1.0, 1.0 + max(q, 1)).reshape(*shape, 1, 1) * 10],
            [torch.arange(1.0, 1.0 + max(q, 1)).reshape(*shape, 1, 1)],
        )
        self.temporal, self.conv = torch.zeros(8, 1, 1), torch.zeros(8, 1, 1)
        state = SimpleNamespace(temporal=self.temporal, conv=[self.conv])
        mk._req_pool = SimpleNamespace(mamba2_layer_cache=lambda _layer: state)
        mk._lin = SimpleNamespace(_translate_mamba_indices=lambda ids: ids + 1)
        return mk

    def test_decode_snapshot_writes_the_translated_track_slot(self):
        mk = self._kernel(q=1)
        mk.snapshot(torch.tensor([4]))
        self.assertEqual(self.temporal[5].item(), 10.0)
        self.assertEqual(self.conv[5].item(), 1.0)

    def test_verify_snapshot_takes_the_accepted_checkpoint(self):
        mk = self._kernel(q=3)
        mk.snapshot(torch.tensor([2]), torch.tensor([1]))
        self.assertEqual(self.temporal[3].item(), 20.0)
        self.assertEqual(self.conv[3].item(), 2.0)

    def test_skipped_rows_write_only_the_null_slot(self):
        mk = self._kernel(q=3)
        mk.snapshot(torch.tensor([2]), torch.tensor([-1]))
        mk._lin = SimpleNamespace(_translate_mamba_indices=lambda ids: ids)
        mk.snapshot(torch.tensor([-1]), torch.tensor([0]))
        self.assertEqual(self.temporal[1:].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
