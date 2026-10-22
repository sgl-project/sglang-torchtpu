# SPDX-License-Identifier: Apache-2.0
"""TPU relay storage and per-instance direct handoff ownership."""

from sglang.srt.managers.overlap_utils import FutureMap
from sglang_tpu.overlap_utils_tpu import (
    device_accept_resolved_tpu,
    flush_direct_handoff_tpu,
    publish_tpu,
    resolve_device_accept_seq_lens_tpu,
    stash_tpu,
)


class TpuFutureMap(FutureMap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tpu_direct_handoff = None
        self.tpu_published_seq_lens = None

    def _store(self, buf, indices, values):
        # Keep row offsets as tensor data rather than specialized scatter positions.
        values = values.to(device=buf.device, dtype=buf.dtype, non_blocking=True)
        buf.index_copy_(0, indices.to(buf.device, non_blocking=True), values)

    def prepare_prefill_inputs(self, batch):
        if any(batch.multimodal_inputs or ()):
            batch.mm_input_ids_cpu = batch.prefill_input_ids_cpu

    def _record_indices_stream(self, indices):
        # One queue owns relay issue order; host indices have no device stream.
        pass

    def _resolve_spec_extras(self, batch):
        if not device_accept_resolved_tpu(self, batch):
            super()._resolve_spec_extras(batch)

    def _resolve_device_seq_lens(self, batch):
        return resolve_device_accept_seq_lens_tpu(self, batch)

    def _keep_committed_seq_lens(self, batch):
        # DFLASH's host mirror is the committed verify-window base. Re-reading
        # published lengths would both sync and shift that pre-gathered window.
        return (
            self.spec_algo.is_dflash_family()
            and batch.spec_info.tpu_host_seq_lens_is_committed
        )

    def stash(self, future_indices, payload):
        if not stash_tpu(self, future_indices, payload):
            super().stash(future_indices, payload)

    def publish(self, future_indices, new_seq_lens, confidence=None):
        super().publish(future_indices, new_seq_lens, confidence)
        publish_tpu(self, future_indices, new_seq_lens)

    def stash_bonus_tokens(self, indices, bonus_tokens):
        flush_direct_handoff_tpu(self)
        super().stash_bonus_tokens(indices, bonus_tokens)

    def resolve_mixed_spec_tails(self, batch):
        flush_direct_handoff_tpu(self)
        super().resolve_mixed_spec_tails(batch)
