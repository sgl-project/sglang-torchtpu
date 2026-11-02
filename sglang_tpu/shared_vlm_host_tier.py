"""Shared CPU storage beneath each engine's private TPU embedding cache."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from sglang_tpu.mm_embedding import EmbeddingSpan, PaddedEmbedding
from sglang_tpu.mm_schedule_tpu import _HostEntry, _sha256_rows, _VlmHostTier
from sglang_tpu.shared_host_cache import HostCacheLease, SharedHostCache

logger = logging.getLogger(__name__)


class _SharedEntry(_HostEntry):
    __slots__ = ("lease", "upload_done", "upload_start_ns", "device_buffers")

    def __init__(self, lease: HostCacheLease):
        super().__init__(lease.buffers, lease.spans, lease.nbytes)
        self.lease = lease
        self.upload_done = None
        self.upload_start_ns = 0
        self.device_buffers = []

    def close(self):
        self.buffers = []
        self.device_buffers = []
        self.lease.close()


class VlmSharedHostTier(_VlmHostTier):
    def __init__(self, cap_bytes, verify, stats_s, directory, namespace):
        self.shared = SharedHostCache(directory, cap_bytes, namespace)
        self._async_error = None
        super().__init__(cap_bytes, verify, stats_s)
        self._uploads = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vlm-shared-upload"
        )
        logger.info(
            "[vlm-shared-host] directory=%s total_cap_mib=%d namespace=%s",
            directory,
            cap_bytes >> 20,
            namespace,
        )

    def _check_error(self):
        if self._async_error is not None:
            raise RuntimeError("Shared VLM cache worker failed") from self._async_error

    def write_back(self, key, emb):
        self._check_error()
        super().write_back(key, emb)

    def _copy_to_host(self, key, emb, t_sub):
        try:
            with (
                torch.inference_mode(),
                torch.accelerator.device_index(self._dev_idx),
                self._dm.stream(self._stream),
            ):
                entry = self._to_host(emb)
                event = self._dm.Event()
                event.record()
                event.synchronize()
            outcome = self.shared.put(key, entry.buffers, entry.spans)
            with self._lock:
                self._pending.discard(key)
                self.c[f"shared_put_{outcome}"] += 1
                if outcome == "inserted":
                    self.c["wb_done"] += 1
                    self.c["wb_bytes"] += entry.nbytes
                    self.c["wb_lat_ns"] += time.perf_counter_ns() - t_sub
        except Exception as error:
            with self._lock:
                self._pending.discard(key)
                self._async_error = error
                self.c["wb_error"] += 1
            logger.exception("[vlm-shared-host] write-back failed")
            raise

    def lookup(self, key, device):
        self._check_error()
        with self._lock:
            pinned = self._pending_emb.get(key) if self.pin_pending else None
        if pinned is not None:
            self.c["pin_hit"] += 1
            return pinned, None
        lookup_start = time.perf_counter_ns()
        lease = self.shared.get(key)
        self.c["shared_lookup_ns"] += time.perf_counter_ns() - lookup_start
        if lease is None:
            self.c["host_miss"] += 1
            return None
        entry = _SharedEntry(lease)
        try:
            entry.upload_start_ns = time.perf_counter_ns()
            dev_buffers = [b.to(device, non_blocking=True) for b in entry.buffers]
            if self.verify:
                entry.device_buffers = dev_buffers
            entry.upload_done = self._dm.Event()
            entry.upload_done.record()
            if entry.spans is None:
                embedding = dev_buffers[0]
            else:
                embedding = PaddedEmbedding(
                    tuple(
                        EmbeddingSpan(dev_buffers[i], start, length)
                        for i, start, length in entry.spans
                    )
                )
            self.c["host_hit"] += 1
            self.c["h2d_bytes"] += entry.nbytes
            self.c["h2d_enqueue_ns"] += time.perf_counter_ns() - entry.upload_start_ns
            if not self.verify:
                self._uploads.submit(self._finish_upload, entry)
            return embedding, entry
        except BaseException:
            # A partial upload can still be reading the source even if a later upload fails.
            self._dm.synchronize()
            entry.close()
            raise

    def _finish_upload(self, entry):
        try:
            entry.upload_done.synchronize()
            with self._lock:
                self.c["shared_h2d_complete_ns"] += (
                    time.perf_counter_ns() - entry.upload_start_ns
                )
                self.c["shared_h2d_complete"] += 1
        except Exception as error:
            self._async_error = error
            logger.exception("[vlm-shared-host] upload completion failed")
            raise
        finally:
            entry.close()

    def verify_pair(self, key, fresh, entry):
        try:
            full_diff_before = self.c["verify_full_diff"]
            ok = super().verify_pair(key, fresh, entry)
            upload_ok = all(
                _sha256_rows(device.to("cpu")) == _sha256_rows(host)
                for device, host in zip(entry.device_buffers, entry.buffers)
            )
            self.c[
                "shared_upload_verify_ok" if upload_ok else "shared_upload_verify_bad"
            ] += 1
            if (
                not ok
                or not upload_ok
                or self.c["verify_full_diff"] != full_diff_before
            ):
                raise RuntimeError(
                    f"Shared VLM embedding differs from a fresh encode: {key}"
                )
            return ok
        finally:
            self._finish_upload(entry)

    def _host_stats(self):
        stats = self.shared.stats()
        return stats["entries"], stats["nbytes"], len(self._pending)

    def maybe_log(self):
        self._check_error()
        super().maybe_log()
