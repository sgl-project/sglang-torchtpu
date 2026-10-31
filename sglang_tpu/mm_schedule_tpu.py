"""TPU embedding scheduling that retains padded buffers through cache and chunks."""

from __future__ import annotations

import collections
import hashlib
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.mm_schedule import (
    PerImageRequestInfo,
    _acknowledge_deferred_cuda_ipc_cache_hits,
    _count_mm_tokens_in_extend,
    _discard_mismatched_cached_embedding,
    _get_precomputed_embedding,
)
from sglang.srt.managers.mm_schedule import (
    _move_items_to_device as _default_move_items_to_device,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.multimodal.evs import EVSEmbeddingResult
from sglang.srt.multimodal.transport.cuda_ipc import BORROW_CUDA_IPC_FEATURE_KEY
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.utils.async_probe import maybe_assert_sum
from sglang_tpu.mm_embedding import (
    EmbeddingSpan,
    MultimodalEmbedding,
    PaddedEmbedding,
    TpuEmbeddingResult,
    concat_embeddings,
    flatten_embedding,
    slice_embedding,
    split_embedding,
)

logger = logging.getLogger(__name__)

DataEmbeddingFunc = Callable[
    [List[MultimodalDataItem]],
    MultimodalEmbedding | List[MultimodalEmbedding] | EVSEmbeddingResult,
]


def _move_items_to_device(
    items: List[MultimodalDataItem],
    device: torch.device,
    data_embedding_func: DataEmbeddingFunc,
) -> None:
    """Let encoders with CPU patchification choose their own H2D boundary.

    Muse assigns each image to one encoder-DP rank before uploading patches.
    Moving every image on every rank first defeats that contract. Other
    encoders keep the shared scheduler's upload behavior.
    """
    owner = getattr(data_embedding_func, "__self__", None)
    if getattr(owner, "keeps_mm_features_on_host", False):
        return
    _default_move_items_to_device(items, device, data_embedding_func)


# Host-RAM tier under the device VLM embedding cache (env-gated; default off = unchanged behaviour).
#   SGLANG_VLM_HOST_TIER_MB=<MiB>     private host copy (per scheduler process = per replica) of every per-image /
#                                     per-clip embedding encoded here; LRU by bytes.
#   SGLANG_VLM_HOST_TIER_VERIFY=1     on a host hit also encode the item fresh, compare sha256 of the host copy with the
#                                     fresh encode (valid rows and whole padded buffers), then serve the host copy.
#   SGLANG_VLM_HOST_TIER_STATS_S=<s>  period of the "[vlm-host-tier]" stats line (default 60). Setting only this
#                                     variable counts device-tier hits/misses/encodes without a host tier.
#   SGLANG_VLM_HOST_TIER_PIN_PENDING=1  STRADDLE-PIN: while an encode's write-back is in flight, a lookup that misses
#                                     both tiers is served that encode's own device embedding (the object the
#                                     write-back job already holds) instead of encoding the item again ("pin_hit").
#                                     The reference is dropped when the write-back job ends (landed, too big or
#                                     failed), so it holds no memory the job does not already hold.
# Write-back: after an encode is inserted into the device cache, one worker thread copies the item's buffers to the
# host (non_blocking D2H on the worker's own TPU stream + event sync, as sglang_tpu.result_copy_worker does), so the
# scheduler thread only enqueues. Lookup: on a device miss, a host hit is uploaded (non_blocking H2D, host source kept
# referenced) and re-inserted into the device cache where a device hit would have been used. Bytes are copied, never
# recomputed: a served host entry is bit-identical to the encode that produced it.
# Engines: the tier is built only when one scheduler process is the whole engine (tp_size == 1 and pp_size == 1);
# see _single_process_engine.
class _HostEntry:
    __slots__ = ("buffers", "spans", "nbytes")

    def __init__(self, buffers, spans, nbytes):
        self.buffers = buffers  # host tensors
        self.spans = spans  # PaddedEmbedding: tuple of (buffer index, start, length); plain tensor: None
        self.nbytes = nbytes


def _sha256_rows(t: torch.Tensor) -> str:
    t = t.contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


class _VlmHostTier:
    def __init__(
        self,
        cap_bytes: int,
        verify: bool,
        stats_s: float,
        pin_pending: Optional[bool] = None,
    ):
        self.cap = cap_bytes
        self.verify = verify and cap_bytes > 0
        if pin_pending is None:
            pin_pending = os.environ.get("SGLANG_VLM_HOST_TIER_PIN_PENDING", "0") == "1"
        self.pin_pending = pin_pending and cap_bytes > 0
        # STRADDLE-PIN: key -> the embedding its pending write-back job is copying.
        self._pending_emb = {}
        self.stats_s = stats_s
        self.c = collections.Counter()
        self.sizes = collections.Counter()  # device-cache insert bytes histogram (MiB, rounded)
        self._t_last = time.monotonic()
        self._lock = threading.Lock()
        self._lru: "collections.OrderedDict" = collections.OrderedDict()
        self._size = 0
        self._pending = set()
        self._recent_sources = collections.deque(maxlen=256)
        self._pool = None
        if cap_bytes > 0:
            self._dm = torch.get_device_module("tpu")
            self._dev_idx = torch.accelerator.current_device_index()
            self._stream = self._dm.Stream(device=self._dev_idx)
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-host-tier")
        logger.info(
            "[vlm-host-tier] enabled=%s cap_mib=%d verify=%s stats_s=%s pin_pending=%s",
            cap_bytes > 0, cap_bytes >> 20, self.verify, stats_s, self.pin_pending,
        )

    @property
    def enabled(self) -> bool:
        return self._pool is not None

    def note_device_insert(self, nbytes: int) -> None:
        self.sizes[round(nbytes / (1 << 20), 1)] += 1

    # ---- write-back (scheduler thread enqueues; worker copies) ----
    def write_back(self, key, emb: MultimodalEmbedding) -> None:
        with self._lock:
            if key in self._lru or key in self._pending:
                self.c["wb_skip_present"] += 1
                return
            self._pending.add(key)
            if self.pin_pending:
                self._pending_emb[key] = emb
            self.c["wb_submit"] += 1
        job = self._copy_and_unpin if self.pin_pending else self._copy_to_host
        self._pool.submit(job, key, emb, time.perf_counter_ns())

    def _copy_and_unpin(self, key, emb, t_sub: int) -> None:
        try:
            self._copy_to_host(key, emb, t_sub)
        finally:
            # After the job: the key is now in the LRU (preferred by lookup) or was dropped (too big / failed).
            with self._lock:
                self._pending_emb.pop(key, None)

    def _copy_to_host(self, key, emb, t_sub: int) -> None:
        try:
            with torch.inference_mode(), torch.accelerator.device_index(
                self._dev_idx
            ), self._dm.stream(self._stream):
                entry = self._to_host(emb)
                event = self._dm.Event()
                event.record()
                event.synchronize()
        except Exception:  # never let the tier take the server down; the device tier is unaffected
            logger.exception("[vlm-host-tier] write-back failed")
            with self._lock:
                self._pending.discard(key)
                self.c["wb_error"] += 1
            return
        with self._lock:
            self._pending.discard(key)
            if entry.nbytes > self.cap:
                self.c["wb_too_big"] += 1
                return
            while self._size + entry.nbytes > self.cap and self._lru:
                _, old = self._lru.popitem(last=False)
                self._size -= old.nbytes
                self.c["evict"] += 1
            self._lru[key] = entry
            self._size += entry.nbytes
            self.c["wb_done"] += 1
            self.c["wb_bytes"] += entry.nbytes
            self.c["wb_lat_ns"] += time.perf_counter_ns() - t_sub

    @staticmethod
    def _to_host(emb: MultimodalEmbedding) -> _HostEntry:
        if isinstance(emb, PaddedEmbedding):
            buffers, index, spans = [], {}, []
            for span in emb.spans:
                k = id(span.buffer)
                if k not in index:
                    index[k] = len(buffers)
                    buffers.append(span.buffer.to("cpu", non_blocking=True))
                spans.append((index[k], span.start, span.length))
            nbytes = sum(b.numel() * b.element_size() for b in buffers)
            return _HostEntry(buffers, tuple(spans), nbytes)
        t = emb.to("cpu", non_blocking=True)
        return _HostEntry([t], None, t.numel() * t.element_size())

    # ---- lookup on a device miss (scheduler thread) ----
    def lookup(
        self, key, device
    ) -> Optional[Tuple[MultimodalEmbedding, Optional[_HostEntry]]]:
        """(embedding, host entry) on a host hit; (embedding, None) when the embedding a
        pending write-back is copying is served (STRADDLE-PIN)."""
        pinned = None
        with self._lock:
            entry = self._lru.get(key)
            if entry is not None:
                self._lru.move_to_end(key)
            elif self.pin_pending:
                pinned = self._pending_emb.get(key)
        if pinned is not None:
            self.c["pin_hit"] += 1
            return pinned, None
        if entry is None:
            self.c["host_miss"] += 1
            return None
        t0 = time.perf_counter_ns()
        dev_bufs = [b.to(device, non_blocking=True) for b in entry.buffers]
        self._recent_sources.append(entry.buffers)  # keep the host sources of non_blocking uploads alive
        if entry.spans is None:
            emb = dev_bufs[0]
        else:
            emb = PaddedEmbedding(
                tuple(EmbeddingSpan(dev_bufs[i], start, length) for i, start, length in entry.spans)
            )
        self.c["host_hit"] += 1
        self.c["h2d_bytes"] += entry.nbytes
        self.c["h2d_enqueue_ns"] += time.perf_counter_ns() - t0
        return emb, entry

    # ---- verification (SGLANG_VLM_HOST_TIER_VERIFY=1) ----
    def verify_pin(self, key, fresh: MultimodalEmbedding, pinned: MultimodalEmbedding) -> bool:
        """A pinned embedding against a fresh encode of the same item (valid rows)."""

        def rows(emb):
            return emb.to_host() if isinstance(emb, PaddedEmbedding) else emb.to("cpu")

        a, b = rows(pinned), rows(fresh)
        ok = (
            tuple(a.shape) == tuple(b.shape)
            and a.dtype == b.dtype
            and _sha256_rows(a) == _sha256_rows(b)
        )
        self.c["verify_pin_ok" if ok else "verify_pin_bad"] += 1
        if not ok:
            logger.warning("[vlm-host-tier] PIN VERIFY MISMATCH key=%s", key)
        return ok

    def verify_pair(self, key, fresh: MultimodalEmbedding, entry: _HostEntry) -> bool:
        if entry.spans is None:
            host_rows = entry.buffers[0]
            fresh_rows = fresh.to("cpu")
            full_ok = None
        else:
            host_rows = torch.cat([entry.buffers[i][s : s + n] for i, s, n in entry.spans])
            fresh_rows = fresh.to_host() if isinstance(fresh, PaddedEmbedding) else fresh.to("cpu")
            full_ok = None
            if isinstance(fresh, PaddedEmbedding):
                fbufs = list({id(sp.buffer): sp.buffer for sp in fresh.spans}.values())
                if len(fbufs) == len(entry.buffers):
                    full_ok = all(
                        tuple(f.shape) == tuple(h.shape) and _sha256_rows(f.to("cpu")) == _sha256_rows(h)
                        for f, h in zip(fbufs, entry.buffers)
                    )
        ok = (
            tuple(host_rows.shape) == tuple(fresh_rows.shape)
            and host_rows.dtype == fresh_rows.dtype
            and _sha256_rows(host_rows) == _sha256_rows(fresh_rows)
        )
        self.c["verify_ok" if ok else "verify_bad"] += 1
        if full_ok is not None:
            self.c["verify_full_ok" if full_ok else "verify_full_diff"] += 1
        if not ok:
            logger.warning(
                "[vlm-host-tier] VERIFY MISMATCH key=%s host=%s fresh=%s", key, tuple(host_rows.shape), tuple(fresh_rows.shape)
            )
        return ok

    def maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._t_last < self.stats_s:
            return
        self._t_last = now
        with self._lock:
            c = dict(self.c)
            host_entries, host_bytes, pending = len(self._lru), self._size, len(self._pending)
        cache = mm_schedule.embedding_cache
        logger.info(
            "[vlm-host-tier] stats %s host_entries=%d host_mib=%.1f pending=%d dev_entries=%d dev_mib=%.1f insert_mib_hist=%s",
            " ".join(f"{k}={v}" for k, v in sorted(c.items())),
            host_entries, host_bytes / (1 << 20), pending,
            len(cache) if cache is not None else -1,
            (cache.current_size if cache is not None else 0) / (1 << 20),
            dict(sorted(self.sizes.items())),
        )


_HOST_TIER: Optional[_VlmHostTier] = None
_HOST_TIER_INIT = False


def _single_process_engine() -> bool:
    """Whether this scheduler process is the whole engine (tp_size == 1, pp_size == 1).

    The tier is private to its process, and its hit/miss outcome depends on when
    write-backs land. In a multi-process engine the processes that run one batch
    could then pass different miss lists to the encoder, and an encoder with
    collectives inside the encode (Muse Spark Omni's encoder-DP over
    attn_tp_group, a weights-TP ViT) would hang or exchange the wrong rows. A
    width that is not published counts as 1: no process groups exist then.
    """
    parallel = get_parallel()
    for name in ("tp_size", "pp_size"):
        try:
            if getattr(parallel, name) != 1:
                return False
        except ValueError:  # "config namespace 'parallel' not published"
            pass
    return True


def _host_tier() -> Optional[_VlmHostTier]:
    global _HOST_TIER, _HOST_TIER_INIT
    if not _HOST_TIER_INIT:
        _HOST_TIER_INIT = True
        cap_mib = int(os.environ.get("SGLANG_VLM_HOST_TIER_MB", "0") or 0)
        stats = os.environ.get("SGLANG_VLM_HOST_TIER_STATS_S")
        if (cap_mib > 0 or stats) and not _single_process_engine():
            logger.warning(
                "[vlm-host-tier] off: SGLANG_VLM_HOST_TIER_* is ignored because this "
                "engine runs several processes; the tier needs tp_size=1 and pp_size=1"
            )
            cap_mib, stats = 0, None
        if cap_mib > 0 or stats:
            _HOST_TIER = _VlmHostTier(
                cap_mib << 20,
                os.environ.get("SGLANG_VLM_HOST_TIER_VERIFY", "0") == "1",
                float(stats or 60),
            )
    return _HOST_TIER


def _patch_budget() -> int:
    """Patches one vision-encoder call may carry."""
    from sglang_tpu.graph_runner.vit_tpu_compile_runner import (
        precompiled_max_patch_count,
    )

    return precompiled_max_patch_count()


def _item_patch_grid(item: MultimodalDataItem) -> object | None:
    if item.is_video():
        return item.model_specific_data.get("video_grid_thw")
    if item.is_image():
        return item.model_specific_data.get("image_grid_thw")
    return None


def _item_patch_count(item: MultimodalDataItem) -> int:
    grid = _item_patch_grid(item)
    if grid is None:
        raise ValueError("multimodal item does not expose a vision patch grid")
    return int(torch.as_tensor(grid).prod().item())


def _split_items_by_patch_budget(
    items: List[MultimodalDataItem], budget: int
) -> List[List[MultimodalDataItem]]:
    """Greedily pack items into groups whose total patch count fits the budget."""
    groups: List[List[MultimodalDataItem]] = []
    current: List[MultimodalDataItem] = []
    current_patches = 0
    for item in items:
        patches = _item_patch_count(item)
        if current and current_patches + patches > budget:
            groups.append(current)
            current, current_patches = [], 0
        current.append(item)
        current_patches += patches
    if current:
        groups.append(current)
    return groups


def encode_vision_items(
    data_embedding_func: DataEmbeddingFunc,
    items: List[MultimodalDataItem],
) -> MultimodalEmbedding | List[MultimodalEmbedding] | EVSEmbeddingResult:
    """Encode items as several calls that each fit the precompiled patch budget.

    A batch is concatenated into one encoder call, and TPU precompiles that call
    for a bounded set of patch counts, so a batch above the largest one falls off
    the set and recompiles.
    """
    # Some external processors derive patch geometry from feature tensors and
    # own their batching. Apply this generic split only when every item exposes
    # the grid metadata it requires.
    if any(_item_patch_grid(item) is None for item in items):
        return data_embedding_func(items)

    budget = _patch_budget()
    groups = _split_items_by_patch_budget(items, budget)
    if len(groups) == 1:
        return data_embedding_func(items)

    logger.info(
        "Splitting the vision batch: %d items / %d patches exceed the encoder "
        "patch budget (%d), so it runs as %d calls of %s patches.",
        len(items),
        sum(_item_patch_count(item) for item in items),
        budget,
        len(groups),
        [sum(_item_patch_count(item) for item in group) for group in groups],
    )
    outputs = [data_embedding_func(group) for group in groups]
    if isinstance(outputs[0], list):
        return [embedding for output in outputs for embedding in output]
    return concat_embeddings([flatten_embedding(output) for output in outputs])


def get_embedding_chunk(
    embedding: MultimodalEmbedding,
    extend_prefix_len: int,
    extend_seq_len: int,
    items_offset: List[Tuple[int, int]],
) -> Tuple[MultimodalEmbedding, int, int]:
    start_index, end_index = 0, 0
    extend_start_index = extend_prefix_len
    extend_end_index = extend_prefix_len + extend_seq_len - 1

    for start, end in items_offset:
        if extend_start_index >= start and extend_start_index <= end:
            start_index += extend_start_index - start
        elif extend_start_index > end:
            start_index += end - start + 1

        if extend_end_index >= start and extend_end_index <= end:
            end_index += extend_end_index - start + 1
        elif extend_end_index > end:
            end_index += end - start + 1
    # some models' embedding is 3-dim, reshape it to 2-dim
    embedding = flatten_embedding(embedding)
    embedding_chunk = slice_embedding(embedding, start_index, end_index)
    return embedding_chunk, start_index, end_index


def _flatten_embedding_result(
    embedding: MultimodalEmbedding | List[MultimodalEmbedding],
) -> MultimodalEmbedding:
    if isinstance(embedding, list):
        if not embedding:
            raise ValueError(
                "DataEmbeddingFunc returned an empty per-item list; expected "
                "one entry per input item"
            )
        flat = [flatten_embedding(e) for e in embedding]
        return flat[0] if len(flat) == 1 else concat_embeddings(flat)
    return embedding


def _embedding_token_count(embedding: MultimodalEmbedding) -> int:
    # Vision encoders may return [tokens, hidden] or a higher-rank tensor.  The
    # scheduler always consumes the flattened token dimension.
    return flatten_embedding(embedding).shape[0]


def _get_chunked_embedding_full(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items_per_req: List[MultimodalDataItem],
    items_offset: List[Tuple[int, int]],
    extend_prefix_len: int,
    extend_seq_len: int,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[Optional[MultimodalEmbedding], torch.Tensor]:
    item_hashes = [item.hash for item in embedding_items_per_req]
    embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)
    embedding_per_req = mm_schedule.embedding_cache.get(item_hashes)

    # A compact feature hash can collide for inputs with different token
    # counts.  Never feed a stale cache entry into the scheduler: the length
    # mismatch would otherwise surface much later in _adjust_embedding_length
    # as an unrecoverable prefill crash.
    if embedding_per_req is not None and not isinstance(
        embedding_per_req, EVSEmbeddingResult
    ):
        expected_token_count = sum(end - start + 1 for start, end in items_offset)
        cached_token_count = _embedding_token_count(embedding_per_req.embedding)
        if cached_token_count != expected_token_count:
            _discard_mismatched_cached_embedding(
                embedding_items_hash, expected_token_count, cached_token_count
            )
            embedding_per_req = None

    if embedding_per_req is None:
        _move_items_to_device(embedding_items_per_req, device, data_embedding_func)
        embedding = data_embedding_func(embedding_items_per_req)
        if isinstance(embedding, list):
            # This path caches the combined per-request embedding, so the
            # per-item form is flattened here.
            embedding = _flatten_embedding_result(embedding)
        embedding_per_req = (
            TpuEmbeddingResult(embedding=embedding)
            if isinstance(embedding, (torch.Tensor, PaddedEmbedding))
            else embedding
        )
        mm_schedule.embedding_cache.set(embedding_items_hash, embedding_per_req)
    else:
        _acknowledge_deferred_cuda_ipc_cache_hits(embedding_items_per_req)

    if isinstance(embedding_per_req, EVSEmbeddingResult):
        item = embedding_items_per_req[0]
        input_ids, items_offset = (
            embedding_per_req.redistribute_pruned_frames_placeholders(
                input_ids,
                items_offset,
                item=item,
                extend_prefix_len=extend_prefix_len,
                extend_seq_len=extend_seq_len,
            )
        )

    embedding_per_req_chunk, _, _ = get_embedding_chunk(
        embedding=embedding_per_req.embedding,
        extend_prefix_len=extend_prefix_len,
        extend_seq_len=extend_seq_len,
        items_offset=items_offset,
    )
    return embedding_per_req_chunk, input_ids


def _batch_encode_per_image_misses(
    data_embedding_func: DataEmbeddingFunc,
    per_image_requests: List[PerImageRequestInfo],
    device: torch.device,
) -> Dict[Tuple[Optional[int], int], MultimodalEmbedding]:
    unique_misses: Dict[Tuple[Optional[int], int], Tuple[MultimodalDataItem, int]] = {}
    hash_to_embedding: Dict[Tuple[Optional[int], int], MultimodalEmbedding] = {}
    tier = _host_tier()
    host_verify: Dict[
        Tuple[Optional[int], int], Tuple[MultimodalEmbedding, Optional[_HostEntry]]
    ] = {}

    # Phase 1a: find overlapping items per request and collect cache misses
    for req_info in per_image_requests:
        chunk_start = req_info.extend_prefix_len
        chunk_end = chunk_start + req_info.extend_seq_len  # exclusive
        overlapping = []
        if req_info.extend_seq_len > 0:
            for idx, (item, (start, end)) in enumerate(
                zip(req_info.items, req_info.items_offset)
            ):
                if end >= chunk_start and start < chunk_end:
                    overlapping.append((idx, item, start, end))
        req_info.overlapping = overlapping

        for _idx, item, start, end in overlapping:
            expected_token_count = end - start + 1
            cache_key = (item.hash, expected_token_count)
            if cache_key in hash_to_embedding:
                continue
            cached = mm_schedule.embedding_cache.get_single(item.hash)
            if tier is not None:
                tier.c["dev_hit" if cached is not None else "dev_miss"] += 1
            if cached is not None:
                cached_embedding = cached.embedding
                cached_token_count = _embedding_token_count(cached_embedding)
                if cached_token_count == expected_token_count:
                    hash_to_embedding[cache_key] = cached_embedding
                else:
                    _discard_mismatched_cached_embedding(
                        item.hash, expected_token_count, cached_token_count
                    )
                    unique_misses[cache_key] = (item, expected_token_count)
            elif cache_key in unique_misses or cache_key in host_verify:
                pass
            elif tier is not None and tier.enabled and (
                hit := tier.lookup(cache_key, device)
            ) is not None:
                host_embedding, host_entry = hit
                if tier.verify:
                    # Encode it fresh as a miss as well; compared and replaced below.
                    host_verify[cache_key] = hit
                    unique_misses[cache_key] = (item, expected_token_count)
                else:
                    restored = TpuEmbeddingResult(embedding=host_embedding)
                    tier.note_device_insert(restored.storage_size())
                    mm_schedule.embedding_cache.set(item.hash, restored)
                    hash_to_embedding[cache_key] = host_embedding
            elif cache_key not in unique_misses:
                if (
                    start >= chunk_start
                    and end < chunk_end
                    and item.can_defer_cuda_ipc_feature_reconstruction()
                ):
                    item.model_specific_data[BORROW_CUDA_IPC_FEATURE_KEY] = True
                unique_misses[cache_key] = (item, expected_token_count)

    # Phase 1b: single ViT call for all unique cache misses
    if unique_misses:
        ordered_cache_keys = list(unique_misses.keys())
        miss_items = [unique_misses[key][0] for key in ordered_cache_keys]
        token_counts = [unique_misses[key][1] for key in ordered_cache_keys]

        _move_items_to_device(miss_items, device, data_embedding_func)
        all_miss_embedding = _encode_vision_items(data_embedding_func, miss_items)

        if isinstance(all_miss_embedding, list):
            # Preserve per-image physical buffers when the encoder returns a list.
            assert len(all_miss_embedding) == len(miss_items), (
                f"per-item embedding count {len(all_miss_embedding)} != "
                f"cache-miss item count {len(miss_items)}"
            )
            split_embeddings = [flatten_embedding(emb) for emb in all_miss_embedding]
        else:
            split_embeddings = split_embedding(all_miss_embedding, token_counts)
        for cache_key, emb in zip(ordered_cache_keys, split_embeddings):
            if cache_key in host_verify:
                # Verification: serve the host copy, after comparing it with this fresh encode.
                host_embedding, host_entry = host_verify[cache_key]
                if host_entry is None:
                    tier.verify_pin(cache_key, emb, host_embedding)
                else:
                    tier.verify_pair(cache_key, emb, host_entry)
                restored = TpuEmbeddingResult(embedding=host_embedding)
                tier.note_device_insert(restored.storage_size())
                mm_schedule.embedding_cache.set(cache_key[0], restored)
                hash_to_embedding[cache_key] = host_embedding
                continue
            inserted = TpuEmbeddingResult(embedding=emb)
            mm_schedule.embedding_cache.set(cache_key[0], inserted)
            # Keep a local ref (no extra GPU memory) so assembly never fails due to LRU eviction.
            hash_to_embedding[cache_key] = emb
            if tier is not None:
                tier.c["encodes"] += 1
                tier.note_device_insert(inserted.storage_size())
                if tier.enabled:
                    tier.write_back(cache_key, emb)

    if tier is not None:
        tier.maybe_log()
    return hash_to_embedding


def _assemble_per_image_chunk(
    overlapping: List[Tuple[int, MultimodalDataItem, int, int]],
    hash_to_embedding: Dict[Tuple[Optional[int], int], MultimodalEmbedding],
    extend_prefix_len: int,
    extend_seq_len: int,
) -> Optional[MultimodalEmbedding]:
    if not overlapping:
        return None

    chunk_start = extend_prefix_len
    chunk_end = extend_prefix_len + extend_seq_len  # exclusive

    chunk_slices = []
    for _idx, item, start, end in overlapping:
        cache_key = (item.hash, end - start + 1)
        emb = hash_to_embedding[cache_key]  # shape: (end - start + 1, hidden)
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end - 1)  # inclusive
        local_start = overlap_start - start
        local_end = overlap_end - start + 1  # exclusive for slicing
        chunk_slices.append(slice_embedding(emb, local_start, local_end))

    if len(chunk_slices) == 1:
        return chunk_slices[0]
    return concat_embeddings(chunk_slices)


def _get_chunked_prefill_embedding(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    input_ids: torch.Tensor,
) -> tuple[MultimodalEmbedding | None, torch.Tensor]:
    device = input_ids.device
    # FIXME(Xinyuan): temporary workaround for eagle3
    max_iterations = min(len(items_size) - 1, len(prefix_length))

    # Phase 0: classify requests into per-image vs full/EVS path
    per_image_requests = []  # batched ViT encoding
    full_path_requests = []  # per-request encoding (EVS etc.)
    all_chunks: List[Tuple[int, MultimodalEmbedding]] = []

    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset

        extend_prefix_len = prefix_length[i]
        extend_seq_len = extend_length[i] if i < len(extend_length) else 0
        if extend_seq_len <= 0:
            continue

        # Skip if all items already prefilled.
        if all(offset_end < prefix_length[i] for _, offset_end in items_offset):
            continue

        req_info = PerImageRequestInfo(
            req_idx=i,
            items=embedding_items_per_req,
            items_offset=items_offset,
            extend_prefix_len=extend_prefix_len,
            extend_seq_len=extend_seq_len,
        )

        is_per_image = all(len(item.offsets) == 1 for item in embedding_items_per_req)
        if is_per_image:
            per_image_requests.append(req_info)
        else:
            full_path_requests.append(req_info)

    # Phase 1: batch encode all per-image cache misses in ONE ViT call
    hash_to_embedding: Dict[Tuple[Optional[int], int], MultimodalEmbedding] = {}
    if per_image_requests:
        hash_to_embedding = _batch_encode_per_image_misses(
            data_embedding_func, per_image_requests, device
        )

    # Phase 2: assemble per-request chunks in original request order
    for req_info in per_image_requests:
        chunk = _assemble_per_image_chunk(
            req_info.overlapping,
            hash_to_embedding,
            req_info.extend_prefix_len,
            req_info.extend_seq_len,
        )
        if chunk is not None:
            all_chunks.append((req_info.req_idx, chunk))

    for req_info in full_path_requests:
        chunk_embedding, input_ids = _get_chunked_embedding_full(
            data_embedding_func,
            req_info.items,
            req_info.items_offset,
            req_info.extend_prefix_len,
            req_info.extend_seq_len,
            input_ids,
            device,
        )
        if chunk_embedding is not None:
            all_chunks.append((req_info.req_idx, chunk_embedding))

    # Sort by original request index to maintain correct output order
    all_chunks.sort(key=lambda x: x[0])
    embedding_list = [chunk for _, chunk in all_chunks]

    if len(embedding_list) == 0:
        return None, input_ids
    if len(embedding_list) == 1:
        return embedding_list[0], input_ids
    return concat_embeddings(embedding_list), input_ids


def _get_multimodal_mask(
    input_ids: torch.Tensor,
    placeholder_tensor: torch.Tensor,
    input_ids_cpu: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if input_ids_cpu is not None:
        if input_ids_cpu.device.type != "cpu":
            raise ValueError("MM mask input_ids_cpu must be on CPU")
        num_prefill_tokens = input_ids_cpu.numel()
        if num_prefill_tokens > input_ids.numel():
            raise ValueError("MM CPU token prefix exceeds the input token count")
        mask = torch.zeros(input_ids.shape[0], 1, dtype=torch.bool, device="cpu")
        # Decode tails and bucket padding cannot contain image placeholders.
        mask[:num_prefill_tokens, 0] = torch.isin(
            input_ids_cpu, placeholder_tensor.to("cpu")
        )
        return mask
    return torch.isin(input_ids.to("cpu"), placeholder_tensor.to("cpu")).unsqueeze(-1)


def _adjust_embedding_length(
    embedding: MultimodalEmbedding,
    num_mm_tokens_in_input_ids: int,
    logger,
) -> MultimodalEmbedding:
    num_mm_tokens_in_embedding = embedding.shape[0]
    if num_mm_tokens_in_input_ids != num_mm_tokens_in_embedding:
        logger.warning(
            f"Number of tokens in multimodal embedding does not match those in the input text. "
            f"Got {num_mm_tokens_in_input_ids} tokens in the text but {num_mm_tokens_in_embedding} "
            f"tokens from multimodal embeddings."
        )
        if num_mm_tokens_in_input_ids < num_mm_tokens_in_embedding:
            chunked_prefill_size = get_schedule().chunked_prefill_size
            if chunked_prefill_size != -1:
                logger.warning(
                    "You may want to avoid this issue by raising `chunked_prefill_size`, or disabling chunked prefill"
                )
            # extract from the end: this is a compromise
            if isinstance(embedding, PaddedEmbedding):
                embedding = embedding.slice(
                    num_mm_tokens_in_embedding - num_mm_tokens_in_input_ids,
                    num_mm_tokens_in_embedding,
                )
            elif embedding.dim() == 2:
                embedding = embedding[-num_mm_tokens_in_input_ids:, :]
            else:
                num_multimodal = num_mm_tokens_in_input_ids // embedding.shape[0]
                embedding = embedding[-num_multimodal:, :]
        else:
            raise RuntimeError(
                f"Insufficient multimodal embedding length: {num_mm_tokens_in_input_ids=} vs {num_mm_tokens_in_embedding=}. This is an internal error"
            )
    return embedding


def get_embedding_and_mask(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    input_ids_cpu: Optional[torch.Tensor] = None,
) -> Tuple[MultimodalEmbedding | None, torch.Tensor | None, torch.Tensor]:
    original_input_ids = input_ids
    num_mm_tokens_in_input_ids = _count_mm_tokens_in_extend(
        prefix_length,
        extend_length,
        items_offset_list,
    )

    # 1. Get embedding
    embedding = _get_precomputed_embedding(
        embedding_items, items_size, prefix_length, extend_length, items_offset_list
    )
    if embedding is None:
        embedding, input_ids = _get_chunked_prefill_embedding(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
            input_ids,
        )
        if embedding is None:
            return None, None, input_ids
    # 2. Get mask
    special_multimodal_mask = _get_multimodal_mask(
        input_ids,
        placeholder_tensor,
        input_ids_cpu if input_ids is original_input_ids else None,
    )
    # 3. Adjust embedding length if needed
    if input_ids is not original_input_ids:
        # EVS rewrites placeholder spans after pruning, making the original offsets stale.
        num_mm_tokens_in_input_ids = special_multimodal_mask.sum().item()
    else:
        maybe_assert_sum(
            special_multimodal_mask,
            num_mm_tokens_in_input_ids,
            "MM placeholder count derived from offsets does not match input_ids",
        )
    embedding = _adjust_embedding_length(embedding, num_mm_tokens_in_input_ids, logger)
    return embedding, special_multimodal_mask, input_ids


# Kept as a separate name for the cache-miss encoding boundary.
_encode_vision_items = encode_vision_items
