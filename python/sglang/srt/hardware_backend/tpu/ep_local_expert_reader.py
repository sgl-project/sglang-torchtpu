# SPDX-License-Identifier: Apache-2.0
"""EP-local safetensors reads: each TPU EP rank reads only its own routed experts."""

import json
import logging
import os
import re
import struct
import threading
from typing import Dict, List, Optional, Set, Tuple

import safetensors
import torch

from sglang.srt.utils import is_tpu, log_info_on_rank0

logger = logging.getLogger(__name__)

_is_tpu = is_tpu()

# Bound on a single readinto() call while filling a span buffer.
_READ_CHUNK_BYTES = 64 * 1024 * 1024
# Kept tensors at most this far apart are read with one seek + read.
_SPAN_MERGE_GAP_BYTES = 16 * 1024 * 1024
_EXPERT_ID_KEY_RE = re.compile(r"^(.*)\.experts\.(\d+)\.")

_SAFETENSORS_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "U16": torch.uint16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "I64": torch.int64,
    "U64": torch.uint64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E4M3FNUZ": torch.float8_e4m3fnuz,
    "F8_E5M2FNUZ": torch.float8_e5m2fnuz,
    "C64": torch.complex64,
}
if hasattr(torch, "float8_e8m0fnu"):
    _SAFETENSORS_DTYPES["F8_E8M0"] = torch.float8_e8m0fnu


def tpu_local_routed_expert_range(model: torch.nn.Module) -> Optional[Tuple[int, int]]:
    """Routed-expert ids [start, end) this EP rank stores, for EP-local reads.

    For models that set ``supports_tpu_ep_local_expert_reads`` (every
    ``.experts.<id>.`` checkpoint key loads into FusedMoE expert ``<id>``), the
    range is ``local_routed_expert_range`` of their FusedMoE layers. None (read
    everything) off TPU, without EP, with EPLB metadata or elastic EP join, when
    the layers disagree, when a layer also loads ids outside its routed range
    (fused shared experts, parameters that need every expert), or when
    SGLANG_TPU_DISABLE_EP_LOCAL_EXPERT_READS is true.
    """
    if not _is_tpu or not getattr(model, "supports_tpu_ep_local_expert_reads", False):
        return None
    from sglang.srt.environ import envs

    def whole_shards(reason: str) -> None:
        # An opted-in model reading whole shards can need far more host memory.
        log_info_on_rank0(
            logger, f"EP-local expert reads off ({reason}); reading whole shards."
        )

    if envs.SGLANG_TPU_DISABLE_EP_LOCAL_EXPERT_READS.get():
        return whole_shards("SGLANG_TPU_DISABLE_EP_LOCAL_EXPERT_READS")
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.utils import local_routed_expert_range
    from sglang.srt.runtime_context import get_exec

    if get_global_expert_location_metadata() is not None:
        return whole_shards("expert location metadata")
    if get_exec().moe.ep_join_mode == "scale":
        return whole_shards("elastic EP join")
    moes = [m for m in model.modules() if isinstance(m, FusedMoE)]
    if any(
        m._has_fused_shared
        or any(
            getattr(p, "_sglang_require_global_experts", False) for p in m.parameters()
        )
        for m in moes
    ):
        return whole_shards("a MoE layer loads experts outside its routed range")
    layouts = {
        (m.moe_ep_size, m._expert_storage_rank, m._num_local_routed) for m in moes
    }
    if len(layouts) != 1:
        return whole_shards("MoE layers disagree on the expert layout")
    ep_size, storage_rank, num_local_routed = layouts.pop()
    if ep_size <= 1:
        return None
    return local_routed_expert_range(storage_rank, num_local_routed)


class EpLocalExpertReader:
    """Reads safetensors shards skipping routed experts outside one EP rank's range.

    Only files listed in the checkpoint index are filtered (others, e.g. an MTP
    shard added beside the index, are read whole). ``verify()`` checks that every
    local expert the index lists was read.
    """

    def __init__(self, hf_weights_files: List[str], expert_range: Tuple[int, int]):
        self.expert_range = expert_range
        index_path = os.path.join(
            os.path.dirname(hf_weights_files[0]), "model.safetensors.index.json"
        )
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        self.indexed_files = set(weight_map.values())
        # Index entries for this rank's routed experts: all must be read.
        self.expected: Set[str] = {
            name for name in weight_map if self._local_expert_id(name) not in (None, -1)
        }
        self.seen: Set[str] = set()
        self._seen_lock = threading.Lock()

    @classmethod
    def maybe_create(
        cls, hf_weights_files: List[str], expert_range: Optional[Tuple[int, int]]
    ) -> Optional["EpLocalExpertReader"]:
        if expert_range is None or not hf_weights_files:
            return None
        index_path = os.path.join(
            os.path.dirname(hf_weights_files[0]), "model.safetensors.index.json"
        )
        if not os.path.isfile(index_path):
            return None
        log_info_on_rank0(
            logger,
            f"Reading only this EP rank's routed experts: ids [{expert_range[0]}, "
            f"{expert_range[1]}) on rank 0.",
        )
        return cls(hf_weights_files, expert_range)

    def _local_expert_id(self, name: str) -> Optional[int]:
        m = _EXPERT_ID_KEY_RE.match(name)
        if m is None:
            return None
        eid = int(m.group(2))
        return eid if self.expert_range[0] <= eid < self.expert_range[1] else -1

    def handles(self, st_file: str) -> bool:
        return os.path.basename(st_file) in self.indexed_files

    def read(self, st_file: str) -> Dict[str, torch.Tensor]:
        with open(st_file, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
            data_base = 8 + header_len
            header.pop("__metadata__", None)
            data_end = max((m["data_offsets"][1] for m in header.values()), default=0)
            file_size = os.fstat(f.fileno()).st_size
            if file_size < data_base + data_end:
                raise IOError(
                    f"{st_file} is truncated: its header needs "
                    f"{data_base + data_end} bytes, the file has {file_size}."
                )
            kept, unviewable = [], []
            for name, meta in header.items():
                if self._local_expert_id(name) == -1:
                    continue
                if meta["dtype"] not in _SAFETENSORS_DTYPES:
                    unviewable.append(name)
                    continue
                start, end = meta["data_offsets"]
                kept.append((start, end, name, meta))
            kept.sort(key=lambda item: item[0])

            spans = []
            for item in kept:
                if not spans or item[0] - spans[-1][1] > _SPAN_MERGE_GAP_BYTES:
                    spans.append([item[0], item[1], [item]])
                else:
                    spans[-1][1] = max(spans[-1][1], item[1])
                    spans[-1][2].append(item)

            result = {}
            for span_start, span_end, items in spans:
                f.seek(data_base + span_start)
                buf = bytearray(span_end - span_start)
                view, pos = memoryview(buf), 0
                while pos < len(buf):
                    n = f.readinto(view[pos : pos + _READ_CHUNK_BYTES])
                    if not n:
                        raise IOError(f"{st_file} shrank while being read")
                    pos += n
                for start, end, name, meta in items:
                    dtype = _SAFETENSORS_DTYPES[meta["dtype"]]
                    if end == start:
                        result[name] = torch.empty(meta["shape"], dtype=dtype)
                        continue
                    result[name] = torch.frombuffer(
                        buf,
                        dtype=dtype,
                        count=(end - start) // dtype.itemsize,
                        offset=start - span_start,
                    ).view(meta["shape"])
        if unviewable:
            # Dtypes missing from _SAFETENSORS_DTYPES are read through safetensors
            # (mmap).
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as sf:
                for name in unviewable:
                    result[name] = sf.get_tensor(name)
        with self._seen_lock:  # read() runs on loader threads
            self.seen.update(result)
        return result

    def verify(self) -> None:
        missing = sorted(self.expected - self.seen)
        if missing:
            raise RuntimeError(
                f"EP-local expert read missed {len(missing)} local expert tensors "
                f"listed in the index, e.g. {missing[:3]}."
            )
