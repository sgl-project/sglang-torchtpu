# SPDX-License-Identifier: Apache-2.0
"""V4.1 adapters for SGLang's existing host Engram configuration.

Why: Engram construction and hashing have no backend factory; the GPU host
     table uses CUDA registration and device-driven gathers.
How: AROUND host mapping, construction, hash/lookup, history and worker entry;
     AFTER runner initialization binds a host coordinator. The TPU compile
     policy records the padded shape; the model stages rows at each segment.
     AROUND the device gate uses bitcast/select for the signed square root.
Upstream PR or issue: not filed (local prototype of the Engram backend seam).
Remove when: upstream exposes Engram factories and a host-input preparation seam.
"""

from __future__ import annotations

import mmap

import torch


def _enabled() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get()


def init_host_table(original, self, layout, nbytes, name, group):
    if not _enabled():
        return original(self, layout, nbytes, name, group)
    if layout not in ("shared", "per_rank"):
        raise ValueError(
            f"Invalid SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT={layout!r}; "
            "expected 'shared' or 'per_rank'"
        )
    self.layout, self.nbytes, self.group = layout, nbytes, group
    self.dirty = False
    shared = layout == "shared"
    # Reuse SGLang's memfd ownership and PID-namespace contract. CPU gathers
    # read this mapping directly; only the packed rows are uploaded to TPU.
    self.fd = self._open_shared_fd(nbytes, name) if shared else None
    self.mm = mmap.mmap(
        self.fd if shared else -1,
        nbytes,
        flags=mmap.MAP_SHARED if shared else mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    if hasattr(mmap, "MADV_HUGEPAGE"):
        self.mm.madvise(mmap.MADV_HUGEPAGE)
    self.bytes = torch.frombuffer(self.mm, dtype=torch.uint8)
    if shared:
        group.barrier()


def _validate_parallel(parallel):
    if any(
        getattr(parallel, name) != 1
        for name in ("pp_size", "attn_cp_size", "attn_dcp_size")
    ):
        raise NotImplementedError("Host Engram requires PP1/CP1/DCP1")


def init_embedding(original, self, num_embeddings, dim, layer_id):
    if not _enabled():
        return original(self, num_embeddings, dim, layer_id)
    from sglang.srt.runtime_context import get_parallel

    # Reject unsupported layouts before allocating the large host tables.
    _validate_parallel(get_parallel())
    from sglang.srt.environ import envs

    if envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get() != "shared":
        raise ValueError("V4.1 DP-local lookup requires one shared full host table")
    with torch.device("cpu"):
        # The core constructor owns layout selection, TP row ownership and
        # checkpoint loading. Only its host mapping needs a backend adapter.
        original(self, num_embeddings, dim, layer_id)
    self.register_buffer("_tpu_engram_packed", None, persistent=False)
    self._tpu_host_engram = True


def init_hasher(original, self, *args, **kwargs):
    if not _enabled():
        return original(self, *args, **kwargs)
    with torch.device("cpu"):
        original(self, *args, **kwargs)
    self._tpu_host_engram = True


def init_history(original, self, num_req_slots, device):
    if getattr(self, "_tpu_host_engram", False):
        device = "cpu"
    return original(self, num_req_slots, device)


def prepare_history(original, self, batch):
    if getattr(self.engram_hasher, "_tpu_host_engram", False):
        # The host snapshot contains the full lookback of this scheduled chunk.
        return None
    return original(self, batch)


def hash_forward(original, self, input_ids, forward_batch):
    if not getattr(self, "_tpu_host_engram", False):
        return original(self, input_ids, forward_batch)
    # Lookups were completed before graph entry. Only the shape is consumed
    # by EngramEmbedding; no int64 hash or history mutation enters the graph.
    return input_ids[:, None, None].expand(
        -1, self.multipliers.shape[0], self.offsets.shape[1]
    )


def embedding_forward(
    original, self, indices, forward_batch=None, *, cp_all_tokens=False
):
    if not getattr(self, "_tpu_host_engram", False):
        return original(self, indices, forward_batch, cp_all_tokens=cp_all_tokens)
    from sglang_tpu.engram import unpack_rows

    if cp_all_tokens:
        raise NotImplementedError(
            "Host Engram does not yet support context parallelism"
        )
    if self._tpu_engram_packed is None:
        raise RuntimeError("Engram rows must be staged before model forward")
    values = unpack_rows(
        self._tpu_engram_packed,
        tokens=indices.shape[0],
        heads=indices.shape[1],
        dim=self.dim,
    )
    return values


def gate_forward(original, x, kv, q_weight, k_weight, eps, clamp_value):
    from sglang_tpu.engram import engram_gate

    return engram_gate(x, kv, q_weight, k_weight, eps, clamp_value)


def bind_coordinator(result, runner):
    hasher = runner.ngram_embedding_manager.engram_hasher
    if not getattr(hasher, "_tpu_host_engram", False):
        return
    from sglang.srt.layers.engram import Engram
    from sglang.srt.runtime_context import get_parallel
    from sglang_tpu.engram import HostEngramCoordinator

    parallel = get_parallel()
    _validate_parallel(parallel)
    if runner.server_args.nnodes != 1:
        raise NotImplementedError("Initial Flash validation requires one host")
    embeddings = [
        (module.layer_hash_index, module.embed)
        for module in runner.model.modules()
        if isinstance(module, Engram)
    ]
    coordinator = HostEngramCoordinator(hasher, embeddings, runner.device)
    runner.host_engram = coordinator
    runner.model.model.host_engram = coordinator
    if not runner.server_args.enable_torch_compile:
        previous = getattr(runner.model, "prepare_forward_batch", None)

        def prepare(forward_batch):
            if previous is not None:
                previous(forward_batch)
            coordinator.begin_step(forward_batch.input_ids.shape[0])

        runner.model.prepare_forward_batch = prepare


def worker_forward(original, worker, batch, *args, **kwargs):
    coordinator = getattr(worker.model_runner, "host_engram", None)
    if coordinator is None:
        return original(worker, batch, *args, **kwargs)
    if batch is None:
        raise NotImplementedError("Host Engram needs a scheduled request batch")
    requests, positions = coordinator.prepare(batch)
    try:
        result = original(worker, batch, *args, **kwargs)
        coordinator.remember(requests, positions, result)
        return result
    finally:
        coordinator.clear_step()


def hook_specs():
    # Do not import the GPU Engram module for unrelated models when disabled.
    if not _enabled():
        return []
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.layers.engram._HostTable.__init__",
            init_host_table,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramEmbedding.__init__",
            init_embedding,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramEmbedding.forward",
            embedding_forward,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.engram_gate",
            gate_forward,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramHasher.__init__",
            init_hasher,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramHasher.init_history",
            init_history,
            HookType.AROUND,
        ),
        (
            "sglang.srt.layers.engram.EngramHasher.forward",
            hash_forward,
            HookType.AROUND,
        ),
        (
            (
                "sglang.srt.model_executor.model_runner_components."
                "ngram_embedding_manager.NgramEmbeddingManager._prepare_engram_history"
            ),
            prepare_history,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.model_runner.ModelRunner.init_ngram_embedding_manager",
            bind_coordinator,
            HookType.AFTER,
        ),
        (
            "sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation",
            worker_forward,
            HookType.AROUND,
        ),
    ]
