# SPDX-License-Identifier: Apache-2.0
"""Serving side of the K3 megakernel: per-rank weights on device, the exported
multi-device Pallas calls, and the per-step inputs taken from sglang's forward
batch and pools.

The kernel reads and writes sglang's own per-layer pools in place (KDA conv /
recurrent state rows, MLA latent pages): torch_tpu copies donated inputs of a
multi-device custom call in full every call, while plain inputs are passed by
reference, so the pools go in as plain inputs and are written through.

Two kernel variants share the weights: decode (one token) and the token loop
for bs=1 prefill chunks of up to R tokens: rows mode (R-token blocks through every stage,
``StackKernel(prefill_blocks)``) when the kernel has it, else the token loop (one stack pass per token).
"""

import contextlib
import inspect
import logging
import os
import time

import ml_dtypes
import msgspec
import numpy as np

logger = logging.getLogger(__name__)

_PAGE_TABLE_ALIGN = 16  # the kernel streams pages in 16-page blocks
# One runtime per process (one TP rank); the custom op looks it up here.
_RUNTIMES: list = []


def _cpu_tensor(a: np.ndarray):
    import torch

    a = np.ascontiguousarray(a)
    if a.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16)
    if a.dtype == ml_dtypes.float8_e4m3fn:
        return torch.from_numpy(a.view(np.uint8)).view(torch.float8_e4m3fn)
    return torch.from_numpy(a)


def _to_device(a: np.ndarray):
    return _cpu_tensor(a).to("tpu")


def _prefetch_dense(
    model_path: str,
    num_layers: int,
    lane: int,
    lanes: int,
    threads: int = 16,
    chunk: int = 32 << 20,
) -> float:
    """Pull this lane's share of the dense checkpoint tensors into the page cache (memmap
    faults over NFS run at ~25 MB/s per reader). Returns GiB read."""
    from concurrent.futures import ThreadPoolExecutor

    from sglang_tpu.k3_megakernel.weights import tensor_ranges

    jobs = [
        (p, o, min(chunk, off + size - o))
        for p, off, size in tensor_ranges(model_path, num_layers)
        for o in range(off, off + size, chunk)
    ][lane::lanes]

    def read(job):
        path, offset, size = job
        fd = os.open(path, os.O_RDONLY)
        try:
            return len(os.pread(fd, size, offset))
        finally:
            os.close(fd)

    with ThreadPoolExecutor(threads) as pool:
        return sum(pool.map(read, jobs)) / 2**30


def _stream_experts_to_device(model_path: str, spec, rank: int, procs: int) -> dict:
    """Routed-expert families ``[moe_layers, expert_slots, ...]`` on device. Layers stream from
    ``weights.stream_expert_layers`` (no persistent cache) into one host array per family, which is
    uploaded whole: under torch_tpu's deferred eager mode, per-layer slice copies into a device
    tensor materialized as one graph needing a second full-size buffer (13L TP8: 27.6 GB OOM)."""
    import torch

    from sglang_tpu.k3_megakernel.config import MoeLayout
    from sglang_tpu.k3_megakernel.weights import stream_expert_layers

    layout = MoeLayout(spec.dims, spec.topo, replicas=spec.relax.expert_replicas)
    layers = range(1, spec.num_layers)
    host = None
    t0 = time.time()
    for i, arrays in stream_expert_layers(
        model_path, layout, layers, rank, procs=procs
    ):
        if host is None:
            host = {
                n: np.empty((len(layers),) + a.shape, a.dtype)
                for n, a in arrays.items()
            }
        for n, a in arrays.items():
            host[n][i] = a
        if rank == 0 and ((i + 1) % 10 == 0 or i + 1 == len(layers)):
            logger.info(
                "K3 megakernel: experts of %d/%d MoE layers read (%.0fs).",
                i + 1,
                len(layers),
                time.time() - t0,
            )
    out = {}
    for n in sorted(host):
        out[n] = _to_device(host.pop(n))
        torch.accelerator.synchronize()
    return out


# HBM left for sglang after the megakernel weights: non-layer weights (embed / lm_head shards, vision
# tower), the bs=1 KV and mamba pools, activations. Arbitrary; 13L TP8 used ~1 GB for the pools.
_HBM_MARGIN = 6 << 30


def _check_hbm(spec, dense_bytes: int, rank: int) -> None:
    """Fail before the (long) expert stream when the kernel weights cannot fit on this core."""
    from sglang.srt.platforms import current_platform
    from sglang_tpu.k3_megakernel.config import MoeLayout
    from sglang_tpu.k3_megakernel.weights import _expert_family_shapes

    layout = MoeLayout(spec.dims, spec.topo, replicas=spec.relax.expert_replicas)
    experts = sum(
        int(np.prod(shape)) * np.dtype(dt).itemsize
        for shape, dt in _expert_family_shapes(layout, 1, spec.num_layers - 1).values()
    )
    free, total = current_platform.get_available_memory()
    need = dense_bytes + experts
    msg = (
        f"K3 megakernel rank {rank}: weights need {need / 2**30:.1f} GiB (dense {dense_bytes / 2**30:.1f}, "
        f"experts {experts / 2**30:.1f}), HBM free {free / 2**30:.1f} of {total / 2**30:.1f} GiB"
    )
    if need + _HBM_MARGIN > free:
        raise RuntimeError(
            msg + f"; need {_HBM_MARGIN / 2**30:.0f} GiB more for sglang. Try "
            "SGLANG_TPU_OPT_K3_MEGAKERNEL_RELAX=dense_fp8 or a smaller model view."
        )
    if rank == 0:
        logger.info(msg)


@contextlib.contextmanager
def _host_slot(lane: int, slots: int):
    """Host-wide semaphore over the ranks of one host: lanes sharing ``lane % slots`` take turns."""
    import fcntl

    with open(f"/dev/shm/k3mk_serve_load_{lane % slots}.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextlib.contextmanager
def _eager_now():
    """Run each torch_tpu op as it comes (sglang serves with DEFER_AND_FUSE, which fuses the
    weight uploads into graphs that hold extra full-size copies)."""
    import torch
    from torch_tpu._internal import execution_mode

    old = execution_mode.eager_mode
    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_NEVER
    try:
        yield
    finally:
        torch.accelerator.synchronize()
        execution_mode.eager_mode = old


# StackKernel options serving accepts from SGLANG_TPU_K3_MEGAKERNEL_KERNEL_OPTS: schedule and
# exact / f32-reorder options validated end to end. Everything else is benchmark-only (debug
# skips, phase markers, test outputs), set by the runtime (rows, captures, collective id, head)
# or known broken (moe_shared_after), and raises.
SERVING_KERNEL_OPTIONS = frozenset(
    {
        "wire_relax",
        "prefetch_attn",
        "prefetch_moe",
        "moe_prefetch_late",
        "handoff",
        "attn_region",
        "attn_ring_mib",
        "attn_chunk_mib",
        "oproj_early",
        "oproj_full_gather",
        "stream_budget",
        "stream_slots",
        "stream_budgets",
        "moe_stream_slots",
        "mixture_tree",
        "mixture_fused_score",
        "mixture_early_scores",
        "mla_w1_split",
        "mla_w1_chunk",
        "mla_early_kv",
        "mla_async_write",
        "mla_kv_window",
        "mla_kv_block_pages",
        "kda_mxu",
        "kda_fast_core",
        "gather_bf16",
        "flat_gather",
        "row_gathers",
        "replica_balance",
        "moe_fast_topk",
        "moe_early_dma",
        "moe_pair_down",
        "moe_overlap_pair",
        "moe_ring",
        "moe_fill_vmem",
        "moe_late_handoff",
        "moe_split_refill",
        "moe_region_m",
        "moe_up_fill",
        "prefetch_latent_up",
        "moe_up_tree",
    }
)

# Tokens per rows-mode prefill block (StackKernel(verify_rows=8, verify_mode="rows")).
_PREFILL_BLOCK = 8


def prefill_rows(chunked_prefill_size: int | None, prefill_mode: str) -> int:
    """Kernel rows of a prefill chunk: the chunked-prefill size, which caps every extend
    sglang schedules for one request (mixed chunks are rejected at startup). Without a positive
    chunk size an extend could hold a whole prompt, so it is rejected instead of being cut."""
    if chunked_prefill_size is None or chunked_prefill_size <= 0:
        raise ValueError(
            "SGLANG_TPU_ENABLE_K3_DECODE_MEGAKERNEL=1 needs --chunked-prefill-size > 0 (the kernel's "
            f"prefill rows; got {chunked_prefill_size})"
        )
    if prefill_mode not in ("auto", "rows", "loop"):
        raise ValueError(
            f"SGLANG_TPU_K3_MEGAKERNEL_PREFILL must be auto, rows or loop (got {prefill_mode!r})"
        )
    if prefill_mode == "rows" and chunked_prefill_size % _PREFILL_BLOCK:
        raise ValueError(
            f"SGLANG_TPU_K3_MEGAKERNEL_PREFILL=rows needs --chunked-prefill-size a multiple of "
            f"{_PREFILL_BLOCK} (got {chunked_prefill_size})"
        )
    return chunked_prefill_size


def kernel_options(text: str, *, allowed: frozenset | None = None) -> dict:
    """``"key=value;..."`` -> StackKernel keywords (values as Python literals, else strings)."""
    import ast

    from sglang_tpu.k3_megakernel.kernel.stack import StackKernel

    kw = {}
    for item in filter(None, (x.strip() for x in text.split(";"))):
        key, _, value = item.partition("=")
        try:
            kw[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kw[key] = value
    accepted = inspect.signature(StackKernel.__init__).parameters
    unknown = [
        k
        for k in kw
        if k not in accepted or k in ("token_rows", "verify_rows", "prefill_blocks")
    ]
    if unknown:
        raise ValueError(
            f"SGLANG_TPU_K3_MEGAKERNEL_KERNEL_OPTS: StackKernel does not take {unknown}"
        )
    if allowed is not None:
        refused = sorted(k for k in kw if k not in allowed)
        if refused:
            raise ValueError(
                f"SGLANG_TPU_K3_MEGAKERNEL_KERNEL_OPTS: {refused} are benchmark-only, set by the "
                f"runtime or unsupported in serving; serving accepts {sorted(allowed)}"
            )
    return kw


class K3MegakernelRuntime:
    """One per TP rank. Built after sglang's (non-layer) weights load; ``forward`` runs a bs=1
    decode step or a bs=1 prefill chunk through the whole decoder stack."""

    def __init__(
        self,
        *,
        model_path: str,
        num_layers: int,
        mla_layers: tuple,
        tp_rank: int,
        tp_size: int,
        nnodes: int,
        prefill_rows: int,
        prefill_mode: str,
        verify_mode: str,
        kernel_opts: str,
        expert_procs: int,
        load_slots: int,
    ):
        from sglang_tpu.k3_megakernel.config import (
            K3MK_COLLECTIVE_ID,
            Relax,
            Topology,
        )
        from sglang_tpu.k3_megakernel.kernel.stack import (
            StackSpec,
            input_names,
        )
        from sglang_tpu.k3_megakernel.weights import Checkpoint, load_stack

        lanes = tp_size // max(nnodes, 1)
        self.topo = Topology(hosts=tp_size // lanes, lanes=lanes)
        self.rank, self.world = tp_rank, tp_size
        self.spec = StackSpec(
            self.topo, num_layers, tuple(mla_layers), relax=Relax.from_env()
        )
        self.opts = kernel_options(kernel_opts, allowed=SERVING_KERNEL_OPTIONS)
        self.prefill_mode, self.verify_mode = prefill_mode, verify_mode
        # The sglang path embeds and samples itself.
        self.opts.update(fused_head=False, collective_id=K3MK_COLLECTIVE_ID)
        # Kernel variants, built on first use: (kind, captured layers) -> index into ``self.variants``
        # (the custom op's ``variant`` argument). Kinds: "decode", "prefill", "verify".
        self.variants: list[_Variant] = []
        self._variant_index: dict = {}
        self.prefill_rows = prefill_rows
        self.prefill_block = (
            _prefill_kernel(self.spec, prefill_rows, {}, prefill_mode)[1]
            if prefill_rows > 1
            else 0
        )
        self.names = input_names(self.spec)
        self.kda_layers, self.mla_layers = self.spec.kda_layers, self.spec.mla_layers

        t0 = time.time()
        gib = _prefetch_dense(model_path, num_layers, tp_rank % lanes, lanes)
        _barrier()
        t1 = time.time()
        weights = load_stack(
            Checkpoint(model_path), self.spec, [tp_rank], routed_experts=False
        )
        _check_hbm(self.spec, sum(v[0].nbytes for v in weights.values()), tp_rank)
        with _eager_now():
            self.weights = {k: _to_device(v[0]) for k, v in weights.items()}
            del weights
            t2 = time.time()
            # Each rank holds its whole expert set in host memory before the upload (93L TP32: ~58 GB);
            # at most ``load_slots`` ranks of a host do so at once.
            with _host_slot(tp_rank % lanes, load_slots):
                self.weights.update(
                    _stream_experts_to_device(
                        model_path, self.spec, tp_rank, expert_procs
                    )
                )
            self.tables = _to_device(np.asarray(self.spec.tables(), np.int32))
        self._calls: dict = {}
        _RUNTIMES.append(self)
        self.index = len(_RUNTIMES) - 1
        _register_op()
        self.prepare()
        logger.info(
            "K3 megakernel: rank %d/%d, %s, %d layers (MLA %s), relax %s, kernel options %s, prefill "
            "rows %d (block %d); weights on device: prefetch %.1f GiB %.0fs, dense %.0fs, experts %.0fs.",
            tp_rank,
            tp_size,
            self.topo,
            num_layers,
            mla_layers,
            self.spec.relax,
            self.opts,
            prefill_rows,
            self.prefill_block,
            gib,
            t1 - t0,
            t2 - t1,
            time.time() - t2,
        )

    # ----- the exported calls -----
    def prepare(self, captures: tuple = (), verify_rows: int = 0) -> None:
        """Build the kernel variants serving needs, outside any traced forward (dynamo cannot build
        them): decode and prefill, and with DSpark the capturing prefill and verify."""
        self._add_variant("decode", ())
        if self.prefill_rows > 1:
            self._add_variant("prefill", ())
        if captures:
            self._add_variant("prefill", captures)
            self._add_variant("verify", captures, verify_rows)

    def variant(self, kind: str, captures: tuple, rows: int = 0) -> int:
        key = (kind, captures, rows if kind == "verify" else 0)
        if key not in self._variant_index:
            raise RuntimeError(
                f"K3 megakernel: no {key} kernel variant prepared (have {list(self._variant_index)})"
            )
        return self._variant_index[key]

    def _add_variant(self, kind: str, captures: tuple, rows: int = 0) -> None:
        from sglang_tpu.k3_megakernel.kernel.stack import StackKernel

        key = (kind, captures, rows)
        if key not in self._variant_index:
            cap = (
                dict(capture_layers=captures, capture_mixture=True) if captures else {}
            )
            if kind == "decode":
                kernel, block = StackKernel(self.spec, **self.opts), 0
            elif kind == "prefill":
                kernel, block = _prefill_kernel(
                    self.spec,
                    self.prefill_rows,
                    dict(self.opts, **cap),
                    self.prefill_mode,
                )
            else:  # verify: ``rows`` rows, one KDA checkpoint per row
                # verify_mode "loop": one stack pass per row (bit-identical to decode steps; parity checks).
                loop = self.verify_mode == "loop"
                kernel = StackKernel(
                    self.spec,
                    verify_rows=rows,
                    verify_mode="loop" if loop else "rows",
                    **cap,
                    **self.opts,
                )
                block = 0 if loop else rows
            self._variant_index[key] = len(self.variants)
            self.variants.append(
                _Variant(kind=kind, kernel=kernel, block=block, captures=captures)
            )

    def _build(self, variant: int, args):
        import jax
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P
        from torch_tpu._internal.pallas import get_pallas_mesh
        from torch_tpu._internal.pallas import pallas as pallas_impl

        from sglang_tpu import pallas_jax_op as pjo
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        v = self.variants[variant]
        kernel = v.kernel
        shapes = {
            n: (tuple(a.shape), _jax_dtype(a.dtype)) for n, a in zip(self.names, args)
        }
        _check_pools(kernel, shapes)
        with tpu_info_for_cpu_export():
            call, n_out = kernel.pallas_call(shapes, alias_states=False)
        mesh = get_pallas_mesh(("x",), (self.world,))
        spec = P("x")
        sharded = jax.shard_map(
            lambda *xs: call(*xs),
            mesh=mesh,
            in_specs=(spec,) * len(self.names),
            out_specs=(spec,) * n_out,
            check_vma=False,
        )

        def fn(*xs: jax.Array):
            return sharded(*xs)

        out_sh = (NamedSharding(mesh, spec),) * n_out
        jit_fn = jax.jit(fn, keep_unused=True, out_shardings=out_sh)
        x0 = shapes["x0"][0]
        tag = (
            f"k3mk_{v.kind}_L{self.spec.num_layers}_x{'x'.join(map(str, x0))}_c{len(v.captures)}"
            f"_p{shapes['pages'][0][0]}"
        )
        trace_key = pallas_impl._get_kernel_invocation_key(
            tag, [], {"relax": str(self.spec.relax)}
        )
        callable_ = pjo._CpuExportJaxCallable(
            name=tag,
            jit_fn=jit_fn,
            trace_key=trace_key,
            mesh=mesh,
            input_partition_specs=(spec,) * len(self.names),
            static_argnums=(),
            donate_argnums=[],
        )
        callable_.sglang_output_shardings = out_sh
        logger.info("K3 megakernel: built %s", tag)
        return callable_

    def run(self, variant: int, args):
        """The custom op's implementation (also traced once by torch_tpu under compile): (final rows,
        captures)."""
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        key = (variant, tuple(args[self.names.index("pages")].shape))
        if key not in self._calls:
            self._calls[key] = self._build(variant, args)
        with tpu_info_for_cpu_export():
            outs = self._calls[key](*args)
        return outs[0], outs[1]

    # ----- one step -----
    def forward(self, model, input_ids, forward_batch, inputs_embeds=None):
        """bs=1 decode (one token), bs=1 extend (``T`` padded tokens, at most R real ones) or bs=1
        target verify (W rows, one KDA checkpoint per row): ``[T, H]`` final hidden states, plus the
        packed DSpark captures ``[T, K * H]`` when the model captures layers."""
        import torch
        import torch.nn.functional as F

        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        full_md = backend.full_attn_backend.forward_metadata
        lin = backend.linear_attn_backend
        pages = full_md.page_indices.to(torch.int32)
        pad = (-pages.shape[0]) % _PAGE_TABLE_ALIGN
        if pad:
            pages = F.pad(pages, (0, pad))
        mode = forward_batch.forward_mode
        captures = tuple(sorted(model.dspark_layers_to_capture or ()))
        kv_len = full_md.kv_lens[0].to(torch.int32)
        slot = (
            lin._state_indices[0].to(torch.int32).clamp(min=0)
        )  # padded / warmup rows carry -1
        read, base = kda_state_rows(
            backend.req_to_token_pool.mamba_pool,
            mode,
            slot,
            forward_batch.req_pool_indices[:1].to(slot.device),
        )
        x0 = (
            model.embed_tokens(input_ids)
            if inputs_embeds is None
            else inputs_embeds.to(torch.bfloat16)
        )
        t = x0.shape[0]
        zero = torch.zeros_like(slot)
        # meta = [kv_len of row 0, KDA read row, KDA write base, rows, write stride, reset]
        if mode.is_decode():
            # DSpark serves only extend and verify; a decode (warmup) gets zero captures.
            variant = self.variant("decode", ())
            meta = torch.stack([kv_len, read, base, zero + 1, zero, zero])
        elif mode.is_target_verify():
            variant = self.variant("verify", captures, t)
            meta = torch.stack([kv_len - t + 1, read, base, zero + t, zero + 1, zero])
            if not self.variants[variant].block:
                x0 = x0[:, None]
        else:
            variant = self.variant("prefill", captures)
            rows = self.prefill_rows
            # Real query rows of request 0 (the batch is padded to a token bucket). sglang caps each
            # extend at --chunked-prefill-size == rows (prefill_rows() and the startup checks), so
            # the clamp and the x0[:rows] slice below only drop padding, never real tokens.
            n = (
                (full_md.cu_q_lens[1] - full_md.cu_q_lens[0])
                .to(torch.int32)
                .clamp(max=rows)
            )
            prefix = kv_len - n
            # A fresh request starts KDA from zero, as sglang's has_initial_state = prefix > 0.
            meta = torch.stack(
                [prefix + 1, read, base, n, zero, (prefix == 0).to(torch.int32)]
            )
            x0 = F.pad(x0, (0, 0, 0, rows - t)) if t < rows else x0[:rows]
            blk = self.variants[variant].block
            x0 = x0.reshape(rows // blk, blk, -1) if blk else x0[:, None]
        pools = backend.req_to_token_pool
        kv_pool = backend.token_to_kv_pool
        state = {}
        for k, layer in enumerate(self.kda_layers):
            cache = pools.mamba2_layer_cache(layer)
            state[f"kda_conv_{k}"] = cache.conv[0]
            state[f"kda_rec_{k}"] = cache.temporal
        for j, layer in enumerate(self.mla_layers):
            state[f"mla_cache_{j}"] = kv_pool.get_fused_kv_buffer(layer)
        inputs = dict(
            tables=self.tables, meta=meta, pages=pages, x0=x0, **self.weights, **state
        )
        out, cap = torch.ops.k3mk.stack(
            [inputs[n] for n in self.names], self.index, variant
        )
        out = _fit_rows(out.reshape(-1, out.shape[-1]), t)
        if not captures:
            return out
        k = len(captures)
        if mode.is_decode():
            aux = out.new_zeros((t, k * out.shape[-1]))
        else:
            aux = _fit_rows(
                cap.reshape(k, -1, cap.shape[-1])
                .transpose(0, 1)
                .reshape(-1, k * cap.shape[-1]),
                t,
            )
        buf = forward_batch.aux_hidden_states_buffer
        if buf is not None:
            buf.copy_(aux)
            aux = buf
        return out, aux


def kda_state_rows(mamba_pool, mode, slot, req_pool_indices):
    """(read row, write base) of one request's KDA state, as the TPU GDN backend uses them
    (``sglang_tpu.mamba_pool``, ``TpuGDNAttnBackend._state_rows``).

    Slots own one state row each; with speculation every request pool row also owns a
    ``checkpoint_width`` verify window after the slots. A verify resumes from the committed
    row (the slot row, or the window row its last commit selected) and writes the state after
    row t to window row t. Extend and decode read and write the slot row (extend's metadata
    has already reset the slot's read offset to ``AT_SLOT``).

    ``slot`` is a 0-d int32 tensor, ``req_pool_indices`` the request's ``[1]`` pool row.
    """
    import torch

    if mode.is_target_verify() and mamba_pool.checkpoint_read_offsets is not None:
        # index_select over a [1] index: a 0-d tensor index is data-dependent under dynamo
        read = mamba_pool.committed_state_rows(slot.reshape(1), req_pool_indices)
        base = mamba_pool.window_bases(req_pool_indices)
        return read.to(torch.int32)[0], base.to(torch.int32)[0]
    return slot, slot


def _check_pools(kernel, shapes: dict) -> None:
    """sglang's KDA / MLA pools, read and written in place, must have the layout the kernel was
    built for (per row / page; the leading slot or page count is free)."""
    import jax.numpy as jnp

    from sglang_tpu.k3_megakernel.kernel.stages.mla import PAGE

    kd = kernel.kda
    expect = {
        "kda_conv_": ((3, 1, 3 * kd.hd), jnp.float32),
        "kda_rec_": ((kd.heads, 128, 128), jnp.float32),
        "mla_cache_": ((PAGE // 2, 2, 640), jnp.bfloat16),
    }
    bad = []
    for name, (shape, dtype) in shapes.items():
        for prefix, (tail, want) in expect.items():
            if name.startswith(prefix) and (
                tuple(shape[-len(tail) :]) != tail
                or jnp.dtype(dtype) != jnp.dtype(want)
            ):
                bad.append(
                    f"{name} {shape} {jnp.dtype(dtype).name} (want [..., {tail}] {jnp.dtype(want).name})"
                )
    if bad:
        raise ValueError(
            "K3 megakernel: sglang's state / KV pools do not have the layout the kernel reads "
            "(check --page-size 128, bf16 KV cache, f32 KDA state): " + "; ".join(bad)
        )


def _fit_rows(x, t: int):
    import torch.nn.functional as F

    return F.pad(x, (0, 0, 0, t - x.shape[0])) if x.shape[0] < t else x[:t]


class _Variant(msgspec.Struct, frozen=True):
    kind: str
    kernel: object
    block: int  # rows per block of the rows-mode kernel (0: token loop / decode)
    captures: tuple


def _register_op() -> None:
    """``k3mk::stack(args, runtime, variant) -> (final rows, captures)`` bf16 as one torch custom op
    (the variant and the page-table length pick a cached compiled call), so dynamo keeps it opaque
    inside the fullgraph-compiled forward. The pools are deliberately not declared as mutated:
    functionalization would copy them back every step; the kernel writes them in place and
    nothing else in the graph reads them."""
    import torch

    if len(_RUNTIMES) > 1:
        return

    @torch.library.custom_op("k3mk::stack", mutates_args=(), device_types=["tpu"])
    def stack(
        args: list[torch.Tensor], runtime: int, variant: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _RUNTIMES[runtime].run(variant, args)

    @stack.register_fake
    def _(args, runtime, variant):
        rt = _RUNTIMES[runtime]
        v = rt.variants[variant]
        x0 = args[rt.names.index("x0")]
        h = x0.shape[-1]
        if v.kind == "decode":
            return x0.new_empty((1, h)), x0.new_empty((1, 1, h))
        k = max(1, len(v.captures))
        # prefill: [R, 1, H] (loop) | [blocks, block, H] (rows); verify: [W, H]
        return x0.new_empty(x0.shape), x0.new_empty((k,) + tuple(x0.shape))


def _prefill_kernel(spec, rows: int, opts: dict, mode: str):
    """(kernel, rows per block) for prefill chunks of ``rows`` tokens. ``mode``: "rows" (blocks of 8 tokens
    through every stage), "loop" (one stack pass per token), "auto" (rows when this kernel has it, else loop)."""
    from sglang_tpu.k3_megakernel.kernel.stack import StackKernel

    params = inspect.signature(StackKernel.__init__).parameters
    block = _PREFILL_BLOCK
    can_rows = "prefill_blocks" in params and rows % block == 0
    if mode == "rows" and not can_rows:
        raise ValueError(
            "SGLANG_TPU_K3_MEGAKERNEL_PREFILL=rows: this kernel has no rows-mode prefill"
        )
    if mode in ("rows", "auto") and can_rows:
        return StackKernel(
            spec,
            verify_rows=block,
            verify_mode="rows",
            prefill_blocks=rows // block,
            **opts,
        ), block
    loop = {"token_rows": rows} if "token_rows" in params else {"verify_rows": rows}
    return StackKernel(spec, **loop, **opts), 0


def _barrier() -> None:
    import torch

    from sglang.srt.distributed.parallel_state import get_tp_group

    torch.distributed.barrier(group=get_tp_group().cpu_group)


def _jax_dtype(torch_dtype):
    import jax.numpy as jnp
    import torch

    return {
        torch.bfloat16: jnp.bfloat16,
        torch.float32: jnp.float32,
        torch.int32: jnp.int32,
        torch.uint8: jnp.uint8,
        torch.float8_e4m3fn: jnp.float8_e4m3fn,
    }[torch_dtype]


def build_runtime(causal_lm, *, owns_weights: bool = True) -> K3MegakernelRuntime:
    """Construct the runtime for ``KimiK3LinearForCausalLM`` after weight load. Without
    ``owns_weights`` (sglang keeps its layers and runs prefill) only the decode kernel is built."""
    from sglang.srt.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from sglang.srt.runtime_context import get_model, get_parallel, get_schedule
    from sglang_tpu.envs import envs as tpu_envs

    model = causal_lm.model
    num_layers = len(model.layers)
    # From the config: with the megakernel owning the weights the layers are placeholders.
    mla = tuple(i for i in range(num_layers) if not model.config.is_kda_layer(i))
    prefill_mode = tpu_envs.SGLANG_TPU_K3_MEGAKERNEL_PREFILL.get()
    load_slots = tpu_envs.SGLANG_TPU_K3_MEGAKERNEL_EXPERT_LOAD_SLOTS.get()
    if load_slots < 1:
        raise ValueError(
            f"SGLANG_TPU_K3_MEGAKERNEL_EXPERT_LOAD_SLOTS must be >= 1 (got {load_slots})"
        )
    return K3MegakernelRuntime(
        model_path=get_model().model_path,
        num_layers=num_layers,
        mla_layers=mla,
        tp_rank=get_tensor_model_parallel_rank(),
        tp_size=get_tensor_model_parallel_world_size(),
        nnodes=get_parallel().nnodes,
        prefill_rows=(
            prefill_rows(get_schedule().chunked_prefill_size, prefill_mode)
            if owns_weights
            else 0
        ),
        prefill_mode=prefill_mode,
        verify_mode=tpu_envs.SGLANG_TPU_K3_MEGAKERNEL_VERIFY.get(),
        kernel_opts=tpu_envs.SGLANG_TPU_K3_MEGAKERNEL_KERNEL_OPTS.get(),
        expert_procs=tpu_envs.SGLANG_TPU_K3_MEGAKERNEL_EXPERT_LOAD_PROCS.get(),
        load_slots=load_slots,
    )
