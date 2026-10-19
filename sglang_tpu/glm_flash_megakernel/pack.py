"""Pack an HF safetensors GLM-5.3-Flash checkpoint into the layout `weights.load` reads.

    python -m sglang_tpu.glm_flash_megakernel.pack <hf_checkpoint_dir> <out_packed_dir> \
        [--tp 8] [--layers 0,3,45] [--workers 16]

After packing, `repack_channelwise` adds per-output-channel K-major copies of the
expert and shared-expert weights (`<name>_ch` / `<name>_chs`, listed in
`manifest_ch.rank<r>.json`), which the kernel's default config streams.

The output is exactly what `layout.PackedLayout.arrays()` declares: one `.npy` per
(array, rank) for sharded arrays and one per array for replicated ones, plus a
`manifest.json` carrying the format version, the per-rank file table, and the two
slab indices (`small_map`, `small_f32_map`).  `weights.to_oracle_dict` is the
inverse of this file; `test_pack.py` holds the two to each other.

Bytes are never converted.  Every checkpoint tensor is read as raw bits (bf16 as
uint16, fp8 as uint8, f32 as f32), sliced, concatenated and written; nothing is
requantised, so the packed fp8 bytes, bf16 bytes and f32 scales are the
checkpoint's own.  The `.npy` files therefore carry the storage descriptor
(`<u2`, `|u1`, `<f4`) and `_load_npy` views them as the manifest dtype.

Streaming: tensors are read one at a time with `pread` from the safetensors files
and written with `pwrite` at their final offset, so the working set is a few
tensors per worker, never the model.  Layers are independent jobs and run on a
thread pool; written ranges are handed to writeback immediately so the page cache
does not grow with the output.  The manifest is written last, so an interrupted
pack is not loadable.

Subsets: `--layers` fills only the listed layers (checkpoint indices; the MTP
layer is `num_hidden_layers`); every array still has its full declared shape and
the unlisted layer slots are zero (sparse holes on disk).  The Python entry point
`pack()` additionally takes a reduced `spec` with `layer_src` / `expert_src`
maps, which is how the round-trip test packs a few real layers and experts into a
small layout without materialising a 40 GiB stack.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import os
import struct
import sys
import threading
import time

import numpy as np

from .config import WEIGHT_PREFIX, Glm53FlashSpec, validate_against_manifest
from .layout import FORMAT_VERSION, PackedLayout
from .weights import SHARD_AXIS

# checkpoint dtype -> raw storage dtype (same itemsize, bits untouched)
RAW = {"BF16": np.dtype("<u2"), "F8_E4M3": np.dtype("u1"), "F32": np.dtype("<f4")}
# layout dtype name -> storage dtype written into the .npy header
STORE = {"bfloat16": np.dtype("<u2"), "float8_e4m3fn": np.dtype("u1"),
         "float32": np.dtype("<f4")}
# layout dtype name -> the checkpoint dtype it must come from
CKPT_OF = {"bfloat16": "BF16", "float8_e4m3fn": "F8_E4M3", "float32": "F32"}

_FADV = getattr(os, "POSIX_FADV_DONTNEED", None)


class PackError(RuntimeError):
    pass


# ----------------------------------------------------------------- checkpoint
class Checkpoint:
    """Per-tensor reads of a sharded safetensors checkpoint, without mmap."""

    def __init__(self, ckpt_dir: str):
        self.dir = ckpt_dir
        self.meta: dict[str, dict] = {}
        files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors"))
        if not files:
            raise PackError(f"no .safetensors files in {ckpt_dir}")
        self._fds: dict[str, int] = {}
        for fn in files:
            path = os.path.join(ckpt_dir, fn)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            base = 8 + n
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                if k in self.meta:
                    raise PackError(f"tensor {k} appears in two shards")
                b, e = v["data_offsets"]
                self.meta[k] = {"shape": list(v["shape"]), "dtype": v["dtype"],
                                "file": path, "off": base + b, "nbytes": e - b}
            self._fds[path] = os.open(path, os.O_RDONLY)
        self.used: set[str] = set()
        self._lock = threading.Lock()

    def close(self):
        for fd in self._fds.values():
            os.close(fd)

    def header_manifest(self) -> dict:
        return {k: {"shape": v["shape"], "dtype": v["dtype"]} for k, v in self.meta.items()}

    def _info(self, name, shape, dtype):
        m = self.meta.get(name)
        if m is None:
            raise PackError(f"checkpoint has no tensor {name}")
        if shape is not None and list(shape) != m["shape"]:
            raise PackError(f"{name}: checkpoint shape {m['shape']} != expected {list(shape)}")
        if dtype is not None and m["dtype"] != dtype:
            raise PackError(f"{name}: checkpoint dtype {m['dtype']} != expected {dtype}")
        if m["dtype"] not in RAW:
            raise PackError(f"{name}: unsupported dtype {m['dtype']}")
        return m

    def get(self, name: str, shape=None, dtype=None) -> np.ndarray:
        """The tensor as raw bits, in its checkpoint shape."""
        m = self._info(name, shape, dtype)
        return self.rows(name, 0, m["shape"][0] if m["shape"] else 1, _m=m)

    def rows(self, name: str, r0: int, r1: int, shape=None, dtype=None, _m=None) -> np.ndarray:
        """Leading-axis rows [r0, r1) of a tensor, read without touching the rest."""
        m = _m or self._info(name, shape, dtype)
        raw = RAW[m["dtype"]]
        shp = m["shape"]
        inner = int(np.prod(shp[1:])) if len(shp) > 1 else 1
        if raw.itemsize * inner * (shp[0] if shp else 1) != m["nbytes"]:
            raise PackError(f"{name}: header size disagrees with shape/dtype")
        rb = raw.itemsize * inner
        n = (r1 - r0) * rb
        buf = np.empty(n, np.uint8)
        fd = self._fds[m["file"]]
        got = 0
        mv = memoryview(buf)
        while got < n:
            k = os.preadv(fd, [mv[got:]], m["off"] + r0 * rb + got)
            if k <= 0:
                raise PackError(f"{name}: short read")
            got += k
        with self._lock:
            self.used.add(name)
        out = buf.view(raw)
        return out.reshape(()) if not shp else out.reshape([r1 - r0] + shp[1:])


# --------------------------------------------------------------------- output
class NpyOut:
    """A preallocated .npy written at row offsets with pwrite (thread-safe)."""

    def __init__(self, path: str, shape: tuple[int, ...], dtype: np.dtype):
        self.path, self.shape, self.dtype = path, tuple(shape), dtype
        with open(path, "wb") as f:
            np.lib.format.write_array_header_2_0(
                f, {"descr": np.lib.format.dtype_to_descr(dtype),
                    "fortran_order": False, "shape": self.shape})
            self.base = f.tell()
        self.row_bytes = int(np.prod(self.shape[1:])) * dtype.itemsize
        self.nbytes = self.row_bytes * self.shape[0]
        self.fd = os.open(path, os.O_WRONLY)
        os.ftruncate(self.fd, self.base + self.nbytes)   # unwritten rows read as 0
        self.written = 0
        self._lock = threading.Lock()

    def put(self, i0: int, block: np.ndarray):
        """Write rows [i0, i0+len(block)); `block` is [k, *shape[1:]] in storage dtype."""
        if block.dtype != self.dtype:
            raise PackError(f"{self.path}: dtype {block.dtype} != {self.dtype}")
        if tuple(block.shape[1:]) != self.shape[1:] or i0 < 0 or i0 + block.shape[0] > self.shape[0]:
            raise PackError(f"{self.path}: block {block.shape} at {i0} does not fit {self.shape}")
        data = np.ascontiguousarray(block).view(np.uint8).reshape(-1)
        off = self.base + i0 * self.row_bytes
        mv = memoryview(data)
        done = 0
        while done < len(mv):
            done += os.pwrite(self.fd, mv[done:], off + done)
        if _FADV is not None:
            os.posix_fadvise(self.fd, off, len(mv), _FADV)   # start writeback, drop clean
        with self._lock:
            self.written += len(mv)

    def close(self):
        os.fdatasync(self.fd)
        if _FADV is not None:
            os.posix_fadvise(self.fd, 0, 0, _FADV)
        os.close(self.fd)


def as_store(x: np.ndarray, dt_name: str) -> np.ndarray:
    want = STORE[dt_name]
    if x.dtype.itemsize != want.itemsize:
        raise PackError(f"itemsize mismatch {x.dtype} vs {dt_name}")
    return x.view(want)


# ------------------------------------------------------------------ slab plan
def slab_plan(spec) -> dict[str, tuple[int, int, tuple[int, ...]]]:
    """tag -> (row offset, rows, shape) in the bf16 slab.

    Mirrors `PackedLayout.small_rows` term by term: a common block, then one region
    shared by the KDA and DSA tensors (a layer is one kind or the other, which is
    why the height is `max(kda, dsa)`), then the three MTP norms.
    """
    H = spec.hidden_size
    nc = (2 + spec.hc_mult) * spec.hc_mult

    def rows(n):
        return -(-n // H)

    m: dict[str, tuple[int, int, tuple[int, ...]]] = {}
    off = 0

    def add(tag, shape):
        nonlocal off
        n = rows(int(np.prod(shape)))
        m[tag] = (off, n, tuple(shape))
        off += n

    add("input_layernorm", (H,))
    add("post_attention_layernorm", (H,))
    add("hc_attn_fn", (nc, spec.hc_mult * H))
    add("hc_ffn_fn", (nc, spec.hc_mult * H))
    common = off
    add("f_a_proj", (spec.linear_head_dim, H))
    add("g_a_proj", (spec.linear_head_dim, H))
    add("o_norm", (spec.linear_head_dim,))
    kda_end = off
    off = common
    add("q_a_layernorm", (spec.q_lora_rank,))
    add("kv_a_layernorm", (spec.kv_lora_rank,))
    add("indexer_wk", (spec.index_head_dim, H))
    add("indexer_k_norm_w", (spec.index_head_dim,))
    add("indexer_k_norm_b", (spec.index_head_dim,))
    add("indexer_weights_proj", (spec.index_n_heads, H))
    add("indexer_ape", (spec.index_kpool, spec.index_head_dim))
    add("indexer_compress_gate", (spec.index_head_dim, H))
    off = max(off, kda_end)
    add("enorm", (H,))
    add("hnorm", (H,))
    add("shared_head_norm", (H,))
    if off != PackedLayout(spec, 1).small_rows():
        raise PackError(f"slab plan has {off} rows, layout.small_rows() says "
                        f"{PackedLayout(spec, 1).small_rows()}")
    return m


def slab_f32_plan(spec) -> dict[str, tuple[int, int]]:
    """tag -> (offset, n) in row 0 of the float32 slab; identical for every layer."""
    nc = (2 + spec.hc_mult) * spec.hc_mult
    m, off = {}, 0
    for tag, n in (("hc_attn_base", nc), ("hc_attn_scale", 3),
                   ("hc_ffn_base", nc), ("hc_ffn_scale", 3),
                   ("router_bias", spec.n_routed_experts)):
        m[tag] = (off, n)
        off += n
    if off > spec.hidden_size or PackedLayout(spec, 1).small_f32_rows() != 1:
        raise PackError("float32 slab no longer fits one row; `small_f32` reads row 0 only")
    return m


def f32_tags(spec, L: int) -> list[str]:
    if L == spec.mtp_layer_idx:
        return ["router_bias"]
    tags = ["hc_attn_base", "hc_attn_scale", "hc_ffn_base", "hc_ffn_scale"]
    if spec.mlp_layer_types[L] == "sparse":
        tags.append("router_bias")
    return tags


# --------------------------------------------------------------------- packer
@dataclasses.dataclass
class Packer:
    ck: Checkpoint
    spec: Glm53FlashSpec
    lay: PackedLayout
    out: dict[str, list[NpyOut]]          # name -> per-rank writers (len 1 if replicated)
    layer_src: dict[int, int]
    expert_src: list[int]
    smap: dict
    fmap: dict

    @property
    def tp(self):
        return self.lay.tp

    # -- slicing helpers on raw arrays ------------------------------------
    def sh(self, x: np.ndarray, axis: int, r: int) -> np.ndarray:
        n = x.shape[axis]
        if n % self.tp:
            raise PackError(f"axis {axis} of {x.shape} not divisible by tp={self.tp}")
        per = n // self.tp
        return np.take(x, np.arange(r * per, (r + 1) * per), axis=axis)

    def sh_scale(self, s: np.ndarray, axis: int, r: int, w_len: int, blk: int) -> np.ndarray:
        """The scale blocks covering rank r's contiguous slice of a weight axis."""
        if s.shape[axis] != -(-w_len // blk):
            raise PackError(f"scale axis {axis} {s.shape} does not cover {w_len}/{blk}")
        per = w_len // self.tp
        if per % blk == 0:
            k = per // blk
            return np.take(s, np.arange(r * k, (r + 1) * k), axis=axis)
        if blk % per == 0:
            # A block spans several ranks: each rank gets a copy of its block.
            # `to_oracle_dict` concatenates these copies and so does NOT invert
            # this case; it does not occur for this checkpoint at tp=8.
            return np.take(s, [(r * per) // blk], axis=axis)
        raise PackError(f"per-rank width {per} and block {blk} do not nest")

    def put(self, name: str, i: int, per_rank):
        """per_rank: callable r -> raw array of the per-device shape[1:]."""
        shape, dt, _ = self.lay.arrays()[name]
        ws = self.out[name]
        for r, w in enumerate(ws):
            a = as_store(per_rank(r), dt)
            if tuple(a.shape) != tuple(shape[1:]):
                raise PackError(f"{name}[{i}] rank {r}: {a.shape} != {tuple(shape[1:])}")
            w.put(i, a[None])

    def rep(self, name: str, i: int, a: np.ndarray):
        if SHARD_AXIS[name] is not None:
            raise PackError(f"{name} is sharded")
        self.put(name, i, lambda r: a)

    def T(self, ck_layer: int, suffix: str, shape, dtype):
        return self.ck.get(f"{WEIGHT_PREFIX}layers.{ck_layer}.{suffix}", shape, dtype)

    # -- per-layer job ----------------------------------------------------
    def layer(self, L: int):
        s, lay = self.spec, self.lay
        H = s.hidden_size
        bs0, bs1 = s.weight_block_size
        src = self.layer_src[L]
        is_mtp = L == s.mtp_layer_idx
        nc = (2 + s.hc_mult) * s.hc_mult
        BF, F8, F32 = "BF16", "F8_E4M3", "F32"
        tp = self.tp

        slab = np.zeros((lay.small_rows(), H), np.uint16)
        slab32 = np.zeros((lay.small_f32_rows(), H), np.float32)

        def to_slab(tag, x):
            off, n, shape = self.smap[tag]
            if tuple(x.shape) != tuple(shape):
                raise PackError(f"slab {tag}: {x.shape} != {shape}")
            if x.dtype != np.uint16:
                raise PackError(f"slab {tag} must be bf16")
            flat = slab[off: off + n].reshape(-1)
            flat[: x.size] = x.reshape(-1)

        def to_slab32(tag, x):
            off, n = self.fmap[tag]
            if x.shape != (n,) or x.dtype != np.float32:
                raise PackError(f"f32 slab {tag}: {x.shape} {x.dtype}")
            slab32[0, off: off + n] = x

        to_slab("input_layernorm", self.T(src, "input_layernorm.weight", [H], BF))
        to_slab("post_attention_layernorm", self.T(src, "post_attention_layernorm.weight", [H], BF))
        if not is_mtp:
            for t in ("attn", "ffn"):
                to_slab(f"hc_{t}_fn", self.T(src, f"hc_{t}_fn", [nc, s.hc_mult * H], BF))
                to_slab32(f"hc_{t}_base", self.T(src, f"hc_{t}_base", [nc], F32))
                to_slab32(f"hc_{t}_scale", self.T(src, f"hc_{t}_scale", [3], F32))

        A = "self_attn."
        if not is_mtp and s.layer_types[L] == "linear_attention":
            k = s.kda_layers.index(L)
            Q, dh, K = s.kda_qkv_dim, s.linear_head_dim, s.linear_conv_kernel_dim
            q, kk, v = (self.T(src, A + f"{p}_proj.weight", [Q, H], BF) for p in "qkv")
            self.put("kda_qkv", k, lambda r: np.concatenate(
                [self.sh(q, 0, r), self.sh(kk, 0, r), self.sh(v, 0, r)], 0))
            del q, kk, v
            o = self.T(src, A + "o_proj.weight", [H, Q], BF)
            self.put("kda_o", k, lambda r: self.sh(o, 1, r))
            fb = self.T(src, A + "f_b_proj.weight", [Q, dh], BF)
            gb = self.T(src, A + "g_b_proj.weight", [Q, dh], BF)
            self.put("kda_fg_b", k, lambda r: np.concatenate([self.sh(fb, 0, r), self.sh(gb, 0, r)], 0))
            dt = self.T(src, A + "dt_bias", [Q], F32)
            self.put("kda_dt_bias", k, lambda r: self.sh(dt, 0, r)[None])
            cv = [self.T(src, A + f"{p}_conv1d.weight", [Q, 1, K], BF)[:, 0, :] for p in "qkv"]
            self.put("kda_conv", k, lambda r: np.concatenate([self.sh(c, 0, r) for c in cv], 0))
            b = self.T(src, A + "b_proj.weight", [s.linear_num_heads, H], BF)
            self.put("kda_b", k, lambda r: self.sh(b, 0, r))
            al = self.T(src, A + "A_log", [s.linear_num_heads], F32)
            self.put("kda_alog", k, lambda r: self.sh(al, 0, r)[None])
            to_slab("f_a_proj", self.T(src, A + "f_a_proj.weight", [dh, H], BF))
            to_slab("g_a_proj", self.T(src, A + "g_a_proj.weight", [dh, H], BF))
            to_slab("o_norm", self.T(src, A + "o_norm.weight", [dh], BF))
        else:
            if not is_mtp and s.indexer_types[L] != "full":
                raise PackError(f"layer {L}: indexer type {s.indexer_types[L]!r} has no "
                                "own weights; the packed layout has no rule for it")
            d = (s.dsa_layers + (s.mtp_layer_idx,)).index(L)
            nh, ql, kvl = s.num_attention_heads, s.q_lora_rank, s.kv_lora_rank
            qk, vd, nope, rope = s.qk_head_dim, s.v_head_dim, s.qk_nope_head_dim, s.qk_rope_head_dim
            qa = self.T(src, A + "q_a_proj.weight", [ql, H], F8)
            qas = self.T(src, A + "q_a_proj.weight_scale_inv", [-(-ql // bs0), -(-H // bs1)], F32)
            self.rep("dsa_q_a", d, qa)
            self.rep("dsa_q_a_scale", d, qas)
            kva = self.T(src, A + "kv_a_proj_with_mqa.weight", [kvl + rope, H], F8)
            kvas = self.T(src, A + "kv_a_proj_with_mqa.weight_scale_inv",
                          [-(-(kvl + rope) // bs0), -(-H // bs1)], F32)
            self.rep("dsa_kv_a", d, kva)
            self.rep("dsa_kv_a_scale", d, kvas)
            qb = self.T(src, A + "q_b_proj.weight", [nh * qk, ql], F8)
            qbs = self.T(src, A + "q_b_proj.weight_scale_inv", [-(-nh * qk // bs0), -(-ql // bs1)], F32)
            self.put("dsa_q_b", d, lambda r: self.sh(qb, 0, r))
            self.put("dsa_q_b_scale", d, lambda r: self.sh_scale(qbs, 0, r, nh * qk, bs0))
            kvb = self.T(src, A + "kv_b_proj.weight", [nh * (nope + vd), kvl], BF)
            self.put("dsa_kv_b", d, lambda r: self.sh(kvb, 0, r))
            o = self.T(src, A + "o_proj.weight", [H, nh * vd], F8)
            os_ = self.T(src, A + "o_proj.weight_scale_inv", [-(-H // bs0), -(-nh * vd // bs1)], F32)
            self.put("dsa_o", d, lambda r: self.sh(o, 1, r))
            self.put("dsa_o_scale", d, lambda r: self.sh_scale(os_, 1, r, nh * vd, bs1))
            to_slab("q_a_layernorm", self.T(src, A + "q_a_layernorm.weight", [ql], BF))
            to_slab("kv_a_layernorm", self.T(src, A + "kv_a_layernorm.weight", [kvl], BF))
            I = A + "indexer."
            ih, idh = s.index_n_heads, s.index_head_dim
            self.rep("indexer_wq_b", d, self.T(src, I + "wq_b.weight", [ih * idh, ql], BF))
            to_slab("indexer_wk", self.T(src, I + "wk.weight", [idh, H], BF))
            to_slab("indexer_k_norm_w", self.T(src, I + "k_norm.weight", [idh], BF))
            to_slab("indexer_k_norm_b", self.T(src, I + "k_norm.bias", [idh], BF))
            to_slab("indexer_weights_proj", self.T(src, I + "weights_proj.weight", [ih, H], BF))
            to_slab("indexer_ape", self.T(src, I + "index_kpool_compress_ape", [s.index_kpool, idh], BF))
            to_slab("indexer_compress_gate", self.T(src, I + "index_kpool_compress_gate", [idh, H], BF))

        if is_mtp:
            to_slab("enorm", self.T(src, "enorm.weight", [H], BF))
            to_slab("hnorm", self.T(src, "hnorm.weight", [H], BF))
            to_slab("shared_head_norm", self.T(src, "shared_head.norm.weight", [H], BF))
            eh = self.T(src, "eh_proj.weight", [H, 2 * H], BF)
            self.put("mtp_eh_proj", 0, lambda r: self.sh(eh, 1, r))

        M = "mlp."
        if is_mtp or s.mlp_layer_types[L] == "sparse":
            m = lay.moe_layers.index(L)
            E, mi = s.n_routed_experts, s.moe_intermediate_size
            if s.n_shared_experts != 1:
                raise PackError("shared_* arrays are sized for exactly one shared expert")
            ck_E = self.ck.meta[f"{WEIGHT_PREFIX}layers.{src}.{M}gate.weight"]["shape"][0]
            g = self.T(src, M + "gate.weight", [ck_E, H], BF)[self.expert_src]
            self.rep("moe_router", m, g)
            to_slab32("router_bias", self.T(src, M + "gate.e_score_correction_bias", [ck_E], F32)[self.expert_src])
            gus_rows = []
            dns_rows = []
            for e in range(E):
                EP = M + f"experts.{self.expert_src[e]}."
                gw = self.T(src, EP + "gate_proj.weight", [mi, H], F8)
                uw = self.T(src, EP + "up_proj.weight", [mi, H], F8)
                dw = self.T(src, EP + "down_proj.weight", [H, mi], F8)
                gs = self.T(src, EP + "gate_proj.weight_scale_inv", [-(-mi // bs0), -(-H // bs1)], F32)
                us = self.T(src, EP + "up_proj.weight_scale_inv", [-(-mi // bs0), -(-H // bs1)], F32)
                ds = self.T(src, EP + "down_proj.weight_scale_inv", [-(-H // bs0), -(-mi // bs1)], F32)
                slot = lay.expert_slot(L, e)
                self.put("expert_gate_up", slot, lambda r: np.concatenate(
                    [self.sh(gw, 0, r), self.sh(uw, 0, r)], 0))
                self.put("expert_down", slot, lambda r: self.sh(dw, 1, r))
                gus_rows.append([np.concatenate([self.sh_scale(gs, 0, r, mi, bs0),
                                                 self.sh_scale(us, 0, r, mi, bs0)], 0)
                                 for r in range(tp)])
                dns_rows.append([self.sh_scale(ds, 1, r, mi, bs1) for r in range(tp)])
            self.put("expert_gate_up_scale", m, lambda r: np.stack([x[r] for x in gus_rows]))
            self.put("expert_down_scale", m, lambda r: np.stack([x[r] for x in dns_rows]))
            SP = M + "shared_experts."
            sg = self.T(src, SP + "gate_proj.weight", [mi, H], F8)
            su = self.T(src, SP + "up_proj.weight", [mi, H], F8)
            sd = self.T(src, SP + "down_proj.weight", [H, mi], F8)
            sgs = self.T(src, SP + "gate_proj.weight_scale_inv", [-(-mi // bs0), -(-H // bs1)], F32)
            sus = self.T(src, SP + "up_proj.weight_scale_inv", [-(-mi // bs0), -(-H // bs1)], F32)
            sds = self.T(src, SP + "down_proj.weight_scale_inv", [-(-H // bs0), -(-mi // bs1)], F32)
            self.put("shared_gate_up", m, lambda r: np.concatenate([self.sh(sg, 0, r), self.sh(su, 0, r)], 0))
            self.put("shared_gate_up_scale", m, lambda r: np.concatenate(
                [self.sh_scale(sgs, 0, r, mi, bs0), self.sh_scale(sus, 0, r, mi, bs0)], 0))
            self.put("shared_down", m, lambda r: self.sh(sd, 1, r))
            self.put("shared_down_scale", m, lambda r: self.sh_scale(sds, 1, r, mi, bs1))
        else:
            dd = [i for i, t in enumerate(s.mlp_layer_types) if t == "dense"].index(L)
            F = s.intermediate_size
            g = self.T(src, M + "gate_proj.weight", [F, H], F8)
            u = self.T(src, M + "up_proj.weight", [F, H], F8)
            gs = self.T(src, M + "gate_proj.weight_scale_inv", [-(-F // bs0), -(-H // bs1)], F32)
            us = self.T(src, M + "up_proj.weight_scale_inv", [-(-F // bs0), -(-H // bs1)], F32)
            self.put("dense_gate_up", dd, lambda r: np.concatenate([self.sh(g, 0, r), self.sh(u, 0, r)], 0))
            self.put("dense_gate_up_scale", dd, lambda r: np.concatenate(
                [self.sh_scale(gs, 0, r, F, bs0), self.sh_scale(us, 0, r, F, bs0)], 0))
            del g, u
            dn = self.T(src, M + "down_proj.weight", [H, F], F8)
            dns = self.T(src, M + "down_proj.weight_scale_inv", [-(-H // bs0), -(-F // bs1)], F32)
            self.put("dense_down", dd, lambda r: self.sh(dn, 1, r))
            self.put("dense_down_scale", dd, lambda r: self.sh_scale(dns, 1, r, F, bs1))

        self.rep("smalls", L, slab)
        self.rep("smalls_f32", L, slab32)
        return L

    # -- embedding, head, final norm --------------------------------------
    def globals_(self, chunk_rows: int = 8192):
        s, lay = self.spec, self.lay
        H, V = s.hidden_size, s.vocab_size
        emb = f"{WEIGHT_PREFIX}embed_tokens.weight"
        head = emb if (s.tie_word_embeddings and "lm_head.weight" not in self.ck.meta) else "lm_head.weight"
        self.ck._info(emb, [V, H], "BF16")
        self.ck._info(head, [V, H], "BF16")
        for r0 in range(0, V, chunk_rows):
            r1 = min(V, r0 + chunk_rows)
            x = self.ck.rows(emb, r0, r1)
            for r, w in enumerate(self.out["embed_tokens"]):
                w.put(r0, self.sh(x, 1, r)[:, None, :])
        vs = lay.vocab_padded // self.tp
        for r, w in enumerate(self.out["lm_head"]):
            lo, hi = r * vs, min(V, (r + 1) * vs)       # rows past V stay zero
            for r0 in range(lo, hi, chunk_rows):
                r1 = min(hi, r0 + chunk_rows)
                w.put(r0 - lo, self.ck.rows(head, r0, r1))
        self.rep("final_norm", 0, self.ck.get(f"{WEIGHT_PREFIX}norm.weight", [H], "BF16"))
        return "globals"


# ----------------------------------------------------------------- entry point
def pack(ckpt_dir: str, out_dir: str, *, tp: int = 8, layers=None, spec=None,
         layer_src: dict[int, int] | None = None, expert_src=None, workers: int = 16,
         log=None) -> dict:
    """Pack `ckpt_dir` into `out_dir`; returns the manifest.

    `layers` are *packed* layer indices to fill (default: all, including MTP).
    `spec` / `layer_src` / `expert_src` let a reduced spec be packed from a subset of
    the checkpoint's layers and experts (packed index -> checkpoint index).
    """
    log = log or (lambda msg: print(msg, flush=True))   # progress must reach a nohup log
    t0 = time.time()
    ck = Checkpoint(ckpt_dir)
    full_spec = Glm53FlashSpec.from_checkpoint(ckpt_dir)
    problems = validate_against_manifest(full_spec, ck.header_manifest())
    if problems:
        raise PackError("checkpoint disagrees with the spec:\n  " + "\n  ".join(problems[:20]))
    spec = spec or full_spec
    spec.validate()
    n_all = spec.num_hidden_layers + 1
    if layer_src is None:
        if spec is not full_spec:
            raise PackError("a reduced spec needs layer_src")
        layer_src = {L: L for L in range(n_all)}
    if expert_src is None:
        expert_src = list(range(spec.n_routed_experts))
    if len(expert_src) != spec.n_routed_experts:
        raise PackError("expert_src must name one checkpoint expert per packed expert")
    layers = sorted(range(n_all) if layers is None else set(layers))
    bad = [L for L in layers if not 0 <= L < n_all]
    if bad:
        raise PackError(f"layers out of range 0..{n_all - 1}: {bad}")

    lay = PackedLayout(spec, tp)
    arrays = lay.arrays()
    if set(arrays) != set(SHARD_AXIS):
        raise PackError(f"layout and SHARD_AXIS disagree: {set(arrays) ^ set(SHARD_AXIS)}")
    smap, fmap = slab_plan(spec), slab_f32_plan(spec)

    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "manifest.json")
    if os.path.exists(mpath):
        os.remove(mpath)            # the old manifest must not describe new files

    out: dict[str, list[NpyOut]] = {}
    ranks: dict[str, dict] = {str(r): {} for r in range(tp)}
    for name, (shape, dt, _) in arrays.items():
        if SHARD_AXIS[name] is None:
            fn = f"{name}.npy"
            out[name] = [NpyOut(os.path.join(out_dir, fn), shape, STORE[dt])]
            for r in range(tp):
                ranks[str(r)][name] = {"file": fn, "dtype": dt, "shape": list(shape)}
        else:
            out[name] = []
            for r in range(tp):
                fn = f"{name}.rank{r}.npy"
                out[name].append(NpyOut(os.path.join(out_dir, fn), shape, STORE[dt]))
                ranks[str(r)][name] = {"file": fn, "dtype": dt, "shape": list(shape)}

    pk = Packer(ck, spec, lay, out, layer_src, list(expert_src), smap, fmap)
    log(f"packing {len(layers)} layers + embed/head into {out_dir} (tp={tp}, "
        f"{workers} workers, {sum(lay.budget().values()) / 2**30:.2f} GiB/device)")
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(pk.globals_)] + [ex.submit(pk.layer, L) for L in layers]
        for i, f in enumerate(cf.as_completed(futs)):
            what = f.result()
            log(f"  [{i + 1}/{len(futs)}] {what} done  {time.time() - t0:7.1f}s")
    for ws in out.values():
        for w in ws:
            w.close()

    manifest = {
        "format_version": FORMAT_VERSION,
        "tp": tp,
        "ranks": ranks,
        "small_map": {k: [v[0], v[1], list(v[2])] for k, v in smap.items()},
        "small_f32_map": {str(L): {t: list(fmap[t]) for t in f32_tags(spec, L)}
                          for L in range(n_all)},
        "shard_axis": dict(SHARD_AXIS),
        "vocab_size": spec.vocab_size,
        "vocab_padded": lay.vocab_padded,
        "num_hidden_layers": spec.num_hidden_layers,
        "mtp_layer_idx": spec.mtp_layer_idx,
        "packed_layers": layers,
        "layer_src": {str(k): v for k, v in layer_src.items()},
        "expert_src": list(expert_src) if list(expert_src) != list(range(spec.n_routed_experts)) else None,
        "source": os.path.abspath(ckpt_dir),
        "storage": "raw bits; .npy descr is the same-itemsize unsigned/f32 type, "
                   "view as the manifest dtype",
    }
    tmp = mpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, mpath)

    unused = None
    if layers == list(range(n_all)) and spec is full_spec:
        unused = sorted(k for k in ck.meta if k not in ck.used
                        and (k.startswith(WEIGHT_PREFIX) or k == "lm_head.weight"))
        if unused:
            log(f"WARNING: {len(unused)} language-model tensors were not packed, e.g. {unused[:5]}")
        else:
            log("every language-model checkpoint tensor was packed")
    ck.close()
    log(f"done in {time.time() - t0:.1f}s")
    return dict(manifest, unused_tensors=unused)


def repack_channelwise(out_dir: str, rank: int, chunk: int = 32, log=print) -> None:
    """Block-scaled fp8 experts -> per-output-channel fp8, K-major, for one rank.

    Dequantizes the 128x128 blocks, requantizes per output channel (scale =
    amax / 448), stores [.., K, N] fp8 plus an f32 scale [.., N], and writes
    manifest_ch.rank<r>.json.  The block-scaled files are left as they are."""
    import ml_dtypes

    f8 = np.dtype(ml_dtypes.float8_e4m3fn)
    man = json.load(open(os.path.join(out_dir, "manifest.json")))
    out_man = {}

    def load(name):
        e = man["ranks"][str(rank)][name]
        a = np.load(os.path.join(out_dir, e["file"]), mmap_mode="r")
        want = {"float8_e4m3fn": f8, "float32": np.dtype(np.float32)}[e["dtype"]]
        return a if a.dtype == want else a.view(want)

    for wname in ("shared_gate_up", "shared_down", "expert_gate_up", "expert_down"):
        w = load(wname)                                            # [M, N, K] fp8
        s_ = load(wname + "_scale")
        s_ = np.asarray(s_).reshape(-1, s_.shape[-2], s_.shape[-1])   # [M, N/128, K/128]
        M, N, K = w.shape
        qf, sf = f"{wname}_ch.rank{rank}.npy", f"{wname}_chs.rank{rank}.npy"
        q = np.lib.format.open_memmap(os.path.join(out_dir, qf + ".tmp"), mode="w+",
                                      dtype=np.uint8, shape=(M, K, N))
        sc = np.lib.format.open_memmap(os.path.join(out_dir, sf + ".tmp"), mode="w+",
                                       dtype=np.float32, shape=(M, N))
        for i in range(0, M, chunk):
            j = min(M, i + chunk)
            wf = np.asarray(w[i:j]).astype(np.float32).reshape(j - i, N // 128, 128, K // 128, 128)
            wf *= s_[i:j][:, :, None, :, None]
            wf = wf.reshape(j - i, N, K)
            ch = np.maximum(np.abs(wf).max(axis=2), 1e-12) / 448.0          # [c, N]
            q8 = (wf / ch[:, :, None]).astype(f8)
            q[i:j] = np.ascontiguousarray(q8.transpose(0, 2, 1)).view(np.uint8)
            sc[i:j] = ch
        q.flush()
        sc.flush()
        del q, sc
        os.replace(os.path.join(out_dir, qf + ".tmp"), os.path.join(out_dir, qf))
        os.replace(os.path.join(out_dir, sf + ".tmp"), os.path.join(out_dir, sf))
        out_man[wname + "_ch"] = {"file": qf, "dtype": "float8_e4m3fn", "shape": [M, K, N]}
        out_man[wname + "_chs"] = {"file": sf, "dtype": "float32", "shape": [M, N]}
        log(f"rank {rank} {wname}: [{M},{N},{K}] -> [{M},{K},{N}]")
    with open(os.path.join(out_dir, f"manifest_ch.rank{rank}.json"), "w") as f:
        json.dump(out_man, f, indent=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--layers", default=None,
                    help="comma-separated checkpoint layer indices to fill (MTP = "
                         "num_hidden_layers); default all. Unlisted slots are zero.")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    layers = None if a.layers is None else [int(x) for x in a.layers.split(",") if x]
    pack(a.ckpt_dir, a.out_dir, tp=a.tp, layers=layers, workers=a.workers)
    with cf.ProcessPoolExecutor(max_workers=a.tp) as ex:
        list(ex.map(repack_channelwise, [a.out_dir] * a.tp, range(a.tp)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
