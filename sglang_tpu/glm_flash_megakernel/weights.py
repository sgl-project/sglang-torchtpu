"""Load the packed weights onto the device mesh, and read the small slab back out.

The packed arrays are memory-mapped and handed straight to `jax.device_put` with an
explicit per-rank placement, so each device receives exactly its own shard and no
array is ever materialised whole on the host.  The format version in the manifest is
checked; a directory packed by an older layout is rejected rather than misread.
"""

from __future__ import annotations

import dataclasses
import json
import os

import jax
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

# np.save writes an ml_dtypes dtype as an opaque void descriptor, so np.load gives
# back `|V2` unless the dtype is restored explicitly.  Handing that to JAX fails
# with "Dtype |V2 is not a valid JAX array type", which looks like a JAX problem
# and is really a .npy round-trip problem.
NP_DTYPE = {
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "float8_e4m3fn": np.dtype(ml_dtypes.float8_e4m3fn),
    "float32": np.dtype(np.float32),
}


def _load_npy(path: str, dtype_name: str) -> np.ndarray:
    a = np.load(path, mmap_mode="r")
    want = NP_DTYPE[dtype_name]
    return a if a.dtype == want else a.view(want)

from .layout import FORMAT_VERSION, PackedLayout

# Which axis of each packed array is the sharded one.  `None` means the array is
# replicated: every rank holds identical bytes and the mesh axis is not involved.
SHARD_AXIS: dict[str, int | None] = {
    "embed_tokens": 2,
    "lm_head": 0,
    "final_norm": None,
    "expert_gate_up": 1,
    "expert_down": 2,
    "expert_gate_up_scale": 2,
    "expert_down_scale": 3,
    "shared_gate_up": 1,
    "shared_down": 2,
    "shared_gate_up_scale": 1,
    "shared_down_scale": 2,
    "kda_qkv": 1,
    "kda_o": 2,
    "kda_fg_b": 1,
    "kda_dt_bias": 2,
    "kda_conv": 1,
    "kda_b": 1,
    "kda_alog": 2,
    "dsa_q_b": 1,
    "dsa_q_b_scale": 1,
    "dsa_kv_b": 1,
    "dsa_o": 2,
    "dsa_o_scale": 2,
    "dense_gate_up": 1,
    "dense_gate_up_scale": 1,
    "dense_down": 2,
    "dense_down_scale": 2,
    "moe_router": None,
    "dsa_q_a": None,
    "dsa_q_a_scale": None,
    "dsa_kv_a": None,
    "dsa_kv_a_scale": None,
    "indexer_wq_b": None,
    "mtp_eh_proj": 2,
    "smalls": None,
    "smalls_f32": None,
    # Per-output-channel K-major experts (repack_ch.py); a down projection's scales
    # differ per rank (each rank quantizes its own input shard), hence sharded.
    "expert_gate_up_ch": 2, "expert_gate_up_chs": 1,
    "expert_down_ch": 1, "expert_down_chs": 1,
    "shared_gate_up_ch": 2, "shared_gate_up_chs": 1,
    "shared_down_ch": 1, "shared_down_chs": 1,
}

# ch_experts: load the *_ch experts (manifest_ch.rank<r>.json) instead of
# the block-scaled ones; both together would not fit next to the rest.
CH_REPLACES = {
    "expert_gate_up": "expert_gate_up_ch", "expert_gate_up_scale": "expert_gate_up_chs",
    "expert_down": "expert_down_ch", "expert_down_scale": "expert_down_chs",
    "shared_gate_up": "shared_gate_up_ch", "shared_gate_up_scale": "shared_gate_up_chs",
    "shared_down": "shared_down_ch", "shared_down_scale": "shared_down_chs",
}


class FormatMismatch(RuntimeError):
    pass


@dataclasses.dataclass
class WeightView:
    """The per-device arrays plus the static slab index.

    Inside `shard_map` the arrays arrive as explicit arguments (closing over a
    globally-sharded array there trips a Manual-vs-Explicit mesh conflict), so the
    view is rebuilt around them while the maps stay Python-side constants.
    """

    arrays: dict
    small_map: dict
    small_f32_map: dict
    layout: PackedLayout

    def small(self, layer: int, tag: str):
        off, nrows, shape = self.small_map[tag]
        flat = self.arrays["smalls"][layer, off : off + nrows].reshape(-1)
        n = int(np.prod(shape))
        return flat[:n].reshape(shape)

    def small_f32(self, layer: int, tag: str):
        off, n = self.small_f32_map[layer][tag]
        return self.arrays["smalls_f32"][layer, 0, off : off + n]


@dataclasses.dataclass
class PackedWeights:
    """Device-resident packed arrays plus the index into the small slab."""

    arrays: dict[str, jax.Array]
    small_map: dict[str, tuple[int, int, tuple[int, ...]]]
    small_f32_map: dict[int, dict[str, tuple[int, int]]]
    layout: PackedLayout
    mesh: jax.sharding.Mesh
    manifest: dict

    # ------------------------------------------------------------- slab access
    def small(self, layer: int, tag: str) -> jax.Array:
        """A tensor out of the per-layer bf16 slab, in its original shape."""
        off, nrows, shape = self.small_map[tag]
        flat = self.arrays["smalls"][layer, off : off + nrows].reshape(-1)
        n = int(np.prod(shape))
        return flat[:n].reshape(shape)

    def small_f32(self, layer: int, tag: str) -> jax.Array:
        off, n = self.small_f32_map[layer][tag]
        return self.arrays["smalls_f32"][layer, 0, off : off + n]

    def view(self, arrays: dict) -> WeightView:
        """Rebuild the accessor around the arrays shard_map handed to the body."""
        return WeightView(arrays, self.small_map, self.small_f32_map, self.layout)

    def partition_specs(self, axis: str = "tp") -> dict:
        out = {}
        for name, a in self.arrays.items():
            ax = SHARD_AXIS[name]
            parts = [None] * a.ndim
            if ax is not None:
                parts[ax] = axis
            out[name] = P(*parts)
        return out

    def nbytes_per_device(self) -> int:
        total = 0
        for name, a in self.arrays.items():
            total += a.nbytes if SHARD_AXIS[name] is None else a.nbytes // self.layout.tp
        return total


# Shards in flight during upload: enough to keep the reads parallel, few enough
# that the host staging copies stay far below the pod's memory limit.
_UPLOAD_WINDOW = 2


def _put_shards(shards, mesh):
    """At most _UPLOAD_WINDOW ranks' shards on the host at a time: device_put is asynchronous and
    keeps its host buffer alive until the copy lands, so materialising all eight
    at once holds every rank's slice (~39 GB each) on the host together with the
    page cache, enough to exhaust a 880 GiB pod."""
    out = []
    for r, d in enumerate(mesh.devices.flat):
        out.append(jax.device_put(np.asarray(shards[r]), d))
        if r >= _UPLOAD_WINDOW - 1:
            out[r - _UPLOAD_WINDOW + 1].block_until_ready()
    for a in out:
        a.block_until_ready()
    return out


def load(packed_dir: str, spec, mesh: jax.sharding.Mesh, *, axis: str = "tp",
         only: set[str] | None = None, ch_experts: bool = True) -> PackedWeights:
    manifest = json.load(open(os.path.join(packed_dir, "manifest.json")))
    if manifest["format_version"] != FORMAT_VERSION:
        raise FormatMismatch(
            f"packed weights are {manifest['format_version']!r}, this build expects "
            f"{FORMAT_VERSION!r}; repack rather than reinterpret"
        )
    tp = manifest["tp"]
    if mesh.shape[axis] != tp:
        raise FormatMismatch(f"packed for tp={tp}, mesh axis {axis} is {mesh.shape[axis]}")

    layout = PackedLayout(spec, tp)
    expected = set(layout.arrays())
    present = set(manifest["ranks"]["0"].keys())
    if expected != present:
        raise FormatMismatch(
            f"packed directory is missing {sorted(expected - present)} and has "
            f"unexpected {sorted(present - expected)}; repack with the current "
            f"layout ({FORMAT_VERSION})")
    names = list(manifest["ranks"]["0"].keys())
    if ch_experts:
        for r in range(tp):
            manifest["ranks"][str(r)].update(
                json.load(open(os.path.join(packed_dir, f"manifest_ch.rank{r}.json"))))
        names = [n for n in names if n not in CH_REPLACES] + list(CH_REPLACES.values())
    if only:
        names = [n for n in names if n in only]

    arrays: dict[str, jax.Array] = {}
    for name in names:
        ax = SHARD_AXIS[name]
        shards = []
        for r in range(tp):
            entry = manifest["ranks"][str(r)][name]
            path = os.path.join(packed_dir, entry["file"])
            shards.append(_load_npy(path, entry["dtype"]))
            if ax is None:
                break
        if ax is None:
            # np.memmap is not a JAX value; np.asarray materialises the (small,
            # replicated) array without copying the big sharded ones.
            arrays[name] = jax.device_put(
                np.asarray(shards[0]), NamedSharding(mesh, P(*([None] * shards[0].ndim)))
            )
        else:
            spec_parts = [None] * shards[0].ndim
            spec_parts[ax] = axis
            sharding = NamedSharding(mesh, P(*spec_parts))
            arrays[name] = jax.make_array_from_single_device_arrays(
                tuple(
                    s if i != ax else s * tp
                    for i, s in enumerate(shards[0].shape)
                ),
                sharding,
                _put_shards(shards, mesh),
            )

    small_map = {k: (v[0], v[1], tuple(v[2])) for k, v in manifest["small_map"].items()}
    small_f32_map = {
        int(k): {kk: (vv[0], vv[1]) for kk, vv in v.items()}
        for k, v in manifest["small_f32_map"].items()
    }
    return PackedWeights(arrays, small_map, small_f32_map, layout, mesh, manifest)


def to_oracle_dict(packed_dir: str, spec, layers: list[int], tp: int = 8) -> dict[str, np.ndarray]:
    """Rebuild float64 checkpoint-named tensors from the packed shards, on the host.

    This is what lets the float64 oracle be driven by the *packed* weights rather
    than by the checkpoint, which is the comparison that actually tests the packing
    and the reference path against the same numbers.
    """
    manifest = json.load(open(os.path.join(packed_dir, "manifest.json")))
    if manifest["format_version"] != FORMAT_VERSION:
        raise FormatMismatch(manifest["format_version"])
    lay = PackedLayout(spec, tp)

    def part(name: str, rank: int) -> np.ndarray:
        e = manifest["ranks"][str(rank)][name]
        return _load_npy(os.path.join(packed_dir, e["file"]), e["dtype"])

    def full(name: str) -> np.ndarray:
        ax = SHARD_AXIS[name]
        if ax is None:
            return np.asarray(part(name, 0))
        return np.concatenate([np.asarray(part(name, r)) for r in range(tp)], axis=ax)

    def unstack(name: str, n_blocks: int, axis: int) -> list[np.ndarray]:
        """Undo "concatenate n tensors, then shard that axis".

        `kda_qkv`, `*_gate_up` and friends pack several tensors head-to-tail and
        then shard the concatenated axis, so each rank holds
        `[q_shard, k_shard, v_shard]`.  Concatenating the ranks naively yields
        `q0 k0 v0 q1 k1 v1 ...` and slicing that into thirds silently mixes
        tensors -- which looks exactly like a numerical bug three layers later.
        Slice each rank first, then join.
        """
        parts = [np.asarray(part(name, r)) for r in range(tp)]
        per = parts[0].shape[axis] // n_blocks
        out = []
        for j in range(n_blocks):
            sl = [slice(None)] * parts[0].ndim
            sl[axis] = slice(j * per, (j + 1) * per)
            out.append(np.concatenate([p_[tuple(sl)] for p_ in parts], axis=axis))
        return out

    def f64(x) -> np.ndarray:
        return np.asarray(x).astype(np.float64)

    def deq(w: np.ndarray, s: np.ndarray, block) -> np.ndarray:
        """Blockwise fp8 -> float64, the dequantisation the kernel does implicitly."""
        w = f64(w)
        s = np.asarray(s, np.float64)
        s = np.repeat(np.repeat(s, block[0], axis=0), block[1], axis=1)
        return w * s[: w.shape[0], : w.shape[1]]

    small_map = {k: (v[0], v[1], tuple(v[2])) for k, v in manifest["small_map"].items()}
    small_f32_map = {int(k): {kk: tuple(vv) for kk, vv in v.items()}
                     for k, v in manifest["small_f32_map"].items()}
    smalls = part("smalls", 0)
    smalls_f32 = part("smalls_f32", 0)

    def slab(layer: int, tag: str) -> np.ndarray:
        off, n, shape = small_map[tag]
        flat = np.asarray(smalls[layer, off : off + n]).reshape(-1)
        return f64(flat[: int(np.prod(shape))].reshape(shape))

    def slab32(layer: int, tag: str) -> np.ndarray:
        off, n = small_f32_map[layer][tag]
        return np.asarray(smalls_f32[layer, 0, off : off + n], np.float64)

    w: dict[str, np.ndarray] = {}
    w["embed_tokens.weight"] = f64(full("embed_tokens")[:, 0, :])
    w["lm_head.weight"] = f64(full("lm_head")[: spec.vocab_size])
    w["norm.weight"] = f64(full("final_norm")[0])

    bs = spec.weight_block_size
    kda_i = {l: i for i, l in enumerate(spec.kda_layers)}
    dsa_i = {l: i for i, l in enumerate(spec.dsa_layers + (spec.mtp_layer_idx,))}
    moe_i = {l: i for i, l in enumerate(lay.moe_layers)}
    dense_i = {l: i for i, l in enumerate(
        [i for i, t in enumerate(spec.mlp_layer_types) if t == "dense"])}

    cache: dict[str, np.ndarray] = {}

    def F(name: str) -> np.ndarray:
        if name not in cache:
            cache[name] = full(name)
        return cache[name]

    for L in layers:
        Pp = f"layers.{L}."
        is_mtp = L == spec.mtp_layer_idx
        w[Pp + "input_layernorm.weight"] = slab(L, "input_layernorm")
        w[Pp + "post_attention_layernorm.weight"] = slab(L, "post_attention_layernorm")
        if not is_mtp:
            for t in ("attn", "ffn"):
                w[Pp + f"hc_{t}_fn"] = slab(L, f"hc_{t}_fn")
                w[Pp + f"hc_{t}_base"] = slab32(L, f"hc_{t}_base")
                w[Pp + f"hc_{t}_scale"] = slab32(L, f"hc_{t}_scale")

        A = Pp + "self_attn."
        if not is_mtp and spec.layer_types[L] == "linear_attention":
            k = kda_i[L]
            for j, nm in enumerate(("q_proj", "k_proj", "v_proj")):
                w[A + nm + ".weight"] = f64(unstack("kda_qkv", 3, 1)[j][k])
            w[A + "o_proj.weight"] = f64(F("kda_o")[k])
            fb, gb = unstack("kda_fg_b", 2, 1)
            w[A + "f_b_proj.weight"] = f64(fb[k])
            w[A + "g_b_proj.weight"] = f64(gb[k])
            w[A + "dt_bias"] = np.asarray(F("kda_dt_bias")[k, 0], np.float64)
            w[A + "A_log"] = np.asarray(F("kda_alog")[k, 0], np.float64)
            w[A + "f_a_proj.weight"] = slab(L, "f_a_proj")
            w[A + "g_a_proj.weight"] = slab(L, "g_a_proj")
            w[A + "b_proj.weight"] = f64(F("kda_b")[k])
            for j, nm in enumerate(("q_conv1d", "k_conv1d", "v_conv1d")):
                w[A + nm + ".weight"] = f64(
                    unstack("kda_conv", 3, 1)[j][k])[:, None, :]
            w[A + "o_norm.weight"] = slab(L, "o_norm")
        else:
            d = dsa_i[L]
            w[A + "q_a_proj.weight"] = deq(F("dsa_q_a")[d], F("dsa_q_a_scale")[d], bs)
            w[A + "q_b_proj.weight"] = deq(F("dsa_q_b")[d], F("dsa_q_b_scale")[d], bs)
            w[A + "kv_a_proj_with_mqa.weight"] = deq(
                F("dsa_kv_a")[d], F("dsa_kv_a_scale")[d], bs)
            w[A + "kv_b_proj.weight"] = f64(F("dsa_kv_b")[d])
            w[A + "o_proj.weight"] = deq(F("dsa_o")[d], F("dsa_o_scale")[d], bs)
            w[A + "q_a_layernorm.weight"] = slab(L, "q_a_layernorm")
            w[A + "kv_a_layernorm.weight"] = slab(L, "kv_a_layernorm")
            I = A + "indexer."
            w[I + "wq_b.weight"] = f64(F("indexer_wq_b")[d])
            w[I + "wk.weight"] = slab(L, "indexer_wk")
            w[I + "k_norm.weight"] = slab(L, "indexer_k_norm_w")
            w[I + "k_norm.bias"] = slab(L, "indexer_k_norm_b")
            w[I + "weights_proj.weight"] = slab(L, "indexer_weights_proj")
            w[I + "index_kpool_compress_ape"] = slab(L, "indexer_ape")
            w[I + "index_kpool_compress_gate"] = slab(L, "indexer_compress_gate")

        if is_mtp:
            w[Pp + "enorm.weight"] = slab(L, "enorm")
            w[Pp + "hnorm.weight"] = slab(L, "hnorm")
            w[Pp + "shared_head.norm.weight"] = slab(L, "shared_head_norm")
            w[Pp + "eh_proj.weight"] = f64(F("mtp_eh_proj")[0])

        M = Pp + "mlp."
        if is_mtp or spec.mlp_layer_types[L] == "sparse":
            m = moe_i[L]
            w[M + "gate.weight"] = f64(F("moe_router")[m])
            w[M + "gate.e_score_correction_bias"] = slab32(L, "router_bias")
            eg, eu = unstack("expert_gate_up", 2, 1)
            egs, eus = unstack("expert_gate_up_scale", 2, 2)
            dn, dns = F("expert_down"), F("expert_down_scale")
            for e in range(spec.n_routed_experts):
                slot = lay.expert_slot(L, e)
                EP = M + f"experts.{e}."
                w[EP + "gate_proj.weight"] = deq(eg[slot], egs[m, e], bs)
                w[EP + "up_proj.weight"] = deq(eu[slot], eus[m, e], bs)
                w[EP + "down_proj.weight"] = deq(dn[slot], dns[m, e], bs)
            sg, su = unstack("shared_gate_up", 2, 1)
            sgs, sus = unstack("shared_gate_up_scale", 2, 1)
            SP = M + "shared_experts."
            w[SP + "gate_proj.weight"] = deq(sg[m], sgs[m], bs)
            w[SP + "up_proj.weight"] = deq(su[m], sus[m], bs)
            w[SP + "down_proj.weight"] = deq(F("shared_down")[m], F("shared_down_scale")[m], bs)
        else:
            dd = dense_i[L]
            dg, du = unstack("dense_gate_up", 2, 1)
            dgs, dus = unstack("dense_gate_up_scale", 2, 1)
            w[M + "gate_proj.weight"] = deq(dg[dd], dgs[dd], bs)
            w[M + "up_proj.weight"] = deq(du[dd], dus[dd], bs)
            w[M + "down_proj.weight"] = deq(F("dense_down")[dd], F("dense_down_scale")[dd], bs)
    return w
