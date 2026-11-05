"""Convert Inkling2's routed experts (NVFP4 / MXFP8) to block-FP8 once, so TPU boots skip host requant.

Usage (CPU only; run with JAX_PLATFORMS=cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0, python -m sglang_tpu.models.inkling2_convert_fp8):
  ... SRC DST --layers 0-23 [--rest]    # one host per layer range
  ... SRC DST --finalize                # after every range: index, configs, small files

Experts become FP8 E4M3 [E, N, K] with FP32 128x128 block scales (`<w>_weight_scale_inv`), w13 already
de-interleaved to [gate | up]. The math is the loader's own (sglang_tpu.models.inkling2), so the result is
bit-identical to the runtime requant. Every other tensor is copied unchanged.
"""

import argparse
import json
import os
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sglang_tpu.models.inkling2 import _deinterleave_gate_up, _dequant_mxfp8, _dequant_nvfp4, _quant_block_fp8

EXPERTS = ".mlp.experts."
CHUNK = 32  # experts per dequant step; a whole NVFP4 layer at once needs ~100 GB of temporaries


def layer_of(name):
    return int(name.split("layers.", 1)[1].split(".", 1)[0])


class Source:
    def __init__(self, src):
        self.src = src
        self.weight_map = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
        self.handles = {}

    def get(self, name):
        shard = self.weight_map[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(f"{self.src}/{shard}", framework="pt")
        return self.handles[shard].get_tensor(name)


def convert_expert(src, prefix, w, precision):
    """(fp8 weight, fp32 scale_inv) for `<prefix><w>_weight`, built CHUNK experts at a time."""
    base = f"{prefix}{w}_weight"
    if precision == "nvfp4":
        weight, scale = src.get(base), src.get(base + "_scale")
        gscale = src.get(base + "_amax").float() / src.get(base + "_outer_scale_denominator").float().reshape(-1)[0]
        dequant = lambda s: _dequant_nvfp4(weight[s], scale[s], gscale[s])
    else:
        weight, scale = src.get(base), src.get(base + "_scale")
        dequant = lambda s: _dequant_mxfp8(weight[s], scale[s])
    qs, ss = [], []
    for e in range(0, weight.shape[0], CHUNK):
        t = dequant(slice(e, e + CHUNK))
        if w == "w13":
            t = _deinterleave_gate_up(t)
        q, s = _quant_block_fp8(t)
        qs.append(q)
        ss.append(s)
    return torch.cat(qs), torch.cat(ss)


def precision_of(quant, prefix):
    groups = {g["compute_dtype"]: set(g["tensors"]) for g in quant["model"]["parameter_groups"]}
    w13 = prefix + "w13_weight"
    return "nvfp4" if w13 in groups.get("nvfp4", ()) else "mxfp8" if w13 in groups.get("mxfp8", ()) else None


def convert_layers(src, dst, lo, hi, quant):
    for layer in range(lo, hi + 1):
        out = f"{dst}/experts-layer-{layer:02d}.safetensors"
        if os.path.exists(out):
            print(f"layer {layer}: exists, skipped", flush=True)
            continue
        prefix = f"model.llm.layers.{layer}.mlp.experts."
        precision = precision_of(quant, prefix)
        t0 = time.time()
        tensors = {}
        for w in ("w13", "w2"):
            q, s = convert_expert(src, prefix, w, precision)
            tensors[f"{prefix}{w}_weight"], tensors[f"{prefix}{w}_weight_scale_inv"] = q, s
        save_file(tensors, out + ".tmp")
        os.replace(out + ".tmp", out)  # a crash never leaves a half-written shard that looks done
        print(f"layer {layer}: {precision} -> fp8 in {time.time() - t0:.0f}s", flush=True)


def copy_rest(src, dst):
    out = f"{dst}/model-rest.safetensors"
    names = [n for n in src.weight_map if EXPERTS not in n]
    save_file({n: src.get(n) for n in names}, out + ".tmp", metadata={"format": "pt"})
    os.replace(out + ".tmp", out)
    print(f"copied {len(names)} non-expert tensors", flush=True)


def finalize(src_dir, dst, quant):
    weight_map = {}
    for f in sorted(os.listdir(dst)):
        if f.endswith(".safetensors") and not f.startswith("mtp"):
            with safe_open(f"{dst}/{f}", framework="pt") as h:
                weight_map.update({n: f for n in h.keys()})
    json.dump({"metadata": {}, "weight_map": weight_map}, open(f"{dst}/model.safetensors.index.json", "w"), indent=1)
    quant["tpu_expert_format"] = {"format": "fp8_e4m3_block", "block": [128, 128], "w13": "gate_up"}
    json.dump(quant, open(f"{dst}/hf_quant_config.json", "w"))
    for f in os.listdir(src_dir):
        if f.endswith(".safetensors") and not f.startswith("mtp") or f in ("model.safetensors.index.json", "hf_quant_config.json"):
            continue
        p = f"{src_dir}/{f}"
        (shutil.copytree if os.path.isdir(p) else shutil.copy2)(p, f"{dst}/{f}")
    print(f"index: {len(weight_map)} tensors; tpu_expert_format added", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layers", help="inclusive range, e.g. 0-23")
    ap.add_argument("--rest", action="store_true", help="also write the non-expert tensors")
    ap.add_argument("--finalize", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    quant = json.load(open(f"{a.src}/hf_quant_config.json"))
    src = Source(a.src)
    if a.finalize:
        return finalize(a.src, a.dst, quant)
    if a.rest:
        copy_rest(src, a.dst)
    if a.layers:
        lo, hi = map(int, a.layers.split("-"))
        convert_layers(src, a.dst, lo, hi, quant)


if __name__ == "__main__":
    main()
