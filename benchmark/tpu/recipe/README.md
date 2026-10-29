# TPU serving recipes

A recipe is one model's serving configuration: parallelism, quantization,
attention backend, compile buckets, and the runtime environment. CI launches
these; so can you.

**New recipes are Python**, under `configs/<model>/`. The shell recipes still
here are legacy and leave as their tests move (see
[Legacy shell recipes](#legacy-shell-recipes)).

## Running one

From the repo root:

```bash
python -m benchmark.tpu.launch_server --config recipe/configs/gpt_oss_20b/perf.py
```

- `--dry-run` prints the resolved command and environment as JSON and exits.
  Use it to diff two recipes, or to see what a flag renders to.
- `--runtime-envs '{"HF_HUB_OFFLINE":"1"}'` overrides `runtime_env` entries;
  `null` as a value removes a variable.
- `--config recipe/configs/<model>/pd_x.py:PREFILL_CONFIG` launches one engine of
  a P/D pair on this host, without the router.

`--config` resolves module paths only under `recipe/`, `recipe.`, or
`benchmark.tpu.recipe.`, and the repo root must be on `sys.path` — run the
module from the repo root, not from `benchmark/tpu/`.

## The format

A recipe module defines `CONFIG`, a dict of two dicts:

```python
CONFIG = {
    "server_args": {...},   # becomes sglang.launch_server CLI flags
    "runtime_env": {...},   # becomes the server process's environment
}
```

`server_args` keys are `ServerArgs` field names; `configs/config_eval.py`
renders them to flags (`max_running_requests` -> `--max-running-requests`):

| Value | Renders as |
| --- | --- |
| `True` | the bare switch, `--enable-torch-compile` |
| `False` or `None` | nothing — the flag is omitted |
| a list | the flag once, then each value: `--tpu-compile-bs 1 4 8` |
| a dict | the flag and compact JSON |
| anything else | the flag and `str(value)` |

`None` is how a recipe opts out of a base setting. On a bf16 checkpoint write
`"quantization": None`, not the shell recipes' `QUANTIZATION=unquant`: both
leave the weights unquantized, and `None` is the one spelling in Python.

A P/D recipe defines `GROUP = {"hosts": N, "prefill_engines": N, "decode_engines": N}`
plus `PREFILL_CONFIG` and `DECODE_CONFIG` instead of a single `CONFIG`.

### Inheritance

`configs/base.py` holds what every TPU server needs; `_FP8_BASE_CONFIG` adds the
fp8 serving defaults. A recipe copies one and overrides:

```python
CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update({...})
```

`deepcopy` is load-bearing: `update()` on a shared dict would edit the base for
every recipe imported afterwards in the same process. The CPU gate
(`test/registered/unit/test_tpu_config_eval.py`) fails when a recipe mutates
its base.

`configs/base.py:LIBTPU_FLAGS` is the XLA flag set, as an ordered tuple. The
string it joins to is part of the compile-cache key, so a recipe that needs a
different flag substitutes it **in place** rather than appending — see
`configs/qwen3_5_397b/perf.py`. `benchmark/tpu/common_env.sh` carries the shell
launcher's copy of the same set.

### How a model's recipes are laid out

`server.py` owns how the model is served. Each variant derives from it and
changes only what the measurement needs — a context length, a concurrency, an
eval's parser — never the parallelism or the backend. A variant that has to
diverge on serving says why in a comment.

| Variant | For |
| --- | --- |
| `server.py` | the deployment; what the cookbook page renders from |
| `accuracy.py`, `accuracy_<benchmark>.py` | the nightly accuracy gates |
| `perf.py` | the nightly throughput sweep |
| `pd_*.py` | disaggregated prefill/decode gates |
| `singlehost_spec.py`, `pcp.py`, `pp4_*.py` | one feature's gate |

## Index

| Model | Variants | Run by |
| --- | --- | --- |
| `deepseek_r1_0528/` | `accuracy`, `perf` | `nightly-tpu-multihost-accuracy`, `nightly-tpu-perf-deepseek-r1-0528` |
| `deepseek_v3_1/` | `accuracy`, `perf` | `nightly-tpu-multihost-accuracy`, `nightly-tpu-perf-deepseek-v3-1` |
| `deepseek_v4_flash.py` | (flat) | `stage-b-test-tpu-4-v7` |
| `gemma4_26b_a4b/` | `accuracy`, `perf` | `nightly-tpu-accuracy-gemma4-26b-a4b-*`, `nightly-tpu-perf-gemma4-26b-a4b` |
| `gemma4_31b/` | `server`, `accuracy`, `perf` | `nightly-tpu-accuracy-gemma4-31b-*`, `nightly-tpu-perf-gemma4-31b` |
| `gpt_oss_120b/` | `server`, `accuracy`, `accuracy_{aime,gpqa,gsm8k,mmlupro}`, `singlehost_spec`, `pd_singlehost`, `pd_singlehost_{spec,kv_offload_spec,kv_offload_dflash,mxint8_multiturn}`, `pd_singlehost_pp_*` | `nightly-tpu-accuracy-gpt-oss-120b-*`, `tpu-integration`, `external-nightly-1host` |
| `gpt_oss_20b/` | `server`, `accuracy`, `perf`, `spec`, `spec_dpa`, `prefix_cache_kv_offload`, `pcp`, `pp4_pcp2`, `pp4_tp2` | `nightly-tpu-accuracy-gpt-oss-20b`, `nightly-tpu-perf-gpt-oss-20b`, `stage-b-test-tpu-4-v7` |
| `kimi_k26/` | `mmmu`, `perf` | `nightly-tpu-multihost-accuracy`, `nightly-tpu-perf-kimi-k26` |
| `qwen3_0_6b.py` | (flat) | the compile-tracer e2e; single chip |
| `qwen3_5_397b/` | `server`, `perf` | `nightly-tpu-perf-qwen3-5-397b` |
| `qwen3_coder_30b_a3b_fp8.py` | (flat) | `stage-b-test-tpu-4-v7` |
| `qwen3_vl_30b_a3b/` | `pd_singlehost` | `tpu-integration` |

A flat `configs/<model>.py` is the older layout for a model with exactly one
recipe; a model that grows a second variant moves to a directory.

## How tests reach a recipe

Three forms, and a rename has to update whichever a test uses:

- `CONFIG_MODULE` / `config_module` — a path string the fixture hands the Python
  launcher.
- `MULTIHOST_LAUNCH["launch_script"]` — the JobSet runs this on every host of the
  slice. `.github/tpu-ci/external/render_external_jobset.py` reads the same
  field, so a rename touches the test and `.github/tpu-ci/` together.
- `MODEL_CONFIG["launch"]["config"]` — the Python launcher, run from the repo
  root. This is the form a new test takes.
- `MODEL_CONFIG["launch"]["recipe"]` — `bash <script>`, shell recipes only.

## Legacy shell recipes

These wrap `benchmark/tpu/launch_server.sh` with env vars. Each goes away when
its tests move to a Python config; do not add more.

| Recipe | Tests |
| --- | --- |
| `gpt_oss_120b/server.sh` | the gpt-oss-120b single-host perf and feature gates |
| `qwen3_coder_480b/server.sh` | Qwen3-Coder-480B single-host gates |
| `qwen3_vl_8b/server.sh` | Qwen3-VL-8B EAGLE3 |
| `qwen3_vl_30b_a3b/server.sh` | the single-host MMMU-Pro gate; paired with `configs/qwen3_vl_30b_a3b/pd_singlehost.py` |
| `qwen3_30b_a3b_fp8_pcp.sh`, `qwen3_30b_a3b_fp8_mh.sh` | Qwen3-30B-A3B PCP / multi-host |
| `qwen3_coder_30b_a3b_fp8.sh` | superseded by `configs/qwen3_coder_30b_a3b_fp8.py` |
| `gpt_oss_120b_spec_mh.sh`, `gpt_oss_120b_dflash_mh.sh` | gpt-oss-120b multi-host JobSet launch groups |
| `mimo_v25_pro_mh.sh` | MiMo-V2.5-Pro multi-host GSM8K |
| `qwen3_coder_480b_agentic_mh.sh`, `qwen3_coder_480b_agentic_spec_mh.sh` | Qwen3-Coder-480B multi-host agentic |

`bench_matrix/` and `qwen3_coder_480b/nightly/` are a different thing: sweep
drivers for the external nightly benchmark legs, not serving recipes. They stay
shell, and their behaviour is pinned by `test/registered/unit/test_bench_matrix.py`.
