# TPU serving recipes

Per-model serving configurations, as Python modules under `configs/`. Each
module builds a `CONFIG` dict — `server_args` plus the `runtime_env` the server
needs — by copying a shared base (`configs/base.py`) and overriding the fields
that are specific to the model and host shape. These are the launch examples
for each model; CI drives the same modules, so what you run is what is tested.

## Launch

Run from the repo root. `--config` takes the path relative to `benchmark/tpu`:

```bash
python3 -m benchmark.tpu.recipe.bench_matrix.serve_group \
    --config recipe/configs/gpt_oss_120b/server.py
```

Add `--dry-run` to print the resolved `sglang.launch_server` command — with the
inherited base args filled in — without launching. This is the fastest way to
see exactly what a recipe sets before committing a host to it:

```bash
python3 -m benchmark.tpu.recipe.bench_matrix.serve_group \
    --config recipe/configs/gpt_oss_120b/server.py --dry-run
```

### With Bazel

Each serve config — single-host and multihost alike — also has a `py_binary`
that pins it, so a model can be launched without naming the config path.
`bazel run` serves the model; extra `serve_group` args (e.g. `--dry-run`) go
after `--`:

```bash
bazel run //benchmark/tpu/recipe:gpt_oss_120b -- --dry-run
```

## Single-host serve configs

The per-model examples most people want. Each has a `Sample usage` docstring
and a matching `//benchmark/tpu/recipe:<target>` binary.

| config | bazel target | model | shape |
|---|---|---|---|
| `configs/gpt_oss_120b/server.py` | `//benchmark/tpu/recipe:gpt_oss_120b` | GPT-OSS-120B (MXFP4 MoE) | 1 host, tp8 / ep8 |
| `configs/qwen3_coder_480b/server.py` | `//benchmark/tpu/recipe:qwen3_coder_480b` | Qwen3-Coder-480B (FP8 MoE) | 1 host, tp8 / ep8 |
| `configs/qwen3_coder_30b_a3b_fp8.py` | `//benchmark/tpu/recipe:qwen3_coder_30b` | Qwen3-Coder-30B (FP8 MoE) | 1 host, tp8 / ep8 |
| `configs/qwen3_0_6b.py` | `//benchmark/tpu/recipe:qwen3_0_6b` | Qwen3-0.6B (tracer e2e) | single chip |
| `configs/mimo/launch_mimo_25.py` | `//benchmark/tpu/recipe:mimo_25` | MiMo-V2.5 (FP8 MoE) | 1 host, tp8 / ep8 / dp2 |

## Layout

- `configs/base.py` — `_BASE_CONFIG` and `_FP8_BASE_CONFIG`: the shared TPU
  serving defaults every recipe starts from. Model configs `deepcopy` a base
  and `update` it, so a field not mentioned in a recipe comes from here.
- `configs/<model>/...` — grouped families: `gpt_oss_120b/` (single-host serve
  plus the two-host `multihost_base` for the spec variants), `qwen3_coder_480b/`
  (serve, bench, and the `nightly/` agentic P/D sweep), `mimo/`, `bench_matrix/`.
- `configs/*_mh.py` — multi-host variants, launched by the CI JobSet across
  hosts rather than by hand. Each has a matching `//benchmark/tpu/recipe:<name>`
  binary; every host runs it, and `serve_group.py` derives its node rank from
  `TPU_WORKER_ID`.
- `bench_matrix/` — the concurrency/prefix-cache sweep machinery
  (`serve_group.py` is the launcher above; `bench_leg.py`, `run_sweep.py` drive
  the matrix legs).

## Customizing

The `.py` configs are fixed values, not env-parameterized like the old shell
recipes. To vary a recipe, edit the fields in its `server_args.update({...})`
block (or copy the module and override), then `--dry-run` to confirm the
resolved command before launching.
