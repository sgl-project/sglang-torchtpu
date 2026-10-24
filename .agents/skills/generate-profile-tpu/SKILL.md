---
name: generate-profile-tpu
description: Capture xprof traces of an SGLang server on TPU (torch_tpu). Warms the server up, takes one capture during prefill and one during decode through /start_profile, checks the files for dropped events, and reads them with the xprof CLI. Use this instead of generate-profile on TPU, where send_one --profile asks for a CUDA trace and records no TPU activity.
---

# Generate an xprof profile of an SGLang server on TPU

On TPU, `/start_profile` writes an xprof trace (`*.xplane.pb`) for each rank,
not a Chrome trace. This skill starts a server, warms it up, takes one capture
while a wave of requests is prefilled and one while it decodes, checks the
files, and hands them to the xprof CLI.

## Before you start

- Profile in a run of its own, never during a benchmark. Traced steps are
  slower, and the server runs no batches while it writes a capture to disk
  (15 to 25 s for one GLM-5.3-Flash capture). Latency measured across a
  capture means nothing.
- The TPU may be shared. `fuser /dev/vfio/*` lists the processes that hold it;
  no output means it is free. Wait until it is free. Never kill a process you
  did not start, and do not use `pkill -f sglang`: it kills every SGLang
  server on the host, not just yours.
- Read the model's page under `docs/cookbook/` and launch the recipe it names,
  sized for the concurrency you will profile. A profile of a configuration
  nobody deploys tells you little.

## Step 1: Launch the server

From the repo root, start the Python launcher in the background with its log
in a file:

```bash
RUN=/path/to/run && mkdir -p "$RUN"
nohup python -m benchmark.tpu.launch_server \
    --config recipe/configs/<model>/<recipe>.py \
    > "$RUN/server.log" 2>&1 < /dev/null &
echo $! > "$RUN/launcher.pid"
```

- `MODEL`, `HOST` and `PORT` in the environment override the recipe's model
  path, host and port. A recipe's model path is often a CI path, so you may
  need `MODEL=<local checkpoint>`. Clear them if your shell sets them for
  something else.
- `--config` also takes a JSON config or `@file.json`. Use one when you need
  sizes the recipe does not have, such as `max_running_requests` for your
  concurrency. `--dry-run` prints the command and environment it would run.
- The launcher starts the server in a new process group and stops that whole
  group when it gets SIGTERM, which is how Step 5 stops it.

Wait until the log says `The server is fired up and ready to roll!`. Boot
compiles every graph, so it takes minutes (GLM-5.3-Flash at TP8 took about
8 minutes with a warm compile cache). Check that the launcher is still alive
while you wait, so a crash does not look like a slow boot.

## Step 2: Take the captures

Run the bundled script against the server:

```bash
MODEL=<checkpoint path> SERVER_LOG="$RUN/server.log" OUTPUT_DIR="$RUN/xprof" \
CONCURRENCY=64 ISL=8192 OSL=1024 PREFILL_STEPS=1 DECODE_STEPS=5 \
PYTHON=<python with sglang> \
    .agents/skills/generate-profile-tpu/scripts/capture_xprof.sh
```

It does what the next section describes. It sends an untraced warmup wave and
then a traced wave, so it takes two waves' time. At the end it prints the
profiler log lines, the number of recompiles during the traced wave, and one
line per rank file. It exits non-zero if a capture did not start or lost
events.

### What the script does, if you need to do it by hand

A capture is one call:

```bash
curl -sS -f "http://127.0.0.1:30000/start_profile" -H 'Content-Type: application/json' \
    -d '{"output_dir": "/abs/path/xprof", "num_steps": 5, "profile_prefix": "decode"}'
```

- Give `output_dir` as an absolute path. The server resolves a relative path
  from its own working directory.
- Leave out `activities`. On TPU the server then uses `["CPU", "TPU"]`, which
  traces both the device and the host. `["TPU"]` alone skips the host. A list
  with `"GPU"` asks for a CUDA torch trace and records no TPU activity.
  `python -m sglang.test.send_one --profile` always sends `["CPU", "GPU"]`,
  and `python -m sglang.profiler` does too unless you pass `--no-gpu --tpu`.
- The call returns as soon as the capture starts. The server stops it by
  itself after `num_steps` forward steps. Each rank then logs
  `Stop profiling...` and `Profiling done. Traces are saved to: ...`.
- Only one capture can run at a time. A second call while one is running
  fails with `Profiling is already in progress`.

When to start each capture:

- Warm up first, with the same request shape. A capture taken while a graph
  compiles shows the compile, not the model. The TPU recipes set
  `TORCH_LOGS=+recompiles`, so any compile during the traced wave shows up as
  a `recompil` line in the server log. There should be none.
- Send the traced wave with new prompts (another `--seed`). If the prompts
  are already in the prefix cache, there is little prefill left to trace.
- Prefill: start at the first `Prefill batch` log line of the traced wave.
- Decode: start at the first `Decode batch` line after that. While a waiting
  request still fits, the scheduler runs prefill before decode, so by then
  every request that fits is decoding. Do not wait for `#running-req` to
  equal the concurrency: the server may run fewer at once than
  `max_running_requests`. GLM-5.3-Flash sized for 64 ran 63 and kept one
  queued until the first request finished. Decode lines print every 40 steps
  (`--decode-log-interval`), so the output length must be well above
  40 + `num_steps`.

How many steps:

- A capture has a size limit (a 2 GB proto). When a capture goes over it, the
  profiler drops the rest of the trace and leaves a `Trace Buffers Dropped`
  event. The device timeline then shows the dropped part as idle time.
- SparseCore events fill most of that space. With libtpu 0.0.50 the files
  held every SparseCore and every tile, although
  `SGLANG_TPU_PROFILE_NUM_SPARSE_CORES` defaults to 1.
- GLM-5.3-Flash on tpu7x-8, 64 requests of 8k tokens: one capture held one
  8k-token prefill step (0.34 s) or 8 decode steps (20 ms each). Asking for
  2 prefill steps or 10 decode steps lost the end of the capture. The
  script's defaults, 1 prefill step and 5 decode steps, fit.

### Other ways to start a capture

- `{"profile_by_stage": true, "num_steps": N}` in one call captures the next
  N prefill steps and then the next N decode steps, in directories named
  `<prefix>-<id>-EXTEND-...` and `<prefix>-<id>-DECODE-...`. Both stages get
  the same N. A prefill step that arrives during the decode capture, such as
  a late request, ends the decode capture early.
  `SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS`, set at launch, holds the decode
  capture until a batch has that many requests. If the server never reaches
  that batch size, the decode capture never starts.
- `SGLANG_TPU_ENABLE_PROFILE_AUTO=1` at launch needs no HTTP call. Each rank
  captures the first prefill and decode batches after launch
  (`SGLANG_TPU_PROFILE_AUTO_STEPS` steps each) into
  `SGLANG_TORCH_PROFILER_DIR`. It is meant for P/D engines and CI. It records
  whatever runs first, not a warmed-up steady state. See
  `sglang_tpu/profile_auto.py`.

## Step 3: Check the files

Each rank writes its own directory:

```
$OUTPUT_DIR/<prefix>-<profile id>-TP-<r>[-DP-<r>][-PP-<r>][-EP-<r>]/plugins/profile/<YYYY_MM_DD_HH_MM_SS>/<host>.xplane.pb
```

- A tpu7x chip has two TensorCores, and each rank runs on one of them. With
  the default settings, one rank per chip records the device trace for both
  TensorCores of its chip (`/device:TPU:0`, `/device:TPU:1`, and their
  SparseCores). The other rank's file holds only the host. At TP8 on
  tpu7x-8, ranks 0, 2, 4 and 6 had the device trace and ranks 1, 3, 5 and 7
  had only the host. For GLM-5.3-Flash, one prefill step gave device files of
  240 to 350 MB and host files of 32 MB; 5 decode steps gave 200 MB and 8 MB.
- Any file that contains `Trace Buffers Dropped` lost events. Capture again
  with fewer steps:

  ```bash
  grep -l -a -F 'Trace Buffers Dropped' "$OUTPUT_DIR"/*/plugins/profile/*/*.xplane.pb
  ```

The script marks host-only files and files that lost events in its last lines.

## Step 4: Read the traces

Run the xprof CLI with `uvx`; nothing needs to be installed in the server's
environment. Pass one rank's run directory. xprof writes a cache file next to
the trace.

```bash
R=$(ls -d "$OUTPUT_DIR"/decode-*-TP-0{,-*}/plugins/profile/* 2>/dev/null | head -1)
uvx --from xprof xprof get_overview "$R"
uvx --from xprof xprof get_top_hlo_ops "$R" --limit=20
# Web UI (trace viewer, op profile) for every rank, on port 8791:
uvx --from xprof xprof server --logdir="$OUTPUT_DIR" --port=8791
```

- `get_top_hlo_ops` gives each op the Python source line that produced it
  (`source_file`, `source_line`). This ties a fusion or a Pallas kernel back
  to the layer in `sglang_tpu/` that emitted it.
- `IDLE` is time the device ran nothing. If it tops the list, first check
  the file for dropped events (Step 3), because the dropped part of a trace
  also counts as idle.
- xprof ships its own agent skill for deeper analysis.
  `uvx --from xprof xprof install_skills --target_dir=<dir>` unpacks it.

## Step 5: Stop the server

```bash
kill -TERM "$(cat "$RUN/launcher.pid")"
```

Then run `fuser /dev/vfio/*` and check that none of your server's processes
still hold the TPU. If the launcher was killed with SIGKILL, its server keeps
running in its own process group. Stop it with `kill -TERM -- -<pgid>`, using
the process group of the TPU holders that you started
(`ps -o pgid= -p <pid>`).

## Knobs

Set these in the server's environment at launch. They change no compiled
program, so changing them does not recompile anything.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SGLANG_TPU_PROFILE_NUM_CHIPS` | 1 | Chips each rank traces. 0 or less traces all of them. |
| `SGLANG_TPU_PROFILE_NUM_SPARSE_CORES` | 1 | SparseCores each rank traces. 0 or less traces all of them. With libtpu 0.0.50 the files still held all four SparseCores of the chip. |
| `SGLANG_TPU_PROFILE_PYTHON_TRACER` | 0 | 1 adds Python frames to the host trace. It uses a lot of host time. |
