# Kimi-K3 megakernel standalone server: DSpark speculative decoding checks

Manual (TPU) checks of `sglang_tpu/k3_megakernel/server` with the DSpark draft. All commands run from
the tree root with `PYTHONPATH=$TREE/python:$TREE` and one host's TPU environment (see
`sglang_tpu/k3_megakernel/server/run1h.sh`); model options follow `--` as for `server/app.py`.

## Serve with DSpark

```bash
# single host, 13-layer stack (run1h.sh logs to $OUT_ROOT/<tag>/server.log)
bash sglang_tpu/k3_megakernel/server/run1h.sh dspark13 --port 30210 --capacity 4096 \
    --spec-draft dspark --draft-path /path/to/dspark_draft --spec-rounds 8 [--bench-hooks] \
    -- --layers 13 --expert-load stream
# 4 hosts, 93 layers: server/run4h.sh on every host with the same server arguments
```

`--spec-rounds` is the number of draft + verify rounds per device call, `--draft-tp 8` runs one draft copy
per host instead of one over the whole mesh (default). The server log's `job N: {...}` lines carry
`accept_length` (tokens per round, bonus included) and `ms_per_round`.

`--bench-hooks` enables the per-request benchmark / test fields; without it a request using them gets
a 400. They are for the checks below only (output meaningless):
`k3_force_acc=N` (every round emits N tokens whatever the target says: the PR #8 timing protocol),
`k3_draft="zero"` (verify token 0 drafts without running the draft) and `k3_stop_ids=[ids]` (extra
ids treated like EOS).

## Checks

| Script | What |
| --- | --- |
| `spec_check.py` | Engine only (no HTTP): plain decode vs spec decode at several rounds per call, 3 prompts. `SPECCHK_OK True` = output independent of the rounds per call. `--verify-mode loop`: spec output must also equal plain decode (`"equal": true`). |
| `spec_smoke.py` | HTTP smoke against a running server started with `--bench-hooks` (`--spec` for a DSpark server): max_tokens, stop strings, EOS, stream == non-stream, forced acceptance. |
| `spec_bench.py` | `force`: forced-acceptance timing (ms per round) over HTTP; `stats`: accept length / ms per round from a server log. |
| `test_dspark_draft.py` | The JAX draft alone against sglang DSpark reference dumps (`--compat`), plus draft latency (`--time`). |

```bash
python3 test/manual/k3_megakernel/spec_check.py --draft dspark --draft-path /path/to/dspark_draft \
    --rounds 1,8 -- --layers 13 --expert-load stream
python3 test/manual/k3_megakernel/spec_check.py --draft zero,replay --verify-mode loop \
    -- --layers 13 --expert-load stream
python3 test/manual/k3_megakernel/spec_smoke.py 30210 --spec
python3 test/manual/k3_megakernel/spec_bench.py force 30210 6 2000 3 dspark
python3 test/manual/k3_megakernel/spec_bench.py stats $OUT_ROOT/dspark13/server.log
```
