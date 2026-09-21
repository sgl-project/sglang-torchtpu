# Raiden runtime verification

This package contains opt-in Raiden event instrumentation and the generated,
dependency-free runtime monitor.

The human-authored LTL formulas and generator live in
`scripts/raiden_runtime_verification/`. Spot is needed only to regenerate the monitor,
not by the serving runtime.

Set `SGLANG_TPU_RAIDEN_RUNTIME_VERIFICATION_FUZZ_MAX_DELAY_MS` to a positive number
to inject an independently sampled uniform delay before each instrumented
method executes. The maximum is in milliseconds and is read once per process;
unset or zero disables the delay.
