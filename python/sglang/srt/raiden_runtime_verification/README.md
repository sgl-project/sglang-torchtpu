# Raiden runtime verification

This package contains opt-in Raiden event instrumentation and the generated,
dependency-free runtime monitor.

The human-authored LTL formulas and generator live in
`scripts/raiden_runtime_verification/`. Spot is needed only to regenerate the monitor,
not by the serving runtime.
