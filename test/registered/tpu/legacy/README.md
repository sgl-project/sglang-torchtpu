# Legacy TPU tests

Retired tests, kept for reference but not run: each registers with
`disabled="<reason>"`, which both `run_suite.py` and `discover_multihost.py`
honor as a skip. Being here means the coverage is no longer wanted -- not that
the test is broken and awaiting a fix.

To revive one, drop its `disabled=` argument and move the file to the directory
matching its suite (`e2e/`, `nightly-accuracy/`, ...).
