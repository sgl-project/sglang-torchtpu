# TPU CI timing dashboard

Static, dependency-free dashboard inspired by Live Actions. PR and nightly views
default to the last 24 hours for PRs and 3 days for nightly runs, with duration trends, GitHub wait, TPU admission
wait/startup/execution, and expandable per-file test timings. Use the time filter
for longer history. Cards are Workflow runs, Median Queue time (submission to first TPU job start in either tab), Median execution time (TPU container execution, including setup), and
Success rate. Queue medians use workflow attempts. PR execution medians use
individual jobs; nightly execution medians use the sum of observed test-job
execution times within each completed workflow run. These nightly totals count
parallel jobs separately and flag missing timings as incomplete totals.
GitHub runner wait and Kueue admission wait remain separate in job details.
PR duration charts include passed runs only. Nightly duration charts include
scheduled completed runs (including partial/failed results), with a separate
curve and legend entry for each workflow and accuracy schedule selection; outcome is shown in the point's tooltip. Nightly mixed results show
Partial with passed/total test-file counts, falling back to labeled job counts
when any job lacks test-file results. Per-file time means each registered test **file**, not each
unittest method. Parallel jobs are never summed into elapsed CI wall time;
the nightly execution card deliberately reports summed job execution time.

The Nightly tab excludes manual triggers and runs with unknown trigger metadata.
Accuracy curves/rows distinguish **10 AM PT** (`day-10am`) and **5 PM PT**
(`day-5pm`) using the workflow's actual `plan` log selection. Delayed start times
never determine the selection. Missing plan output displays selection unknown.
Manual runs remain in stored history.

## Preview

```bash
python3 scripts/ci/tpu_dashboard/collect.py --days 1 --output /tmp/tpu-dashboard/data.json
cp scripts/ci/tpu_dashboard/web/* /tmp/tpu-dashboard/
python3 -m http.server 8765 --directory /tmp/tpu-dashboard
```

Open http://localhost:8765/ for real data, or http://localhost:8765/?sample=1 for
clearly labeled sample data (no API access needed for the sample).
For larger backfills, supply `GH_TOKEN` or `GITHUB_TOKEN` via the environment;
never put it in the page or a command argument. The collector supports `--days 90`
and `--history path/to/previous/data.json` and writes output atomically.

## Storage and automation

`tpu-ci-dashboard.yml` runs on completion of TPU workflows, hourly, and manually.
Collection always covers at least three days for nightly workflows.
Completion-triggered updates use GitHub's native concurrency queue (up to 100
pending runs), so a newer event does not replace an older pending update.
It stores 90 days of normalized JSON on the `ci-metrics` branch (the longest range
the page offers; the whole file is fetched per page view and must stay under
GitHub's 100 MB push limit) and uploads a
30-day preview artifact. If GitHub rate-limits collection, completed attempts
are saved with a warning so a later collection can resume. All attempts are keyed independently; completed records
are reused. GitHub Actions logs may expire, so older runs can have missing test
records. Historical TPU phase timings are unavailable until instrumentation is
deployed. Missing fields are unknown, never zero. In-progress records are snapshots,
not a live queue monitor. Awaiting-trigger attempts are hidden from dashboard rows, counts, and metrics;
they remain in stored history.

The existing externally generated nightly correctness/accuracy/performance dashboard
is separate and unchanged. This workflow never checks out or executes triggering PR
code. Artifacts are read as bounded JSON data; no archive files are extracted.

## Public hosting

Inspect the repository's existing Pages site before enabling deployment. If it is
already used, deploy this dashboard from a separate dedicated repository or integrate
its `/ci/` directory into that site's existing deployment; a Pages deployment
replaces the whole site. For an unused Pages site, select GitHub Actions as the
publishing source and set repository variable `TPU_CI_TIMING_PAGES_ENABLED=true`.
Then dispatch **TPU CI Timing Dashboard**. Link its deployment URL from the repo
About website field or README. Pages deployment is disabled by default, so this
change cannot overwrite an existing site. No tokens are shipped to viewers.

## Measurements

- Waiting in both tabs: current attempt submission (`created_at` from the attempt-specific
  GitHub API response) to the earliest TPU suite execution marker. This includes
  GitHub wait, image build, GKE/Kueue wait, pod startup, and test discovery. All
  observed TPU starts are considered; a job that never starts does not suppress
  timings from other jobs. Missing starts make the first-observed value an estimate.
  The marker is emitted immediately before running the first batch of files.
  Manual delay before requesting a rerun is excluded. Historical logs can provide
  an estimate from the timestamp when the runner observes the pod's enabled-tests
  announcement. These are labeled log estimates; replay/buffering can delay the
  observation. Exact test-start markers take precedence. Jobs without a suite
  announcement can fall back to a logged Running pod observation, also labeled
  as an estimate. Direct container start timestamps are used when available.
- Historical GKE wait + startup estimate: log-observed JobSet creation to the
  first logged Running pod row. This includes admission polling and pod startup,
  so it is not an exact Kueue admission measurement. Shown separately in job
  details and never substituted for exact TPU queue time.
- GitHub job wait: job `created_at` to `started_at`; dependency wait is not included.
- Job runtime includes setup, cluster wait, execution, and teardown.
- TPU queue: JobSet creation to Kueue admission.
- TPU startup: admission to test-container start.
- TPU execution: container start to termination, including in-container setup.
- Historical execution estimate: elapsed log-observed suite start to its
  TIMINGS END record, including time occupied by concurrent files without
  summing their durations. Buffering/replay can affect this estimate; direct
  container execution timestamps take precedence. A missing completion record
  stays unknown.
- Per-file durations: existing `TIMINGS` records, preserving passed/failed outcomes.

Telemetry is best effort and cannot turn a passing test into a failure. Cancellation
may leave unknown timings. Multi-host execution spans first start to last termination.
No estimates are substituted for missing measurements.

## Checks

```bash
python3 -m unittest discover -s scripts/ci/tpu_dashboard -p 'test_*.py'
node --test scripts/ci/tpu_dashboard/web/*.test.*
python3 -B -m unittest discover -s .github/tpu-ci/tests
```
