/* Dependency-free rendering; exported timing helpers also run under node --test. */
'use strict';

function duration(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
}

function formatDuration(value) {
  if (duration(value) === null) return '—';
  const seconds = Math.round(value);
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, '0')}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function percentile(values, fraction) {
  const known = values.filter(value => duration(value) !== null).sort((a, b) => a - b);
  if (!known.length) return null;
  const index = (known.length - 1) * fraction;
  return known[Math.floor(index)] + (known[Math.ceil(index)] - known[Math.floor(index)]) * (index % 1);
}

function mean(values) {
  const known = values.filter(value => duration(value) !== null);
  return known.length ? known.reduce((a, b) => a + b, 0) / known.length : null;
}

function runTime(run) {
  return Date.parse(run.completed_at || run.started_at || run.created_at);
}

function selectRuns(runs, lane, days, now = Date.now()) {
  return runs.filter(run => run.classification !== 'awaiting_trigger' && (run.lane !== 'nightly' || run.event === 'schedule') && run.lane === lane && runTime(run) >= now - days * 86400000 && runTime(run) <= now)
    .sort((a, b) => runTime(b) - runTime(a));
}

function runDuration(run) {
  if (!run.started_at || !run.completed_at) return null;
  return duration((Date.parse(run.completed_at) - Date.parse(run.started_at)) / 1000);
}

function submittedAt(run) {
  return run.submitted_at || run.created_at;
}

function startObservation(job) {
  for (const [key, estimated, kind] of [['test_started_at', false, 'test'], ['test_started_at_estimate', true, 'test'], ['started_at', false, 'TPU'], ['pod_started_at_estimate', true, 'TPU']]) {
    const at = job.tpu?.[key];
    if (Number.isFinite(Date.parse(at))) return {at, estimated, kind};
  }
  return null;
}

function submissionWait(run, job) {
  const start = Date.parse(submittedAt(run));
  const testStart = Date.parse(startObservation(job)?.at);
  return Number.isFinite(start) && Number.isFinite(testStart) ? duration((testStart - start) / 1000) : null;
}

function waitIsEstimated(run) {
  return run.jobs.some(job => startObservation(job)?.estimated || ((job.is_tpu_test || job.tpu) && submissionWait(run, job) === null));
}

function runWait(run) {
  const tests = run.jobs.filter(job => job.is_tpu_test || job.tpu);
  if (!tests.length) return null;
  const waits = tests.map(job => submissionWait(run, job));
  const observed = waits.filter(wait => wait !== null);
  return observed.length ? Math.min(...observed) : null;
}

function executionTime(job) {
  return duration(job.tpu?.execution_seconds) ?? duration(job.tpu?.execution_seconds_estimate);
}

function testJobs(run) {
  return run.jobs.filter(job => (job.is_tpu_test || job.tpu || job.tests?.length || (job.name && !/(^| \/ )(build|decide|pr-test-tpu-finish|plan|discover[^/]*|collect-full-evaluations)$/.test(job.name))) && job.conclusion !== 'skipped');
}

function executionSummary(run) {
  const jobs = testJobs(run);
  const known = jobs.map(executionTime).filter(value => value !== null);
  return {seconds: known.length ? known.reduce((a, b) => a + b, 0) : null,
    incomplete: known.length < jobs.length,
    estimated: jobs.some(job => executionTime(job) !== null && duration(job.tpu?.execution_seconds) === null)};
}

function workflowLabel(run) {
  const name = run.workflow_name || String(run.name || 'Unknown workflow').replace(/ \[publish\]$/, '');
  if (name !== 'Accuracy Test (TPU)') return name;
  const selection = {'day-10am': '10 AM PT', 'day-5pm': '5 PM PT'}[run.workflow_selection] || run.workflow_selection || 'selection unknown';
  return `${name} · ${selection}`;
}

function nightlyResults(run) {
  const jobs = testJobs(run);
  const allFiles = jobs.length && jobs.every(job => job.tests?.length && job.tests.every(test => typeof test.passed === 'boolean') && (job.conclusion === 'success' || job.tests.some(test => test.passed === false)));
  const files = allFiles ? jobs.flatMap(job => job.tests) : [];
  const passed = allFiles ? files.filter(test => test.passed).length : jobs.filter(job => job.conclusion === 'success').length;
  const total = allFiles ? files.length : jobs.length;
  const partial = (passed > 0 && passed < total) || (jobs.some(job => job.conclusion === 'success') && jobs.some(job => job.conclusion !== 'success'));
  return {passed, total, unit: allFiles ? 'test files' : 'jobs', partial};
}

function safeGithubUrl(value) {
  try {
    const url = new URL(value);
    return url.protocol === 'https:' && url.hostname === 'github.com' && !url.username && !url.password ? url.href : null;
  } catch { return null; }
}

function sampleData(now = Date.now()) {
  const iso = ms => new Date(ms).toISOString();
  const runs = [];
  const files = ['test/registered/tpu/test_attention.py', 'test/registered/tpu/test_model_runner.py', 'test/registered/tpu/test_quantization.py', 'test/registered/tpu/test_sampling.py'];
  for (const lane of ['pr', 'nightly']) {
    for (let i = 0; i < 40; i++) {
      const id = 100000 + i + (lane === 'nightly' ? 1000 : 0);
      const ageHours = i < 8 ? i * 2.65 + 0.2 : 24 + (i - 8) * 61;
      const completed = now - ageHours * 3600000;
      const runtime = (lane === 'nightly' ? 1550 : 880) + ((i * 137) % 650);
      const started = completed - runtime * 1000;
      const classification = i === 2 && lane === 'pr' ? 'awaiting_trigger' : i === 5 ? 'skipped' : 'executed';
      const conclusion = classification === 'awaiting_trigger' ? null : classification === 'skipped' ? 'skipped' : i % 7 === 4 ? 'failure' : 'success';
      const jobs = classification === 'executed' ? [0, 1].map(j => {
        const wait = 25 + ((i * 31 + j * 23) % 200);
        const queue = i % 9 === 3 ? null : 40 + ((i * 67 + j * 43) % 260);
        return {
          id: id * 10 + j, name: j ? 'TPU · v6e-8 / integration' : 'TPU · v6e-8 / unit tests', url: `https://github.com/sgl-project/sglang-torchtpu/actions/runs/${id}/job/${id * 10 + j}`,
          status: 'completed', conclusion: j && conclusion === 'failure' ? 'failure' : 'success', created_at: iso(started - wait * 1000), started_at: iso(started + j * 25000), completed_at: iso(completed - j * 12000),
          github_wait_seconds: wait, runtime_seconds: runtime - j * 37,
          is_tpu_test: true, tpu: {test_started_at: iso(started + (400 + j * 25) * 1000), queue_seconds: queue, startup_seconds: 100 + (i * 13) % 70, execution_seconds: runtime - 400 - j * 30},
          tests: files.map((file, n) => ({file, elapsed: 50 + ((i * 37 + n * 71 + j * 19) % 240), passed: !(j && conclusion === 'failure' && n === 1)})),
        };
      }) : [];
      runs.push({id, attempt: i === 6 ? 2 : 1, lane, event: lane === 'nightly' ? 'schedule' : 'pull_request', workflow_selection: i % 4 === 1 ? 'day-10am' : 'day-5pm', workflow_name: lane === 'nightly' ? (i % 2 ? 'Accuracy Test (TPU)' : 'Nightly Test (TPU)') : 'PR Test (TPU)', name: lane === 'pr' ? 'TPU PR CI' : 'TPU Nightly CI', pr: lane === 'pr' ? 12600 + i : null,
        sha: (0xa18e29b + i * 539).toString(16).padEnd(40, '0'), url: `https://github.com/sgl-project/sglang-torchtpu/actions/runs/${id}`, created_at: iso(started - 150000),
        started_at: classification === 'awaiting_trigger' ? null : iso(started), completed_at: classification === 'awaiting_trigger' ? null : iso(completed),
        status: classification === 'awaiting_trigger' ? 'queued' : 'completed', conclusion, classification, jobs});
    }
  }
  return {schema_version: 1, generated_at: iso(now), runs, warnings: []};
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {duration, formatDuration, percentile, selectRuns, runDuration, safeGithubUrl, sampleData, runWait};
}

if (typeof document !== 'undefined') {
  const $ = id => document.getElementById(id);
  const state = {lane: 'pr', days: 1, ranges: {pr: 1, nightly: 3}, data: {runs: []}};
  const sample = new URLSearchParams(window.location.search).get('sample') === '1';
  const dateFormat = new Intl.DateTimeFormat(undefined, {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'});
  const axisDateFormat = new Intl.DateTimeFormat(undefined, {month: 'short', day: 'numeric'});
  const timeFormat = new Intl.DateTimeFormat(undefined, {hour: '2-digit', minute: '2-digit'});
  const dateText = date => Number.isFinite(Date.parse(date)) ? dateFormat.format(new Date(date)) : '—';
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  };
  const link = (label, url) => {
    const node = el('a', '', label);
    const safe = safeGithubUrl(url);
    if (safe) { node.href = safe; node.target = '_blank'; node.rel = 'noopener noreferrer'; }
    else { node.removeAttribute('href'); node.title = 'GitHub link unavailable'; }
    return node;
  };
  function status(run) {
    if (run.lane === 'nightly' && run.status === 'completed' && nightlyResults(run).partial) return {label: 'Partial', kind: 'partial', symbol: '◐'};
    if (run.classification === 'awaiting_trigger') return {label: 'Awaiting trigger', kind: 'pending', symbol: '◷'};
    if (run.classification === 'skipped' || run.conclusion === 'skipped') return {label: 'Skipped', kind: 'skipped', symbol: '⊘'};
    if (run.conclusion === 'success') return {label: 'Passed', kind: 'success', symbol: '✓'};
    if (run.conclusion === 'failure' || run.conclusion === 'timed_out' || run.conclusion === 'action_required') return {label: run.conclusion === 'timed_out' ? 'Timed out' : 'Failed', kind: 'failure', symbol: '×'};
    if (run.conclusion === 'cancelled') return {label: 'Cancelled', kind: 'skipped', symbol: '⊘'};
    if (run.status === 'completed') return {label: 'Unknown', kind: 'skipped', symbol: '?'};
    return {label: run.status === 'in_progress' ? 'Running' : 'Queued', kind: 'pending', symbol: '◷'};
  }
  function metric(label, value, detail, symbol) {
    const card = el('article', 'metric');
    const title = el('div', 'metric-label', label);
    title.append(el('span', 'metric-symbol', symbol));
    card.append(title, el('div', 'metric-value', value), el('div', 'metric-detail', detail));
    return card;
  }
  function renderMetrics(runs) {
    const executed = runs.filter(r => r.classification === 'executed');
    const completed = executed.filter(r => r.status === 'completed' && ['success', 'failure', 'timed_out', 'cancelled', 'action_required'].includes(r.conclusion));
    const passed = completed.filter(r => r.conclusion === 'success').length;
    const jobs = executed.flatMap(r => r.jobs);
    const queue = executed.map(runWait).filter(v => duration(v) !== null);
    const estimated = executed.filter(run => runWait(run) !== null && waitIsEstimated(run)).length;
    const queueDetail = `${queue.length} attempt${queue.length === 1 ? '' : 's'} · submission → first observed TPU start${estimated ? ` · ${estimated} estimate${estimated === 1 ? '' : 's'}` : ''}`;
    const summaries = completed.map(executionSummary).filter(summary => summary.seconds !== null);
    const execution = state.lane === 'nightly' ? summaries.map(summary => summary.seconds) : jobs.map(executionTime).filter(v => v !== null);
    const executionEstimates = jobs.filter(job => duration(job.tpu?.execution_seconds) === null && duration(job.tpu?.execution_seconds_estimate) !== null).length;
    const executionDetail = state.lane === 'nightly' ? `${execution.length} run totals · summed job time (parallel counted)${summaries.some(s => s.estimated) ? ' · log estimates' : ''}${summaries.some(s => s.incomplete) ? ` · ${summaries.filter(s => s.incomplete).length} incomplete totals` : ''}` : executionEstimates ? `${execution.length} jobs · ${executionEstimates} log estimate${executionEstimates === 1 ? '' : 's'} · suite wall time` : `${execution.length} measured jobs · includes container setup`;
    const skipped = runs.filter(r => r.classification === 'skipped').length;
    $('metrics').replaceChildren(
      metric('Workflow runs', String(runs.length), `${executed.length} executed · ${skipped} skipped`, '▦'),
      metric('Median Queue time', formatDuration(percentile(queue, .5)), `P95 ${formatDuration(percentile(queue, .95))} · ${queueDetail}`, '⇥'),
      metric('Median execution time', formatDuration(percentile(execution, .5)), `P95 ${formatDuration(percentile(execution, .95))} · ${executionDetail}`, '◷'),
      metric('Success rate', completed.length ? `${Math.round(passed / completed.length * 100)}%` : '—', `${passed} passed / ${completed.length} completed executions`, '✓'),
    );
  }

  function chart(target, runs, series, description) {
    const width = 560, height = 230, left = 51, right = 18, top = 25, bottom = 37;
    const end = Date.now(), start = end - state.days * 86400000;
    const points = runs.filter(r => r.classification === 'executed').slice().reverse();
    const seriesPoints = series.map(s => ({...s, values: points.filter(run => !s.matches || s.matches(run)).map(run => ({time: runTime(run), value: s.value(run), run}))}));
    const values = seriesPoints.flatMap(s => s.values.map(p => p.value)).filter(v => duration(v) !== null);
    if (!values.length) {
      $(target).replaceChildren(el('div', 'chart-empty', 'No measured timings in this time range.'));
      return;
    }
    const max = Math.max(...values, 60) * 1.18;
    const x = value => left + (value - start) / (end - start) * (width - left - right);
    const y = value => height - bottom - value / max * (height - top - bottom);
    const svgEl = (tag, attrs = {}, text) => {
      const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
      if (text !== undefined) node.textContent = text;
      return node;
    };
    const svg = svgEl('svg', {viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': description});
    svg.append(svgEl('title', {}, description), svgEl('desc', {}, `${points.length} executed runs. Hover or focus a point for its measured duration. Missing measurements are omitted.`));
    for (let i = 0; i <= 4; i++) {
      const value = max * i / 4, yy = y(value);
      svg.append(svgEl('line', {x1: left, x2: width - right, y1: yy, y2: yy, class: 'grid-line'}));
      svg.append(svgEl('text', {x: left - 10, y: yy + 3, 'text-anchor': 'end'}, value >= 3600 ? `${(value / 3600).toFixed(1)}h` : `${Math.round(value / 60)}m`));
    }
    for (let i = 0; i <= 4; i++) {
      const time = start + (end - start) * i / 4;
      svg.append(svgEl('text', {x: x(time), y: height - 10, 'text-anchor': i === 0 ? 'start' : i === 4 ? 'end' : 'middle'}, (state.days === 1 ? timeFormat : axisDateFormat).format(new Date(time))));
    }
    for (const s of seriesPoints) {
      // Separate segments preserve visible gaps for unknown measurements.
      let segment = [];
      const drawSegment = () => {
        if (segment.length > 1) svg.append(svgEl('polyline', {points: segment.join(' '), fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}));
        segment = [];
      };
      for (const p of s.values) {
        if (duration(p.value) === null) { drawSegment(); continue; }
        segment.push(`${x(p.time)},${y(p.value)}`);
      }
      drawSegment();
      for (const p of s.values) {
        if (duration(p.value) === null) continue;
        const label = `${s.name}: ${formatDuration(p.value)}${s.estimate?.(p.run) ? ' (estimate)' : ''} · ${p.run.name} · ${p.run.conclusion || p.run.status} · ${dateText(new Date(p.time).toISOString())}`;
        const point = svgEl('circle', {cx: x(p.time), cy: y(p.value), r: 3.5, fill: s.color, tabindex: 0, 'aria-label': label});
        point.append(svgEl('title', {}, label));
        svg.append(point);
      }
    }
    $(target).replaceChildren(svg);
  }
  function phase(label, seconds) {
    const box = el('div', 'phase');
    box.append(el('span', '', label), el('strong', '', formatDuration(seconds)));
    return box;
  }
  function renderJob(job, run) {
    const details = el('details', 'job');
    const summary = el('summary', 'job-summary');
    const s = status(job);
    const title = el('span', 'job-title');
    title.append(el('span', 'chevron', '›'), el('span', `status-icon ${s.kind}`, s.symbol), el('span', '', job.name || 'Unnamed job'));
    const meta = el('span', 'job-meta');
    meta.append(el('span', `job-status ${s.kind}`, s.label), el('span', 'run-duration', formatDuration(job.runtime_seconds)));
    summary.append(title, meta);
    const body = el('div', 'job-body');
    const phases = el('div', 'phase-grid');
    phases.append(phase('GitHub wait', job.github_wait_seconds), phase('TPU queue', job.tpu?.queue_seconds), phase('TPU startup', job.tpu?.startup_seconds), phase(duration(job.tpu?.execution_seconds) !== null ? 'TPU execution' : 'TPU execution (log estimate)', executionTime(job)), phase('Job runtime', job.runtime_seconds));
    const observation = startObservation(job);
    phases.prepend(phase(`Submission → ${observation?.kind || 'TPU'} start${observation?.estimated ? ' (log estimate)' : ''}`, submissionWait(run, job)));
    if (duration(job.tpu?.cluster_wait_startup_estimate_seconds) !== null) phases.append(phase('GKE wait + startup (log estimate)', job.tpu.cluster_wait_startup_estimate_seconds));
    body.append(phases);
    const tests = Array.isArray(job.tests) ? job.tests.slice().sort((a, b) => (duration(b.elapsed) ?? -1) - (duration(a.elapsed) ?? -1)) : [];
    if (tests.length) {
      const table = el('table', 'tests-table');
      const caption = el('caption', 'skip-link', 'Test-file timings, slowest first');
      const head = el('thead');
      const headings = el('tr');
      for (const text of ['TEST FILE', 'DURATION', 'RESULT']) { const th = el('th', '', text); th.scope = 'col'; headings.append(th); }
      head.append(headings);
      const rows = el('tbody');
      const max = Math.max(...tests.map(t => duration(t.elapsed) ?? 0), 1);
      for (const test of tests) {
        const row = el('tr');
        const file = el('td', '', test.file || 'Unknown file');
        if (duration(test.elapsed) !== null) { const bar = el('div', 'test-bar'); bar.style.width = `${test.elapsed / max * 90}%`; file.append(bar); }
        row.append(file, el('td', '', formatDuration(test.elapsed)), el('td', test.passed === true ? 'success' : test.passed === false ? 'failure' : 'muted', test.passed === true ? 'Passed' : test.passed === false ? 'Failed' : '—'));
        rows.append(row);
      }
      table.append(caption, head, rows);
      body.append(table);
    } else body.append(el('p', 'no-tests', 'No per-file timings reported for this job.'));
    const jobLink = el('p', 'no-tests'); jobLink.append(link('Open job logs ↗', job.url)); body.append(jobLink);
    details.append(summary, body);
    return details;
  }
  function renderRun(run) {
    const details = el('details', 'run');
    const summary = el('summary', 'run-summary');
    const s = status(run);
    const identity = el('div');
    const title = el('div', 'workflow-title');
    title.append(el('span', `status-icon ${s.kind}`, s.symbol), el('span', 'name', run.lane === 'nightly' ? workflowLabel(run) : run.name || 'Unnamed workflow'));
    const subtitle = el('div', 'run-subtitle');
    if (run.pr) subtitle.append(el('span', '', `PR #${run.pr}`));
    subtitle.append(el('span', '', String(run.sha || 'Unknown commit').slice(0, 7)), el('span', '', `#${run.id} · attempt ${run.attempt || 1}`));
    identity.append(title, subtitle);
    const results = run.lane === 'nightly' ? nightlyResults(run) : null;
    const statusLabel = results?.total ? `${s.label} · ${results.passed}/${results.total} ${results.unit} passed` : s.label;
    summary.append(identity, el('span', `status-pill ${s.kind}`, statusLabel), el('span', 'run-duration', run.classification === 'executed' ? formatDuration(runDuration(run)) : '—'), el('span', 'run-time', dateText(run.started_at || run.created_at)), el('span', 'chevron', '›'));
    const body = el('div', 'run-body');
    const heading = el('div', 'run-body-top');
    heading.append(el('span', '', `Started ${dateText(run.started_at)} · Completed ${dateText(run.completed_at)}`), link('View GitHub Actions run ↗', run.url));
    body.append(heading);
    if (run.jobs.length) for (const job of run.jobs) body.append(renderJob(job, run));
    else body.append(el('p', 'no-tests', run.classification === 'awaiting_trigger' ? 'This workflow is awaiting its CI trigger. No jobs have executed.' : 'No executed jobs were reported for this run.'));
    details.append(summary, body);
    return details;
  }
  function render() {
    document.querySelectorAll('[data-days]').forEach(button => {
      const active = Number(button.dataset.days) === state.days;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    const runs = selectRuns(state.data.runs, state.lane, state.days);
    $('pr-count').textContent = selectRuns(state.data.runs, 'pr', state.ranges.pr).length;
    $('nightly-count').textContent = selectRuns(state.data.runs, 'nightly', state.ranges.nightly).length;
    $('run-count').textContent = runs.length;
    renderMetrics(runs);
    const durationRuns = runs.filter(r => r.status === 'completed' && (state.lane === 'nightly' || r.conclusion === 'success'));
    $('duration-description').textContent = state.lane === 'pr' ? 'Elapsed wall time · passed runs only' : 'Elapsed wall time · completed runs, including partial results';
    const workflowName = workflowLabel;
    const colors = ['#a99aff', '#54d6dd', '#efbc70', '#f28eb2', '#79d6a5'];
    const workflows = [...new Set(runs.map(workflowName))].sort();
    const durationSeries = state.lane === 'nightly' ? workflows.map((name, index) => ({
      name, color: colors[index % colors.length], value: runDuration, matches: run => workflowName(run) === name,
    })) : [{name: 'Duration', color: colors[0], value: runDuration}];
    $('duration-keys').replaceChildren(...durationSeries.map(series => {
      const key = el('span', 'chart-key');
      const swatch = el('i'); swatch.style.background = series.color;
      key.append(swatch, document.createTextNode(series.name));
      return key;
    }));
    chart('duration-chart', durationRuns, durationSeries, state.lane === 'pr' ? 'Workflow elapsed wall time for passed PR runs only' : 'Separate workflow curves for completed nightly runs, including partial results');
    $('wait-description').textContent = 'Submission to first TPU job start · historical log estimates included';
    $('wait-keys').textContent = 'Submission → first TPU job start';
    chart('wait-chart', runs, [{name: 'Submission → first TPU job start', color: '#efbc70', value: runWait, estimate: waitIsEstimated}], 'Workflow submission through build, resource wait and startup to first TPU job execution');
    $('runs').replaceChildren(...(runs.length ? runs.map(renderRun) : [el('div', 'empty-state', `No ${state.lane === 'pr' ? 'pull request' : 'nightly'} runs in the last ${state.days === 1 ? '24 hours' : `${state.days} days`}. Try a longer time range.`)]));
    $('dashboard').setAttribute('aria-busy', 'false');
  }
  document.querySelectorAll('[data-lane]').forEach(button => button.addEventListener('click', () => {
    state.lane = button.dataset.lane;
    state.days = state.ranges[state.lane];
    document.querySelectorAll('[data-lane]').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', b === button); });
    render();
  }));
  document.querySelectorAll('[data-days]').forEach(button => button.addEventListener('click', () => {
    state.days = Number(button.dataset.days);
    state.ranges[state.lane] = state.days;
    document.querySelectorAll('[data-days]').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', b === button); });
    render();
  }));
  async function load() {
    try {
      let data;
      if (sample) data = sampleData();
      else {
        const response = await fetch('./data.json', {cache: 'no-store'});
        if (!response.ok) throw new Error(`Snapshot request returned HTTP ${response.status}.`);
        data = await response.json();
      }
      if (data.schema_version !== 1 || !Array.isArray(data.runs)) throw new Error('Unsupported or invalid dashboard snapshot.');
      if (!data.runs.every(run => run && typeof run === 'object' && Array.isArray(run.jobs) && run.jobs.every(job => job && typeof job === 'object'))) throw new Error('The snapshot has invalid workflow or job records.');
      state.data = data;
      $('data-badge').textContent = sample ? 'SAMPLE DATA' : 'COLLECTED DATA';
      $('data-badge').className = `badge ${sample ? 'sample' : 'live'}`;
      $('updated').textContent = `Updated ${dateText(data.generated_at)}`;
      $('updated').title = data.generated_at || 'Unknown collection time';
      if (sample) {
        $('notice').hidden = false;
        $('notice').append(el('strong', '', 'You’re viewing sample data. '), document.createTextNode('These deterministic examples demonstrate the dashboard and do not represent actual CI results. '));
        const real = el('a', '', 'View collected data ↗'); real.href = './'; $('notice').append(real);
        $('sample-link').hidden = true;
      } else if (Array.isArray(data.warnings) && data.warnings.length) {
        $('notice').hidden = false;
        $('notice').textContent = `Collection notes: ${data.warnings.map(w => typeof w === 'string' ? w : JSON.stringify(w)).join(' · ')}`;
      }
      render();
    } catch (error) {
      document.body.dataset.error = 'true';
      $('data-badge').textContent = 'DATA UNAVAILABLE';
      $('updated').textContent = 'Snapshot could not be loaded';
      $('notice').hidden = false;
      $('notice').replaceChildren(document.createTextNode(`${error.message} Publish data.json next to this page, or `));
      const demo = el('a', '', 'explore the sample dashboard ↗'); demo.href = '?sample=1'; $('notice').append(demo);
      $('runs').replaceChildren(el('div', 'empty-state', 'Workflow data is unavailable. Sample mode is always available using the link above.'));
      $('dashboard').setAttribute('aria-busy', 'false');
    }
  }
  load();
}
