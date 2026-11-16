const test = require('node:test');
const assert = require('node:assert/strict');
const { duration, formatDuration, percentile, selectRuns, runDuration, safeGithubUrl, sampleData } = require('./app.js');

test('missing and negative durations stay unknown while measured zero is valid', () => {
  assert.equal(duration(null), null);
  assert.equal(duration(undefined), null);
  assert.equal(duration(-2), null);
  assert.equal(duration('4'), null);
  assert.equal(duration(0), 0);
  assert.equal(formatDuration(null), '—');
  assert.equal(formatDuration(0), '0s');
});

test('workflow duration uses wall time rather than summing parallel jobs', () => {
  assert.equal(runDuration({ started_at: '2026-11-06T01:00:00Z', completed_at: '2026-11-06T01:10:00Z', jobs: [{runtime_seconds: 600}, {runtime_seconds: 600}] }), 600);
  assert.equal(runDuration({ jobs: [{runtime_seconds: 600}] }), null);
});

test('time window uses completion then start time, lane and deterministic newest first', () => {
  const now = Date.parse('2026-11-06T12:00:00Z');
  const runs = [
    {id: 1, lane:'pr', started_at:'2026-11-04T10:00:00Z', completed_at:'2026-11-06T11:00:00Z'},
    {id: 2, lane:'pr', started_at:'2026-11-06T10:00:00Z'},
    {id: 3, lane:'nightly',event:'schedule', completed_at:'2026-11-06T11:50:00Z'},
    {id: 4, lane:'pr', completed_at:'2026-11-05T11:59:59Z'},
    {id: 5, lane:'pr', completed_at:'2026-11-07T00:00:00Z'},
  ];
  assert.deepEqual(selectRuns(runs, 'pr', 1, now).map(r => r.id), [1, 2]);
});

test('percentiles ignore unknown values and preserve zero', () => {
  assert.equal(percentile([null, undefined, 0, 10, 20], .5), 10);
  assert.equal(percentile([null], .95), null);
});

test('only HTTPS GitHub links are accepted', () => {
  assert.equal(safeGithubUrl('https://github.com/org/repo/actions/runs/1'), 'https://github.com/org/repo/actions/runs/1');
  for (const url of ['javascript:alert(1)', 'https://github.com.evil.test/run', 'https://github.com@evil.test', 'http://github.com/a', 'https://example.com']) assert.equal(safeGithubUrl(url), null);
});

test('sample data is reproducible and includes both lanes, unknowns and old history', () => {
  const now = Date.parse('2026-11-06T12:00:00Z');
  const data = sampleData(now);
  assert.deepEqual(data, sampleData(now));
  assert.equal(data.schema_version, 1);
  assert.ok(selectRuns(data.runs, 'pr', 1, now).length > 0);
  assert.ok(selectRuns(data.runs, 'nightly', 1, now).length > 0);
  assert.ok(selectRuns(data.runs, 'pr', 90, now).length > selectRuns(data.runs, 'pr', 1, now).length);
  assert.ok(data.runs.some(r => r.classification === 'awaiting_trigger'));
});

// Exercise the real renderer without adding a browser dependency to the dashboard.
async function renderPage(search, fetch) {
  const {readFileSync} = require('node:fs');
  const vm = require('node:vm');
  class Element {
    constructor(tag) { this.tagName = tag; this.children = []; this.attributes = {}; this.dataset = {}; this.style = {}; this.listeners = {}; this._text = ''; this.classList = {toggle() {}}; }
    set textContent(value) { this._text = String(value); this.children = []; }
    get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
    append(...children) { this.children.push(...children); }
    prepend(...children) { this.children.unshift(...children); }
    replaceChildren(...children) { this._text = ''; this.children = children; }
    setAttribute(key, value) { this.attributes[key] = String(value); }
    removeAttribute(key) { delete this.attributes[key]; }
    addEventListener(type, listener) { this.listeners[type] = listener; }
  }
  const ids = new Map();
  const lanes = ['pr', 'nightly'].map(lane => { const node = new Element('button'); node.dataset.lane = lane; return node; });
  const days = [1, 3, 7, 30, 90].map(day => { const node = new Element('button'); node.dataset.days = String(day); return node; });
  const document = {
    body: new Element('body'),
    getElementById: id => { if (!ids.has(id)) ids.set(id, new Element('div')); return ids.get(id); },
    createElement: tag => new Element(tag),
    createElementNS: (_, tag) => new Element(tag),
    createTextNode: text => { const node = new Element('text'); node.textContent = text; return node; },
    querySelectorAll: selector => selector === '[data-lane]' ? lanes : days,
  };
  vm.runInNewContext(readFileSync(`${__dirname}/app.js`, 'utf8'), {document, window: {location: {search}}, URL, URLSearchParams, Intl, Date, fetch});
  await new Promise(resolve => setImmediate(resolve));
  return {document, ids, lanes, days};
}

test('explicit sample mode renders charts and nested jobs without fetching real data', async () => {
  const page = await renderPage('?sample=1', () => { throw new Error('Sample mode must not fetch'); });
  assert.equal(page.ids.get('data-badge').textContent, 'SAMPLE DATA');
  assert.equal(page.ids.get('duration-chart').children[0].tagName, 'svg');
  assert.match(page.ids.get('runs').textContent, /TPU startup/);
  assert.match(page.ids.get('runs').textContent, /test_attention.py/);
  const initialCount = Number(page.ids.get('run-count').textContent);
  page.days.at(-1).listeners.click();
  assert.ok(Number(page.ids.get('run-count').textContent) > initialCount);
  page.lanes[1].listeners.click();
  assert.match(page.ids.get('runs').textContent, /Nightly Test \(TPU\)|Accuracy Test \(TPU\)/);
  assert.equal(page.ids.get('dashboard').attributes['aria-busy'], 'false');
});

test('a missing snapshot shows an error and sample link without inventing live results', async () => {
  const page = await renderPage('', async () => ({ok:false, status:404}));
  assert.equal(page.document.body.dataset.error, 'true');
  assert.equal(page.ids.get('data-badge').textContent, 'DATA UNAVAILABLE');
  assert.match(page.ids.get('notice').textContent, /HTTP 404/);
  assert.ok(page.ids.get('notice').children.some(node => node.href === '?sample=1'));
  assert.doesNotMatch(page.ids.get('runs').textContent, /TPU PR CI/);
});

test('untrusted workflow names render as plain text and external run links are rejected', async () => {
  const name = '<img src=x onerror=alert(1)>';
  const data = {schema_version:1, generated_at:new Date().toISOString(), runs:[{id:1, attempt:1, lane:'pr', name, classification:'executed', status:'completed', conclusion:'success', started_at:new Date(Date.now()-60000).toISOString(), completed_at:new Date().toISOString(), url:'javascript:alert(1)', jobs:[]}]};
  const page = await renderPage('', async () => ({ok:true, json:async () => data}));
  assert.equal(page.ids.get('data-badge').textContent, 'COLLECTED DATA');
  assert.match(page.ids.get('runs').textContent, /<img src=x onerror=alert\(1\)>/);
  const runLink = page.ids.get('runs').children[0].children[1].children[0].children[1];
  assert.equal(runLink.href, undefined);
});

test('awaiting-trigger attempts are excluded from rows, counts and metrics in both lanes', async () => {
  const now = Date.now();
  const records = sampleData(now).runs;
  for (const lane of ['pr', 'nightly']) {
    const pending = {id: 'pending', lane, classification: 'awaiting_trigger', created_at: new Date(now).toISOString()};
    assert.ok(!selectRuns([...records, pending], lane, 90, now).some(r => r.classification === 'awaiting_trigger'));
  }
  const page = await renderPage('?sample=1', () => { throw new Error('No fetch'); });
  assert.doesNotMatch(page.ids.get('runs').textContent, /Awaiting trigger/);
  assert.doesNotMatch(page.ids.get('metrics').textContent, /awaiting/);
  assert.equal(Number(page.ids.get('run-count').textContent), Number(page.ids.get('pr-count').textContent));
});

test('nightly queue uses submission-to-start and execution uses phase measurements', async () => {
  const data = {schema_version:1, generated_at:new Date().toISOString(), runs:[{
    id:1, lane:'nightly',event:'schedule', classification:'executed', status:'completed', conclusion:'success',
    created_at:new Date(Date.now()-600000).toISOString(),
    started_at:new Date(Date.now()-600000).toISOString(), completed_at:new Date().toISOString(),
    jobs:[{name:'test', github_wait_seconds:30, runtime_seconds:600, tpu:{queue_seconds:10, startup_seconds:80, execution_seconds:300, test_started_at:new Date(Date.now()-480000).toISOString()}}, {name:'unknown', tpu:null}]
  }]};
  const page = await renderPage('', async () => ({ok:true, json:async () => data}));
  page.lanes[1].listeners.click();
  const cards = page.ids.get('metrics').children;
  assert.match(cards[0].textContent, /Workflow runs/);
  assert.match(cards[1].textContent, /Median Queue time/);
  assert.equal(cards[1].children[1].textContent, '2m 00s');
  assert.match(cards[1].textContent, /1 attempt/);
  assert.match(cards[2].textContent, /Median execution time/);
  assert.equal(cards[2].children[1].textContent, '5m 00s');
  assert.match(cards[2].textContent, /summed job time/);
  assert.match(cards[3].textContent, /Success rate/);
  data.runs[0].jobs[0].tpu = null;
  const missing = await renderPage('', async () => ({ok:true, json:async () => data}));
  missing.lanes[1].listeners.click();
  assert.equal(missing.ids.get('metrics').children[1].children[1].textContent, '—');
  assert.equal(missing.ids.get('metrics').children[2].children[1].textContent, '—');
});


test('PR defaults to 24h and nightly to 3d, preserving each lane selection', async () => {
  const page = await renderPage('?sample=1', () => { throw new Error('No fetch'); });
  assert.equal(page.days[0].attributes['aria-pressed'], 'true');
  page.lanes[1].listeners.click();
  assert.equal(page.days[1].attributes['aria-pressed'], 'true');
  page.days[2].listeners.click();
  page.lanes[0].listeners.click();
  assert.equal(page.days[0].attributes['aria-pressed'], 'true');
  page.lanes[1].listeners.click();
  assert.equal(page.days[2].attributes['aria-pressed'], 'true');
});

test('PR waiting spans attempt submission through first TPU test, including build and pod startup', () => {
  const {runWait} = require('./app.js');
  const run = {submitted_at:'2026-11-06T01:00:00Z', jobs:[
    {is_tpu_test:true, tpu:{test_started_at:'2026-11-06T01:12:00Z', queue_seconds:60}},
    {is_tpu_test:true, tpu:{test_started_at:'2026-11-06T01:10:00Z', queue_seconds:30}},
    {name:'build', runtime_seconds:300}
  ]};
  assert.equal(runWait(run), 600);
  run.jobs[0].tpu.test_started_at = null;
  assert.equal(runWait(run), 600);
  run.jobs[1].tpu.test_started_at = null;
  assert.equal(runWait(run), null);
  run.jobs = [];
  assert.equal(runWait(run), null);
});

test('PR duration shows only passed attempts; nightly duration includes completed partial runs', async () => {
  const now = Date.now();
  const data = {schema_version:1, generated_at:new Date(now).toISOString(), runs:[]};
  for (const lane of ['pr','nightly']) for (const conclusion of ['success','failure']) data.runs.push({
    id:data.runs.length+1,lane,event:'schedule',classification:'executed',status:'completed',conclusion,
    created_at:new Date(now-120000).toISOString(), started_at:new Date(now-100000).toISOString(),completed_at:new Date(now-1000).toISOString(),
    jobs:[{is_tpu_test:true,tpu:{test_started_at:new Date(now-60000).toISOString()}}]
  });
  const page = await renderPage('',async()=>({ok:true,json:async()=>data}));
  for (const lane of [0,1]) {
    page.lanes[lane].listeners.click();
    const svg=page.ids.get('duration-chart').children[0];
    assert.equal(svg.children.filter(node=>node.tagName==='circle').length,lane===0?1:2);
    assert.equal(page.ids.get('runs').children.length,2);
    assert.match(page.ids.get('metrics').children[3].textContent,/50%/);
    if(lane===0) assert.equal(page.ids.get('metrics').children[1].children[1].textContent,'1m 00s');
  }
});

test('nightly mixed results are partial and show passed/total test files, with job fallback', async () => {
  const base={id:1,lane:'nightly',event:'schedule',classification:'executed',status:'completed',conclusion:'failure',completed_at:new Date().toISOString()};
  const jobs=[{is_tpu_test:true,conclusion:'success',tests:[{passed:true}]},{is_tpu_test:true,conclusion:'failure',tests:[{passed:true},{passed:false}]}];
  const data={schema_version:1,runs:[{...base,jobs}]};
  const page=await renderPage('',async()=>({ok:true,json:async()=>data}));
  page.lanes[1].listeners.click();
  assert.match(page.ids.get('runs').children[0].children[0].textContent,/Partial/);
  assert.match(page.ids.get('runs').children[0].children[0].textContent,/2\/3 test files passed/);
  jobs[1].tests=[];
  const fallback=await renderPage('',async()=>({ok:true,json:async()=>data}));
  fallback.lanes[1].listeners.click();
  assert.match(fallback.ids.get('runs').children[0].children[0].textContent,/Partial.*1\/2 jobs passed/);
});

test('active nightlies stay Running and include unfinished jobs in the denominator', async () => {
  const data={schema_version:1,runs:[{id:1,lane:'nightly',event:'schedule',classification:'executed',status:'in_progress',started_at:new Date().toISOString(),jobs:[
    {is_tpu_test:true,conclusion:'success',tests:[{passed:true}]},
    {is_tpu_test:true,conclusion:'failure',tests:[{passed:false}]},
    {is_tpu_test:true,status:'in_progress',conclusion:null,tests:[]}
  ]}]};
  const page=await renderPage('',async()=>({ok:true,json:async()=>data}));page.lanes[1].listeners.click();
  assert.match(page.ids.get('runs').children[0].children[0].textContent,/Running.*1\/3 jobs passed/);
  assert.doesNotMatch(page.ids.get('runs').children[0].children[0].textContent,/Partial/);
});

test('nightly passed/total excludes planning and result-collection jobs', async () => {
  const data={schema_version:1,runs:[{id:1,lane:'nightly',event:'schedule',classification:'executed',status:'completed',conclusion:'failure',completed_at:new Date().toISOString(),jobs:[
    {name:'plan',conclusion:'success',tests:[]},
    {name:'discover-multihost',conclusion:'success',tests:[]},
    {name:'collect-full-evaluations',conclusion:'success',tests:[]},
    {name:'build / build',conclusion:'success',tests:[]},
    {name:'test_tpu_model / test_tpu_model',conclusion:'failure',tests:[]}
  ]}]};
  const page=await renderPage('',async()=>({ok:true,json:async()=>data}));page.lanes[1].listeners.click();
  assert.match(page.ids.get('runs').children[0].children[0].textContent,/Failed.*0\/1 jobs passed/);
});

test('nightly duration connects completed runs within each workflow and labels each curve', async () => {
  const now=Date.now();
  const runs=[['Accuracy',true],['Nightly',true],['Accuracy',true],['Nightly',true],['Accuracy',false]].map(([workflow,passed],i)=>({
    id:i,lane:'nightly',event:'schedule',name:`custom title ${i}`,workflow_name:workflow,classification:'executed',status:'completed',conclusion:passed?'success':'failure',
    started_at:new Date(now-(i+1)*600000).toISOString(),completed_at:new Date(now-i*600000-60000).toISOString(),jobs:[]
  }));
  const page=await renderPage('',async()=>({ok:true,json:async()=>({schema_version:1,runs})}));page.lanes[1].listeners.click();
  const svg=page.ids.get('duration-chart').children[0];
  const lines=svg.children.filter(node=>node.tagName==='polyline');
  assert.equal(lines.length,2);
  assert.notEqual(lines[0].attributes.stroke,lines[1].attributes.stroke);
  assert.equal(svg.children.filter(node=>node.tagName==='circle').length,5);
  assert.match(page.ids.get('duration-keys').textContent,/Accuracy/);
  assert.match(page.ids.get('duration-keys').textContent,/Nightly/);
});

test('historical PR wait uses a labeled log estimate, preferring exact starts when available', async () => {
  const {runWait}=require('./app.js');
  const run={id:1,lane:'pr',classification:'executed',status:'completed',conclusion:'success',created_at:new Date(Date.now()-180000).toISOString(),completed_at:new Date().toISOString(),jobs:[
    {is_tpu_test:true,tpu:{test_started_at_estimate:new Date(Date.now()-60000).toISOString()}}
  ]};
  assert.ok(Math.abs(runWait(run)-120)<1);
  const page=await renderPage('',async()=>({ok:true,json:async()=>({schema_version:1,runs:[run]})}));
  assert.match(page.ids.get('metrics').children[1].textContent,/1 estimate/);
  run.jobs[0].tpu.test_started_at=new Date(Date.now()-120000).toISOString();
  assert.ok(Math.abs(runWait(run)-60)<1);
});


test('execution card uses labeled historical wall-time estimates without summing parallel files', async () => {
  const job={is_tpu_test:true,tpu:{execution_seconds_estimate:300},tests:[{elapsed:200,passed:true},{elapsed:200,passed:true}]};
  const data={schema_version:1,runs:[{id:1,lane:'pr',classification:'executed',status:'completed',conclusion:'success',completed_at:new Date().toISOString(),jobs:[job]}]};
  const page=await renderPage('',async()=>({ok:true,json:async()=>data}));
  assert.equal(page.ids.get('metrics').children[2].children[1].textContent,'5m 00s');
  assert.match(page.ids.get('metrics').children[2].textContent,/1 log estimate/);
  job.tpu.execution_seconds=240;
  const exact=await renderPage('',async()=>({ok:true,json:async()=>data}));
  assert.equal(exact.ids.get('metrics').children[2].children[1].textContent,'4m 00s');
});

test('wait labels follow the selected source when a log estimate and container start both exist', async () => {
  const now=Date.now();
  const job={is_tpu_test:true,tpu:{started_at:new Date(now-120000).toISOString(),test_started_at_estimate:new Date(now-60000).toISOString()}};
  const data={schema_version:1,runs:[{id:1,lane:'pr',classification:'executed',status:'completed',conclusion:'success',created_at:new Date(now-180000).toISOString(),completed_at:new Date(now).toISOString(),jobs:[job]}]};
  const page=await renderPage('',async()=>({ok:true,json:async()=>data}));
  assert.equal(page.ids.get('metrics').children[1].children[1].textContent,'2m 00s');
  assert.match(page.ids.get('metrics').children[1].textContent,/1 estimate/);
  delete job.tpu.test_started_at_estimate;
  const exact=await renderPage('',async()=>({ok:true,json:async()=>data}));
  assert.equal(exact.ids.get('metrics').children[1].children[1].textContent,'1m 00s');
  assert.doesNotMatch(exact.ids.get('runs').textContent,/Submission → test start \(log estimate\)/);
});


test('nightly view excludes manual and unknown triggers without affecting PRs', () => {
  const now=Date.now(), completed_at=new Date(now).toISOString();
  const runs=[{id:1,lane:'nightly',event:'schedule',completed_at},{id:2,lane:'nightly',event:'workflow_dispatch',completed_at},{id:3,lane:'nightly',completed_at},{id:4,lane:'pr',event:'pull_request',completed_at}];
  assert.deepEqual(selectRuns(runs,'nightly',3,now).map(r=>r.id),[1]);
  assert.deepEqual(selectRuns(runs,'pr',1,now).map(r=>r.id),[4]);
});

test('nightly execution is median of summed job execution per run, not median individual job', async () => {
  const now=Date.now();
  const data={schema_version:1,runs:[{id:1,lane:'nightly',event:'schedule',classification:'executed',status:'completed',conclusion:'success',completed_at:new Date(now).toISOString(),jobs:[
    {is_tpu_test:true,tpu:{execution_seconds:600}}, {is_tpu_test:true,tpu:{execution_seconds:1200}}, {name:'build / build',runtime_seconds:9999}
  ]}]};
  const page=await renderPage('',async()=>({ok:true,json:async()=>data}));page.lanes[1].listeners.click();
  assert.equal(page.ids.get('metrics').children[2].children[1].textContent,'30m 00s');
  assert.match(page.ids.get('metrics').children[2].textContent,/1 run totals.*summed job time/);
});

test('accuracy schedule selections get distinct row labels and duration curves', async () => {
  const now=Date.now();
  const runs=['day-10am','day-5pm'].flatMap((workflow_selection,k)=>[0,1].map(i=>({id:k*2+i,lane:'nightly',event:'schedule',workflow_name:'Accuracy Test (TPU)',workflow_selection,name:'Accuracy Test (TPU)',classification:'executed',status:'completed',conclusion:'failure',started_at:new Date(now-(i+2)*600000).toISOString(),completed_at:new Date(now-(i+1)*600000).toISOString(),jobs:[]})));
  const page=await renderPage('',async()=>({ok:true,json:async()=>({schema_version:1,runs})}));page.lanes[1].listeners.click();
  assert.match(page.ids.get('duration-keys').textContent,/10 AM PT/);
  assert.match(page.ids.get('duration-keys').textContent,/5 PM PT/);
  assert.match(page.ids.get('runs').textContent,/10 AM PT/);
  assert.match(page.ids.get('runs').textContent,/5 PM PT/);
  assert.equal(page.ids.get('duration-chart').children[0].children.filter(n=>n.tagName==='polyline').length,2);
});
