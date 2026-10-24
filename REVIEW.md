# Writing a PR description

How to write the description of a pull request to `sgl-project/sglang-torchtpu`.
A reviewer should be able to read it in two minutes, run what it shows, and get
the same result. For what reviewers check in the code, see
[AGENTS.md](AGENTS.md).

## Rules

1. **Keep the template's sections**, in this order: Motivation, Modifications,
   Accuracy Tests, Speed Tests and Profiling, Checklist. Add a short "Notes for
   reviewers" only if there is something a reviewer must know.

2. **Keep it short.** Aim for what a reviewer reads in two minutes, usually
   under 70 lines. If a draft feels complete, cut it to about a third.

3. **Describe the PR as it is at its head, not its history.** The description
   becomes the squash-merge commit message. When review changes the PR,
   rewrite the description so it is still true. Do not add a "Review changes"
   list, notes per commit, test runs on earlier commits, or before/after tables
   of what an earlier version of the PR did. Those go in commit messages and
   review replies.

4. **Describe what the PR does, not what it does not do.** Do not list limits
   or caveats that the code does not have, and do not add "left to a
   follow-up" notes. The one exception is the "Not run" line in Notes for
   reviewers.

5. **Modifications: one entry per concept.** Use a short lettered list, one
   entry per main part (for example: a. index op, b. sparse attention op,
   c. KV pool), with two or three sentences each.
   - Do not list files, classes, or methods. Reviewers read the diff.
   - Spend words on the design point a reviewer needs before reading the code,
     for example why the cache is not copied. A small table is fine for that.
   - Say what the diff does not show: a new startup rejection, a changed
     default, a lazy import.
   - Put supporting changes (argument resolution, config, recipe) in one
     closing sentence.

6. **Accuracy Tests: show a run a reviewer can repeat.** Three things:
   - the command that starts the server;
   - one simple prompt sent with `curl`, such as "What is the capital of
     France?";
   - the output it returned.

   Give any other check one line: what ran and the verdict, for example
   "kernel tests pass on TPU" or "outputs match the dense backend". No pass
   counts, lists of test files, timings, logprob tables, or `<details>` blocks
   of evidence. Those belong in the tests, in review replies, or in a linked
   doc.

7. **Speed Tests: give the exact benchmark command, then one table.**
   - Use the standard client, `benchmark/tpu/bench_serving.sh`, and paste the
     command line that was run.
   - Use the standard workload: 8k input tokens and 1k output tokens
     (`ISL=8192 OSL=1024`), on the recipe's default context.
   - Report the client's own summary numbers: output throughput, TTFT, TPOT.
   - Show one table. No second tables, no analysis of where the time goes, and
     no restating a configuration that the named recipe already holds.
   - Do not compare with other backends or other models. Compare with the code
     before the PR only when the PR is a performance change.

8. **Say which commit the results come from.** Name the commit the tests and
   numbers ran at, normally the head, and say whether CI agrees.

9. **Serve with the recipe as shipped.** Use the model's recipe under
   `benchmark/tpu/recipe/`. If the demo needs a different setting, use a knob
   that the recipe or script already has, and show it in the command: an
   environment variable the recipe reads (such as
   `GLM_FLASH_DENSE_MAX_CONTEXT`) or a `bench_serving.sh` variable. Also show
   the environment variables the feature needs. Do not add an override that
   has no knob; if the default is wrong for the demo, fix the recipe in the PR.

10. **Link what you mention.** A function, a line, a commit, a PR, or a review
    thread gets a link: a permalink at a fixed commit with line anchors
    (`.../blob/<sha>/path/to/file.py#L10-L20`), or the thread's
    `#discussion_r<id>` URL.

11. **Write in plain English.** Full sentences, ordinary words, one idea per
    sentence. Say "the indexer picks the keys" rather than a compressed phrase
    a reader has to decode.

12. **No tool attribution.** Do not add "Generated with ..." footers or
    co-author lines for coding tools to the description or the commit
    messages. (Vendored files keep their "Adapted from" header.)

## Template

````markdown
## Motivation

<Two or three sentences: what was missing, and what this PR makes work.>

## Modifications

**a. <Main part>**

<Two or three sentences on what it does.>

**b. <Main part>**

<Two or three sentences on what it does.>

**c. <Main part>**

<Two or three sentences. If one design point needs explaining, explain it
here, with a small table if that helps.>

Smaller changes that go with these: <one sentence>.

## Accuracy Tests

`<model>` on `<hardware>`, at `<commit>`.

Serve, from the repo root:

```bash
python -m benchmark.tpu.launch_server --config recipe/configs/<model>/server.py
# The recipe starts the server with: <the key flags>
```

Send a prompt:

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "default",
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "max_tokens": 64, "temperature": 0
}'
```

Output:

```json
{"choices": [{"message": {"role": "assistant", "content": "The capital of France is Paris."}}]}
```

<One line per other check: what ran, and the verdict.>

## Speed Tests and Profiling

```bash
cd benchmark/tpu
MODEL=<model> DATASET_NAME=random-ids TOKENIZE_PROMPT=1 \
    ISL=8192 OSL=1024 BATCH_SIZE=<n> NUM_PROMPTS=<n> ./bench_serving.sh
```

| Input / output tokens | Concurrency | Output throughput | Mean TTFT | Mean TPOT |
|---|---|---|---|---|
| 8,192 / 1,024 | <n> | <tok/s> | <ms> | <ms> |

## Checklist

<the template's checklist, ticked honestly>

## Notes for reviewers

- Not run: <hardware or paths this PR was not tested on>.
- <Anything else a reviewer must know.>
````
