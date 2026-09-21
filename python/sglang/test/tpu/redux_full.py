"""Full Redux 2.0 with the same explicit label corrections as the September run."""


def run_full_redux(base_url):
    import json
    import tempfile
    from collections import Counter
    from pathlib import Path

    from sgl_eval.evals._multichoice import make_score_one_fn
    from sgl_eval.predictions import PredictionsWriter
    from sgl_eval.runner import run_examples
    from sgl_eval.sampler import ChatCompletionSampler
    from sgl_eval.types import Example, GenConfig

    from sglang.test.tpu.redux_dataset import prepare

    out = Path(tempfile.mkdtemp(prefix="mmlu-redux-"))
    prepare(out)
    rows = [
        json.loads(line) for line in (out / "dataset.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 5700
    by_id = {r["id"]: r for r in rows}
    examples = [
        Example(
            id=r["id"],
            inputs={
                "problem": r["question"]
                + "\n\n"
                + "\n".join(
                    f'{"ABCD"[i]}. {choice}' for i, choice in enumerate(r["choices"])
                )
            },
            target="ABCD"[r["scoring_answer"]],
            meta={"subject": r["subject"]},
        )
        for r in rows
    ]
    sampler = ChatCompletionSampler(base_url=base_url.rstrip("/") + "/v1")
    gen = GenConfig(temperature=0.6, top_p=0.95, max_tokens=65536)

    def sample_fn(ex, rep):
        prompt = (
            ex.inputs["problem"]
            + "\n\nThink step by step, then give your final answer as a single letter A, B, C, or D inside \\boxed{}."
        )
        return sampler([{"role": "user", "content": prompt}], gen)

    print(
        "Starting full MMLU-Redux-2.0: 5700 rows, 57 subjects, repeat=1, threads=128, max_tokens=65536",
        flush=True,
    )
    with PredictionsWriter(out, 1) as writer:
        result = run_examples(
            name="mmlu-redux-2.0",
            examples=examples,
            sample_fn=sample_fn,
            score_one_fn=make_score_one_fn(),
            num_threads=128,
            n_repeats=1,
            on_sample_scored=writer,
        )
    valid = [r for r in result.per_example if by_id[r.example.id]["scorable"]]
    correct = sum(r.scores[0] for r in valid)
    summary = {
        "dataset": json.loads((out / "dataset_manifest.json").read_text()),
        "completed": result.num_examples,
        "partial": result.partial,
        "scorable_completed": len(valid),
        "correct": correct,
        "corrected_valid_accuracy": correct / len(valid) if valid else None,
        "elapsed_seconds": result.latency,
        "output_tokens_per_second": result.output_throughput,
        "finish_reasons": dict(
            Counter(s.finish_reason for r in result.per_example for s in r.samples)
        ),
        "protocol": "All 5700 rows generated; EM excludes invalid/ambiguous rows; wrong_groundtruth fixed only for unambiguous index/letter annotations. One zero-shot CoT response per row.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    from sglang.test.tpu.full_eval_artifacts import archive_results

    archive_results(out)
    if result.partial or result.num_examples != 5700 or len(valid) != 5431:
        raise RuntimeError("Incomplete Redux evaluation or changed scoring population")
    if result.aggregate.get("error_rate", 0) != 0:
        raise RuntimeError("Redux evaluation contains request errors")
    return {
        "score": summary["corrected_valid_accuracy"],
        "latency": result.latency,
        "output_throughput": result.output_throughput,
    }
