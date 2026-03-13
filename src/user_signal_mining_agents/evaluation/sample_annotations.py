"""Script to sample outputs for human annotation, creating blinded pairs."""

import argparse
import json
import random
from pathlib import Path

from .. import console as con
from ..config import get_settings
from ..schemas import HumanAnnotationTask, SynthesisResult


def _load_overall_preference(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    scores = payload.get("scores")
    if not isinstance(scores, dict):
        return None

    overall = scores.get("overall_preference")
    if isinstance(overall, int | float):
        return float(overall)
    return None


def sample_for_annotation(
    num_samples: int = 30,
    output_dir: Path | None = None,
    seed: int = 17,
) -> None:
    settings = get_settings()
    out_dir = output_dir or settings.run_artifacts_dir / "_human_annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    available_tasks = []

    # Gather all completed run directories.
    for run_dir in settings.run_artifacts_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue

        baseline_path = run_dir / "baseline.json"
        pipeline_path = run_dir / "pipeline.json"
        judge_p_path = run_dir / "judge_pipeline.json"
        judge_b_path = run_dir / "judge_baseline.json"

        if not all(p.exists() for p in (baseline_path, pipeline_path, judge_p_path, judge_b_path)):
            continue

        try:
            baseline = SynthesisResult.model_validate_json(baseline_path.read_text(encoding="utf-8"))
            pipeline = SynthesisResult.model_validate_json(pipeline_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        judge_b_overall = _load_overall_preference(judge_b_path)
        judge_p_overall = _load_overall_preference(judge_p_path)
        if judge_b_overall is None or judge_p_overall is None:
            continue

        diff = judge_p_overall - judge_b_overall
        if diff > 0:
            category = "pipeline_win"
        elif diff < 0:
            category = "baseline_win"
        else:
            category = "tie"

        available_tasks.append(
            {
                "prompt_id": run_dir.name,
                "baseline": baseline,
                "pipeline": pipeline,
                "category": category,
            }
        )

    con.step("sampling", f"Found {len(available_tasks)} completed prompts.")

    if not available_tasks:
        con.warning("No valid evaluation results found to sample.")
        return

    sampled_tasks = []

    # Stratified sampling.
    samples_per_category = max(1, num_samples // 3)
    categories = {"pipeline_win": [], "baseline_win": [], "tie": []}

    for task in available_tasks:
        categories[task["category"]].append(task)

    for tasks in categories.values():
        k = min(len(tasks), samples_per_category)
        if k:
            sampled_tasks.extend(rng.sample(tasks, k))

    # If we still need more, sample randomly from the rest.
    remaining_needed = num_samples - len(sampled_tasks)
    if remaining_needed > 0:
        remaining_tasks = [task for task in available_tasks if task not in sampled_tasks]
        k = min(len(remaining_tasks), remaining_needed)
        if k:
            sampled_tasks.extend(rng.sample(remaining_tasks, k))

    tasks_created = []
    for task in sampled_tasks:
        is_a_baseline = rng.choice([True, False])

        system_a = task["baseline"] if is_a_baseline else task["pipeline"]
        system_b = task["pipeline"] if is_a_baseline else task["baseline"]
        mapping = {
            "system_a": "baseline" if is_a_baseline else "pipeline",
            "system_b": "pipeline" if is_a_baseline else "baseline",
        }

        evidence_dict = {snippet.snippet_id: snippet for snippet in system_a.retrieved_evidence}
        for snippet in system_b.retrieved_evidence:
            if snippet.snippet_id not in evidence_dict:
                evidence_dict[snippet.snippet_id] = snippet

        task_id = f"task_{task['prompt_id']}"
        annotation_task = HumanAnnotationTask(
            task_id=task_id,
            prompt=task["baseline"].prompt,
            retrieved_evidence=list(evidence_dict.values()),
            system_a_focus_points=system_a.focus_points,
            system_b_focus_points=system_b.focus_points,
            ground_truth_mapping=mapping,
        )

        out_path = out_dir / f"{task_id}.json"
        out_path.write_text(annotation_task.model_dump_json(indent=2), encoding="utf-8")
        tasks_created.append(annotation_task)

    con.cached("DONE", f"Wrote {len(tasks_created)} annotation tasks to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample model outputs for human annotation")
    parser.add_argument("--num", type=int, default=30, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for deterministic task sampling")
    args = parser.parse_args()
    sample_for_annotation(num_samples=args.num, seed=args.seed)
