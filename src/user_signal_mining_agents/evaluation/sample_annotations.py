"""Script to sample outputs for human annotation, creating blinded pairs."""

import json
import random
from pathlib import Path
from typing import Literal

from ..config import get_settings
from ..schemas import (
    HumanAnnotationTask,
    JudgeResult,
    SynthesisResult,
)
from .. import console as con


def sample_for_annotation(
    num_samples: int = 30,
    output_dir: Path | None = None,
) -> None:
    settings = get_settings()
    out_dir = output_dir or settings.run_artifacts_dir / "_human_annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    available_tasks = []

    # Gather all completed run directories
    for run_dir in settings.run_artifacts_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue

        baseline_path = run_dir / "baseline.json"
        pipeline_path = run_dir / "pipeline.json"
        judge_p_path = run_dir / "judge_pipeline.json"
        judge_b_path = run_dir / "judge_baseline.json"

        if not all(
            p.exists()
            for p in (baseline_path, pipeline_path, judge_p_path, judge_b_path)
        ):
            continue

        try:
            baseline = SynthesisResult.model_validate_json(baseline_path.read_text(encoding="utf-8"))
            pipeline = SynthesisResult.model_validate_json(pipeline_path.read_text(encoding="utf-8"))
            judge_b = JudgeResult.model_validate_json(judge_b_path.read_text(encoding="utf-8"))
            judge_p = JudgeResult.model_validate_json(judge_p_path.read_text(encoding="utf-8"))
        except Exception:
            # Skip invalid/corrupt JSON
            continue

        diff = judge_p.scores.overall_avg - judge_b.scores.overall_avg
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
        con.print("[yellow]No valid evaluation results found to sample.[/yellow]")
        return
        
    sampled_tasks = []

    # Stratified sampling
    samples_per_category = max(1, num_samples // 3)
    categories = {"pipeline_win": [], "baseline_win": [], "tie": []}
    
    for task in available_tasks:
        categories[task["category"]].append(task)
        
    for cat, tasks in categories.items():
        k = min(len(tasks), samples_per_category)
        sampled_tasks.extend(random.sample(tasks, k))
        
    # If we still need more, sample randomly from the rest
    remaining_needed = num_samples - len(sampled_tasks)
    if remaining_needed > 0:
        remaining_tasks = [t for t in available_tasks if t not in sampled_tasks]
        k = min(len(remaining_tasks), remaining_needed)
        sampled_tasks.extend(random.sample(remaining_tasks, k))
        
    # Create the blinded annotations
    tasks_created = []
    
    for i, t in enumerate(sampled_tasks, 1):
        is_a_baseline = random.choice([True, False])
        
        system_a = t["baseline"] if is_a_baseline else t["pipeline"]
        system_b = t["pipeline"] if is_a_baseline else t["baseline"]
        
        mapping = {
            "system_a": "baseline" if is_a_baseline else "pipeline",
            "system_b": "pipeline" if is_a_baseline else "baseline"
        }
        
        # Merge retrieved evidence (in case it differs)
        evidence_dict = {
            e.snippet_id: e for e in system_a.retrieved_evidence
        }
        for e in system_b.retrieved_evidence:
            if e.snippet_id not in evidence_dict:
                evidence_dict[e.snippet_id] = e
                
        merged_evidence = list(evidence_dict.values())
        
        task_id = f"task_{t['prompt_id']}"
        
        annotation_task = HumanAnnotationTask(
            task_id=task_id,
            prompt=t["baseline"].prompt,
            retrieved_evidence=merged_evidence,
            system_a_focus_points=system_a.focus_points,
            system_b_focus_points=system_b.focus_points,
            ground_truth_mapping=mapping,
        )
        
        out_path = out_dir / f"{task_id}.json"
        out_path.write_text(annotation_task.model_dump_json(indent=2), encoding="utf-8")
        tasks_created.append(annotation_task)
        
    con.cached("DONE", f"Wrote {len(tasks_created)} annotation tasks to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sample model outputs for human annotation")
    parser.add_argument("--num", type=int, default=30, help="Number of samples to generate")
    args = parser.parse_args()
    sample_for_annotation(num_samples=args.num)
