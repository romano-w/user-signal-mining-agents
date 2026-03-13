# Human Annotation Reports

Store exported reviewer JSON files in `exports/` and generated analysis outputs in `analysis/`.

Recommended workflow:
- Generate blinded tasks with `uv run usm sample-annotations --num 20 --seed 17`.
- Have each reviewer annotate the same task set with `uv run usm annotate-human --annotator-id <id>`.
- Save the exported reviewer JSON files as `exports/<annotator_id>.json`.
- Run `uv run usm analyze-human-annotations --export-a exports/<annotator_a>.json --export-b exports/<annotator_b>.json --output-dir reports/human_annotation/analysis`.

Notes:
- Autosaves under `artifacts/runs/_human_annotations/_results/` are gitignored scratch files.
- The tracked artifacts to commit are the reviewer export JSON files and the generated analysis reports.
