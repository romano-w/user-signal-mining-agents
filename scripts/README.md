Current script entrypoints:

- `build_yelp_subset.py`: stream Yelp businesses and reviews, keep restaurants, and write chunked snippet JSONL output.
- `build_index.py`: generate Sentence Transformers embeddings and persist a dense local index.
- `ci/run_research_gate.py`: execute standardized `branch-ready`, `integration-ready`, and `main-ready` CI gate flows.
- `ci/check_metric_guardrails.py`: fail when judge artifacts show critical metric regressions.

Dataset preparation:

- `uv run usm fetch-yelp-dataset`: download the official Yelp JSON zip, extract the tar, and unpack the JSON files.
- `uv run usm fetch-yelp-dataset --skip-download`: use an existing local `yelp_dataset.tar` and only extract it.

CI gate commands:

- `uv run python scripts/ci/run_research_gate.py --level branch-ready`
- `uv run python scripts/ci/run_research_gate.py --level integration-ready`
- `uv run python scripts/ci/run_research_gate.py --level main-ready`
- `uv run python scripts/ci/check_metric_guardrails.py --runs-dir artifacts/runs`

Planned next:

- `run_experiment.py`: run baseline and multi-step pipeline variants over the founder prompt set.
