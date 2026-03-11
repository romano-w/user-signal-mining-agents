Current script entrypoints:

- `build_yelp_subset.py`: stream Yelp businesses and reviews, keep restaurants, and write chunked snippet JSONL output
- `build_index.py`: generate Sentence Transformers embeddings and persist a dense local index

Dataset preparation:

- `uv run usm fetch-yelp-dataset`: download the official Yelp JSON zip, extract the tar, and unpack the JSON files
- `uv run usm fetch-yelp-dataset --skip-download`: use an existing local `yelp_dataset.tar` and only extract it

Planned next:

- `run_experiment.py`: run baseline and multi-step pipeline variants over the founder prompt set
