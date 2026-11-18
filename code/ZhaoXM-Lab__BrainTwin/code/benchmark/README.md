# Benchmark

These scripts call the agent in `braintwin/`. They do not reimplement literature retrieval or the knowledge graph.

- `Multiagent_system_test.py` reads `data/benchmarks/singlecell_DEG.json`, `singlecell_region.json`, `genomics_SCQ.json`, and `image_5K.json`. Answers are written to `benchmark/results/`.
- `evaluation_utils.py` computes accuracy, precision, recall, and F1 for multiple-choice labels.

Question files are listed in `DATA.md`. Download `benchmarks.zip` from https://doi.org/10.6084/m9.figshare.35014504 and unzip it into `data/benchmarks/`. `results/` starts empty.
