# Thalamus-Decision

Thalamus-Decision is a typed decision model for biology and medicine. It reads bulk and single-cell expression
profiles alongside text and JSON records, and answers closed questions with calibrated probabilities:

- **choice**: one of 2–255 named options;
- **yes/no**;
- **score**: a level of an ordered scale.

With a target error of 5% or 1%, each answer also says whether it can be automated or should go to a person. The
current release is Thalamus-Decision-9B.

[Model on Hugging Face](https://huggingface.co/ingenix/Thalamus-Decision-9B) · [Thalamus-Reasoning](https://github.com/IngenixAI/Thalamus-Reasoning) · Paper: *Thalamus-Decision: A Typed Decision Model with a Biological Modality* (preprint in preparation)

This repository holds the inference engine: an HTTP API and MCP server, a Python API, a small client and the
evaluation scripts. The training code and training data are not released.

## Thalamus-Decision and Thalamus-Reasoning

Thalamus-Decision is built on [Thalamus-Reasoning](https://github.com/IngenixAI/Thalamus-Reasoning): Thalamus-Decision-9B
is fine-tuned from [Thalamus-Reasoning-9B](https://huggingface.co/ingenix/Thalamus-Reasoning-9B) and reads expression
profiles in exactly the same way. The two models answer different kinds of requests:

| | Thalamus-Reasoning | Thalamus-Decision |
|---|---|---|
| Answers | free text: explanations, comparisons, conversations | one of the options you give, with a calibrated probability for each |
| Output per question | generated text | probabilities, the answer, its confidence, automate or defer at a target error |
| Cost | generation, token by token | one forward pass per option order, no generation |
| Typical use | exploring samples, asking open questions | pipelines and agents that need decisions they can act on or route to a person |
| Profiles | 20,029-value vectors from `tr-embed`; DepMap cell lines by name; `@sample_id` | the same vectors, files and references |

The engine installs the `thalamus-reasoning` package and uses its code for every step that touches a profile:
- `tr-embed` builds profile vectors from your own data;
- `thalamus_reasoning.load_profiles` reads `.npz` files;
- the DepMap resolver finds cell lines by name;
- `tr-chat`'s message preparation places profiles in text.

The model repository carries the same `preprocessing/`, `depmap/` and `eval/` files as Thalamus-Reasoning-9B. So a
profile you prepared for Thalamus-Reasoning means the same to Thalamus-Decision.

## Two ways to give the model a profile

**1. Name a cancer cell line.** The model repository includes expression profiles of 1,673 cell lines from DepMap.
Write a cell line in the question's state, and its profile is attached at its first mention:
- common spellings work: `MCF7`, `mcf-7`, `HCT 116`, `ACH-000019`, `CVCL_0031`;
- names that are also ordinary words or gene symbols (`RD`, `KELLY`) count only when written as `@RD` or "RD cells";
- lines that are not in DepMap (for example HEK293T) stay plain text.

These are the rules of Thalamus-Reasoning's chat.

**2. Use your own samples.** Convert your expression table (GEO counts, TPM, `.h5ad`) into profile vectors with
`tr-embed`, load them into the server or the Python API, and refer to the samples by ID (`@GSM1001`).

## Install

You need Linux with an NVIDIA GPU of at least 24 GB, and Python 3.11–3.13.

```bash
git clone https://github.com/IngenixAI/Thalamus-Decision.git && cd Thalamus-Decision
uv sync --extra fast --extra mcp --extra eval    # engine, packed readout kernels, MCP, evaluation
```

With pip: `pip install -e ".[fast,mcp,eval]"`.

The `fast` extra installs FlashAttention 2, flash-linear-attention and causal-conv1d. They give the packed readout
the model was calibrated with, and compile against your PyTorch. Without them every prompt runs on its own: the same
answers up to bf16 rounding, several times slower.

The lock pins PyTorch 2.13, the version FlashAttention 2.8 builds against. Where no prebuilt FlashAttention wheel
matches your CUDA and Python, it compiles from source: that needs the CUDA toolkit (`nvcc`) and a lot of memory, so limit
the parallel compile jobs, for example `MAX_JOBS=4 uv sync --extra fast --extra mcp --extra eval`.

Model files (about 19 GB) are downloaded from the Hub on first use. To fetch them ahead of time:

```bash
hf download ingenix/Thalamus-Decision-9B --local-dir Thalamus-Decision-9B
export THALAMUS_DECISION_HOME=$PWD/Thalamus-Decision-9B
```

## Prepare your own profiles

`tr-embed` comes with the `thalamus-reasoning` dependency and works exactly as in Thalamus-Reasoning:

```bash
uv run tr-embed GSE123_counts.tsv.gz --out mine.npz     # bulk: TPM, FPKM, read counts or CPM
uv run tr-embed cells.h5ad --out cells.npz               # single cells: raw counts
uv run tr-embed cells.h5ad --pseudobulk-by donor --out donors.npz
```

These files stand for your own data and are not in this repository.

What `tr-embed` does:
- matches your genes to the 19,260-gene panel;
- detects bulk vs single-cell data, and TPM vs read counts;
- normalizes as in training;
- writes the vectors (`mine.npz`) and a report (`mine.report.json`) with gene coverage and warnings.

Read the report before trusting answers. Also read Thalamus-Reasoning's
[data preparation guide](https://github.com/IngenixAI/Thalamus-Reasoning/blob/main/skills/thalamus-reasoning-prepare-profiles/SKILL.md);
preparing data correctly matters more than anything else.

## Use from Python

```python
from thalamus_decision import ThalamusDecision

td = ThalamusDecision.load()       # $THALAMUS_DECISION_HOME or ingenix/Thalamus-Decision-9B

# DepMap cell lines by name: their profiles are attached as in Thalamus-Reasoning's chat
td.ask("Which line has the more mesenchymal expression program?", ["Line A", "Line B"],
       state="Line A: MCF7\nLine B: MDA-MB-231")
# {"type": "choice", "choice": "Line B", "confidence": 0.999, "probabilities": {...}, ...}

# Your own samples, from tr-embed (mine.npz is your own file)
td.load_profiles("mine.npz")
td.ask("Which sample shows the stronger interferon response?", ["Sample A", "Sample B", "neither"],
       state="Sample A: @GSM1001\nSample B: @GSM1004", max_error=0.05)

# Several typed questions about one state
td.decide(state="Cell line: MCF7", questions={
    "tissue": {"type": "choice", "instructions": "Which tissue does this cell line come from?",
               "criteria": ["breast", "lung", "colon", "skin"], "max_error": 0.05},
    "er":     {"type": "noul", "instructions": "Is it oestrogen-receptor positive?"},
    "grade":  {"type": "score", "instructions": "How proliferative is it?",
               "criteria": ["low", "intermediate", "high"]}})
```

Explicit vectors work as in `ThalamusReasoning.ask`: put one `<omics>` per vector in the state and pass the vectors in
order (`state="Expression profile: <omics>", vectors=[v]`). `td.decide_batch([...])` answers many requests in shared
forwards.

## Serve: HTTP API and MCP

```bash
uv run td-serve --model ingenix/Thalamus-Decision-9B --profiles mine.npz --host 0.0.0.0 --port 8000
# ... READY http://myhost:8000 (Thalamus-Decision-9B, calibration CALIBRATION.json, 12 samples)
```

- **Start-up.** Loading takes under a minute from a local copy. The warm-up then compiles the kernels for 12 packed
  shapes, about 3 minutes on an H100. `GET /health` answers `loading` (HTTP 503) until the `READY` line.
- **Security.** The server has no authentication: run it on a trusted network or behind an authenticating proxy.
- **Slurm.** `scripts/serve.sbatch` runs the server on one GPU and writes its URL to a file.

| option | default | what |
|---|---|---|
| `--model` | `$THALAMUS_DECISION_HOME`, else `ingenix/Thalamus-Decision-9B` | local folder or Hub repo id |
| `--profiles FILE.npz ...` | none | `tr-embed` outputs; their samples become `@sample_id` |
| `--profile-store DIR` | none | keeps uploaded samples across restarts |
| `--embed on\|off` | off | run `tr-embed` on uploaded tables on the server, as Thalamus-Reasoning's web app does |
| `--depmap on\|off` | on | resolve DepMap cell-line names |
| `--mcp on\|off` | on | MCP at `/mcp` |
| `--host`, `--port` | 127.0.0.1, 8000 | `--host 0.0.0.0` accepts other machines |
| `--log-decisions FILE` | none | one JSON line per request, without vectors |

### `POST /v1/decide`

A request is a `state` (text, or a JSON object) and named `questions`:

```bash
curl -s http://127.0.0.1:8000/v1/decide -H 'Content-Type: application/json' -d '{
  "state": "Sample A: @GSM1001\nSample B: @GSM1004",
  "questions": {"ifn": {"type": "choice", "instructions": "Which sample shows the stronger interferon response?",
                        "criteria": ["Sample A", "Sample B", "neither"], "max_error": 0.05}}}'
```

| question field | values |
|---|---|
| `type` | `choice`, `noul` (yes/no) or `score` (ordered levels, lowest first) |
| `instructions` | the question text |
| `criteria` | `choice`, `score`: a list of options, or `{key: description}`. `noul`: optional, `[what yes means, what no means]` |
| `max_error` | 0.05 or 0.01: adds a `decision` (automate or defer) |
| `category` | the threshold category for `max_error` (listed by `GET /v1/models`); default: by type |
| `rank` | k: also return the k most probable options |
| `raw` | true: also return the uncalibrated probabilities of both option orders |

Profiles go into a request in one of three ways:

| way | request | what the model sees |
|---|---|---|
| references (default) | `@sample_id` and cell-line names in `state` | the first mention of each gets its profile right after it (`MCF7<omics>`), as in Thalamus-Reasoning's chat |
| explicit vectors | `"vectors": [v1, v2]` (20,029 floats each, or base64 of little-endian float32) and one `<omics>` per vector in `state` | the state as written, as `ThalamusReasoning.ask(text, vectors)` |
| profile list | `"profiles": ["@GSM1001", "MCF7"]` | one line `Expression profile: <omics>` per profile before the state |

The response has, per question:
- `probabilities` (calibrated), the answer and its `confidence`;
- with `max_error`, a `decision`: `automated` true or false, the certified threshold, and the error measured on test
  data;
- `profiles.notes`: which samples and cell lines were attached.

`noul` is the probability of yes; `score` is the expected level, and `level` the most probable one.

**Other endpoints:**

| endpoint | what |
|---|---|
| `POST /v1/profiles` | add samples: `{"npz": base64 of a tr-embed .npz}`, `{"vectors": {id: [...]}}`, or (with `--embed on`) `{"table": base64, "filename": ...}` |
| `GET /v1/profiles`, `GET /v1/profiles/{id}` | the loaded samples |
| `GET /v1/lines?q=breast` | DepMap cell lines with a profile |
| `POST /v1/resolve` | which samples and cell lines a text names, without a decision |
| `GET /v1/models`, `GET /health` | model, calibration and categories; status |

`src/thalamus_decision/client.py` is a client that needs only the Python standard library. `td-client --url URL demo`
runs example requests.

### MCP (Claude Code and other agents)

```bash
# HTTP. Tools: decide, decide_many, resolve, list_profiles, find_cell_lines
claude mcp add --transport http thalamus-decision http://myhost:8000/mcp

# stdio bridge: also upload_profiles, which sends a tr-embed .npz from your machine
claude mcp add thalamus-decision-local -- uv run --directory /path/to/thamalus-decision td-mcp --url http://myhost:8000
```

## How a profile becomes a vector

Each profile is a 20,029-value vector, built by `tr-embed` as in Thalamus-Reasoning. A linear layer turns it into
one token of the prompt.

| Block | Size | Computation |
|---|---|---|
| Data-type flag | 1 | bulk = 0, single-cell (cells or pseudobulk) = 1; z-scored |
| Expression | 19,260 | panel genes → scale to 10⁶ total → log1p → minus per-gene training mean → ÷ 1.535 |
| Geneformer-V2-104M | 768 | `<cls>` embedding (second-to-last layer) of the rank-value-encoded profile; z-scored |

## Confidence and decisions

- **Two option orders.** Every question is read with the options in the given and in the reversed order, and the two
  are averaged.
- **Calibration.** The probabilities are temperature-scaled with the model's `CALIBRATION.json`: one temperature per
  question type, and for `choice` per number of options.
- **Thresholds.** For `max_error` 0.05 or 0.01, the confidence threshold is certified on held-out calibration data,
  per category (`omics choice`, `omics yes/no`, `biology text (pathways, ontologies)`, ...) or per type.
- **Defer.** `automated: false` means: send the case to a person.
- **The targets are not guarantees.** On shifted test data the realised error was 6.4% at the 5% target and 1.9% at
  the 1% target. Each answer's `decision` reports its category's test error and whether it held (`holds_on_test`).
- **Without profiles,** set `category` to the closest one (`GET /v1/models` lists them).

## Evaluation

**The evaluation subsets of the model repository.** These are the same files as in Thalamus-Reasoning-9B, asked as
typed decisions with the released vectors (`uv run python eval/benchmark.py --benchmark all`, about 6 minutes on one
H100):

| Task | Metric | Thalamus-Decision-9B | Thalamus-Reasoning-9B |
|---|---|---|---|
| Tabula Sapiens, cell type (top 20) | macro AUROC | 0.975 | 0.920 |
| Tabula Sapiens, tissue | macro AUROC | 0.915 | 0.874 |
| Human Diseases, disease subtype (229 labels) | macro AUROC | 0.965 | 0.943 |
| Human Diseases, tissue | macro AUROC | 0.994 | 0.987 |
| GEO-OmicsQA, yes/no (2 × 500 questions) | accuracy | 0.777 | 0.765 |

- **How each model answers.** Thalamus-Decision answers one `choice` question over all candidate labels, or one
  `noul` question. Thalamus-Reasoning's scores come from its card: it scores each label by log-likelihood and generates
  its yes/no answers.
- **No overlap with training.** None of these profiles or questions is in Thalamus-Decision's training data.

**Held-out decision sets** (the paper's test sets; macro accuracy, %):

| Benchmark | Thalamus-Decision-9B | MedDecider-9B |
|---|---|---|
| Biology questions about a measured sample (43 sets) | 83.0 | 40.0 (text only) |
| Ontology and pathway tasks (8 sets) | 94.1 | 77.0 |
| Public text-only biology benchmarks (10 sets) | 69.3 | 70.3 |
| Medicine (24 sets) | 72.2 | 70.8 |
| Follows a planted wrong answer (%, lower is better) | 16.1 | 56.8 |

**General decisions** (public suites written for typed decision APIs, sent in their own request shape; all items,
an unanswered item scores 0):

| Suite | Metric | Thalamus-Decision-9B | MedDecider-9B |
|---|---|---|---|
| OpenDecisions | capability | 69.3 | 65.1 |
| JevBench public v1.2 | capability | 65.4 | 63.4 |
| JevBench (Benchmark Heaven) | accuracy | 82.2 | 78.8 |
| JevBench mini (metamorphic) | coherence | 81.7 | 76.2 |
| Jev Skills | JevExtra | 77.0 | 60.0 |
| Jev Skills | Rules7 | 81.1 | 68.0 |
| DecideBench | mean of 3 modes | 93.6 | 93.9 |
| Decision Index 0.2.1 | index (chance-corrected, 0 = random) | 49.9 | 36.7 |
| jev-rerank-bench | 8 sets, nDCG@10 | 66.4 | 64.1 |
| rag-hallu-bench | macro AUROC ×100 | 89.7 | 90.2 |
| unsafe-eval | mean balanced accuracy | 83.6 | 82.1 |

MedDecider-9B can show at most 10 options; the 17% of Decision Index questions with more score 0 for it. The training data include the training splits of three Decision Index benchmarks (WinoGrande, ESCI, ContractNLI).

## Checks

```bash
TD_TEST_MODEL=/path/to/Thalamus-Decision-9B uv run pytest -q    # CPU: profiles, requests, calibration, HTTP, MCP
```

- **No weights needed.** The tests use only the model folder:
  `hf download ingenix/Thalamus-Decision-9B --exclude "*.safetensors"`.
- **Same profile handling.** They check, among other things, that cell-line names and `@samples` resolve exactly as
  in Thalamus-Reasoning's chat.
- **Same answers as the evaluation pipeline.** On 919 test items (82 profile sets and 10 text-only sets), the engine
  matched the probabilities the model's calibration was fitted on: 100% of top answers, largest difference 2.3e-6.
  It then ran at 27 decisions/s on one H100.
- **`tr-embed` vectors.** With every profile in the form `tr-embed` writes, the 82 held-out profile sets scored as in
  the evaluation pipeline (+0.06 points, 95% CI −0.10 to +0.23).

## Limitations

- For research only, not for clinical decisions.
- Human RNA-seq only. Microarrays, other assays and other species are out of distribution. `tr-embed` checks the
  input and reports what it detected and any problems; read its report before trusting answers.
- The model chooses among the options you give; it does not explain or generate text. Offer "cannot be determined"
  when the state may not contain the answer.
- Its general knowledge is that of a 9B model. Put the facts and rules it needs into the state, and compute dates
  and numbers in code.
- DepMap profiles describe cancer cell lines in culture, not tumours.

## Support

Questions and bug reports: issues in this repository. Contact: [info@ingenix.ai](mailto:info@ingenix.ai).

## License

Code is released under the [Apache License 2.0](https://github.com/IngenixAI/Thalamus-Decision/blob/HEAD/LICENSE) (see [NOTICE](https://github.com/IngenixAI/Thalamus-Decision/blob/HEAD/NOTICE)). The model weights are released
under the [Ingenix.ai License](https://huggingface.co/ingenix/Thalamus-Decision-9B/blob/main/LICENSE) and derive from
[Thalamus-Reasoning-9B](https://huggingface.co/ingenix/Thalamus-Reasoning-9B) and [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
(Apache 2.0). Data files in the model repository keep their own terms; see its README.

## Citation

```bibtex
@misc{kinas2026thalamusdecision,
  title  = {Thalamus-Decision: A Typed Decision Model with a Biological Modality},
  author = {Kinas, Remigiusz and Pikusa, Michał and Powalski, Rafał and Możwiłło, Aleksandra and
            Sypetkowski, Maciej and Smoliński, Łukasz and Hofman, Bartłomiej},
  year   = {2026},
  note   = {Ingenix.ai, preprint in preparation}
}

@article{sypetkowski2026thalamus,
  title   = {Thalamus-Reasoning: A Multimodal Large Language Model for Multi-Sample Omics Reasoning},
  author  = {Sypetkowski, Maciej and Smoliński, Łukasz and Kinas, Remigiusz and Krawczyk, Joanna and
             Hofman, Bartłomiej and Możwiłło, Aleksandra and Pikusa, Michał and Jetka, Tomasz and
             Powalski, Rafał},
  journal = {arXiv preprint arXiv:2605.06728},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.06728}
}
```

Please also cite [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B) and
[Geneformer](https://huggingface.co/ctheodoris/Geneformer). If you use the DepMap profiles or the evaluation data,
cite DepMap, CellWhisperer, Tabula Sapiens and PerturbQA.
