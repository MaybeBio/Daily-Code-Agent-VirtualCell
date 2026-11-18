# BrainTwin

Code for the literature index, the knowledge graph, and the agent. Weights and databases are not included. Download them from https://doi.org/10.6084/m9.figshare.35014504 and follow [DATA.md](https://github.com/ZhaoXM-Lab/BrainTwin/blob/HEAD/DATA.md).

That record currently has the benchmark questions, the literature index, the Neo4j dump, and the LoRA adapter. The full-parameter checkpoint is not uploaded yet. It will be added to the same record. The embedding model is a separate download.

## Layout

- `braintwin/`: the agent, literature retrieval, and graph retrieval. See [braintwin/README.md](https://github.com/ZhaoXM-Lab/BrainTwin/blob/HEAD/braintwin/README.md).
- `benchmark/`: question benchmarks. The scripts call the agent and write answers to `benchmark/results/`.
- `cases/`: genomics, single-cell, and MRI cases, in separate folders. The cross-scale report is not included.
- `data/`: reserved location for the literature index, embedding model, graph export, benchmarks, and case inputs.
- `checkpoints/`: reserved location for model weights.

## One question

```bash
pip install -r requirements.txt
export BRAINTWIN_LLM_URL="your-openai-compatible-origin"
export BRAINTWIN_MODEL_ID="the-model-id-reported-by-that-server"
export BRAINTWIN_NEO4J_URL="your-neo4j-bolt-url"
export BRAINTWIN_NEO4J_PASSWORD="your-password"
python braintwin/examples/simple_qa.py
```

`BRAINTWIN_LLM_URL` is the server origin. The client appends `/v1`. Do not put a host name in the code.
