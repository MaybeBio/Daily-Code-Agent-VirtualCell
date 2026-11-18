# BrainTwin core

- `Document_RAG.py` retrieves passages from the local literature index and asks the model to answer from them.
- `Graph_RAG.py` writes a Cypher query for genes, cell types, brain regions, and SNPs.
- `multiagent_system/` routes each question to the literature index, the knowledge graph, or a direct answer.

Importing this package does not open a database connection. The graph connection is created on the first query.

## Before you run it

1. Download the files listed in `DATA.md` from https://doi.org/10.6084/m9.figshare.35014504 and place them where that file says. The full-parameter checkpoint is not on that record yet.
2. Export `BRAINTWIN_LLM_URL`, `BRAINTWIN_MODEL_ID`, `BRAINTWIN_NEO4J_URL`, and `BRAINTWIN_NEO4J_PASSWORD`.
3. Start an OpenAI-compatible server with the weights in `checkpoints/qwen3_8b_braintwin/`. Point `BRAINTWIN_LLM_URL` at that server. The value is an origin, without `/v1`.

No service host is stored in the code.

## Ask one question

From the repository root:

```bash
python braintwin/examples/simple_qa.py
```

Or pass your own question:

```bash
python braintwin/examples/simple_qa.py "List brain subregions that contain microglia"
```

The first default question uses the literature index. A gene, cell-type, or SNP question uses the knowledge graph.

```python
import sys
sys.path.insert(0, "braintwin")
from multiagent_system.main import create_system

system = create_system()
print(system.query("Which genes are associated with SNP rs429358?"))
```

## Environment

| Variable | Role |
|---|---|
| `BRAINTWIN_LLM_URL` | Required. OpenAI-compatible origin. The client appends `/v1`. |
| `BRAINTWIN_MODEL_ID` | Required. Model id reported by that server. |
| `BRAINTWIN_LLM_API_KEY` | Optional. Defaults to `EMPTY`. |
| `BRAINTWIN_NEO4J_URL` | Required. Bolt URL of your Neo4j server. |
| `BRAINTWIN_NEO4J_USER` | Optional. Defaults to `neo4j`. |
| `BRAINTWIN_NEO4J_PASSWORD` | Password for that server. |
| `BRAINTWIN_EMBEDDING` | Optional override for `data/embedding/multilingual-e5-large-instruct`. |
| `BRAINTWIN_LITERATURE_INDEX` | Optional override for `data/literature_index`. |
