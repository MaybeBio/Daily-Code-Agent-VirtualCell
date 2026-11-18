import os
import sys

from tqdm import tqdm
import numpy as np

import json
from typing import TypedDict, Dict, Any
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import StateGraph, START, END
from LLM_utils import CustomOpenAIProxyChat
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import GraphCypherQAChain


cypher_prompt = PromptTemplate.from_template("""You are an expert Cypher engineer for a Neo4j database. Your job in this step is to produce a syntactically correct Cypher query that answers the question using the schema below.
Do NOT answer in natural language. Do NOT explain. Output must be ONLY a single fenced code block with language `cypher`.
# RULES
- Use ONLY labels / relationship types / properties present in the schema.
- Do NOT use APOC or any functions not standard to Cypher unless present in the schema.
- Prefer exact matches on names; for case-insensitive matching, use toLower() on both sides.
- Return the MINIMAL fields needed to answer the question.
- If multiple steps are needed, compose them in one query with WITH.
- No natural language, no commentary, no markdown outside the code block.

# SCHEMA
Node properties:
    CellType [CellType_Name: STRING]
    Gene [Gene_Name: STRING, Chromosome_id: STRING, start_position: STRING, end_position: STRING]
    BrainRegion [id: STRING, Region_Name: STRING]
    SNP [SNP_Name: STRING, Chromosome_id: STRING, Chromosome_position: STRING]
    BrainSubRegion [Subregion_Name: STRING]
    GOTerm [Go_Term_Name: STRING]
    Trait [Trait_Name: STRING]
Relationship properties:
    CONTAINS [rank: INTEGER]
The relationships:
    (:Gene)-[:ASSOCIATE_WITH_TRAIT]->(:Trait)
    (:Gene)-[:ANNOTATAED_AS]->(:GOTerm)
    (:Gene)-[:IS_MARKER]->(:CellType)
    (:SNP)-[:ASSOCIATE_WITH_TRAIT]->(:Trait)
    (:SNP)-[:ASSOCIATE_WITH_GENE]->(:Gene)
    (:BrainSubRegion)-[:IS_PART_OF]->(:BrainRegion)
    (:BrainSubRegion)-[:CONTAINS]->(:CellType)

# FEW-SHOT EXAMPLES (follow structure exactly)
Q: List cell types with expressed genes PDCD10, CD34, HBB, PTPRC:
```cypher
MATCH (g:Gene)-[:IS_MARKER]->(ct:CellType)
WHERE g.Gene_Name IN ["PDCD10", "CD34", "HBB", "PTPRC"]
RETURN ct.CellType_Name AS celltype
```

Q: List brain subregions containing the cell type 'Microglia':
```cypher
MATCH (sr:BrainSubRegion)-[:CONTAINS]->(c:CellType {{CellType_Name: "Microglia"}})
RETURN sr.Subregion_Name AS subregion
```

Q: Which cell types are contained in subregion "Hippocampus-CA1"?
```cypher
MATCH (sr:BrainSubRegion {{Subregion_Name: "Hippocampus-CA1"}})-[:CONTAINS]->(c:CellType)
RETURN c.CellType_Name AS cell_type
```

Q: Count SNPs associated with gene "APOE".
```cypher
MATCH (g:Gene {{Gene_Name: "APOE"}})<-[:ASSOCIATE_WITH_GENE]-(s:SNP)
RETURN count(s) AS snp_count
```

# YOUR TURN
Question: {query}

""")


EMPTY_GRAPH_ANSWER = "The knowledge graph returned no records for this query."

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Answer the question using only the records in Information.\n"
        "Do not use an example, a stock, a manager, or ownership as the subject of the answer.\n"
        "If Information is empty, reply with exactly this sentence and nothing else:\n"
        f"{EMPTY_GRAPH_ANSWER}\n"
        "If Information is not empty, use only those records. Do not add facts that are absent from Information.\n\n"
        "Information:\n{context}\n\n"
        "Question: {question}\n"
        "Helpful Answer:"
    ),
)


def _graph_context_empty(steps: Any) -> bool:
    saw_context = False
    for step in steps or []:
        if not isinstance(step, dict) or "context" not in step:
            continue
        saw_context = True
        if step.get("context"):
            return False
    return saw_context


class GraphQAState(TypedDict, total=False):
    question: str
    qa_raw: Dict[str, Any]
    qa_result_text: str
    final_answer: str



def build_grag_application(
    llm,
    graph,
    *,
    cypher_prompt=cypher_prompt,
    final_llm=None,
    top_k: int = 100,
    verbose: bool = False,
    allow_dangerous_requests: bool = True,
    validate_cypher: bool = True,
    return_intermediate_steps: bool = True,
    ):
    """
    LangGraph app that chains two stages with a single .invoke({"question": ...}):
    1) GraphCypherQAChain (KG retrieval/answer)
    """
    qa_chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        verbose=verbose,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        return_intermediate_steps=return_intermediate_steps,
        allow_dangerous_requests=allow_dangerous_requests,
        validate_cypher=validate_cypher,
        top_k=top_k,
    )


    final_llm = final_llm or llm


    refine_prompt = ChatPromptTemplate.from_messages([
    (
    "system",
    "You are a meticulous expert. Use the knowledge-graph QA output to answer clearly and concisely."
    ),
    (
    "human",
    (
    "Question:\n{question}\n\n"
    "Knowledge-Graph QA Output:\n{qa_result}\n\n"
    "If information is missing or uncertain, answer the question based on your own knowledge."
    "If the question is a choice question, answer it with a choice in A/B/C/D."
    ),
    ),
    ])

    parser = StrOutputParser()


    # ---------------- Nodes ----------------
    def qa_node(state: GraphQAState) -> GraphQAState:
        q = state["question"]
        out = qa_chain.invoke({"query": q}) # returns dict with "result" and maybe "intermediate_steps"
        # pack readable text for the refiner
        result = out.get("result", "")
        steps = out.get("intermediate_steps")
        if _graph_context_empty(steps) or "neo4j stocks" in str(result).lower():
            result = EMPTY_GRAPH_ANSWER
            out = {**out, "result": result}
        if steps:
            qa_result_text = f"Result: {result}\n\nIntermediateSteps: {steps}"
        else:
            qa_result_text = str(result)
        return {
        **state,
        "qa_raw": out,
        "qa_result_text": qa_result_text,
        "final_answer": parser.parse(str(result)),
        }


    def refine_node(state: GraphQAState) -> GraphQAState:
        messages = refine_prompt.invoke({
        "question": state["question"],
        "qa_result": state.get("qa_result_text", "<empty>")
        })
        text = final_llm.invoke(messages)
        # Some LLMs return BaseMessage/AIMessage; normalize to string
        final_text = getattr(text, "content", text)
        final_text = parser.parse(final_text)
        return {**state, "final_answer": final_text}


    # --------------- Graph wiring ---------------
    graph_builder = StateGraph(GraphQAState)
    graph_builder.add_node("qa", qa_node)


    graph_builder.add_edge(START, "qa")
    graph_builder.add_edge("qa", END)


    app = graph_builder.compile()
    
    return app