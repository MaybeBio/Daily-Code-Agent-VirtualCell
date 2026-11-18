"""Route a question and collect the answer."""
import json
from typing import Annotated, Optional, TypedDict, List, Literal
from operator import add
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class GraphState(TypedDict, total=False):
    """State passed between workflow nodes."""
    messages: Annotated[list[BaseMessage], add]
    route: str
    router_answer: Optional[str]
    kg_output: Optional[str]
    kg_details: Optional[dict]
    rag_output: Optional[str]
    rag_details: Optional[dict]
    final_answer: Optional[str]


class MultiAgentWorkflow:
    """Chooses literature retrieval, graph retrieval, or a direct reply."""
    
    def __init__(self, config, llm_manager, rag_manager):
        self.config = config
        self.llm_manager = llm_manager
        self.rag_manager = rag_manager
        self.workflow = None
        self.app = None
        self.cfg = None
        self._initialize_workflow()
    
    def _initialize_workflow(self):
        """Compile the workflow."""
        try:
            self.router_prompt = self._get_router_prompt()
            
            self.workflow = self._build_workflow()
            
            checkpointer = MemorySaver()
            self.app = self.workflow.compile(checkpointer=checkpointer)
            self.cfg = {"configurable": {"thread_id": self.config.thread_id}}
            
            print("Multi-agent workflow initialized successfully")
            
        except Exception as e:
            print(f"Error initializing workflow: {e}")
            raise
    
    def _get_router_prompt(self) -> str:
        """Router instructions."""
        return r"""You are a router that must choose exactly one route.
Allowed routes (lowercase): "kg", "rag", "direct".

Routing rules (priority order):
- If the question is about single-cell, cell-type, brain-region distribution of cell-types, genomics, transcriptomics, genes, SNPs, GO, cell types, etc. → answer: kg
- If the question is about medical imaging, clinical diagnosis or  cognitive/behavioral neuroscience experiments (e.g., fMRI, EEG, decoding accuracy, brain regions, memory tasks, cognitive performance) . → answer: rag
- If the question is NOT about science, like chit-chat → route: direct  And ALSO provide a concise reply (≤25 words).

Output requirements:
- Return ONLY a compact JSON object on a single line, no markdown, no extra text.
- Keys must be exactly: {"route":"<kg|rag|direct>", "answer":"<text or empty>"}
- If route != "direct", then "answer" MUST be "" (empty string).
- Use only lowercase for the route.

Examples:
Q: "Define transcriptomics."
{"route":"kg","answer":""}

Q: "How is Alzheimer's diagnosed on MRI?"
{"route":"rag","answer":""}

Q: "hi"
{"route":"direct","answer":"Hello!"}

Q: "what is an apple?"
{"route":"direct","answer":"An apple is an edible fruit from Malus domestica, commonly eaten fresh."}

Now route the following message strictly following the rules above.
"""
    
    def _build_workflow(self) -> StateGraph:
        """Wire the workflow graph."""
        workflow = StateGraph(GraphState)

        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("kg", self._kg_node)
        workflow.add_node("rag", self._rag_node)
        workflow.add_node("final", self._final_node)
        
        workflow.add_edge(START, "supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            lambda s: s["route"],
            {
                "kg": "kg",
                "rag": "rag",
                "end": "final",
                "direct": "final",
            },
        )
        workflow.add_edge("kg", "final")
        workflow.add_edge("rag", "final")
        
        return workflow
    
    def _supervisor_node(self, state: GraphState) -> GraphState:
        """Choose a route for the latest user message."""
        last = self._last_user_message(state["messages"])
        quick_direct = {
            "hi": "Hello!",
            "hello": "Hello!",
            "hey": "Hello!",
            "thanks": "You're welcome.",
            "thank you": "You're welcome.",
        }
        normalized = last.strip().lower()
        if normalized in quick_direct:
            return {"route": "direct", "router_answer": quick_direct[normalized]}

        router_llm = self.llm_manager.get_base_llm()
        
        resp = router_llm.invoke([
            {"role": "system", "content": self.router_prompt},
            {"role": "user", "content": last},
        ])
        
        try:
            data = json.loads(resp.content)
        except Exception:
            data = {"route": "direct", "answer": "Sorry, I didn't catch that."}
        
        route = str(data.get("route", "direct")).strip().lower()
        answer = (data.get("answer") or "").strip()
        
        if route not in {"kg", "rag", "direct"}:
            route, answer = "direct", (answer or "Let me help with that.")
        
        return {"route": route, "router_answer": answer}
    
    def _kg_node(self, state: GraphState) -> GraphState:
        """Query the knowledge graph."""
        last_user = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
        query = last_user.content if last_user else "Describe the KG."
        
        result = self.rag_manager.query_graph_rag(query)
        return {
            "kg_output": result.get("final_answer", "Error: No answer in response"),
            "kg_details": result.get("details", {}),
        }
    
    def _rag_node(self, state: GraphState) -> GraphState:
        """Query the literature index."""
        last_user = next((m for m in reversed(state["messages"]) if m.type == "human"), None)
        query = last_user.content if last_user else "General imaging question."
        
        result = self.rag_manager.query_document_rag(query)
        return {
            "rag_output": result.get("answer", "Error: No answer in response"),
            "rag_details": {
                "query": result.get("query", {}),
                "sources": result.get("sources", []),
            },
        }
    
    def _final_node(self, state: GraphState) -> GraphState:
        """Copy the selected answer into the final state."""
        route = state.get("route")
        if route == "kg":
            ans = (state.get("kg_output") or "").strip()
        elif route == "rag":
            ans = (state.get("rag_output") or "").strip()
        else:  # direct
            ans = (state.get("router_answer") or "").strip()
        
        if not ans:
            ans = "Not enough information was returned to answer that."
        
        return {
            "final_answer": ans,
            "messages": [AIMessage(content=ans)],
        }
    
    def _last_user_message(self, messages):
        """Return the latest user message."""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                return m.content
        return messages[-1].content if messages else ""
    
    def query(self, question: str, stream: bool = False):
        """Run one question through the workflow."""
        if stream:
            return self.app.stream(
                {"messages": [HumanMessage(content=question)]},
                config=self.cfg,
            )
        else:
            result = self.app.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=self.cfg,
            )
            return result.get("final_answer", "No answer generated")

    def query_with_trace(self, question: str) -> dict:
        """Return the answer and the knowledge-graph trace without changing query()."""
        result = self.app.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=self.cfg,
        )
        return {
            "answer": result.get("final_answer", "No answer generated"),
            "kg_details": result.get("kg_details") or {},
        }
    
    def get_app(self):
        """Return the compiled app."""
        return self.app
    
    def get_config(self):
        """Return the thread config."""
        return self.cfg
