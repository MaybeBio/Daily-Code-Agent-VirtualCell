"""Literature and knowledge-graph retrieval."""
from typing import Optional, Dict, Any
from langchain_community.graphs import Neo4jGraph


def _extract_graph_details(response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract readable KG query details from GraphCypherQAChain output."""
    details: Dict[str, Any] = {}
    steps = response.get("intermediate_steps") or []
    details["intermediate_steps"] = steps

    for step in steps:
        if not isinstance(step, dict):
            continue
        query = step.get("query")
        if query:
            details["cypher"] = query
            break

    return details

class RAGManager:
    """Loads the literature index and the knowledge graph on first use."""
    
    def __init__(self, config, llm_manager):
        self.config = config
        self.llm_manager = llm_manager
        self.document_rag = None
        self.graph_rag = None
        self.vector_store = None
        self.graph = None
        self._document_init_attempted = False
        self._graph_init_attempted = False
        print("RAG systems will be initialized lazily")
    
    def _initialize_rag_systems(self):
        """Load both retrieval backends."""
        try:
            self._initialize_document_rag()
            self._initialize_graph_rag()
            
            print("RAG systems initialized successfully")
            
        except Exception as e:
            print(f"Error initializing RAG systems: {e}")
            raise
    
    def _initialize_document_rag(self):
        """Load the literature index."""
        if self.document_rag is not None:
            return
        self._document_init_attempted = True
        try:
            from Document_RAG import load_vector_store, build_rag_application
            
            self.vector_store = load_vector_store(model_path=self.config.model_path, store_path=self.config.vector_store_path)
            if self.vector_store is None:
                print("Warning: Could not load vector store")
                return
            
            self.document_rag = build_rag_application(
                self.llm_manager.get_ft_llm(), 
                self.vector_store
            )
            
            print(f"Document RAG initialized with {self.vector_store.index.ntotal} documents")
            
        except Exception as e:
            print(f"Error initializing Document RAG: {e}")
            self.document_rag = None
    
    def _initialize_graph_rag(self):
        """Connect to Neo4j and build the graph query app."""
        if self.graph_rag is not None:
            return
        self._graph_init_attempted = True
        try:
            from Graph_RAG import build_grag_application
            
            neo4j_config = self.config.get_neo4j_config()
            self.graph = Neo4jGraph(
                url=neo4j_config["url"],
                username=neo4j_config["username"],
                password=neo4j_config["password"]
            )
            
            graph_rag_config = self.config.get_graph_rag_config()
            self.graph_rag = build_grag_application(
                graph=self.graph,
                llm=self.llm_manager.get_ft_llm(),
                top_k=graph_rag_config["top_k"],
                verbose=graph_rag_config["verbose"]
            )
            
            print("Graph RAG initialized successfully")
            
        except Exception as e:
            print(f"Error initializing Graph RAG: {e}")
            self.graph_rag = None
    
    def query_document_rag(self, question: str) -> Dict[str, Any]:
        """Query the literature index."""
        if not self.document_rag and not self._document_init_attempted:
            self._initialize_document_rag()
        if not self.document_rag:
            return {"answer": "Document RAG not available", "sources": []}
        
        try:
            response = self.document_rag.invoke({"question": question})
            return {
                "answer": response.get("answer", "No answer found"),
                "query": response.get("query", {}),
                "sources": response.get("sources", [])
            }
        except Exception as e:
            return {"answer": f"Error querying Document RAG: {e}", "sources": []}
    
    def query_graph_rag(self, question: str) -> Dict[str, Any]:
        """Query the knowledge graph."""
        if not self.graph_rag and not self._graph_init_attempted:
            self._initialize_graph_rag()
        if not self.graph_rag:
            return {"final_answer": "Graph RAG not available"}
        
        try:
            response = self.graph_rag.invoke({"question": question})
            return {
                "final_answer": response.get("final_answer", "No answer found"),
                "qa_raw": response.get("qa_raw", {}),
                "qa_result_text": response.get("qa_result_text", ""),
                "details": _extract_graph_details(response.get("qa_raw", {}))
            }
        except Exception as e:
            return {"final_answer": f"Error querying Graph RAG: {e}"}
    
    def get_document_rag(self):
        """Return the literature app, loading it if needed."""
        if not self.document_rag and not self._document_init_attempted:
            self._initialize_document_rag()
        return self.document_rag
    
    def get_graph_rag(self):
        """Return the graph app, loading it if needed."""
        if not self.graph_rag and not self._graph_init_attempted:
            self._initialize_graph_rag()
        return self.graph_rag
