"""Configuration."""
import os
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name}.")
    return value


class Config:
    """Runtime settings. Service endpoints come only from the environment."""

    def __init__(self):
        self.full_url = _required("BRAINTWIN_LLM_URL")
        self.base_url = self.full_url
        self.api_key = os.environ.get("BRAINTWIN_LLM_API_KEY", "EMPTY")
        self.model_id = _required("BRAINTWIN_MODEL_ID")
        self.checkpoint_dir = str(CHECKPOINTS / "qwen3_8b_braintwin")

        self.neo4j_url = _required("BRAINTWIN_NEO4J_URL")
        self.neo4j_username = os.environ.get("BRAINTWIN_NEO4J_USER", "neo4j")
        self.neo4j_password = os.environ.get("BRAINTWIN_NEO4J_PASSWORD", "")

        # Vector store paths are relative to this repository.
        self.model_path = os.environ.get(
            "BRAINTWIN_EMBEDDING",
            str(DATA / "embedding" / "multilingual-e5-large-instruct"),
        )
        self.vector_store_path = os.environ.get(
            "BRAINTWIN_LITERATURE_INDEX",
            str(DATA / "literature_index"),
        )
        
        self.graph_rag_top_k = 100
        self.graph_rag_verbose = False
        self.thread_id = "bio-imaging-qa-001"
        self.project_root = str(ROOT)

    def get_llm_config(self) -> Dict[str, Any]:
        """LLM client settings."""
        return {
            "full_url": self.full_url,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model_id": self.model_id
        }
    
    def get_neo4j_config(self) -> Dict[str, str]:
        """Neo4j settings."""
        return {
            "url": self.neo4j_url,
            "username": self.neo4j_username,
            "password": self.neo4j_password
        }
    
    def get_graph_rag_config(self) -> Dict[str, Any]:
        """Graph RAG settings."""
        return {
            "top_k": self.graph_rag_top_k,
            "verbose": self.graph_rag_verbose
        }
