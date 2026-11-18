"""Agent entry point."""
from .config import Config
from .llm_utils import LLMManager
from .rag_manager import RAGManager
from .multiagent_workflow import MultiAgentWorkflow


class MultiAgentSystem:
    """Routes a question to the literature index, the knowledge graph, or a direct answer."""

    def __init__(self, config_path: str = None):
        """Build the agent. config_path is unused and kept for callers."""
        self.config = Config()
        self.llm_manager = LLMManager(self.config)
        self.rag_manager = RAGManager(self.config, self.llm_manager)
        self.workflow = MultiAgentWorkflow(self.config, self.llm_manager, self.rag_manager)
        
        print("Multi-agent system initialized successfully!")
    
    def query(self, question: str, stream: bool = False):
        """Ask one question. stream=True yields text chunks."""
        return self.workflow.query(question, stream=stream)
    
    def test_system(self):
        """Check the chat client, both retrieval backends, and the workflow."""
        print("Testing system components...")

        print("1. Testing LLM connections...")
        llm_ok = self.llm_manager.test_connection()
        
        print("2. Testing RAG systems...")
        doc_rag_ok = self.rag_manager.get_document_rag() is not None
        graph_rag_ok = self.rag_manager.get_graph_rag() is not None
        
        print(f"   Document RAG: {'✓' if doc_rag_ok else '✗'}")
        print(f"   Graph RAG: {'✓' if graph_rag_ok else '✗'}")
        
        print("3. Testing workflow...")
        workflow_ok = self.workflow.app is not None
        print(f"   Workflow: {'✓' if workflow_ok else '✗'}")
        
        overall_status = llm_ok and doc_rag_ok and graph_rag_ok and workflow_ok
        print(f"\nOverall system status: {'✓ Ready' if overall_status else '✗ Issues detected'}")
        
        return overall_status
    
    def get_components(self):
        """Return the config, clients, retrieval manager, and workflow."""
        return {
            'config': self.config,
            'llm_manager': self.llm_manager,
            'rag_manager': self.rag_manager,
            'workflow': self.workflow
        }


def create_system(config_path: str = None) -> MultiAgentSystem:
    """Create an agent. config_path is unused and kept for callers."""
    return MultiAgentSystem(config_path)
