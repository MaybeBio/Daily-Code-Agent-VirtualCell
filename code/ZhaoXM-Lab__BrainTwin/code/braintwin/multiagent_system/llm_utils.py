"""LLM client helpers."""
import sys
import os
from typing import Optional
from langchain_openai import ChatOpenAI

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from LLM_utils import CustomOpenAIProxyChat
except ImportError:
    print("Warning: CustomOpenAIProxyChat was not found. Using the fallback client.")
    CustomOpenAIProxyChat = None

class LLMManager:
    """Holds the chat clients used by the agent."""

    def __init__(self, config):
        self.config = config
        self.llm_base = None
        self.llm_ft = None
        self._initialize_llms()

    def _initialize_llms(self):
        """Create the chat clients."""
        try:
            if CustomOpenAIProxyChat:
                self.llm_base = CustomOpenAIProxyChat(
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    model=self.config.model_id
                )

                self.llm_ft = CustomOpenAIProxyChat(
                    base_url=self.config.full_url,
                    api_key=self.config.api_key,
                    model=self.config.model_id
                )
            else:
                self.llm_base = ChatOpenAI(
                    base_url=self.config.base_url + "/v1",
                    api_key=self.config.api_key,
                    model=self.config.model_id
                )
            
                self.llm_ft = ChatOpenAI(
                    base_url=self.config.full_url + "/v1",
                    api_key=self.config.api_key,
                    model=self.config.model_id
                )
            
            print("LLM instances initialized successfully")
            
        except Exception as e:
            print(f"Error initializing LLMs: {e}")
            raise
    
    def get_base_llm(self) -> Optional[ChatOpenAI]:
        """Return the base chat client."""
        return self.llm_base
    
    def get_ft_llm(self) -> Optional[ChatOpenAI]:
        """Return the fine-tuned chat client."""
        return self.llm_ft
    
    def test_connection(self) -> bool:
        """Send a short probe to the chat client."""
        try:
            if self.llm_base:
                response = self.llm_base.invoke("Hi")
                print(f"Base LLM response: {response.content}")
            
            if self.llm_ft:
                response = self.llm_ft.invoke("Hi")
                print(f"FT LLM response: {response.content}")
            
            return True
        except Exception as e:
            print(f"LLM connection test failed: {e}")
            return False
