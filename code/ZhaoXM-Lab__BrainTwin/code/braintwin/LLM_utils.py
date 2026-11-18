import os
import sys
import json
import numpy as np
from typing import List, Union, Dict, Any, Optional
from pydantic import Field, BaseModel
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult, ChatGeneration, AIMessage
import requests

# ======== Custom LLM ========
class CustomOpenAIProxyChat(BaseChatModel):
    api_key: str = Field(...)
    base_url: str = Field(...)
    model: str = Field(default="gpt-4")
    streaming: bool = Field(default=True)

    @property
    def _llm_type(self) -> str:
        return "custom-openai-proxy"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        converted = []
        for msg in messages:
            role = msg.type.replace("human", "user").replace("ai", "assistant")
            item = {"role": role, "content": msg.content}
            if msg.type == "ai" and hasattr(msg, "tool_calls"):
                item["tool_calls"] = msg.tool_calls
            if msg.type == "tool":
                item["tool_call_id"] = getattr(msg, "tool_call_id", None)
            converted.append(item)
        return converted

    def _generate(self, messages: List[BaseMessage], stop=None, tools=None, tool_choice=None, **kwargs) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": 0.1,
            "stream": self.streaming,
        }
        if not self.streaming:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        for key in ("max_tokens", "frequency_penalty", "presence_penalty", "repetition_penalty"):
            if kwargs.get(key) is not None:
                payload[key] = kwargs[key]
        if tools: 
            payload["tools"] = tools
            payload["tool_choice"] = "auto" if tool_choice is None else tool_choice
        if tool_choice: payload["tool_choice"] = tool_choice

        headers = {"Content-Type": "application/json", "Authorization": self.api_key}
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=self.streaming,
            timeout=180,
        )
        if response.status_code >= 400 or not response.content:
            raise RuntimeError(
                f"LLM HTTP {response.status_code}: {response.text[:400]}"
            )

        if self.streaming:
            full_content = ""
            tool_calls = None
            for line in response.iter_lines():
                if not line: continue
                try:
                    decoded = line.decode("utf-8").strip()
                    if decoded.startswith("data: "):
                        data = decoded[6:]
                        if data == "[DONE]": break
                        delta = json.loads(data)["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content: full_content += content
                        if "tool_calls" in delta:
                            if tool_calls is None:
                                tool_calls = []
                            tool_calls.extend(delta["tool_calls"])
                except: continue
            
            if tool_calls:
                message = AIMessage(content=full_content, tool_calls=tool_calls)
            else:
                message = AIMessage(content=full_content)
        else:
            response_data = response.json()
            message_data = response_data["choices"][0]["message"]
            full_content = message_data.get("content", "")
            tool_calls = message_data.get("tool_calls")
            
            if tool_calls:
                message = AIMessage(content=full_content, tool_calls=tool_calls)
            else:
                message = AIMessage(content=full_content)

        return ChatResult(generations=[ChatGeneration(message=message)])
        