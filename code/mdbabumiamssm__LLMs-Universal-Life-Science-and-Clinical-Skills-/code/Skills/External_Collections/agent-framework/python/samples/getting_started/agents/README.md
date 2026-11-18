<!--
# COPYRIGHT NOTICE
# This file is part of the "Universal Biomedical Skills" project.
# Copyright (c) 2026 MD BABU MIA, PhD <md.babu.mia@mssm.edu>
# All Rights Reserved.
#
# This code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
#
# Provenance: Authenticated by MD BABU MIA

-->

# Agent Examples

This folder contains examples demonstrating how to create and use agents with different chat clients from the Agent Framework. Each sub-folder focuses on a specific provider and client type, showing various capabilities like function tools, code interpreter, thread management, structured outputs, image processing, web search, Model Context Protocol (MCP) integration, and more.

## Examples by Provider

### Azure AI Foundry Examples

| Folder | Description |
|--------|-------------|
| **[`azure_ai_agent/`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/python/samples/getting_started/agents/azure_ai_agent)** | Create agents using Azure AI Agent Service (based on `azure-ai-agents` V1 package) including function tools, code interpreter, MCP integration, thread management, and more. |
| **[`azure_ai/`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/python/samples/getting_started/agents/azure_ai)** | Create agents using Azure AI Agent Service (based on `azure-ai-projects` [V2](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-projects/CHANGELOG.md#200b1-2025-11-11) package) including function tools, code interpreter, MCP integration, thread management, and more. |

### Microsoft Copilot Studio Examples

| Folder | Description |
|--------|-------------|
| **[`copilotstudio/`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/python/samples/getting_started/agents/copilotstudio)** | Create agents using Microsoft Copilot Studio with streaming and non-streaming responses, authentication handling, and explicit configuration options |

### Azure OpenAI Examples

| Folder | Description |
|--------|-------------|
| **[`azure_openai/`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/python/samples/getting_started/agents/azure_openai)** | Create agents using Azure OpenAI APIs with multiple client types (Assistants, Chat, and Responses clients) supporting function tools, code interpreter, thread management, and more |

### OpenAI Examples

| Folder | Description |
|--------|-------------|
| **[`openai/`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/python/samples/getting_started/agents/openai)** | Create agents using OpenAI APIs with comprehensive examples including Assistants, Chat, and Responses clients featuring function tools, code interpreter, file search, web search, MCP integration, image analysis/generation, structured outputs, reasoning, and thread management |

### Anthropic Examples

| Folder | Description |
|--------|-------------|
| **[`anthropic/`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/python/samples/getting_started/agents/anthropic)** | Create agents using Anthropic models through OpenAI Chat Client configuration, demonstrating tool calling capabilities |

### Custom Implementation Examples

| Folder | Description |
|--------|-------------|
| **[`custom/`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/python/samples/getting_started/agents/custom)** | Create custom agents and chat clients by extending the base framework classes, showing complete control over agent behavior and backend integration |


<!-- AUTHOR_SIGNATURE: 9a7f3c2e-MD-BABU-MIA-2026-MSSM-SECURE -->