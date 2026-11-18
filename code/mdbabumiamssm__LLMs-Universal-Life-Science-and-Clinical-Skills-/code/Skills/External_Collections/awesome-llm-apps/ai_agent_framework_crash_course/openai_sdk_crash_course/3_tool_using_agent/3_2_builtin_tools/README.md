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

# Built-in Tools Agent

Demonstrates using OpenAI Agents SDK built-in tools like WebSearchTool and CodeInterpreterTool.

## 🎯 What This Demonstrates

- **WebSearchTool**: Real-time web search capabilities
- **CodeInterpreterTool**: Code execution and mathematical computation
- **Built-in Tool Integration**: Using pre-configured SDK tools
- **Tool Combination**: Leveraging multiple tools in one agent

## 🚀 Quick Start

1. **Install OpenAI Agents SDK**:
   ```bash
   pip install openai-agents
   ```

2. **Set up environment**:
   ```bash
   cp ../env.example .env
   # Edit .env and add your OpenAI API key
   ```

3. **Run the agent**:
   ```python
   from agents import Runner
   from agent import root_agent
   
   result = Runner.run_sync(root_agent, "What's the latest news about AI and calculate 15% of 200?")
   print(result.final_output)
   ```

## 💡 Key Concepts

- **WebSearchTool()**: Search the web for current information
- **CodeInterpreterTool()**: Execute Python code and calculations
- **Tool Instantiation**: Creating tool instances with default configurations
- **Multi-tool Agents**: Combining different tool types

## 🧪 Available Tools

### WebSearchTool
- Search for current information on the internet
- Useful for factual questions requiring recent data
- Automatically formats search results for agent use

### CodeInterpreterTool
- Execute Python code in a secure environment
- Perfect for mathematical calculations
- Can handle data analysis and complex computations

## 🔗 Next Steps

- [Function Tools](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/awesome-llm-apps/ai_agent_framework_crash_course/openai_sdk_crash_course/3_tool_using_agent/3_1_function_tools/README.md) - Custom function tools
- [Agents as Tools](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/awesome-llm-apps/ai_agent_framework_crash_course/openai_sdk_crash_course/3_tool_using_agent/3_3_agents_as_tools/README.md) - Advanced orchestration patterns


<!-- AUTHOR_SIGNATURE: 9a7f3c2e-MD-BABU-MIA-2026-MSSM-SECURE -->