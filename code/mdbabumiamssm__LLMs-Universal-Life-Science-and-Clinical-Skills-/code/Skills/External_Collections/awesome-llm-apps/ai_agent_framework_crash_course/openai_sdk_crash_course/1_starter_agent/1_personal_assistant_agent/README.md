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

# Personal Assistant Agent

A basic personal assistant agent demonstrating the fundamental concepts of agent creation with the OpenAI Agents SDK.

## 🎯 What This Demonstrates

- **Basic Agent Definition**: Creating a simple agent with name and instructions
- **Model Configuration**: Using the default GPT-4o model
- **Simple Instructions**: Basic conversational capabilities

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

2. **Run the agent**:
   ```python
   from agents import Runner
   from agent import root_agent
   
   result = Runner.run_sync(root_agent, "Hello, introduce yourself!")
   print(result.final_output)
   ```

## 💡 Key Concepts

- **Agent Definition**: The `Agent()` class with basic parameters
- **Instructions**: Natural language instructions that guide agent behavior
- **Model Selection**: Default model usage (gpt-4o)

## 🔗 Next Steps

This agent demonstrates the absolute basics. For more advanced features, see:
- [Execution Demo Agent](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/awesome-llm-apps/ai_agent_framework_crash_course/openai_sdk_crash_course/1_starter_agent/execution_demo_agent/README.md) - Different execution methods
- [Tutorial 2: Structured Output](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/awesome-llm-apps/ai_agent_framework_crash_course/openai_sdk_crash_course/2_structured_output_agent/README.md) - Pydantic schema outputs
- [Tutorial 3: Tool Using Agent](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/awesome-llm-apps/ai_agent_framework_crash_course/openai_sdk_crash_course/3_tool_using_agent/README.md) - Adding tools and functions


<!-- AUTHOR_SIGNATURE: 9a7f3c2e-MD-BABU-MIA-2026-MSSM-SECURE -->