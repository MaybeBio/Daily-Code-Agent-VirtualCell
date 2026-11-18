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

# Conversation Management

Demonstrates manual conversation threading with `to_input_list()` and automatic management with Sessions.

## 🎯 What This Demonstrates

- **Manual Threading**: Using `result.to_input_list()` for conversation history
- **Automatic Sessions**: Using `SQLiteSession` for memory management
- **Conversation Context**: Maintaining state across multiple turns
- **Thread Management**: Different approaches to conversation flow

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
   import asyncio
   from agent import manual_conversation_example, session_conversation_example
   
   # Test manual conversation management
   asyncio.run(manual_conversation_example())
   ```

## 💡 Key Concepts

- **to_input_list()**: Manual conversation history management
- **SQLiteSession**: Automatic conversation persistence
- **Context Preservation**: Maintaining conversation state
- **Session Storage**: In-memory vs persistent storage

## 🔗 Next Steps

- [Execution Methods](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/awesome-llm-apps/ai_agent_framework_crash_course/openai_sdk_crash_course/4_running_agents/4_1_execution_methods/README.md) - Basic execution patterns
- [Streaming Events](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/awesome-llm-apps/ai_agent_framework_crash_course/openai_sdk_crash_course/4_running_agents/4_4_streaming_events/README.md) - Real-time processing


<!-- AUTHOR_SIGNATURE: 9a7f3c2e-MD-BABU-MIA-2026-MSSM-SECURE -->