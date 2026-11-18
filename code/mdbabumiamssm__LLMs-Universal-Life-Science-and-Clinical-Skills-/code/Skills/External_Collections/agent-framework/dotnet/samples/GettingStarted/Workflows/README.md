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

# Workflow Getting Started Samples

The getting started with workflow samples demonstrate the fundamental concepts and functionalities of workflows in Agent Framework.

## Samples Overview

### Foundational Concepts - Start Here

Please begin with the [Foundational](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/_Foundational) samples in order. These three samples introduce the core concepts of executors, edges, agents in workflows, streaming, and workflow construction.

> The folder name starts with an underscore (`_Foundational`) to ensure it appears first in the explorer view.

| Sample | Concepts |
|--------|----------|
| [Executors and Edges](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/_Foundational/01_ExecutorsAndEdges) | Minimal workflow with basic executors and edges |
| [Streaming](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/_Foundational/02_Streaming) | Extends workflows with event streaming |
| [Agents](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/_Foundational/03_AgentsInWorkflows) | Use agents in workflows |
| [Agentic Workflow Patterns](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/_Foundational/04_AgentWorkflowPatterns) | Demonstrates common agentic workflow patterns |
| [Multi-Service Workflows](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/_Foundational/05_MultiModelService) | Shows using multiple AI services in the same workflow |
| [Sub-Workflows](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/_Foundational/06_SubWorkflows) | Demonstrates composing workflows hierarchically by embedding workflows as executors |
| [Mixed Workflow with Agents and Executors](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/_Foundational/07_MixedWorkflowAgentsAndExecutors) | Shows how to mix agents and executors with adapter pattern for type conversion and protocol handling |
| [Writer-Critic Workflow](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/_Foundational/08_WriterCriticWorkflow) | Demonstrates iterative refinement with quality gates, max iteration safety, multiple message handlers, and conditional routing for feedback loops |

Once completed, please proceed to other samples listed below.

> Note that you don't need to follow a strict order after the foundational samples. However, some samples build upon concepts from previous ones, so it's beneficial to be aware of the dependencies.

### Agents

| Sample | Concepts |
|--------|----------|
| [Foundry Agents in Workflows](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/Agents/FoundryAgent) | Demonstrates using Azure Foundry Agents within a workflow |
| [Custom Agent Executors](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/Agents/CustomAgentExecutors) | Shows how to create a custom agent executor for more complex scenarios |
| [Workflow as an Agent](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/Agents/WorkflowAsAnAgent) | Illustrates how to encapsulate a workflow as an agent |

### Concurrent Execution

| Sample | Concepts |
|--------|----------|
| [Fan-Out and Fan-In](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/Concurrent) | Introduces parallel processing with fan-out and fan-in patterns |

### Loop

| Sample | Concepts |
|--------|----------|
| [Looping](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/Loop) | Shows how to create a loop within a workflow |

### Workflow Shared States

| Sample | Concepts |
|--------|----------|
| [Shared States](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/SharedStates) | Demonstrates shared states between executors for data sharing and coordination |

### Conditional Edges

| Sample | Concepts |
|--------|----------|
| [Edge Conditions](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/ConditionalEdges/01_EdgeCondition) | Introduces conditional edges for dynamic routing based on executor outputs |
| [Switch-Case Routing](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/ConditionalEdges/02_SwitchCase) | Extends conditional edges with switch-case routing for multiple paths |
| [Multi-Selection Routing](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/ConditionalEdges/03_MultiSelection) | Demonstrates multi-selection routing where one executor can trigger multiple downstream executors |

> These 3 samples build upon each other. It's recommended to explore them in sequence to fully grasp the concepts.

### Declarative Workflows

| Sample | Concepts |
|--------|----------|
| [Declarative](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/Declarative) | Demonstrates execution of declartive workflows. |

### Checkpointing

| Sample | Concepts |
|--------|----------|
| [Checkpoint and Resume](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/Checkpoint/CheckpointAndResume) | Introduces checkpoints for saving and restoring workflow state for time travel purposes |
| [Checkpoint and Rehydrate](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/Checkpoint/CheckpointAndRehydrate) | Demonstrates hydrating a new workflow instance from a saved checkpoint |
| [Checkpoint with Human-in-the-Loop](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/Checkpoint/CheckpointWithHumanInTheLoop) | Combines checkpointing with human-in-the-loop interactions |

### Human-in-the-Loop

| Sample | Concepts |
|--------|----------|
| [Basic Human-in-the-Loop](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows/HumanInTheLoop/HumanInTheLoopBasic) | Introduces human-in-the-loop interaction using input ports and external requests |


<!-- AUTHOR_SIGNATURE: 9a7f3c2e-MD-BABU-MIA-2026-MSSM-SECURE -->