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

# Get Started with Microsoft Agent Framework for C# Developers

## Samples

- [Getting Started with Agents](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Agents): basic agent creation and tool usage
- [Agent Provider Samples](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/AgentProviders): samples showing different agent providers
- [Workflow Samples](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows): advanced multi-agent patterns and workflow orchestration

## Quickstart

### Basic Agent - .NET

```c#
using System;
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;

var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")!;
var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME")!;

var agent = new AzureOpenAIClient(new Uri(endpoint), new AzureCliCredential())
    .GetOpenAIResponseClient(deploymentName)
    .CreateAIAgent(name: "HaikuBot", instructions: "You are an upbeat assistant that writes beautifully.");

Console.WriteLine(await agent.RunAsync("Write a haiku about Microsoft Agent Framework."));
```

## Examples & Samples

- [Getting Started with Agents](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Agents): basic agent creation and tool usage
- [Agent Provider Samples](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/AgentProviders): samples showing different agent providers
- [Workflow Samples](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/dotnet/samples/GettingStarted/Workflows): advanced multi-agent patterns and workflow orchestration

## Agent Framework Documentation

- [Documentation](https://learn.microsoft.com/agent-framework/)
- [Agent Framework Repository](https://github.com/microsoft/agent-framework)
- [Design Documents](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/docs/design)
- [Architectural Decision Records](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/agent-framework/docs/decisions)
- [MSFT Learn Docs](https://learn.microsoft.com/agent-framework/overview/agent-framework-overview)


<!-- AUTHOR_SIGNATURE: 9a7f3c2e-MD-BABU-MIA-2026-MSSM-SECURE -->