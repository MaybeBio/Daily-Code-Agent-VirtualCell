# LLM Integration Layer

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](README.md)
- [biomni/eval/__init__.py](biomni/eval/__init__.py)
- [biomni/eval/biomni_eval1.py](biomni/eval/biomni_eval1.py)
- [biomni/know_how/__init__.py](biomni/know_how/__init__.py)
- [biomni/know_how/loader.py](biomni/know_how/loader.py)
- [biomni/llm.py](biomni/llm.py)
- [biomni/model/retriever.py](biomni/model/retriever.py)
- [biomni/version.py](biomni/version.py)
- [tutorials/biomni_101.ipynb](tutorials/biomni_101.ipynb)

</details>



## Purpose and Scope

This document describes Biomni's unified LLM integration layer, which provides a single abstraction for interacting with multiple language model providers. The system supports automatic provider detection based on model names, seamless configuration management, and consistent interfaces across different LLM services. For overall system architecture, see [System Architecture](#2). For configuration details, see [Configuration Management](#2.5).

## Architecture Overview

The LLM integration layer centers around the `get_llm()` function in [biomni/llm.py:13-223](), which serves as a factory for creating language model instances. This function abstracts away provider-specific implementations while maintaining full access to underlying capabilities.

### LLM Provider Factory Architecture

```mermaid
graph TB
    subgraph "Entry Point"
        get_llm["get_llm()"]
    end
    
    subgraph "Configuration Sources"
        BiomniConfig["BiomniConfig"]
        EnvVars["Environment Variables"]
        FunctionParams["Function Parameters"]
        Defaults["Default Values"]
    end
    
    subgraph "Auto-Detection Logic"
        SourceDetection["Source Detection Logic"]
        ModelNameParsing["Model Name Parsing"]
        EnvSourceCheck["LLM_SOURCE Check"]
    end
    
    subgraph "Provider Implementations"
        OpenAI["ChatOpenAI<br/>(OpenAI)"]
        AzureOpenAI["AzureChatOpenAI<br/>(Azure)"]
        Anthropic["ChatAnthropic<br/>(Anthropic)"]
        Gemini["ChatOpenAI<br/>(Gemini via OpenAI API)"]
        Groq["ChatOpenAI<br/>(Groq via OpenAI API)"]
        Ollama["ChatOllama<br/>(Local/Ollama)"]
        Bedrock["ChatBedrock<br/>(AWS Bedrock)"]
        Custom["ChatOpenAI<br/>(Custom base_url)"]
    end
    
    subgraph "Output"
        BaseChatModel["BaseChatModel<br/>(LangChain Interface)"]
    end
    
    get_llm --> BiomniConfig
    get_llm --> EnvVars
    get_llm --> FunctionParams
    get_llm --> Defaults
    
    get_llm --> SourceDetection
    SourceDetection --> ModelNameParsing
    SourceDetection --> EnvSourceCheck
    
    get_llm --> OpenAI
    get_llm --> AzureOpenAI
    get_llm --> Anthropic
    get_llm --> Gemini
    get_llm --> Groq
    get_llm --> Ollama
    get_llm --> Bedrock
    get_llm --> Custom
    
    OpenAI --> BaseChatModel
    AzureOpenAI --> BaseChatModel
    Anthropic --> BaseChatModel
    Gemini --> BaseChatModel
    Groq --> BaseChatModel
    Ollama --> BaseChatModel
    Bedrock --> BaseChatModel
    Custom --> BaseChatModel
```

Sources: [biomni/llm.py:13-223]()

## Supported Providers

The LLM integration layer supports eight distinct provider types through the `SourceType` literal definition. Each provider has specific instantiation logic and configuration requirements.

| Provider | Source Type | Implementation | Base URL | API Key Source |
|----------|-------------|----------------|----------|----------------|
| OpenAI | `"OpenAI"` | `ChatOpenAI` | Default | `OPENAI_API_KEY` |
| Azure OpenAI | `"AzureOpenAI"` | `AzureChatOpenAI` | `OPENAI_ENDPOINT` | `OPENAI_API_KEY` |
| Anthropic | `"Anthropic"` | `ChatAnthropic` | Default | `ANTHROPIC_API_KEY` |
| Google Gemini | `"Gemini"` | `ChatOpenAI` | `generativelanguage.googleapis.com` | `GEMINI_API_KEY` |
| Groq | `"Groq"` | `ChatOpenAI` | `api.groq.com` | `GROQ_API_KEY` |
| Ollama | `"Ollama"` | `ChatOllama` | Default (localhost) | None |
| AWS Bedrock | `"Bedrock"` | `ChatBedrock` | AWS Region | AWS Credentials |
| Custom | `"Custom"` | `ChatOpenAI` | User-specified | User-specified |

Sources: [biomni/llm.py:9-10](), [biomni/llm.py:98-222]()

## Automatic Source Detection

The system implements sophisticated automatic source detection when no explicit `source` parameter is provided. The detection logic follows a hierarchical approach with multiple fallback mechanisms.

### Source Detection Flow

```mermaid
flowchart TD
    Start["get_llm() called<br/>source=None"]
    
    CheckEnv["Check LLM_SOURCE<br/>environment variable"]
    EnvValid{"env_source in<br/>ALLOWED_SOURCES?"}
    UseEnv["source = env_source"]
    
    CheckModel["Analyze model name"]
    
    ClaudeCheck{"model[:7] ==<br/>'claude-'?"}
    UseAnthropic["source = 'Anthropic'"]
    
    GptOssCheck{"model[:7] ==<br/>'gpt-oss'?"}
    UseOllama1["source = 'Ollama'"]
    
    GptCheck{"model[:4] ==<br/>'gpt-'?"}
    UseOpenAI["source = 'OpenAI'"]
    
    AzureCheck{"model.startswith<br/>('azure-')?"}
    UseAzure["source = 'AzureOpenAI'"]
    
    GeminiCheck{"model[:7] ==<br/>'gemini-'?"}
    UseGemini["source = 'Gemini'"]
    
    GroqCheck{"'groq' in<br/>model.lower()?"}
    UseGroq["source = 'Groq'"]
    
    BaseUrlCheck{"base_url is<br/>not None?"}
    UseCustom["source = 'Custom'"]
    
    OllamaPatterns{"model contains<br/>'/' or ollama patterns?"}
    UseOllama2["source = 'Ollama'"]
    
    BedrockPatterns{"model matches<br/>bedrock patterns?"}
    UseBedrock["source = 'Bedrock'"]
    
    Error["raise ValueError:<br/>Unable to determine source"]
    
    Start --> CheckEnv
    CheckEnv --> EnvValid
    EnvValid -->|Yes| UseEnv
    EnvValid -->|No| CheckModel
    
    CheckModel --> ClaudeCheck
    ClaudeCheck -->|Yes| UseAnthropic
    ClaudeCheck -->|No| GptOssCheck
    
    GptOssCheck -->|Yes| UseOllama1
    GptOssCheck -->|No| GptCheck
    
    GptCheck -->|Yes| UseOpenAI
    GptCheck -->|No| AzureCheck
    
    AzureCheck -->|Yes| UseAzure
    AzureCheck -->|No| GeminiCheck
    
    GeminiCheck -->|Yes| UseGemini
    GeminiCheck -->|No| GroqCheck
    
    GroqCheck -->|Yes| UseGroq
    GroqCheck -->|No| BaseUrlCheck
    
    BaseUrlCheck -->|Yes| UseCustom
    BaseUrlCheck -->|No| OllamaPatterns
    
    OllamaPatterns -->|Yes| UseOllama2
    OllamaPatterns -->|No| BedrockPatterns
    
    BedrockPatterns -->|Yes| UseBedrock
    BedrockPatterns -->|No| Error
```

The detection logic includes specific patterns for identifying Ollama models [biomni/llm.py:75-89]() and Bedrock models [biomni/llm.py:90-93]().

Sources: [biomni/llm.py:56-95]()

## Configuration Integration

The LLM integration layer seamlessly integrates with Biomni's configuration system through the optional `config` parameter. When a `BiomniConfig` object is provided, unspecified parameters inherit values from the configuration.

### Parameter Resolution Hierarchy

```mermaid
graph LR
    subgraph "Parameter Sources (Priority Order)"
        P1["1. Function Parameters<br/>(Explicit)"]
        P2["2. BiomniConfig Object<br/>(config.llm_model, etc.)"]
        P3["3. Default Values<br/>(claude-3-5-sonnet-20241022)"]
    end
    
    subgraph "Configuration Fields"
        Model["model<br/>config.llm_model"]
        Temp["temperature<br/>config.temperature"]
        Source["source<br/>config.source"]
        BaseUrl["base_url<br/>config.base_url"]
        ApiKey["api_key<br/>config.api_key"]
    end
    
    subgraph "Final Values"
        FinalModel["Resolved model"]
        FinalTemp["Resolved temperature"]
        FinalSource["Resolved source"]
        FinalBaseUrl["Resolved base_url"]
        FinalApiKey["Resolved api_key"]
    end
    
    P1 --> Model
    P2 --> Model
    P3 --> Model
    Model --> FinalModel
    
    P1 --> Temp
    P2 --> Temp
    P3 --> Temp
    Temp --> FinalTemp
    
    P1 --> Source
    P2 --> Source
    Source --> FinalSource
    
    P1 --> BaseUrl
    P2 --> BaseUrl
    BaseUrl --> FinalBaseUrl
    
    P1 --> ApiKey
    P2 --> ApiKey
    P3 --> ApiKey
    ApiKey --> FinalApiKey
```

The parameter resolution follows a clear precedence: explicit function parameters override config values, which override system defaults.

Sources: [biomni/llm.py:35-54]()

## Provider-Specific Implementation Details

### OpenAI Integration
The OpenAI provider uses the standard `ChatOpenAI` implementation from LangChain with support for temperature control and stop sequences. Models are passed directly without modification.

Sources: [biomni/llm.py:98-105]()

### Azure OpenAI Integration
Azure OpenAI requires additional configuration including endpoint and deployment mapping. The system strips the `"azure-"` prefix from model names and uses API version `"2024-12-01-preview"`.

Sources: [biomni/llm.py:107-122]()

### Anthropic Integration
Anthropic models use `ChatAnthropic` with a default `max_tokens` setting of 8192 and full support for stop sequences and temperature control.

Sources: [biomni/llm.py:124-136]()

### Gemini Integration
Gemini models are accessed through OpenAI-compatible endpoints using the Google Generative Language API base URL. This approach provides consistent interface while accessing Google's models.

Sources: [biomni/llm.py:138-157]()

### Custom Provider Integration
The Custom provider enables integration with any OpenAI-compatible API endpoint. It requires explicit `base_url` specification and supports custom API keys for proprietary or self-hosted models.

Sources: [biomni/llm.py:200-217]()

## Error Handling and Validation

The system implements comprehensive error handling for missing dependencies and invalid configurations. Each provider checks for required packages and raises informative `ImportError` messages with installation instructions.

### Dependency Management

| Provider | Required Package | Import Error Message |
|----------|------------------|---------------------|
| OpenAI/Azure/Gemini/Groq/Custom | `langchain-openai` | "Install with: pip install langchain-openai" |
| Anthropic | `langchain-anthropic` | "Install with: pip install langchain-anthropic" |
| Ollama | `langchain-ollama` | "Install with: pip install langchain-ollama" |
| Bedrock | `langchain-aws` | "Install with: pip install langchain-aws" |

The system also validates source types against `ALLOWED_SOURCES` and provides clear error messages for unsupported configurations.

Sources: [biomni/llm.py:99-104](), [biomni/llm.py:108-113](), [biomni/llm.py:125-130](), [biomni/llm.py:175-180](), [biomni/llm.py:187-192](), [biomni/llm.py:220-222]()

---