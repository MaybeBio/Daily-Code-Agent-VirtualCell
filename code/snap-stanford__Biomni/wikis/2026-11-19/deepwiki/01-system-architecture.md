# System Architecture

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](README.md)
- [biomni/agent/a1.py](biomni/agent/a1.py)
- [biomni/eval/__init__.py](biomni/eval/__init__.py)
- [biomni/eval/biomni_eval1.py](biomni/eval/biomni_eval1.py)
- [biomni/tool/support_tools.py](biomni/tool/support_tools.py)
- [biomni/tool/tool_description/support_tools.py](biomni/tool/tool_description/support_tools.py)
- [biomni/utils.py](biomni/utils.py)
- [biomni/version.py](biomni/version.py)

</details>



## Purpose and Scope

This document provides a comprehensive overview of Biomni's system architecture, describing the major components, their interactions, and the overall design philosophy. It covers the high-level structure of the agent system, execution pipelines, tool ecosystem, and data infrastructure.

For detailed information about specific subsystems, refer to:
- Agent internals and state management: [A1 Agent Core](#2.1)
- Code execution mechanisms: [Execution Pipeline](#2.2)  
- LLM provider integration: [LLM Integration Layer](#2.3)
- Tool discovery and selection: [Tool Registry and Retrieval](#2.4)
- System configuration: [Configuration System](#2.5)
- Dataset management: [Data Infrastructure](#2.6)
- Protocol documents: [Know-How Library](#2.7)

## High-Level Architecture

Biomni implements an agent-based architecture where the `A1` class orchestrates complex biomedical workflows through LLM-driven reasoning, tool execution, and iterative refinement.

```mermaid
graph TB
    subgraph "User Interface"
        GradioUI["Gradio Web Interface<br/>launch_gradio_demo()"]
        PythonAPI["Python API<br/>agent.go()"]
        Notebooks["Jupyter Notebooks"]
    end
    
    subgraph "A1 Agent Core"
        A1Class["A1 Class<br/>biomni/agent/a1.py:56"]
        AgentState["AgentState<br/>TypedDict with messages"]
        StateGraph["LangGraph StateGraph<br/>workflow.compile()"]
        Memory["MemorySaver<br/>checkpoint memory"]
    end
    
    subgraph "LLM Integration"
        GetLLM["get_llm()<br/>biomni/llm/__init__.py"]
        Providers["OpenAI | Anthropic<br/>Gemini | Ollama<br/>Custom | Bedrock"]
        ToolCallParser["Tool Call Parser<br/>parse_tool_calls_from_code()"]
    end
    
    subgraph "Tool & Knowledge Systems"
        ToolRegistry["ToolRegistry<br/>biomni/model/retriever.py"]
        ToolRetriever["ToolRetriever<br/>RAG-based selection"]
        KnowHow["KnowHowLoader<br/>biomni/know_how/__init__.py"]
        Module2API["module2api dict<br/>read_module2api()"]
    end
    
    subgraph "Execution Environment"
        PythonREPL["run_python_repl()<br/>_persistent_namespace"]
        RCode["run_r_code()<br/>Rscript subprocess"]
        BashScript["run_bash_script()<br/>bash subprocess"]
        TimeoutWrapper["run_with_timeout()<br/>threading.Thread"]
    end
    
    subgraph "Data & Configuration"
        BiomniConfig["BiomniConfig<br/>default_config"]
        DataLake["data_lake_dict<br/>env_desc.py"]
        LibraryContent["library_content_dict<br/>env_desc.py"]
        S3Download["check_and_download_s3_files()"]
    end
    
    GradioUI --> PythonAPI
    Notebooks --> PythonAPI
    PythonAPI --> A1Class
    
    A1Class --> AgentState
    A1Class --> StateGraph
    A1Class --> GetLLM
    A1Class --> BiomniConfig
    
    StateGraph --> Memory
    
    GetLLM --> Providers
    
    A1Class --> ToolRegistry
    A1Class --> ToolRetriever
    A1Class --> KnowHow
    A1Class --> Module2API
    
    ToolRegistry --> Module2API
    ToolRetriever --> ToolRegistry
    
    A1Class --> PythonREPL
    PythonREPL --> RCode
    PythonREPL --> BashScript
    
    TimeoutWrapper -.wraps.-> PythonREPL
    TimeoutWrapper -.wraps.-> RCode
    TimeoutWrapper -.wraps.-> BashScript
    
    BiomniConfig --> DataLake
    BiomniConfig --> LibraryContent
    DataLake --> S3Download
```

**Sources:** [biomni/agent/a1.py:1-1000](), [biomni/utils.py:1-300](), [biomni/tool/support_tools.py:1-100]()

## Core Components Overview

### A1 Agent Class

The `A1` class is the primary orchestrator, implementing the main agent logic. It is initialized with configuration parameters and manages the entire execution lifecycle.

| Component | Code Location | Description |
|-----------|---------------|-------------|
| **A1 Class** | [biomni/agent/a1.py:56-224]() | Main agent class with initialization, tool management, and workflow execution |
| **AgentState** | [biomni/agent/a1.py:51-54]() | TypedDict defining agent state structure with messages list and next_step |
| **configure()** | [biomni/agent/a1.py:873-1021]() | Generates system prompt with selected tools, data, and know-how |
| **go()** | [biomni/agent/a1.py:1023-1198]() | Primary execution method that runs the agent workflow |

**Sources:** [biomni/agent/a1.py:51-224]()

### LLM Integration Layer

The system supports multiple LLM providers through a unified factory pattern implemented in `get_llm()`.

```mermaid
graph LR
    A1["A1.__init__()"] --> GetLLM["get_llm()<br/>biomni/llm/__init__.py"]
    GetLLM --> SourceCheck{source parameter}
    
    SourceCheck -->|"OpenAI"| OpenAIInit["ChatOpenAI<br/>langchain_openai"]
    SourceCheck -->|"AzureOpenAI"| AzureInit["AzureChatOpenAI<br/>langchain_openai"]
    SourceCheck -->|"Anthropic"| AnthropicInit["ChatAnthropic<br/>langchain_anthropic"]
    SourceCheck -->|"Ollama"| OllamaInit["ChatOllama<br/>langchain_ollama"]
    SourceCheck -->|"Gemini"| GeminiInit["ChatGoogleGenerativeAI<br/>langchain_google_genai"]
    SourceCheck -->|"Bedrock"| BedrockInit["ChatBedrock<br/>langchain_aws"]
    SourceCheck -->|"Groq"| GroqInit["ChatGroq<br/>langchain_groq"]
    SourceCheck -->|"Custom"| CustomInit["ChatOpenAI with base_url<br/>custom endpoint"]
    
    OpenAIInit --> LLMInstance["Configured LLM Instance"]
    AzureInit --> LLMInstance
    AnthropicInit --> LLMInstance
    OllamaInit --> LLMInstance
    GeminiInit --> LLMInstance
    BedrockInit --> LLMInstance
    GroqInit --> LLMInstance
    CustomInit --> LLMInstance
    
    LLMInstance --> StopSeq["Add stop_sequences"]
    StopSeq --> Return["Return configured LLM"]
```

**Sources:** [biomni/llm/__init__.py](), [biomni/agent/a1.py:197-204]()

### Tool Registry and Retrieval System

The tool system uses a two-tier architecture: `ToolRegistry` for storage and `ToolRetriever` for intelligent selection.

```mermaid
graph TB
    subgraph "Tool Metadata Loading"
        ToolDescFiles["tool_description/*.py<br/>description lists"]
        ReadModule2API["read_module2api()<br/>biomni/utils.py"]
        Module2APIDict["module2api dict<br/>{module: [tool_schemas]}"]
    end
    
    subgraph "Tool Registry"
        Registry["ToolRegistry<br/>biomni/model/retriever.py"]
        RegisterTool["register_tool(schema)"]
        ToolsList["self.tools list"]
        DocumentDF["self.document_df<br/>DataFrame with docid"]
    end
    
    subgraph "Tool Retrieval"
        Retriever["ToolRetriever<br/>biomni/model/retriever.py"]
        RetrieveMethod["retrieve(query, tool_registry)<br/>returns selected tool IDs"]
        LLMSelection["LLM-based selection<br/>default_config.llm"]
    end
    
    subgraph "Tool Injection"
        SystemPrompt["System Prompt Generation<br/>A1.configure()"]
        TextifyAPI["textify_api_dict()<br/>formats tools as text"]
        InjectREPL["inject_custom_functions_to_repl()<br/>adds to namespace"]
    end
    
    ToolDescFiles --> ReadModule2API
    ReadModule2API --> Module2APIDict
    
    Module2APIDict --> Registry
    Registry --> RegisterTool
    RegisterTool --> ToolsList
    RegisterTool --> DocumentDF
    
    DocumentDF --> Retriever
    Retriever --> RetrieveMethod
    RetrieveMethod --> LLMSelection
    
    Module2APIDict --> SystemPrompt
    Retriever --> SystemPrompt
    SystemPrompt --> TextifyAPI
    
    SystemPrompt --> InjectREPL
```

**Sources:** [biomni/model/retriever.py](), [biomni/agent/a1.py:873-1021](), [biomni/utils.py:807-900]()

## Agent Workflow Execution

The agent follows a structured workflow implemented using LangGraph's `StateGraph`. The workflow alternates between LLM reasoning and code execution.

```mermaid
sequenceDiagram
    participant User
    participant A1go as "A1.go()"
    participant StateGraph as "LangGraph StateGraph"
    participant CallModel as "call_model node"
    participant LLM as "LLM Provider"
    participant ExecuteTools as "execute_tools node"
    participant REPL as "Execution Environment"
    
    User->>A1go: agent.go("task description")
    A1go->>A1go: Initialize AgentState
    A1go->>StateGraph: workflow.invoke(initial_state)
    
    loop Iterative Execution
        StateGraph->>CallModel: Process state
        CallModel->>LLM: Send messages + system prompt
        LLM-->>CallModel: Response with tool calls
        CallModel->>CallModel: Parse tool calls
        CallModel-->>StateGraph: Update messages
        
        StateGraph->>ExecuteTools: Process tool calls
        
        alt Python Code
            ExecuteTools->>REPL: run_python_repl(code)
            REPL->>REPL: exec() in _persistent_namespace
        else R Code (#!R marker)
            ExecuteTools->>REPL: run_r_code(code)
            REPL->>REPL: Rscript subprocess
        else Bash (#!BASH marker)
            ExecuteTools->>REPL: run_bash_script(code)
            REPL->>REPL: bash subprocess
        end
        
        REPL-->>ExecuteTools: Observation/output
        ExecuteTools->>ExecuteTools: Format observation
        ExecuteTools-->>StateGraph: Update messages with observation
        
        StateGraph->>StateGraph: Check should_continue()
        
        alt Has final answer
            StateGraph-->>A1go: END
        else Continue reasoning
            StateGraph->>CallModel: Next iteration
        end
    end
    
    A1go-->>User: Final result
```

**Sources:** [biomni/agent/a1.py:1023-1198](), [biomni/agent/a1.py:1261-1500]()

### State Graph Node Definitions

The workflow is defined by three primary nodes:

| Node | Code Location | Function |
|------|---------------|----------|
| **call_model** | [biomni/agent/a1.py:1261-1360]() | Sends current state to LLM and parses tool calls from response |
| **execute_tools** | [biomni/agent/a1.py:1362-1467]() | Executes parsed tool calls and returns observations |
| **should_continue** | [biomni/agent/a1.py:1469-1477]() | Conditional edge determining whether to continue or end |

**Sources:** [biomni/agent/a1.py:1261-1477]()

## Multi-Language Execution Architecture

Biomni supports Python, R, and Bash execution through a unified interface with persistent state management and timeout protection.

```mermaid
graph TB
    subgraph "Code Generation & Detection"
        LLMOutput["LLM Generated Code"]
        LanguageDetect["Detect Language Marker"]
        
        LLMOutput --> LanguageDetect
    end
    
    subgraph "Python Execution Path"
        PythonREPL["run_python_repl()<br/>biomni/tool/support_tools.py:13"]
        PersistentNS["_persistent_namespace dict<br/>line 7"]
        ExecFunc["exec(code, namespace)"]
        StdoutCapture["StringIO stdout capture"]
        
        PythonREPL --> PersistentNS
        PersistentNS --> ExecFunc
        ExecFunc --> StdoutCapture
    end
    
    subgraph "R Execution Path"
        RCode["run_r_code()<br/>biomni/utils.py:25"]
        TempRFile["tempfile.NamedTemporaryFile<br/>.R extension"]
        RscriptCmd["subprocess.run<br/>['Rscript', temp_file]"]
        
        RCode --> TempRFile
        TempRFile --> RscriptCmd
    end
    
    subgraph "Bash Execution Path"
        BashScript["run_bash_script()<br/>biomni/utils.py:56"]
        TempShFile["tempfile.NamedTemporaryFile<br/>.sh extension"]
        ChmodExec["os.chmod 0o755"]
        BashRun["subprocess.run<br/>[temp_file]"]
        
        BashScript --> TempShFile
        TempShFile --> ChmodExec
        ChmodExec --> BashRun
    end
    
    subgraph "Timeout Protection"
        TimeoutWrapper["run_with_timeout()<br/>biomni/utils.py:183"]
        ThreadQueue["queue.Queue"]
        DaemonThread["threading.Thread<br/>daemon=True"]
        ThreadJoin["thread.join(timeout)"]
        
        TimeoutWrapper --> ThreadQueue
        TimeoutWrapper --> DaemonThread
        DaemonThread --> ThreadJoin
    end
    
    LanguageDetect -->|"No marker"| PythonREPL
    LanguageDetect -->|"#!R"| RCode
    LanguageDetect -->|"#!BASH or #!CLI"| BashScript
    
    TimeoutWrapper -.wraps.-> PythonREPL
    TimeoutWrapper -.wraps.-> RCode
    TimeoutWrapper -.wraps.-> BashScript
    
    StdoutCapture --> Result["Return output"]
    RscriptCmd --> Result
    BashRun --> Result
```

**Key Implementation Details:**

- **Persistent Python Namespace:** Variables and imports persist across executions via `_persistent_namespace` global dictionary ([biomni/tool/support_tools.py:7]())
- **Language Detection:** Code is parsed for language markers (`#!R`, `#!BASH`, `#!CLI`) to route to appropriate executor ([biomni/agent/a1.py:1382-1396]())
- **Timeout Protection:** All execution wrapped with `run_with_timeout()` using daemon threads and queue-based result passing ([biomni/utils.py:183-243]())
- **Custom Function Injection:** User-added tools via `add_tool()` are injected into `_persistent_namespace` ([biomni/utils.py:34-43]())

**Sources:** [biomni/tool/support_tools.py:1-45](), [biomni/utils.py:25-243](), [biomni/agent/a1.py:1362-1467]()

## Configuration and Initialization Flow

The system uses a hierarchical configuration system with multiple override mechanisms.

```mermaid
graph TB
    subgraph "Configuration Sources"
        EnvFile[".env file<br/>load_dotenv()"]
        EnvVars["Environment Variables<br/>OPENAI_API_KEY, etc."]
        DefaultConfig["BiomniConfig<br/>default_config instance"]
        ConstructorArgs["A1.__init__() arguments<br/>llm, source, path, etc."]
    end
    
    subgraph "Configuration Resolution"
        LoadEnv["Load .env if exists<br/>biomni/agent/a1.py:46-48"]
        ResolveParams["Resolve parameters<br/>lines 83-98"]
        ConfigPrecedence["Precedence:<br/>1. Constructor args<br/>2. default_config<br/>3. Environment vars"]
    end
    
    subgraph "System Initialization"
        CreateDirs["Create data directories<br/>lines 150-162"]
        CheckS3["check_and_download_s3_files()<br/>data_lake + benchmark"]
        LoadModule2API["read_module2api()<br/>tool schemas"]
        InitLLM["get_llm()<br/>provider initialization"]
        InitRetriever["ToolRetriever() if enabled"]
        InitKnowHow["KnowHowLoader()"]
    end
    
    subgraph "Runtime Configuration"
        ConfigureMethod["A1.configure()<br/>line 873"]
        GenSystemPrompt["Generate system prompt"]
        SelectResources["Select tools/data/knowhow"]
        InjectFunctions["inject_custom_functions_to_repl()"]
    end
    
    EnvFile --> LoadEnv
    EnvVars --> LoadEnv
    LoadEnv --> DefaultConfig
    
    DefaultConfig --> ResolveParams
    ConstructorArgs --> ResolveParams
    ResolveParams --> ConfigPrecedence
    
    ConfigPrecedence --> CreateDirs
    CreateDirs --> CheckS3
    CheckS3 --> LoadModule2API
    LoadModule2API --> InitLLM
    InitLLM --> InitRetriever
    InitRetriever --> InitKnowHow
    
    InitKnowHow --> ConfigureMethod
    ConfigureMethod --> GenSystemPrompt
    GenSystemPrompt --> SelectResources
    SelectResources --> InjectFunctions
```

**Configuration Hierarchy:**

1. **Constructor arguments** to `A1()` take highest precedence
2. **`default_config`** values used if constructor argument is `None`
3. **Environment variables** used for API keys and provider settings
4. **`.env` file** loaded if present in working directory

**Important Note:** Constructor arguments to `A1()` only affect agent reasoning LLM. Database queries and retrieval operations always use `default_config` values. For system-wide configuration, modify `default_config` before creating agent instances.

**Sources:** [biomni/agent/a1.py:46-224](), [biomni/config/__init__.py]()

## Data Infrastructure

Biomni maintains a structured data lake with automatic S3 synchronization and commercial mode filtering.

```mermaid
graph LR
    subgraph "Data Lake Structure"
        DataLakeDir["data/biomni_data/data_lake/<br/>local storage"]
        BenchmarkDir["data/biomni_data/benchmark/<br/>evaluation datasets"]
    end
    
    subgraph "Data Metadata"
        EnvDesc["env_desc.py<br/>data_lake_dict"]
        EnvDescCM["env_desc_cm.py<br/>commercial mode version"]
        LibraryDict["library_content_dict<br/>software descriptions"]
    end
    
    subgraph "Data Management"
        S3Bucket["S3 Bucket<br/>biomni-release.s3.amazonaws.com"]
        CheckDownload["check_and_download_s3_files()<br/>biomni/utils.py"]
        ExpectedFiles["expected_data_lake_files list"]
    end
    
    subgraph "Commercial Mode Filtering"
        CommercialFlag["commercial_mode parameter<br/>A1.__init__()"]
        ImportSwitch{commercial_mode?}
        AcademicData["All datasets<br/>including non-commercial"]
        CommercialData["Licensed datasets only<br/>filtered list"]
    end
    
    EnvDesc --> DataLakeDir
    EnvDescCM --> DataLakeDir
    LibraryDict --> DataLakeDir
    
    S3Bucket --> CheckDownload
    ExpectedFiles --> CheckDownload
    CheckDownload --> DataLakeDir
    CheckDownload --> BenchmarkDir
    
    CommercialFlag --> ImportSwitch
    ImportSwitch -->|False| EnvDesc
    ImportSwitch -->|True| EnvDescCM
    EnvDesc --> AcademicData
    EnvDescCM --> CommercialData
```

**Data Lake Components:**

| Component | Description | Location |
|-----------|-------------|----------|
| **data_lake_dict** | Dictionary mapping filenames to descriptions for ~77 datasets | [biomni/env_desc.py]() or [biomni/env_desc_cm.py]() |
| **library_content_dict** | Dictionary of available Python packages, R packages, and CLI tools | [biomni/env_desc.py]() |
| **S3 Synchronization** | Automatic download of missing files from S3 bucket | [biomni/utils.py:845-920]() |
| **Commercial Mode** | Filters datasets based on licensing restrictions | [biomni/agent/a1.py:100-113]() |

**Sources:** [biomni/agent/a1.py:100-194](), [biomni/env_desc.py](), [biomni/utils.py:845-920]()

## Tool Extension Architecture

Users can dynamically add custom tools, data, and software to extend agent capabilities.

```mermaid
graph TB
    subgraph "Tool Addition Flow"
        UserTool["User defines function"]
        AddTool["agent.add_tool(api)"]
        InspectSource["inspect.getsource(api)"]
        GenSchema["function_to_api_schema(code, llm)<br/>LLM generates schema"]
        
        UserTool --> AddTool
        AddTool --> InspectSource
        InspectSource --> GenSchema
    end
    
    subgraph "Registration"
        RegisterTR["ToolRegistry.register_tool(schema)"]
        UpdateM2A["Add to module2api dict"]
        CustomFuncs["Store in _custom_functions"]
        CustomTools["Store in _custom_tools"]
        BuiltinsNS["Add to builtins._biomni_custom_functions"]
        
        GenSchema --> RegisterTR
        GenSchema --> UpdateM2A
        GenSchema --> CustomFuncs
        GenSchema --> CustomTools
        GenSchema --> BuiltinsNS
    end
    
    subgraph "Data Addition Flow"
        UserData["User provides data dict<br/>{path: description}"]
        AddData["agent.add_data(data)"]
        CustomDataStore["_custom_data storage"]
        UpdateDLDict["Update data_lake_dict"]
        
        UserData --> AddData
        AddData --> CustomDataStore
        AddData --> UpdateDLDict
    end
    
    subgraph "Software Addition Flow"
        UserSW["User provides software dict<br/>{name: description}"]
        AddSoftware["agent.add_software(software)"]
        CustomSWStore["_custom_software storage"]
        UpdateLibDict["Update library_content_dict"]
        
        UserSW --> AddSoftware
        AddSoftware --> CustomSWStore
        AddSoftware --> UpdateLibDict
    end
    
    subgraph "MCP Integration Flow"
        MCPConfig["MCP config YAML file"]
        AddMCP["agent.add_mcp(config_path)"]
        DiscoverTools["discover_mcp_tools_sync()<br/>auto-discover from server"]
        WrapperFunc["make_mcp_wrapper()<br/>async to sync wrapper"]
        MCPModule["Create sys.modules entry<br/>mcp_servers.{name}"]
        
        MCPConfig --> AddMCP
        AddMCP --> DiscoverTools
        DiscoverTools --> WrapperFunc
        WrapperFunc --> MCPModule
        MCPModule --> RegisterTR
    end
    
    subgraph "Final Steps"
        ReconfigureAgent["agent.configure()<br/>regenerate system prompt"]
    end
    
    RegisterTR --> ReconfigureAgent
    UpdateM2A --> ReconfigureAgent
    UpdateDLDict --> ReconfigureAgent
    UpdateLibDict --> ReconfigureAgent
    MCPModule --> ReconfigureAgent
```

**Extension Methods:**

| Method | Purpose | Code Location |
|--------|---------|---------------|
| **add_tool(api)** | Add custom Python function as tool | [biomni/agent/a1.py:225-348]() |
| **add_data(data)** | Add dataset to data lake | [biomni/agent/a1.py:676-721]() |
| **add_software(software)** | Add software description | [biomni/agent/a1.py:777-820]() |
| **add_mcp(config_path)** | Integrate MCP server tools | [biomni/agent/a1.py:350-587]() |

**Key Features:**

- **Automatic Schema Generation:** LLM analyzes function source code to generate API schema ([biomni/utils.py:251-290]())
- **MCP Auto-Discovery:** MCP servers can be queried for available tools instead of manual configuration ([biomni/agent/a1.py:382-413]())
- **Persistent Storage:** Custom additions stored in instance attributes and persist across `go()` calls
- **System Prompt Regeneration:** `configure()` automatically called after adding resources to update agent context

**Sources:** [biomni/agent/a1.py:225-820](), [biomni/utils.py:251-290]()

## Evaluation Framework Integration

BiomniEval1 provides standardized benchmarking across 10 task types with 433 instances.

```mermaid
graph TB
    subgraph "BiomniEval1 Class"
        EvalInit["BiomniEval1.__init__()"]
        LoadDF["Load parquet from HuggingFace<br/>hf://datasets/biomni/Eval1"]
        InstanceMap["Create instance_map dict<br/>{(task_name, task_id): df_idx}"]
        
        EvalInit --> LoadDF
        LoadDF --> InstanceMap
    end
    
    subgraph "Evaluation Methods"
        Evaluate["evaluate(task_name, task_id, answer)"]
        GetInstance["get_instance(task_name, task_id)"]
        BatchEval["batch_evaluate(evaluations)"]
        
        InstanceMap --> Evaluate
        InstanceMap --> GetInstance
        Evaluate --> BatchEval
    end
    
    subgraph "Task-Specific Reward Logic"
        ComputeReward["_compute_reward(task, answer, gt)"]
        
        GWASGene["GWAS Causal Gene<br/>exact match, case-insensitive"]
        CRISPRDel["CRISPR Delivery<br/>letter match a-f"]
        RareDisease["Rare Disease Diagnosis<br/>JSON OMIM_ID match"]
        PatientGene["Patient Gene Detection<br/>list intersection"]
        LabBench["Lab Bench QA<br/>letter match A-Z"]
        
        Evaluate --> ComputeReward
        
        ComputeReward --> GWASGene
        ComputeReward --> CRISPRDel
        ComputeReward --> RareDisease
        ComputeReward --> PatientGene
        ComputeReward --> LabBench
    end
    
    subgraph "Dataset Structure"
        DataFrame["Merged Parquet DataFrame"]
        
        Columns["Columns:<br/>- instance_id<br/>- task_instance_id<br/>- task_name<br/>- split (train/val)<br/>- prompt<br/>- answer"]
        
        DataFrame --> Columns
    end
```

**Evaluation Interface:**

```python
from biomni.eval import BiomniEval1

evaluator = BiomniEval1()
score = evaluator.evaluate('gwas_causal_gene_opentargets', 0, 'BRCA1')
# Returns: 1.0 (correct) or 0.0 (incorrect)
```

**Task Types and Validation:**

| Task Category | Evaluation Method | Code Location |
|---------------|-------------------|---------------|
| **GWAS Causal Gene** | Exact string match (case-insensitive) | [biomni/eval/biomni_eval1.py:78-80]() |
| **CRISPR Delivery** | Letter match (a-f, case-insensitive) | [biomni/eval/biomni_eval1.py:74-76]() |
| **Rare Disease Diagnosis** | JSON OMIM_ID field match | [biomni/eval/biomni_eval1.py:94-117]() |
| **Patient Gene Detection** | List intersection of causal_gene field | [biomni/eval/biomni_eval1.py:123-155]() |
| **Lab Bench Q&A** | Letter match (A-Z, case-insensitive) | [biomni/eval/biomni_eval1.py:90-92]() |

**Sources:** [biomni/eval/biomni_eval1.py:1-325]()

## System Initialization Sequence

This diagram shows the complete initialization sequence when creating an `A1` agent instance.

```mermaid
sequenceDiagram
    participant User
    participant A1Init as "A1.__init__()"
    participant LoadEnv as "load_dotenv()"
    participant BiomniConfig as "default_config"
    participant PathSetup as "Path Management"
    participant S3Sync as "S3 Synchronization"
    participant ModuleLoader as "read_module2api()"
    participant LLMFactory as "get_llm()"
    participant ToolSys as "Tool System"
    participant KnowHow as "KnowHowLoader"
    participant Configure as "A1.configure()"
    
    User->>A1Init: A1(path, llm, source, ...)
    
    A1Init->>LoadEnv: Load .env if exists
    LoadEnv-->>A1Init: Environment variables loaded
    
    A1Init->>BiomniConfig: Get default values
    BiomniConfig-->>A1Init: Configuration defaults
    
    A1Init->>A1Init: Resolve parameters<br/>(constructor > default_config > env)
    
    A1Init->>A1Init: Import env_desc or env_desc_cm<br/>based on commercial_mode
    
    A1Init->>PathSetup: Check/create directories
    PathSetup->>PathSetup: os.makedirs(benchmark_dir)
    PathSetup->>PathSetup: os.makedirs(data_lake_dir)
    PathSetup-->>A1Init: Directories ready
    
    A1Init->>S3Sync: check_and_download_s3_files()
    S3Sync->>S3Sync: Check local vs expected files
    
    loop For each missing file
        S3Sync->>S3Sync: Download from S3 with progress bar
    end
    
    S3Sync-->>A1Init: Data lake synchronized
    
    A1Init->>ModuleLoader: read_module2api()
    ModuleLoader->>ModuleLoader: Load tool_description/*.py
    ModuleLoader-->>A1Init: module2api dict
    
    A1Init->>LLMFactory: get_llm(llm, source, ...)
    LLMFactory->>LLMFactory: Initialize provider client
    LLMFactory->>LLMFactory: Add stop_sequences
    LLMFactory-->>A1Init: Configured LLM instance
    
    alt use_tool_retriever enabled
        A1Init->>ToolSys: Initialize ToolRegistry
        ToolSys-->>A1Init: Registry ready
        A1Init->>ToolSys: Initialize ToolRetriever
        ToolSys-->>A1Init: Retriever ready
    end
    
    A1Init->>KnowHow: KnowHowLoader()
    KnowHow->>KnowHow: Load know_how/*.md files
    
    alt commercial_mode enabled
        KnowHow->>KnowHow: Filter non-commercial documents
    end
    
    KnowHow-->>A1Init: Know-how library loaded
    
    A1Init->>Configure: self.configure()
    Configure->>Configure: Build workflow StateGraph
    Configure->>Configure: Generate system prompt template
    Configure->>Configure: inject_custom_functions_to_repl()
    Configure-->>A1Init: Agent configured
    
    A1Init-->>User: A1 instance ready
```

**Sources:** [biomni/agent/a1.py:46-224](), [biomni/utils.py:845-920]()

## Summary

Biomni's architecture is designed around several key principles:

1. **Modularity:** Clear separation between agent core, execution environment, tool system, and data infrastructure
2. **Extensibility:** Dynamic tool, data, and software addition without code modification
3. **Multi-LLM Support:** Provider-agnostic design through factory pattern
4. **Persistent Execution:** Python REPL maintains state across multiple executions
5. **Safety:** Timeout protection and error handling for all code execution
6. **Knowledge-Augmented:** Integration of structured know-how documents and data lake metadata

The system's design enables complex biomedical workflows while maintaining flexibility for different deployment scenarios (academic vs. commercial, different LLM providers, custom tool additions).

---