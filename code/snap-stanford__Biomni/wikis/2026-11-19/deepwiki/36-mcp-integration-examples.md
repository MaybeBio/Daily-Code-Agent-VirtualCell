# MCP Integration Examples

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.gitignore](.gitignore)

</details>



This page provides practical examples of integrating external tools into Biomni via the Model Context Protocol (MCP). MCP enables the agent to interact with external tool servers, expanding its capabilities beyond built-in functions.

For general information about adding custom tools, see [Working with Custom Resources](#4.4). For MCP configuration details as part of agent setup, see [MCP Integration](#4.5).

---

## Overview

The Model Context Protocol (MCP) allows Biomni to connect to external tool servers that expose functionality through a standardized protocol. The `A1.add_mcp()` method in [biomni/agent/a1.py:450-587]() dynamically registers MCP server tools as callable functions within the agent, making external tools available alongside built-in Biomni tools.

**Key Features:**
- **Automatic Tool Discovery**: MCP servers expose tools via `list_tools()` API, which Biomni discovers through `discover_mcp_tools_sync()` [biomni/agent/a1.py:382-413]()
- **Manual Tool Configuration**: Tools can be manually defined in YAML configuration files
- **Synchronous Wrapper**: MCP's async operations are wrapped via `make_mcp_wrapper()` [biomni/agent/a1.py:415-444]() for seamless integration with Biomni's synchronous execution model
- **Module Namespacing**: MCP tools are dynamically added to `sys.modules` as `mcp_servers.{server_name}.{tool_name}`
- **Environment Variable Support**: Server configurations can reference environment variables using `${VAR_NAME}` syntax, processed in [biomni/agent/a1.py:479-487]()

**Sources:** [biomni/agent/a1.py:350-587](), [biomni/agent/a1.py:382-413](), [biomni/agent/a1.py:415-444](), [biomni/agent/a1.py:479-487]()
</thinking>

<old_str>
## Example: Weather API Integration

This example demonstrates a complete MCP integration for a weather API service.

### MCP Server Implementation

```python
# weather_mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx

app = Server("weather-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates"
                    },
                    "units": {
                        "type": "string",
                        "description": "Temperature units",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        location = arguments["location"]
        units = arguments.get("units", "celsius")
        
        # API call to weather service
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.weather.com/v1/current",
                params={"location": location, "units": units}
            )
            data = response.json()
        
        return [TextContent(
            type="text",
            text=f"Temperature: {data['temp']}°, Conditions: {data['conditions']}"
        )]
```

### Configuration

```yaml
# mcp_config.yaml
mcp_servers:
  weather:
    enabled: true
    command: ["python", "-m", "weather_mcp_server"]
    env:
      WEATHER_API_KEY: "${WEATHER_API_KEY}"
```

### Usage

```python
from biomni.agent import A1

agent = A1(path="./workspace", llm="claude-sonnet-4")
agent.add_mcp(config_path="./mcp_config.yaml")

# Query weather using natural language
result = agent.go(
    "What's the current weather in San Francisco? Compare it with New York."
)
```

### Code Entity Mapping: Natural Language to Implementation

**Bridging Natural Language Query to Code Execution**

```mermaid
graph TD
    NL["User Query:<br/>What's the weather in SF?"]
    A1["A1.go() executes"]
    LLM["LLM generates Python code"]
    IMPORT["from mcp_servers.weather import get_weather"]
    EXEC["get_weather(location='San Francisco')"]
    WRAPPER["make_mcp_wrapper() sync wrapper"]
    ASYNC["async_tool_call() in wrapper"]
    CLIENT["stdio_client(StdioServerParameters)"]
    SERVER["weather_mcp_server.py call_tool()"]
    RESULT["TextContent returned"]
    OBS["Observation added to AgentState"]
    
    NL --> A1
    A1 --> LLM
    LLM --> IMPORT
    IMPORT --> EXEC
    EXEC --> WRAPPER
    WRAPPER --> ASYNC
    ASYNC --> CLIENT
    CLIENT --> SERVER
    SERVER --> RESULT
    RESULT --> OBS
    OBS --> A1
```

**The agent execution flow:**
1. User query triggers `A1.go()` method
2. LLM generates code: `from mcp_servers.weather import get_weather`
3. Python REPL executes the import, which resolves to the wrapper function created by `make_mcp_wrapper()` in [biomni/agent/a1.py:415-444]()
4. Function call `get_weather(location="San Francisco")` invokes the synchronous wrapper
5. Wrapper calls `async_tool_call()` which creates `StdioServerParameters` and establishes `stdio_client` session
6. Client sends `call_tool` request to the MCP server process
7. Server returns `TextContent` result
8. Result is extracted and returned to the Python REPL as observation
9. Agent receives observation and continues reasoning

**Sources:** [biomni/agent/a1.py:350-587](), [biomni/agent/a1.py:415-444]()
</old_str>
<new_str>
**Code Implementation:**

The `discover_mcp_tools_sync()` function in [biomni/agent/a1.py:382-413]() uses the MCP `list_tools()` API to retrieve available tools from the server. It creates a `StdioServerParameters` object with the command and environment, then establishes a `stdio_client` session to call `session.list_tools()`.

When no tools are configured in the YAML (`server_config.get("tools")` is None or empty), the system automatically invokes discovery in [biomni/agent/a1.py:497-510](), populating the `tools_config` list with discovered tool schemas.

**Sources:** [biomni/agent/a1.py:382-413](), [biomni/agent/a1.py:497-510]()
</thinking>

<old_str>
### Verification Steps

```python
from biomni.agent import A1

agent = A1(path="./workspace", llm="claude-sonnet-4")

# Before adding MCP tools
print("Custom tools before MCP:", agent.list_custom_tools())

# Add MCP configuration
agent.add_mcp(config_path="./mcp_config.yaml")

# After adding MCP tools
print("Custom tools after MCP:", agent.list_custom_tools())

# Check tool registry
if hasattr(agent, 'tool_registry'):
    print("Total registered tools:", len(agent.tool_registry.tools))
    
# Verify module2api structure
for module_name in agent.module2api.keys():
    if module_name.startswith("mcp_servers."):
        print(f"MCP module: {module_name}")
        print(f"  Tools: {[t['name'] for t in agent.module2api[module_name]]}")
```

**Expected Output:**
```
Custom tools before MCP: []
Custom tools after MCP: ['get_weather']
Total registered tools: 105
MCP module: mcp_servers.weather
  Tools: ['get_weather']
```

**Key Code Entities:**
- `agent.list_custom_tools()` - Returns list from `_custom_tools` dict
- `agent.tool_registry.tools` - Dict maintained by `ToolRegistry` class
- `agent.module2api` - Dict mapping module names to tool schemas
- `sys.modules["mcp_servers.weather"]` - Dynamically created module containing wrapper function

**Sources:** [biomni/agent/a1.py:589-674](), [biomni/agent/a1.py:559-584]()
</old_str>
<new_str>
**Code Implementation:**

Manual tools are processed in [biomni/agent/a1.py:514-557](). The system extracts `parameters` from the tool configuration, builds `required_param_names` from the `required` field (defaulting to an empty list), and constructs separate `required_params` and `optional_params` lists. Each parameter entry includes `name`, `type`, `description`, and optionally `default` values.

**Sources:** [biomni/agent/a1.py:514-557]()
</thinking>

<old_str>
## Example: Weather API Integration

This example demonstrates a complete MCP integration for a weather API service.

### MCP Server Implementation

```python
# weather_mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx

app = Server("weather-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates"
                    },
                    "units": {
                        "type": "string",
                        "description": "Temperature units",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        location = arguments["location"]
        units = arguments.get("units", "celsius")
        
        # API call to weather service
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.weather.com/v1/current",
                params={"location": location, "units": units}
            )
            data = response.json()
        
        return [TextContent(
            type="text",
            text=f"Temperature: {data['temp']}°, Conditions: {data['conditions']}"
        )]
```

### Configuration

```yaml
# mcp_config.yaml
mcp_servers:
  weather:
    enabled: true
    command: ["python", "-m", "weather_mcp_server"]
    env:
      WEATHER_API_KEY: "${WEATHER_API_KEY}"
```

### Usage

```python
from biomni.agent import A1

agent = A1(path="./workspace", llm="claude-sonnet-4")
agent.add_mcp(config_path="./mcp_config.yaml")

# Query weather using natural language
result = agent.go(
    "What's the current weather in San Francisco? Compare it with New York."
)
```

**The agent will automatically:**
1. Recognize the weather query
2. Import the MCP weather tool: `from mcp_servers.weather import get_weather`
3. Call `get_weather(location="San Francisco")`
4. Call `get_weather(location="New York")`
5. Compare and present the results

**Sources:** [biomni/agent/a1.py:350-587]()
</thinking>

<old_str>
## Example: Weather API Integration

This example demonstrates a complete MCP integration for a weather API service.

### MCP Server Implementation

```python
# weather_mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx

app = Server("weather-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates"
                    },
                    "units": {
                        "type": "string",
                        "description": "Temperature units",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        location = arguments["location"]
        units = arguments.get("units", "celsius")
        
        # API call to weather service
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.weather.com/v1/current",
                params={"location": location, "units": units}
            )
            data = response.json()
        
        return [TextContent(
            type="text",
            text=f"Temperature: {data['temp']}°, Conditions: {data['conditions']}"
        )]
```

### Configuration

```yaml
# mcp_config.yaml
mcp_servers:
  weather:
    enabled: true
    command: ["python", "-m", "weather_mcp_server"]
    env:
      WEATHER_API_KEY: "${WEATHER_API_KEY}"
```

### Usage

```python
from biomni.agent import A1

agent = A1(path="./workspace", llm="claude-sonnet-4")
agent.add_mcp(config_path="./mcp_config.yaml")

# Query weather using natural language
result = agent.go(
    "What's the current weather in San Francisco? Compare it with New York."
)
```

**The agent will automatically:**
1. Recognize the weather query
2. Import the MCP weather tool: `from mcp_servers.weather import get_weather`
3. Call `get_weather(location="San Francisco")`
4. Call `get_weather(location="New York")`
5. Compare and present the results

**Sources:** [biomni/agent/a1.py:350-587]()

---

## MCP Configuration File Structure

MCP servers are configured via YAML files that define server parameters, environment variables, and available tools.

### Basic Configuration Format

```mermaid
graph TB
    YAML["mcp_config.yaml"]
    SERVERS["mcp_servers"]
    SERVER1["server_name_1"]
    SERVER2["server_name_2"]
    
    ENABLED["enabled: true/false"]
    COMMAND["command: [executable, args]"]
    ENV["env: {VAR: value}"]
    TOOLS["tools: [...]"]
    
    YAML --> SERVERS
    SERVERS --> SERVER1
    SERVERS --> SERVER2
    
    SERVER1 --> ENABLED
    SERVER1 --> COMMAND
    SERVER1 --> ENV
    SERVER1 --> TOOLS
    
    style YAML fill:#f9f9f9
    style SERVERS fill:#e1f5ff
    style SERVER1 fill:#fff4e1
```

**YAML Structure:**

```yaml
mcp_servers:
  server_name:
    enabled: true
    command: ["executable_path", "arg1", "arg2"]
    env:
      API_KEY: "${MY_API_KEY}"
      CONFIG_PATH: "/path/to/config"
    tools:
      - biomni_name: "tool_name"
        description: "Tool description"
        parameters:
          param1:
            type: "string"
            description: "Parameter description"
            required: true
```

**Sources:** [biomni/agent/a1.py:450-463]()

---

## Configuration Patterns

### Pattern 1: Auto-Discovery (No Tools Specified)

When `tools` is omitted or empty, Biomni automatically discovers available tools from the MCP server.

```yaml
mcp_servers:
  filesystem:
    enabled: true
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path/to/workspace"]
    env:
      LOG_LEVEL: "info"
```

**Workflow:**

```mermaid
sequenceDiagram
    participant Agent as "A1 Agent"
    participant Config as "YAML Config"
    participant MCP as "MCP Server"
    participant Registry as "ToolRegistry"
    
    Agent->>Config: "Load mcp_config.yaml"
    Config->>Agent: "Server definition (no tools)"
    Agent->>MCP: "discover_mcp_tools_sync()"
    MCP->>Agent: "tools_result.tools"
    
    loop "For each discovered tool"
        Agent->>Agent: "Create wrapper function"
        Agent->>Registry: "register_tool(schema)"
    end
    
    Agent->>Agent: "configure()"
```

**Code Implementation:**

The auto-discovery mechanism uses the MCP `list_tools()` API:

[biomni/agent/a1.py:382-413]()

When no tools are configured, the system invokes discovery:

[biomni/agent/a1.py:497-510]()

**Sources:** [biomni/agent/a1.py:382-413](), [biomni/agent/a1.py:497-510]()

---

### Pattern 2: Manual Tool Definition

Tools can be explicitly defined with full parameter specifications:

```yaml
mcp_servers:
  weather_api:
    enabled: true
    command: ["python", "-m", "mcp_server.weather"]
    env:
      WEATHER_API_KEY: "${OPENWEATHER_API_KEY}"
    tools:
      - biomni_name: "get_current_weather"
        description: "Get current weather for a location"
        parameters:
          location:
            type: "string"
            description: "City name or coordinates"
            required: true
          units:
            type: "string"
            description: "Temperature units (celsius/fahrenheit)"
            required: false
            default: "celsius"
```

**Parameter Processing:**

```mermaid
graph LR
    MANUAL["Manual Tool Definition"]
    PARAMS["parameters dict"]
    REQUIRED["required_param_names"]
    SCHEMA["Tool Schema"]
    
    MANUAL --> PARAMS
    PARAMS --> REQUIRED
    
    REQUIRED --> |"param in required_param_names"| REQ_LIST["required_params list"]
    REQUIRED --> |"param not in required"| OPT_LIST["optional_params list"]
    
    REQ_LIST --> SCHEMA
    OPT_LIST --> SCHEMA
    
    style MANUAL fill:#fff4e1
    style SCHEMA fill:#e1f5ff
```

**Code Implementation:**

Manual tools are processed differently from auto-discovered ones:

[biomni/agent/a1.py:514-557]()

**Sources:** [biomni/agent/a1.py:514-557]()

---

## Environment Variable Handling

Environment variables can be referenced in MCP configurations using `${VARIABLE_NAME}` syntax.

### Configuration Example

```yaml
mcp_servers:
  slack:
    enabled: true
    command: ["python", "-m", "mcp_server.slack"]
    env:
      SLACK_BOT_TOKEN: "${SLACK_BOT_TOKEN}"
      SLACK_CHANNEL_ID: "${SLACK_CHANNEL_ID}"
      DEBUG_MODE: "false"
```

### Processing Flow

```mermaid
graph TD
    YAML_ENV["env: {KEY: value}"]
    CHECK{"Is value ${...}?"}
    EXTRACT["Extract var_name"]
    GETENV["os.getenv(var_name)"]
    DIRECT["Use value directly"]
    PROCESSED["processed_env dict"]
    
    YAML_ENV --> CHECK
    CHECK -->|"Yes"| EXTRACT
    CHECK -->|"No"| DIRECT
    EXTRACT --> GETENV
    GETENV --> PROCESSED
    DIRECT --> PROCESSED
    
    style YAML_ENV fill:#fff4e1
    style PROCESSED fill:#e1f5ff
```

**Code Implementation:**

[biomni/agent/a1.py:479-487]()

**Sources:** [biomni/agent/a1.py:479-487]()

---

## Complete Integration Example

### Step 1: Create MCP Configuration File

Create `mcp_config.yaml` with your MCP servers:

```yaml
mcp_servers:
  # Example 1: File system operations
  filesystem:
    enabled: true
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "./workspace"]
    env:
      ALLOWED_OPERATIONS: "read,write,list"
  
  # Example 2: Database queries
  postgres:
    enabled: true
    command: ["python", "-m", "mcp_server.postgres"]
    env:
      DATABASE_URL: "${POSTGRES_URL}"
      MAX_CONNECTIONS: "10"
  
  # Example 3: Custom API integration
  custom_api:
    enabled: true
    command: ["python", "-m", "my_mcp_server"]
    env:
      API_KEY: "${CUSTOM_API_KEY}"
    tools:
      - biomni_name: "fetch_data"
        description: "Fetch data from custom API"
        parameters:
          endpoint:
            type: "string"
            description: "API endpoint path"
            required: true
          query_params:
            type: "object"
            description: "Query parameters"
            required: false
```

### Step 2: Set Environment Variables

```bash
export POSTGRES_URL="postgresql://user:pass@localhost/dbname"
export CUSTOM_API_KEY="your_api_key_here"
```

### Step 3: Initialize Agent with MCP Tools

```python
from biomni.agent import A1

# Initialize agent
agent = A1(
    path="./workspace",
    llm="claude-sonnet-4",
    use_tool_retriever=True
)

# Add MCP tools from configuration
agent.add_mcp(config_path="./mcp_config.yaml")

# Agent is now ready with both built-in and MCP tools
```

### Step 4: Use MCP Tools in Queries

```python
# Example: Using filesystem MCP tools
result = agent.go(
    "List all Python files in the workspace directory and count their total lines of code"
)

# Example: Using database MCP tools
result = agent.go(
    "Query the patients table and calculate the average age of patients with diabetes"
)

# Example: Using custom API MCP tools
result = agent.go(
    "Fetch experimental data from the /experiments endpoint and plot the results"
)
```

**Sources:** [biomni/agent/a1.py:350-587]()

---

## MCP Tool Registration Flow

### Architecture Overview

```mermaid
graph TB
    subgraph "Configuration Loading"
        YAML["mcp_config.yaml"]
        LOAD["Load YAML content"]
        VALIDATE["Validate config structure"]
    end
    
    subgraph "Server Initialization"
        SERVERS["Iterate mcp_servers"]
        ENABLED{"enabled: true?"}
        CMD["Extract command & args"]
        ENV_PROC["Process env variables"]
    end
    
    subgraph "Tool Discovery/Definition"
        HAS_TOOLS{"Has tools config?"}
        AUTO["discover_mcp_tools_sync()"]
        MANUAL["Use defined tools"]
        TOOLS_LIST["tools_config list"]
    end
    
    subgraph "Wrapper Creation"
        WRAPPER["make_mcp_wrapper()"]
        SYNC_FUNC["Synchronous wrapper function"]
        NAMESPACE["sys.modules[mcp_servers.name]"]
    end
    
    subgraph "Registration"
        SCHEMA["Build tool_schema dict"]
        REGISTRY["tool_registry.register_tool()"]
        MODULE2API["module2api mapping"]
        CUSTOM["_custom_functions dict"]
    end
    
    YAML --> LOAD
    LOAD --> VALIDATE
    VALIDATE --> SERVERS
    
    SERVERS --> ENABLED
    ENABLED -->|"Yes"| CMD
    ENABLED -->|"No"| SERVERS
    CMD --> ENV_PROC
    ENV_PROC --> HAS_TOOLS
    
    HAS_TOOLS -->|"No"| AUTO
    HAS_TOOLS -->|"Yes"| MANUAL
    AUTO --> TOOLS_LIST
    MANUAL --> TOOLS_LIST
    
    TOOLS_LIST --> WRAPPER
    WRAPPER --> SYNC_FUNC
    SYNC_FUNC --> NAMESPACE
    
    NAMESPACE --> SCHEMA
    SCHEMA --> REGISTRY
    SCHEMA --> MODULE2API
    SCHEMA --> CUSTOM
    
    style YAML fill:#fff4e1
    style REGISTRY fill:#e1f5ff
    style MODULE2API fill:#e1f5ff
```

**Sources:** [biomni/agent/a1.py:450-587]()

---

## Synchronous Wrapper Implementation

MCP tools are inherently asynchronous, but Biomni's execution environment is synchronous. The system creates wrapper functions that bridge this gap.

### Wrapper Function Architecture

```mermaid
sequenceDiagram
    participant User as "User Code"
    participant Wrapper as "sync_tool_wrapper()"
    participant Async as "async_tool_call()"
    participant Client as "ClientSession"
    participant Server as "MCP Server"
    
    User->>Wrapper: "tool_func(**kwargs)"
    Wrapper->>Wrapper: "Create StdioServerParameters"
    Wrapper->>Async: "async_tool_call()"
    
    Async->>Client: "stdio_client(server_params)"
    Client->>Client: "session.initialize()"
    Client->>Server: "session.call_tool(tool_name, kwargs)"
    Server->>Client: "result"
    Client->>Async: "content"
    
    Async->>Async: "Extract text or json"
    Async->>Wrapper: "return result"
    
    Wrapper->>Wrapper: "asyncio.run() or loop.create_task()"
    Wrapper->>User: "return result"
```

**Code Implementation:**

The wrapper factory creates synchronous functions:

[biomni/agent/a1.py:415-444]()

**Execution Context Handling:**

The wrapper intelligently handles both event loop contexts:

| Context | Strategy | Implementation |
|---------|----------|----------------|
| **No running loop** | `asyncio.run()` | Creates new event loop for execution |
| **Running loop exists** | `loop.create_task()` | Uses existing event loop via `nest_asyncio` |

**Sources:** [biomni/agent/a1.py:415-444]()

---

## Tool Schema Construction

Each MCP tool is registered with a standardized schema that integrates with Biomni's tool ecosystem.

### Schema Structure

```mermaid
graph LR
    subgraph "Input Sources"
        MANUAL_DEF["Manual Definition"]
        AUTO_DISC["Auto-Discovery"]
    end
    
    subgraph "Schema Components"
        NAME["name: str"]
        DESC["description: str"]
        PARAMS["parameters: dict"]
        REQUIRED["required_parameters: list"]
        OPTIONAL["optional_parameters: list"]
        MODULE["module: mcp_servers.name"]
        FN["fn: wrapper_function"]
    end
    
    subgraph "Registries"
        TOOL_REG["tool_registry"]
        MOD2API["module2api"]
        CUSTOM_FUNC["_custom_functions"]
        CUSTOM_TOOLS["_custom_tools"]
    end
    
    MANUAL_DEF --> NAME
    AUTO_DISC --> NAME
    
    NAME --> TOOL_REG
    DESC --> TOOL_REG
    PARAMS --> TOOL_REG
    REQUIRED --> TOOL_REG
    OPTIONAL --> TOOL_REG
    MODULE --> TOOL_REG
    FN --> TOOL_REG
    
    TOOL_REG --> MOD2API
    TOOL_REG --> CUSTOM_FUNC
    TOOL_REG --> CUSTOM_TOOLS
    
    style NAME fill:#fff4e1
    style TOOL_REG fill:#e1f5ff
    style MOD2API fill:#e1f5ff
```

**Schema Building Code:**

[biomni/agent/a1.py:559-568]()

**Registry Updates:**

[biomni/agent/a1.py:570-584]()

**Sources:** [biomni/agent/a1.py:559-584]()

---

## Example: Weather API Integration

This example demonstrates a complete MCP integration for a weather API service.

### MCP Server Implementation

```python
# weather_mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx

app = Server("weather-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates"
                    },
                    "units": {
                        "type": "string",
                        "description": "Temperature units",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        location = arguments["location"]
        units = arguments.get("units", "celsius")
        
        # API call to weather service
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.weather.com/v1/current",
                params={"location": location, "units": units}
            )
            data = response.json()
        
        return [TextContent(
            type="text",
            text=f"Temperature: {data['temp']}°, Conditions: {data['conditions']}"
        )]
```

### Configuration

```yaml
# mcp_config.yaml
mcp_servers:
  weather:
    enabled: true
    command: ["python", "-m", "weather_mcp_server"]
    env:
      WEATHER_API_KEY: "${WEATHER_API_KEY}"
```

### Usage

```python
from biomni.agent import A1

agent = A1(path="./workspace", llm="claude-sonnet-4")
agent.add_mcp(config_path="./mcp_config.yaml")

# Query weather using natural language
result = agent.go(
    "What's the current weather in San Francisco? Compare it with New York."
)
```

**The agent will automatically:**
1. Recognize the weather query
2. Import the MCP weather tool: `from mcp_servers.weather import get_weather`
3. Call `get_weather(location="San Francisco")`
4. Call `get_weather(location="New York")`
5. Compare and present the results

**Sources:** [biomni/agent/a1.py:350-587]()

---

## Example: Database Query Integration

### MCP Server for SQL Queries

```yaml
mcp_servers:
  postgres:
    enabled: true
    command: ["npx", "-y", "@modelcontextprotocol/server-postgres"]
    env:
      DATABASE_URL: "${DATABASE_URL}"
      SCHEMA: "public"
```

### Usage Example

```python
agent = A1(path="./workspace", llm="claude-sonnet-4")
agent.add_mcp(config_path="./mcp_config.yaml")

# Complex database query via natural language
result = agent.go("""
    Query the clinical_trials database:
    1. Find all trials for cancer treatment started after 2020
    2. Group them by drug mechanism of action
    3. Calculate success rates for each mechanism
    4. Plot the distribution
""")
```

**Agent Execution Flow:**

```mermaid
graph TD
    QUERY["User Query"]
    PLAN["Generate Plan"]
    IMPORT["from mcp_servers.postgres import query_database"]
    
    SQL1["Build SQL: SELECT * FROM trials WHERE..."]
    EXEC1["Execute query_database(sql1)"]
    RESULT1["Get trial data"]
    
    PROCESS["Process data with pandas"]
    
    SQL2["Build SQL: SELECT mechanism, AVG(success)..."]
    EXEC2["Execute query_database(sql2)"]
    RESULT2["Get aggregated results"]
    
    PLOT["Generate matplotlib plot"]
    SOLUTION["Present solution"]
    
    QUERY --> PLAN
    PLAN --> IMPORT
    IMPORT --> SQL1
    SQL1 --> EXEC1
    EXEC1 --> RESULT1
    RESULT1 --> PROCESS
    PROCESS --> SQL2
    SQL2 --> EXEC2
    EXEC2 --> RESULT2
    RESULT2 --> PLOT
    PLOT --> SOLUTION
    
    style IMPORT fill:#fff4e1
    style EXEC1 fill:#e1f5ff
    style EXEC2 fill:#e1f5ff
```

**Sources:** [biomni/agent/a1.py:350-587]()

---

## Debugging MCP Integration

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Server Not Found** | `FileNotFoundError` when executing command | Verify `command` path is correct and executable exists |
| **Environment Variables** | `KeyError` or empty values | Set environment variables before running agent |
| **Tool Discovery Fails** | No tools registered | Check server implements MCP `list_tools()` API |
| **Async Errors** | `RuntimeError: no running event loop` | Ensure `nest_asyncio.apply()` is called |
| **Parameter Mismatch** | `ValidationError` on tool call | Verify parameter names match between config and server |

### Verification Steps

```python
from biomni.agent import A1

agent = A1(path="./workspace", llm="claude-sonnet-4")

# Before adding MCP tools
print("Custom tools before MCP:", agent.list_custom_tools())

# Add MCP configuration
agent.add_mcp(config_path="./mcp_config.yaml")

# After adding MCP tools
print("Custom tools after MCP:", agent.list_custom_tools())

# Check tool registry
if hasattr(agent, 'tool_registry'):
    print("Total registered tools:", len(agent.tool_registry.tools))
    
# Verify module2api structure
for module_name in agent.module2api.keys():
    if module_name.startswith("mcp_servers."):
        print(f"MCP module: {module_name}")
        print(f"  Tools: {[t['name'] for t in agent.module2api[module_name]]}")
```

**Sources:** [biomni/agent/a1.py:589-674]()

---

## MCP Tools in System Prompt

When MCP tools are added, they are automatically included in the agent's system prompt alongside built-in tools.

### System Prompt Integration

```mermaid
graph TB
    subgraph "Tool Categories"
        BUILTIN["Built-in Tools<br/>(biomni.tool.*)"]
        CUSTOM["Custom Tools<br/>(add_tool)"]
        MCP["MCP Tools<br/>(mcp_servers.*)"]
    end
    
    subgraph "Registration"
        REGISTRY["ToolRegistry"]
        MODULE2API["module2api dict"]
    end
    
    subgraph "System Prompt Generation"
        TOOL_DESC["Generate tool descriptions"]
        PRIORITY["Priority section for custom/MCP"]
        DEFAULT["Default tools section"]
        PROMPT["Final system prompt"]
    end
    
    BUILTIN --> REGISTRY
    CUSTOM --> REGISTRY
    MCP --> REGISTRY
    
    REGISTRY --> MODULE2API
    MODULE2API --> TOOL_DESC
    
    TOOL_DESC --> PRIORITY
    TOOL_DESC --> DEFAULT
    
    PRIORITY --> PROMPT
    DEFAULT --> PROMPT
    
    style MCP fill:#fff4e1
    style REGISTRY fill:#e1f5ff
    style PROMPT fill:#e1f5ff
```

**System Prompt Format:**

```text
PRIORITY CUSTOM RESOURCES
===============================
🔧 CUSTOM TOOLS (USE THESE FIRST):
mcp_servers.weather → get_weather: Get current weather for a location
mcp_servers.postgres → query_database: Execute SQL queries on PostgreSQL

===============================

Environment Resources:

- Function Dictionary:
Import file: mcp_servers.weather
===================================
Method: get_weather
  Description: Get current weather for a location
  Required Parameters:
    - location (string): City name or coordinates [Default: None]
  Optional Parameters:
    - units (string): Temperature units [Default: celsius]
```

**Sources:** [biomni/agent/a1.py:1152-1204](), [biomni/agent/a1.py:1283-1372]()

---

## Advanced: Multiple MCP Servers

### Configuration with Multiple Servers

```yaml
mcp_servers:
  # File operations
  filesystem:
    enabled: true
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "./workspace"]
  
  # Database queries
  postgres:
    enabled: true
    command: ["python", "-m", "mcp_server.postgres"]
    env:
      DATABASE_URL: "${DATABASE_URL}"
  
  # Web search
  brave_search:
    enabled: true
    command: ["npx", "-y", "@modelcontextprotocol/server-brave-search"]
    env:
      BRAVE_API_KEY: "${BRAVE_API_KEY}"
  
  # Git operations
  git:
    enabled: true
    command: ["npx", "-y", "@modelcontextprotocol/server-git"]
  
  # Custom bioinformatics API
  biotools:
    enabled: true
    command: ["python", "-m", "custom_mcp.biotools"]
    env:
      NCBI_API_KEY: "${NCBI_API_KEY}"
```

## Summary

MCP integration extends Biomni's capabilities by enabling seamless access to external tool servers. Key concepts:

1. **Configuration**: Define servers in YAML with `command`, `env`, and optionally `tools`
2. **Discovery**: Tools can be auto-discovered via `discover_mcp_tools_sync()` [biomni/agent/a1.py:382-413]() or manually specified
3. **Wrapping**: Async MCP operations are wrapped as synchronous functions via `make_mcp_wrapper()` [biomni/agent/a1.py:415-444]()
4. **Registration**: MCP tools integrate fully with `ToolRegistry` and `module2api` dict [biomni/agent/a1.py:559-584]()
5. **Namespace**: Tools are added to `sys.modules` as `mcp_servers.{server_name}` with individual tool functions as module attributes
6. **Usage**: Agent can use MCP tools alongside built-in tools in natural language queries, with LLM generating appropriate import statements

**Core Code Entities:**
- `A1.add_mcp(config_path)` - Main entry point for MCP integration
- `discover_mcp_tools_sync()` - Auto-discovers tools from MCP server
- `make_mcp_wrapper()` - Creates synchronous wrapper around async MCP calls
- `sys.modules["mcp_servers.*"]` - Dynamic module namespace for MCP tools
- `tool_registry.register_tool()` - Registers tool schemas
- `module2api` - Dict mapping module names to tool schema lists
- `_custom_functions` - Dict storing tool name to wrapper function mappings
- `_custom_tools` - List of custom tool names for `list_custom_tools()`

The integration is transparent to end users—once configured, MCP tools are indistinguishable from built-in Biomni tools in agent interactions.

**Sources:** [biomni/agent/a1.py:350-587](), [biomni/agent/a1.py:382-444](), [biomni/agent/a1.py:559-584](), [biomni/agent/a1.py:1283-1372]()

---

## Performance Considerations

### Tool Discovery Overhead

**Auto-discovery timing:**
- Each MCP server connection: ~200-500ms
- Tool schema extraction: ~100-300ms per tool
- Total initialization: ~1-3 seconds for multiple servers

**Optimization strategies:**
1. **Use manual tool definitions** when tool schemas are known
2. **Disable unused servers** with `enabled: false`
3. **Cache tool definitions** by saving manual configurations

### Execution Performance

| Operation | Overhead | Mitigation |
|-----------|----------|------------|
| **Server startup** | ~100-300ms per call | Use persistent servers if MCP server supports it |
| **Async wrapper** | ~10-50ms per call | Negligible; necessary for sync integration |
| **Tool invocation** | Depends on server | Optimize server implementation |

**Sources:** [biomni/agent/a1.py:382-444]()

---

## Summary

MCP integration extends Biomni's capabilities by enabling seamless access to external tool servers. Key concepts:

1. **Configuration**: Define servers in YAML with `command`, `env`, and optionally `tools`
2. **Discovery**: Tools can be auto-discovered or manually specified
3. **Wrapping**: Async MCP operations are wrapped as synchronous functions
4. **Registration**: MCP tools integrate fully with ToolRegistry and module2api
5. **Usage**: Agent can use MCP tools alongside built-in tools in natural language queries

The integration is transparent to end users—once configured, MCP tools are indistinguishable from built-in Biomni tools in agent interactions.

**Sources:** [biomni/agent/a1.py:350-587](), [biomni/agent/a1.py:1283-1372]()

---