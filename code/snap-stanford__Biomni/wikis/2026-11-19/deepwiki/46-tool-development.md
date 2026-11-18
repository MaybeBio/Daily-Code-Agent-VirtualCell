# Tool Development

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.pre-commit-config.yaml](.pre-commit-config.yaml)
- [CONTRIBUTION.md](CONTRIBUTION.md)
- [biomni/agent/a1.py](biomni/agent/a1.py)
- [biomni/agent/env_collection.py](biomni/agent/env_collection.py)
- [biomni/agent/qa_llm.py](biomni/agent/qa_llm.py)
- [biomni/biorxiv_scripts/extract_biorxiv_tasks.py](biomni/biorxiv_scripts/extract_biorxiv_tasks.py)
- [biomni/biorxiv_scripts/process_all_subjects.py](biomni/biorxiv_scripts/process_all_subjects.py)
- [biomni/tool/support_tools.py](biomni/tool/support_tools.py)
- [biomni/tool/tool_description/support_tools.py](biomni/tool/tool_description/support_tools.py)
- [biomni/utils.py](biomni/utils.py)

</details>



This document provides a comprehensive guide for developing new tools to extend Biomni's capabilities. It covers both static tool development (adding tools to the codebase) and dynamic tool addition (registering tools at runtime).

For information about using existing tools, see page 5. For details on external tool integration via MCP, see page 4.5. For contributing guidelines, see page 8.2.

## Tool Development Overview

Biomni supports two approaches to tool development:

1. **Static Tool Development**: Adding tools to the codebase in `biomni/tool/` with descriptions in `biomni/tool/tool_description/`
2. **Dynamic Tool Addition**: Registering tools at runtime using `agent.add_tool()` or `agent.add_mcp()`

Both approaches use automatic schema generation and integration with the Tool Registry for retrieval-augmented planning.

### Static Tool Development Workflow

```mermaid
graph TD
    IMPLEMENT["Implement Function<br/>biomni/tool/{domain}.py"] --> TEST["Test Locally"]
    TEST --> DESCRIBE["Create Tool Description<br/>tool_description/{domain}.py"]
    DESCRIBE --> CATEGORY["Choose Domain Category"]
    CATEGORY --> VALIDATE["Create Test Prompt"]
    VALIDATE --> PR["Submit Pull Request"]
    
    subgraph "Domain Categories"
        DATABASE["database.py"]
        GENOMICS["genomics.py"]
        PHARMACOLOGY["pharmacology.py"]
        CANCER["cancer_biology.py"]
        SYSTEMS["systems_biology.py"]
        MOLECULAR["molecular_biology.py"]
        SUPPORT["support_tools.py"]
        PROTOCOLS["protocols.py"]
        LAB["lab_automation.py"]
    end
    
    CATEGORY --> DATABASE
    CATEGORY --> GENOMICS
    CATEGORY --> PHARMACOLOGY
    CATEGORY --> CANCER
    CATEGORY --> SYSTEMS
    CATEGORY --> MOLECULAR
    CATEGORY --> SUPPORT
    CATEGORY --> PROTOCOLS
    CATEGORY --> LAB
```

**Static Tool Development Process from Implementation to Integration**

Sources: [CONTRIBUTION.md:14-35](), [biomni/utils.py:845-876]()

### Dynamic Tool Addition Workflow

```mermaid
graph TD
    USER["User Code"] --> ADD_TOOL["agent.add_tool(function)"]
    ADD_TOOL --> INSPECT["inspect.getsource(api)"]
    INSPECT --> SCHEMA_GEN["function_to_api_schema()"]
    SCHEMA_GEN --> LLM["LLM generates schema"]
    LLM --> REGISTER["tool_registry.register_tool()"]
    REGISTER --> MODULE2API["Add to module2api"]
    MODULE2API --> NAMESPACE["Inject into execution namespace"]
    NAMESPACE --> CONFIGURE["agent.configure()"]
    
    subgraph "Schema Fields"
        NAME["name: function_name"]
        DESC["description: str"]
        REQ_PARAMS["required_parameters: list"]
        OPT_PARAMS["optional_parameters: list"]
        MODULE["module: str"]
    end
    
    SCHEMA_GEN --> NAME
    SCHEMA_GEN --> DESC
    SCHEMA_GEN --> REQ_PARAMS
    SCHEMA_GEN --> OPT_PARAMS
    SCHEMA_GEN --> MODULE
```

**Dynamic Tool Addition via add_tool() Method**

Sources: [biomni/agent/a1.py:225-348](), [biomni/utils.py:251-291]()

## Creating New Tools

### Simple Tool Implementation Pattern

Tools are implemented as Python functions with clear type hints and docstrings. The implementation pattern follows these principles:

| Component | Purpose | Example |
|-----------|---------|---------|
| Type Hints | Define parameter and return types | `def tool(param: str) -> str:` |
| Docstring | Explain functionality and usage | Triple-quoted documentation |
| Error Handling | Graceful failure with informative messages | Try-except blocks |
| Return Format | Consistent string output | Success message or error details |

```mermaid
graph TD
    FUNCTION["Python Function"] --> SIGNATURE["Type-Hinted Signature"]
    SIGNATURE --> DOCSTRING["Comprehensive Docstring"]
    DOCSTRING --> VALIDATION["Input Validation"]
    VALIDATION --> LOGIC["Core Implementation"]
    LOGIC --> ERROR_HANDLER["try-except Block"]
    ERROR_HANDLER --> RETURN["String Return Value"]
    
    subgraph "Example: run_python_repl Flow"
        GET_NAMESPACE["Get _persistent_namespace"]
        EXEC_CODE["exec(code, namespace)"]
        CAPTURE_OUT["Capture stdout"]
        CAPTURE_ERR["Capture stderr"]
    end
    
    LOGIC --> GET_NAMESPACE
    GET_NAMESPACE --> EXEC_CODE
    EXEC_CODE --> CAPTURE_OUT
    EXEC_CODE --> CAPTURE_ERR
    CAPTURE_OUT --> RETURN
    CAPTURE_ERR --> RETURN
```

**Tool Implementation Pattern with Type Safety and Error Handling**

Sources: [biomni/tool/support_tools.py:8-68]()

### Tool Function Requirements

All tool functions must adhere to these requirements:

```python
# Required structure
def tool_name(
    required_param: type,
    optional_param: type = default_value
) -> str:
    """Brief description.
    
    Args:
        required_param: Description
        optional_param: Description with default
    
    Returns:
        Success message or error details
    """
    try:
        # Implementation
        result = perform_operation(required_param, optional_param)
        return f"Success: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
```

The function should return a string that is either:
- A success message with results
- An error message with diagnostic information

Sources: [biomni/tool/support_tools.py:1-100]()

### Database Integration Tools

Database tools follow a standardized pattern using helper functions for consistent error handling and LLM integration:

```mermaid
graph TD
    PROMPT["Natural Language Prompt"] --> SCHEMA["Load API Schema<br/>schema_db/*.pkl"]
    SCHEMA --> LLM_QUERY["_query_llm_for_api()"]
    LLM_QUERY --> ENDPOINT["Generate API Endpoint"]
    ENDPOINT --> REST_API["_query_rest_api()"]
    REST_API --> FORMAT["_format_query_results()"]
    FORMAT --> RESPONSE["Structured Response"]
    
    subgraph "Example: query_uniprot"
        UNIPROT_SCHEMA["uniprot.pkl"]
        UNIPROT_API["UniProt REST API"]
        PROTEIN_DATA["Protein Information"]
    end
    
    SCHEMA --> UNIPROT_SCHEMA
    REST_API --> UNIPROT_API
    RESPONSE --> PROTEIN_DATA
    
    subgraph "Helper Functions"
        HELPER1["_query_llm_for_api()<br/>Lines 36-109"]
        HELPER2["_query_rest_api()<br/>Lines 111-207"]
        HELPER3["_format_query_results()<br/>Lines 327-456"]
    end
    
    LLM_QUERY --> HELPER1
    REST_API --> HELPER2
    FORMAT --> HELPER3
```

**Database Tool Integration Pattern with Helper Functions**

Sources: [biomni/tool/database.py:36-109](), [biomni/tool/database.py:111-207](), [biomni/tool/database.py:458-551]()

## Tool Description Schema

Every tool requires a corresponding description in the `tool_description` module. The schema follows a standardized format that enables automatic tool discovery and validation.

### Description Structure

| Field | Type | Purpose | Example |
|-------|------|---------|---------|
| `name` | string | Function name | `"query_uniprot"` |
| `description` | string | What the tool does | `"Query the UniProt REST API..."` |
| `required_parameters` | array | Must-have parameters | `[{"name": "prompt", "type": "str", ...}]` |
| `optional_parameters` | array | Optional parameters | `[{"name": "max_results", "type": "int", ...}]` |

### Parameter Schema Format

```mermaid
graph TD
    PARAMETER["Parameter Definition"] --> NAME["name: str"]
    PARAMETER --> TYPE["type: str"]
    PARAMETER --> DESC["description: str"]
    PARAMETER --> DEFAULT["default: Any"]
    
    subgraph "Parameter Types"
        SIMPLE["str, int, bool"]
        COMPLEX["List[str], dict"]
        SPECIAL["pandas.DataFrame"]
    end
    
    TYPE --> SIMPLE
    TYPE --> COMPLEX
    TYPE --> SPECIAL
    
    subgraph "Example: query_uniprot prompt"
        PROMPT_NAME["'prompt'"]
        PROMPT_TYPE["'str'"]
        PROMPT_DESC["'Natural language query...'"]
        PROMPT_DEFAULT["None"]
    end
    
    NAME --> PROMPT_NAME
    TYPE --> PROMPT_TYPE
    DESC --> PROMPT_DESC
    DEFAULT --> PROMPT_DEFAULT
```

**Parameter Schema Structure with Examples**

Sources: [biomni/tool/tool_description/database.py:17-25](), [biomni/tool/tool_description/support_tools.py:6-13]()

### Auto-Generation of Tool Schemas

Tool descriptions can be automatically generated using the `function_to_api_schema` utility, which uses an LLM to analyze function signatures and docstrings:

```mermaid
graph TD
    FUNC_CODE["Function Source Code"] --> PROMPT["Schema Generation Prompt"]
    PROMPT --> LLM_CALL["llm.with_structured_output(api_schema)"]
    LLM_CALL --> STRUCTURED["Structured Output"]
    STRUCTURED --> PARSE["ast.literal_eval()"]
    PARSE --> RETRY{"Parse Success?"}
    RETRY -->|No| RETRY_LOOP["Retry up to 7 times"]
    RETRY_LOOP --> LLM_CALL
    RETRY -->|Yes| SCHEMA["API Schema Dict"]
    
    subgraph "Schema Components"
        NAME["name: str"]
        DESC["description: str"]
        REQ["required_parameters: list"]
        OPT["optional_parameters: list"]
    end
    
    SCHEMA --> NAME
    SCHEMA --> DESC
    SCHEMA --> REQ
    SCHEMA --> OPT
```

**Automatic Schema Generation Using LLM**

Usage example:

```python
from biomni.utils import function_to_api_schema
from biomni.llm import get_llm
import inspect

# Get function source code
function_code = inspect.getsource(my_function)

# Generate schema
llm = get_llm('claude-sonnet-4-20250514')
schema = function_to_api_schema(function_code, llm)

# Schema contains: name, description, required_parameters, optional_parameters
```

The function uses a carefully crafted prompt that instructs the LLM to:
- Follow the function signature strictly
- Set None for parameters without defaults (not null)
- Use capitalized True/False for booleans
- Keep descriptions clear and succinct
- Avoid creating fake optional parameters

Sources: [biomni/utils.py:251-291](), [CONTRIBUTION.md:26-32]()

## Database Integration Pattern

Database tools in Biomni use a three-layer architecture to translate natural language queries into API calls:

```mermaid
graph TD
    NL_QUERY["Natural Language Query"] --> LOAD_SCHEMA["Load API Schema<br/>from schema_db/*.pkl"]
    LOAD_SCHEMA --> TEMPLATE["System Prompt Template"]
    TEMPLATE --> LLM_QUERY["_query_llm_for_api()"]
    LLM_QUERY --> EXTRACT_JSON["Extract JSON response"]
    EXTRACT_JSON --> REST_API["_query_rest_api()"]
    REST_API --> HTTP_REQUEST["requests.request()"]
    HTTP_REQUEST --> FORMAT["_format_query_results()"]
    FORMAT --> RETURN["Formatted Result String"]
    
    subgraph "Helper Functions"
        H1["_query_llm_for_api()<br/>Lines 36-109"]
        H2["_query_rest_api()<br/>Lines 111-207"]
        H3["_format_query_results()<br/>Lines 327-456"]
    end
    
    LLM_QUERY --> H1
    REST_API --> H2
    FORMAT --> H3
    
    subgraph "Error Handling Flow"
        HTTP_ERROR["RequestException"]
        EXTRACT_ERR["Extract error info"]
        TRY_JSON["Try response.json()"]
        FALLBACK_TEXT["Fallback to text"]
        STRUCT_ERR["Structured error dict"]
    end
    
    HTTP_REQUEST --> HTTP_ERROR
    HTTP_ERROR --> EXTRACT_ERR
    EXTRACT_ERR --> TRY_JSON
    TRY_JSON --> FALLBACK_TEXT
    FALLBACK_TEXT --> STRUCT_ERR
```

**Three-Layer Database Tool Architecture**

### LLM-Powered Natural Language to API Translation

The `_query_llm_for_api` function converts natural language to structured API parameters:

| Step | Action | Implementation |
|------|--------|----------------|
| 1. Load Schema | Read API schema from pickle file | `load_pickle(schema_path)` |
| 2. Build Prompt | Insert schema into template | `template.format(schema=schema, prompt=prompt)` |
| 3. Invoke LLM | Get structured response | `default_config.llm.invoke(system_prompt)` |
| 4. Parse JSON | Extract API parameters | `json.loads()` with retry logic |
| 5. Return | Provide API endpoint and params | Dictionary with endpoint, method, params |

The system prompt template instructs the LLM to:
- Analyze the natural language query
- Map it to appropriate API endpoints
- Extract required and optional parameters
- Return valid JSON matching the schema

Sources: [biomni/tool/database.py:36-109]()

### Database Helper Functions Reference

Three core helper functions provide consistent behavior across all database tools:

| Function | Purpose | Key Parameters | Return Type |
|----------|---------|----------------|-------------|
| `_query_llm_for_api()` | Convert NL to API call | `prompt`, `schema_path`, `template` | `dict` with endpoint info |
| `_query_rest_api()` | Execute HTTP requests | `endpoint`, `method`, `params`, `headers` | API response or error dict |
| `_format_query_results()` | Reduce response size | `result`, `max_results`, `include_fields` | Formatted string |

### REST API Execution and Error Handling

The `_query_rest_api` function handles all HTTP interactions:

```mermaid
graph TD
    PARAMS["API Parameters"] --> BUILD_REQ["Build Request"]
    BUILD_REQ --> METHOD_CHECK{"HTTP Method"}
    METHOD_CHECK -->|GET| GET_REQ["requests.get()"]
    METHOD_CHECK -->|POST| POST_REQ["requests.post()"]
    METHOD_CHECK -->|PUT| PUT_REQ["requests.put()"]
    METHOD_CHECK -->|DELETE| DEL_REQ["requests.delete()"]
    
    GET_REQ --> CHECK_STATUS["response.raise_for_status()"]
    POST_REQ --> CHECK_STATUS
    PUT_REQ --> CHECK_STATUS
    DEL_REQ --> CHECK_STATUS
    
    CHECK_STATUS --> SUCCESS{"Success?"}
    SUCCESS -->|Yes| RETURN_RESP["Return response"]
    SUCCESS -->|No| CATCH_ERR["Catch RequestException"]
    
    CATCH_ERR --> TRY_JSON["Try response.json()"]
    TRY_JSON --> HAS_JSON{"Valid JSON?"}
    HAS_JSON -->|Yes| STRUCT_ERR["Structured error dict"]
    HAS_JSON -->|No| TEXT_ERR["Use response.text"]
    
    subgraph "Error Dictionary Structure"
        ERR_MSG["error: Error message"]
        QUERY["query_info: API endpoint"]
        RESP_TEXT["response_text: Raw response"]
    end
    
    STRUCT_ERR --> ERR_MSG
    STRUCT_ERR --> QUERY
    TEXT_ERR --> RESP_TEXT
```

**REST API Execution with Comprehensive Error Handling**

The error handling pattern ensures:
- All HTTP exceptions are caught and structured
- API error responses are parsed when available
- Diagnostic information is preserved for debugging
- Users receive actionable error messages

Sources: [biomni/tool/database.py:111-207]()

### Result Formatting

The `_format_query_results` function reduces response sizes for LLM context management:

```python
# Truncates large responses while preserving structure
# Handles both list and dict responses
# Includes continuation indicators ("...")
```

This prevents context overflow when APIs return large datasets.

Sources: [biomni/tool/database.py:327-456]()

## Development Workflow and Quality Control

### Pre-commit Hooks and Code Quality

The project uses automated code quality checks to maintain consistency:

```mermaid
graph TD
    COMMIT["Git Commit"] --> PRECOMMIT["Pre-commit Hooks"]
    PRECOMMIT --> BIOME["Biome Format<br/>JS/JSON formatting"]
    PRECOMMIT --> RUFF["Ruff Check + Format<br/>Python linting"]
    PRECOMMIT --> HOOKS["General Hooks"]
    
    HOOKS --> PRIVATE_KEY["Detect Private Keys"]
    HOOKS --> AST_CHECK["Python AST Check"]
    HOOKS --> EOF_FIX["End of File Fixer"]
    HOOKS --> LINE_END["Mixed Line Ending Fix"]
    HOOKS --> WHITESPACE["Trailing Whitespace"]
    HOOKS --> CASE_CONFLICT["Case Conflict Check"]
    HOOKS --> LARGE_FILES["Large Files Check"]
    HOOKS --> YAML_CHECK["YAML/TOML Check"]
    HOOKS --> MERGE_CHECK["Merge Conflict Check"]
    HOOKS --> BRANCH_PROTECT["No Commit to Main"]
    
    subgraph "Ruff Configuration"
        FIX["--fix"]
        EXIT_NON_ZERO["--exit-non-zero-on-fix"]
        UNSAFE["--unsafe-fixes"]
    end
    
    RUFF --> FIX
    RUFF --> EXIT_NON_ZERO
    RUFF --> UNSAFE
```

**Automated Code Quality Pipeline**

Sources: [.pre-commit-config.yaml:8-34]()

### Testing Requirements

Before submitting tools, developers must create test prompts to validate functionality:

| Test Component | Purpose | Example |
|----------------|---------|---------|
| Function Test | Verify implementation | Unit test with sample inputs |
| Agent Test | Verify A1 integration | Test prompt using the tool |
| Error Test | Verify error handling | Invalid inputs and edge cases |

### Software Dependencies

For tools requiring additional software, create installation scripts:

```mermaid
graph TD
    NEW_TOOL["New Tool Requiring Software"] --> TEST_LOCAL["Test Locally<br/>No Conflicts"]
    TEST_LOCAL --> CREATE_SCRIPT["Create Installation Script"]
    CREATE_SCRIPT --> APPEND_SCRIPT["Append to new_software_VERSION.sh"]
    APPEND_SCRIPT --> UPDATE_ENV["Add to library_content_dict"]
    UPDATE_ENV --> SCREENSHOT["Screenshot: No Conflicts"]
    SCREENSHOT --> SUBMIT_PR["Submit PR with Evidence"]
    
    subgraph "Installation Script Location"
        SCRIPT_PATH["biomni_env/new_software_\{VERSION\}.sh"]
    end
    
    APPEND_SCRIPT --> SCRIPT_PATH
    
    subgraph "Environment Description"
        ENV_DESC["biomni/env_desc.py"]
        LIB_DICT["library_content_dict"]
    end
    
    UPDATE_ENV --> ENV_DESC
    ENV_DESC --> LIB_DICT
```

**Software Dependency Management Workflow**

Sources: [CONTRIBUTION.md:56-64]()

## Code Execution Tools

Biomni supports execution across multiple environments with standardized interfaces and timeout protection.

### Multi-Language Execution Architecture

```mermaid
graph TD
    A1_AGENT["A1 Agent Execution Node"] --> DETECT["Language Detection"]
    DETECT --> PYTHON["Python REPL"]
    DETECT --> R_EXEC["R Code Execution"]
    DETECT --> BASH_EXEC["Bash Script Execution"]
    
    PYTHON --> PERSIST_NS["_persistent_namespace"]
    PERSIST_NS --> EXEC_PY["exec(code, namespace)"]
    EXEC_PY --> CAPTURE["Capture stdout/stderr"]
    
    R_EXEC --> R_TEMP["tempfile.NamedTemporaryFile(.R)"]
    R_TEMP --> RSCRIPT["subprocess.run(['Rscript', file])"]
    RSCRIPT --> R_CLEANUP["os.unlink(temp_file)"]
    
    BASH_EXEC --> BASH_TEMP["tempfile.NamedTemporaryFile(.sh)"]
    BASH_TEMP --> ADD_SHEBANG["Add #!/bin/bash if missing"]
    ADD_SHEBANG --> SET_E["Add set -e"]
    SET_E --> CHMOD["os.chmod(file, 0o755)"]
    CHMOD --> SHELL_RUN["subprocess.run([file], shell=True)"]
    SHELL_RUN --> BASH_CLEANUP["os.unlink(temp_file)"]
    
    subgraph "Timeout Protection"
        TIMEOUT_FUNC["run_with_timeout()"]
        THREAD["threading.Thread"]
        QUEUE["queue.Queue for result"]
        TIMEOUT_CHECK["thread.join(timeout)"]
    end
    
    EXEC_PY --> TIMEOUT_FUNC
    RSCRIPT --> TIMEOUT_FUNC
    SHELL_RUN --> TIMEOUT_FUNC
```

**Multi-Language Code Execution with Timeout Protection**

### Execution Environment Details

| Environment | Function | Temporary File | Cleanup | Timeout Support |
|-------------|----------|----------------|---------|-----------------|
| Python | `run_python_repl` | No (uses namespace) | N/A | Yes |
| R | `run_r_code` | `.R` file | `os.unlink()` | No (direct subprocess) |
| Bash | `run_bash_script` | `.sh` file | `os.unlink()` | No (direct subprocess) |
| CLI | `run_cli_command` | No | N/A | No |

### Python REPL with Persistent State

The Python execution environment maintains state across code blocks:

```python
# Variables and imports persist between executions
# Located in global _persistent_namespace dictionary
# Allows multi-step workflows without re-initialization
```

The A1 agent injects custom tools into this namespace before execution, making them available to generated code.

Sources: [biomni/tool/support_tools.py:1-100]()

### Timeout Implementation

The `run_with_timeout` function uses threading to prevent infinite execution:

```mermaid
graph TD
    FUNC_CALL["run_with_timeout(func, timeout=600)"] --> QUEUE["Create queue.Queue()"]
    QUEUE --> THREAD_FUNC["Define thread_func wrapper"]
    THREAD_FUNC --> START_THREAD["Start daemon thread"]
    START_THREAD --> JOIN["thread.join(timeout)"]
    JOIN --> ALIVE_CHECK{"thread.is_alive()?"}
    ALIVE_CHECK -->|No| GET_RESULT["result_queue.get()"]
    ALIVE_CHECK -->|Yes| TIMEOUT_MSG["Return timeout error"]
    ALIVE_CHECK -->|Yes| TRY_KILL["Try to terminate thread"]
    
    GET_RESULT --> RETURN_SUCCESS["Return result"]
    
    subgraph "Thread Termination (Best Effort)"
        GET_ID["thread.ident"]
        ASYNC_EXC["PyThreadState_SetAsyncExc"]
        RAISE_EXIT["Raise SystemExit in thread"]
    end
    
    TRY_KILL --> GET_ID
    GET_ID --> ASYNC_EXC
    ASYNC_EXC --> RAISE_EXIT
```

**Threading-Based Timeout Protection**

Note: Thread termination in Python is not 100% reliable. The function uses daemon threads and attempts `PyThreadState_SetAsyncExc` as a best-effort approach.

Sources: [biomni/utils.py:183-243]()

### Bash Script Execution

The `run_bash_script` function includes automatic safety features:

| Feature | Purpose | Implementation |
|---------|---------|----------------|
| Shebang injection | Ensure bash interpreter | Prepends `#!/bin/bash` if missing |
| Error propagation | Exit on first error | Adds `set -e` if not present |
| Executable permissions | Allow execution | `os.chmod(file, 0o755)` |
| Environment inheritance | Preserve context | `env=os.environ.copy()` |
| Working directory | Maintain path context | `cwd=os.getcwd()` |

Sources: [biomni/utils.py:56-145]()

## Tool Registry Architecture

### Module-Based Organization

The tool system organizes functionality by biological domain:

```mermaid
graph TD
    REGISTRY["Tool Registry"] --> FIELDS["Domain Fields"]
    
    FIELDS --> LIT["literature"]
    FIELDS --> BIO["biochemistry"]
    FIELDS --> CANCER["cancer_biology"]
    FIELDS --> GENETICS["genetics"]
    FIELDS --> GENOMICS["genomics"]
    FIELDS --> IMMUNO["immunology"]
    FIELDS --> PHARMA["pharmacology"]
    FIELDS --> SUPPORT["support_tools"]
    FIELDS --> DB["database"]
    
    subgraph "Module Structure"
        MODULE["biomni.tool.{field}"]
        DESC["tool_description.{field}"]
        API["description variable"]
    end
    
    FIELDS --> MODULE
    MODULE --> DESC
    DESC --> API
```

**Domain-Based Tool Organization**

Sources: [biomni/utils.py:845-873]()

This architecture enables the A1 agent to dynamically discover and utilize tools across the entire biomedical research spectrum while maintaining type safety and user-friendly error handling.

---