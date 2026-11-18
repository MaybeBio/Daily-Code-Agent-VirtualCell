# PDF Report Generation

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [README.md](README.md)
- [biomni/eval/__init__.py](biomni/eval/__init__.py)
- [biomni/eval/biomni_eval1.py](biomni/eval/biomni_eval1.py)
- [biomni/version.py](biomni/version.py)

</details>



## Purpose and Scope

This document describes the PDF report generation system in Biomni, which converts agent conversation histories into formatted PDF documents. The system captures complete conversation traces including reasoning steps, code execution blocks, observations, plots, and solutions. For information about the agent execution workflow itself, see [Execution Pipeline](#2.2). For details on plot generation within tools, see [Tool Reference](#5).

**Sources:** [biomni/agent/a1.py:2055-2132]()

---

## Overview

The PDF report generation system provides a complete audit trail of agent interactions by converting conversation history into professionally formatted PDF documents. The system tracks execution results, captures generated plots as base64-encoded images, and applies structured styling to different content types.

### Key Components

| Component | Primary Function | Key Code Entity |
|-----------|-----------------|-----------------|
| Save API | Entry point for PDF generation | `A1.save_conversation_history()` |
| Markdown Generator | Converts conversation to markdown | `A1._generate_markdown_content()` |
| PDF Converter | Converts markdown to PDF | `convert_markdown_to_pdf()` |
| Execution Tracker | Stores code execution results | `A1._execution_results` |
| Content Formatter | Formats different content types | `format_execute_tags_in_content()` |
| CSS Styler | Applies styling to PDF | `get_pdf_css_content()` |

**Sources:** [biomni/agent/a1.py:2055-2132](), [biomni/utils.py:1922-2366]()

---

## PDF Generation Workflow

```mermaid
graph TD
    A1["User calls<br/>agent.save_conversation_history()"] --> B1{"Check<br/>save_pdf"}
    B1 -->|False| END1["Exit<br/>(no file saved)"]
    B1 -->|True| C1["Create directory<br/>if needed"]
    
    C1 --> D1["Call<br/>_generate_markdown_content()"]
    
    D1 --> E1["Get messages from<br/>_conversation_state or log"]
    E1 --> F1["Process each message<br/>via _process_message()"]
    
    F1 --> G1{"Message<br/>type?"}
    G1 -->|"HumanMessage"| H1["Format as<br/>user query"]
    G1 -->|"AIMessage with<br/>&lt;execute&gt;"| H2["Format code block<br/>+ Find matching plots<br/>from _execution_results"]
    G1 -->|"AIMessage with<br/>&lt;observation&gt;"| H3["Format as<br/>terminal output"]
    G1 -->|"AIMessage with<br/>&lt;solution&gt;"| H4["Format as<br/>summary block"]
    
    H1 --> I1["Append to<br/>markdown_content"]
    H2 --> I1
    H3 --> I1
    H4 --> I1
    
    I1 --> J1{"More<br/>messages?"}
    J1 -->|Yes| F1
    J1 -->|No| K1["Create temp<br/>markdown file"]
    
    K1 --> L1["Call<br/>convert_markdown_to_pdf()"]
    
    L1 --> M1{"weasyprint<br/>available?"}
    M1 -->|Yes| N1["Use weasyprint.HTML<br/>with custom CSS"]
    M1 -->|No| M2{"markdown2pdf<br/>available?"}
    
    M2 -->|Yes| N2["Use markdown2pdf"]
    M2 -->|No| M3{"pandoc<br/>available?"}
    
    M3 -->|Yes| N3["Use subprocess<br/>with pandoc"]
    M3 -->|No| ERROR["Raise ImportError<br/>(no converter available)"]
    
    N1 --> O1["Generate PDF<br/>with 60s timeout"]
    N2 --> O1
    N3 --> O1
    
    O1 --> P1["Clean up temp<br/>markdown file"]
    P1 --> END2["Print success<br/>message"]
    
    style A1 fill:#e1f5ff
    style D1 fill:#fff4e1
    style L1 fill:#f0e1ff
    style O1 fill:#e1ffe1
```

**Sources:** [biomni/agent/a1.py:2055-2132](), [biomni/agent/a1.py:2133-2166](), [biomni/utils.py:1922-2000]()

---

## Core API: save_conversation_history()

### Method Signature

The `save_conversation_history()` method is defined in the `A1` agent class:

```python
def save_conversation_history(
    self, 
    filepath: str, 
    include_images: bool = True, 
    save_pdf: bool = True
) -> None
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | Required | Path where to save the PDF file (without extension). If path doesn't end with `.pdf`, it will be appended automatically. |
| `include_images` | `bool` | `True` | Whether to include captured plots and images in the output. |
| `save_pdf` | `bool` | `True` | Whether to save as PDF. If `False`, no file is saved. |

### Return Value

Returns `None`. Prints status messages to console including file path and step count.

### Timeout Protection

The method includes a 60-second timeout for PDF generation using `signal.SIGALRM` to prevent hanging on large documents.

**Sources:** [biomni/agent/a1.py:2055-2132]()

---

## Message Processing Pipeline

```mermaid
graph LR
    A["_conversation_state<br/>or self.log"] --> B["_get_messages_for_processing()"]
    
    B --> C1{"Source?"}
    C1 -->|"conversation_state"| D1["_normalize_conversation_state_messages()"]
    C1 -->|"log"| D2["_normalize_log_messages()"]
    
    D1 --> E["Normalized message list<br/>(content, type, original)"]
    D2 --> E
    
    E --> F["Loop: _process_message()<br/>for each message"]
    
    F --> G["clean_message_content()<br/>(remove ANSI codes)"]
    
    G --> H{"should_skip_message()?"}
    H -->|Yes| F
    H -->|No| I{"Message<br/>type?"}
    
    I -->|"human"| J1["Increment step_number<br/>Add ## Step X"]
    I -->|"ai with<br/>&lt;execute&gt;"| J2["format_execute_tags_in_content()<br/>parse_tool_calls_with_modules()"]
    I -->|"ai with<br/>&lt;observation&gt;"| J3["format_observation_as_terminal()"]
    I -->|"other ai"| J4["format_lists_in_text()<br/>format_solution_tags_in_content()"]
    
    J1 --> K["Append to<br/>markdown_content"]
    J2 --> L{"has_execution_results()?"}
    J3 --> K
    J4 --> K
    
    L -->|Yes| M["find_matching_execution()"]
    L -->|No| K
    
    M --> N{"Images<br/>present?"}
    N -->|Yes| O["Append base64<br/>images to content"]
    N -->|No| K
    
    O --> K
    
    K --> P{"More<br/>messages?"}
    P -->|Yes| F
    P -->|No| Q["Return complete<br/>markdown_content"]
    
    style B fill:#e1f5ff
    style F fill:#fff4e1
    style L fill:#f0e1ff
    style Q fill:#e1ffe1
```

**Sources:** [biomni/agent/a1.py:2168-2186](), [biomni/agent/a1.py:2133-2166](), [biomni/utils.py:1021-1066]()

---

## Execution Results Tracking

### Execution Entry Structure

During code execution, the agent stores execution results with metadata in `_execution_results`:

```python
execution_entry = {
    "triggering_message": last_message,    # AI message containing <execute>
    "images": execution_plots,             # List of base64 encoded images
    "timestamp": datetime.now().isoformat()
}
```

### Plot Capture Workflow

```mermaid
graph TD
    A["Code execution<br/>in execute() node"] --> B["_clear_execution_plots()"]
    
    B --> C["_inject_custom_functions_to_repl()"]
    
    C --> D["run_with_timeout()<br/>run_python_repl(code)"]
    
    D --> E["get_captured_plots()<br/>from support_tools"]
    
    E --> F["Create execution_entry<br/>with plots + metadata"]
    
    F --> G["Append to<br/>_execution_results[]"]
    
    G --> H["Later: find_matching_execution()<br/>retrieves plots for PDF"]
    
    style D fill:#e1f5ff
    style E fill:#fff4e1
    style F fill:#f0e1ff
```

**Sources:** [biomni/agent/a1.py:1465-1547](), [biomni/agent/a1.py:1522-1543]()

---

## Content Formatting System

The system applies different formatting to different content types using specialized functions in `biomni/utils.py`:

### Content Type Formatting

| Content Type | Formatting Function | Output Style | CSS Class |
|--------------|-------------------|--------------|-----------|
| Code Execution | `format_execute_tags_in_content()` | Tool call block with syntax highlighting | `.tool-call-highlight` |
| Observations | `format_observation_as_terminal()` | Terminal-style output | `.title-text.observation` |
| Solutions | `format_solution_tags_in_content()` | Summary block | `.title-text.summary` |
| Plans | `format_lists_in_text()` | Checkbox list | `.title-text.plan` |
| Parsing Errors | `create_parsing_error_html()` | Error box | `.parsing-error-box` |

### Execute Tag Formatting Pipeline

```mermaid
graph TD
    A["Content with<br/>&lt;execute&gt;...&lt;/execute&gt;"] --> B["format_execute_tags_in_content()"]
    
    B --> C["Regex extract<br/>code_content"]
    
    C --> D["detect_code_language_and_tool()<br/>(Python/R/Bash)"]
    
    D --> E["clean_code_content()<br/>(remove #!R, #!BASH markers)"]
    
    E --> F["parse_tool_calls_with_modules()<br/>(detect imported tools)"]
    
    F --> G["create_tool_call_block()"]
    
    G --> H{"Tools<br/>detected?"}
    
    H -->|Yes| I["format_detected_tools()<br/>(e.g., 'scanpy → preprocess')"]
    H -->|No| J["format_default_tool_name()<br/>(e.g., 'Python REPL')"]
    
    I --> K["Generate HTML with<br/>.tool-call-highlight class"]
    J --> K
    
    K --> L["Return formatted<br/>HTML block"]
    
    style B fill:#e1f5ff
    style F fill:#fff4e1
    style G fill:#f0e1ff
```

**Sources:** [biomni/utils.py:1316-1360](), [biomni/utils.py:1362-1542]()

### Tool Detection Details

The `parse_tool_calls_with_modules()` function uses regex patterns to detect tool usage:

- **Import statements**: `from biomni.tool.genomics import analyze_cells`
- **Direct calls**: `result = analyze_cells(data)`
- **Module patterns**: `biomni.tool.database.query_uniprot()`

Output format: List of `(tool_name, module_name)` tuples

**Sources:** [biomni/utils.py:1156-1249]()

---

## PDF Conversion Strategies

### Conversion Library Priority

The system attempts multiple PDF conversion strategies in order:

```mermaid
graph TD
    A["convert_markdown_to_pdf()<br/>(markdown_path, pdf_path)"] --> B["Read markdown<br/>content from file"]
    
    B --> C["Try: import weasyprint"]
    
    C --> D1{"weasyprint<br/>available?"}
    
    D1 -->|Yes| E1["Convert markdown to HTML<br/>using markdown.markdown()"]
    
    E1 --> F1["Add CSS from<br/>get_pdf_css_content()"]
    
    F1 --> G1["Create HTML document<br/>with styling"]
    
    G1 --> H1["weasyprint.HTML(string=html_doc)"]
    
    H1 --> I1["html.write_pdf()<br/>with FontConfiguration<br/>and optimize_images=True"]
    
    I1 --> SUCCESS["PDF file<br/>written"]
    
    D1 -->|ImportError| D2["Try: import markdown2pdf"]
    
    D2 -->|Available| E2["markdown2pdf()<br/>direct conversion"]
    E2 --> SUCCESS
    
    D2 -->|ImportError| D3["Try: subprocess.run()<br/>with pandoc"]
    
    D3 -->|Available| E3["pandoc markdown_path<br/>-o pdf_path"]
    E3 --> SUCCESS
    
    D3 -->|Not available| ERROR["Raise ImportError:<br/>No PDF library found"]
    
    style C fill:#e1f5ff
    style H1 fill:#fff4e1
    style SUCCESS fill:#e1ffe1
    style ERROR fill:#ffe1e1
```

### Library Comparison

| Library | Priority | Advantages | Performance |
|---------|----------|------------|-------------|
| `weasyprint` | 1st (preferred) | Best layout control, custom CSS, image embedding | Slower but high quality |
| `markdown2pdf` | 2nd (fallback) | Simple API, automatic conversion | Medium speed |
| `pandoc` | 3rd (last resort) | Widely available, command-line tool | Fast but less control |

**Sources:** [biomni/utils.py:1922-2000]()

---

## CSS Styling System

### Style Architecture

The `get_pdf_css_content()` function returns comprehensive CSS for PDF generation with optimized typography and layout:

```mermaid
graph LR
    A["get_pdf_css_content()"] --> B["Base Typography<br/>(body, headings, code)"]
    A --> C["Content Blocks<br/>(.tool-call-highlight)"]
    A --> D["Title-Text Containers<br/>(.title-text)"]
    A --> E["Specialized Styles<br/>(.plan, .observation)"]
    A --> F["Error Styles<br/>(.parsing-error-box)"]
    
    B --> G["Font: sans-serif<br/>Size: 9pt<br/>Line-height: 1.4"]
    
    C --> H["Code execution blocks<br/>with syntax highlighting"]
    
    D --> I["Unified container<br/>for observations,<br/>plans, solutions"]
    
    E --> J["Soft pastel colors:<br/>Blue (plans)<br/>Purple (observations)<br/>Orange (solutions)"]
    
    F --> K["Red error boxes<br/>with monospace font"]
    
    style A fill:#e1f5ff
    style D fill:#fff4e1
    style E fill:#f0e1ff
```

### Key CSS Classes

| CSS Class | Purpose | Color Scheme | Applied To |
|-----------|---------|--------------|------------|
| `.tool-call-highlight` | Code execution blocks | Gray background | `<execute>` tags |
| `.title-text.plan` | Plan checklists | Soft blue (#e3f2fd) | Plan with checkboxes |
| `.title-text.observation` | Execution results | Soft purple (#f3e5f5) | `<observation>` tags |
| `.title-text.summary` | Final solutions | Soft orange (#fff3e0) | `<solution>` tags |
| `.parsing-error-box` | Error messages | Red (#ffebee) | Parsing errors |

### Font Specifications

- **Body text**: 9pt sans-serif, line-height 1.4
- **Code blocks**: 8pt Monaco/Menlo/Ubuntu Mono, monospace
- **Headings**: Hierarchical sizing from 16pt (h1) to 9pt (h5/h6)
- **Code in text**: 8pt with light gray background

**Sources:** [biomni/utils.py:2002-2366]()

---

## Observation Formatting

### Terminal Output Handling

The `format_observation_as_terminal()` function processes observation content with length limits:

**Maximum Length**: 10,000 characters (approximately 2 A4 pages)

```mermaid
graph TD
    A["format_observation_as_terminal()<br/>(content)"] --> B["Extract content from<br/>&lt;observation&gt; tags"]
    
    B --> C{"Content<br/>empty?"}
    C -->|Yes| SKIP["Return None"]
    
    C -->|No| D{"Length ><br/>10,000 chars?"}
    
    D -->|Yes| E["Truncate to 10,000<br/>Add '[Output truncated...]' notice"]
    D -->|No| F["Keep full content"]
    
    E --> G{"Contains<br/>'data:image/'?"}
    F --> G
    
    G -->|Yes| H["process_observation_with_images()"]
    G -->|No| I["Format as terminal<br/>```terminal block```"]
    
    H --> J["Split text and image parts"]
    J --> K["Text → terminal blocks<br/>Images → markdown ![Plot]()"]
    
    I --> L["Wrap in .title-text.observation<br/>HTML container"]
    K --> L
    
    L --> M["Return formatted HTML"]
    
    style A fill:#e1f5ff
    style H fill:#fff4e1
    style L fill:#f0e1ff
```

**Sources:** [biomni/utils.py:1584-1645](), [biomni/utils.py:1647-1706]()

---

## List and Plan Formatting

### Checkbox List Processing

The system automatically detects and formats plan checklists with checkbox symbols:

**Input Pattern**: `1. [✓] Completed step` or `1. [ ] Pending step`

**Output Format**: HTML list with styled checkboxes

```mermaid
graph TD
    A["format_lists_in_text()<br/>(text)"] --> B["Remove bold formatting<br/>from plan titles<br/>(**Plan:**, etc.)"]
    
    B --> C["remove_emojis_from_text()<br/>(🔧, 📊, ⚙️, etc.)"]
    
    C --> D["identify_list_blocks()<br/>(split text into blocks)"]
    
    D --> E["Loop through blocks"]
    
    E --> F{"Is checkbox<br/>list?"}
    
    F -->|Yes| G["format_single_list()<br/>(numbered items)"]
    F -->|No| H["Keep as plain text"]
    
    G --> I["Detect plan title<br/>(Plan, Updated Plan, etc.)"]
    
    I --> J["Convert checkboxes:<br/>[✓] → &lt;strong&gt;[x]&lt;/strong&gt;<br/>[✗] → &lt;strong&gt;[ ]&lt;/strong&gt;<br/>[ ] → &lt;strong&gt;[ ]&lt;/strong&gt;"]
    
    J --> K["Wrap in<br/>.title-text.plan container"]
    
    H --> L["Append to result"]
    K --> L
    
    L --> M{"More<br/>blocks?"}
    M -->|Yes| E
    M -->|No| N["Return formatted text"]
    
    style D fill:#e1f5ff
    style G fill:#fff4e1
    style K fill:#f0e1ff
```

**Sources:** [biomni/utils.py:1746-1793](), [biomni/utils.py:1795-1850](), [biomni/utils.py:1853-1920]()

---

## Error Handling

### Timeout Protection

The system includes multiple timeout mechanisms:

| Timeout Type | Duration | Purpose | Implementation |
|--------------|----------|---------|----------------|
| PDF Generation | 60 seconds | Prevent hanging on large documents | `signal.SIGALRM` in `save_conversation_history()` |
| Code Execution | Configurable (default 600s) | Prevent infinite loops | `run_with_timeout()` in `execute()` node |

### Error Cases and Handling

```mermaid
graph TD
    A["save_conversation_history()"] --> B{"save_pdf<br/>== True?"}
    
    B -->|No| END1["Print 'disabled'<br/>Return early"]
    
    B -->|Yes| C["Try: Generate markdown"]
    
    C --> D{"Success?"}
    D -->|No| ERR1["Print exception<br/>Cleanup temp file"]
    
    D -->|Yes| E["Set 60s timeout<br/>signal.alarm(60)"]
    
    E --> F["Try: convert_markdown_to_pdf()"]
    
    F --> G{"weasyprint<br/>available?"}
    G -->|Yes| H["Try: weasyprint conversion"]
    G -->|No| I["Try: markdown2pdf"]
    
    I --> J{"Success?"}
    J -->|No| K["Try: pandoc"]
    
    K --> L{"Success?"}
    L -->|No| ERR2["Raise ImportError:<br/>No PDF library"]
    
    H --> M{"Timeout?"}
    J -->|Yes| M
    L -->|Yes| M
    
    M -->|Yes| ERR3["Print 'timed out'<br/>after 60s"]
    M -->|No| N["Cancel alarm<br/>signal.alarm(0)"]
    
    N --> O["Print success message<br/>+ step count"]
    
    O --> P["Finally: cleanup<br/>os.unlink(temp_file)"]
    
    ERR1 --> P
    ERR2 --> P
    ERR3 --> P
    
    style C fill:#e1f5ff
    style F fill:#fff4e1
    style O fill:#e1ffe1
    style ERR1 fill:#ffe1e1
    style ERR2 fill:#ffe1e1
    style ERR3 fill:#ffe1e1
```

**Sources:** [biomni/agent/a1.py:2055-2132](), [biomni/utils.py:1922-2000]()

---

## Usage Examples

### Basic PDF Generation

```python
from biomni.agent import A1

# Initialize agent
agent = A1(path="./workspace")

# Execute some task
agent.go("Analyze gene expression data for BRCA1")

# Save conversation history as PDF
agent.save_conversation_history("results/analysis_report.pdf")
# Output: results/analysis_report.pdf
```

### PDF Without Images

```python
# Generate PDF without including plots/images
agent.save_conversation_history(
    "reports/text_only.pdf",
    include_images=False
)
```

### Disable PDF Generation

```python
# Skip PDF generation (e.g., for testing)
agent.save_conversation_history(
    "output.pdf",
    save_pdf=False
)
# Output: "PDF saving is disabled. No file will be saved."
```

### Automatic Extension Handling

```python
# Extension is added automatically if missing
agent.save_conversation_history("report")
# Creates: report.pdf

agent.save_conversation_history("report.pdf")
# Creates: report.pdf (no duplicate extension)

agent.save_conversation_history("report.md")
# Creates: report.pdf (replaces .md with .pdf)
```

**Sources:** [biomni/agent/a1.py:2055-2132]()

---

## Integration with Agent Workflow

### Conversation State Storage

The PDF generation system accesses conversation data from two sources:

| Source | Access Pattern | Priority |
|--------|---------------|----------|
| `_conversation_state` | Set by `go()` and `go_stream()` after execution completes | 1st (preferred) |
| `self.log` | Populated during streaming with `pretty_print()` output | 2nd (fallback) |

### Data Flow During Execution

```mermaid
graph TD
    A["agent.go(prompt)"] --> B["LangGraph workflow<br/>execution loop"]
    
    B --> C["generate() node<br/>LLM response"]
    C --> D["execute() node<br/>run code"]
    
    D --> E["get_captured_plots()<br/>from support_tools"]
    
    E --> F["Create execution_entry:<br/>{triggering_message,<br/>images, timestamp}"]
    
    F --> G["Append to<br/>_execution_results[]"]
    
    C --> H["pretty_print(message)<br/>format for display"]
    H --> I["Append to<br/>self.log[]"]
    
    B --> J["Store final_state"]
    J --> K["self._conversation_state<br/>= final_state"]
    
    K --> L["Later: save_conversation_history()"]
    G --> L
    I --> L
    
    L --> M["_get_messages_for_processing()"]
    M --> N{"_conversation_state<br/>exists?"}
    
    N -->|Yes| O["Use conversation_state<br/>(LangChain messages)"]
    N -->|No| P["Fallback to self.log<br/>(formatted strings)"]
    
    O --> Q["_generate_markdown_content()"]
    P --> Q
    
    style B fill:#e1f5ff
    style F fill:#fff4e1
    style K fill:#f0e1ff
    style Q fill:#e1ffe1
```

**Sources:** [biomni/agent/a1.py:1753-1783](), [biomni/agent/a1.py:1465-1547](), [biomni/agent/a1.py:2168-2186]()

---

## Performance Considerations

### Optimization Strategies

The system includes several optimizations for large documents:

| Optimization | Implementation | Benefit |
|--------------|----------------|---------|
| Minimal markdown extensions | Only `fenced_code` extension | Faster HTML conversion |
| Observation truncation | 10,000 character limit | Fits within 2 A4 pages |
| Image optimization | `optimize_images=True` in weasyprint | Smaller PDF file size |
| Timeout protection | 60-second limit | Prevents hanging |
| Temporary file cleanup | `os.unlink()` in finally block | Prevents disk bloat |

### Memory Management

- **Execution results**: Stored in memory during conversation
- **Base64 images**: Embedded directly in markdown (no external files)
- **Temporary files**: Created and deleted automatically
- **Plot clearing**: `_clear_execution_plots()` before each execution

**Sources:** [biomni/agent/a1.py:2055-2132](), [biomni/utils.py:1922-2000](), [biomni/utils.py:1584-1645]()

---

## File System Interaction

### Path Handling

```mermaid
graph TD
    A["save_conversation_history(filepath)"] --> B{"Directory<br/>exists?"}
    
    B -->|No| C["os.makedirs(directory,<br/>exist_ok=True)"]
    B -->|Yes| D["Proceed"]
    C --> D
    
    D --> E{"filepath ends<br/>with .pdf?"}
    
    E -->|Yes| F["pdf_path = filepath"]
    E -->|No| G{"filepath ends<br/>with .md?"}
    
    G -->|Yes| H["Remove .md extension"]
    G -->|No| I["Use as-is"]
    
    H --> J["pdf_path = base + '.pdf'"]
    I --> J
    
    J --> K["Create temp markdown file:<br/>tempfile.NamedTemporaryFile(<br/>suffix='.md', delete=False)"]
    
    K --> L["Write markdown_content<br/>to temp file"]
    
    L --> M["convert_markdown_to_pdf(<br/>temp_markdown_path,<br/>pdf_path)"]
    
    M --> N["Finally: os.unlink(<br/>temp_markdown_path)"]
    
    style K fill:#e1f5ff
    style M fill:#fff4e1
    style N fill:#f0e1ff
```

**Sources:** [biomni/agent/a1.py:2074-2132]()

---