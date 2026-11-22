# Configuration System

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [ci/tests/test_uce/test_gene_embeddings.py](ci/tests/test_uce/test_gene_embeddings.py)
- [examples/run_models/configs/geneformer_config.yaml](examples/run_models/configs/geneformer_config.yaml)
- [examples/run_models/configs/hyena_dna_config.yaml](examples/run_models/configs/hyena_dna_config.yaml)
- [examples/run_models/configs/scgpt_config.yaml](examples/run_models/configs/scgpt_config.yaml)
- [examples/run_models/configs/uce_config.yaml](examples/run_models/configs/uce_config.yaml)
- [helical/models/geneformer/geneformer_config.py](helical/models/geneformer/geneformer_config.py)
- [helical/models/hyena_dna/standalone_hyenadna.py](helical/models/hyena_dna/standalone_hyenadna.py)
- [helical/models/scgpt/scgpt_config.py](helical/models/scgpt/scgpt_config.py)
- [helical/models/scgpt/scgpt_utils.py](helical/models/scgpt/scgpt_utils.py)
- [helical/models/uce/gene_embeddings.py](helical/models/uce/gene_embeddings.py)
- [helical/models/uce/uce_config.py](helical/models/uce/uce_config.py)

</details>



The Configuration System in Helical manages model-specific parameters, file paths, and resource requirements for all supported foundation models. It provides a consistent interface for initializing and customizing models while abstracting away the complexities of model-specific configurations and file management.

## Overview

The Configuration System serves several key purposes:

1. Defines model-specific parameters with sensible defaults
2. Manages file paths for model weights and associated resources
3. Specifies which files need to be downloaded for each model
4. Validates model parameters and ensures compatibility
5. Provides a consistent interface across different model types

This page focuses on the configuration architecture and usage patterns. For information on model-specific parameters, see the corresponding model pages in sections [4. RNA Models](#4) and [5. DNA Models](#5).

```mermaid
flowchart TD
    A["User Code"] --> B["Model Configuration"]
    B --> C{"Model Type?"}
    
    C -->|"RNA Models"| D1["GeneformerConfig"]
    C -->|"RNA Models"| D2["scGPTConfig"]
    C -->|"RNA Models"| D3["UCEConfig"]
    C -->|"RNA Models"| D4["HelixmRNAConfig"]
    C -->|"RNA Models"| D5["Mamba2mRNAConfig"]
    
    C -->|"DNA Models"| E1["HyenaDNAConfig"]
    C -->|"DNA Models"| E2["CaduceusConfig"] 
    C -->|"DNA Models"| E3["Evo2Config"]
    
    D1 --> F["Configure Default Parameters"]
    D2 --> F
    D3 --> F
    D4 --> F
    D5 --> F
    E1 --> F
    E2 --> F
    E3 --> F
    
    F --> G["Set File Paths"]
    G --> H["Define Download Requirements"]
    H --> I["Create Configuration Dictionary"]
    I --> J["Initialize Model"]
    J --> K["Model Ready for Use"]
```

Sources: [helical/models/geneformer/geneformer_config.py:7-146](), [helical/models/uce/uce_config.py:20-148](), [helical/models/scgpt/scgpt_config.py:6-103]()

## Configuration Architecture

All model configuration classes in Helical follow a consistent pattern. Each configuration class:

1. Defines parameters with default values via constructor arguments
2. Maps model names/versions to specific parameter sets
3. Creates a list of required files to download
4. Generates appropriate file paths based on the cache directory
5. Constructs a configuration dictionary that can be passed to model constructors

```mermaid
classDiagram
    class ModelConfig {
        +__init__(model_specific_params)
        +model_map
        +list_of_files_to_download
        +config
    }
    
    class GeneformerConfig {
        +model_name: str
        +batch_size: int
        +emb_layer: int
        +emb_mode: str
        +device: str
        +model_map
        +list_of_files_to_download
        +files_config
        +config
    }
    
    class scGPTConfig {
        +pad_token: str
        +batch_size: int
        +emb_mode: str
        +fast_transformer: bool
        +model_map
        +list_of_files_to_download
        +config
    }
    
    class UCEConfig {
        +model_name: str
        +batch_size: int
        +species: str
        +gene_embedding_model: str
        +model_map
        +list_of_files_to_download
        +config
    }
    
    ModelConfig <|-- GeneformerConfig
    ModelConfig <|-- scGPTConfig
    ModelConfig <|-- UCEConfig
```

Sources: [helical/models/geneformer/geneformer_config.py:7-146](), [helical/models/uce/uce_config.py:20-148](), [helical/models/scgpt/scgpt_config.py:6-103]()

## Configuration Classes

### Common Structure

While there isn't a formal base configuration class, all model configurations follow a similar structure. Each configuration class has:

1. A constructor with model-specific parameters
2. A `model_map` dictionary that maps model names to parameter sets
3. A `list_of_files_to_download` listing required files
4. A `config` dictionary with all necessary parameters for model initialization

Example from GeneformerConfig:

```mermaid
flowchart TD
    A["GeneformerConfig"] --> B["__init__(model_name, batch_size, etc.)"]
    B --> C["Validate model_name"]
    C --> D["Determine model_version"]
    D --> E["Generate list_of_files_to_download"]
    E --> F["Set model_dir paths"]
    F --> G["Create files_config dictionary"]
    G --> H["Create config dictionary"] 
```

Sources: [helical/models/geneformer/geneformer_config.py:40-146]()

### Model-Specific Configurations

#### GeneformerConfig

The GeneformerConfig class configures Geneformer models with parameters like:
- `model_name`: Which pre-trained model to use
- `batch_size`: Batch size for processing
- `emb_layer`: Which embedding layer to use
- `emb_mode`: Embedding extraction mode (cls, cell, gene)
- Device and accelerator settings

This configuration also handles model version differences (v1 vs v2) and validates that the embedding mode is compatible with the chosen model.

Sources: [helical/models/geneformer/geneformer_config.py:7-146]()

#### scGPTConfig

The scGPTConfig class manages configurations for the scGPT model, including:
- Tokenization parameters (`pad_token`, `mask_value`, `pad_value`)
- Model architecture parameters (`nlayers`, `nheads`, `embsize`)
- Training parameters (`dropout`, `batch_size`)
- Hardware settings (`device`, `accelerator`)

Sources: [helical/models/scgpt/scgpt_config.py:6-103]()

#### UCEConfig

The UCEConfig class configures the Universal Cell Embedding model with:
- `model_name`: Which pre-trained model to use
- `species`: The species for the data (human, mouse, etc.)
- `gene_embedding_model`: The protein embedding model to use
- Token and padding parameters
- Model dimension parameters
- Hardware settings

This configuration also handles species-specific gene embeddings.

Sources: [helical/models/uce/uce_config.py:20-148](), [helical/models/uce/gene_embeddings.py:15-112]()

## Configuration Usage

### Creating Model Configurations

Model configurations can be created either programmatically or loaded from YAML files. 

#### Programmatic Configuration

To create a configuration programmatically:

```python
from helical.models.geneformer import GeneformerConfig

# Create with default parameters
config = GeneformerConfig()

# Create with custom parameters
config = GeneformerConfig(
    model_name="gf-12L-95M-i4096",
    batch_size=16,
    emb_mode="cell",
    device="cuda"
)
```

#### YAML Configuration

YAML files can also define configurations:

```yaml
# Geneformer config example
model_name: "gf-12L-30M-i2048"
batch_size: 24
emb_layer: -1
emb_mode: "cell"
device: "cpu"
accelerator: False
```

These YAML configurations can be loaded and used to instantiate model configurations.

Sources: [examples/run_models/configs/geneformer_config.yaml:1-6](), [examples/run_models/configs/scgpt_config.yaml:1-17](), [examples/run_models/configs/uce_config.yaml:1-18]()

### Configuration to Model Flow

The configuration system plays a central role in model initialization. Here's the typical flow:

```mermaid
sequenceDiagram
    participant User
    participant Config as Configuration Class
    participant Downloader
    participant ModelLoader
    participant Model

    User->>Config: Create configuration
    Config->>Config: Validate parameters
    Config->>Config: Set file paths
    Config->>Config: Define download requirements
    User->>Downloader: Use list_of_files_to_download
    Downloader->>Downloader: Download missing files
    User->>ModelLoader: Pass configuration
    ModelLoader->>ModelLoader: Load model files using paths
    ModelLoader->>Model: Initialize with parameters
    Model->>User: Return initialized model
```

Sources: [helical/models/geneformer/geneformer_config.py:98-117](), [helical/models/uce/uce_config.py:114-120](), [helical/models/scgpt/scgpt_utils.py:72-111]()

## File Management

The configuration system is responsible for managing file paths and download requirements. Each model configuration:

1. Uses a central cache directory defined in `helical.constants.paths.CACHE_DIR_HELICAL`
2. Creates model-specific subdirectories
3. Generates a list of files that need to be downloaded
4. Provides paths to these files for the model initialization process

### Cache Structure

```mermaid
flowchart TD
    A["CACHE_DIR_HELICAL<br>~/.cache/helical/models/"] --> B1["geneformer/"]
    A --> B2["scgpt/"]
    A --> B3["uce/"]
    A --> B4["hyena_dna/"]
    
    B1 --> C1["v1/"]
    B1 --> C2["v2/"]
    
    C1 --> D1["gf-6L-30M-i2048/"]
    C1 --> D2["gf-12L-30M-i2048/"]
    
    C2 --> D3["gf-12L-95M-i4096/"]
    C2 --> D4["gf-20L-95M-i4096/"]
    
    B2 --> C3["scGPT_CP/"]
    
    B3 --> C4["protein_embeddings/"]
    B3 --> C5["all_tokens.torch"]
    B3 --> C6["4layer_model.torch"]
    
    C4 --> E1["Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"]
    C4 --> E2["Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt"]
```

Sources: [helical/models/geneformer/geneformer_config.py:119-131](), [helical/models/uce/uce_config.py:114-122](), [helical/models/uce/gene_embeddings.py:15-28]()

## Common Configuration Parameters

While each model has specific parameters, there are common parameters that appear across most configurations:

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `model_name` | Name or identifier of the pre-trained model | "gf-12L-30M-i2048", "4layer_model" |
| `batch_size` | Batch size for processing | 16, 24, 32 |
| `device` | Device to run the model on | "cpu", "cuda" |
| `accelerator` | Whether to use an accelerator | True, False |
| `emb_mode` | How embeddings are extracted | "cls", "cell", "gene" |

Sources: [helical/models/geneformer/geneformer_config.py:134-146](), [helical/models/uce/uce_config.py:124-148](), [helical/models/scgpt/scgpt_config.py:82-103]()

## Parameter Validation and Defaults

Configuration classes validate parameters and set appropriate defaults:

1. Model name validation: Checking against available models
   ```python
   if model_name not in self.model_map:
       raise ValueError(f"Model name {model_name} not found in available models: {self.model_map.keys()}")
   ```

2. Parameter compatibility validation:
   ```python
   model_version = "v2" if "95M" in model_name else "v1"
   if model_version == "v1" and emb_mode == "cls":
       raise ValueError(f"Verson 1 Geneformer models do not support the CLS embedding mode")
   ```

3. Default values through constructor parameters:
   ```python
   def __init__(
       self,
       model_name: Literal["gf-6L-30M-i2048", "gf-12L-30M-i2048", ...] = "gf-12L-30M-i2048",
       batch_size: int = 24,
       emb_layer: int = -1,
       # ... more parameters with defaults
   ):
   ```

Sources: [helical/models/geneformer/geneformer_config.py:86-96](), [helical/models/uce/uce_config.py:109-112]()

## Summary

The Configuration System in Helical provides a consistent approach to model configuration across different model types. It handles parameter management, file paths, and download requirements, creating a streamlined experience for initializing and using models while abstracting away the complexities of model-specific configurations.

When using Helical, you typically:
1. Create a configuration object with your desired parameters
2. Pass this configuration to the appropriate model loader
3. The configuration handles file paths and validates parameters
4. The model is initialized with the configuration

This architecture allows for flexibility in how models are configured while maintaining a consistent interface across the framework.

---