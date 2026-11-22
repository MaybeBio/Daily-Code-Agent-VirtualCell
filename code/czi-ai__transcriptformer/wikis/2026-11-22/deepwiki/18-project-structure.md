# Project Structure

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.gitignore](.gitignore)
- [pyproject.toml](pyproject.toml)
- [src/transcriptformer/__init__.py](src/transcriptformer/__init__.py)
- [src/transcriptformer/data/__init__.py](src/transcriptformer/data/__init__.py)

</details>



This page provides an overview of the TranscriptFormer repository's file organization and directory structure. Understanding the project structure is essential for developers who want to contribute to the project or customize the models for their specific needs. For information about configuration classes used in the project, see [Configuration Classes](#6.1).

## Top-Level Organization

The TranscriptFormer repository follows a modern Python package structure with a src-based layout. This layout separates the package code from other project files and helps maintain a clean distribution.

```mermaid
graph TD
    Root["transcriptformer/"]
    Root --> Src["src/"]
    Root --> Tests["tests/"]
    Root --> Configs["configs/"]
    Root --> Notebooks["notebooks/"]
    Root --> Docs["docs/"]
    Root --> ProjectFiles["Project Files
    (pyproject.toml, README.md, etc.)"]
    
    Src --> TranscriptformerPkg["transcriptformer/"]
```

Sources: [pyproject.toml](), [.gitignore]()

## Package Organization

The `src/transcriptformer` directory contains the main package code organized into several subpackages that align with the system's architecture.

```mermaid
graph TD
    TranscriptformerPkg["src/transcriptformer/"]
    TranscriptformerPkg --> Models["models/"]
    TranscriptformerPkg --> Data["data/"]
    TranscriptformerPkg --> Inference["inference/"]
    TranscriptformerPkg --> Config["config/"]
    TranscriptformerPkg --> Utils["utils/"]
    TranscriptformerPkg --> CLI["cli/"]
    
    Models --> BaseModel["base.py"]
    Models --> ModelVariants["variants/"]
    Models --> ModelUtils["utils.py"]
    
    Data --> Loading["loading.py"]
    Data --> Processing["processing.py"]
    Data --> Tokenization["tokenization.py"]
    Data --> Vocabulary["vocabulary.py"]
    Data --> Datasets["datasets.py"]
    
    Inference --> Engine["engine.py"]
    Inference --> Pipeline["pipeline.py"]
    
    Config --> ModelConfig["model_config.py"]
    Config --> DataConfig["data_config.py"]
    Config --> InferenceConfig["inference_config.py"]
```

Sources: [src/transcriptformer/__init__.py](), [src/transcriptformer/data/__init__.py]()

## Key Components and Their Roles

The TranscriptFormer codebase is organized around several key components, each with a specific responsibility in the system:

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| Models | Defines the TranscriptFormer model architecture variants | models/base.py, models/variants/*.py |
| Data | Handles data loading, processing, and tokenization | data/loading.py, data/processing.py, data/tokenization.py |
| Inference | Manages the inference process | inference/engine.py, inference/pipeline.py |
| Config | Contains configuration classes for models, data, and inference | config/*.py |
| Utils | Provides utility functions used across the codebase | utils/*.py |
| CLI | Implements the command-line interface | cli/*.py |

## Data Flow and Component Interaction

The following diagram illustrates how the different components interact during the typical inference workflow:

```mermaid
flowchart TD
    subgraph "CLI Layer"
        CLI["cli.main"]
    end
    
    subgraph "Configuration"
        HydraConfig["hydra config"]
        ModelConfig["ModelConfig"]
        DataConfig["DataConfig"]
        InferenceConfig["InferenceConfig"]
    end
    
    subgraph "Data Processing"
        DataLoader["data.loading.load_data"]
        DataProcessor["data.processing.process_data"]
        Tokenizer["data.tokenization.tokenize"]
        Dataset["data.datasets.AnnDataset"]
    end
    
    subgraph "Model"
        TranscriptformerModel["models.TranscriptFormer"]
    end
    
    subgraph "Inference"
        InferenceEngine["inference.engine.InferenceEngine"]
        Pipeline["inference.pipeline.run_inference"]
    end
    
    CLI --> HydraConfig
    HydraConfig --> ModelConfig
    HydraConfig --> DataConfig
    HydraConfig --> InferenceConfig
    
    CLI --> Pipeline
    Pipeline --> InferenceEngine
    InferenceEngine --> TranscriptformerModel
    InferenceEngine --> Dataset
    
    Dataset --> DataLoader
    Dataset --> DataProcessor
    Dataset --> Tokenizer
    
    ModelConfig --> TranscriptformerModel
    DataConfig --> DataProcessor
    DataConfig --> Tokenizer
    InferenceConfig --> InferenceEngine
```

Sources: [pyproject.toml]()

## Configuration System

TranscriptFormer uses Hydra for configuration management. Configuration files are stored in the `configs/` directory and organized hierarchically to match the structure of the code.

```mermaid
graph TD
    ConfigRoot["configs/"]
    ConfigRoot --> ModelConfigs["model/"]
    ConfigRoot --> DataConfigs["data/"]
    ConfigRoot --> InferenceConfigs["inference/"]
    ConfigRoot --> Default["config.yaml"]
    
    ModelConfigs --> Metazoa["metazoa.yaml"]
    ModelConfigs --> Exemplar["exemplar.yaml"]
    ModelConfigs --> Sapiens["sapiens.yaml"]
    
    DataConfigs --> Processing["processing.yaml"]
    DataConfigs --> Tokenization["tokenization.yaml"]
    
    InferenceConfigs --> Default["default.yaml"]
    InferenceConfigs --> OOD["ood.yaml"]
```

Sources: [.gitignore:180-181]()

## Development Tools and Project Files

The repository includes several files that support the development workflow and package distribution:

| File | Purpose |
|------|---------|
| pyproject.toml | Defines package metadata, dependencies, and build configuration |
| .gitignore | Specifies files and directories to exclude from version control |
| .pre-commit-config.yaml | Configures pre-commit hooks for code quality |
| README.md | Contains project documentation and usage instructions |

The project uses modern Python tooling:

- **Ruff** for linting and formatting, configured in pyproject.toml
- **Pytest** for testing
- **Hatchling** as the build backend

Sources: [pyproject.toml:1-97](), [.gitignore:1-183]()

## Test Structure

Tests are organized to mirror the package structure, ensuring comprehensive test coverage across all components:

```mermaid
graph TD
    TestRoot["tests/"]
    TestRoot --> ModelTests["test_models/"]
    TestRoot --> DataTests["test_data/"]
    TestRoot --> InferenceTests["test_inference/"]
    TestRoot --> ConfTests["test_config/"]
    TestRoot --> UtilTests["test_utils/"]
    TestRoot --> IntegrationTests["test_integration/"]
    
    ModelTests --> TestBaseModel["test_base.py"]
    ModelTests --> TestVariants["test_variants.py"]
    
    DataTests --> TestLoading["test_loading.py"]
    DataTests --> TestProcessing["test_processing.py"]
    DataTests --> TestTokenization["test_tokenization.py"]
    
    InferenceTests --> TestEngine["test_engine.py"]
    InferenceTests --> TestPipeline["test_pipeline.py"]
```

Sources: [pyproject.toml:98-99]()

## Package Distribution

The project is configured as a Python package that can be installed via pip. The build configuration in `pyproject.toml` specifies that the package code is located in the `src/transcriptformer` directory.

```mermaid
graph TD
    Build["Build Process"]
    Build --> Hatchling["Hatchling Build Backend"]
    Hatchling --> SourcePackage["Source Package (.tar.gz)"]
    Hatchling --> WheelPackage["Wheel Package (.whl)"]
    
    SourcePackage --> PyPI["PyPI Repository"]
    WheelPackage --> PyPI
    
    PyPI --> Installation["pip install transcriptformer"]
```

Sources: [pyproject.toml:1-5](), [pyproject.toml:50-51]()

## Output and Artifact Management

The repository structure also accounts for various outputs and artifacts generated during model training and inference:

- **lightning_logs/** - Contains logs generated by PyTorch Lightning during training
- **outputs/** - Stores outputs from inference runs
- **mlruns/** - Used for experiment tracking with MLflow
- **multirun/** - Contains outputs from Hydra's multirun feature

These directories are included in the .gitignore file to prevent large binary files from being committed to the repository.

Sources: [.gitignore:171-176]()

## Summary

The TranscriptFormer project follows a well-organized structure that separates concerns and promotes maintainability. The src-based layout with clear module boundaries aligns with modern Python best practices. The configuration system, powered by Hydra, provides flexibility while maintaining control over model behavior. This organization facilitates both research experimentation and production deployment of the TranscriptFormer models.

---