# Documentation

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [.readthedocs.yaml](.readthedocs.yaml)
- [docs/model_cards/caduceus.md](docs/model_cards/caduceus.md)
- [docs/model_cards/geneformer.md](docs/model_cards/geneformer.md)
- [docs/model_cards/helix_mrna.md](docs/model_cards/helix_mrna.md)
- [docs/model_cards/hyena_dna.md](docs/model_cards/hyena_dna.md)
- [docs/model_cards/mamba2_mrna.md](docs/model_cards/mamba2_mrna.md)
- [docs/model_cards/scgpt.md](docs/model_cards/scgpt.md)
- [docs/model_cards/uce.md](docs/model_cards/uce.md)
- [docs/notebooks/Cell-Gene-Cls-embedding-generation.ipynb](docs/notebooks/Cell-Gene-Cls-embedding-generation.ipynb)
- [docs/notebooks/Cell-Type-Annotation.ipynb](docs/notebooks/Cell-Type-Annotation.ipynb)
- [docs/notebooks/Cell-Type-Classification-Fine-Tuning.ipynb](docs/notebooks/Cell-Type-Classification-Fine-Tuning.ipynb)
- [docs/notebooks/Evo-2.ipynb](docs/notebooks/Evo-2.ipynb)
- [docs/notebooks/Geneformer-vs-UCE.ipynb](docs/notebooks/Geneformer-vs-UCE.ipynb)
- [docs/notebooks/Genegpt-sample-run.ipynb](docs/notebooks/Genegpt-sample-run.ipynb)
- [docs/notebooks/Helix-mRNA.ipynb](docs/notebooks/Helix-mRNA.ipynb)
- [docs/notebooks/Hyena-DNA-Inference.ipynb](docs/notebooks/Hyena-DNA-Inference.ipynb)
- [docs/notebooks/HyenaDNA-Fine-Tuning.ipynb](docs/notebooks/HyenaDNA-Fine-Tuning.ipynb)

</details>



This page describes the documentation system used within the Helical framework. It covers the structure and components of documentation, guidelines for writing and maintaining documentation, and the tools and processes used to generate the documentation. For information about setting up the development environment, see [Development and Deployment](#9).

## Overview

Helical's documentation system consists of several components that work together to provide comprehensive information for users and developers:

1. **Model Cards**: Standardized documentation for each supported model
2. **Jupyter Notebooks**: Interactive examples and tutorials
3. **Technical Documentation**: Documentation of the codebase structure and APIs
4. **Read the Docs Integration**: Web-based documentation hosting

The documentation follows a standard structure to ensure consistency across all models, making it easier for users to find information regardless of which model they're working with.

### Documentation Architecture

```mermaid
flowchart TD
    A["Documentation System"]
    A --> B["Model Cards"]
    A --> C["Jupyter Notebooks"]
    A --> D["Technical Documentation"]
    A --> E["ReadTheDocs Integration"]
    
    B --> B1["Model Details"]
    B --> B2["Training Data"]
    B --> B3["Usage Examples"]
    B --> B4["Performance Metrics"]
    
    C --> C1["Usage Examples"]
    C --> C2["Model Comparisons"]
    C --> C3["Downstream Tasks"]
    
    D --> D1["API Reference"]
    D --> D2["System Architecture"]
    D --> D3["Installation Guide"]
    
    E --> E1["Web-based Docs"]
    E --> E2["Versioned Docs"]
    E --> E3["Search Functionality"]
```

Sources: [.readthedocs.yaml](). [docs/model_cards/geneformer.md](). [docs/model_cards/uce.md](). [docs/notebooks/Geneformer-vs-UCE.ipynb]().

## Model Cards

Model cards are the primary documentation for each AI model in the Helical framework. They provide standardized information about a model's capabilities, training process, intended use cases, and limitations.

### Model Card Structure

```mermaid
flowchart TD
    A["Model Card"]
    A --> B["Model Details\n(Name, Version, Description)"]
    A --> C["Model Developers\n(Authors, Contact Info)"]
    A --> D["Model Type\n(Architecture, Domain)"]
    A --> E["Model Purpose\n(Use Cases, Applications)"]
    A --> F["Training Data\n(Sources, Preprocessing)"]
    A --> G["Model Performance\n(Evaluation Metrics)"]
    A --> H["Model Limitations\n(Known Issues)"]
    A --> I["Usage Examples\n(Code Snippets)"]
    A --> J["Citation\n(Publication Info)"]
```

Sources: [docs/model_cards/geneformer.md](). [docs/model_cards/uce.md](). [docs/model_cards/scgpt.md](). [docs/model_cards/helix_mrna.md](). [docs/model_cards/mamba2_mrna.md](). [docs/model_cards/caduceus.md](). [docs/model_cards/hyena_dna.md]().

### Model Card Components

| Section | Purpose | Contents |
|---------|---------|----------|
| Model Details | Basic information | Name, version, description |
| Model Developers | Attribution | Authors, institutions, contact information |
| Model Type | Technical classification | Architecture, domain, input/output types |
| Model Purpose | Use cases | Intended uses, capabilities, applications |
| Training Data | Data information | Data sources, preprocessing, biases |
| Model Performance | Evaluation results | Metrics, benchmarks, comparison |
| Model Limitations | Constraints | Known issues, limitations, edge cases |
| How to Use | Instructions | Code examples, installation, configuration |
| Citation | Academic attribution | BibTeX citations, publication information |

Sources: [docs/model_cards/geneformer.md](). [docs/model_cards/uce.md](). [docs/model_cards/scgpt.md]().

## Creating and Updating Documentation

### Model Card Creation Process

```mermaid
sequenceDiagram
    participant Dev as "Developer"
    participant Card as "Model Card"
    participant Repo as "GitHub Repository"
    participant RTD as "ReadTheDocs"
    participant User as "End User"
    
    Dev->>Card: Create model card using template
    Dev->>Card: Fill in model details, performance metrics, etc.
    Dev->>Card: Add code examples
    Dev->>Repo: Commit model card to repository
    Repo->>RTD: Trigger documentation build
    RTD->>RTD: Build documentation
    RTD->>User: Serve updated documentation
    User->>Card: Access model card
```

Sources: [.readthedocs.yaml](). [docs/model_cards/geneformer.md](). [docs/model_cards/uce.md]().

### Documentation Updates

The documentation is automatically built and updated whenever changes are pushed to the main branch of the repository. This is configured through the `.readthedocs.yaml` file, which specifies the build process and dependencies.

```mermaid
flowchart LR
    A["Code/Doc Changes"] --> B["Git Commit"]
    B --> C["Push to Main Branch"]
    C --> D["ReadTheDocs Webhook"]
    D --> E["Build Documentation"]
    E --> F["Deploy to ReadTheDocs"]
    F --> G["Updated Public Docs"]
```

Sources: [.readthedocs.yaml]().

## Jupyter Notebooks

Jupyter notebooks provide interactive examples and tutorials for using the Helical framework. They demonstrate how to use the different models for various tasks, such as cell type classification, model comparisons, and downstream tasks.

### Notebook Structure

Each notebook typically includes:

1. Installation instructions
2. Imports and setup
3. Data loading and preprocessing
4. Model initialization and configuration
5. Model execution (inference or fine-tuning)
6. Results visualization and analysis

### Notebook Examples

The Helical repository includes several example notebooks:

| Notebook | Purpose | Models Used |
|----------|---------|-------------|
| Geneformer-vs-UCE | Comparing models for cell-type classification | Geneformer, UCE |
| Cell-Type-Annotation | Cell type annotation prediction | scGPT, Geneformer |
| Cell-Type-Classification-Fine-Tuning | Classification fine-tuning | Geneformer, scGPT |
| Hyena-DNA-Inference | Downstream prediction | HyenaDNA |
| HyenaDNA-Fine-Tuning | Fine-tuning for classification | HyenaDNA |
| Evo-2 | mRNA downstream tasks | Evo2, Helix-mRNA |

Sources: [docs/notebooks/Geneformer-vs-UCE.ipynb](). [docs/notebooks/Cell-Type-Annotation.ipynb](). [docs/notebooks/Cell-Type-Classification-Fine-Tuning.ipynb](). [docs/notebooks/Hyena-DNA-Inference.ipynb](). [docs/notebooks/HyenaDNA-Fine-Tuning.ipynb](). [docs/notebooks/Evo-2.ipynb](). [docs/notebooks/Cell-Gene-Cls-embedding-generation.ipynb]().

## Documentation Infrastructure

### Read the Docs Configuration

The Helical documentation is hosted on Read the Docs, which automatically builds and publishes the documentation. The configuration is specified in the `.readthedocs.yaml` file.

```yaml
version: 2
build:
  os: ubuntu-22.04
  tools:
    python: "3.11"

mkdocs:
  configuration: mkdocs.yml
  fail_on_warning: false

python:
   install:
   - requirements: docs/requirements.txt
   - method: pip
     path: .
```

This configuration specifies:
- The build environment (Ubuntu 22.04 with Python 3.11)
- The documentation tool (MkDocs)
- The dependencies required to build the documentation

Sources: [.readthedocs.yaml]().

### Documentation Code to Site Flow

```mermaid
flowchart TD
    A["Documentation Sources"]
    A --> B1["Model Cards (.md)"]
    A --> B2["Jupyter Notebooks (.ipynb)"]
    A --> B3["Technical Docs (.md)"]
    A --> B4["Configuration Files"]
    
    B1 --> C["Documentation Build Process"]
    B2 --> C
    B3 --> C
    B4 --> C
    
    C --> D["MkDocs Static Site Generator"]
    D --> E["HTML/CSS/JS Output"]
    E --> F["ReadTheDocs Hosting"]
    F --> G["Public Documentation Site"]
    
    G --> H1["Model Reference"]
    G --> H2["API Documentation"]
    G --> H3["Tutorials"]
    G --> H4["Examples"]
```

Sources: [.readthedocs.yaml]().

## Documentation Best Practices

### Model Card Guidelines

When creating or updating model cards:

1. Follow the standard template and structure
2. Include all required sections
3. Provide accurate and up-to-date information
4. Include code examples for common use cases
5. Document limitations and biases
6. Include performance metrics and evaluation results
7. Add proper citations and attribution

### Code Example Standards

Code examples should:

1. Be complete and runnable
2. Include imports and setup
3. Use consistent style and formatting
4. Include comments explaining key steps
5. Show both basic and advanced usage
6. Handle errors and edge cases
7. Follow the Helical API conventions

Example from Geneformer model card:

```python
from helical.models.geneformer import Geneformer, GeneformerConfig
import anndata as ad

# Example configuration
model_config = GeneformerConfig(model_name="gf-12L-95M-i4096", batch_size=10)
geneformer_v2 = Geneformer(model_config)

# Example usage for base pretrained model
ann_data = ad.read_h5ad("anndata_file.h5ad")
dataset = geneformer_v2.process_data(ann_data)
embeddings = geneformer_v2.get_embeddings(dataset)
print("Base model embeddings shape:", embeddings.shape)
```

Sources: [docs/model_cards/geneformer.md:189-202]().

## Documentation Tools and Technologies

### Tools Used

| Tool | Purpose | Usage |
|------|---------|-------|
| MkDocs | Static site generator | Converts Markdown to HTML |
| Read the Docs | Documentation hosting | Builds and serves the documentation |
| Markdown | Markup language | Used for writing documentation |
| Jupyter Notebooks | Interactive examples | Used for tutorials and examples |
| GitHub | Version control | Stores and manages documentation |

### Documentation-to-Code Relationship

```mermaid
flowchart LR
    subgraph "Code Layer"
        A1["Model Implementation"]
        A2["API Design"]
        A3["Configuration"]
    end
    
    subgraph "Documentation Layer"
        B1["Model Cards"]
        B2["API Reference"]
        B3["Configuration Guide"]
        B4["Tutorials"]
    end
    
    subgraph "User Experience Layer"
        C1["Model Selection"]
        C2["Implementation"]
        C3["Troubleshooting"]
        C4["Learning"]
    end
    
    A1 --> B1
    A2 --> B2
    A3 --> B3
    A1 --> B4
    A2 --> B4
    
    B1 --> C1
    B2 --> C2
    B3 --> C2
    B4 --> C4
    B1 --> C3
    B2 --> C3
    B3 --> C3
```

Sources: [docs/model_cards/geneformer.md](). [docs/model_cards/uce.md](). [docs/notebooks/Geneformer-vs-UCE.ipynb]().

## Maintenance and Updates

### Update Process

Documentation should be updated:

1. When new models are added
2. When existing models are updated
3. When APIs change
4. When new features are added
5. When bugs are fixed
6. When new use cases are identified

### Review Process

Documentation updates should be reviewed for:

1. Technical accuracy
2. Completeness
3. Clarity
4. Consistency with the rest of the documentation
5. Code example correctness
6. Proper formatting and structure

## Conclusion

Effective documentation is crucial for the usability and adoption of the Helical framework. By following the guidelines and processes outlined in this document, you can help ensure that the documentation remains comprehensive, accurate, and useful for both new and experienced users.