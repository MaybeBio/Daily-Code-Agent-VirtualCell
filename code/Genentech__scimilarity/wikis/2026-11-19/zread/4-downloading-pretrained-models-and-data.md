

This guide walks you through acquiring the essential pretrained models and data needed to use SCimilarity's powerful cell search and annotation capabilities. The pretrained model enables you to leverage a foundation model trained on millions of single-cell expression profiles, allowing immediate use without requiring extensive computational resources or training time.

Sources: [README.rst](README.rst#L1-L43)

## Understanding What You Need to Download

SCimilarity requires two main components to function: the pretrained model files and your query datasets. The pretrained model includes neural network weights, gene ordering information, and pre-built kNN indices that enable rapid similarity search across millions of cells. Additionally, tutorial datasets are available to help you get started with hands-on examples.

![SCimilarity Workflow](https://github.com/Genentech/scimilarity/blob/main/docs/_static/search_thumbnail_1.png?raw=true)

The architecture follows a clear separation of concerns: the model provides the embedding and search infrastructure, while your data represents the query states you want to investigate. When you download the pretrained model package, you receive everything needed for inference, including the encoder architecture weights and supporting metadata files.

Sources: [cell_embedding.py](src/scimilarity/cell_embedding.py#L1-L80), [nn_models.py](src/scimilarity/nn_models.py#L82-L100)

## Downloading the Pretrained Model

The SCimilarity pretrained model v1.1 is hosted on Zenodo and contains all necessary components for cell embedding, similarity search, and annotation. Use the following commands to download and extract the model:

```bash
# Download the model tarball
curl -L -o /models/model_v1.1.tar.gz \
  https://zenodo.org/records/10685499/files/model_v1.1.tar.gz?download=1

# Extract the archive
tar -xzvf /models/model_v1.1.tar.gz
```

This is a large file (several gigabytes), so download and extraction may take several minutes depending on your internet connection and disk speed. Ensure you have at least 10GB of free disk space available.

```mermaid
flowchart LR
    A[Zenodo Repository] --> B[Download model_v1.1.tar.gz]
    B --> C[Extract to /models/model_v1.1]
    C --> D[Verify Required Files]
    D --> E[Ready for Use]
    
    style A fill:#e3f2fd
    style B fill:#bbdefb
    style C fill:#90caf9
    style D fill:#64b5f6
    style E fill:#2196f3
```

> [!TIP]
> Choose a location with sufficient disk space and fast I/O performance, such as an SSD, as the kNN index will be frequently accessed during queries. The model files should remain uncompressed for optimal performance.

Sources: [cell_search_tutorial_1.ipynb](docs/notebooks/cell_search_tutorial_1.ipynb#L20-L30)

## Model File Structure

After extraction, the model directory contains several critical files that the SCimilarity API expects to find. Understanding this structure helps troubleshoot setup issues:

| File | Purpose | Required By |
|------|---------|-------------|
| `encoder.ckpt` | Neural network weights | CellEmbedding, CellQuery, CellAnnotation |
| `gene_order.tsv` | Canonical gene ordering | CellEmbedding |
| `layer_sizes.json` | Network architecture dimensions | CellEmbedding |
| `label_ints.csv` | Cell type label mappings | CellAnnotation |
| `cell_embedding` | Pre-computed embeddings (TileDB) | CellQuery |
| `cell_metadata` | Cell metadata database (TileDB) | CellQuery |
| `knn_index` | HNSW kNN search index | CellQuery, CellSearchKNN |

```
/models/model_v1.1/
├── encoder.ckpt              # PyTorch model weights
├── gene_order.tsv            # Gene identifiers
├── layer_sizes.json          # Network dimensions
├── label_ints.csv            # Cell type mappings
├── cell_embedding/           # TileDB array with embeddings
├── cell_metadata/            # TileDB array with metadata
└── knn_index                 # HNSW index file
```

When you initialize a CellQuery or CellAnnotation object, it automatically looks for these files in the specified model directory [cell_embedding.py](src/scimilarity/cell_embedding.py#L24-L48). The gene order file is particularly important—it defines the canonical gene ordering that your query data must match for accurate embeddings.

Sources: [cell_embedding.py](src/scimilarity/cell_embedding.py#L31-L48), [cell_search_knn.py](src/scimilarity/cell_search_knn.py#L40-L63)

## Downloading Tutorial Datasets

To follow along with the SCimilarity tutorials and verify your installation, download the provided tutorial datasets. These are smaller, curated datasets that demonstrate various capabilities:

```bash
# Download the cell search tutorial dataset
curl -L -o '/data/GSE136831_subsample.h5ad' \
  https://zenodo.org/records/13685881/files/GSE136831_subsample.h5ad?download=1
```

Additional tutorial datasets are available for different use cases:
- **Cell Search Tutorial**: IPF lung fibroblast data from Adams et al. 2020
- **Cell Annotation Tutorial**: Annotated datasets for classification tasks
- **Training Tutorial**: Sample data for training custom models

These datasets are in AnnData format (.h5ad files), which is the standard format for single-cell analysis in Python. The tutorial notebooks provide complete examples of how to load, process, and query these datasets.

Sources: [cell_search_tutorial_1.ipynb](docs/notebooks/cell_search_tutorial_1.ipynb#L1-L45)

## System Requirements and Memory Considerations

Running SCimilarity with the pretrained model demands substantial system resources due to the scale of the precomputed embeddings and kNN indices:

| Component | Minimum Requirement | Recommended |
|-----------|---------------------|-------------|
| System RAM | 64 GB | 128 GB+ |
| Disk Space | 20 GB | SSD with fast I/O |
| CPU | Multi-core | 8+ cores |
| GPU | Not required | Optional for training |

> [!TIP]
> If you experience crashes during model loading or kNN queries, verify your dependency versions. Older versions of hnswlib or numpy can cause memory issues. Consider using the provided Docker container or Conda environment file for a stable setup.

The memory requirements stem from loading the entire kNN index into memory for fast similarity search. The index enables queries across 23.4 million cells in seconds, but this speed comes at the cost of significant RAM usage. For environments with limited memory, consider using TileDB vector search which can operate with lower memory requirements at the cost of slower query performance.

Sources: [install.rst](docs/install.rst#L30-L40), [cell_search_knn.py](src/scimilarity/cell_search_knn.py#L40-L63)

## Verifying Your Downloaded Model

After downloading and extracting the model, verify that all required files are present and accessible:

```python
import os
from scimilarity import CellQuery

# Check model directory contents
model_path = "/models/model_v1.1"
expected_files = [
    "encoder.ckpt",
    "gene_order.tsv", 
    "layer_sizes.json",
    "label_ints.csv"
]

print("Verifying model files...")
for filename in expected_files:
    filepath = os.path.join(model_path, filename)
    exists = "✓" if os.path.exists(filepath) else "✗"
    print(f"  {exists} {filename}")

# Try initializing CellQuery to validate integrity
try:
    cq = CellQuery(model_path=model_path)
    print("\n✓ Model loaded successfully!")
    print(f"  Latent dimension: {cq.latent_dim}")
    print(f"  Number of genes: {cq.n_genes}")
except Exception as e:
    print(f"\n✗ Error loading model: {e}")
```

This verification script checks for essential files and attempts to initialize the CellQuery class, which performs additional validation such as checking the gene order file format and loading the encoder weights. If successful, it will display the model's latent dimension and gene count.

Sources: [cell_embedding.py](src/scimilarity/cell_embedding.py#L19-L70), [cell_query.py](src/scimilarity/cell_query.py#L31-L92)

## Using Downloaded Models with Docker

The SCimilarity Docker container provides a pre-configured environment but requires you to mount your downloaded model and data directories. The container includes four preset bind points for easy integration:

```bash
docker pull ghcr.io/genentech/scimilarity:latest

docker run -it --platform linux/amd64 -p 8888:8888 \
    -v /path/to/workspace:/workspace \
    -v /path/to/data:/data \
    -v /path/to/models:/models \
    ghcr.io/genentech/scimilarity:latest start-notebook
```

| Bind Point | Purpose | Local Path Example |
|------------|---------|-------------------|
| `/models` | Pretrained model files | `/local/scimilarity/models` |
| `/data` | scRNA-seq datasets | `/local/scimilarity/data` |
| `/workspace` | Notebooks and scripts | `/local/scimilarity/workspace` |
| `/scratch` | Temporary files | `/tmp/scimilarity` |

The Docker approach ensures all dependencies are correctly configured, including versions of hnswlib, numpy, and PyTorch that are known to work well with SCimilarity. This eliminates the common issue of crashes due to incompatible library versions.

Sources: [install.rst](docs/install.rst#L60-L99), [Dockerfile](docker/Dockerfile#L1-L50)

## Setting Up a Conda Environment

For users preferring a Conda-based setup, SCimilarity provides an environment configuration file that specifies compatible package versions:

```bash
# Download the environment file
wget https://github.com/Genentech/scimilarity/raw/main/docs/_static/environment.yaml

# Create the Conda environment
conda env create -f environment.yaml

# Activate the environment
conda activate scimilarity

# Install the SCimilarity package
pip install scimilarity
```

The environment file includes precise version specifications for critical dependencies like PyTorch, hnswlib, and numpy, preventing compatibility issues that can arise from version mismatches. After activating the environment, you can proceed with downloading the model and data as described above.

Sources: [install.rst](docs/install.rst#L42-L58), [environment.yaml](docs/_static/environment.yaml#L1-L30)

## Integration with SCimilarity APIs

Once downloaded, the pretrained model integrates seamlessly with SCimilarity's main APIs through simple initialization:

```python
from scimilarity import CellQuery, CellAnnotation

# Initialize for cell search queries
model_path = "/models/model_v1.1"
cq = CellQuery(model_path=model_path)

# Initialize for cell type annotation
ca = CellAnnotation(model_path=model_path)

# The model is now ready for queries
embeddings = cq.get_embeddings(your_data_matrix)
nearest = cq.search_nearest(embeddings, k=100)
```

The initialization process automatically loads the encoder weights, reads the gene order file, and prepares the kNN index for fast similarity search. For CellAnnotation, additional label mappings are loaded to enable cell type classification. Both classes inherit from CellEmbedding, which handles the core model loading logic.

Sources: [cell_query.py](src/scimilarity/cell_query.py#L31-L92), [cell_annotation.py](src/scimilarity/cell_annotation.py#L23-L70)

## Troubleshooting Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Model not found | FileNotFoundError for encoder.ckpt | Verify extraction completed without errors |
| Gene mismatch | Shape errors during embedding | Use `align_dataset` to match gene order |
| Memory crashes | Process killed during kNN load | Increase RAM or reduce memory_budget parameter |
| Slow queries | Queries taking minutes instead of seconds | Use HNSW index instead of exhaustive search |
| Import errors | Missing dependency errors | Use Docker or Conda environment file |

The most common issue is gene order mismatch between your query data and the pretrained model's expected gene order. The `align_dataset` utility function handles this by reordering and filtering your data to match the model's canonical gene list:

```python
from scimilarity.utils import align_dataset

# Your data may have different genes
aligned_data = align_dataset(
    data=your_anndata,
    target_gene_order=cq.gene_order,
    keep_obsm=True
)
```

Sources: [utils.py](src/scimilarity/utils.py#L50-L134), [cell_search_tutorial_1.ipynb](docs/notebooks/cell_search_tutorial_1.ipynb#L50-L120)

## Next Steps

After successfully downloading and verifying the pretrained model, you're ready to explore SCimilarity's capabilities:

- [Quick Start](2-quick-start) - Begin with a hands-on tutorial showing basic cell search functionality
- [Docker Usage](5-docker-usage) - Learn advanced Docker configurations and deployment strategies  
- [Cell Embedding Fundamentals](6-cell-embedding-fundamentals) - Understand how gene expression is transformed into latent representations
- [Cell Query API Reference](9-cell-query-api-reference) - Detailed documentation of query methods and parameters

The pretrained model provides immediate access to SCimilarity's foundation model capabilities, enabling you to search across millions of annotated cells and perform cell type annotation without any training. Download the tutorial datasets and work through the notebooks to build familiarity with the API before applying SCimilarity to your own research questions.
