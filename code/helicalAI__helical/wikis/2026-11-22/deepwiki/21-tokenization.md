# Tokenization

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [ci/tests/test_geneformer/test_geneformer_model.py](ci/tests/test_geneformer/test_geneformer_model.py)
- [ci/tests/test_geneformer/test_geneformer_tokenizer.py](ci/tests/test_geneformer/test_geneformer_tokenizer.py)
- [ci/tests/test_scgpt/test_scgpt_model.py](ci/tests/test_scgpt/test_scgpt_model.py)
- [helical/models/geneformer/geneformer_tokenizer.py](helical/models/geneformer/geneformer_tokenizer.py)
- [helical/models/geneformer/geneformer_utils.py](helical/models/geneformer/geneformer_utils.py)

</details>



This page documents the tokenization systems used within Helical to convert biological data (DNA sequences, RNA sequences, and gene expression values) into numerical token sequences that can be processed by foundation models. Tokenization is a critical preprocessing step that bridges biological data with neural network inputs.

For information about other data preprocessing steps that occur before tokenization, see [Data Preprocessing](#7.1).

## Tokenization Overview

In computational biology, tokenization transforms raw biological data into numerical tokens that foundation models can process. This transformation is essential because neural networks operate on numerical data, while biological information is typically represented as sequences, counts, or expression values.

```mermaid
flowchart TD
    subgraph "Natural Language Space"
        A["Biological Data"]
        A1["Gene Expression Matrix (AnnData)"]
        A2["DNA/RNA Sequences"]
        A --> A1
        A --> A2
    end
    
    subgraph "Tokenization Process"
        B["Data Preprocessing"]
        C["Generate Tokens"]
        D["Format Token Sequence"]
        B --> C
        C --> D
    end
    
    subgraph "Code Entity Space"
        E["TranscriptomeTokenizer.tokenize_data()"]
        F["tokenize_cell()"]
        G["format_cell_features()"]
    end
    
    A1 --> B
    A2 --> B
    
    B -.-> E
    C -.-> F
    D -.-> G
    
    D --> Z["Model-Ready Token Sequences"]
```

Sources:
- [helical/models/geneformer/geneformer_tokenizer.py:308-347]()
- [helical/models/geneformer/geneformer_tokenizer.py:64-82]()

## Model-Specific Tokenization Approaches

Different models in Helical tokenize biological data using distinct approaches:

| Model | Data Type | Tokenization Approach | Special Tokens | Input Size |
|-------|-----------|----------------------|----------------|------------|
| Geneformer v1 | Gene expression | Rank-based gene expression | None | 2048 |
| Geneformer v2 | Gene expression | Rank-based gene expression | `<cls>`, `<eos>` | 4096 |
| scGPT | Gene expression | Gene ID and count pairs | `<cls>` | Variable |
| HyenaDNA | DNA sequences | k-mer or character-level | Model-specific | Variable |
| Caduceus | DNA sequences | k-mer embeddings | Model-specific | Variable |

## Geneformer Tokenization

Geneformer's tokenization process transforms gene expression data into tokens through the `TranscriptomeTokenizer` class. This process is particularly designed for single-cell RNA-seq data.

```mermaid
flowchart TD
    A["AnnData Object"] --> B["Map Gene Symbols to Ensembl IDs"]
    B --> C["Normalize Expression Values"]
    C --> D["Filter Non-zero Expressions"]
    D --> E["Rank Genes by Expression"]
    E --> F["Convert to Token IDs"]
    F --> G["Add Special Tokens (v2 only)"]
    G --> H["Truncate to Model Input Size"]
    H --> I["Create Dataset"]
    
    subgraph "TranscriptomeTokenizer Methods"
        J["tokenize_data()"]
        K["tokenize_anndata()"]
        L["tokenize_cell()"]
        M["format_cell_features()"]
    end
    
    A -.-> J
    C -.-> K
    E -.-> L
    G -.-> M
```

Sources:
- [helical/models/geneformer/geneformer_tokenizer.py:196-304]()
- [helical/models/geneformer/geneformer_tokenizer.py:392-481]()

### Rank-Based Gene Expression Tokenization

Geneformer uses a unique approach where genes are ranked by their normalized expression values, and these ranks become the token sequence:

1. Normalize gene expression by dividing by cell-specific factors and gene-specific median values
2. Filter out genes with zero expression
3. Sort genes in descending order of normalized expression
4. Represent each gene by its token ID from a predefined dictionary

```mermaid
sequenceDiagram
    participant A as "Gene Expression"
    participant B as "Normalized Expression"
    participant C as "Ranked Gene Sequence"
    participant D as "Token Sequence"
    
    A->>B: Normalize by median values
    B->>C: Sort in descending order
    C->>D: Map genes to token IDs
    Note over D: Final token sequence
```

Sources:
- [helical/models/geneformer/geneformer_tokenizer.py:64-82]()
- [helical/models/geneformer/geneformer_tokenizer.py:427-479]()

### Special Tokens and Version Differences

Geneformer models come in two versions with different tokenization characteristics:

```mermaid
classDiagram
    class GeneformerBase {
        +process_data()
        +get_embeddings()
    }
    
    class GeneformerV1 {
        +model_input_size: 2048
        +special_tokens: None
        +embedding_modes: "cell", "gene"
    }
    
    class GeneformerV2 {
        +model_input_size: 4096
        +special_tokens: "<cls>", "<eos>"
        +embedding_modes: "cell", "gene", "cls"
    }
    
    GeneformerBase <|-- GeneformerV1
    GeneformerBase <|-- GeneformerV2
```

V2 models add special tokens to sequences:
- `<cls>` token at the beginning of the sequence
- `<eos>` token at the end of the sequence

These special tokens enable additional functionality like the "cls" embedding mode, which uses the embedding of just the first (`<cls>`) token.

Sources:
- [helical/models/geneformer/geneformer_tokenizer.py:525-536]()
- [ci/tests/test_geneformer/test_geneformer_model.py:129-135]()

## scGPT Tokenization

The scGPT model handles gene expression data differently from Geneformer:

```mermaid
flowchart TD
    A["AnnData Object"] --> B["Map Genes to Vocabulary IDs"]
    B --> C["Filter Out Genes Not in Vocabulary"]
    C --> D["Extract Gene IDs and Expression Counts"]
    D --> E["Create Token Sequences"]
    
    subgraph "scGPT Methods"
        F["process_data()"]
        G["ensure_data_validity()"]
    end
    
    A -.-> F
    B -.-> G
```

Key differences in scGPT tokenization:
- Maps gene symbols directly to vocabulary IDs without Ensembl ID conversion
- Filters out genes not present in the model's vocabulary
- Creates a dataset with gene IDs and their corresponding expression counts

Sources:
- [ci/tests/test_scgpt/test_scgpt_model.py:25-40]()

## Token Dictionaries and Mappings

Each model uses dictionaries to map between biological entities and token IDs:

```mermaid
classDiagram
    class GeneformerTokenizer {
        +gene_token_dict: Maps Ensembl IDs to tokens
        +token_to_ensembl_dict: Maps tokens to Ensembl IDs
        +gene_median_dict: Contains median expression values
    }
    
    class scGPTTokenizer {
        +vocab: Maps gene symbols to vocabulary IDs
        +vocab_id_to_str: Maps IDs back to gene symbols
    }
    
    class TokenMapping {
        +Map biological entities to numeric tokens
        +Support reverse mapping for interpretation
    }
    
    TokenMapping <|-- GeneformerTokenizer
    TokenMapping <|-- scGPTTokenizer
```

These dictionaries are typically loaded from pre-built files during model initialization and are critical for the tokenization process.

Sources:
- [helical/models/geneformer/geneformer_tokenizer.py:256-303]()
- [ci/tests/test_scgpt/test_scgpt_model.py:19-21]()

## Special Cases in Tokenization

The tokenization process handles several special cases:

### 1. Handling Missing and Duplicate Genes

```mermaid
flowchart TD
    A["Input Genes"] --> B{"In Vocabulary?"}
    B -->|"Yes"| C["Include in Tokenization"]
    B -->|"No"| D["Filter Out"]
    
    E["Duplicate Ensembl IDs"] --> F{"Collapse Enabled?"}
    F -->|"Yes"| G["Sum Expression Values"]
    F -->|"No"| H["Raise Error"]
```

Sources:
- [helical/models/geneformer/geneformer_tokenizer.py:85-186]()

### 2. Input Sequence Length Handling

```mermaid
flowchart TD
    A["Token Sequence"] --> B{"Length > Model Input Size?"}
    B -->|"Yes"| C["Truncate Sequence"]
    B -->|"No"| D["Use As Is"]
    C --> E{"Special Tokens?"}
    D --> E
    E -->|"Yes"| F["Add <cls> and <eos>"]
    E -->|"No"| G["Keep Sequence As Is"]
```

Sources:
- [helical/models/geneformer/geneformer_tokenizer.py:525-540]()

## From Tokens to Embeddings

After tokenization, models convert token sequences into embeddings:

```mermaid
flowchart TD
    A["Token Sequence"] --> B["Model Forward Pass"]
    B --> C["Hidden States"]
    C --> D{"Embedding Mode?"}
    
    D -->|"cell"| E["Average All Token Embeddings"]
    D -->|"gene"| F["Extract Per-Gene Embeddings"]
    D -->|"cls"| G["Use First Token Embedding"]
    
    subgraph "Embedding Computation Functions"
        H["_compute_embeddings_depending_on_mode()"]
        I["mean_nonpadding_embs()"]
    end
    
    E -.-> H
    E -.-> I
    F -.-> H
    G -.-> H
```

The embedding mode determines how token embeddings are processed:
- **cell mode**: Average embeddings across all tokens (excluding special tokens)
- **gene mode**: Extract individual gene embeddings
- **cls mode**: Use only the embedding of the `<cls>` token (only available in models with special tokens)

Sources:
- [helical/models/geneformer/geneformer_utils.py:41-99]()
- [helical/models/geneformer/geneformer_utils.py:323-336]()
- [ci/tests/test_scgpt/test_scgpt_model.py:126-171]()

## Using Tokenization in Helical

In practice, tokenization is usually handled internally by a model's `process_data` method:

```python
# Geneformer example
from helical.models.geneformer import Geneformer, GeneformerConfig

config = GeneformerConfig(model_name="gf-12L-95M-i4096")
model = Geneformer(config)

# This handles tokenization internally
tokenized_dataset = model.process_data(anndata_object, gene_names="gene_symbols")
embeddings = model.get_embeddings(tokenized_dataset)

# scGPT example
from helical.models.scgpt import scGPT, scGPTConfig

model = scGPT(scGPTConfig())
tokenized_dataset = model.process_data(anndata_object, gene_names="gene_names")
embeddings = model.get_embeddings(tokenized_dataset)
```

Sources:
- [ci/tests/test_geneformer/test_geneformer_model.py:60-82]()
- [ci/tests/test_geneformer/test_geneformer_model.py:149-150]()
- [ci/tests/test_scgpt/test_scgpt_model.py:25-40]()
- [ci/tests/test_scgpt/test_scgpt_model.py:149-150]()

## Conclusion

Tokenization is a fundamental step in the Helical framework that bridges biological data with neural network inputs. Each model in Helical implements its own tokenization strategy optimized for specific biological data types and model architectures. The framework provides consistent APIs for tokenization across different models while handling the unique requirements of each.

---