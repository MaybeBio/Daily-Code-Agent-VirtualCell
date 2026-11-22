

TranscriptFormer treats each cell as a sequence of gene tokens, where every gene is mapped to a discrete vocabulary index backed by a pretrained protein embedding. The **gene tokenizer and vocabulary** layer is the bridge between raw AnnData expression matrices and the transformer encoder — it resolves gene identifiers to integer indices, constructs the embedding matrix from ESM2 protein representations, and defines the special tokens that structure the input sequence. Understanding this layer is essential for configuring inference, interpreting model outputs, and extending the model to new species.

## Vocabulary Architecture

The vocabulary is a flat dictionary mapping string identifiers (Ensembl gene IDs or special token names) to integer indices. It is constructed in two phases: first, **special tokens** are assigned the lowest indices (starting from 0), then **gene tokens** fill the remaining positions sequentially. This ordering guarantees that special token indices are stable across vocabularies of different sizes — a property that the model relies on when constructing masks, right-shifting sequences, and performing embedding surgery.

```mermaid
flowchart TD
    A[ESM2 Embedding Files<br/>.pkl or .h5] -->|load_embeddings| B[Gene Name → Embedding Dict]
    B --> C[Deduplication<br/>Remove duplicate gene names]
    C --> D[build_gene_vocab_from_list]
    E[SPECIAL_TOKENS<br/>unknown, START, END, RD,<br/>CELL, PAD, ⟨] --> D
    D --> F[gene_vocab_dict<br/>String → Integer Index]
    C --> G[Special Token Embeddings<br/>Random, seeded]
    C --> H[Gene Embeddings<br/>From ESM2]
    G --> I[Concatenated Embedding Matrix<br/>[special_embs | gene_embs]]
    H --> I
    F --> J[GeneVocab Dataclass<br/>vocab_dict, vocab_size,<br/>pad_idx, start_idx, end_idx, cell_idx]
    I --> J
```

The seven special tokens reserve indices 0–6, with an eighth separator token `⟨` at index 7. Their roles within the model are distinct: `[START]` and `[END]` delimit the gene sequence for autoregressive decoding, `[PAD]` marks positions that the attention mask should ignore, `[CELL]` serves as a cell-level token for auxiliary conditioning, `[RD]` functions as a read token, and `unknown` captures any gene not present in the vocabulary. The `⟨` token acts as a delimiter within the sequence.

Sources: [vocab.py](src/transcriptformer/tokenizer/vocab.py#L8-L9), [vocab.py](src/transcriptformer/tokenizer/vocab.py#L42-L74), [dataclasses.py](src/transcriptformer/data/dataclasses.py#L203-L214)

## Special Tokens Reference

| Token | Default Index | Role in Model |
|-------|--------------|---------------|
| `unknown` | 0 | Fallback for out-of-vocabulary genes |
| `[START]` | 1 | Prepended during autoregressive right-shift of gene tokens |
| `[END]` | 2 | Appended to mark sequence termination for the gene ID head |
| `[RD]` | 3 | Read token used in sequence structure |
| `[CELL]` | 4 | Cell-level conditioning token in auxiliary position |
| `[PAD]` | 5 | Padding token — masked out in attention and loss computation |
| `⟨` | 6 | Delimiter token within the gene sequence |

> [!TIP]
> When extending the vocabulary through embedding surgery, existing special token embeddings are preserved from the original model rather than re-randomized. This ensures that the model's learned behavior around sequence boundaries is not disrupted when adapting to new species.

Sources: [vocab.py](src/transcriptformer/tokenizer/vocab.py#L8-L9), [model.py](src/transcriptformer/model/model.py#L65-L75)

## Embedding Matrix Construction

The embedding matrix is built by `construct_gene_embeddings()`, which takes a list of pretrained embedding file paths and the special tokens list. It loads each file (supporting both `.pkl` pickle and `.h5` HDF5 formats), concatenates gene names and their corresponding ESM2 embedding vectors, and deduplicates any genes that appear across multiple files (keeping only the first occurrence and logging a warning for duplicates). Special token embeddings are generated using a **seeded random number generator** (`np.random.RandomState(42)`) so that they are reproducible across runs, and they are prepended to the gene embedding rows so that index alignment with the vocabulary dictionary is exact.

The final matrix has shape `(vocab_size, embedding_dim)`, where `vocab_size = len(special_tokens) + len(unique_genes)` and `embedding_dim` matches the ESM2 output dimension (2560 for ESM2-3B). An assertion verifies that the matrix row count equals the vocabulary length, guarding against misalignment bugs.

Sources: [vocab.py](src/transcriptformer/tokenizer/vocab.py#L12-L74), [utils.py](src/transcriptformer/utils/utils.py#L11-L38)

## BatchGeneTokenizer

The `BatchGeneTokenizer` is the runtime component that converts arrays of gene name strings into tensors of vocabulary indices. It wraps a `np.vectorize` call over the vocabulary dictionary lookup, mapping each gene name to its index. Any gene name not found in the vocabulary resolves to the `unknown` token index (0), and a warning is logged with the count of unrecognized genes. This design enables the tokenizer to handle **open-vocabulary inputs gracefully** — genes absent from the pretrained ESM2 mapping still receive a valid embedding (the randomly initialized `unknown` vector) rather than causing a runtime error.

```python
# Simplified tokenization flow
tokenizer = BatchGeneTokenizer(gene_vocab)
gene_tokens = tokenizer(gene_names_array)  # np.array of strings → torch.Tensor of indices
# Any unmapped gene → index 0 ("unknown"), with warning logged
```

Sources: [tokenizer.py](src/transcriptformer/tokenizer/tokenizer.py#L7-L25)

## BatchObsTokenizer

Alongside gene tokenization, TranscriptFormer supports **auxiliary (observation-level) tokenization** through `BatchObsTokenizer`. This tokenizer operates on categorical metadata columns (e.g., `assay`, `tissue`) stored in `adata.obs`. Each auxiliary column has its own sub-vocabulary (loaded from JSON files), and the tokenizer maps each metadata value to its corresponding index. The result is a tensor of shape `(n_aux_columns,)` per cell, which is concatenated before the gene embeddings in the input sequence.

Sources: [tokenizer.py](src/transcriptformer/tokenizer/tokenizer.py#L28-L34), [vocab.py](src/transcriptformer/tokenizer/vocab.py#L95-L115)

## Vocabulary Integration with the Model

The vocabulary and embedding matrix are loaded together via `load_vocabs_and_embeddings()`, which is called during model initialization. This function reads the ESM2 mapping file paths from configuration, constructs the gene vocabulary and embedding matrix, and returns both the vocab dictionaries and a `torch.tensor` embedding matrix. The model then wraps these into structured dataclasses:

- **`GeneVocab`**: Holds `vocab_dict`, `vocab_size`, and the critical index constants (`pad_idx`, `start_idx`, `end_idx`, `cell_idx`). These indices are used throughout the forward pass for right-shifting, masking, and padding.
- **`AuxVocab`**: Holds `vocab_dict`, `vocab_size`, `pad_ids`, and `aux_seq_len`. The `pad_ids` are the `unknown` token indices from each auxiliary sub-vocabulary, used to construct the attention pad mask.

The `PretrainedEmbeddings` layer receives the full embedding matrix, projects it to the model's `embed_dim`, and applies normalization and dropout. The embedding matrix is **frozen** (`freeze=True`) during training, preserving the ESM2 protein representations as stable anchors for gene identity.

```mermaid
flowchart LR
    subgraph Data Pipeline
        A[AnnData<br/>gene_names + counts] -->|BatchGeneTokenizer| B[gene_token_indices]
        A -->|BatchObsTokenizer| C[aux_token_indices]
    end
    subgraph Model Forward
        B -->|right-shift + embed| D[Gene Embeddings]
        C -->|embed| E[Aux Embeddings]
        D --> F[Concatenate<br/>[aux | gene]]
        E --> F
        F --> G[TranscriptEncoder<br/>Flex Attention]
        G --> H[CountDecoderHead + GeneIDHead]
    end
    subgraph Vocabulary
        V[gene_vocab_dict] -.->|index lookup| B
        V -.->|pad_idx, start_idx| D
        V -.->|end_idx| H
    end
```

Sources: [vocab.py](src/transcriptformer/tokenizer/vocab.py#L77-L93), [model.py](src/transcriptformer/model/model.py#L60-L90), [dataclasses.py](src/transcriptformer/data/dataclasses.py#L203-L237)

## Gene Name Resolution and Filtering

Before tokenization occurs, gene names undergo a resolution pipeline in the dataloader. The `load_gene_features()` function reads gene identifiers from `adata.var[gene_col_name]` (default column: `ensembl_id`), strips Ensembl version suffixes (e.g., `ENSG00000141510.5` → `ENSG00000141510`), and handles duplicate gene names that may arise from version stripping. If `remove_duplicate_genes` is set, duplicates are removed keeping the first occurrence; otherwise, a `ValueError` is raised.

The `filter_to_vocab` option further constrains the expression matrix to only include genes present in the vocabulary. Genes not in the vocabulary are filtered from the columns of the expression matrix *before* tokenization, which means they never reach the `unknown` token path in the `BatchGeneTokenizer`. This is a design choice: `filter_to_vocab` removes out-of-vocabulary genes entirely (their expression counts are dropped), while the `unknown` token in the tokenizer serves as a safety net for any genes that slip through.

Sources: [dataloader.py](src/transcriptformer/data/dataloader.py#L220-L272), [dataloader.py](src/transcriptformer/data/dataloader.py#L33-L69)

## Vocabulary and Embedding Surgery

The vocabulary is not immutable — it can be replaced at inference time through **embedding surgery**, which is the mechanism for adapting TranscriptFormer to species not seen during training. The `change_embedding_layer()` function constructs a new vocabulary and embedding matrix from a different set of ESM2 protein embeddings, while preserving the special token embeddings from the original model. The new vocabulary re-indexes all gene names with an offset equal to the number of preserved special tokens, and the old special token embeddings are prepended to maintain index alignment.

This architecture means that the vocabulary is the single source of truth for gene-to-index mapping, and any change to it must be accompanied by a corresponding change to the embedding matrix. The `GeneVocab` dataclass on the model is updated in place, and the `padding_idx` of the new embedding layer is set to the original `pad_idx` value.

Sources: [embedding_surgery.py](src/transcriptformer/model/embedding_surgery.py#L1-L80)

## Configuration Reference

The vocabulary and tokenization behavior is controlled through the `DataConfig` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gene_col_name` | `ensembl_id` | Column in `adata.var` containing gene identifiers |
| `special_tokens` | `SPECIAL_TOKENS` | List of special tokens prepended to the vocabulary |
| `esm2_mappings` | — | List of ESM2 embedding filenames |
| `esm2_mappings_path` | — | Directory containing ESM2 embedding files |
| `filter_to_vocabs` | `true` | Filter expression matrix to vocabulary genes only |
| `pad_zeros` | `true` | Replace zero-count gene tokens with `[PAD]` index |
| `gene_pad_token` | `[PAD]` | Token name used for padding |
| `aux_cols` | `null` | Comma-separated auxiliary metadata columns |
| `aux_vocab_path` | — | Directory containing `*_vocab.json` files |
| `remove_duplicate_genes` | `false` | Remove duplicates instead of raising error |

Sources: [dataclasses.py](src/transcriptformer/data/dataclasses.py#L107-L173), [inference_config.yaml](conf/inference_config.yaml#L22-L36)

## Next Steps

With the vocabulary and tokenization pipeline understood, the next logical step is to examine how tokenized data flows through the full loading and preprocessing pipeline in [Data Loading and Preprocessing](12-data-loading-and-preprocessing). For the downstream consumer of vocabulary indices, see [Flex Attention Transformer Encoder](6-flex-attention-transformer-encoder) and how special tokens shape the attention masks. To understand how the vocabulary is swapped for new species, see [Embedding Surgery for New Species](10-embedding-surgery-for-new-species).