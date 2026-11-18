# Genomics and Single-Cell Tools

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [biomni/tool/genomics.py](biomni/tool/genomics.py)
- [biomni/tool/tool_description/genomics.py](biomni/tool/tool_description/genomics.py)

</details>



## Overview

This page documents the genomics and single-cell analysis tools in Biomni, covering scRNA-seq analysis, variant analysis, gene embeddings, ChIP-seq peak calling, chromatin interaction analysis, and comparative genomics. These tools integrate state-of-the-art bioinformatics algorithms with LLM-guided annotation capabilities.

All functions are implemented in [biomni/tool/genomics.py]() with structured tool descriptions in [biomni/tool/tool_description/genomics.py](). For database query tools, see page 5.1. For lab automation tools, see page 5.3.

## Tool Architecture

**Figure 1: Genomics Tool Organization by Function**

The genomics toolkit provides 21 functions organized into four categories:

```mermaid
graph TB
    GenomicsModule["biomni/tool/genomics.py<br/>21 functions"]
    
    GenomicsModule --> scRNACategory["Single-cell RNA-seq<br/>7 functions"]
    GenomicsModule --> GeneCategory["Gene Analysis<br/>5 functions"]
    GenomicsModule --> ChromatinCategory["Chromatin & Epigenomics<br/>4 functions"]
    GenomicsModule --> CompGenCategory["Comparative Genomics<br/>1 function"]
    
    scRNACategory --> Transfer["unsupervised_celltype_transfer_between_scRNA_datasets"]
    scRNACategory --> LLMAnnotate["annotate_celltype_scRNA"]
    scRNACategory --> Azimuth["annotate_celltype_with_panhumanpy"]
    scRNACategory --> ScviEmbed["create_scvi_embeddings_scRNA"]
    scRNACategory --> HarmonyEmbed["create_harmony_embeddings_scRNA"]
    scRNACategory --> UCEEmbed["get_uce_embeddings_scRNA"]
    scRNACategory --> IMAMap["map_to_ima_interpret_scRNA"]
    
    GeneCategory --> SpeciesConv["interspecies_gene_conversion"]
    GeneCategory --> ESMEmbed["generate_gene_embeddings_with_ESM_models"]
    GeneCategory --> RNASeq["get_rna_seq_archs4"]
    GeneCategory --> GSEA["gene_set_enrichment_analysis"]
    GeneCategory --> GSEAList["get_gene_set_enrichment_analysis_supported_database_list"]
    
    ChromatinCategory --> HiC["analyze_chromatin_interactions"]
    ChromatinCategory --> ChIPSeq["perform_chipseq_peak_calling_with_macs2"]
    ChromatinCategory --> Motif["find_enriched_motifs_with_homer"]
    ChromatinCategory --> Overlap["analyze_genomic_region_overlap"]
    
    CompGenCategory --> CompGen["analyze_comparative_genomics_and_haplotypes"]
```

**Sources:** [biomni/tool/genomics.py:1-2165](), [biomni/tool/tool_description/genomics.py:1-657]()

## Single-cell RNA-seq Analysis

The scRNA-seq toolkit provides seven functions for cell type annotation, batch correction, and embedding generation. These tools support complete workflows from raw counts to annotated cell types.

### Cell Type Annotation Methods

**Table 1: Cell Type Annotation Functions**

| Function | Algorithm | Input Format | Output | Implementation |
|----------|-----------|--------------|--------|----------------|
| `unsupervised_celltype_transfer_between_scRNA_datasets` | popV multi-method ensemble (10 classifiers) | Annotated reference + query AnnData | `popv_output/predictions.csv` | Lines 18-105 |
| `annotate_celltype_scRNA` | LLM marker gene interpretation | Clustered AnnData with leiden | `annotated.h5ad` with `cell_type` column | Lines 463-597 |
| `annotate_celltype_with_panhumanpy` | Azimuth Neural Network | Raw AnnData | `annotated_cell_metadata.csv` + embeddings | Lines 600-795 |

**Sources:** [biomni/tool/genomics.py:18-795](), [biomni/tool/tool_description/genomics.py:1-95]()

### Cell Type Transfer with popV

**Figure 2: popV Multi-Method Annotation Pipeline**

The `unsupervised_celltype_transfer_between_scRNA_datasets` function implements reference-based label transfer with 10 classifier options:

```mermaid
graph TB
    RefH5ad["path_to_annotated_h5ad<br/>AnnData with ref_labels_key"]
    QueryH5ad["path_to_not_annotated_h5ad<br/>AnnData to annotate"]
    
    LoadRef["sc.read_h5ad<br/>Line 49"]
    LoadQuery["sc.read_h5ad<br/>Line 53"]
    
    AddCounts["Add adata.layers['counts']<br/>Required for scVI/scANVI<br/>Lines 57-58"]
    
    ProcessQuery["popv.preprocessing.Process_Query<br/>Lines 66-75"]
    
    subgraph ClassifierFlags["Boolean Flags (Lines 78-89)"]
        CELLTYPIST["CELLTYPIST=False"]
        KNN_BBKNN["KNN_BBKNN=False"]
        KNN_HARMONY["KNN_HARMONY=False"]
        KNN_SCANORAMA["KNN_SCANORAMA=False"]
        KNN_SCVI["KNN_SCVI=False"]
        ONCLASS["ONCLASS=False"]
        RandomForest["Random_Forest=False"]
        SCANVI_POPV["SCANVI_POPV=True (default)"]
        SVM["Support_Vector=False"]
        XGBoost["XGboost=False"]
    end
    
    AnnotateData["popv.annotation.annotate_data<br/>Lines 97-101"]
    
    PredictionsCSV["output_folder/popv_output/predictions.csv<br/>Line 103"]
    
    RefH5ad --> LoadRef
    QueryH5ad --> LoadQuery
    LoadRef --> AddCounts
    LoadQuery --> AddCounts
    AddCounts --> ProcessQuery
    ProcessQuery --> ClassifierFlags
    ClassifierFlags --> AnnotateData
    AnnotateData --> PredictionsCSV
```

**Classifier Selection:** Each of the 10 methods is controlled by a boolean parameter. Default is `SCANVI_POPV=True` only. Multiple methods can be enabled simultaneously for ensemble predictions. The function uses `popv.settings.n_jobs` for parallelization [biomni/tool/genomics.py:60-101]().

**Sources:** [biomni/tool/genomics.py:18-105](), [biomni/tool/tool_description/genomics.py:462-576]()

### LLM-Based Cell Type Annotation

**Figure 3: LLM-Guided Cluster Annotation Workflow**

The `annotate_celltype_scRNA` function uses LLM reasoning to assign cell types based on differentially expressed genes:

```mermaid
sequenceDiagram
    participant H5ad as "adata_filename (AnnData)"
    participant RankGenes as "sc.tl.rank_genes_groups"
    participant DataLake as "data_lake_path/czi_census_datasets_v4.parquet"
    participant LLM as "get_llm(llm)"
    participant ChainOp as "PromptTemplate | LLM (Lines 545-546)"
    participant Output as "data_dir/annotated.h5ad"
    
    H5ad->>RankGenes: Compute top 20 markers per leiden cluster
    RankGenes-->>ChainOp: markers dict (Lines 515-523)
    
    DataLake->>ChainOp: Load czi_celltype_set (Lines 526-529)
    
    Note over ChainOp: Prompt template with data_info, cluster_info, czi_celltype
    
    loop For each leiden cluster (Lines 554-580)
        ChainOp->>LLM: chain.invoke with cluster_info
        LLM-->>ChainOp: response.content or response.text
        
        alt Parse and validate
            ChainOp->>ChainOp: Split by semicolon into name, score, reason
            ChainOp->>ChainOp: Check if name in czi_celltype_set
            ChainOp->>ChainOp: Accept annotation
        else Retry with validation prompt
            ChainOp->>ChainOp: Append validation message to cluster_info
            ChainOp->>LLM: Retry chain.invoke
        end
    end
    
    ChainOp->>Output: adata.obs['cell_type'], adata.obs['cell_type_reason']
```

**Ontology Validation:** The function loads cell type names from the CZI Census dataset [biomni/tool/genomics.py:526-529]() and validates LLM outputs against this ontology [biomni/tool/genomics.py:572-579](). Invalid predictions trigger a retry with explicit validation feedback.

**Output Format:** The LLM must respond with `"name; score; reason"` format. Both `cell_type` and `cell_type_reason` columns are added to `adata.obs` [biomni/tool/genomics.py:591-593]().

**Sources:** [biomni/tool/genomics.py:463-597](), [biomni/tool/tool_description/genomics.py:1-54]()

### Batch Integration and Embeddings

**Table 2: Embedding Generation Functions**

| Function | Algorithm | Output Location | Implementation |
|----------|-----------|-----------------|----------------|
| `create_scvi_embeddings_scRNA` | scVI VAE (Lines 813-817) + scANVI semi-supervised (Lines 822-833) | `adata.obsm['X_scVI']`, `adata.obsm['X_scANVI']` | Lines 801-839 |
| `create_harmony_embeddings_scRNA` | Harmony iterative PCA correction (Line 851) | `adata.obsm['X_harmony']` | Lines 842-857 |
| `get_uce_embeddings_scRNA` | UCE foundation model via `eval_single_anndata.main` | `adata.obsm['X_uce']` | Lines 861-911 |
| `map_to_ima_interpret_scRNA` | kNN on UCE embeddings to IMA reference | `adata.obs['mapped_cell_type']`, `adata.obs['mapping_confidence']` | Lines 914-971 |

**UCE + IMA Workflow:** The `get_uce_embeddings_scRNA` function checks if `{basename}_uce_adata.h5ad` exists to avoid recomputation [biomni/tool/genomics.py:886-889](). The `map_to_ima_interpret_scRNA` function requires `X_uce` in `adata.obsm` [biomni/tool/genomics.py:922-923]() and uses `sklearn.neighbors.NearestNeighbors` to find nearest cells in the IMA atlas [biomni/tool/genomics.py:936-943]().

**Sources:** [biomni/tool/genomics.py:801-971](), [biomni/tool/tool_description/genomics.py:96-215]()

### Azimuth Neural Network Annotation

**Figure 4: Isolated Environment Execution for panhumanpy**

The `annotate_celltype_with_panhumanpy` function runs panhumanpy in an isolated conda environment due to dependency conflicts:

```mermaid
graph TB
    CheckPanhumanEnv["conda_env_exists('panhumanpy_env')<br/>Lines 639-644"]
    
    CreatePanhumanEnv["Create conda env python=3.10<br/>conda run pip install panhumanpy<br/>Lines 646-653"]
    
    TempScript["Write script to tempfile.mkdtemp()<br/>run_panhumanpy.py<br/>Lines 662-769"]
    
    subgraph IsolatedExec["Subprocess Execution (Lines 773-774)"]
        CondaRun["conda run -n panhumanpy_env python script_path"]
        
        subgraph ScriptActions["Script Actions (Lines 688-761)"]
            LoadAnnData["sc.read_h5ad(adata_path)"]
            InitAzimuth["ph.AzimuthNN(adata, feature_names_col)"]
            GetMetadata["azimuth.cells_meta"]
            
            OptionalEmbed["if umap: azimuth.azimuth_embed()"]
            OptionalUMAP["if umap: azimuth.azimuth_umap()"]
            OptionalRefine["if refine: azimuth.azimuth_refine()"]
            
            SaveOutputs["Save metadata.csv, embeddings.npy, umap.npy"]
            PackAdata["azimuth.pack_adata(save_path)"]
        end
        
        WriteJSON["Write result.json with log or error"]
    end
    
    ReadJSON["Read result.json from temp_dir<br/>Lines 780-790"]
    CleanupTemp["shutil.rmtree(temp_dir)<br/>Line 793"]
    
    CheckPanhumanEnv -->|Missing| CreatePanhumanEnv
    CheckPanhumanEnv -->|Exists| TempScript
    CreatePanhumanEnv --> TempScript
    TempScript --> IsolatedExec
    IsolatedExec --> ReadJSON
    ReadJSON --> CleanupTemp
```

**Dependency Isolation:** The panhumanpy package requires dependencies that conflict with the main `biomni_e1` environment [biomni/tool/genomics.py:655-659](). The function uses `subprocess.run` with `conda run -n panhumanpy_env` [biomni/tool/genomics.py:773-774]() to execute in isolation.

**Communication Protocol:** The main process writes a Python script to a temporary directory [biomni/tool/genomics.py:662-769](), executes it via subprocess, and reads results from `result.json` [biomni/tool/genomics.py:780-790](). The temporary directory is cleaned up after execution [biomni/tool/genomics.py:793]().

**Sources:** [biomni/tool/genomics.py:600-795](), [biomni/tool/tool_description/genomics.py:55-94]()

## Gene Analysis Tools

### Interspecies Gene Conversion

**Figure 5: BioMart Homology Mapping with Fallback Strategy**

The `interspecies_gene_conversion` function maps ENSEMBL gene IDs between 13 species using the Ensembl BioMart API:

```mermaid
graph TB
    GeneList["gene_list: list of ENSEMBL IDs"]
    SpeciesMaps["species_mapping dict (Lines 158-172)<br/>homolog_mapping dict (Lines 174-188)"]
    
    ValidateSpecies["Validate source_species and target_species<br/>Lines 190-198"]
    
    GetDataset["pybiomart.Dataset(source_dataset_name)<br/>Lines 209"]
    
    PrimaryQuery["dataset.query with filters={'link_ensembl_gene_id': gene_list}<br/>Lines 211-213"]
    
    CheckEmpty["Check if mapping.empty<br/>Line 215"]
    
    FallbackQuery["Query all homologs then filter by gene_list<br/>Lines 216-217"]
    
    BuildMapping["Create gene_mapping dict from filtered results<br/>Lines 219-220"]
    
    ConvertGenes["Apply mapping to gene_list, use np.nan for missing<br/>Lines 222-227"]
    
    SaveCSV["Save to {source_species}_to_{target_species}_gene_conversion.csv<br/>Lines 231-234"]
    
    GeneList --> SpeciesMaps
    SpeciesMaps --> ValidateSpecies
    ValidateSpecies --> GetDataset
    GetDataset --> PrimaryQuery
    PrimaryQuery --> CheckEmpty
    CheckEmpty -->|Empty| FallbackQuery
    CheckEmpty -->|Has Data| BuildMapping
    FallbackQuery --> BuildMapping
    BuildMapping --> ConvertGenes
    ConvertGenes --> SaveCSV
```

**Species Support:** The function defines two dictionaries: `species_mapping` maps common names to Ensembl dataset names (e.g., "human" → "hsapiens_gene_ensembl") [biomni/tool/genomics.py:158-172](), and `homolog_mapping` maps to homolog attribute names [biomni/tool/genomics.py:174-188](). Supports 13 species total.

**Fallback Strategy:** If the primary filtered query returns empty results [biomni/tool/genomics.py:215](), the function queries all homologs without filters [biomni/tool/genomics.py:216-217]() then filters locally. A second exception handler [biomni/tool/genomics.py:238-281]() provides additional robustness.

**Sources:** [biomni/tool/genomics.py:108-281](), [biomni/tool/tool_description/genomics.py:577-605]()

### Protein Language Model Embeddings

**Figure 6: ESM Model Embedding Generation with Memory-Safe Batching**

The `generate_gene_embeddings_with_ESM_models` function generates protein embeddings using Facebook AI's ESM models:

```mermaid
graph TB
    EnsemblIDs["ensembl_gene_ids: list of ENSG IDs"]
    
    FetchIsoforms["_fetch_isoform_sequences(ensembl_gene_id)<br/>REST API call to Ensembl<br/>Lines 284-311"]
    
    LoadESM["esm.pretrained.load_model_and_alphabet(model_name)<br/>batch_converter = alphabet.get_batch_converter()<br/>Lines 343-344"]
    
    DeviceSelect["device = torch.device('cuda' if available else 'cpu')<br/>Lines 348-350"]
    
    FilterSeq["Filter isoforms by max_sequence_length (default 1024)<br/>Lines 357-361"]
    
    subgraph BatchLoop["Batch Processing Loop (Lines 376-430)"]
        PrepBatch["batch_data = all_data[i:i+batch_size]"]
        Convert["batch_labels, batch_strs, batch_tokens = batch_converter"]
        Forward["results = model(batch_tokens, repr_layers=[layer])"]
        ExtractLayer["token_representations = results['representations'][layer]"]
        MeanPool["sequence_embedding = token_representations[j, 1:tokens_len-1].mean(0)"]
        StoreEmbed["gene_embeddings[gene_name].append(sequence_embedding)"]
    end
    
    OOMHandler["RuntimeError: 'out of memory'<br/>torch.cuda.empty_cache()<br/>Process one by one<br/>Lines 403-430"]
    
    AverageIsoforms["Stack and mean across isoforms per gene<br/>gene_avg_embedding[gene] = stacked_embeddings.mean(dim=0)<br/>Lines 434-442"]
    
    SavePT["torch.save(torch_embeddings, save_path)<br/>Lines 445-458"]
    
    EnsemblIDs --> FetchIsoforms
    FetchIsoforms --> LoadESM
    LoadESM --> DeviceSelect
    DeviceSelect --> FilterSeq
    FilterSeq --> BatchLoop
    BatchLoop -->|OOM Error| OOMHandler
    BatchLoop --> AverageIsoforms
    OOMHandler --> AverageIsoforms
    AverageIsoforms --> SavePT
```

**ESM Model Selection:** The function accepts any ESM model name via the `model_name` parameter (default: "esm2_t6_8M_UR50D"). Billion-parameter ESM models require FSDP or 80GB GPU [biomni/tool/genomics.py:341-343](). The `layer` parameter specifies which transformer layer to extract embeddings from (default: 6, typically the last layer) [biomni/tool/genomics.py:314-321]().

**Memory-Safe Implementation:** Uses batch_size=1 by default to minimize memory [biomni/tool/genomics.py:319](). On RuntimeError with "out of memory" [biomni/tool/genomics.py:403](), the function calls `torch.cuda.empty_cache()` and falls back to single-sequence processing [biomni/tool/genomics.py:406-430]().

**Sources:** [biomni/tool/genomics.py:284-460](), [biomni/tool/tool_description/genomics.py:606-656]()

### Gene Set Enrichment Analysis

The `gene_set_enrichment_analysis` function wraps `gget.enrichr` for pathway and ontology enrichment:

```mermaid
graph LR
    Genes["genes: list<br/>Gene symbols"]
    Database["database: str<br/>pathway/transcription/ontology<br/>diseases_drugs/celltypes<br/>kinase_interactions"]
    Background["background_list: optional<br/>Background genes"]
    
    GGET["gget.enrichr<br/>Lines 1062"]
    
    Results["Top K pathways<br/>DataFrame with:<br/>rank, path_name, p_val<br/>z_score, combined_score<br/>overlapping_genes, adj_p_val"]
    
    Genes --> GGET
    Database --> GGET
    Background --> GGET
    GGET --> Results
```

**Database Options:** Use `get_gene_set_enrichment_analysis_supported_database_list()` to retrieve all available databases [biomni/tool/genomics.py:1017-1018](). Common options include KEGG_2021_Human (pathway), GO_Biological_Process_2021 (ontology), ChEA_2016 (transcription), etc.

**Sources:** [biomni/tool/genomics.py:974-1088](), [biomni/tool/tool_description/genomics.py:216-282]()

## Chromatin Analysis Tools

### Hi-C Chromatin Interaction Analysis

The `analyze_chromatin_interactions` function identifies enhancer-promoter interactions and TADs from Hi-C data:

```mermaid
graph TB
    HiCFile["hic_file_path<br/>.cool or .hic format"]
    RegElements["regulatory_elements_bed<br/>Enhancers, promoters, CTCF sites"]
    
    LoadCooler["cooler.Cooler<br/>Lines 1125-1129"]
    CheckWeights["Check for balancing weights<br/>Lines 1131-1148"]
    
    LoadBED["pd.read_csv BED file<br/>Lines 1175-1185"]
    
    subgraph InteractionAnalysis["Interaction Analysis (Lines 1199-1266)"]
        MatrixFetch["Get balanced matrix per chromosome<br/>Lines 1211"]
        FilterRegions["Filter enhancers and promoters<br/>Lines 1218-1223"]
        
        LoopPairs["For each enhancer-promoter pair"]
        CalcStrength["interaction_strength = matrix[enh_bin, prom_bin]<br/>Lines 1237"]
        CalcExpected["expected = mean of diagonal at distance<br/>Lines 1244-1246"]
        CalcEnrichment["fold_enrichment = strength / expected<br/>Lines 1249"]
        FilterSig["Keep if fold_enrichment > 2<br/>Lines 1252"]
    end
    
    subgraph TADCalling["TAD Calling (Lines 1299-1377)"]
        InsulationScore["Calculate insulation score<br/>window_size = 5<br/>Lines 1324-1328"]
        ZScore["stats.zscore normalization<br/>Lines 1332"]
        FindMinima["Find local minima as boundaries<br/>Lines 1336-1342"]
        DefineTADs["TADs between boundaries<br/>Lines 1346-1361"]
    end
    
    SaveResults["Save interactions TSV<br/>Save TADs BED<br/>Save contact matrix NPY<br/>Lines 1268-1420"]
    
    HiCFile --> LoadCooler
    LoadCooler --> CheckWeights
    RegElements --> LoadBED
    
    CheckWeights --> InteractionAnalysis
    LoadBED --> InteractionAnalysis
    InteractionAnalysis --> SaveResults
    
    CheckWeights --> TADCalling
    TADCalling --> SaveResults
```

**Input Format:** The function accepts Hi-C data in `.cool` (cooler format) or `.hic` format. Regulatory elements should be provided in BED format with naming convention including "enhancer" or "promoter" keywords [biomni/tool/genomics.py:1182-1185]().

**Interaction Significance:** Interactions are considered significant if fold enrichment over expected (based on genomic distance) is greater than 2 [biomni/tool/genomics.py:1252-1265]().

**TAD Detection:** The function uses a simple insulation score method with window size 5 and local minima detection [biomni/tool/genomics.py:1309-1342](). For production use, specialized tools like tadbit or HiCExplorer are recommended [biomni/tool/genomics.py:1306-1307]().

**Sources:** [biomni/tool/genomics.py:1091-1421](), [biomni/tool/tool_description/genomics.py:283-312]()

### ChIP-seq Peak Calling with MACS2

The `perform_chipseq_peak_calling_with_macs2` function wraps the MACS2 command-line tool:

```mermaid
graph LR
    ChIPFile["chip_seq_file<br/>BAM/BED format"]
    ControlFile["control_file<br/>Input control"]
    
    Params["Parameters:<br/>genome_size: hs/mm/etc<br/>q_value: 0.05 default<br/>output_name: prefix"]
    
    CheckMACS2["Check MACS2 installed<br/>Lines 1679-1692"]
    
    BuildCmd["Build command:<br/>macs2 callpeak -t -c -n -g -q<br/>Lines 1695-1712"]
    
    RunMACS2["subprocess.run<br/>timeout=300s<br/>Lines 1728"]
    
    Outputs["Output files:<br/>*_peaks.narrowPeak<br/>*_summits.bed<br/>*_peaks.xls<br/>Lines 1719-1724"]
    
    ChIPFile --> BuildCmd
    ControlFile --> BuildCmd
    Params --> BuildCmd
    BuildCmd --> CheckMACS2
    CheckMACS2 --> RunMACS2
    RunMACS2 --> Outputs
```

**Output Files:**
- `*_peaks.narrowPeak`: BED6+4 format with peak locations and statistics [biomni/tool/genomics.py:1770-1772]()
- `*_summits.bed`: Summit positions (highest point) of each peak [biomni/tool/genomics.py:1773]()
- `*_peaks.xls`: Tab-delimited detailed information [biomni/tool/genomics.py:1774]()

**Sources:** [biomni/tool/genomics.py:1617-1791](), [biomni/tool/tool_description/genomics.py:342-380]()

### Motif Discovery with HOMER

The `find_enriched_motifs_with_homer` function runs HOMER's findMotifsGenome.pl:

```mermaid
graph TB
    PeakFile["peak_file<br/>BED format"]
    Genome["genome: hg38/mm10/etc<br/>Line 1796"]
    
    Params["Parameters:<br/>motif_length: 8,10,12<br/>num_motifs: 10<br/>threads: 4<br/>background_file: optional"]
    
    BuildCmd["Build HOMER command:<br/>findMotifsGenome.pl<br/>-size 200 -len -S -p<br/>Lines 1850-1867"]
    
    RunHOMER["subprocess.run<br/>Lines 1875"]
    
    ParseResults["Parse results:<br/>homerResults.html<br/>motif*.motif files<br/>knownResults.html<br/>Lines 1886-1948"]
    
    PeakFile --> BuildCmd
    Genome --> BuildCmd
    Params --> BuildCmd
    BuildCmd --> RunHOMER
    RunHOMER --> ParseResults
```

**Search Region:** HOMER searches 200bp centered on each peak by default [biomni/tool/genomics.py:1855-1856](). The function parses motif consensus sequences, matching transcription factors, and p-values from the output files [biomni/tool/genomics.py:1900-1925]().

**Sources:** [biomni/tool/genomics.py:1794-1962](), [biomni/tool/tool_description/genomics.py:381-435]()

### Genomic Region Overlap Analysis

The `analyze_genomic_region_overlap` function uses pybedtools for pairwise region comparisons:

```mermaid
graph TB
    RegionSets["region_sets: list<br/>BED files or coordinate lists"]
    
    CreateBedTools["Create pybedtools.BedTool objects<br/>Lines 1992-2017"]
    
    subgraph PairwiseAnalysis["Pairwise Overlap (Lines 2031-2152)"]
        IntersectU["bedtools[i].intersect(bedtools[j], u=True)<br/>Unique overlapping features<br/>Lines 2038-2041"]
        
        IntersectWO["bedtools[i].intersect(bedtools[j], wo=True)<br/>With overlap width<br/>Lines 2048"]
        
        ParseWidth["Extract overlap width from last column<br/>Lines 2057-2094"]
        
        CalcStats["Calculate:<br/>overlap_bp, pct_of_set1, pct_of_set2<br/>Lines 2099-2112"]
        
        SaveDetail["Save detailed overlaps to BED<br/>Lines 2115-2116"]
    end
    
    Summary["Save summary TSV<br/>Lines 2154-2157"]
    
    RegionSets --> CreateBedTools
    CreateBedTools --> PairwiseAnalysis
    PairwiseAnalysis --> Summary
```

**Input Flexibility:** Each region set can be either a path to a BED file or a list of tuples `(chrom, start, end)` or `(chrom, start, end, name)` [biomni/tool/genomics.py:1969-2015]().

**Overlap Metrics:** For each pair, the function reports:
- Number of overlapping regions from each set
- Total overlap size in base pairs
- Percentage of each set covered by overlaps [biomni/tool/genomics.py:2099-2112]()

**Sources:** [biomni/tool/genomics.py:1965-2164](), [biomni/tool/tool_description/genomics.py:436-460]()

## Comparative Genomics Tools

The `analyze_comparative_genomics_and_haplotypes` function performs whole-genome comparison and haplotype analysis:

```mermaid
graph TB
    SampleFASTAs["sample_fasta_files: list<br/>Whole-genome sequences"]
    RefGenome["reference_genome_path<br/>Reference FASTA"]
    
    CheckBWA["Check BWA availability<br/>Lines 1464-1473"]
    
    subgraph AlignmentWorkflow["Alignment & Variant Calling (Lines 1476-1531)"]
        IndexRef["bwa index reference<br/>Lines 1469"]
        AlignSample["bwa mem alignment<br/>Lines 1492-1497"]
        AlternativeComp["Fallback: Direct sequence comparison<br/>Lines 1506-1527"]
    end
    
    subgraph ComparativeAnalysis["Comparative Analysis (Lines 1533-1570)"]
        IdentifyShared["Identify shared genomic regions"]
        IdentifyUnique["Identify unique genomic regions"]
        WriteComparison["Write comparative_analysis.txt<br/>Lines 1547-1569"]
    end
    
    subgraph HaplotypeAnalysis["Haplotype Analysis (Lines 1572-1604)"]
        GroupSamples["Group samples by haplotype patterns"]
        WriteHaplotypes["Write haplotype_analysis.txt<br/>Lines 1576-1603"]
    end
    
    SampleFASTAs --> CheckBWA
    RefGenome --> CheckBWA
    CheckBWA --> AlignmentWorkflow
    AlignmentWorkflow --> ComparativeAnalysis
    ComparativeAnalysis --> HaplotypeAnalysis
```

**BWA Integration:** If BWA is available, the function performs proper alignment and variant calling [biomni/tool/genomics.py:1490-1504](). Otherwise, it falls back to direct sequence comparison using BioPython's SeqIO [biomni/tool/genomics.py:1509-1527]().

**Output Files:**
- `comparative_analysis.txt`: Shared and unique genomic regions [biomni/tool/genomics.py:1547-1569]()
- `haplotype_analysis.txt`: Haplotype grouping and characteristics [biomni/tool/genomics.py:1576-1603]()

**Sources:** [biomni/tool/genomics.py:1424-1614](), [biomni/tool/tool_description/genomics.py:313-341]()

## Dependencies and External Tools

**Figure 7: Python Package Dependencies**

```mermaid
graph TB
    GenomicsPy["biomni/tool/genomics.py"]
    
    GenomicsPy --> Import1["import scanpy as sc (Line 10)"]
    GenomicsPy --> Import2["import esm (Line 4)"]
    GenomicsPy --> Import3["import gget (Line 5)"]
    GenomicsPy --> Import4["import gseapy (Line 6)"]
    GenomicsPy --> Import5["import torch (Line 11)"]
    GenomicsPy --> Import6["from pybiomart import Dataset (Line 12)"]
    GenomicsPy --> Import7["from biomni.llm import get_llm (Line 15)"]
    
    subgraph DynamicImports["Dynamic Imports (Conditional)"]
        popvImport["import popv (Line 43, inside function)"]
        scviImport["import scvi (Line 804, inside function)"]
        harmonyImport["from harmony import harmonize (Line 844)"]
        coolerImport["import cooler (Line 1110)"]
        pybedtoolsImport["import pybedtools (Line 1986)"]
    end
    
    Import1 --> scanpyFuncs["scRNA-seq functions"]
    Import2 --> esmFunc["generate_gene_embeddings_with_ESM_models"]
    Import3 --> ggetFuncs["get_rna_seq_archs4, gene_set_enrichment_analysis"]
    Import4 --> gseapyFunc["gene_set_enrichment_analysis"]
    Import5 --> torchFuncs["ESM embeddings, GPU management"]
    Import6 --> biomartFunc["interspecies_gene_conversion"]
    Import7 --> llmFunc["annotate_celltype_scRNA"]
```

**Sources:** [biomni/tool/genomics.py:1-16]()

### Command-Line Tool Integration

**Table 3: External CLI Tool Integrations**

| CLI Tool | Function | Availability Check | Invocation | Fallback Strategy |
|----------|----------|-------------------|------------|-------------------|
| MACS2 | `perform_chipseq_peak_calling_with_macs2` | Lines 1679-1692 | `subprocess.run(['macs2', 'callpeak', ...])` (Line 1728) | Error message, return log |
| HOMER | `find_enriched_motifs_with_homer` | Implicit in subprocess call | `subprocess.run(['findMotifsGenome.pl', ...])` (Line 1875) | Error message, return log |
| BWA | `analyze_comparative_genomics_and_haplotypes` | Lines 1465-1473 | `subprocess.run(['bwa', 'mem', ...])` (Lines 1492-1497) | Direct sequence comparison (Lines 1506-1527) |

**Sources:** [biomni/tool/genomics.py:1465-1473, 1679-1692, 1873-1875]()

### Command-Line Tool Integration

| CLI Tool | Function | Installation Check | Usage Pattern |
|----------|----------|-------------------|---------------|
| MACS2 | `perform_chipseq_peak_calling_with_macs2` | Lines 1679-1692 | `subprocess.run(['macs2', 'callpeak', ...])` |
| HOMER | `find_enriched_motifs_with_homer` | Lines 1873-1875 | `subprocess.run(['findMotifsGenome.pl', ...])` |
| BWA | `analyze_comparative_genomics_and_haplotypes` | Lines 1465-1473 | `subprocess.run(['bwa', 'mem', ...])` with fallback |

**Error Handling:** All CLI tool integrations check for tool availability before execution and provide fallback strategies or clear error messages [biomni/tool/genomics.py:1465-1473, 1679-1692]().

**Sources:** [biomni/tool/genomics.py:1465-1473, 1679-1692, 1873-1875]()

## Common Patterns and Usage

### LLM Integration for Biological Reasoning

The `annotate_celltype_scRNA` function demonstrates LLM integration:

```python
# Lines 544-546
llm = get_llm(llm)  # From biomni.llm
prompt = PromptTemplate(input_variables=["cluster_info"], template=prompt_template)
chain = prompt | llm
```

The LLM is used iteratively to predict cell types with validation against the CZI Census ontology [biomni/tool/genomics.py:554-580](). This pattern of LLM-guided biological annotation can be extended to other tasks.

**Sources:** [biomni/tool/genomics.py:463-597]()

### Isolated Environment Execution

The `annotate_celltype_with_panhumanpy` function uses subprocess to execute code in a separate conda environment:

```python
# Lines 655-659
PANHUMANPY_ENV = "panhumanpy_env"
if not conda_env_exists(PANHUMANPY_ENV):
    create_panhumanpy_env(PANHUMANPY_ENV)

# Lines 773-774
run_cmd = ["conda", "run", "-n", PANHUMANPY_ENV, "python", script_path]
subprocess.run(run_cmd, check=True)
```

This pattern isolates conflicting dependencies while maintaining integration with the main agent system [biomni/tool/genomics.py:600-795]().

**Sources:** [biomni/tool/genomics.py:600-795]()

### Progress Logging and Result Tracking

All functions return a multi-line string documenting execution steps:

```python
# Lines 45-46, 103-105 (example pattern)
steps = []
steps.append("Starting unsupervised cell type transfer using popV")
# ... processing ...
return "\n".join(steps)
```

This provides transparency for the agent's reasoning process and makes results reproducible [biomni/tool/genomics.py:45-105]().

**Sources:** [biomni/tool/genomics.py:18-2164]()

---