#!/usr/bin/env python3
"""Build a disease-risk report from an SNP file."""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
import scanpy as sc
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

NEO4J_URI = os.environ["BRAINTWIN_NEO4J_URL"]
NEO4J_AUTH = (os.environ.get("BRAINTWIN_NEO4J_USER", "neo4j"), os.environ.get("BRAINTWIN_NEO4J_PASSWORD", ""))
LLM_BASE_URL = os.environ["BRAINTWIN_LLM_URL"].rstrip("/") + "/v1"
LLM_API_KEY = os.environ.get("BRAINTWIN_LLM_API_KEY", "EMPTY")
LLM_MODEL = os.environ["BRAINTWIN_MODEL_ID"]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA

SC_DATA_PATH = os.environ.get(
    "BRAINTWIN_STAB2_H5AD",
    str(DATA / "singlecell" / "stab2_hs_anno.h5ad"),
)

REPORT_DIR = "Report_files"
os.makedirs(REPORT_DIR, exist_ok=True)

CANDIDATE_DISEASE = {
    "Degenerative diseases": ["Alzheimer's disease", "Parkinson's disease", "Lewy body disease", "Dementia"],
    "Mental disorders": ["Anxiety disorder", "Panic disorder", "Bipolar disorder (MTAG)", "Major depressive disorder", 
                        "Schizophrenia (MTAG)", "Obsessive-compulsive disorder", "Tourette syndrome", 
                        "Attention deficit hyperactivity disorder", "Autism"],
    "fetal diseases": ["Neonatal abstinence syndrome", "Kleine-Levin syndrome"],
    "brain tumor": ["Glioma", "Neuroblastoma"],
    "brain event": ["Stroke", "Cerebral aneurysm"],
    "neural diseases": ["Epilepsy", "Neuromyelitis optica"]
}

PRIMARY_KEY = {
    "SNP": "SNP_Name",
    "Gene": "Gene_Name",
    "Trait": "Trait_Name",
    "CellType": "CellType_Name",
    "BrainRegion": "Region_Name",
    "BrainSubRegion": "Subregion_Name",
    "GOTerm": "Go_Term_Name"
}


def read_snp_file(file_path):
    """Read an SNP file."""
    logger.info(f"Reading SNP file: {file_path}")
    with open(file_path) as f:
        lines = f.readlines()

    header_line = None
    for i, line in enumerate(lines):
        if line.startswith("#") and not line.startswith("##"):
            header_line = i
            break

    pvar_df = pd.read_csv(file_path,
                          sep="\t",
                          header=header_line,
                          comment=None)

    pvar_df.columns = [col.lstrip("#") for col in pvar_df.columns]
    pvar_df.index = pvar_df['ID']
    logger.info(f"Read {len(pvar_df)} SNPs.")
    return pvar_df


def get_all_snps_from_db(driver):
    """Load SNP names stored in the graph."""
    logger.info("Loading SNP names from Neo4j.")
    query = """
    MATCH (s:SNP)-[]-(t:Trait)
    RETURN DISTINCT s.SNP_Name AS name
    ORDER BY name
    """
    with driver.session() as session:
        result = session.run(query)
        all_snp = [record["name"] for record in result]
    logger.info(f"The graph contains {len(all_snp)} SNPs.")
    return all_snp


def check_snp_disease_association(disease, personal_snp, all_snp, driver):
    """Test overlap between the sample SNPs and one disease."""
    query = """
    MATCH (s:SNP)-[]-(t:Trait {Trait_Name: $trait_name})
    RETURN s.SNP_Name AS name
    """
    with driver.session() as session:
        result = session.run(
            query,
            trait_name=disease.replace("'", "\'")
        )
        result = [record["name"] for record in result]

    disease_snp = set(result)
    overlap = disease_snp & personal_snp
    
    if len(disease_snp) == 0:
        pval = 1.0
    else:
        pval = hypergeom.sf(len(overlap) - 1, len(all_snp), len(disease_snp), len(personal_snp))
    
    return (disease, ";".join(list(overlap)), f"{len(overlap)}/{len(disease_snp)}", pval)


def get_genes_from_snps(snp_list, driver):
    """Map SNPs to genes."""
    query = """
    MATCH (s:SNP)-[]->(g:Gene)
    WHERE s.SNP_Name IN $snp_list
    WITH g, COLLECT(DISTINCT s.SNP_Name) AS snps
    RETURN g.Gene_Name AS g_name, 
           size(snps) AS covered_snp_count,
           snps AS covered_snps
    ORDER BY covered_snp_count DESC;
    """
    with driver.session() as session:
        result = session.run(query, snp_list=snp_list)
        hit_gene_list = [[record["g_name"], record["covered_snp_count"], ";".join(record["covered_snps"])] 
                         for record in result]
    return pd.DataFrame(hit_gene_list, columns=['gene', 'snp_num', 'snp'])


def get_go_terms_from_genes(gene_list, driver):
    """Map genes to GO terms."""
    query = """
    MATCH (g:Gene)-[]->(go:GOTerm)
    WHERE g.Gene_Name IN $gene_list
    WITH go, COLLECT(DISTINCT g.Gene_Name) AS genes
    RETURN go.Go_Term_Name AS GO_ID, 
           size(genes) AS covered_gene_count,
           genes AS covered_genes
    ORDER BY covered_gene_count DESC;
    """
    with driver.session() as session:
        result = session.run(query, gene_list=gene_list)
        hit_GO_list = [[record["GO_ID"], record["covered_gene_count"], ";".join(record["covered_genes"])] 
                       for record in result]
    return pd.DataFrame(hit_GO_list, columns=['go', 'gene_num', 'gene'])


def get_celltypes_from_genes(gene_list, driver):
    """Map genes to cell types."""
    query = """
    MATCH (g:Gene)-[]->(ct:CellType)
    WHERE g.Gene_Name IN $gene_list
    WITH ct, COLLECT(DISTINCT g.Gene_Name) AS genes
    RETURN ct.CellType_Name AS ct_name, 
           size(genes) AS covered_gene_count,
           genes AS covered_genes
    ORDER BY covered_gene_count DESC;
    """
    with driver.session() as session:
        result = session.run(query, gene_list=gene_list)
        hit_celltype_list = [[record["ct_name"], record["covered_gene_count"], ";".join(record["covered_genes"])] 
                            for record in result]
    return pd.DataFrame(hit_celltype_list, columns=['celltype', 'gene_num', 'gene'])


def compute_dotplot_data(adata, groupby):
    """Summarize scores by group."""
    groups = adata.obs[groupby].unique().tolist()
    result = pd.DataFrame(index=["z_score", "fraction"], columns=groups, dtype=float)

    for grp in groups:
        idx = adata.obs[groupby] == grp
        X = adata.obs.score[idx]

        result.loc["z_score", grp] = X.mean(axis=0)

        result.loc["fraction", grp] = (X > 0).mean(axis=0)
    
    result.loc["z_score"] = result.loc["z_score"].apply(
        lambda x: (x - result.loc["z_score"].mean()) / result.loc["z_score"].std()
    )
    return result.transpose()


def generate_summary_section(df_result, llm):
    """Write the summary section."""
    logger.info("Writing the summary.")
    summary_prompt = f"""
You are an expert in human genetics, statistical genomics, GWAS interpretation, and disease risk communication.

You will receive:
1. A Markdown table summarizing enrichment p-values across diseases.

Your tasks:
- Write a clear, concise SUMMARY section for a genetic risk assessment report.
- Interpret the key findings:
  * Which diseases show statistically significant enrichment (FDR < 0.05)?
  * Which are borderline or weak?
  * Which show no genetic evidence of elevated risk?
- Provide an overall narrative explaining the genetic risk landscape.
- Highlight notable disease categories (e.g., neurodegenerative, psychiatric, immune-related).
- Keep the writing objective, non-alarming, and scientifically accurate.

Return the final output in **Markdown format** with this structure:

# Summary

<LLM-generated summary paragraphs here>

---

Input Table:
{df_result.loc[:, ['disease', 'hit_snp_number', 'pval_adjusted']].to_markdown(index=False)}
"""
    summary = llm.invoke(summary_prompt)
    return summary.content


def generate_disease_description(disease, llm):
    """Describe one disease."""
    logger.info(f"Describing {disease}.")
    prompt = f"Please describe what is {disease} briefly."
    response = llm.invoke(prompt)
    return response.content


def generate_genetic_effects(disease, df_hit_gene, df_hit_go, llm):
    """Interpret the genetic evidence."""
    logger.info(f"Interpreting genetic evidence for {disease}.")
    genetic_prompt = f"""
You are a professional expert in human genetics, neurobiology, and disease mechanism interpretation. Your task is to produce a clear, 
clinically useful explanation of how an individual's genetic variants (SNPs and associated genes) may influence their risk for a specific disease.

You will receive:
- A disease name
- A set of risk SNPs carried by the individual
- The genes associated with these SNPs
- Functional annotations for these gene.

Your task:
1. Summarize the overall genetic signal for this disease in this individual.
2. Explain the critical genes and their biological function in the context of this disease.
3. Avoid deterministic statements. This is risk interpretation, not diagnosis.

Input:
disease: {disease}
associated_gene: {df_hit_gene.loc[0:50, :].to_markdown(index=False)}
associated_go: {df_hit_go.loc[0:50, :].to_markdown(index=False)}
"""
    genetic_response = llm.invoke(genetic_prompt)
    return genetic_response.content


def generate_cellular_effects(disease, hit_gene_list, score_distri_region_fine, score_distri_celltype, llm):
    """Interpret cell-type and region scores."""
    logger.info(f"Interpreting cellular evidence for {disease}.")
    cellular_prompt = f"""
You are an expert in neuroscience, single-cell biology, and neurodegeneration. 
I will provide you with weighted scores that quantify the likelihood that specific brain regions and cell types are affected, based on a set of genes associated with {disease}.  
Your task is to analyze these scores and produce a mechanistic interpretation of the affected brain regions, vulnerable cell types, and downstream functional consequences.

------------------------------------
Input data:
• Gene list used to compute scores:
{hit_gene_list}

• Weighted scores for brain regions (higher = more likely affected):
{score_distri_region_fine.z_score.to_markdown()}

• Weighted scores for cell types (higher = more likely affected):
{score_distri_celltype.z_score.to_markdown()}
------------------------------------

Please analyze the data according to the following requirements:

1. Identify the most vulnerable cell types, and explain what aspects of their molecular function or physiology make them susceptible to disruptions implied by the gene list.

2. Identify which brain regions are most likely to be affected, and explain the biological relevance of these regions in the context of disease mechanisms 

3. Integrate the brain-region and cell-type analyses to infer potential pathophysiological processes, such as synaptic dysfunction, impaired plasticity, metabolic stress, myelin changes, 
   neuroinflammation, or neurotransmission imbalance.

4. Based on the gene list and their known biological roles, infer which molecular pathways or functional modules are most likely to be impacted. Provide a mechanistic interpretation linking genes → cell vulnerability → region-level dysfunction → potential behavioral or cognitive outcomes.

The output should be structured, precise, and phrased in professional scientific language.
"""
    cellular_response = llm.invoke(cellular_prompt)
    return cellular_response.content


def generate_imaging_correlates(cellular_response, llm):
    """Suggest imaging correlates."""
    logger.info("Writing imaging correlates.")
    image_prompt = f"""
Based on {cellular_response}, please extend the analysis by generating hypothesis-driven suggestions about potential MRI-relevant changes that could *theoretically* be observed in this patient. 

Your task is to identify MRI-detectable structural or functional changes that clinicians might observe, and to suggest imaging markers that should be monitored.

The output should be written in professional scientific language. These should not be framed as interpretations or diagnoses, since no actual neuroimaging data are provided. Instead, focus on mechanistic predictions and possible imaging correlates that could be explored in future studies.
"""
    image_response = llm.invoke(image_prompt)
    return image_response.content


def generate_disease_report(disease, hit_snp_list, driver, adata_raw, llm, report_dir, disease_idx):
    """Write the section for one disease."""
    logger.info(f"Analyzing {disease}.")
    
    df_hit_gene = get_genes_from_snps(hit_snp_list, driver)
    logger.info(f"Found {len(df_hit_gene)} genes.")
    
    if len(df_hit_gene) == 0:
        logger.warning(f"No genes for {disease}. Skipping the detailed section.")
        return ""
    
    hit_gene_list = list(df_hit_gene.gene)
    
    df_hit_go = get_go_terms_from_genes(hit_gene_list, driver)
    logger.info(f"Found {len(df_hit_go)} GO terms.")
    
    disease_description = generate_disease_description(disease, llm)
    
    genetic_effects = generate_genetic_effects(disease, df_hit_gene, df_hit_go, llm)
    
    logger.info("Scoring genes in the single-cell data.")
    if "score"  in adata_raw.obs.columns:
        del adata_raw.obs["score"]
    sc.tl.score_genes(adata_raw, gene_list=hit_gene_list)
    
    score_distri_celltype = compute_dotplot_data(adata_raw, "cell_type").sort_values(by="z_score", ascending=False)
    score_distri_region = compute_dotplot_data(adata_raw, "region").sort_values(by="z_score", ascending=False)
    score_distri_region_fine = compute_dotplot_data(adata_raw, "region_fine").sort_values(by="z_score", ascending=False)
    
    cellular_response = generate_cellular_effects(disease, hit_gene_list, score_distri_region_fine, score_distri_celltype, llm)
    
    logger.info(f"Saving score plots for {disease}.")
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    sc.pl.dotplot(adata_raw, "score", groupby="cell_type", show=False, ax=axes[0], 
                  title="Cell type", standard_scale='var')
    sc.pl.dotplot(adata_raw, "score", groupby="region", show=False, ax=axes[1], 
                  title="Brain Region", standard_scale='var')
    sc.pl.dotplot(adata_raw, "score", groupby="region_fine", show=False, ax=axes[2], 
                  title="Brain SubRegion", standard_scale='var')
    plt.tight_layout()
    image_path = os.path.join(report_dir, f"{disease}_scores.png")
    plt.savefig(image_path)
    plt.close()
    logger.info(f"Saved the plot to {image_path}.")
    
    image_response = generate_imaging_correlates(cellular_response, llm)
    
    disease_report = f"""

## {disease_idx}. {disease}

### Description
{disease_description}

### Genetic effects
{genetic_effects}

#### Genes associated with carried snps (TOP 50)
{df_hit_gene.loc[0:50, :].to_markdown(index=False)}

#### GO terms associated with these genes (TOP 50)
{df_hit_go.loc[0:50, :].to_markdown(index=False)}

### Cellular effects
{cellular_response}

<div>
  <img src="{disease}_scores.png" width="800">
</div>

### Expected imaging correlates
{image_response}

---
"""
    return disease_report


def generate_snp_report(snp_file_path, output_path=None):
    """Write the SNP report. output_path defaults to Report_files/report.md."""
    logger.info("=" * 60)
    logger.info("Starting the SNP report.")
    logger.info("=" * 60)
    
    logger.info("Connecting to Neo4j.")
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    logger.info("Creating the chat client.")
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL
    )
    
    if output_path is None:
        output_path = os.path.join(REPORT_DIR, "report.md")
    else:
        if os.path.isdir(output_path):
            REPORT_DIR = output_path
            output_path = os.path.join(output_path, "report.md")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    pvar_df = read_snp_file(snp_file_path)
    
    all_snp = get_all_snps_from_db(driver)
    
    logger.info("Keeping SNPs that are present in the graph.")
    pvar_df = pvar_df.loc[pvar_df.index.isin(all_snp), :]
    logger.info(f"{len(pvar_df)} SNPs remain.")
    
    if len(pvar_df) == 0:
        logger.error("No SNPs remain, so no report was written.")
        driver.close()
        return
    
    personal_snp = set(pvar_df.index)
    
    logger.info("Testing disease overlap.")
    result = []
    for disease_list in CANDIDATE_DISEASE.values():
        for disease_name in disease_list:
            result.append(check_snp_disease_association(disease_name, personal_snp, all_snp, driver))
    
    df_result = pd.DataFrame(result, columns=['disease', 'snp', 'hit_snp_number', 'pval'])
    
    logger.info("Applying FDR correction.")
    reject, pvals_corrected, _, _ = multipletests(df_result['pval'], method='fdr_bh')
    df_result['pval_adjusted'] = pvals_corrected
    df_result['significant'] = reject
    df_result = df_result.sort_values('pval_adjusted', ascending=True)
    
    significant_diseases = df_result[df_result['significant'] == True].copy()
    logger.info(f"Found {len(significant_diseases)} diseases with FDR < 0.05.")
    
    if len(significant_diseases) == 0:
        logger.warning("No disease passed FDR < 0.05.")
        significant_diseases = df_result.head(1)
        logger.info(f"Using the smallest p value: {significant_diseases.iloc[0]['disease']}")
    
    report = generate_summary_section(df_result, llm)
    report += f"\n\n{df_result.loc[:, ['disease', 'hit_snp_number', 'pval_adjusted', 'snp']].to_markdown(index=False)}\n"
    
    logger.info("Loading single-cell data.")
    adata = sc.read_h5ad(SC_DATA_PATH)
    adata_raw = adata.raw.to_adata()
    del adata
    logger.info("Single-cell data loaded.")
    
    report += "\n\n# Analysis of risk diseases\n"
    
    disease_idx = 1
    for idx, row in significant_diseases.iterrows():
        disease = row['disease']
        hit_snp_list = row['snp'].split(";") if row['snp'] else []
        
        if len(hit_snp_list) == 0:
            logger.warning(f"No SNPs for {disease}. Skipping.")
            continue
        
        disease_report = generate_disease_report(
            disease, hit_snp_list, driver, adata_raw, llm, 
            REPORT_DIR, disease_idx
        )
        report += disease_report
        disease_idx += 1
    
    
    
    logger.info(f"Writing the report to {output_path}.")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info("=" * 60)
    logger.info("Report finished.")
    logger.info("=" * 60)
    
    driver.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Write an SNP disease-risk report.")
    parser.add_argument("snp_file_path", type=str, help="Path to the SNP file.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output report path.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.snp_file_path):
        logger.error(f"SNP file was not found: {args.snp_file_path}")
        sys.exit(1)
    
    generate_snp_report(args.snp_file_path, args.output)

