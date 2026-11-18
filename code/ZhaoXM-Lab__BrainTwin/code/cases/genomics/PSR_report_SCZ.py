# %%
from cyvcf2 import VCF
import pandas as pd
import os
from tqdm import tqdm

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import DATA

PSR_dir = str(DATA / "genomics" / "prs")
vsf_dir = str(DATA / "genomics")

# %%
def load_pgs(pgs_file):
    pgs = pd.read_csv(
        pgs_file,
        sep="\t",
        comment="#"
    )
    if "chr_name" in pgs.columns and "chr_position" in pgs.columns:
        # Keep the columns used for scoring.
        pgs = pgs[["chr_name", "chr_position", "effect_allele", "effect_weight"]]
        pgs.dropna(subset=["chr_name", "chr_position"], inplace=True)
    else:
        pgs = pgs[["hm_chr", "hm_pos", "effect_allele", "effect_weight"]]
        pgs.rename(columns={"hm_chr": "chr_name", "hm_pos": "chr_position"}, inplace=True)
        pgs.dropna(subset=["chr_name", "chr_position"], inplace=True)
        pgs.chr_name = pgs.chr_name.astype(int).astype(str)
        pgs.chr_position = pgs.chr_position.astype(int).astype(str)
    # Index: chr:pos -> (effect_allele, beta)
    pgs_dict = {
        f"{row.chr_name}:{row.chr_position}": (row.effect_allele, row.effect_weight)
        for _, row in pgs.iterrows()
    }
    return pgs_dict

# %%
def calculate_prs(vcf, pgs_dict):
    prs = 0.0
    matched_snps = 0
    for variant in (vcf):
        key = f"{variant.CHROM.strip('chr')}:{variant.POS}"

        if key not in pgs_dict:
            continue

        effect_allele, beta = pgs_dict[key]

        # REF / ALT
        alleles = [variant.REF] + variant.ALT

        if effect_allele not in alleles:
            continue

        # Genotype, for example (0, 1).
        gt = variant.genotypes[0][:2]

        # Dosage.
        dosage = sum(
            1 for g in gt
            if g >= 0 and alleles[g] == effect_allele
        )

        prs += dosage * beta
        matched_snps += 1

    return prs, matched_snps

# %% [markdown]
# ## load PRS files

# %%
pgs_file = "AD_PGS003958_hmPOS_GRCh38.txt"
AD_pgs_dict = load_pgs(os.path.join(PSR_dir, pgs_file))

# %%
pgs_file = "ASD_PGS000327_hmPOS_GRCh38.txt"
ASD_pgs_dict = load_pgs(os.path.join(PSR_dir, pgs_file))

# %%
pgs_file = "SCZ_PGS000133_hmPOS_GRCh38.txt"
SCZ_pgs_dict = load_pgs(os.path.join(PSR_dir, pgs_file))

# %%
pgs_file = "MDD_PGS003578_hmPOS_GRCh38.txt"
MDD_pgs_dict = load_pgs(os.path.join(PSR_dir, pgs_file))

# %% [markdown]
# ## scoring samples


# %%
_dir = "SCZ_subset2_indiv_gvcf"
vcf_list = [file for file in os.listdir(vsf_dir+"/"+_dir) if file.endswith(".vcf.gz")]
result_dict = {}
for file in tqdm(vcf_list):
    sample_id = file.split(".")[0]
    scores = []
    matched_numbers = []
    for pgs_dict in [AD_pgs_dict, ASD_pgs_dict, SCZ_pgs_dict, MDD_pgs_dict]:
        vcf = VCF(os.path.join(vsf_dir, _dir, file))
        prs, matched_snps = calculate_prs(vcf, pgs_dict)
        scores.append(prs)
        matched_numbers.append(matched_snps)
    result_dict[sample_id] = {"scores": scores, "matched_numbers": matched_numbers}
    
# %%
import json
with open("SZC_subset2.json", "w") as f:
    json.dump(result_dict, f)



