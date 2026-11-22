"""Gene expression reconstruction of a trained scVI model on the evaluation datasets.

usage (from bash/):
  python ../eval_scripts/LDVAE_eval.py <eval data dir> <model training adata> <model dir>
      <output dir> <seed> <AR: T|F> <latent dim> <atlas cell count> <alpha> <epochs> [bins]
e.g. for a checkpoint downloaded from Zenodo:
  python ../eval_scripts/LDVAE_eval.py ../data/sctab \
      ../data/sctab/bloodbase_100_atlas_seed42_ARTrue_train_adata_2000_2000HVGs.h5ad \
      <model dir> ../result/test/sctab/scVI-reconstruction-evals-fig2 42 T 64 100 0.0001 300
The training adata is written by main.py when the model is trained, or by
bash/run_scvi_weights.sh. bash/run_scvi.sh runs this script for the
models it trains.
"""
import scanpy as sc
import json
import os
import pickle
import sys

from LDVAE_eval_class import LDVAE_eval


final_eval_path = sys.argv[1]
model_adata = sys.argv[2]
model_path = sys.argv[3]
eval_output_dir = sys.argv[4]
seed = int(sys.argv[5])
ARtype = sys.argv[6]
latent_dim = sys.argv[7]
Atlas_cell_count = sys.argv[8]
alpha = sys.argv[9]
epoch_num = sys.argv[10]
bins = sys.argv[11] if len(sys.argv) > 11 else "100"

balancing_method = None
# model_path = <run_dir>/best; the run's config.json holds the balancing method
config_file = os.path.join(os.path.dirname(os.path.normpath(model_path)), "config.json")
if os.path.exists(config_file):
    with open(config_file) as f:
        balancing_method = json.load(f)["balancing_method"] or "Original Combination"
elif "class_balancing" in model_path:
    balancing_method = "class_balancing"
elif "geometric_sketching" in model_path:
    balancing_method = "geometric_sketching"
else:
    balancing_method = "Original Combination"
    # raise ValueError("Invalid balancing method")

# check if eval_output_dir exists, if not create it
if not os.path.exists(eval_output_dir):
    os.makedirs(eval_output_dir)

model_adata = sc.read_h5ad(model_adata)
# Load datasets
sctab_train_10pct_heart_nohematopoietic = sc.read(os.path.join(final_eval_path, "sctab_train_10pct_heart_nohematopoietic.h5ad"))
sctab_train_10pct_kidney_nohematopoietic = sc.read(os.path.join(final_eval_path, "sctab_train_10pct_kidney_nohematopoietic.h5ad"))
Neurons_H1830002_10Ksubset = sc.read(os.path.join(final_eval_path, "Neurons_H1830002_10Ksubset.h5ad"))

# Initialize LDVAE_eval object for evaluation
ldvae_eval = LDVAE_eval(model_path, model_adata)


# Run evaluations using LDVAE_eval (NOT scvi_model)
print(f"Running evaluation for Model {model_path}")

print("Reconstruction Evaluation")
print("scTab Heart")
ldvae_eval.get_reconstruction_r2(sctab_train_10pct_heart_nohematopoietic, ['cell_type'], 'scTab_Heart_reconstruction', 'All')

print("scTab Kidney")
ldvae_eval.get_reconstruction_r2(sctab_train_10pct_kidney_nohematopoietic, ['cell_type'], 'scTab_Kidney_reconstruction', 'All')

print("Neurons")
ldvae_eval.get_reconstruction_r2(Neurons_H1830002_10Ksubset, ['supercluster_term'], 'Neurons_reconstruction', 'All')

# Save results
output_file = os.path.join(
    eval_output_dir,
    f'Reconstruction_seed_{seed}_ARtype_{ARtype}_latent_dim_{latent_dim}_'
    f'Atlas_cell_count_{Atlas_cell_count}_alpha_{alpha}_bins_{bins}_epoch_{epoch_num}_'
    f'{balancing_method}_bloodbase_eval.pkl',
)
with open(output_file, 'wb') as f:
    pickle.dump(ldvae_eval.evals, f)

print(f"Evaluation results saved to {output_file}")
