#!/bin/bash
# AR resampling weights of the scVI AR models during training.
#
# usage: bash run_scvi_weights.sh <stage>
#   stage: weights      resampling weight of every training cell, per checkpoint
#          diagnostics  normalized entropy and ESS of these weights, per checkpoint
#          umap         UMAP of all training cells coloured by the weights
#                       (atlas count 10000, seed 42, epochs 0 80 160 240)
#
# Models: the scVI AR models of run_scvi.sh (method AR: checkpoint every
# 20 epochs), ../saved_models/scvi/sctab/<hash>/{epoch<N>,best}/; the weights are
# computed as in training (posterior mean, max over latent dimensions, bins 100,
# alpha 1e-4). Checkpoints: epochs 0, 20, ..., 280 and best.
# Output:
#   weights, diagnostics  ../result/scvi_weights/{resampling_weights,weight_diagnostics}/sctab/<seed>/scvi/
#                         (read by eval_scripts/plot_weight_diagnostics.ipynb and
#                         plot_resampling_weight_correlation.ipynb; the weights of seed 42,
#                         atlas 10000 also by plot_weight_histogram.py, Fig. 2D)
#   umap                  ../figures/umap_w_annotated/sctab/seed<seed>/high_dim/
#
# The jobs run one at a time: every call rewrites the training data file
# ../data/sctab/bloodbase_<atlas>_atlas_seed<seed>_ARTrue_train_adata_2000_2000HVGs.h5ad
# (same cells and genes as in training).
#
# Optional environment variables:
#   SEEDS="42"             only these seeds (default 42 43 44 45 46; umap: 42)
#   ATLAS_COUNTS="100"     only these atlas cell counts (default 0 1 10 100 1000 10000 50000; umap: 10000)
#   DRY_RUN=1              print the commands instead of running them
#   SCAR_VENV=~/env/scar-env  activate this virtualenv (after module load
#                          python/3.10) instead of the conda env scar-env
#   WANDB_MODE=offline     passed on to wandb (main.py logs in to wandb in every mode)

stage=$1

case "$stage" in
    weights|diagnostics|umap) ;;
    *) sed -n '2,30p' "$0"; exit 1 ;;
esac

cd "$(dirname "$0")" || exit 1

if [ -z "$DRY_RUN" ]; then
    if [ -n "$SCAR_VENV" ]; then
        module load python/3.10 || exit 1
        source "$SCAR_VENV/bin/activate" || exit 1
    else
        eval "$(conda shell.bash hook)" && conda activate scar-env || exit 1
    fi
fi

## Settings of the scVI AR models (as run_scvi.sh) ##
data='sctab'
latent_dim=64
num_epoch=300
batch_size=4096
lr=0.00005
bins=100
alpha=0.0001
hvg_count=2000

## Grids ##
if [ "$stage" == 'umap' ]; then
    seeds=(42); atlas_counts=(10000); tracked_epochs=(0 80 160 240)
else
    seeds=(42 43 44 45 46); atlas_counts=(0 1 10 100 1000 10000 50000)
    tracked_epochs=(0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 best)
fi
[ -n "$SEEDS" ] && read -ra seeds <<< "$SEEDS"
[ -n "$ATLAS_COUNTS" ] && read -ra atlas_counts <<< "$ATLAS_COUNTS"

# umap runs from the repository root: scanpy saves its figures to ./figures/ and
# the other umap figures go to <root>/figures/
if [ "$stage" == 'umap' ]; then
    cd .. || exit 1
    root='./'; main='main.py'; log_dir=bash/log/$(date '+%Y%m%d')
else
    root='../'; main='../main.py'; log_dir=log/$(date '+%Y%m%d')
    weights_dir='../result/scvi_weights'
fi
mkdir -p "$log_dir"
failed=0

## main.py arguments of one AR model (they give its run directory) ##
for seed in "${seeds[@]}"; do
    for atlas in "${atlas_counts[@]}"; do
        model=(--root "$root" --data "$data" --out_path "${root}saved_models/" --model_name scvi
               --model AR --AR True --w_update max_w --bins $bins --alpha $alpha
               --latent_dim $latent_dim --num_epoch $num_epoch --batch_size $batch_size --lr $lr
               --hvg_count $hvg_count --ood True --seed "$seed" --atlas_count "$atlas" --test True)
        for t in "${tracked_epochs[@]}"; do
            name="${stage}-atlas${atlas}-seed${seed}-epoch${t}"
            case $stage in
                weights|diagnostics)
                    if [ "$stage" == 'weights' ]; then
                        sub=resampling_weights; flags=(--store_resampling_weights True --resampling_weight_path)
                    else
                        sub=weight_diagnostics; flags=(--store_weight_diagnostics True --weight_diagnostics_path)
                    fi
                    dir="${weights_dir}/${sub}/${data}/${seed}/scvi"
                    file="${sub}_${data}_AR_epoch${num_epoch}_seed${seed}_scvi_ldim${latent_dim}_atlascount${atlas}_trackedepoch${t}.csv"
                    flags+=("${dir}/${file}")
                    [ -z "$DRY_RUN" ] && mkdir -p "$dir" ;;
                umap)
                    flags=(--plot_umap_annotated_with_w True) ;;
            esac
            cmd=(python -u "$main" "${model[@]}" --tracked_epoch "$t" "${flags[@]}")
            if [ -n "$DRY_RUN" ]; then
                printf '%q ' "${cmd[@]}"; echo
                continue
            fi
            echo "${name}"
            "${cmd[@]}" > "${log_dir}/${name}.txt" 2> "${log_dir}/${name}.err" \
                || { echo "Error in ${name}, see ${log_dir}/${name}.err"; failed=$((failed + 1)); }
        done
    done
done

[ -n "$DRY_RUN" ] && exit 0
if [ "$failed" -gt 0 ]; then
    echo "${failed} jobs failed (logs in ${log_dir})"
    exit 1
fi
echo "All ${stage} jobs finished."
