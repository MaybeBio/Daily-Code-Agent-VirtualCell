#!/bin/bash
# scVI on scTab: standard and AR training, the class-balancing and
# geometric-sketching baselines, and evaluation of gene expression
# reconstruction and cell type classification.
#
# usage: bash run_scvi.sh <train|eval-recon|eval-class|balance>
#
# Methods Standard and AR train scVI without and with AR (bins 100, alpha 1e-4)
# on the unbalanced training data. main.py builds their training data and writes
# it to ../data/sctab/bloodbase_*_AR<AR>_train_adata_*HVGs.h5ad:
#   models   ../saved_models/scvi/sctab/<hash>/  (index: ../saved_models/scvi/sctab/runs.csv)
#   results  ../result/test/scVI-{reconstruction,classification}-evals
#
# Baselines class_balancing and geometric_sketching train standard scVI on
# balanced data (scripts/balance_data.py): the validation data are the
# Standard/AR validation cells (10% blood cells), only the training data are
# balanced (90k rows), and both keep raw counts. All four methods use the same
# environment.
#   data     ../data/sctab/balanced_data/<method>/<seed>/  (train, valid, composition csv)
#   results  ../result/test/scVI-{reconstruction,classification}-evals-balanced-data
#
# The train stage writes runs.csv after training; the eval stages read the
# model directories from it.
#
# Standard and AR (the default methods):
#   bash run_scvi.sh train
#   bash run_scvi.sh eval-recon
#   bash run_scvi.sh eval-class
# Baselines, after the Standard/AR training (balance reads the training data it
# writes), then the plots and tables with
# eval_scripts/generate_plots_scvi_models_balanced_data.ipynb:
#   bash run_scvi.sh balance
#   METHODS="class_balancing geometric_sketching" bash run_scvi.sh train
#   METHODS="class_balancing geometric_sketching" bash run_scvi.sh eval-recon
#   METHODS="class_balancing geometric_sketching" bash run_scvi.sh eval-class
#
# Optional environment variables:
#   SCAR_VENV=~/env/scar-env  activate this virtualenv (after module load
#                          python/3.10) instead of the conda env scar-env
#   NUM_GPUS=4             run up to 4 jobs at once, one per GPU (default 1;
#                          balance and eval stages always run one at a time)
#   WRITE_JOBS=jobs/dir    do not run; write one standalone script per job
#   SEEDS="42 43"          override the seeds
#   ATLAS_COUNTS="0 1000"  override the atlas cell counts
#   METHODS="AR"           override the methods (default Standard AR; balance stage:
#                          class_balancing geometric_sketching)
#   WANDB_MODE, WANDB_ENTITY   passed on to wandb (training)

stage=$1

case "$stage" in
    balance|train|eval-recon|eval-class) ;;
    *) sed -n '2,48p' "$0"; exit 1 ;;
esac

cd "$(dirname "$0")" || exit 1

## Environment: virtualenv if SCAR_VENV is set, otherwise conda ##
if [ -n "$SCAR_VENV" ]; then
    activate_env=('module load python/3.10'
                  "source $(printf '%q' "$SCAR_VENV")/bin/activate")
else
    activate_env=('eval "$(conda shell.bash hook)"'
                  'conda activate scar-env')
fi
# writing job scripts does not need the environment
if [ -z "$WRITE_JOBS" ]; then
    for line in "${activate_env[@]}"; do eval "$line" || exit 1; done
fi
# wandb settings given by the caller are passed on to every job
wandb_env=()
for v in WANDB_MODE WANDB_ENTITY WANDB_CACHE_DIR WANDB_DATA_DIR; do
    [ -n "${!v}" ] && wandb_env+=("$v=$(printf '%q' "${!v}")")
done
[ ${#wandb_env[@]} -gt 0 ] && export "${wandb_env[@]%%=*}"

## Paths (relative to bash/) ##
root='../'
data='sctab'
out_path='../saved_models/'
sctab_dir='../data/sctab'
balanced_dir="${sctab_dir}/balanced_data"
recon_out=../result/test/scVI-reconstruction-evals-balanced-data
class_out=../result/test/scVI-classification-evals-balanced-data
recon_out_main=../result/test/scVI-reconstruction-evals
class_out_main=../result/test/scVI-classification-evals

## Settings ##
latent_dim=64
num_epoch=300
batch_size=4096
alpha=0.0001
hvg_count=2000
num_gpus=${NUM_GPUS:-1}
# balance and eval stages write shared files, so they run one at a time
[[ "$stage" =~ ^(balance|eval-recon|eval-class)$ ]] && num_gpus=1

## Grids ##
seeds=(42 43 44 45 46)
atlas_counts=(0 1 10 100 1000 10000 50000)
# default methods: Standard and AR; the balance stage builds the baselines' data
if [ "$stage" == 'balance' ]; then
    balancing_methods=(class_balancing geometric_sketching)
else
    balancing_methods=(Standard AR)
fi
[ -n "$SEEDS" ] && read -ra seeds <<< "$SEEDS"
[ -n "$ATLAS_COUNTS" ] && read -ra atlas_counts <<< "$ATLAS_COUNTS"
[ -n "$METHODS" ] && read -ra balancing_methods <<< "$METHODS"

today=$(date '+%Y%m%d')
log_dir=log/$today
mkdir -p "$log_dir"
failed_file="${log_dir}/failed-scvi-balanced-${stage}-$(date '+%H%M%S').txt"
counter=0
[ -n "$WRITE_JOBS" ] && mkdir -p "$WRITE_JOBS"

## Write one job as a standalone script: write_job <log name> <command...> ##
write_job() {
    local name=$1; shift
    local file="${WRITE_JOBS}/${name}.sh"
    {
        echo '#!/bin/bash'
        echo "# scVI job: ${name}"
        echo "# GPU: set CUDA_VISIBLE_DEVICES before running, e.g. CUDA_VISIBLE_DEVICES=0 bash ${name}.sh"
        echo "cd $(printf '%q' "$PWD") || exit 1"
        printf '%s\n' "${activate_env[@]}"
        [ ${#wandb_env[@]} -gt 0 ] && echo "export ${wandb_env[*]}"
        echo "mkdir -p ${log_dir}"
        printf '%q' "$1"; shift; printf ' %q' "$@"
        echo " > ${log_dir}/${name}.txt 2> ${log_dir}/${name}.err"
    } > "$file"
    chmod +x "$file"
}

## Run one job: run_job <log name> <command...> ##
run_job() {
    local name=$1; shift
    if [ -n "$WRITE_JOBS" ]; then write_job "$name" "$@"; counter=$((counter + 1)); return; fi
    local gpu_id=$((counter % num_gpus))
    counter=$((counter + 1))
    echo "[GPU ${gpu_id}] ${name}"
    (
        CUDA_VISIBLE_DEVICES=${gpu_id} "$@" \
            > "${log_dir}/${name}.txt" 2> "${log_dir}/${name}.err"
        if [ $? -ne 0 ] || grep -q "Traceback\|main.py: error:" "${log_dir}/${name}.err"; then
            echo "Error in ${name}, see ${log_dir}/${name}.err"
            echo "${name}" >> "${failed_file}"
        fi
    ) &
    # wait for each batch of num_gpus jobs before starting the next one
    if (( counter % num_gpus == 0 )); then wait; fi
}

## main.py arguments shared by all scVI runs ##
main_common=(--root "$root" --data "$data" --out_path "$out_path"
             --model_name scvi --batch_size $batch_size --hvg_count $hvg_count --ood True)

## Evaluation of one model: eval_scvi <log name> <model dir> <train adata> <AR T|F>
##   <latent dim> <atlas count> <alpha> <epochs> [main] (positional, as the eval scripts read them;
##   main: results of the Standard/AR models go to the main result directories) ##
eval_scvi() {
    local name=$1 model=$2 adata=$3 ar=$4 ldim=$5 atlas=$6 a=$7 ep=$8
    local r_out=$recon_out c_out=$class_out
    [ "${9:-}" == 'main' ] && r_out=$recon_out_main && c_out=$class_out_main
    if [ "$stage" == 'eval-recon' ]; then
        run_job "$name" python -u ../eval_scripts/LDVAE_eval.py \
            "$sctab_dir" "$adata" "$model" "$r_out" "$seed" "$ar" "$ldim" "$atlas" "$a" "$ep"
    else
        run_job "$name" python -u ../eval_scripts/zero_shot_classification.py \
            scVI "$model" "$seed" "$c_out" "$ar" "$ldim" "$atlas" "$adata" "$a" "$ep"
    fi
}

# best model of a run, looked up in runs.csv by its settings:
# model_dir <balancing method, empty for Standard/AR> <arm: Naive or AR>
runs_csv="${out_path}scvi/${data}/runs.csv"
model_dir() {
    local hashes
    hashes=$(awk -F, -v m="$1" -v arm="$2" -v a="$atlas" -v s="$seed" -v e=$num_epoch \
                 -v l=$latent_dim -v b=$batch_size -v h=$hvg_count '
        NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
        $c["balancing_method"] == m && $c["arm"] == arm && $c["atlas_count"] == a &&
        $c["seed"] == s && $c["num_epoch"] == e && $c["latent_dim"] == l &&
        $c["batch_size"] == b && $c["hvg_count"] == h { print $c["hash"] }' "$runs_csv")
    if [ "$(echo "$hashes" | grep -c .)" -ne 1 ]; then
        echo "expected one run in ${runs_csv} for method '${1}' arm ${2} atlas ${atlas} seed ${seed}," \
             "found: ${hashes:-none}" >&2
        return 1
    fi
    echo "${out_path}scvi/${data}/${hashes}/best"
}

# runs.csv: one row per trained run (written after training, read by the eval stages)
write_runs_index() {
    python -c "import sys; sys.path.insert(0, '..'); from scripts.utils import write_runs_index; \
write_runs_index('${out_path}', 'scvi', '${data}')"
}
if [[ "$stage" == eval-* ]] && [ -z "$WRITE_JOBS" ]; then
    write_runs_index || exit 1
fi
[[ "$stage" == eval-* ]] && [ ! -f "$runs_csv" ] && { echo "${runs_csv} not found: train first"; exit 1; }

for seed in "${seeds[@]}"; do
    for atlas in "${atlas_counts[@]}"; do
        for method in "${balancing_methods[@]}"; do
            if [[ "$method" =~ ^(Standard|AR)$ ]]; then
                # Standard and AR scVI models, compared with the baselines;
                # AR models keep a checkpoint every 20 epochs, Standard every 100
                if [ "$method" == 'AR' ]; then ar=True; ckpt=20; else ar=False; ckpt=100; fi
                tag="scvi-${method}-atlas${atlas}-seed${seed}"
                case $stage in
                    balance) ;;
                    train)
                        run_job "train-${tag}" python -u ../main.py "${main_common[@]}" \
                            --train True --test False --predict False --checkpoint $ckpt \
                            --AR $ar --latent_dim $latent_dim --num_epoch $num_epoch \
                            --seed "$seed" --atlas_count "$atlas" --alpha $alpha ;;
                    eval-*)
                        if [ "$ar" == True ]; then arm=AR; else arm=Naive; fi
                        model=$(model_dir '' $arm) || exit 1
                        eval_scvi "${stage}-${tag}" "$model" \
                            "${sctab_dir}/bloodbase_${atlas}_atlas_seed${seed}_AR${ar}_train_adata_${hvg_count}_${hvg_count}HVGs.h5ad" \
                            "${ar:0:1}" $latent_dim "$atlas" $alpha $num_epoch main ;;
                esac
                continue
            fi
            tag="balanced-${method}-atlas${atlas}-seed${seed}"
            split="${balanced_dir}/${method}/${seed}/bloodbase_${atlas}_atlas_seed${seed}_${hvg_count}HVGs_${method}"
            case $stage in
                balance)
                    run_job "balance-${tag}" python -u ../scripts/balance_data.py \
                        --data "$data" --root "$root" \
                        --anndata_blood_file data/sctab/BaseModel_scTabBloodOnly_seed0_bloodbase_TrainingData.h5ad \
                        --anndata_atlas_file data/sctab/BaseModel_scTabAll_seed0_allbase_TrainingData.h5ad \
                        --obs_class_label tissue --training_data_count 100000 \
                        --seed "$seed" --atlas_count "$atlas" --hvg_count $hvg_count \
                        --balancing_method "$method" ;;
                train)
                    run_job "train-${tag}" python -u ../main.py "${main_common[@]}" \
                        --train True --checkpoint 100 \
                        --balanced_data True --balancing_method "$method" \
                        --train_adata_path "${split}_train_adata.h5ad" \
                        --valid_adata_path "${split}_valid_adata.h5ad" \
                        --AR False --latent_dim $latent_dim --num_epoch $num_epoch \
                        --seed "$seed" --atlas_count "$atlas" --alpha $alpha ;;
                eval-*)
                    model=$(model_dir "$method" Naive) || exit 1
                    eval_scvi "${stage}-${tag}" "$model" \
                        "${split}_train_adata.h5ad" F $latent_dim "$atlas" $alpha $num_epoch ;;
            esac
        done
    done
done
wait
[ "$stage" == train ] && [ -z "$WRITE_JOBS" ] && write_runs_index

if [ -n "$WRITE_JOBS" ]; then
    echo "Wrote ${counter} job scripts to ${WRITE_JOBS}/"
    exit 0
fi
if [ -f "${failed_file}" ]; then
    echo "$(wc -l < "${failed_file}") failed jobs are listed in ${failed_file}"
    exit 1
fi
echo "All balanced ${stage} jobs finished."
