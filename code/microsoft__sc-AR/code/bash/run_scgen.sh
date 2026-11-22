#!/bin/bash
# Train, predict, evaluate and summarize the scGen designed-data experiments.
#
# usage: bash run_scgen.sh <stage> <dataset>
#   stage:   train | predict | eval | merge | latent | rotation | plots
#   dataset: species | pbmc | lps-hpoly | all
#
# Data: the designed data files written by make_designed_data.sh,
#   ../data/designed/<data>/<data>-test<held out>-main<abundant>-per<percent>-seed<seed>.h5ad
# One design per held-out group (DESIGNS below): the held-out group is the test
# group, the abundant group keeps all its cells in training and the other groups
# are rare (percent = 0.01 and 0.05 of their cells in training); seeds 700-704.
# Arms: Naive (Standard), Uniform, AR-mu (AR-max), AR-mean-mu, AR-geomean-mu,
# AR-knn-mu; AR weights from the posterior mean.
#
# Stages, in this order:
#   train    train every arm: ../saved_models/scgen/<data>/<hash>/, then runs.csv
#   predict  predict the stimulated cells of the held-out group and of each rare
#            group (--in_dist_group) with every arm
#   eval     evaluate the predictions: ../result/test/<data>/jobs/
#   merge    one csv per design: ../result/test/<data>/ (eval_scripts/merge_eval.py)
#   latent   latent-space clustering: ../result/latent/ (eval_scripts/latent_clustering.py)
#   rotation rotation sensitivity of the AR-max weights: ../result/rotation/
#            (eval_scripts/rotation_invariance.py)
#   plots    by-group plots (../result/figures/<data>/), latent ARI tables
#            (../result/latent/), the AR weight ESS figure (../result/figures/scgen_ess/)
#            and the rotation table (../result/rotation/);
#            run with dataset 'all' after merge, latent and rotation for all datasets
# All stages pass the same run settings (common below), so predict and eval find
# the run directory written by train (its name is a hash of these settings).
#
# Optional environment variables:
#   NUM_GPUS=4             run up to 4 jobs at once, one per GPU (default 1)
#   TEST_GROUPS="mouse"    only the designs with these held-out groups (default: all)
#   SEEDS="700"            only these seeds (default 700 701 702 703 704; merge
#                          and plots expect all five)
#   WRITE_JOBS=jobs/train  train, predict, eval: do not run; write one standalone
#                          script per job into that directory (e.g. to submit them separately)
#   SCAR_VENV=~/env/scar-env  activate this virtualenv (after module load
#                          python/3.10) instead of the conda env scar-env
#   WANDB_MODE=offline     passed on to wandb (default: wandb's own, i.e. online)
#   WANDB_ENTITY=<team>    wandb entity/team to log to (default: your own account)
# On a SLURM cluster, submit the train, predict and eval jobs with submit_scgen_slurm.sh.

stage=$1
dataset=$2

if [[ ! "$stage" =~ ^(train|predict|eval|merge|latent|rotation|plots)$ ]] || [ -z "$dataset" ]; then
    sed -n '2,43p' "$0"; exit 1
fi

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
# avoid mixed OpenMP runtimes in sklearn (KMeans in the latent stage)
export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE} OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

## Paths (relative to bash/) ##
root='../'
design_dir='../data/designed'
out_path='../saved_models/'

## Settings shared by all scGen runs ##
latent_dim=64
optimizer=adam
lr=0.00005
weight_decay=0.00005
batch_size=2048
bins=100           # AR-max, AR-mean, AR-geomean: histogram bins per latent dimension
alpha=0.0001       # AR-max, AR-mean, AR-geomean: smoothing of the histogram densities
checkpoint=500
ood=True
variable_con=False
num_gpus=${NUM_GPUS:-1}

## Experiment ##
seeds=(700 701 702 703 704)
con_percent_list=(0.01 0.05)
arms=(Naive Uniform AR-mu AR-mean-mu AR-geomean-mu AR-knn-mu)
[ -n "$SEEDS" ] && read -ra seeds <<< "$SEEDS"

## Datasets ##
# list: all cell groups; designs: <held out>:<abundant> (abundant = the remaining
# group with the most cells); num_epoch per dataset
set_dataset() {
    case $1 in
        species)   list=('mouse' 'pig' 'rabbit' 'rat'); num_epoch=1000
                   designs=('mouse:rat' 'pig:rat' 'rabbit:rat' 'rat:rabbit') ;;
        pbmc)      list=('CD4T' 'B' 'NK' 'CD14+Mono' 'Dendritic' 'CD8T' 'FCGR3A+Mono'); num_epoch=1800
                   designs=('CD4T:FCGR3A+Mono' 'B:CD4T' 'NK:CD4T' 'CD14+Mono:CD4T' 'Dendritic:CD4T'
                            'CD8T:CD4T' 'FCGR3A+Mono:CD4T') ;;
        lps-hpoly) list=('Endocrine' 'Enterocyte' 'Enterocyte.Progenitor' 'Goblet' 'Stem' 'TA' 'TA.Early' 'Tuft'); num_epoch=2000
                   designs=('Endocrine:Stem' 'Enterocyte:Stem' 'Enterocyte.Progenitor:Stem' 'Goblet:Stem'
                            'Stem:TA.Early' 'TA:Stem' 'TA.Early:Stem' 'Tuft:Stem') ;;
        *) echo "unknown dataset: $1"; exit 1 ;;
    esac
}
if [ "$dataset" == 'all' ]; then datasets=('species' 'pbmc' 'lps-hpoly'); else datasets=("$dataset"); fi
for data in "${datasets[@]}"; do set_dataset "$data"; done

today=$(date '+%Y%m%d')
log_dir=log/$today
mkdir -p "$log_dir"
failed_file="${log_dir}/failed-${stage}-${dataset}-$(date '+%H%M%S').txt"
counter=0

## Summary stages: run from the repository root (script defaults and the result
## file patterns below are relative to it), one after the other ##
summary() {
    local name=$1; shift
    echo "${name}"
    python -u "$@" > "bash/${log_dir}/${name}.txt" 2> "bash/${log_dir}/${name}.err" \
        || { echo "Error in ${name}, see bash/${log_dir}/${name}.err"; exit 1; }
}
if [[ "$stage" =~ ^(merge|latent|rotation|plots)$ ]]; then
    [ -n "$WRITE_JOBS" ] && { echo "WRITE_JOBS is for the train, predict and eval stages"; exit 1; }
    cd .. || exit 1
    seed_args=(--seeds "${seeds[@]}")
    # latent, rotation: runs.csv of the trained runs (also when train ran through submit_scgen_slurm.sh)
    if [[ "$stage" =~ ^(latent|rotation)$ ]]; then
        for data in "${datasets[@]}"; do
            summary "runs-index-${data}" -c "from scripts.utils import write_runs_index; \
write_runs_index('saved_models/', 'scgen', '${data}')"
        done
    fi
    for data in "${datasets[@]}"; do
        set_dataset "$data"
        case $stage in
            merge)  summary "merge-${data}" eval_scripts/merge_eval.py "$data" --write ;;
            latent)
                for d in "${designs[@]}"; do
                    held=${d%%:*}
                    [ -n "$TEST_GROUPS" ] && [[ " $TEST_GROUPS " != *" $held "* ]] && continue
                    summary "latent-${data}-test${held}" eval_scripts/latent_clustering.py \
                        --data "$data" --held_out "$held" --percents "${con_percent_list[@]}" "${seed_args[@]}"
                done ;;
            rotation)
                for d in "${designs[@]}"; do
                    held=${d%%:*}
                    [ -n "$TEST_GROUPS" ] && [[ " $TEST_GROUPS " != *" $held "* ]] && continue
                    summary "rotation-${data}-test${held}" eval_scripts/rotation_invariance.py \
                        --data "$data" --held_out "$held" --percents "${con_percent_list[@]}" "${seed_args[@]}"
                done ;;
            plots)
                summary "plots-${data}" eval_scripts/generate_plot.py --data "$data" "${seed_args[@]}" \
                    --results "result/test/${data}/${data}-oodTrue-variable_conFalse-designed-test"*"-main"*.csv ;;
        esac
    done
    if [ "$stage" == 'plots' ] && [ "$dataset" == 'all' ]; then
        summary plots-latent-ari eval_scripts/generate_plot.py --latent_ari_tables "${seed_args[@]}" \
            --results result/latent/*/*/latent_clustering.csv
        summary plots-ess eval_scripts/plot_scgen_ess.py
        summary plots-rotation eval_scripts/rotation_invariance.py --table "${seed_args[@]}"
    fi
    echo "All ${stage} steps finished."
    exit 0
fi

[ -n "$WRITE_JOBS" ] && mkdir -p "$WRITE_JOBS"

## Write one job as a standalone script: write_job <log name> <main.py arguments...> ##
write_job() {
    local name=$1; shift
    local file="${WRITE_JOBS}/${name}.sh"
    {
        echo '#!/bin/bash'
        echo "# scGen job: ${name}"
        echo "# GPU: set CUDA_VISIBLE_DEVICES before running, e.g. CUDA_VISIBLE_DEVICES=0 bash ${name}.sh"
        echo "cd $(printf '%q' "$PWD") || exit 1"
        printf '%s\n' "${activate_env[@]}"
        [ ${#wandb_env[@]} -gt 0 ] && echo "export ${wandb_env[*]}"
        echo "export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK} OMP_NUM_THREADS=${OMP_NUM_THREADS}"
        echo "mkdir -p ${log_dir}"
        printf 'python ../main.py'; printf ' %q' "$@"
        echo " > ${log_dir}/${name}.txt 2> ${log_dir}/${name}.err"
    } > "$file"
    chmod +x "$file"
}

## Run one job: run_job <log name> <main.py arguments...> ##
run_job() {
    local name=$1; shift
    if [ -n "$WRITE_JOBS" ]; then write_job "$name" "$@"; counter=$((counter + 1)); return; fi
    local gpu_id=$((counter % num_gpus))
    counter=$((counter + 1))
    echo "[GPU ${gpu_id}] ${name}"
    (
        CUDA_VISIBLE_DEVICES=${gpu_id} python ../main.py "$@" \
            > "${log_dir}/${name}.txt" 2> "${log_dir}/${name}.err"
        if [ $? -ne 0 ] || grep -q "Traceback\|main.py: error:" "${log_dir}/${name}.err"; then
            echo "Error in ${name}, see ${log_dir}/${name}.err"
            echo "${name}" >> "${failed_file}"
        fi
    ) &
    # wait for each batch of num_gpus jobs before starting the next one
    if (( counter % num_gpus == 0 )); then wait; fi
}

models=$(IFS=,; echo "${arms[*]}")

for data in "${datasets[@]}"; do
    set_dataset "$data"
    for d in "${designs[@]}"; do
        test_data=${d%%:*}; abundant=${d#*:}
        [ -n "$TEST_GROUPS" ] && [[ " $TEST_GROUPS " != *" $test_data "* ]] && continue
        # training groups: all groups but the held-out one, in list order; the rare
        # groups are the training groups other than the abundant one
        train_groups=(); rare=()
        for g in "${list[@]}"; do
            [ "$g" == "$test_data" ] && continue
            train_groups+=("$g")
            [ "$g" != "$abundant" ] && rare+=("$g")
        done
        train_data=$(IFS=,; echo "${train_groups[*]}")

        for seed in "${seeds[@]}"; do
            for con_percent in "${con_percent_list[@]}"; do
                design_file="${design_dir}/${data}/${data}-test${test_data}-main${abundant}-per${con_percent}-seed${seed}.h5ad"
                [ -f "$design_file" ] || \
                    { echo "designed data file not found: $design_file (run make_designed_data.sh)"; exit 1; }
                # main.py checks these against the design stored in the data file
                common=(--data "$data" --train_data "$train_data" --test_data "$test_data"
                        --ood $ood --variable_con $variable_con --con_percent $con_percent
                        --seed $seed --designed_data True --h5ad_adata_file "$design_file"
                        --model_name scgen --latent_dim $latent_dim --num_epoch $num_epoch
                        --optimizer $optimizer --lr $lr --weight_decay $weight_decay
                        --batch_size $batch_size --bins $bins --alpha $alpha
                        --root $root --out_path $out_path)
                tag="${data}-test${test_data}-main${abundant}-per${con_percent}-seed${seed}"

                if [ "$stage" == 'train' ]; then
                    for arm in "${arms[@]}"; do
                        # arm -> --AR / --w_update, as scripts/utils.py::set_arm
                        case $arm in
                            Naive)         ar=False; w_update='max_w' ;;
                            Uniform)       ar=True;  w_update='uniform' ;;
                            AR-mu)         ar=True;  w_update='max_w' ;;
                            AR-mean-mu)    ar=True;  w_update='mean_w' ;;
                            AR-geomean-mu) ar=True;  w_update='geomean_w' ;;
                            AR-knn-mu)     ar=True;  w_update='knn_w' ;;
                        esac
                        run_job "train-${tag}-${arm}" "${common[@]}" --train True \
                            --AR $ar --w_update $w_update --checkpoint $checkpoint
                    done
                else
                    case $stage in
                        predict) flags=(--predict True) ;;
                        eval)    flags=(--test True) ;;
                    esac
                    # held-out group, then the test cells of each rare group
                    for group in '' "${rare[@]}"; do
                        in_dist=(); name="${stage}-${tag}"
                        [ -n "$group" ] && { in_dist=(--in_dist_group "$group"); name="${name}-indist${group}"; }
                        run_job "$name" "${common[@]}" "${flags[@]}" "${in_dist[@]}" \
                            --model "$models"
                    done
                fi
            done
        done
    done
done
wait

if [ -n "$WRITE_JOBS" ]; then
    echo "Wrote ${counter} job scripts to ${WRITE_JOBS}/"
    exit 0
fi
# runs.csv: one row per trained run (read by the latent stage, which also rewrites it)
if [ "$stage" == 'train' ]; then
    for data in "${datasets[@]}"; do
        python -c "import sys; sys.path.insert(0, '..'); from scripts.utils import write_runs_index; \
write_runs_index('${out_path}', 'scgen', '${data}')" || echo "runs.csv not written for ${data}" >> "${failed_file}"
    done
fi
if [ -f "${failed_file}" ]; then
    echo "$(wc -l < "${failed_file}") failed jobs are listed in ${failed_file}"
    exit 1
fi
echo "All ${stage} jobs finished."
