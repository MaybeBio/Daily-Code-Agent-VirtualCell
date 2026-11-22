#!/bin/bash
# Submit the scGen jobs of run_scgen.sh to SLURM as one job array, one GPU per job
# (CPU only with GPU=none, e.g. for predict and eval).
#
# usage: ACCOUNT=<slurm account> bash submit_scgen_slurm.sh <stage> <dataset>
#   stage: train | predict | eval;  dataset: as run_scgen.sh
#   (merge, latent, rotation and plots run directly: bash run_scgen.sh <stage> <dataset>)
#
# The job scripts are written by run_scgen.sh (WRITE_JOBS), so the run arguments
# are exactly those of a local run. jobs.txt in the jobs directory lists them;
# array task i runs line i.
#
# Environment variables (defaults in brackets):
#   ACCOUNT         SLURM account (required)
#   SCAR_VENV       virtualenv activated by every job [none: conda env scar-env]
#   RUNS_DIR        base of the jobs and SLURM log directories [../slurm]
#   JOBS_DIR        where the job scripts go [$RUNS_DIR/jobs/<stage>-<dataset>]
#   GPU             --gpus value, or none for a CPU-only job [1]; some clusters
#                   require a GPU type, e.g. GPU=<type>:1
#   CPUS, MEM       per job [4, 32G]
#   TIME            per job wall time [12:00:00]
#   ARRAY           task indices (lines of jobs.txt), e.g. 1,5,9 or 10-285 [all]
#   MAX_PARALLEL    tasks running at once [50]
#   DEPENDENCY      e.g. afterok:123456 (for predict after train, eval after predict)
#   DRY_RUN=1       write everything and run sbatch --test-only instead of submitting
# Also passed on to run_scgen.sh: TEST_GROUPS, SEEDS, WANDB_MODE, WANDB_ENTITY.

stage=$1
dataset=$2

if [[ ! "$stage" =~ ^(train|predict|eval)$ ]] || [ -z "$dataset" ]; then
    sed -n '2,26p' "$0"; exit 1
fi
if [ -z "$ACCOUNT" ]; then
    echo "set ACCOUNT to your SLURM account"; exit 1
fi

cd "$(dirname "$0")" || exit 1

runs_dir=${RUNS_DIR:-../slurm}
jobs_dir=${JOBS_DIR:-$runs_dir/jobs/${stage}-${dataset}}
slurm_log_dir=$runs_dir/log
gpu=${GPU:-1}
if [ "$gpu" == 'none' ]; then
    gpu_line=''
else
    gpu_line="#SBATCH --gpus=${gpu}"
fi
cpus=${CPUS:-4}
mem=${MEM:-32G}
time_limit=${TIME:-12:00:00}
max_parallel=${MAX_PARALLEL:-50}

if [ -n "$SCAR_VENV" ] && [ ! -f "$SCAR_VENV/bin/activate" ]; then
    echo "virtualenv not found: $SCAR_VENV"; exit 1
fi

## Write the job scripts (old ones in jobs_dir are replaced) ##
mkdir -p "$jobs_dir" "$slurm_log_dir" || exit 1
jobs_dir=$(cd "$jobs_dir" && pwd); slurm_log_dir=$(cd "$slurm_log_dir" && pwd)
rm -f "$jobs_dir"/*.sh "$jobs_dir/jobs.txt"
WRITE_JOBS="$jobs_dir" bash run_scgen.sh "$stage" "$dataset" || exit 1
ls "$jobs_dir"/*.sh 2>/dev/null | sort > "$jobs_dir/jobs.txt"
num_jobs=$(wc -l < "$jobs_dir/jobs.txt")
if [ "$num_jobs" -eq 0 ]; then
    echo "no jobs written"; exit 1
fi

## SLURM array script ##
name="scgen-${stage}-${dataset}"
array_script="$jobs_dir/${name}.sbatch"
cat > "$array_script" <<EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --account=${ACCOUNT}
${gpu_line}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time_limit}
#SBATCH --output=${slurm_log_dir}/%x-%A_%a.out

job=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" $(printf '%q' "$jobs_dir/jobs.txt"))
echo "task \${SLURM_ARRAY_TASK_ID} on \$(hostname): \$job"
nvidia-smi -L 2>/dev/null
start=\$(date +%s)
bash "\$job"
status=\$?
echo "exit status \$status after \$(( \$(date +%s) - start )) s"
exit \$status
EOF

array="${ARRAY:-1-${num_jobs}}%${max_parallel}"
sbatch_args=(--array="$array")
[ -n "$DEPENDENCY" ] && sbatch_args+=(--dependency="$DEPENDENCY")

echo "${num_jobs} job scripts in ${jobs_dir} (list: jobs.txt)"
echo "array ${array}, account ${ACCOUNT}, gpus ${gpu}, time ${time_limit}"
if [ -n "$DRY_RUN" ]; then
    sbatch --test-only "${sbatch_args[@]}" "$array_script"
else
    sbatch "${sbatch_args[@]}" "$array_script"
fi
