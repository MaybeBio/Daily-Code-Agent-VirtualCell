#!/bin/bash
# Designed data for the scGen experiments (scripts/make_designed_data.py): every
# cell group is held out once; the abundant group is the remaining group with
# the most cells (control + stimulated); all other groups are rare at 1 % and
# 5 % (90 % of each rare group and condition in test); seeds 700-704.
# Writes ../data/designed/<dataset>/<dataset>-test<held out>-main<abundant>-per<percent>-seed<seed>.h5ad
# (19 designs x 2 percentages x 5 seeds = 190 files), read by run_scgen.sh.
#
# usage: bash make_designed_data.sh <dataset>
#   dataset: species | pbmc | lps-hpoly | all
#   input:   ../data/species.h5ad, ../data/pbmc.h5ad, ../data/hpoly.h5ad
#
# Optional environment variables:
#   TEST_GROUPS="mouse"    only the designs with these held-out groups (default: all)
#   DRY_RUN=1              print the commands instead of running them
#   SCAR_VENV=~/env/scar-env  activate this virtualenv (after module load
#                          python/3.10) instead of the conda env scar-env

dataset=$1

if [ -z "$dataset" ]; then
    sed -n '2,18p' "$0"; exit 1
fi

# make_designed_data.py runs from the repository root
cd "$(dirname "$0")/.." || exit 1

if [ -z "$DRY_RUN" ]; then
    if [ -n "$SCAR_VENV" ]; then
        module load python/3.10 || exit 1
        source "$SCAR_VENV/bin/activate" || exit 1
    else
        eval "$(conda shell.bash hook)" && conda activate scar-env || exit 1
    fi
fi

## Settings of the designed experiments ##
percents=(0.01 0.05)
test_fraction=0.9
min_cells=1
seeds=(700 701 702 703 704)

## Datasets: input file, cell group column, designs <held out>:<abundant> (as run_scgen.sh) ##
set_dataset() {
    case $1 in
        species)   file=data/species.h5ad; key=species
                   designs=('mouse:rat' 'pig:rat' 'rabbit:rat' 'rat:rabbit') ;;
        pbmc)      file=data/pbmc.h5ad; key=cell_type
                   designs=('CD4T:FCGR3A+Mono' 'B:CD4T' 'NK:CD4T' 'CD14+Mono:CD4T' 'Dendritic:CD4T'
                            'CD8T:CD4T' 'FCGR3A+Mono:CD4T') ;;
        lps-hpoly) file=data/hpoly.h5ad; key=cell_label
                   designs=('Endocrine:Stem' 'Enterocyte:Stem' 'Enterocyte.Progenitor:Stem' 'Goblet:Stem'
                            'Stem:TA.Early' 'TA:Stem' 'TA.Early:Stem' 'Tuft:Stem') ;;
        *) echo "unknown dataset: $1"; exit 1 ;;
    esac
}
if [ "$dataset" == 'all' ]; then datasets=('species' 'pbmc' 'lps-hpoly'); else datasets=("$dataset"); fi
for data in "${datasets[@]}"; do set_dataset "$data"; done

for data in "${datasets[@]}"; do
    set_dataset "$data"
    [ -f "$file" ] || [ -n "$DRY_RUN" ] || { echo "input data not found: $file"; exit 1; }
    for d in "${designs[@]}"; do
        held=${d%%:*}; abundant=${d#*:}
        [ -n "$TEST_GROUPS" ] && [[ " $TEST_GROUPS " != *" $held "* ]] && continue
        cmd=(python scripts/make_designed_data.py --data "$file" --name "$data" --group_key "$key"
             --condition_key condition --held_out "$held" --abundant "$abundant"
             --percents "${percents[@]}" --test_fraction $test_fraction --min_cells $min_cells
             --seeds "${seeds[@]}")
        if [ -n "$DRY_RUN" ]; then
            printf '%q ' "${cmd[@]}"; echo
        else
            echo "=== ${data}: held out ${held}, abundant ${abundant}"
            "${cmd[@]}" || exit 1
        fi
    done
done
