"""Normalized ESS of the AR resampling weights during scGen training, per aggregation rule.

Reads weights_ess.csv (normalized ESS per epoch, written by scripts/scgen_AR/scgen/_scgen.py)
of the designed-sweep runs in <saved_models>/<data>/<hash>/: 19 designs (Species 4, PBMC 7,
H. poly 8), seeds 700-704, 1 % / 5 %, AR rules max / mean / geomean / kNN (weights from the
posterior mean). Uniform has ESS = 1 by construction (dashed reference line). Lines: mean over
designs x seeds; band: +-1 SD. Writes a png/pdf figure and a csv with the per-epoch mean / SD / n.

usage (from the repository root):
  python eval_scripts/plot_scgen_ess.py [--saved_models saved_models/scgen] [--out_dir DIR]
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASETS = {'species': ('Species', 4), 'pbmc': ('PBMC', 7), 'lps-hpoly': ('H. poly', 8)}
# arm -> label, colour (categorical slots 1-4 of the dataviz reference palette), line style
RULES = {'AR-mu': ('AR (max)', '#2a78d6', '-'),
         'AR-mean-mu': ('AR (mean)', '#eb6834', '--'),
         'AR-geomean-mu': ('AR (geometric mean)', '#1baf7a', '-.'),
         'AR-knn-mu': ('AR (kNN)', '#eda100', ':')}
PERCENTS = [0.01, 0.05]
SEEDS = range(700, 705)
# designed data file: <design>-per<percent>-seed<seed>.h5ad (scripts/make_designed_data.py)
DESIGN_FILE = re.compile(r'^(?P<design>.+)-per(?P<percent>[0-9.]+)-seed\d+\.h5ad$')


def load(saved_models):
    rows = []
    for f in glob.glob(os.path.join(saved_models, '*', '*', 'weights_ess.csv')):
        with open(os.path.join(os.path.dirname(f), 'config.json')) as fh:
            cfg = json.load(fh)
        m = DESIGN_FILE.match(cfg.get('data_file') or '')
        # designed runs only; the asserts in main() check the design count per dataset
        if (cfg['data'] not in DATASETS or cfg['arm'] not in RULES or m is None
                or cfg['seed'] not in SEEDS):
            continue
        ess = np.loadtxt(f, delimiter=',', ndmin=1)
        assert len(ess) == cfg['num_epoch'], (f, len(ess))
        rows.append({'data': cfg['data'], 'arm': cfg['arm'], 'percent': float(m['percent']),
                     'design': m['design'], 'seed': cfg['seed'], 'ess': ess})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--saved_models', default='saved_models/scgen',
                        help='scGen runs: <saved_models>/<data>/<hash>/')
    parser.add_argument('--out_dir', default='result/figures/scgen_ess')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = load(args.saved_models)
    # every dataset x % x rule: all designs x 5 seeds, each (design, seed) once
    for (data, pct, arm), g in runs.groupby(['data', 'percent', 'arm']):
        n_designs = DATASETS[data][1]
        assert g['design'].nunique() == n_designs and len(g) == n_designs * len(SEEDS), (data, pct, arm, len(g))
        assert not g.duplicated(['design', 'seed']).any(), (data, pct, arm)
    assert len(runs) == sum(n for _, n in DATASETS.values()) * len(SEEDS) * len(PERCENTS) * len(RULES)
    print(f'{len(runs)} runs')

    plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(len(DATASETS), len(PERCENTS), figsize=(13, 13), sharey=True,
                             constrained_layout=True)
    summary = []
    for r, (data, (title, n_designs)) in enumerate(DATASETS.items()):
        for c, pct in enumerate(PERCENTS):
            ax = axes[r, c]
            ax.axhline(1.0, color='#7a7a75', lw=1.2, ls=(0, (4, 3)), zorder=1)
            if r == 0 and c == 0:
                ax.text(0.98, 0.985, 'Uniform', transform=ax.get_yaxis_transform(), ha='right', va='top',
                        fontsize=12, color='#5b5b57')
            for arm, (label, color, ls) in RULES.items():
                g = runs[(runs.data == data) & (runs.percent == pct) & (runs.arm == arm)]
                ess = np.vstack(g['ess'].values)
                mean, sd = ess.mean(axis=0), ess.std(axis=0, ddof=1)
                epochs = np.arange(1, ess.shape[1] + 1)
                ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.15, lw=0, zorder=2)
                ax.plot(epochs, mean, color=color, ls=ls, lw=2, label=label, zorder=3)
                summary.append(pd.DataFrame({'data': data, 'percent': pct, 'rule': label, 'epoch': epochs,
                                             'ess_mean': mean, 'ess_sd': sd, 'n_runs': len(g)}))
            ax.set_xlim(1, ess.shape[1])
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', color='#e6e5e0', lw=0.8)
            ax.tick_params(colors='#5b5b57')
            if r == 0:
                ax.set_title(f'Rare groups: {int(pct * 100)} % of cells', fontsize=16)
            if r == len(DATASETS) - 1:
                ax.set_xlabel('Epoch')
            if c == 0:
                ax.set_ylabel(f'{title} ({n_designs} designs)\nNormalized ESS')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=len(RULES), frameon=False,
               fontsize=18, handlelength=3.5)
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(args.out_dir, f'scgen_ess_by_rule.{ext}'), dpi=300, facecolor='white')
    summary = pd.concat(summary)
    summary.to_csv(os.path.join(args.out_dir, 'scgen_ess_by_rule.csv'), index=False)
    # last-epoch table for the log
    last = summary.sort_values('epoch').groupby(['data', 'percent', 'rule']).tail(1)
    print(last[['data', 'percent', 'rule', 'epoch', 'ess_mean', 'ess_sd', 'n_runs']].to_string(index=False))
    print('saved to', args.out_dir)


if __name__ == '__main__':
    main()
