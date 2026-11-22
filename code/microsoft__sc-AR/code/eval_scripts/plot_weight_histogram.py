"""Histograms of AR resampling weights per cell group (Fig. 2D).

For each checkpoint, plots the distribution of the log10 min-max-normalized
resampling weights of the training cells, one colour per group of an obs column
(default blood_atlas: Blood vs Atlas), with a kernel density estimate and a
dashed line at the mean of each group. Larger groups are randomly subsampled to
the size of the smallest group for the histogram; the means use all cells.

Input:
    weights   one CSV per checkpoint with a column 'resampling_weight', one row per
              training cell, as written by `bash run_scvi_weights.sh weights`
              (<weights_dir>/sctab/<seed>/scvi/resampling_weights_sctab_AR_epoch300_seed<seed>_
              scvi_ldim64_atlascount<atlas>_trackedepoch<epoch>.csv)
    data      the training h5ad of the same model (same cells, same order), from
              which the group labels are read (obs column --label_key)

Output: <out_dir>/<prefix>_epoch<epoch>.{png,pdf}, one figure per checkpoint, and
<out_dir>/<prefix>_means.csv (mean log10 normalized weight and cell count per group).

Example (from eval_scripts/, Fig. 2D: seed 42, 10,000 atlas cells):
    python plot_weight_histogram.py --seed 42 --atlas_count 10000 --epochs 0 80 160 240
"""
import argparse
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# first font found is used (Arial look of the manuscript figures)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42

PALETTE = {'Blood': 'orchid', 'Atlas': 'cadetblue'}
EPS = 1e-6  # added before log10 so that the minimum weight (0 after scaling) is finite


def read_labels(path, key):
    """Read one obs column of an h5ad file with h5py (categorical or string)."""
    with h5py.File(path, 'r') as f:
        col = f['obs'][key]
        if isinstance(col, h5py.Group):                      # categorical
            cats = col['categories'][()]
            codes = col['codes'][()]
            labels = np.asarray(cats)[codes]
        else:
            labels = col[()]
            if 'categories' in col.attrs:                   # old anndata layout
                labels = np.asarray(f['obs'][col.attrs['categories']][()])[labels]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in labels])


def log_normalized(w):
    """log10 of the min-max-normalized weights."""
    w = np.asarray(w, dtype=float)
    return np.log10((w - w.min()) / (w.max() - w.min()) + EPS)


def plot_histogram(values, labels, save_path, palette=None, bins=100, figsize=(8, 6),
                   dpi=600, seed=42):
    """Histogram + KDE per group, with dashed lines at the group means.

    Args:
        values (np.ndarray): log10 normalized resampling weight per cell
        labels (np.ndarray): group of each cell
        save_path (str): output path without extension (.png and .pdf are written)
        palette (dict, optional): group -> colour (default: PALETTE, else seaborn colours)
        bins (int, optional): number of histogram bins
        figsize (tuple, optional): figure size in inches
        dpi (int, optional): resolution of the png
        seed (int, optional): random state of the subsampling

    Returns:
        pd.DataFrame: mean value and cell count per group (all cells)"""
    df = pd.DataFrame({'value': values, 'group': labels})
    groups = list(dict.fromkeys(labels))  # order of first appearance
    if set(groups) <= set(palette or PALETTE):
        groups = [g for g in (palette or PALETTE) if g in groups]  # Blood before Atlas
    if palette is None:
        palette = PALETTE if set(groups) <= set(PALETTE) else \
            dict(zip(groups, sns.color_palette(n_colors=len(groups))))
    means = df.groupby('group', sort=False)['value'].agg(['mean', 'size']).reset_index()

    # subsample every group to the size of the smallest group
    n_min = df['group'].value_counts().min()
    df = pd.concat([df[df['group'] == g].sample(n=n_min, random_state=seed)
                    if (df['group'] == g).sum() > n_min else df[df['group'] == g]
                    for g in groups], axis=0)

    plt.figure(figsize=figsize)
    ax = sns.histplot(data=df, x='value', hue='group', bins=bins, stat='count', kde=True,
                      palette=palette)
    sns.move_legend(ax, 'lower center', bbox_to_anchor=(.5, 1), ncol=len(groups), title=None,
                    frameon=False, fontsize=20)
    for g, m in zip(means['group'], means['mean']):
        plt.axvline(m, color=palette[g], linestyle='--')
    plt.xlabel('Log(10) Normalized Resampling Weight', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path + '.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close()
    return means


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='seed of the scVI AR model')
    parser.add_argument('--atlas_count', type=int, default=10000,
                        help='number of atlas cells of the scVI AR model')
    parser.add_argument('--epochs', nargs='+', default=['0', '80', '160', '240'],
                        help='checkpoints (tracked epochs) to plot')
    parser.add_argument('--weights_dir', default='../result/scvi_weights/resampling_weights',
                        help='output folder of `run_scvi_weights.sh weights`')
    parser.add_argument('--weights_pattern',
                        default='sctab/{seed}/scvi/resampling_weights_sctab_AR_epoch300_seed{seed}_'
                                'scvi_ldim64_atlascount{atlas}_trackedepoch{epoch}.csv',
                        help='weight file name below --weights_dir')
    parser.add_argument('--data_file', default=None,
                        help='training h5ad of the model (default: ../data/sctab/bloodbase_'
                             '<atlas>_atlas_seed<seed>_ARTrue_train_adata_2000_2000HVGs.h5ad)')
    parser.add_argument('--label_key', default='blood_atlas', help='obs column of the groups')
    parser.add_argument('--bins', type=int, default=100, help='histogram bins of the plot')
    parser.add_argument('--out_dir', default='../figures/weight_histogram')
    parser.add_argument('--prefix', default=None,
                        help='output file prefix (default: sctab_seed<seed>_atlas<atlas>)')
    args = parser.parse_args()

    data_file = args.data_file or (f'../data/sctab/bloodbase_{args.atlas_count}_atlas_seed'
                                   f'{args.seed}_ARTrue_train_adata_2000_2000HVGs.h5ad')
    prefix = args.prefix or f'sctab_seed{args.seed}_atlas{args.atlas_count}'
    os.makedirs(args.out_dir, exist_ok=True)

    labels = read_labels(data_file, args.label_key)
    print('cells per group:', pd.Series(labels).value_counts().to_dict())
    rows = []
    for epoch in args.epochs:
        path = os.path.join(args.weights_dir, args.weights_pattern.format(
            seed=args.seed, atlas=args.atlas_count, epoch=epoch))
        w = pd.read_csv(path)['resampling_weight'].values
        assert len(w) == len(labels), f'{path}: {len(w)} weights, {len(labels)} cells'
        means = plot_histogram(log_normalized(w), labels,
                               os.path.join(args.out_dir, f'{prefix}_epoch{epoch}'),
                               bins=args.bins, seed=42)
        rows.append(means.assign(epoch=epoch))
    out = pd.concat(rows).rename(columns={'mean': 'mean_log10_normalized_weight', 'size': 'n_cells'})
    out.to_csv(os.path.join(args.out_dir, f'{prefix}_means.csv'), index=False)
    print(out.round(3).to_string(index=False))
    print('Wrote', os.path.join(args.out_dir, prefix + '_*'))


if __name__ == '__main__':
    main()
