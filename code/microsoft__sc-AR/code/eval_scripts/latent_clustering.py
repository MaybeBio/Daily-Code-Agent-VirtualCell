"""Latent-space clustering of the designed sweep runs (real cells only), and
the cell type x condition PCA panels.

Designs: every held-out group of a dataset (all its cells are test cells), one
abundant group fully in training, the other groups rare (1 % / 5 %); seeds
700-704. Arms: Standard (Naive), AR-max (AR-mu), AR-mean (AR-mean-mu).

For each run the model is loaded as in scripts/predict.py and all cells of the
run's data file are encoded once (posterior mean μ; the encoder input is the
expression only, so the labels used for encoding do not change the latents).
Task: cell group x condition on all cells of the data file (every group,
condition and split); KMeans (k = number of groups x 2, random_state = run
seed, n_init = 10) scored against '<group>|<condition>' with the adjusted Rand
index (ARI). No predictions or deltas are used.
Output: <out_dir>/<data>/test<held>-main<abundant>/latent_clustering.csv

PCA panels (--pca_seed only; arms Standard and AR-max): PCA (2 components) of
the latent of all cells, one file per arm x colouring: ground truth, or KMeans
clusters under Hungarian-matched ground-truth names
(eval.py::_match_clusters_to_true_labels), ARI written on the KMeans panel.
Colour: one hue per group (tab20 pair), dark = stimulated, light = control.
No axis labels, ticks, title or legend.
Output: <out_dir>/<data>/test<held>-main<abundant>/pca_cell_group_x_condition/
  <data>-test<held>-main<abundant>-per<per>-seed<seed>-<arm>-{groundtruth,kmeans}.{png,pdf}

Runs are found in <out_path>/scgen/<data>/runs.csv (written by
bash/run_scgen.sh, train and latent stages) by data file name and arm.

usage (from the repository root, on a compute node):
  python eval_scripts/latent_clustering.py --data pbmc --held_out B
"""
import argparse
import gc
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eval_scripts.eval import _match_clusters_to_true_labels
from scripts.dataloader import load_and_preprocess_data
from scripts.predict import get_train_test_adata
from scripts.scgen_AR.scgen._scgen import SCGEN
from scripts.utils import set_seed

# plotted arm name -> arm name in runs.csv (scripts/utils.py::arm_name)
ARMS = {'Standard': 'Naive', 'AR-mean': 'AR-mean-mu', 'AR-max': 'AR-mu'}
PCA_ARMS = ('Standard', 'AR-max')
PERS = (0.01, 0.05)
SEEDS = (700, 701, 702, 703, 704)
FILE = re.compile(r'(?P<data>.+)-test(?P<held>.+)-main(?P<abundant>.+)-per(?P<per>[0-9.]+)'
                  r'-seed(?P<seed>\d+)\.h5ad$')
# marker size / alpha of the PCA panels per dataset (Species ~62k cells, PBMC ~17k, H. poly ~5k)
STYLE = {'species': (2, 0.3), 'pbmc': (4, 0.5), 'lps-hpoly': (6, 0.6)}


def parse_file(path):
    m = FILE.match(os.path.basename(str(path)))
    return None if m is None else m.groupdict()


def sweep_designs(data_dir, data, pers, seeds):
    """(held-out, abundant) of every sweep design of the dataset, from the data files."""
    designs = {}
    for f in glob.glob(f'{data_dir}/{data}/*.h5ad'):
        p = parse_file(f)
        if p is None or p['data'] != data:
            continue
        designs.setdefault((p['held'], p['abundant']), set()).add((float(p['per']), int(p['seed'])))
    expected = {(p, s) for p in pers for s in seeds}
    for d, files in designs.items():
        assert expected <= files, f'{data} {d}: data files {sorted(files)}'
    return sorted(designs)


def find_runs(out_path, data, designs, pers, seeds):
    """Run dirs of the chosen arms: {(held, abundant, arm, per, seed): runs.csv row}."""
    names = {v: k for k, v in ARMS.items()}
    index = pd.read_csv(os.path.join(out_path, 'scgen', data, 'runs.csv'), dtype={'hash': str})
    runs = {}
    for _, r in index.iterrows():
        p = parse_file(r['data_file'])
        if p is None or r['arm'] not in names or (p['held'], p['abundant']) not in designs:
            continue
        if float(p['per']) not in pers or int(p['seed']) not in seeds:
            continue
        assert r['finished'] == True, f'{r["hash"]}: not finished'
        assert r['test_data'] == p['held'] and r['seed'] == int(p['seed']), \
            f'{r["hash"]}: run arguments differ from the data file name'
        key = (p['held'], p['abundant'], names[r['arm']], float(p['per']), int(p['seed']))
        assert key not in runs, f'two runs for {key}'
        runs[key] = dict(r, run_dir=os.path.join(out_path, 'scgen', data, r['hash']))
    expected = {(h, a, arm, p, s) for h, a in designs for arm in ARMS for p in pers for s in seeds}
    assert set(runs) == expected, f'missing runs: {sorted(expected - set(runs))[:10]}'
    return runs


def run_args(run, data_file):
    """Arguments of the predict call for the held-out group of this run."""
    p = parse_file(data_file)
    return argparse.Namespace(
        model_name='scgen', data=run['data'], designed_data=True, h5ad_adata_file=data_file,
        train_data=run['train_data'].split(':'), test_data=[run['test_data']],
        ood=True, variable_con=False, con_percent=float(p['per']), seed=int(run['seed']),
        in_dist_group='', train=False, predict=True, test=False, gpu=False, root='.')


def encode(run, args, adata):
    """μ of all cells of the data file."""
    train_adata, _, _, _ = get_train_test_adata(args, adata)
    model = SCGEN.load(os.path.join(run['run_dir'], 'best'), train_adata.copy(),
                       accelerator='cpu', device='auto')
    model.is_trained = True

    label = args.adata_label_cell
    obs = adata.obs
    held = obs[label].astype(str) == args.test_data[0]
    assert (obs.loc[held, 'split'] == 'test').all(), 'held-out cells outside the test split'

    # the held-out group is not a category of the model (no training cells):
    # encode it under a training label (not a network input), as predict.py
    cells = adata.copy()
    train_labels = set(train_adata.obs[label].astype(str))
    unknown = ~cells.obs[label].astype(str).isin(train_labels).values
    assert set(cells.obs.loc[unknown, label].astype(str)) == {args.test_data[0]}
    cells.obs[label] = cells.obs[label].astype(str)
    cells.obs.loc[unknown, label] = sorted(train_labels)[0]
    z = model.get_latent_representation(cells)
    del model, cells
    gc.collect()
    return z


def cluster(z, y, seed):
    """KMeans (k = number of labels) as in eval.py::evaluate_latent_clustering."""
    k = len(np.unique(y))
    set_seed(seed)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(z)
    scores = {'n_cells': len(y), 'k': k, 'ari': adjusted_rand_score(y, km)}
    return scores, km


def panel(xy, lab, order, colours, path, style, ari=None):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.scatter(xy[order, 0], xy[order, 1], s=style[0], alpha=style[1], linewidths=0,
               c=[colours[l] for l in lab[order]], rasterized=True)
    ax.set_axis_off()
    if ari is not None:
        ax.text(0.03, 0.97, f'ARI = {ari:.2f}', transform=ax.transAxes, ha='left', va='top',
                fontsize=11, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path}.{ext}', dpi=600)
    plt.close(fig)


def plot_pca(z, y, cond, groups, km, ari, args, seed, path):
    """Ground-truth and KMeans PCA panels of one run."""
    stim = [c for c in sorted(set(cond)) if c != args.adata_label_unper]
    assert len(stim) == 1 and len(groups) <= 10
    colours = {}
    for i, grp in enumerate(groups):
        colours[f'{grp}|{stim[0]}'] = plt.cm.tab20.colors[2 * i]
        colours[f'{grp}|{args.adata_label_unper}'] = plt.cm.tab20.colors[2 * i + 1]
    xy = PCA(n_components=2, random_state=seed).fit_transform(z)
    order = np.random.default_rng(seed).permutation(len(y))
    style = STYLE[args.data]
    panel(xy, y, order, colours, path.format('groundtruth'), style)
    panel(xy, _match_clusters_to_true_labels(y, km), order, colours, path.format('kmeans'), style, ari)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True, choices=['species', 'pbmc', 'lps-hpoly'])
    parser.add_argument('--held_out', nargs='*', default=None,
                        help='held-out groups to run (default: all sweep designs)')
    parser.add_argument('--out_path', default='saved_models',
                        help='folder of the trained models (main.py --out_path)')
    parser.add_argument('--data_dir', default='data/designed',
                        help='folder of the designed data files (scripts/make_designed_data.py)')
    parser.add_argument('--out_dir', default='result/latent')
    parser.add_argument('--percents', type=float, nargs='+', default=list(PERS))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--pca_seed', type=int, default=700,
                        help='PCA panels only for this seed (none if not in --seeds)')
    args = parser.parse_args()

    designs = sweep_designs(args.data_dir, args.data, args.percents, args.seeds)
    if args.held_out:
        designs = [d for d in designs if d[0] in args.held_out]
        assert {d[0] for d in designs} == set(args.held_out), f'unknown held-out group in {args.held_out}'
    runs = find_runs(args.out_path, args.data, designs, args.percents, args.seeds)

    for held, abundant in designs:
        name = f'test{held}-main{abundant}'
        out = os.path.join(args.out_dir, args.data, name)
        os.makedirs(out, exist_ok=True)
        rows = []
        for per in args.percents:
            for seed in sorted(args.seeds):
                f = f'{args.data_dir}/{args.data}/{args.data}-{name}-per{per}-seed{seed}.h5ad'
                adata = None
                for arm in ARMS:
                    run = runs[(held, abundant, arm, per, seed)]
                    assert run['data_file'] == os.path.basename(f)
                    run_arguments = run_args(run, f)
                    if adata is None:
                        adata = load_and_preprocess_data(run_arguments)
                        # the loader sets the label columns of the dataset
                        labels = {k: getattr(run_arguments, k) for k in
                                  ('adata_label_cell', 'adata_label_per', 'adata_label_unper')}
                    vars(run_arguments).update(labels)
                    z = encode(run, run_arguments, adata)
                    label = run_arguments.adata_label_cell
                    # cell group x condition on all cells (k = groups x conditions)
                    yg = adata.obs[label].astype(str).values
                    cond = adata.obs['condition'].astype(str).values
                    y = np.array([f'{g}|{c}' for g, c in zip(yg, cond)])
                    s, km = cluster(z, y, seed)
                    rows.append(dict(data=args.data, held_out=held, abundant=abundant, arm=arm,
                                     model=run['arm'], per=per, seed=seed, task='cell_group_x_condition',
                                     label=f'{label} x condition',
                                     **s, run_dir=run['run_dir'], data_file=f))
                    print(rows[-1], flush=True)
                    if seed == args.pca_seed and arm in PCA_ARMS:
                        pca_dir = os.path.join(out, 'pca_cell_group_x_condition')
                        os.makedirs(pca_dir, exist_ok=True)
                        plot_pca(z, y, cond, sorted(set(yg)), km, s['ari'], run_arguments, seed,
                                 os.path.join(pca_dir, f'{args.data}-{name}-per{per}-seed{seed}-{arm}-{{}}'))
                    del z
                    gc.collect()
                del adata
                gc.collect()
        pd.DataFrame(rows).to_csv(os.path.join(out, 'latent_clustering.csv'), index=False)


if __name__ == '__main__':
    main()
