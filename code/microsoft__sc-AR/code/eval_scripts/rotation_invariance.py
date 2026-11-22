"""Rotation sensitivity of the AR-max weights of the designed scGen sweep.

The AR-max weights are built from histograms along the latent axes, so they
depend on the basis of the latent space. For each AR-max run (best checkpoint,
all designs, 1 % / 5 %, seeds 700-704), the training cells are encoded once
(posterior mean μ, the latent the weights are computed on in training), and the
weights in the native basis are compared with the weights after random
orthogonal transformations (rotations and reflections) of the latent:

  spearman          Spearman correlation between the native and rotated weights
  top5pct_overlap   fraction of the 5 % highest-weighted cells that stay in the
                    top 5 % after the rotation (expected by chance: 0.05); the
                    weights have ties, which are broken by argsort order

Both are summarized by the median and the 2.5 / 97.5 % quantiles over rotations.
The latents are used in float64 in both bases, so that the two sets of weights
differ only by the rotation.

Output:
  <out_dir>/<data>/test<held>-main<abundant>/rotation_invariance.csv   one row per run
  <out_dir>/rotation_invariance_table.{csv,tex}   (--table) mean and SD of the medians
                                                  over seeds, one row per design x percent

Runs are found in <out_path>/scgen/<data>/runs.csv (written by bash/run_scgen.sh).
Memory: about 17 GB per design for species, 7 GB for pbmc and lps-hpoly (--table: little).

usage (from the repository root):
  python eval_scripts/rotation_invariance.py --data pbmc --held_out B
  python eval_scripts/rotation_invariance.py --table
"""
import argparse
import gc
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eval_scripts.latent_clustering import PERS, SEEDS, encode, parse_file, run_args, sweep_designs
from scripts.AR_resampling_weight_calculator import calculate_resampling_weights
from scripts.dataloader import load_and_preprocess_data

ARM = 'AR-mu'  # AR-max in runs.csv (scripts/utils.py::arm_name)
TOP_FRACTION = 0.05
METRICS = ('spearman', 'top5pct_overlap')
DATA_NAMES = {'species': 'Species', 'pbmc': 'PBMC', 'lps-hpoly': 'H. poly'}


def random_orthogonal_matrix(n_dims, seed=None):
    """Draw a random orthogonal matrix, uniformly (Haar measure): QR of a
    Gaussian matrix, with the column signs fixed so that diag(R) > 0."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((n_dims, n_dims)))
    return q * np.sign(np.diag(r))


def rotation_invariance_check(z, bins=100, smoothing_fac=0.0001, n_rotations=100, seed=0):
    """Compare the AR-max weights of a latent matrix with those after random rotations.

    Args:
        z (np.ndarray): latent matrix (N, K)
        bins (int): histogram bins per latent dimension
        smoothing_fac (float): smoothing added to the bin densities
        n_rotations (int): number of random rotations
        seed (int): seed of the rotations

    Returns:
        dict: median and 2.5 / 97.5 % quantiles over rotations of the Spearman
        correlation and of the overlap of the top 5 %"""

    z = np.asarray(z).astype(np.float64)
    rng = np.random.default_rng(seed)
    native = calculate_resampling_weights(z, bins=bins, smoothing_fac=smoothing_fac)

    top_n = max(1, int(TOP_FRACTION * len(native)))
    native_top = set(np.argsort(native)[-top_n:])
    scores = {name: [] for name in METRICS}

    for _ in range(n_rotations):
        rotation = random_orthogonal_matrix(z.shape[1], seed=int(rng.integers(1e9)))
        rotated = calculate_resampling_weights(z @ rotation, bins=bins, smoothing_fac=smoothing_fac)
        scores['spearman'].append(spearmanr(native, rotated).correlation)
        scores['top5pct_overlap'].append(len(native_top & set(np.argsort(rotated)[-top_n:])) / top_n)

    result = {}
    for name, values in scores.items():
        result[f'{name}_median'] = float(np.median(values))
        result[f'{name}_q025'] = float(np.quantile(values, 0.025))
        result[f'{name}_q975'] = float(np.quantile(values, 0.975))
    return result


def find_runs(out_path, data, designs, pers, seeds):
    """Run dirs of the AR-max arm: {(held, abundant, per, seed): runs.csv row}."""
    index = pd.read_csv(os.path.join(out_path, 'scgen', data, 'runs.csv'), dtype={'hash': str})
    runs = {}
    for _, r in index[index['arm'] == ARM].iterrows():
        p = parse_file(r['data_file'])
        if p is None or (p['held'], p['abundant']) not in designs:
            continue
        if float(p['per']) not in pers or int(p['seed']) not in seeds:
            continue
        assert r['finished'] == True, f'{r["hash"]}: not finished'
        assert r['test_data'] == p['held'] and r['seed'] == int(p['seed']), \
            f'{r["hash"]}: run arguments differ from the data file name'
        key = (p['held'], p['abundant'], float(p['per']), int(p['seed']))
        assert key not in runs, f'two runs for {key}'
        runs[key] = dict(r, run_dir=os.path.join(out_path, 'scgen', data, r['hash']))
    expected = {(h, a, p, s) for h, a in designs for p in pers for s in seeds}
    assert set(runs) == expected, f'missing runs: {sorted(expected - set(runs))[:10]}'
    return runs


def write_table(out_dir, seeds):
    """Mean and SD over seeds of the per-run medians: csv and LaTeX table."""
    files = sorted(glob.glob(os.path.join(out_dir, '*', '*', 'rotation_invariance.csv')))
    assert files, f'no rotation_invariance.csv in {out_dir}'
    runs = pd.concat([pd.read_csv(f) for f in files])
    assert runs['n_rotations'].nunique() == 1, 'runs with different numbers of rotations'

    rows = []
    for data in DATA_NAMES:
        for (held, abundant, per), g in runs[runs['data'] == data].groupby(['held_out', 'abundant', 'per']):
            assert sorted(g['seed']) == sorted(seeds), f'{data} {held} {per}: seeds {sorted(g["seed"])}'
            row = dict(data=data, held_out=held, abundant=abundant, per=per, n_seeds=len(g),
                       n_cells=g['n_cells'].mean())
            for m in METRICS:
                row[f'{m}_mean'] = g[f'{m}_median'].mean()
                row[f'{m}_sd'] = g[f'{m}_median'].std()
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, 'rotation_invariance_table.csv'), index=False)

    body = []
    for _, r in table.iterrows():
        values = [f'{r[f"{m}_mean"]:.3f} $\\pm$ {r[f"{m}_sd"]:.3f}' for m in METRICS]
        body.append(' & '.join([DATA_NAMES[r['data']], r['held_out'], r['abundant'],
                                f'{r["per"] * 100:g}\\%', *values]) + ' \\\\')
    header = ['Dataset', 'Held-out', 'Abundant', 'Rare cells', 'Spearman', 'Top-5\\% overlap']
    caption = (
        'Rotation sensitivity of the AR-max resampling weights of the scGen models. For each '
        'design and percentage of rare cells, the training cells were encoded with the best '
        f'checkpoint (posterior means, $Z$), and the weights were recomputed after '
        f'{runs["n_rotations"].iloc[0]} random orthogonal transformations $R$ (rotations and '
        'reflections) of the latents ($ZR$). Spearman: rank correlation between the weights '
        'before and after the transformation. Top-5\\% overlap: fraction of the 5\\% '
        'highest-weighted cells that stay in the top 5\\% after the transformation (expected by '
        'chance: 0.05). Values are medians over transformations; mean $\\pm$ SD over '
        f'{len(seeds)} seeds. Values below 1 show that the weights partly depend on the basis, '
        'as expected from histograms along the latent axes with max aggregation.')
    with open(os.path.join(out_dir, 'rotation_invariance_table.tex'), 'w') as f:
        f.write('\\begin{table}[t]\n\\centering\n')
        f.write(f'\\caption{{{caption}}}\n\\label{{tab:rotation_invariance_scgen}}\n')
        f.write('\\begin{tabular}{llll' + 'c' * len(METRICS) + '}\n\\toprule\n')
        f.write(' & '.join(header) + ' \\\\\n\\midrule\n')
        f.write('\n'.join(body) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    print(table.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', choices=list(DATA_NAMES))
    parser.add_argument('--held_out', nargs='*', default=None,
                        help='held-out groups to run (default: all sweep designs)')
    parser.add_argument('--table', action='store_true',
                        help='only write the summary table from the per-run csv files in --out_dir')
    parser.add_argument('--out_path', default='saved_models',
                        help='folder of the trained models (main.py --out_path)')
    parser.add_argument('--data_dir', default='data/designed',
                        help='folder of the designed data files (scripts/make_designed_data.py)')
    parser.add_argument('--out_dir', default='result/rotation')
    parser.add_argument('--percents', type=float, nargs='+', default=list(PERS))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--n_rotations', type=int, default=100)
    parser.add_argument('--rotation_seed', type=int, default=0)
    args = parser.parse_args()

    if args.table:
        write_table(args.out_dir, args.seeds)
        return
    if args.data is None:
        parser.error('--data is required unless --table is given')

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
                run = runs[(held, abundant, per, seed)]
                assert run['data_file'] == os.path.basename(f)
                run_arguments = run_args(run, f)
                # the loader sets the label columns of the dataset on the arguments
                adata = load_and_preprocess_data(run_arguments)
                z = encode(run, run_arguments, adata)
                z = z[(adata.obs['split'].astype(str) == 'train').values]
                rows.append(dict(data=args.data, held_out=held, abundant=abundant, per=per, seed=seed,
                                 n_cells=len(z), n_rotations=args.n_rotations,
                                 **rotation_invariance_check(z, n_rotations=args.n_rotations,
                                                             seed=args.rotation_seed),
                                 run_dir=run['run_dir']))
                print(rows[-1], flush=True)
                del adata, z
                gc.collect()
        pd.DataFrame(rows).to_csv(os.path.join(out, 'rotation_invariance.csv'), index=False)


if __name__ == '__main__':
    main()
