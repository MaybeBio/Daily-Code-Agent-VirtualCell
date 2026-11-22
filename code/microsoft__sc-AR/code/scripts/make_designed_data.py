"""Perturbation data with a designed imbalance (scGen AR experiments).

Writes one AnnData file per seed and percentage. Each file holds all cells of
the input data plus obs['split'] (train, valid, test or unused). Cell groups
are the values of --group_key (species for Species, cell_type for PBMC,
cell_label for H. poly):

  held-out group    all cells                               -> test
  abundant group    all cells                               -> train / valid
  rare groups       per condition, in a random order fixed by the seed:
  (all others)      the first <test_fraction> of the cells  -> test
                    the next <percent> of all cells          -> train / valid
                    the remaining cells                     -> unused

The random order depends only on the seed, so within a seed the test cells
are the same at every percentage and the training cells of a smaller
percentage are part of every larger one. Every 5th selected cell is valid
(20 %), so a cell keeps its role at every percentage. The design is stored
in uns['design'].

usage (from the repository root):
  python scripts/make_designed_data.py
  python scripts/make_designed_data.py --data data/pbmc.h5ad --name pbmc \
      --group_key cell_type --held_out B --abundant CD4T
"""
import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc


def assign_split(adata, group_key, condition_key, held_out, abundant, percent, seed,
                 test_fraction, min_cells=0):
    """Role of every cell: 'train', 'valid', 'test' or 'unused'.

    Args:
        adata (AnnData): data with the group and condition columns in obs
        group_key, condition_key (str): obs columns of the cell group and condition
        held_out (str): group used only for testing
        abundant (str): group with all its cells in training
        percent (float): fraction of each rare group and condition in training
        seed (int): seed of the random order
        test_fraction (float): fraction of each rare group and condition in test
        min_cells (int): at least this many cells of each rare group and condition
            in training (train / valid), if the group has them

    Returns:
        np.ndarray: split label per cell"""
    split = np.full(adata.n_obs, 'test', dtype=object)
    rng = np.random.default_rng(seed)
    for group in sorted(adata.obs[group_key].unique()):
        if group == held_out:
            continue
        for condition in sorted(adata.obs[condition_key].unique()):
            cells = np.flatnonzero((adata.obs[group_key] == group) &
                                   (adata.obs[condition_key] == condition))
            order = rng.permutation(cells)
            if group == abundant:
                chosen = order
            else:
                n_test = int(test_fraction * len(order))
                n = max(min_cells, int(percent * len(order)))
                assert n_test + n <= len(order), (group, condition, n_test, n)
                split[order[n_test:]] = 'unused'
                chosen = order[n_test:n_test + n]
            split[chosen] = 'train'
            split[chosen[4::5]] = 'valid'
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default='data/species.h5ad')
    parser.add_argument('--name', default='species', help='dataset name used in the file names')
    parser.add_argument('--group_key', default='species')
    parser.add_argument('--condition_key', default='condition')
    parser.add_argument('--held_out', default='mouse')
    parser.add_argument('--abundant', default='rat')
    parser.add_argument('--percents', nargs='+', type=float, default=[0.01, 0.05])
    parser.add_argument('--test_fraction', type=float, default=0.9)
    parser.add_argument('--seeds', nargs='+', type=int, default=[700, 701, 702, 703, 704])
    parser.add_argument('--min_cells', type=int, default=1,
                        help='at least this many cells of each rare group and condition in training')
    parser.add_argument('--out_dir', default=None, help='default data/designed/<name>')
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join('data', 'designed', args.name)
    os.makedirs(out_dir, exist_ok=True)

    adata = sc.read_h5ad(args.data)
    groups = set(adata.obs[args.group_key])
    for g in (args.held_out, args.abundant):
        if g not in groups:
            raise ValueError(f'{g} is not a value of obs["{args.group_key}"]: {sorted(groups)}')
    rare = sorted(groups - {args.held_out, args.abundant})

    for seed in args.seeds:
        for percent in args.percents:
            out = adata.copy()
            out.obs['split'] = pd.Categorical(
                assign_split(adata, args.group_key, args.condition_key, args.held_out,
                             args.abundant, percent, seed, args.test_fraction,
                             min_cells=args.min_cells),
                categories=['train', 'valid', 'test', 'unused'])
            out.uns['design'] = {'group_key': args.group_key, 'held_out': args.held_out,
                                 'abundant': args.abundant, 'rare': rare,
                                 'percent': percent, 'seed': seed,
                                 'test_fraction': args.test_fraction,
                                 'min_cells': args.min_cells}

            name = f'{args.name}-test{args.held_out}-main{args.abundant}-per{percent}-seed{seed}.h5ad'
            # gzip, as the original data files
            out.write_h5ad(os.path.join(out_dir, name), compression='gzip')
            print(f'== {name}')
            print(pd.crosstab([out.obs[args.group_key], out.obs[args.condition_key]],
                              out.obs['split']))


if __name__ == '__main__':
    main()
