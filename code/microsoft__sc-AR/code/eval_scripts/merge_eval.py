"""Merge the per-job eval CSVs of the designed sweep into one CSV per design and
kind (held-out group / in-distribution rare groups), and check them (read-only
on the inputs).

Input:  <result_dir>/<data>/jobs/<test_id>/test<held>-per<p>-seed<s>-<group>.csv
        (eval.py, --designed_data; test_id = <data>-oodTrue-variable_conFalse-
        designed-main<abundant>[-indist])
Output: <result_dir>/<data>/<data>-oodTrue-variable_conFalse-designed-test<held>-main<abundant>[-indist].csv
Checks per design: every (percent, seed, evaluated group) present (2 x 5 x
groups), each file 6 arms x 3 metrics = 18 rows, no missing values, no
duplicated (percent, seed, group, arm, metric), design columns consistent
(abundant, design_file matching percent and seed), test = held-out group and
in_dist_group = '' (held out) or the rare group.
usage: python eval_scripts/merge_eval.py <data> [--result_dir result/test]
           [--data_dir data/designed] [--write]   (without --write: check only)"""
import argparse
import glob
import os
import re
import sys

import anndata as ad
import pandas as pd

PERS, SEEDS = [0.01, 0.05], [700, 701, 702, 703, 704]
N_ARMS, N_METRICS = 6, 3
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('data')
parser.add_argument('--result_dir', default='result/test')
parser.add_argument('--data_dir', default='data/designed')
parser.add_argument('--write', action='store_true')
args = parser.parse_args()
RES, DATA_DIR, data, write = args.result_dir, args.data_dir, args.data, args.write
problems = []


def check(ok, msg):
    if not ok:
        problems.append(msg)
        print('PROBLEM', msg, flush=True)


designs = sorted({re.search(r'-test(.+)-main(.+)-per', os.path.basename(f)).groups()
                  for f in glob.glob(f'{DATA_DIR}/{data}/{data}-test*-main*-per*-seed7*.h5ad')})
for held, abundant in designs:
    meta = ad.read_h5ad(f'{DATA_DIR}/{data}/{data}-test{held}-main{abundant}-per0.01-seed700.h5ad',
                        backed='r').uns['design']
    rare = list(meta['rare'])
    for kind, groups in (('', [held]), ('-indist', rare)):
        test_id = f'{data}-oodTrue-variable_conFalse-designed-main{abundant}{kind}'
        # several designs share an abundant group: the file name starts with the held-out group
        files = sorted(glob.glob(f'{RES}/{data}/jobs/{test_id}/test{glob.escape(held)}-per*-seed*-*.csv'))
        expected = {f'test{held}-per{p}-seed{s}-{g}.csv' for p in PERS for s in SEEDS for g in groups}
        frames = [pd.read_csv(f).assign(_file=os.path.basename(f)) for f in files]
        got = {d['_file'].iloc[0] for d in frames}
        check(all((d['test'] == held).all() for d in frames), f'{held}/{abundant}{kind}: test column')
        check(got == expected, f'{held}/{abundant}{kind}: {len(got)} job files, expected {len(expected)}; '
                               f'missing {sorted(expected - got)[:5]}')
        if not frames:
            continue
        m = pd.concat(frames, ignore_index=True)
        m['in_dist_group'] = m['in_dist_group'].fillna('')
        rows = m.groupby('_file').size()
        check((rows == N_ARMS * N_METRICS).all(), f'{held}{kind}: rows per job file {sorted(set(rows))}')
        check(m['value'].notna().all(), f'{held}{kind}: missing values')
        check(not m.duplicated(['con_percent', 'seed', 'in_dist_group', 'arm', 'metric']).any(),
              f'{held}{kind}: duplicated rows')
        check(m['arm'].nunique() == N_ARMS and m['metric'].nunique() == N_METRICS, f'{held}{kind}: arms / metrics')
        check(set(m['in_dist_group']) == ({''} if kind == '' else set(rare)), f'{held}{kind}: in_dist_group')
        check((m['abundant'] == abundant).all(), f'{held}{kind}: design columns')
        df_ok = m.apply(lambda r: r['design_file'] ==
                        f'{data}-test{held}-main{abundant}-per{r["con_percent"]}-seed{r["seed"]}.h5ad', axis=1)
        check(df_ok.all(), f'{held}{kind}: design_file does not match percent / seed')
        out = f'{RES}/{data}/{data}-oodTrue-variable_conFalse-designed-test{held}-main{abundant}{kind}.csv'
        print(f'{held}/{abundant}{kind}: {len(got)} job files, {len(m)} rows' + (f' -> {out}' if write else ''),
              flush=True)
        if write:
            m.drop(columns='_file').to_csv(out, index=False)
print('ALL CHECKS PASS' if not problems else f'{len(problems)} PROBLEMS')
sys.exit(1 if problems else 0)
