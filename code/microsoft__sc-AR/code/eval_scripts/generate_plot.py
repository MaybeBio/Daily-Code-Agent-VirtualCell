"""Plots and tables for the scGen designed sweep (Standard, Uniform and the AR
aggregation rules AR[-<rule>]-mu, weights from the posterior mean).

By-group plots (default): reads the merged metric csv files of all designs of
one dataset (eval_scripts/merge_eval.py) and writes into
<out_dir>/designed-by-group/seeds<first>-<last>/, for the held-out groups
('heldout') and the rare groups ('rare'):
  <data>-<kind>-seeds<a>-<b>-stats.csv   one row per metric x group x percentage
                                         x AR arm: mean and SD of the AR arm and
                                         of Standard, mean paired difference
                                         (AR - Standard), paired two-sided t-test
                                         p, and whether AR is better (direction
                                         depends on the metric)
  <data>-<kind>-seeds<a>-<b>-<metric>-bar.{pdf,png}
                                         one panel per cell group, the rare-cell
                                         percentage on the x axis, a '*' above
                                         each AR arm that is better than Standard
                                         with p < 0.05
heldout: each group as the held-out group of its design, paired on seed.
rare:    each group pooled over all designs where it is rare (its test cells),
         paired on design x seed; the per-design test (seeds averaged within
         each design, paired over designs) is added as a robustness check.
Uniform is plotted but not tested. Metrics are listed in METRICS.

Latent ARI tables (--latent_ari_tables): reads the latent_clustering.csv files
of eval_scripts/latent_clustering.py (cell type x condition task) and writes
<out_dir>/latent_ari_tables.md and latent_ari_tables.csv: per dataset, rows =
held-out group (abundant group), columns = percentage x Standard / AR-max /
AR-mean, cell = mean ARI over the seeds. Each AR rule is tested against
Standard with a two-sided paired t-test over the seeds (pair = same seed),
nominal P < 0.05: an AR value is bold if that rule is significantly higher than
Standard; Standard is bold if it is significantly higher than at least one AR
rule.

usage (from the repository root):
  python eval_scripts/generate_plot.py --data pbmc --seeds 700 701 702 703 704 \
      --results result/test/pbmc/pbmc-oodTrue-variable_conFalse-designed-test*-main*.csv
  python eval_scripts/generate_plot.py --latent_ari_tables \
      --results result/latent/*/*/latent_clustering.csv
"""
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import os


ARM_LABELS = {'Naive': 'Standard', 'AR': 'AR', 'Uniform': 'Uniform'}
PALETTE = {'Standard': 'darkgray', 'AR': 'brown', 'Uniform': 'steelblue'}
# metrics in plot order; True if higher is better
METRICS = {'Delta Pearson r': True, '20 up+down DEG Delta Pearson r': True,
           'Energy Distance': False}
# bar plots (mean +- SD, seed points) for every metric
DESIGNED_BAR_METRICS = list(METRICS)
# max rule brown as AR in the other plots, other AR rules in purple / violet / chocolate
DESIGNED_PALETTE = {'AR-mu': 'brown', 'AR-max': 'brown', 'AR-mean-mu': 'purple',
                    'AR-geomean-mu': 'violet', 'AR-knn-mu': 'chocolate'}
# legend names of the AR arms (aggregation rule only)
BY_GROUP_LABELS = {'AR-mu': 'AR (max)', 'AR-mean-mu': 'AR (mean)',
                   'AR-geomean-mu': 'AR (geometric mean)', 'AR-knn-mu': 'AR (kNN)'}
# Uniform tan, every AR rule brown (told apart by hatch)
BY_GROUP_PALETTE = {'Uniform': 'tan', 'AR-mu': 'brown', 'AR-mean-mu': 'brown',
                    'AR-geomean-mu': 'brown', 'AR-knn-mu': 'brown'}
BY_GROUP_HATCHES = {'AR-mean-mu': '//', 'AR-geomean-mu': '..', 'AR-knn-mu': '--'}
KEY =['test', 'arm', 'seed', 'con_percent', 'metric']
# latent ARI tables: arms in column order, dataset names in the section titles
ARI_ARMS = ['Standard', 'AR-max', 'AR-mean']
ARI_DATA = {'species': 'Species', 'pbmc': 'PBMC', 'lps-hpoly': 'H. poly'}


def arm_order(models):
    """Standard, AR, Uniform first (as before), then the AR variants in sorted order."""

    first = [m for m in ['Standard', 'AR', 'Uniform'] if m in models]
    return first + sorted(set(models) - set(first))


def palette_for(models):
    """Fixed colours for Standard, AR and Uniform, a tab20 colour for every other arm."""

    rest = [m for m in models if m not in PALETTE]
    return {**{m: PALETTE[m] for m in models if m in PALETTE},
            **dict(zip(rest, sns.color_palette('tab20', len(rest))))}


def check_metrics(df, allow_missing_metrics):
    """Check that every run (test, con_percent, arm, seed) has all METRICS.

    Args:
        df (pd.DataFrame): results read by load_results
        allow_missing_metrics (bool): warn instead of raising an error

    Raises:
        ValueError: if a metric is missing and allow_missing_metrics is False"""

    unknown = sorted(set(df['metric']) - set(METRICS))
    if unknown:
        print(f'Warning: metrics not used by this script: {unknown}')

    run = ['test', 'con_percent', 'arm', 'seed']
    present = df.groupby(run)['metric'].agg(set)
    missing = present.map(lambda s: [m for m in METRICS if m not in s])
    missing = missing[missing.map(len) > 0]
    if len(missing):
        names = sorted({m for ms in missing for m in ms}, key=list(METRICS).index)
        msg = (f'{len(missing)} of {len(present)} runs miss metrics: {names}. '
               f'Example runs:\n{missing.head().to_string()}\n'
               'Rerun eval.py, or pass --allow_missing_metrics to plot the available metrics only.')
        if not allow_missing_metrics:
            raise ValueError(msg)
        print('Warning: '+msg)

    return


def load_results(paths, n_seeds, allow_missing_metrics=False, seeds=None):
    """Read eval.py csv files and keep one value per test, arm, seed,
    con_percent and metric.

    Args:
        paths (list): csv files written by eval.py (merged by merge_eval.py)
        n_seeds (int): expected number of seeds per arm and group
        allow_missing_metrics (bool, optional): warn instead of raising an error
            when a run misses some of METRICS
        seeds (list, optional): keep only these seeds

    Returns:
        pd.DataFrame: results with a 'model' column (Standard, Uniform, AR arms),
            the evaluated group in 'test'"""

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if seeds:
        df = df[df['seed'].isin(seeds)]
    if 'arm' not in df.columns:
        raise ValueError('No "arm" column: these results were not written by eval.py.')

    df['in_dist_group'] = df['in_dist_group'].fillna('').astype(str)
    # rare training groups, evaluated on their test cells
    df.loc[df['in_dist_group'] != '', 'test'] = df['in_dist_group']
    df['con_percent'] = df['con_percent'].astype(float)
    df['value'] = df['value'].astype(float)

    # eval.py appends, so a rerun of the same evaluation adds duplicate rows
    # designed results: rows of different designs are not duplicates
    key = KEY + [c for c in ('abundant', 'design_file') if c in df.columns]
    n_dup = df.duplicated(key, keep='last').sum()
    if n_dup:
        print(f'Warning: {n_dup} duplicate rows (repeated evaluations); keeping the last of each.')
        df = df.drop_duplicates(key, keep='last')

    check_metrics(df, allow_missing_metrics)

    df['model'] =df['arm'].map(lambda a: ARM_LABELS.get(a, a))

    counts = df.groupby(['test', 'con_percent', 'model', 'metric'])['seed'].nunique()
    incomplete = counts[counts != n_seeds]
    if len(incomplete):
        print(f'Warning: {len(incomplete)} groups do not have {n_seeds} seeds:')
        print(incomplete.to_string())

    return df


def paired_test(df, other, higher_is_better, arm='AR', pair_col='seed'):
    """Paired two-sided t-test of an AR arm vs. another arm, matched on pair_col.

    Args:
        df (pd.DataFrame): rows of one metric and group, with pair_col, 'model', 'value'
        other (str): arm compared with the AR arm
        higher_is_better (bool): direction of the metric
        arm (str, optional): AR arm to test ('mean_AR' and 'sd_AR' refer to it)
        pair_col (str, optional): column of the matched units ('seed'; 'design_file'
            pairs on design x seed when several designs are pooled)

    Returns:
        dict: means, SDs, mean difference (AR - other), p, n and whether AR is better"""

    wide = df.pivot_table(index=pair_col, columns='model', values='value')
    wide = wide[[arm, other]].dropna()
    diff = wide[arm] - wide[other]
    p = ttest_rel(wide[arm], wide[other]).pvalue if len(wide) > 1 else np.nan

    return {'comparison': arm+' vs '+other, 'n_seeds': len(wide),
            'mean_AR': wide[arm].mean(), 'sd_AR': wide[arm].std(),
            'mean_other': wide[other].mean(), 'sd_other': wide[other].std(),
            'mean_diff': diff.mean(), 'sd_diff': diff.std(), 'p': p,
            'AR_better': bool(diff.mean() > 0) == higher_is_better}


def compute_designed_stats(df, metrics, pair_col='seed', per_design=False):
    """Paired tests of every AR arm vs. Standard for each metric, evaluated
    group and percentage (designed experiment).

    Args:
        df (pd.DataFrame): results read by load_results
        metrics (list): metrics to test
        pair_col (str, optional): matched units of the paired test ('seed', or
            'design_file' = design x seed when several designs are pooled; then
            'n_pairs' = number of pairs)
        per_design (bool, optional): also test on design means (seeds averaged
            within each design, paired over designs): 'n_designs',
            'mean_diff_design', 'p_design' (robustness of the pooled test)

    Returns:
        pd.DataFrame: one row per metric x group x percentage x AR arm, with
            'star' (p < 0.05 and the AR arm better) and 'star_worse' (p < 0.05
            and Standard better)"""

    ar_arms = [m for m in arm_order(set(df['model'])) if m.startswith('AR')]
    rows = []
    for metric in metrics:
        df_m = df[df['metric'] == metric]
        for (test, per), df_g in df_m.groupby(['test', 'con_percent']):
            for arm in ar_arms:
                row = {'metric': metric, 'test': test, 'con_percent': per,
                       **paired_test(df_g, 'Standard', METRICS[metric], arm, pair_col)}
                if pair_col != 'seed':
                    row['n_pairs'] = row['n_seeds']
                if per_design:
                    design = df_g.assign(design=df_g['design_file'].str.replace(r'-per.*$', '', regex=True))
                    means = design.groupby(['design', 'model'], as_index=False)['value'].mean()
                    d = paired_test(means, 'Standard', METRICS[metric], arm, 'design')
                    row.update({'n_designs': d['n_seeds'], 'mean_diff_design': d['mean_diff'],
                                'p_design': d['p']})
                rows.append(row)

    stats = pd.DataFrame(rows)
    stats['star'] = (stats['p'] < 0.05) & stats['AR_better']
    stats['star_worse'] = (stats['p'] < 0.05) & ~stats['AR_better']

    return stats


def plot_designed(df, stats, metric, kind, groups, held_out, path, title=None,
                  star_colors=('black', 'red'), star_row=False, labels=None, png_dpi=None,
                  xlabel='Cell percentage of each rare group in training', title_suffix='',
                  palette=None, hatches=None, show_worse=True, direction_arrow=False,
                  ylabel_size=None, title_size=None, xtick_size=None, legend_size=None,
                  bar_edgecolor=None):
    """One panel per evaluated group, percentage on the x axis, arms as hue;
    one '*' above each AR arm that is significantly better than Standard.

    Args:
        df (pd.DataFrame): results read by load_results
        stats (pd.DataFrame): output of compute_designed_stats
        metric (str): metric to plot
        kind (str): 'bar' or 'box'
        groups (list): evaluated groups in panel order
        held_out (list): groups not used in training
        path (str): output file
        title (str, optional): figure title
        star_colors (tuple, optional): colours of the '*' when the AR arm is
            better / worse than Standard
        star_row (bool, optional): put all '*' of a panel in one row above its
            highest bar / box (default: each '*' just above its arm)
        labels (dict, optional): legend names of the arms (default: arm names)
        png_dpi (int, optional): also save a png with this resolution next to path
        xlabel (str, optional): x axis label of the figure (None: no label)
        title_suffix (str, optional): added to each panel title, e.g. ' (OOD)'
        palette (dict, optional): arm colours on top of the default ones
        hatches (dict, optional): bar hatch per arm (bar plots only)
        show_worse (bool, optional): also mark the AR arms worse than Standard
        direction_arrow (bool, optional): add (↑) / (↓) for higher / lower is
            better to the y label
        ylabel_size (float, optional): y label font size (default: matplotlib's)
        title_size (float, optional): panel title font size (default: matplotlib's)
        xtick_size (float, optional): x tick label (percentage) font size
            (default: matplotlib's)
        legend_size (float, optional): legend and star note font size
            (default: matplotlib's)
        bar_edgecolor (str, optional): outline colour of every bar (bar plots
            only; default: no outline)"""

    sub = df[df['metric'] == metric].copy()
    if sub.empty:
        return
    pers = sorted(sub['con_percent'].unique())
    label = {p: f'{100 * p:g} %' for p in pers}
    sub['percent'] = sub['con_percent'].map(label)
    order = [label[p] for p in pers]
    hue_order = arm_order(set(sub['model']))
    palette = {**palette_for(hue_order),
               **{m: c for m, c in {**DESIGNED_PALETTE, **(palette or {})}.items()
                  if m in hue_order}}
    hatches = hatches or {}
    width = 0.8    # seaborn default: dodged hues share this width

    fig, axes = plt.subplots(1, len(groups), squeeze=False,
                             figsize=(len(groups) * (1 + 0.15 * len(hue_order) * len(order)), 4.5))
    for ax, g in zip(axes[0], groups):
        sub_g = sub[sub['test'] == g]
        common = dict(data=sub_g, x='percent', y='value', hue='model', order=order,
                      hue_order=hue_order, palette=palette, ax=ax, legend=False)
        if kind == 'bar':
            sns.barplot(**common, errorbar='sd', alpha=0.8, width=width)
            # one BarContainer per arm in hue_order; the hatch is drawn in the edge
            # colour, so black edge (outline only if bar_edgecolor is set)
            assert len(ax.containers) == len(hue_order), (g, metric)
            for m, bars in zip(hue_order, ax.containers):
                if m in hatches or bar_edgecolor:
                    for bar in bars:
                        bar.set(hatch=hatches.get(m, ''), edgecolor=bar_edgecolor or 'black',
                                linewidth=0.8 if bar_edgecolor else 0)
        else:
            sns.boxplot(**common, showfliers=False, width=width)
        sns.stripplot(**common, dodge=True, size=3, edgecolor='black', linewidth=0.5)

        ymin, ymax = ax.get_ylim()
        pad = 0.03 * (ymax - ymin)
        s = stats[(stats['metric'] == metric) & (stats['test'] == g)
                  & (stats['star'] | (stats['star_worse'] & show_worse))]
        # one row for the whole panel: above its highest value or mean + SD
        top = sub_g.groupby(['con_percent', 'model'])['value'].agg(
            lambda v: max(v.max(), v.mean() + v.std())).max()
        for _, row in s.iterrows():
            arm = row['comparison'].split(' vs ')[0]
            i, j = pers.index(row['con_percent']), hue_order.index(arm)
            vals = sub_g[(sub_g['con_percent'] == row['con_percent']) & (sub_g['model'] == arm)]['value']
            y = (top if star_row else max(vals.max(), vals.mean() + vals.std())) + pad
            ax.text(i - width / 2 + width / len(hue_order) * (j + 0.5), y, '*',
                    ha='center', va='bottom', size=14,
                    color=star_colors[0] if row['star'] else star_colors[1])
            ymax = max(ymax, y + 3 * pad)
        ax.set_ylim(ymin, ymax)

        ax.set_title(g[0].upper() + g[1:] + title_suffix, fontsize=title_size)
        ax.set_xlabel('')
        if xtick_size:
            ax.tick_params(axis='x', labelsize=xtick_size)
        ylabel = f"{metric} ({'↑' if METRICS[metric] else '↓'})" if direction_arrow else metric
        ax.set_ylabel(ylabel if ax is axes[0][0] else '', fontsize=ylabel_size)
        sns.despine(ax=ax)

    if xlabel:
        fig.supxlabel(xlabel)
    # same colour as the bars (seaborn barplot saturation 0.75, alpha 0.8)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=sns.desaturate(palette[m], 0.75), alpha=0.8,
                             hatch=hatches.get(m, ''), edgecolor=bar_edgecolor or 'black',
                             linewidth=0.8 if bar_edgecolor else 0)
               if kind == 'bar' and (m in hatches or bar_edgecolor) else
               plt.Rectangle((0, 0), 1, 1, color=sns.desaturate(palette[m], 0.75), alpha=0.8)
               if kind == 'bar' else plt.Rectangle((0, 0), 1, 1, color=palette[m])
               for m in hue_order]
    fig.legend(handles, [(labels or {}).get(m, m) for m in hue_order],
               bbox_to_anchor=(1, 0.9), loc='upper left', frameon=False, fontsize=legend_size)
    fig.text(1, 0.1, '* AR better', ha='left', va='bottom', color=star_colors[0],
             fontsize=legend_size)
    if show_worse:
        fig.text(1, 0.1, '* Standard better', ha='left', va='top', color=star_colors[1],
                 fontsize=legend_size)
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    if png_dpi:
        fig.savefig(os.path.splitext(path)[0] + '.png', dpi=png_dpi, bbox_inches='tight')
    plt.close(fig)

    return


def report_by_group(df, data, seed_tag, out_dir, metrics):
    """Designed sweep across designs, one panel per cell group (percentages on
    the x axis, arms as hue):
      heldout: each group as the held-out group of its design, paired on seed
      rare:    each group pooled over all designs where it is rare (test cells),
               paired on design x seed; per-design robustness test added
    Writes <data>-heldout-<seed_tag>-* and <data>-rare-<seed_tag>-* (stats csv
    and one figure per metric) into out_dir.

    Args:
        df (pd.DataFrame): results of all designs of one dataset, read by
            load_results (and one seed group)
        data (str): dataset name
        seed_tag (str): e.g. 'seeds700-704'
        out_dir (str): output directory
        metrics (list): metrics to test and plot"""

    if 'Standard' not in set(df['model']):
        raise ValueError('Need the Standard arm.')
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f'Unknown metrics {unknown}; choose from {list(METRICS)}')

    for kind_name, sub, pair_col, per_design in (
            ('heldout', df[df['in_dist_group'] == ''], 'seed', False),
            ('rare', df[df['in_dist_group'] != ''], 'design_file', True)):
        groups = sorted(sub['test'].unique())
        held_out = groups if kind_name == 'heldout' else []
        name = f'{data}-{kind_name}-{seed_tag}'
        stats = compute_designed_stats(sub, metrics, pair_col, per_design)
        stats.to_csv(os.path.join(out_dir, name+'-stats.csv'), index=False)
        for metric in metrics:
            kind = 'bar' if metric in DESIGNED_BAR_METRICS else 'box'
            # seaborn draws the jitter of the seed points from numpy's global
            # random generator: seed it so every run gives the same figure
            np.random.seed(0)
            # no figure title; only AR better than Standard marked, in black
            plot_designed(sub, stats, metric, kind, groups, held_out,
                          os.path.join(out_dir, f"{name}-{metric.replace(' ', '_')}-{kind}.pdf"),
                          star_colors=('black', 'red'), star_row=True,
                          labels=BY_GROUP_LABELS, png_dpi=400, xlabel=None,
                          title_suffix=' (OOD)' if kind_name == 'heldout' else ' (Rare)',
                          palette=BY_GROUP_PALETTE, hatches=BY_GROUP_HATCHES,
                          show_worse=False, direction_arrow=True,
                          # matplotlib defaults: y label 10 + 4, title 12 + 2, x ticks 10 + 2,
                          # legend 10 + 1
                          ylabel_size=14, title_size=14, xtick_size=12, legend_size=11,
                          bar_edgecolor='black')
        print('Wrote', os.path.join(out_dir, name+'-*'))

    return


def latent_ari_tables(paths, out_dir, seeds=None):
    """Mean ARI tables of the latent cell type x condition clustering, with
    each AR rule tested against Standard (two-sided paired t-test over seeds).

    Args:
        paths (list): latent_clustering.csv files (eval_scripts/latent_clustering.py)
        out_dir (str): output directory
        seeds (list, optional): keep only these seeds"""

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df[df['task'] == 'cell_group_x_condition']
    if seeds:
        df = df[df['seed'].isin(seeds)]
    if df.duplicated(['data', 'held_out', 'abundant', 'arm', 'per', 'seed']).any():
        raise ValueError('duplicated runs in the latent_clustering.csv files')
    seed_list = sorted(df['seed'].unique())
    pers = sorted(df['per'].unique())

    rows, lines = [], [
        f'# scGen latent ARI, cell type x condition, all cells (k = 2 x number of groups), '
        f'seeds {seed_list[0]}–{seed_list[-1]}', '',
        f'Cell: mean ARI over {len(seed_list)} seeds. Each AR rule is tested against Standard with a '
        f'two-sided paired t-test over the seeds (pair = same seed), nominal P < 0.05. '
        f'**Bold AR value** = that AR rule is significantly higher than Standard. '
        f'**Bold Standard** = Standard is significantly higher than at least one AR rule. '
        f'Rows: held-out group (abundant group).']
    for data in [d for d in ARI_DATA if d in set(df['data'])]:
        lines += ['', f'## {ARI_DATA[data]}', '',
                  '| Held-out (abundant) | ' + ' | '.join(f'{100 * p:g} % {a}' for p in pers for a in ARI_ARMS) + ' |',
                  '|---|' + '---|' * (len(pers) * len(ARI_ARMS))]
        df_d = df[df['data'] == data]
        for (held, abundant), df_h in df_d.groupby(['held_out', 'abundant']):
            cells = []
            for per in pers:
                wide = df_h[df_h['per'] == per].pivot(index='seed', columns='arm', values='ari')
                if sorted(wide.index) != seed_list or wide[ARI_ARMS].isna().any().any():
                    raise ValueError(f'{data} {held} {per}: not every arm has every seed')
                mean = wide[ARI_ARMS].mean()
                bold = {a: False for a in ARI_ARMS}
                for a in ARI_ARMS[1:]:
                    p = ttest_rel(wide[a], wide['Standard']).pvalue
                    diff = (wide[a] - wide['Standard']).mean()
                    rows.append(dict(data=data, held_out=held, abundant=abundant, per=per,
                                     comparison=f'{a} vs Standard', n_seeds=len(wide),
                                     mean_AR=mean[a], mean_Standard=mean['Standard'],
                                     mean_diff=diff, p=p))
                    if p < 0.05:
                        bold[a if diff > 0 else 'Standard'] = True
                cells += [f'**{mean[a]:.2f}**' if bold[a] else f'{mean[a]:.2f}' for a in ARI_ARMS]
            lines.append(f'| {held} ({abundant}) | ' + ' | '.join(cells) + ' |')

    with open(os.path.join(out_dir, 'latent_ari_tables.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'latent_ari_tables.csv'), index=False)
    print('Wrote', os.path.join(out_dir, 'latent_ari_tables.{md,csv}'))

    return


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=None,
                        help='species, pbmc or lps-hpoly (by-group plots)')
    parser.add_argument('--results', nargs='+', required=True,
                        help='merged eval csv files of all designs of one dataset, or with '
                             '--latent_ari_tables the latent_clustering.csv files')
    parser.add_argument('--out_dir', default=None,
                        help='output directory (default result/figures/<data>/, '
                             'with --latent_ari_tables result/latent/)')
    parser.add_argument('--n_seeds', type=int, default=5, help='expected seeds per group')
    parser.add_argument('--allow_missing_metrics', action='store_true',
                        help='warn instead of stopping when runs miss some metrics')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='use only these seeds (e.g. 700 701 702 703 704)')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='plot / test only these metrics (default METRICS)')
    parser.add_argument('--latent_ari_tables', action='store_true',
                        help='write the latent ARI tables instead of the by-group plots')
    args = parser.parse_args()

    if args.latent_ari_tables:
        out_dir = args.out_dir or os.path.join('result', 'latent')
        os.makedirs(out_dir, exist_ok=True)
        latent_ari_tables(args.results, out_dir, args.seeds)
        return
    if args.data is None:
        raise ValueError('--data is needed for the by-group plots')

    out_dir = args.out_dir or os.path.join('result', 'figures', args.data)
    os.makedirs(out_dir, exist_ok=True)

    df = load_results(args.results, args.n_seeds, args.allow_missing_metrics, args.seeds)

    # all designs of one dataset: one panel per cell group, held out and rare
    if 'design_file' not in df.columns:
        raise ValueError('needs the designed sweep csv files (with design_file)')
    if not df['experiment_id'].str.startswith(f'{args.data}-ood').all():
        raise ValueError(f'needs the csv files of one dataset ({args.data})')
    seeds = sorted(df['seed'].unique())
    seed_tag = f'seeds{seeds[0]}-{seeds[-1]}'
    sub_dir = os.path.join(out_dir, 'designed-by-group', seed_tag)
    os.makedirs(sub_dir, exist_ok=True)
    report_by_group(df, args.data, seed_tag, sub_dir, args.metrics or list(METRICS))

    return


if __name__ == '__main__':
    main()
