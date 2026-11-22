"""Draw batch, annotation and reconstruction figures in the Figure 2 layout.

Each figure has a composite ranking table and four closed metric frames.
Raw MAE is not pooled across datasets because mouse heart uses a different
expression scale; that panel shows within-dataset ranks, with rank 1 at the top.
"""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import pymupdf

import draw_rq1_vertical_review as fig2
import draw_rq1_review as base
import task_layout as t
from audit_panel_alignment import require_matplotlib_panel_alignment

R = base.R
TASKS = {
    'correction': {
        'slug': 'fig03_correction',
        'stem': 'Native-embedding batch assessment',
        'supplement': 'S5--S8',
    },
    'annotation': {
        'slug': 'fig04_annotation',
        'stem': 'Supervised annotation',
        'supplement': 'S9--S12',
    },
    'reconstruction': {
        'slug': 'fig05_reconstruction',
        'stem': 'Supervised reconstruction',
        'supplement': 'S13--S16',
    },
}


def load_task(task):
    lookup = {}
    for panel in t.PANELS:
        path = R / 'data/exports' / panel / f'{task}_scores.csv'
        for row in t.readcsv(path):
            key = row['dataset'], int(row['seed']), row['model']
            assert key not in lookup
            lookup[key] = row
    assert len(lookup) == 576
    metrics = t.METRICS[task]
    raw = np.array([[[[float(lookup[d, s, m][k]) for k, _, _ in metrics]
                       for m in t.MODELS] for s in t.SEEDS] for d in t.DS])
    assert np.isfinite(raw).all()
    ranked = np.empty_like(raw)
    for di, si, ki in np.ndindex(12, 3, 4):
        values = raw[di, si, :, ki]
        higher = metrics[ki][2]
        score = values if higher else -values
        ranked[di, si, :, ki] = [1 + sum(score > v) + (sum(score == v) - 1) / 2 for v in score]
    reference = {(row['panel'], row['model']): float(row['mean_metric_rank'])
                 for row in t.readcsv(R / 'data/all_panel_ranks.csv') if row['task'] == task}
    panel_seed = np.stack([ranked[i * 4:i * 4 + 4].mean(axis=(0, 3)) for i in range(3)])
    for pi, panel in enumerate(t.PANELS):
        for mi, model in enumerate(t.MODELS):
            assert np.isclose(panel_seed[pi, :, mi].mean(), reference[panel, model])
    return raw, ranked


def y_scale(values, higher):
    if higher and values.min() >= 0 and values.max() <= 1:
        return 0, 1, [0, .25, .5, .75, 1], ['0', '0.25', '0.50', '0.75', '1.0']
    return None


def make_figure(task, raw, ranked):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
    })
    order, means, ordinal = fig2.previous.rank_summary(ranked)
    models = [t.MODELS[i] for i in order]
    raw = raw[:, :, order, :]
    ranked = ranked[:, :, order, :]
    ordinal = ordinal[order]
    dataset_rank_means = ranked.mean(axis=(1, 3)).T
    fig = plt.figure(figsize=(fig2.W / 25.4, fig2.H / 25.4))
    axes = [fig2.draw_integrated_table(fig, models, ordinal, dataset_rank_means)]
    seed_means = raw.mean(axis=1)
    rank_means = ranked.mean(axis=1)
    panel_orders = {}
    plot_seq = [3, 2, 0, 1] if task == 'reconstruction' else list(range(len(t.METRICS[task])))
    for plot_i, ki in enumerate(plot_seq):
        key, name, higher = t.METRICS[task][ki]
        shown = seed_means[:, :, ki] if higher else rank_means[:, :, ki]
        direction = -1 if higher else 1
        metric_order = np.argsort(direction * shown.mean(axis=0), kind='stable')
        panel_models = [models[i] for i in metric_order]
        panel_values = shown[:, metric_order]
        panel_orders['bcde'[plot_i]] = panel_models
        column = plot_i % 2
        x, y = fig2.METRIC_X[column], fig2.METRIC_Y[0 if plot_i < 2 else 1]
        ax = fig2.axmm(fig, x, y, fig2.METRIC_W, fig2.METRIC_H)
        label_y = fig2.METRIC_LABEL_Y[0 if plot_i < 2 else 1]
        fig2.panel_letter(fig, column, label_y, 'bcde'[plot_i])
        title = name if higher else name + ' rank'
        fig.text((x + 39) / fig2.W, label_y / fig2.H, title, fontsize=8,
                 ha='center', va='bottom', color=fig2.INK)
        bp = ax.boxplot([panel_values[:, mi] for mi in range(16)], positions=np.arange(16),
                        widths=.56, whis=(0, 100), showfliers=False, patch_artist=True,
                        manage_ticks=False, boxprops=dict(ec='#31627F', lw=.75),
                        medianprops=dict(color=fig2.INK, lw=.95),
                        whiskerprops=dict(color='#31627F', lw=.7),
                        capprops=dict(color='#31627F', lw=.7))
        for mi, model in enumerate(panel_models):
            color = fig2.METHOD_COLORS[model]
            edge = tuple(.75 * np.array(fig2.to_rgb(color)))
            bp['boxes'][mi].set_facecolor(color)
            bp['boxes'][mi].set_edgecolor(edge)
            bp['medians'][mi].set_color(fig2.contrasting_ink(fig2.to_rgb(color)))
            for line in bp['whiskers'][2 * mi:2 * mi + 2] + bp['caps'][2 * mi:2 * mi + 2]:
                line.set_color(edge)
            assert np.isclose(bp['medians'][mi].get_ydata()[0], np.median(panel_values[:, mi]))
        ax.set_xlim(-.6, 15.6)
        ax.set_xticks(range(16), [t.display(model) for model in panel_models])
        ax.tick_params(axis='x', which='both', top=False, labeltop=False, bottom=True,
                       labelbottom=True, length=2.5, width=.8, pad=1.5, direction='out',
                       labelsize=8)
        ax.tick_params(axis='y', which='both', right=False, labelright=False, left=True,
                       labelleft=True, length=3, width=.8, pad=2, direction='out',
                       labelsize=8)
        for label in ax.get_xticklabels():
            label.set_rotation(48)
            label.set_ha('right')
            label.set_va('top')
            label.set_rotation_mode('anchor')
            label.set_fontweight('normal')
            label.set_fontsize(8)
            label.set_clip_on(False)
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_color(fig2.INK)
            side.set_linewidth(.8)
        if higher:
            assert np.all(np.diff(panel_values.mean(axis=0)) <= 1e-12)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.0'])
        else:
            assert np.all(np.diff(panel_values.mean(axis=0)) >= -1e-12)
            ax.set_ylim(16.5, 0.5)
            ax.set_yticks([1, 4, 8, 12, 16])
        ax.grid(axis='y', color='#E5EAEE', lw=.45, zorder=0)
        ax.set_axisbelow(True)
        axes.append(ax)
    return fig, axes, models, means, ordinal, dataset_rank_means, panel_orders


def caption(task, meta):
    metrics = ', '.join(name if high else name + ' rank' for _, name, high in t.METRICS[task])
    mae = (' The MAE panel shows within-dataset ranks because raw MAE is not on a shared scale '
           'across datasets; rank 1 is at the top and is better.') if task == 'reconstruction' else ''
    return (f'{meta["stem"]}. a, Composite ranking table. Overall and each context have an '
            'ordinal-rank number and bar; each context also has four dataset columns. Bar length '
            'is proportional to 17 minus the ordinal rank; longer bars are better. Dataset cells '
            'give mean ranks over four metrics and three seeds; darker cells indicate lower, better '
            'mean ranks. These rankings have no uncertainty bars. b--e, Closed frames for '
            f'{metrics}. Each score panel orders methods by descending mean score across twelve '
            'datasets. Each box contains twelve dataset means after averaging three seeds, with '
            'quartile boxes, median lines and minimum--maximum whiskers. Higher scores are better '
            'except where the panel shows ranks.' + mae + ' Full method names are tilted on the '
            'x-axis, and each method has one fixed color. Individual seed symbols are omitted. '
            f'Dataset-resolved three-seed boxplots remain in Supplementary Figs.~{meta["supplement"]}. '
            'No confidence intervals or hypothesis tests are shown.')


def main():
    sys.path.insert(0, str(R / 'analysis/vendor/scientific_visualization/scripts'))
    from figure_export import export_figure
    from style_presets import style_context
    for task, meta in TASKS.items():
        out = R / 'figures/task_rank_review' / meta['slug']
        out.mkdir(parents=True, exist_ok=True)
        raw, ranked = load_task(task)
        with style_context('default', palette_name='okabe_ito_on_white'):
            plt.rcParams.update({'font.family': 'sans-serif',
                                 'font.sans-serif': ['Arial', 'Arial'],
                                 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                                 'ytick.labelsize': 8, 'text.color': fig2.INK,
                                 'axes.labelcolor': fig2.INK, 'axes.edgecolor': '#81909B',
                                 'axes.linewidth': .65, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                                 'axes.grid': False})
            fig, axes, models, means, ordinal, tiles, orders = make_figure(task, raw, ranked)
        text = caption(task, meta)
        (out / 'caption.txt').write_text(text + '\n')
        export_figure(fig, out / meta['slug'], formats=['pdf', 'svg', 'png'], dpi=600,
                      font_mode='truetype', bbox_inches=None, transparent=False, overwrite=True,
                      write_manifest=True, provenance={'task': task, 'script': Path(__file__).name})
        fig.savefig(out / 'preview.png', dpi=300, facecolor='white')
        for filename, dpi in [(meta['slug'] + '.png', 600), ('preview.png', 300)]:
            with Image.open(out / filename) as image:
                rgb = image.convert('RGB')
            rgb.save(out / filename, dpi=(dpi, dpi))
        require_matplotlib_panel_alignment(
            fig, axes=axes, panel_ids=list('abcde'),
            row_groups=[['b', 'c'], ['d', 'e']], column_groups=[['b', 'd'], ['c', 'e']],
            json_out=str(out / 'alignment.json'), overlay_svg=str(out / 'alignment.svg'),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
        pdf = out / (meta['slug'] + '.pdf')
        with pymupdf.open(pdf) as document:
            assert len(document) == 1 and not document[0].get_images()
        (out / 'verification.json').write_text(json.dumps({
            'task': task, 'methods': models, 'width_mm': fig2.W, 'height_mm': fig2.H,
            'vector_pdf': True, 'png_dpi': 600, 'mae_panel': 'within-dataset rank, 1 at top'
            if task == 'reconstruction' else None,
        }, indent=2))
        plt.close(fig)
        print(task, 'OK', pdf)


if __name__ == '__main__':
    main()
