"""Perturbation figure in the Figure 2 layout: ranking table plus four closed frames.

Four datasets replace the three biological contexts. MAE delta is shown as a
within-dataset rank because Norman/Adamson and the Replogle studies do not share
an expression scale.
"""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image
import pymupdf

import draw_rq1_vertical_review as fig2
import perturbation_rq as pert
import task_layout as t
from audit_panel_alignment import require_matplotlib_panel_alignment

R = fig2.R
SLUG = 'fig06_perturbation_rank'
OUT = R / 'figures/perturbation_rank_review' / SLUG
DS = pert.DS
LABELS = ['Norman', 'Adamson', 'K562', 'RPE1']
METRICS = [
    ('Pearson_delta', 'Pearson Δ', True),
    ('Spearman_delta', 'Spearman Δ', True),
    ('MAE_delta', 'MAE Δ rank', False),
    ('DES', 'DES', True),
]
COLUMN_COLORS = ['#457B9D', '#C45C86', '#2A9D8F', '#C48A2A']


def load():
    rows = t.readcsv(R / 'data/exports/perturbation/perturbation_scores.csv')
    lookup = {(row['dataset'], int(row['seed']), row['model']): row for row in rows}
    assert len(lookup) == 192
    raw = np.array([[[[float(lookup[d, s, m][key]) for key, _, _ in METRICS]
                       for m in t.MODELS] for s in t.SEEDS] for d in DS])
    assert np.isfinite(raw).all()
    ranked = np.empty_like(raw)
    for di, si, ki in np.ndindex(4, 3, 4):
        values = raw[di, si, :, ki]
        higher = METRICS[ki][2]
        ranked[di, si, :, ki] = t.ranks(values, high=higher)
    reference = {row['model']: float(row['mean_metric_rank'])
                 for row in t.readcsv(R / 'data/all_panel_ranks.csv') if row['task'] == 'perturbation'}
    for mi, model in enumerate(t.MODELS):
        assert np.isclose(ranked[:, :, mi, :].mean(), reference[model])
    return raw, ranked


def ordinal_ranks(ranked):
    sums = np.column_stack([ranked.sum(axis=(0, 1, 3))] +
                           [ranked[i].sum(axis=(0, 2)) for i in range(4)])
    ordinal = np.column_stack([t.ranks(sums[:, j]) for j in range(5)])
    means = sums / np.array([48, 12, 12, 12, 12])
    order = np.argsort(sums[:, 0], kind='stable')
    return order, means, ordinal


def draw_table(fig, models, ordinal, dataset_means):
    left, bottom, width, height = fig2.TABLE['left'], fig2.TABLE['bottom'], fig2.TABLE['width'], fig2.TABLE['height']
    header_y, header_h, label_y = fig2.TABLE['header_y'], fig2.TABLE['header_h'], fig2.TABLE['label_y']
    ax = fig2.axmm(fig, left, bottom, width, height)
    ax.set(xlim=(0, width), ylim=(15.5, -.5))
    ax.axis('off')
    for i, model in enumerate(models):
        ax.add_patch(Rectangle((0, i - .5), width, 1, fc='#E8ECEF' if i % 2 == 0 else '#F8F9FA', lw=0))
        ax.text(24.5, i, t.display(model), ha='right', va='center', fontsize=8, color=fig2.INK)

    def header(x, w, label, color):
        fig.add_artist(Rectangle(((left + x) / fig2.W, header_y / fig2.H), w / fig2.W, header_h / fig2.H,
                                transform=fig.transFigure, fc=color, lw=0))
        fig.text((left + x + w / 2) / fig2.W, (header_y + header_h / 2) / fig2.H, label, ha='center', va='center',
                 fontsize=8, color='white')

    def rank_column(start, maximum, column, color):
        for i in range(16):
            value = ordinal[i, column]
            ax.text(start + 3.4, i, f'{value:g}', ha='right', va='center', fontsize=8, color=fig2.INK)
            ax.add_patch(Rectangle((start + 4.5, i - .30), maximum * (17 - value) / 16, .60,
                                  fc=color, ec='#3E515C', lw=.55))
        fig.text((left + start + (4.5 + maximum) / 2) / fig2.W, label_y / fig2.H, 'Rank',
                 fontsize=8, ha='center', va='bottom', color=fig2.INK)

    fig2.panel_letter(fig, 0, header_y + header_h + 0.8, 'a')
    header(0, 26, 'Method', '#73777C')
    header(28, 24, 'Overall', '#457B9D')
    rank_column(32, 16, 0, '#8CB1C8')
    group = 31.2
    for di, (label, color) in enumerate(zip(LABELS, COLUMN_COLORS)):
        start = 54 + di * group
        header(start, group - .6, label, color)
        rank_column(start + 1.5, 9, di + 1, tuple(.45 + .55 * np.array(fig2.to_rgb(color))))
        cmap = LinearSegmentedColormap.from_list('ds' + str(di), ['#F3F6F8', color])
        tile_x = start + 16.4
        for i in range(16):
            value = dataset_means[i, di]
            rgba = cmap((16 - value) / 15)
            ax.add_patch(Rectangle((tile_x, i - .42), 12.5, .84, fc=rgba, lw=0))
            ax.text(tile_x + 6.25, i, f'{value:.1f}', fontsize=6, ha='center', va='center',
                    color=fig2.contrasting_ink(rgba),
                    fontweight='bold' if np.isclose(value, dataset_means[:, di].min()) else 'normal')
    return ax


def make_figure(raw, ranked):
    order, means, ordinal = ordinal_ranks(ranked)
    models = [t.MODELS[i] for i in order]
    raw = raw[:, :, order, :]
    ranked = ranked[:, :, order, :]
    ordinal = ordinal[order]
    dataset_means = ranked.mean(axis=(1, 3)).T
    fig = plt.figure(figsize=(fig2.W / 25.4, fig2.H / 25.4))
    axes = [draw_table(fig, models, ordinal, dataset_means)]
    seed_means = raw.mean(axis=1)
    rank_means = ranked.mean(axis=1)
    for ki, (_, name, higher) in enumerate(METRICS):
        shown = seed_means[:, :, ki] if higher else rank_means[:, :, ki]
        direction = -1 if higher else 1
        metric_order = np.argsort(direction * shown.mean(axis=0), kind='stable')
        panel_models = [models[i] for i in metric_order]
        panel_values = shown[:, metric_order]
        column = ki % 2
        x, y = fig2.METRIC_X[column], fig2.METRIC_Y[0 if ki < 2 else 1]
        ax = fig2.axmm(fig, x, y, fig2.METRIC_W, fig2.METRIC_H)
        label_y = fig2.METRIC_LABEL_Y[0 if ki < 2 else 1]
        fig2.panel_letter(fig, column, label_y, 'bcde'[ki])
        fig.text((x + 39) / fig2.W, label_y / fig2.H, name, fontsize=8, ha='center', va='bottom', color=fig2.INK)
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
        ax.set_xlim(-.6, 15.6)
        ax.set_xticks(range(16), [t.display(model) for model in panel_models])
        ax.tick_params(axis='x', which='both', top=False, labeltop=False, bottom=True,
                       labelbottom=True, length=2.5, width=.8, pad=1.5, direction='out')
        ax.tick_params(axis='y', which='both', right=False, labelright=False, left=True,
                       labelleft=True, length=3, width=.8, pad=2, direction='out')
        for label in ax.get_xticklabels():
            label.set_rotation(48)
            label.set_ha('right')
            label.set_va('top')
            label.set_rotation_mode('anchor')
            label.set_fontweight('normal')
            label.set_clip_on(False)
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_color(fig2.INK)
            side.set_linewidth(.8)
            side.set_zorder(3)
        if higher:
            ax.set_ylim(0, 1)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.0'])
        else:
            ax.set_ylim(16.5, 0.5)
            ax.set_yticks([1, 4, 8, 12, 16])
        ax.grid(axis='y', color='#E5EAEE', lw=.45, zorder=0)
        ax.set_axisbelow(True)
        axes.append(ax)
    return fig, axes, models


def main():
    sys.path.insert(0, str(R / 'analysis/vendor/scientific_visualization/scripts'))
    from figure_export import export_figure
    from style_presets import style_context
    OUT.mkdir(parents=True, exist_ok=True)
    raw, ranked = load()
    with style_context('default', palette_name='okabe_ito_on_white'):
        plt.rcParams.update({'font.family': 'sans-serif',
                             'font.sans-serif': ['Arial', 'Arial'],
                             'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                             'text.color': fig2.INK, 'axes.edgecolor': '#81909B'})
        fig, axes, models = make_figure(raw, ranked)
    require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=list('abcde'),
        row_groups=[['b', 'c'], ['d', 'e']], column_groups=[['b', 'd'], ['c', 'e']],
        json_out=str(OUT / 'alignment.json'), overlay_svg=str(OUT / 'alignment.svg'),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    export_figure(fig, OUT / SLUG, formats=['pdf', 'svg', 'png'], dpi=600, font_mode='truetype',
                  bbox_inches=None, transparent=False, overwrite=True, write_manifest=True,
                  provenance={'task': 'perturbation', 'script': Path(__file__).name})
    fig.savefig(OUT / 'preview.png', dpi=300, facecolor='white')
    for filename, dpi in [(SLUG + '.png', 600), ('preview.png', 300)]:
        with Image.open(OUT / filename) as image:
            image.convert('RGB').save(OUT / filename, dpi=(dpi, dpi))
    pdf = OUT / (SLUG + '.pdf')
    with pymupdf.open(pdf) as document:
        assert len(document) == 1 and not document[0].get_images()
    caption = ('Perturbation prediction on four held-out-condition datasets. a, Composite ranking '
               'table. Overall and each dataset have an ordinal-rank number and bar. Longer bars are '
               'better. Colored cells give mean ranks over four metrics and three seeds. b--e, Closed '
               'frames for Pearson delta, Spearman delta, MAE-delta rank and DES. Each box contains four '
               'dataset means after averaging three seeds. The MAE panel shows within-dataset ranks because '
               'raw error is not on a shared scale; rank 1 is at the top. Full method names are tilted on '
               'the x-axis. No confidence intervals or hypothesis tests are shown.')
    (OUT / 'caption.txt').write_text(caption + '\n')
    (OUT / 'verification.json').write_text(json.dumps({
        'methods': models, 'width_mm': fig2.W, 'height_mm': fig2.H, 'vector_pdf': True, 'png_dpi': 600,
    }, indent=2))
    plt.close(fig)
    print('OK', pdf)


if __name__ == '__main__':
    main()
