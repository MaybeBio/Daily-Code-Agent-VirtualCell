"""Color each method consistently; metric panels use methods on the x-axis.

576 completed clustering runs, 12 datasets, 3 seeds, 16 methods, 4 metrics.
Bars display ordinal ranks; tiles display dataset composite mean ranks.
Independent review exports only; existing manuscript files remain untouched.
"""
from pathlib import Path
import json
import sys
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle
from PIL import Image
import pymupdf
from pptx import Presentation
from pptx.util import Inches

import draw_rq1_review as base
import draw_rq1_ranktable_review as previous
import task_layout as t
from perturbation_rq import add_slide
from audit_panel_alignment import require_matplotlib_panel_alignment

R = base.R
O = R / 'figures/rq1_vertical_review/fig02_clustering'
SLUG = 'fig02_clustering_vertical_review'
W, H = 190, 172
TABLE = dict(left=7, bottom=104, width=179, height=53,
             header_y=161.6, header_h=4.2, label_y=157.8)
METRIC_X = (12, 104)
METRIC_Y = (75.4, 28)
METRIC_W, METRIC_H = 78, 22
METRIC_LABEL_Y = (98.6, 51.2)
INK = previous.INK
COLORS = previous.COLORS
RED = previous.RED
METHOD_COLORS = {
    'CellFM': '#332288', 'CellPLM': '#F28E2B', 'GeneCompass': '#882255',
    'Geneformer': '#CCAA44', 'LangCell': '#009988', 'Nicheformer': '#AA77BB',
    'Tabula': '#999933', 'TranscriptFormer': '#8C6D31', 'UCE': '#4477AA',
    'scFoundation': '#AA3377', 'scGPT': '#EE6677', 'scLong': '#78A2C5',
    'scPRINT2': '#44AA99', 'PCA_model': '#59A14F', 'UMAP': '#79706E',
    'scVI': '#66CCEE',
}
assert set(METHOD_COLORS) == set(t.MODELS) and len(set(METHOD_COLORS.values())) == 16
METHOD_SHORT = {
    'CellFM': 'CellFM', 'CellPLM': 'CellPLM', 'GeneCompass': 'GeneComp.',
    'Geneformer': 'Geneform.', 'LangCell': 'LangCell', 'Nicheformer': 'Niche.',
    'Tabula': 'Tabula', 'TranscriptFormer': 'TransF.', 'UCE': 'UCE',
    'scFoundation': 'scFound.', 'scGPT': 'scGPT', 'scLong': 'scLong',
    'scPRINT2': 'scP2', 'PCA_model': 'PCA', 'UMAP': 'UMAP', 'scVI': 'scVI',
}
assert set(METHOD_SHORT) == set(t.MODELS) and len(set(METHOD_SHORT.values())) == 16
DATASET_LABELS = t.DS_NAMES
CAPTION = (
    'Clustering performance across biological contexts. a, Composite ranking '
    'table. Overall and each context have an ordinal-rank number and bar; each context also has '
    'four dataset columns. Within-dataset--seed--metric ranks use average ties. Overall summaries '
    'equally average twelve datasets, three seeds and four metrics; each context uses four '
    'datasets. Bar-column numbers are ordinal ranks of these composite means, with average ties, '
    'rather than the mean-rank values reported in the manuscript. Bar length is proportional '
    'to 17 minus the ordinal rank; longer bars are better. '
    'Dataset tiles give mean ranks over four metrics and three seeds, rounded to one decimal; '
    'darker tiles indicate lower, better mean ranks. Bold tile values identify the unrounded minimum in each dataset, including '
    'ties. Column labels are BM, HL, AH, AK, MS, UC, C19, BC, MM, ML, MH and MK. These descriptive rankings have no uncertainty bars. b-e, ARI, NMI, matching accuracy and '
    'accuracy. Panel a orders all sixteen methods by overall composite mean rank. Each of '
    'panels b-e independently orders the sixteen methods by the descending arithmetic mean of '
    'its metric across twelve datasets; exact ties retain the order from panel a. Each box '
    'contains twelve dataset values, after averaging three seeds within each dataset. Boxes '
    'span linearly interpolated quartiles with median lines and observed minimum--maximum '
    'whiskers; higher scores are better. In b-e, full method names are on the x-axis, sorted from high to low '
    'mean score, with one fixed color per method; each box contains 12 dataset means. Exact ties retain the '
    'panel-a order. Each method has one fixed color across the four '
    'metric panels. All method names use regular weight, and PCA, UMAP and scVI use the same '
    'filled-box style as the other methods. Individual seed symbols are omitted. Seed-level '
    'values and unrounded ranking values accompany the figure. Supplementary Figs. S1-S4 retain '
    'dataset-specific three-seed boxplots. Heterogeneous datasets are not asserted to be '
    'independent biological replicates. No confidence intervals or hypothesis tests are shown.'
)


def axmm(fig, x, y, width, height):
    return fig.add_axes([x / W, y / H, width / W, height / H])


# Nature-style panel labels: letter only, one left edge per column, one baseline per row.
LETTER_X = (1.6, 93.6)


def panel_letter(fig, column, y, letter):
    fig.text(LETTER_X[column] / W, y / H, letter, fontsize=8, weight='bold',
             ha='left', va='bottom', color=INK)


def contrasting_ink(rgba):
    rgb = np.array(rgba[:3])
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = linear @ np.array([.2126, .7152, .0722])
    black_luminance = ((np.array(to_rgb(INK)) + .055) / 1.055) ** 2.4 @ np.array([.2126, .7152, .0722])
    return 'white' if 1.05 / (luminance + .05) > (luminance + .05) / (black_luminance + .05) else INK


def draw_integrated_table(fig, models, ordinal, dataset_means):
    # One shared coordinate system prevents independent row or text drift.
    # Horizontal two-letter labels sit on one baseline with Rank, packed under the headers.
    left, bottom, width, height = TABLE['left'], TABLE['bottom'], TABLE['width'], TABLE['height']
    header_y, header_h, label_y = TABLE['header_y'], TABLE['header_h'], TABLE['label_y']
    ax = axmm(fig, left, bottom, width, height)
    ax.set(xlim=(0, width), ylim=(15.5, -.5)); ax.axis('off')
    for i, model in enumerate(models):
        ax.add_patch(Rectangle((0, i - .5), width, 1, fc='#E8ECEF' if i % 2 == 0 else '#F8F9FA', lw=0))
        ax.text(24.5, i, t.display(model), ha='right', va='center', fontsize=8,
                color=INK, fontweight='normal')

    def header(x, w, label, color):
        fig.add_artist(Rectangle(((left + x) / W, header_y / H), w / W, header_h / H,
                                transform=fig.transFigure, fc=color, lw=0))
        fig.text((left + x + w / 2) / W, (header_y + header_h / 2) / H, label, ha='center', va='center',
                 fontsize=8, color='white')

    def rank_column(start, maximum, column, color):
        for i in range(16):
            value = ordinal[i, column]
            ax.text(start + 3.4, i, f'{value:g}', ha='right', va='center', fontsize=8,
                    color=INK, fontweight='normal')
            ax.add_patch(Rectangle((start + 4.5, i - .30), maximum * (17 - value) / 16, .60,
                                  fc=color, ec='#3E515C', lw=.55))
        fig.text((left + start + (4.5 + maximum) / 2) / W, label_y / H, 'Rank',
                 fontsize=8, ha='center', va='bottom', color=INK)

    panel_letter(fig, 0, header_y + header_h + 0.8, 'a')
    header(0, 25, 'Method', '#73777C')
    header(27, 17.3, 'Overall', '#457B9D')
    rank_column(27, 12.5, 0, '#8CB1C8')
    group_width = 133 / 3
    for pi, (label, color) in enumerate(zip(t.PANEL_NAMES, COLORS)):
        start = 46 + pi * group_width
        header(start, group_width - .6, label, color)
        rank_column(start, 8.8, pi + 1, tuple(.42 + .58 * np.array(to_rgb(color))))
        cmap = LinearSegmentedColormap.from_list('dataset_' + str(pi), ['#F3F6F8', color])
        tile_start, tile_pitch = start + 15, 7.15
        for di in range(4):
            col = pi * 4 + di
            tile_x = tile_start + tile_pitch * di
            fig.text((left + tile_x + tile_pitch / 2) / W, label_y / H, DATASET_LABELS[col],
                     fontsize=8, ha='center', va='bottom', color=INK)
            for i in range(16):
                value = dataset_means[i, col]
                rgba = cmap((16 - value) / 15)
                ax.add_patch(Rectangle((tile_x + .25, i - .42), tile_pitch - .5, .84,
                                      fc=rgba, lw=0))
                ax.text(tile_x + tile_pitch / 2, i, f'{value:.1f}', fontsize=6,
                        ha='center', va='center', color=contrasting_ink(rgba),
                        fontweight='bold' if np.isclose(value, dataset_means[:, col].min()) else 'normal')

    return ax


def make_figure(raw, ranked):
    from style_presets import style_context
    order, means, ordinal = previous.rank_summary(ranked)
    models = [t.MODELS[i] for i in order]
    raw, ranked, ordinal = raw[:, :, order, :], ranked[:, :, order, :], ordinal[order]
    dataset_means = ranked.mean(axis=(1, 3)).T
    with style_context('default', palette_name='okabe_ito_on_white'):
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Arial'],
            'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
            'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': '#81909B',
            'axes.linewidth': .65, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.grid': False})
        fig = plt.figure(figsize=(W / 25.4, H / 25.4))
        axes = [draw_integrated_table(fig, models, ordinal, dataset_means)]
        values = raw.mean(axis=1)
        panel_orders = {}
        # Partition agreement (ARI, NMI, Accuracy) then continuous separation (ASW).
        plot_seq = [0, 1, 3, 2]
        for plot_i, ki in enumerate(plot_seq):
            _, name, _ = t.METRICS['clustering'][ki]
            metric_order = np.argsort(-values[:, :, ki].mean(axis=0), kind='stable')
            panel_models = [models[i] for i in metric_order]
            panel_values = values[:, metric_order, ki]
            panel_orders['bcde'[plot_i]] = panel_models
            column = plot_i % 2
            x, y = METRIC_X[column], METRIC_Y[0 if plot_i < 2 else 1]
            ax = axmm(fig, x, y, METRIC_W, METRIC_H)
            label_y = METRIC_LABEL_Y[0 if plot_i < 2 else 1]
            panel_letter(fig, column, label_y, 'bcde'[plot_i])
            fig.text((x + 39) / W, label_y / H, name, fontsize=8, ha='center', va='bottom', color=INK)
            bp = ax.boxplot([panel_values[:, mi] for mi in range(16)], positions=np.arange(16),
                widths=.56, whis=(0, 100), showfliers=False,
                patch_artist=True, manage_ticks=False, boxprops=dict(ec='#31627F', lw=.75),
                medianprops=dict(color=INK, lw=.95), whiskerprops=dict(color='#31627F', lw=.7),
                capprops=dict(color='#31627F', lw=.7))
            for mi, (model, box) in enumerate(zip(panel_models, bp['boxes'])):
                color = METHOD_COLORS[model]
                edge = tuple(.75 * np.array(to_rgb(color)))
                box.set_facecolor(color); box.set_edgecolor(edge)
                bp['medians'][mi].set_color(contrasting_ink(to_rgb(color)))
                for line in bp['whiskers'][2 * mi:2 * mi + 2] + bp['caps'][2 * mi:2 * mi + 2]:
                    line.set_color(edge)
                assert np.isclose(bp['medians'][mi].get_ydata()[0], np.median(panel_values[:, mi]))
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
                side.set_color(INK)
                side.set_linewidth(.8)
                side.set_zorder(3)
            assert np.all(np.diff(panel_values.mean(axis=0)) <= 0)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.0'])
            ax.grid(axis='y', color='#E5EAEE', lw=.45, zorder=0); ax.set_axisbelow(True)
            axes.append(ax)
    return fig, axes, models, means, ordinal, dataset_means, panel_orders


def main():
    sys.path.insert(0, str(R / 'analysis/vendor/scientific_visualization/scripts'))
    from figure_export import export_figure
    O.mkdir(parents=True, exist_ok=True)
    rows, raw, ranked, _, protected = base.evidence()
    for folder in ['figures/rq1_review/fig02_clustering', 'figures/rq1_ranktable_review/fig02_clustering',
                   'figures/rq1_merged_review/fig02_clustering']:
        for path in (R / folder).glob('*'):
            if path.is_file(): protected[path.relative_to(R).as_posix()] = base.sha(path)
    for version in ['ranktable', 'merged']:
        for extension in ['pdf', 'pptx']:
            path = R / f'manuscript/CellBench_Fig2_RQ1_{version}_review.{extension}'
            protected[path.relative_to(R).as_posix()] = base.sha(path)
    fig, axes, models, means, ordinal, dataset_means, panel_orders = make_figure(raw, ranked)
    texts = base.audit_text(fig)
    alignment = require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=list('abcde'),
        row_groups=[['b', 'c'], ['d', 'e']], column_groups=[['b', 'd'], ['c', 'e']],
        json_out=str(O / 'alignment.json'), overlay_svg=str(O / 'alignment.svg'),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    provenance = dict(source='source_seed_values.csv', script='analysis/draw_rq1_vertical_review.py',
        panel_a='ordinal summary bars plus mean-rank dataset tiles in one shared row coordinate system',
        panels_b_e='12 dataset means per box; each dataset value averages 3 seeds',
        method_order_b_e='metric-specific descending arithmetic mean across all 12 datasets',
        method_colors=METHOD_COLORS, baseline_emphasis=False,
        individual_markers=False, whiskers='observed minimum and maximum', significance_tests=False)
    export_figure(fig, O / SLUG, formats=['pdf', 'svg', 'png'], dpi=600, font_mode='truetype',
                  bbox_inches=None, transparent=False, overwrite=True, write_manifest=True, provenance=provenance)
    export_figure(fig, O / (SLUG + '_outlined'), formats=['svg'], font_mode='paths', bbox_inches=None, overwrite=True)
    fig.savefig(O / 'preview.png', dpi=300, facecolor='white')
    for filename, dpi in [(SLUG + '.png', 600), ('preview.png', 300)]:
        with Image.open(O / filename) as image:
            assert image.getchannel('A').getextrema() == (255, 255)
            rgb = image.convert('RGB')
        rgb.save(O / filename, dpi=(dpi, dpi))
    with Image.open(O / 'preview.png') as image:
        image.crop((0, round((H - 122) * 300 / 25.4), image.width, image.height)).save(
            O / 'metric_panels_preview.png', dpi=(300, 300))
    manifest_path = O / (SLUG + '.export.json')
    meta = json.loads(manifest_path.read_text())
    for item in meta['outputs']: item['size_bytes'] = Path(item['path']).stat().st_size
    manifest_path.write_text(json.dumps(meta, indent=2))
    base.writecsv(O / 'source_seed_values.csv', rows)
    previous_data = R / 'figures/rq1_ranktable_review/fig02_clustering'
    for name in ['source_dataset_means.csv', 'source_context_seed_ranks.csv', 'source_ranking_table.csv']:
        (O / name).write_bytes((previous_data / name).read_bytes())
    base.writecsv(O / 'source_dataset_rank_tiles.csv',
        [dict(model=model, **{dataset: float(dataset_means[i, j]) for j, dataset in enumerate(t.DS)})
         for i, model in enumerate(models)])
    base.writecsv(O / 'source_method_colors.csv',
                  [dict(model=model, color=METHOD_COLORS[model]) for model in t.MODELS])
    # Recompute metric means directly from the original unpermuted array.
    source_order = []
    original_values = raw.mean(axis=1)
    plot_seq = [0, 1, 3, 2]
    for plot_i, ki in enumerate(plot_seq):
        metric, _, _ = t.METRICS['clustering'][ki]
        panel = 'bcde'[plot_i]
        for position, model in enumerate(panel_orders[panel]):
            mi = t.MODELS.index(model)
            source_order.append(dict(panel=panel, metric=metric, position=position + 1, model=model,
                display_label=t.display(model),
                mean_score=float(original_values[:, mi, ki].mean()),
                median_score=float(np.median(original_values[:, mi, ki])), color=METHOD_COLORS[model]))
        these = [row for row in source_order if row['panel'] == panel]
        assert len(these) == 16 and np.all(np.diff([row['mean_score'] for row in these]) <= 0)
        assert set(panel_orders[panel]) == set(t.MODELS)
        assert [label.get_text() for label in axes[plot_i + 1].get_xticklabels()] == [t.display(m) for m in panel_orders[panel]]
    base.writecsv(O / 'source_metric_order.csv', source_order)
    # Independently verify every displayed summary and tile against the prior source values.
    prior_table = {row['model']: row for row in t.readcsv(previous_data / 'source_ranking_table.csv')}
    for i, model in enumerate(models):
        for j, name in enumerate(['overall'] + t.PANELS):
            assert np.isclose(ordinal[i, j], float(prior_table[model][name + '_display_rank']))
            assert np.isclose(means[t.MODELS.index(model), j], float(prior_table[model][name + '_mean_rank']))
    assert models == json.loads((previous_data / 'verification.json').read_text())['method_order']
    (O / 'caption.txt').write_text(CAPTION + '\n')
    (O / 'alt_text.txt').write_text('Panel a orders sixteen methods by overall composite mean rank and combines overall '
        'and context rank bars with twelve labeled dataset mean-rank tile columns. UCE ranks first '
        'in both human contexts and PCA ranks first in mouse organs. Panels b-e retain the ARI, '
        'NMI, accuracy and ASW distributions across twelve dataset means, each independently ordered '
        'by descending mean score. Compact method labels on the x-axes map to the full names in panel a. '
        'Method colors are fixed across the four panels; no method names '
        'are bold and no fitted baselines have special box styles.\n')
    (O / 'text_audit.json').write_text(json.dumps(texts, indent=2))
    rec = dict(slug=SLUG, folder=O.relative_to(R).as_posix(), caption=CAPTION,
               width_inches=W / 25.4, height_inches=H / 25.4)
    deck = Presentation(); deck.slide_width = Inches(W / 25.4); deck.slide_height = Inches(H / 25.4)
    add_slide(deck, rec)
    ppt = R / 'manuscript/CellBench_Fig2_RQ1_vertical_review.pptx'; deck.save(ppt)
    pdf = R / 'manuscript/CellBench_Fig2_RQ1_vertical_review.pdf'
    pdf.write_bytes((O / (SLUG + '.pdf')).read_bytes()); plt.close(fig)
    assert all(base.sha(R / path) == value for path, value in protected.items())
    with pymupdf.open(pdf) as document:
        assert len(document) == 1 and not document[0].get_images()
        assert all('Type3' not in str(font) for font in document[0].get_fonts())
        document[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(O / 'pdf_preview.png')
    for suffix in ['.svg', '_outlined.svg']:
        assert '<image' not in (O / (SLUG + suffix)).read_text()
    with zipfile.ZipFile(ppt) as archive:
        assert archive.testzip() is None and len(archive.namelist()) == len(set(archive.namelist()))
    report = dict(status='PASS', source_runs=576, raw_values=2304, table_rank_cells=64,
        dataset_rank_cells=192, raw_metric_boxes=64, raw_metric_n=12, method_order_a=models,
        panel_orders=panel_orders, method_colors=METHOD_COLORS, baseline_emphasis=False,
        width_mm=W, height_mm=H, minimum_font_pt=8, vector_pdf_svg=True, png_dpi=600,
        original_outputs_and_data_unchanged=protected, alignment=alignment['verdict'],
        nonrotated_text_overlaps=0, visual_review='Completed same-assistant inspection of the ranking table, four vertical metric panels, 300 dpi preview and final PDF render.')
    (O / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'original_outputs_and_data_unchanged'}, indent=2))


if __name__ == '__main__':
    main()
