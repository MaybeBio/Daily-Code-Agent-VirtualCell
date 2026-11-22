"""Draw the combined RQ6--RQ7 attribute figure from audited checkpoint attributes and panel ranks."""
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
import task_layout as t
from audit_panel_alignment import require_matplotlib_panel_alignment

R = fig2.R
TASKS = ['clustering', 'correction', 'annotation', 'reconstruction']
TASK_LABELS = ['Clustering', 'Batch', 'Annotation', 'Reconstruction']
INK = fig2.INK
HUMAN = '#457B9D'
MULTI = '#C45C86'


def load():
    rows = t.readcsv(R / 'data/layout_panel_ranks.csv')
    rank = {(r['task'], r['panel'], r['model']): float(r['mean_rank']) for r in rows}
    assoc = t.readcsv(R / 'data/layout_attribute_associations.csv')
    overall = {task: np.array([np.mean([rank[task, p, m] for p in t.PANELS]) for m in t.MODELS[:13]])
               for task in TASKS}
    # Panel ranks are already means; task-wide mean is the mean of the three panel means.
    # The manuscript coefficients use the mean across all dataset-seed-metric blocks, which
    # equals the unweighted mean of the three equal-sized panels.
    human = np.array([rank['annotation', 'organ', m] for m in t.MODELS[:13]])
    mouse = np.array([rank['annotation', 'species', m] for m in t.MODELS[:13]])
    expected = {
        ('parameters', 'clustering'): (-0.24175824175824176, -0.49650349650349657, -0.034965034965034975),
        ('parameters', 'correction'): (0.06043956043956044, -0.04895104895104895, 0.25874125874125875),
        ('parameters', 'annotation'): (-0.12087912087912088, -0.3426573426573427, 0.1188811188811189),
        ('parameters', 'reconstruction'): (-0.14835164835164835, -0.37762237762237766, 0.08391608391608392),
        ('corpus', 'clustering'): (0.32121212121212117, 0.21666666666666667, 0.4666666666666666),
        ('corpus', 'correction'): (-0.19999999999999998, -0.39999999999999997, 0.09999999999999999),
        ('corpus', 'annotation'): (-0.07878787878787878, -0.2833333333333333, 0.15),
        ('corpus', 'reconstruction'): (-0.07878787878787878, -0.2833333333333333, 0.15),
    }
    got = {(r['factor'], r['task']): (float(r['spearman_rho']), float(r['leave_one_out_min']), float(r['leave_one_out_max']))
           for r in assoc if r['panel'] == 'all'}
    for key, value in expected.items():
        assert np.allclose(got[key], value)
    assert np.isclose(human[t.MODELS.index('Geneformer')], 9.895833333333334)
    assert np.isclose(human[t.MODELS.index('LangCell')], 7.125)
    assert np.isclose(mouse[t.MODELS.index('scFoundation')], 1.9791666666666667)
    assert np.isclose(mouse[t.MODELS.index('TranscriptFormer')], 4.78125)
    assert np.isclose(mouse[t.MODELS.index('Nicheformer')], 4.677083333333333)
    assert np.isclose(mouse[t.MODELS.index('GeneCompass')], 14.229166666666666)
    return overall, human, mouse, got


def style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Arial'],
                         'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'text.color': INK, 'axes.edgecolor': INK, 'axes.labelcolor': INK,
                         'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': .8})


def export(fig, folder, slug, axes, caption):
    sys.path.insert(0, str(R / 'analysis/vendor/scientific_visualization/scripts'))
    from figure_export import export_figure
    folder.mkdir(parents=True, exist_ok=True)
    export_figure(fig, folder / slug, formats=['pdf', 'svg', 'png'], dpi=600, font_mode='truetype',
                  bbox_inches=None, transparent=False, overwrite=True, write_manifest=True,
                  provenance={'script': Path(__file__).name})
    fig.savefig(folder / 'preview.png', dpi=300, facecolor='white')
    for filename, dpi in [(slug + '.png', 600), ('preview.png', 300)]:
        with Image.open(folder / filename) as image:
            image.convert('RGB').save(folder / filename, dpi=(dpi, dpi))
    if len(axes) > 1:
        require_matplotlib_panel_alignment(
            fig, axes=axes, panel_ids=list('ab'),
            exemptions=[{
                'panels': ['a', 'b'],
                'checks': ['panel-width', 'column'],
                'reason': 'Each table is sized to its own column content and centered independently.',
            }],
            json_out=str(folder / 'alignment.json'), overlay_svg=str(folder / 'alignment.svg'),
            tolerance_pt=1.5, strict=True)
    pdf = folder / (slug + '.pdf')
    with pymupdf.open(pdf) as document:
        assert len(document) == 1 and not document[0].get_images()
    (folder / 'caption.txt').write_text(caption + '\n')
    with plt.rc_context({'svg.fonttype': 'path'}):
        fig.savefig(folder / (slug + '_outlined.svg'))
    from pptx import Presentation
    from pptx.util import Inches
    from perturbation_rq import add_slide
    width_in, height_in = fig.get_size_inches()
    deck = Presentation()
    deck.slide_width = Inches(width_in)
    deck.slide_height = Inches(height_in)
    add_slide(deck, dict(slug=slug, folder=folder.relative_to(R).as_posix(), caption=caption,
                         width_inches=width_in, height_inches=height_in))
    deck.save(folder / (slug + '.pptx'))
    plt.close(fig)
    print('OK', pdf)


def draw_protocol(overall):
    models = t.MODELS[:13]
    order = np.argsort([overall['annotation'][i] for i in range(13)], kind='stable')
    W, H = 190, 92
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    left, bottom, width, height = 8, 8, 176, 72
    ax = fig.add_axes([left / W, bottom / H, width / W, height / H])
    ax.set(xlim=(0, width), ylim=(12.5, -.5))
    ax.axis('off')
    cmap = LinearSegmentedColormap.from_list('rank', ['#F3F6F8', '#457B9D'])
    fig2.panel_letter(fig, 0, 86.2, 'a')
    headers = [(0, 42, 'Method', '#73777C'), (44, 16, 'Input', '#73777C')]
    for j, label in enumerate(TASK_LABELS):
        headers.append((62 + j * 28, 26, label, ['#457B9D', '#C45C86', '#2A9D8F', '#C48A2A'][j]))
    for x, w, label, color in headers:
        fig.add_artist(Rectangle(((left + x) / W, 81.2 / H), w / W, 4.2 / H,
                                transform=fig.transFigure, fc=color, lw=0))
        fig.text((left + x + w / 2) / W, 83.3 / H, label, ha='center', va='center', fontsize=8, color='white')
    for row, mi in enumerate(order):
        ax.add_patch(Rectangle((0, row - .5), width, 1, fc='#E8ECEF' if row % 2 == 0 else '#F8F9FA', lw=0))
        ax.text(40, row, t.display(models[mi]), ha='right', va='center', fontsize=8, color=INK)
        ax.text(52, row, t.ENCODING[mi], ha='center', va='center', fontsize=8, color=INK)
        for j, task in enumerate(TASKS):
            value = overall[task][mi]
            tile_x = 63 + j * 28
            rgba = cmap((16 - value) / 15)
            ax.add_patch(Rectangle((tile_x, row - .38), 24, .76, fc=rgba, lw=0))
            ax.text(tile_x + 12, row, f'{value:.2f}', ha='center', va='center', fontsize=8,
                    color=fig2.contrasting_ink(rgba))
    caption = ('Input protocol and task rank. Thirteen pretrained checkpoints are ordered by annotation '
               'mean rank. Input codes are A, gene-aligned; V, gene identity and values; R, ranked genes; '
               'C, context; S, sampled genes; P, protein-derived features; F, fixed gene panel; B, expression '
               'bins. Codes overlap and do not define exclusive tokenizer classes. Cells give the mean rank '
               'across three equal-sized biological contexts; darker cells indicate better ranks. Baselines '
               'are excluded. No causal tokenizer effect is estimated.')
    return fig, [ax], caption, R / 'figures/rq678_review/fig07_protocol', 'fig07_protocol'


def draw_size(assoc):
    W, H = 190, 78
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    axes = []
    for column, (factor, title, n) in enumerate([
            ('parameters', 'Parameters', 'n = 13'),
            ('corpus', 'Corpus size', 'n = 10')]):
        x = 28 if column == 0 else 112
        ax = fig.add_axes([x / W, 12 / H, 68 / W, 52 / H])
        fig.text((x - 6) / W, 68.5 / H, 'ab'[column], fontsize=8, weight='bold',
                 ha='left', va='bottom', color=INK)
        fig.text((x + 34) / W, 68.5 / H, title, fontsize=8, ha='center', va='bottom', color=INK)
        fig.text((x + 68) / W, 66.2 / H, n, fontsize=8, ha='right', va='bottom', color=INK)
        for i, task in enumerate(TASKS):
            rho, lo, hi = assoc[factor, task]
            ax.plot([lo, hi], [i, i], color='#457B9D', lw=1.2, solid_capstyle='butt')
            ax.plot(rho, i, 'o', ms=5.5, color='#457B9D', mec=INK, mew=.4)
        ax.axvline(0, color='#8AA0AE', lw=.7)
        ax.set_yticks(range(4), TASK_LABELS)
        ax.set_xlim(-.6, .6)
        ax.set_xticks([-.4, -.2, 0, .2, .4])
        ax.set_xlabel('Spearman rho')
        ax.set_ylim(-.6, 3.6)
        ax.tick_params(length=3, width=.8, direction='out')
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_color(INK)
            side.set_linewidth(.8)
        axes.append(ax)
    caption = ('Checkpoint size and task rank. a, Parameter count, thirteen checkpoints. b, Reported corpus '
               'size, ten retained estimates. Dots are full-set Spearman coefficients; lines span '
               'leave-one-model-out coefficients, a sensitivity range rather than a confidence interval. '
               'Negative coefficients associate larger size with a better rank. Each checkpoint is counted '
               'once per task. Parameter scopes and corpus units differ. No scaling law or causal effect is estimated.')
    return fig, axes, caption, R / 'figures/rq678_review/fig08_size', 'fig08_size'


def draw_species(human, mouse):
    """Table of human-organ versus mouse-organ mean ranks for each representation task."""
    models = t.MODELS[:13]
    rows = t.readcsv(R / 'data/layout_panel_ranks.csv')
    rank = {(r['task'], r['panel'], r['model']): float(r['mean_rank']) for r in rows}
    order = np.argsort(mouse, kind='stable')
    W, H = 190, 92
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    cmap = LinearSegmentedColormap.from_list('rank', ['#F3F6F8', '#457B9D'])
    left, bottom, width, height = 6, 10, 180, 66
    ax = fig.add_axes([left / W, bottom / H, width / W, height / H])
    ax.set(xlim=(0, width), ylim=(12.5, -.5))
    ax.axis('off')
    task_colors = ['#457B9D', '#C45C86', '#2A9D8F', '#C48A2A']
    group = 36.5
    fig.add_artist(Rectangle((left / W, 81.4 / H), 32 / W, 4.2 / H, transform=fig.transFigure, fc='#73777C', lw=0))
    fig.text((left + 16) / W, 83.5 / H, 'Method', ha='center', va='center', fontsize=8, color='white')
    for j, (label, color) in enumerate(zip(TASK_LABELS, task_colors)):
        x = 34 + j * group
        fig.add_artist(Rectangle(((left + x) / W, 81.4 / H), (group - .8) / W, 4.2 / H,
                                transform=fig.transFigure, fc=color, lw=0))
        fig.text((left + x + (group - .8) / 2) / W, 83.5 / H, label, ha='center', va='center', fontsize=8, color='white')
        for k, sub in enumerate(('Human', 'Mouse')):
            sx = x + k * 18
            fig.add_artist(Rectangle(((left + sx) / W, 76.8 / H), 17.2 / W, 3.6 / H,
                                    transform=fig.transFigure, fc=tuple(.72 + .28 * np.array(fig2.to_rgb(color))), lw=0))
            fig.text((left + sx + 8.6) / W, 78.6 / H, sub, ha='center', va='center', fontsize=8, color=INK)
    for row, mi in enumerate(order):
        model = models[mi]
        ax.add_patch(Rectangle((0, row - .5), width, 1, fc='#E8ECEF' if row % 2 == 0 else '#F8F9FA', lw=0))
        name_color = MULTI if mi in t.MULTI else HUMAN
        ax.text(31, row, t.display(model), ha='right', va='center', fontsize=8, color=name_color)
        for j, task in enumerate(TASKS):
            for k, panel in enumerate(('organ', 'species')):
                value = rank[task, panel, model]
                tile_x = 35 + j * group + k * 18
                rgba = cmap((16 - value) / 15)
                ax.add_patch(Rectangle((tile_x, row - .36), 16.4, .72, fc=rgba, lw=0))
                ax.text(tile_x + 8.2, row, f'{value:.2f}', ha='center', va='center', fontsize=8,
                        color=fig2.contrasting_ink(rgba))
    fig.text(6 / W, 5.2 / H, 'Blue names: predominantly human pretraining.   Vermillion names: human/mouse or multispecies.',
             fontsize=8, color=INK)
    assert np.isclose(rank['annotation', 'species', 'scFoundation'], 1.9791666666666667)
    assert np.isclose(rank['annotation', 'organ', 'Geneformer'], 9.895833333333334)
    caption = ('Human-organ and mouse-organ ranks on each representation task. Models are ordered by '
               'mouse-organ annotation rank. Each task has a Human column and a Mouse column; cells are mean '
               'ranks and darker cells are better. Blue names denote predominantly human pretraining; vermillion '
               'names denote reported human/mouse or multispecies pretraining. This compares performance on '
               'human-organ and mouse-organ studies, not direct human-to-mouse transfer. Baselines are excluded. '
               'No causal species effect is estimated.')
    return fig, [ax], caption, R / 'figures/rq678_review/fig09_species', 'fig09_species'


def draw_combined(overall, human, mouse, assoc):
    models = t.MODELS[:13]
    protocol_order = np.argsort([overall['annotation'][i] for i in range(13)], kind='stable')
    species_order = np.argsort(mouse, kind='stable')
    W, H = 190, 198
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    cmap = LinearSegmentedColormap.from_list('rank', ['#F3F6F8', '#457B9D'])

    # a. Input protocol and task ranks.
    left, bottom, width, height = 8, 136, 176, 50
    table = fig.add_axes([left / W, bottom / H, width / W, height / H])
    table.set(xlim=(0, width), ylim=(12.5, -.5))
    table.axis('off')
    fig.text(2 / W, 192.4 / H, 'a', fontsize=8, weight='bold', ha='left', va='bottom', color=INK)
    headers = [(0, 42, 'Method', '#73777C'), (44, 16, 'Input', '#73777C')]
    for j, label in enumerate(TASK_LABELS):
        headers.append((62 + j * 28, 26, label, ['#457B9D', '#C45C86', '#2A9D8F', '#C48A2A'][j]))
    for x, w, label, color in headers:
        fig.add_artist(Rectangle(((left + x) / W, 187.2 / H), w / W, 4.0 / H,
                                transform=fig.transFigure, fc=color, lw=0))
        fig.text((left + x + w / 2) / W, 189.2 / H, label, ha='center', va='center', fontsize=8, color='white')
    for row, mi in enumerate(protocol_order):
        table.add_patch(Rectangle((0, row - .5), width, 1, fc='#E8ECEF' if row % 2 == 0 else '#F8F9FA', lw=0))
        table.text(40, row, t.display(models[mi]), ha='right', va='center', fontsize=8, color=INK)
        table.text(52, row, t.ENCODING[mi], ha='center', va='center', fontsize=8, color=INK)
        for j, task in enumerate(TASKS):
            value = overall[task][mi]
            tile_x = 63 + j * 28
            rgba = cmap((16 - value) / 15)
            table.add_patch(Rectangle((tile_x, row - .36), 24, .72, fc=rgba, lw=0))
            table.text(tile_x + 12, row, f'{value:.2f}', ha='center', va='center', fontsize=8,
                       color=fig2.contrasting_ink(rgba))

    # b, c. Size associations.
    forests = []
    for column, (factor, title, n) in enumerate([
            ('parameters', 'Parameters', 'n = 13'),
            ('corpus', 'Corpus size', 'n = 10')]):
        x = 28 if column == 0 else 112
        ax = fig.add_axes([x / W, 82 / H, 68 / W, 40 / H])
        fig.text((x - 8) / W, 124.6 / H, 'bc'[column], fontsize=8, weight='bold',
                 ha='left', va='bottom', color=INK)
        fig.text((x + 34) / W, 124.6 / H, title, fontsize=8, ha='center', va='bottom', color=INK)
        fig.text((x + 68) / W, 124.6 / H, n, fontsize=8, ha='right', va='bottom', color=INK)
        for i, task in enumerate(TASKS):
            rho, lo, hi = assoc[factor, task]
            ax.plot([lo, hi], [i, i], color='#457B9D', lw=1.2, solid_capstyle='butt')
            ax.plot(rho, i, 'o', ms=5.2, color='#457B9D', mec=INK, mew=.4)
        ax.axvline(0, color='#8AA0AE', lw=.7)
        ax.set_yticks(range(4), TASK_LABELS)
        ax.set_xlim(-.6, .6)
        ax.set_xticks([-.4, -.2, 0, .2, .4])
        ax.set_xlabel('Spearman rho')
        ax.set_ylim(-.65, 3.55)
        ax.tick_params(length=3, width=.8, direction='out')
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_color(INK)
            side.set_linewidth(.8)
        forests.append(ax)

    # d. Species comparison.
    species = fig.add_axes([42 / W, 22 / H, 136 / W, 48 / H])
    fig.text(2 / W, 72.2 / H, 'd', fontsize=8, weight='bold', ha='left', va='bottom', color=INK)
    for row, mi in enumerate(species_order):
        color = MULTI if mi in t.MULTI else HUMAN
        species.plot([human[mi], mouse[mi]], [row, row], color=color, lw=1.1,
                     ls='--' if mi in t.MULTI else '-')
        species.plot(human[mi], row, marker='o', mfc='white', mec=color, ms=5, mew=1.1)
        species.plot(mouse[mi], row, marker='o', mfc=color, mec=color, ms=5)
    species.set_yticks(range(13), [t.display(models[i]) for i in species_order])
    species.set_xlim(.5, 16.5)
    species.set_xticks([1, 4, 8, 12, 16])
    species.set_xlabel('Annotation mean rank (lower is better)', labelpad=1)
    species.set_ylim(-.7, 12.7)
    species.tick_params(length=3, width=.8, direction='out')
    for side in species.spines.values():
        side.set_visible(True)
        side.set_color(INK)
        side.set_linewidth(.8)
    fig.text(42 / W, 8.4 / H, 'Open circles: human organs.   Filled circles: mouse organs.', fontsize=8, color=INK)
    fig.text(42 / W, 3.6 / H, 'Blue: human pretraining.   Vermillion: human/mouse or multispecies.', fontsize=8, color=INK)
    caption = ('Checkpoint attributes and task rank. a, Thirteen pretrained checkpoints ordered by annotation '
               'mean rank. Input codes overlap and are not exclusive tokenizer classes. Cells give the mean rank '
               'across three biological contexts; darker cells indicate better ranks. b, Parameter count, thirteen '
               'checkpoints. c, Reported corpus size, ten retained estimates. Dots are Spearman coefficients; lines '
               'span leave-one-model-out coefficients, a sensitivity range rather than a confidence interval. '
               'Negative coefficients associate larger size with a better rank. d, Human-organ and mouse-organ '
               'annotation ranks. Open circles are human organs and filled circles are mouse organs. Blue solid '
               'lines denote predominantly human pretraining; vermillion dashed lines denote reported human/mouse '
               'or multispecies pretraining. Endpoints are model summaries, not seeds. Baselines are excluded. '
               'No causal effect or scaling law is estimated.')
    return fig, [table, *forests, species], caption


def draw_protocol_size(overall, assoc):
    """One table: input code, parameter count, corpus size and the four task ranks."""
    models = t.MODELS[:13]
    params = {r['model']: int(r['total_parameters']) for r in t.readcsv(R / 'data/checkpoint_parameter_counts.csv')}
    order = sorted(range(13), key=lambda mi: (t.SIZES.get(models[mi]) is None, -(t.SIZES.get(models[mi]) or 0)))
    W, H = 190, 86
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    cmap = LinearSegmentedColormap.from_list('rank', ['#F3F6F8', '#457B9D'])
    left, bottom, width, height = 6, 8, 180, 66
    ax = fig.add_axes([left / W, bottom / H, width / W, height / H])
    ax.set(xlim=(0, width), ylim=(12.5, -.5))
    ax.axis('off')
    pert = {r['model']: float(r['mean_metric_rank'])
            for r in t.readcsv(R / 'data/all_panel_ranks.csv') if r['task'] == 'perturbation'}
    assert np.isclose(pert['scLong'], 5.729166666666667)
    assert np.isclose(pert['scFoundation'], 5.770833333333333)
    headers = [
        (0, 32, 'Method', '#73777C'),
        (33, 20, 'Corpus (M)', '#5C6B73'),
        (54, 18, 'Params (M)', '#5C6B73'),
        (73, 12, 'Input', '#73777C'),
    ]
    task_labels = TASK_LABELS + ['Perturb.']
    task_colors = ['#457B9D', '#C45C86', '#2A9D8F', '#C48A2A', '#6E5A7D']
    for j, label in enumerate(task_labels):
        headers.append((86 + j * 18.8, 18.0, label, task_colors[j]))
    for x, w, label, color in headers:
        fig.add_artist(Rectangle(((left + x) / W, 75.2 / H), w / W, 4.2 / H,
                                transform=fig.transFigure, fc=color, lw=0))
        fig.text((left + x + w / 2) / W, 77.3 / H, label, ha='center', va='center', fontsize=8, color='white')
    for row, mi in enumerate(order):
        model = models[mi]
        ax.add_patch(Rectangle((0, row - .5), width, 1, fc='#E8ECEF' if row % 2 == 0 else '#F8F9FA', lw=0))
        ax.text(30, row, t.display(model), ha='right', va='center', fontsize=8, color=INK)
        corpus = t.SIZES.get(model)
        ax.text(51, row, f'{corpus:.0f}' if corpus else '—', ha='right', va='center', fontsize=8, color=INK)
        ax.text(70, row, f'{params[model] / 1e6:.0f}', ha='right', va='center', fontsize=8, color=INK)
        ax.text(79, row, t.ENCODING[mi], ha='center', va='center', fontsize=8, color=INK)
        values = [overall[task][mi] for task in TASKS] + [pert[model]]
        for j, value in enumerate(values):
            tile_x = 87 + j * 18.8
            rgba = cmap((16 - value) / 15)
            ax.add_patch(Rectangle((tile_x, row - .36), 16.6, .72, fc=rgba, lw=0))
            ax.text(tile_x + 8.3, row, f'{value:.2f}', ha='center', va='center', fontsize=8,
                    color=fig2.contrasting_ink(rgba))
    assert np.isclose(params['scLong'] / 1e6, 1006.1, atol=0.1)
    assert np.isclose(params['scFoundation'] / 1e6, 119.3, atol=0.1)
    caption = ('Input protocol, checkpoint size and task rank in one comparison. Thirteen pretrained '
               'checkpoints are ordered by reported corpus size, largest first; models without a retained '
               'estimate are last. Input codes are A, gene-aligned; V, gene '
               'identity and values; R, ranked genes; C, context; S, sampled genes; P, protein-derived features; '
               'F, fixed gene panel; B, expression bins. Parameters are audited instantiated-model counts in '
               'millions. Corpus is the reported pretraining size in millions of cells; a dash means the estimate '
               'was not retained. The first four rank columns are means across three biological contexts; '
               'Perturb. is the mean rank on four held-out perturbation datasets. Darker cells indicate better '
               'ranks. Models that share an input code, or that have more parameters or a larger corpus, do not '
               'share a rank pattern across tasks. Baselines are excluded. No causal effect or scaling law is estimated.')
    return fig, [ax], caption


def text_mm(fig, text, size=8):
    artist = fig.text(0, 0, text, fontsize=size, color=INK)
    fig.canvas.draw()
    width = artist.get_window_extent(fig.canvas.get_renderer()).width * 25.4 / fig.dpi
    artist.remove()
    return width


def content_width(fig, labels, pad=2.4):
    return max(text_mm(fig, label) for label in labels) + pad


def table_context(models):
    params = {r['model']: int(r['total_parameters']) for r in t.readcsv(R / 'data/checkpoint_parameter_counts.csv')}
    order = sorted(range(13), key=lambda mi: (t.SIZES.get(models[mi]) is None, -(t.SIZES.get(models[mi]) or 0)))
    return params, order


def span(widths, gap):
    return sum(widths) + gap * (len(widths) - 1)


def mm_table(width, height, table_w, table_h, bottom):
    """One axes whose data unit is 1 mm, so a header and its cells share x and width."""
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))
    left = (width - table_w) / 2
    ax = fig.add_axes([left / width, bottom / height, table_w / width, table_h / height])
    ax.set_xlim(0, table_w)
    ax.set_ylim(table_h, 0)
    ax.axis('off')
    ax.set_clip_on(False)
    return fig, ax


def same_edges(fig, header, cell):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    h = header.get_window_extent(renderer)
    c = cell.get_window_extent(renderer)
    scale = 72 / fig.dpi
    assert abs((h.x0 - c.x0) * scale) < 0.4 and abs((h.x1 - c.x1) * scale) < 0.4, (
        f'header {h.x0:.1f}-{h.x1:.1f} cell {c.x0:.1f}-{c.x1:.1f}')


LOOKUP = {'UCE': '87.7', 'scPRINT2': '84.2'}
WIDTH = {
    'Tabula': '192',
    'scLong': '200',
    'LangCell': '256',
    'Geneformer': '512',
    'Nicheformer': '512',
    'UCE': '1280',
    'scFoundation': '3072',
}
TEXT = {'LangCell': 'Y'}


def draw_attributes(overall):
    """Fig. 8: corpus, parameters, lookup share, embedding width, tokenizer, text alignment, and task ranks."""
    models = t.MODELS[:13]
    params, _ = table_context(models)
    task_labels = TASK_LABELS
    task_colors = ['#457B9D', '#C45C86', '#2A9D8F', '#C48A2A']
    values = {}
    for mi, model in enumerate(models):
        task_values = [overall[task][mi] for task in TASKS]
        values[model] = task_values + [float(np.mean(task_values))]
    order = sorted(range(13), key=lambda mi: values[models[mi]][-1])
    probe = plt.figure(figsize=(1, 1))
    probe.canvas.draw()
    corpus_labels = [f'{t.SIZES[m]:.0f}' if m in t.SIZES else '—' for m in models]
    param_labels = [f'{params[m] / 1e6:.0f}' for m in models]
    lookup_labels = [LOOKUP.get(m, '—') for m in models]
    width_labels = [WIDTH.get(m, '—') for m in models]
    text_labels = [TEXT.get(m, '—') for m in models]
    columns = [
        ('Method', content_width(probe, ['Method'] + [t.display(m) for m in models], 3.2), '#73777C', 'white'),
        ('Corpus (M)', content_width(probe, ['Corpus (M)'] + corpus_labels), '#5C6B73', 'white'),
        ('Params (M)', content_width(probe, ['Params (M)'] + param_labels), '#5C6B73', 'white'),
        ('Lookup', content_width(probe, ['Lookup'] + lookup_labels), '#5C6B73', 'white'),
        ('Width', content_width(probe, ['Width'] + width_labels), '#5C6B73', 'white'),
        ('Input', content_width(probe, ['Input'] + list(t.ENCODING)), '#73777C', 'white'),
        ('Text', content_width(probe, ['Text'] + text_labels), '#73777C', 'white'),
        ('Overall', content_width(probe, ['Overall', '16.00']), '#3E515C', 'white'),
    ]
    for label, color in zip(task_labels, task_colors):
        columns.append((label, content_width(probe, [label, '16.00']), color, 'white'))
    plt.close(probe)
    gap, header_h, row_h = 1.0, 4.6, 4.5
    table_w = span([width for _, width, _, _ in columns], gap)
    table_h = header_h + 1.2 + 13 * row_h
    W, H = 190, table_h + 8
    fig, ax = mm_table(W, H, table_w, table_h, 4)
    cmap = LinearSegmentedColormap.from_list('rank', ['#F3F6F8', '#457B9D'])
    xs, header_patches = [], []
    x = 0
    for label, width, color, ink in columns:
        patch = Rectangle((x, 0), width, header_h, fc=color, lw=0, clip_on=False)
        ax.add_patch(patch)
        header_patches.append(patch)
        ax.text(x + width / 2, header_h / 2, label, ha='center', va='center', fontsize=8, color=ink, clip_on=False)
        xs.append(x)
        x += width + gap
    rank_index = 7
    last_cell = None
    for row, mi in enumerate(order):
        model = models[mi]
        y = header_h + 1.2 + row * row_h
        ax.add_patch(Rectangle((0, y), table_w, row_h, fc='#E8ECEF' if row % 2 == 0 else '#F8F9FA', lw=0, zorder=0))
        cy = y + row_h / 2
        corpus = t.SIZES.get(model)
        attrs = [
            (0, t.display(model), 'right'),
            (1, f'{corpus:.0f}' if corpus else '—', 'right'),
            (2, f'{params[model] / 1e6:.0f}', 'right'),
            (3, LOOKUP.get(model, '—'), 'right'),
            (4, WIDTH.get(model, '—'), 'right'),
            (5, t.ENCODING[mi], 'center'),
            (6, TEXT.get(model, '—'), 'center'),
        ]
        for idx, lab, ha in attrs:
            if ha == 'right':
                ax.text(xs[idx] + columns[idx][1] - 1.2, cy, lab, ha='right', va='center', fontsize=8, color=INK)
            else:
                ax.text(xs[idx] + columns[idx][1] / 2, cy, lab, ha='center', va='center', fontsize=8, color=INK)
        shown = [values[model][-1]] + values[model][:-1]
        for j, value in enumerate(shown):
            width = columns[rank_index + j][1]
            rgba = cmap((16 - value) / 15)
            cell = Rectangle((xs[rank_index + j], y + .65), width, row_h - 1.3, fc=rgba, lw=0, clip_on=False)
            ax.add_patch(cell)
            ax.text(xs[rank_index + j] + width / 2, cy, f'{value:.2f}', ha='center', va='center', fontsize=8,
                    color=fig2.contrasting_ink(rgba))
            if row == 0 and j == len(shown) - 1:
                last_cell = cell
    same_edges(fig, header_patches[-1], last_cell)
    assert np.isclose(values['scLong'][-1], np.mean([overall[task][models.index('scLong')] for task in TASKS]))
    assert np.isclose(params['scLong'] / 1e6, 1006.1, atol=0.1)
    assert LOOKUP['UCE'] == '87.7' and LOOKUP['scPRINT2'] == '84.2'
    assert WIDTH['Tabula'] == '192' and WIDTH['scFoundation'] == '3072'
    assert TEXT['LangCell'] == 'Y'
    assert values[models[order[0]]][-1] <= values[models[order[-1]]][-1]
    caption = ('Checkpoint attributes and task ranks. Thirteen pretrained checkpoints are ordered by Overall '
               'rank, lowest first. Overall is the unweighted mean of the four representation-task ranks in the same row. '
               'Corpus and parameters are in millions. Lookup is the '
               'percent of audited parameters inside a frozen protein or gene lookup table, 87.7 in UCE and 84.2 in '
               'scPRINT2; GeneCompass keeps 155.4M regulatory-prior elements in buffers outside Params. Width is '
               'extracted cell-embedding dimensionality. Input codes are '
               'A, gene-aligned; V, gene identity and values; R, ranked genes; C, context; S, sampled genes; '
               'P, protein-derived features; F, fixed gene panel; B, expression bins. Text marks biomedical text '
               'alignment, present only in LangCell. Clustering, Batch, Annotation, and Reconstruction are means '
               'across three biological contexts. Each column header has the same width as the cells beneath it. '
               'Darker cells indicate better ranks. Baselines are excluded. No causal effect or scaling law is estimated.')
    return fig, [ax], caption


def draw_species():
    """Fig. 6: human-organ versus mouse-organ ranks, ordered by Overall mouse rank."""
    models = t.MODELS[:13]
    rank_rows = t.readcsv(R / 'data/layout_panel_ranks.csv')
    panel_rank = {(r['task'], r['panel'], r['model']): float(r['mean_rank']) for r in rank_rows}
    overall_mouse = [
        float(np.mean([panel_rank[task, 'species', models[mi]] for task in TASKS]))
        for mi in range(13)
    ]
    order = sorted(range(13), key=lambda mi: overall_mouse[mi])
    task_colors = ['#3E515C', '#457B9D', '#C45C86', '#2A9D8F', '#C48A2A']
    group_labels = ['Overall'] + TASK_LABELS
    probe = plt.figure(figsize=(1, 1))
    probe.canvas.draw()
    method_w = content_width(probe, ['Method'] + [t.display(m) for m in models], 3.2)
    gap = 1.2
    pair = content_width(probe, ['Mouse', 'Human', '16.00'])
    groups = []
    for label, color in zip(group_labels, task_colors):
        group_w = max(text_mm(probe, label) + 2.4, 2 * pair + gap)
        sub = (group_w - gap) / 2
        groups.append((label, color, group_w, sub))
    plt.close(probe)
    table_w = span([method_w] + [group[2] for group in groups], gap)
    header_h, sub_h, row_h = 4.4, 3.8, 4.5
    table_h = header_h + .4 + sub_h + 1.2 + 13 * row_h
    W, H = 190, table_h + 12
    fig, ax = mm_table(W, H, table_w, table_h, 8)
    cmap = LinearSegmentedColormap.from_list('rank', ['#F3F6F8', '#457B9D'])
    ax.add_patch(Rectangle((0, 0), method_w, header_h + .4 + sub_h, fc='#73777C', lw=0, clip_on=False))
    ax.text(method_w / 2, (header_h + .4 + sub_h) / 2, 'Method', ha='center', va='center', fontsize=8, color='white')
    x = method_w + gap
    header_patches, subs = [], []
    for label, color, group_w, sub in groups:
        patch = Rectangle((x, 0), group_w, header_h, fc=color, lw=0, clip_on=False)
        ax.add_patch(patch)
        header_patches.append(patch)
        ax.text(x + group_w / 2, header_h / 2, label, ha='center', va='center', fontsize=8, color='white')
        shade = tuple(.70 + .30 * np.array(fig2.to_rgb(color)))
        pair_patches = []
        for k, name in enumerate(('Human', 'Mouse')):
            sx = x + k * (sub + gap)
            subpatch = Rectangle((sx, header_h + .4), sub, sub_h, fc=shade, lw=0, clip_on=False)
            ax.add_patch(subpatch)
            pair_patches.append(subpatch)
            ax.text(sx + sub / 2, header_h + .4 + sub_h / 2, name, ha='center', va='center', fontsize=8, color=INK)
        subs.append(pair_patches)
        x += group_w + gap
    data_top = header_h + .4 + sub_h + 1.2
    checks = []
    for row, mi in enumerate(order):
        model = models[mi]
        y = data_top + row * row_h
        ax.add_patch(Rectangle((0, y), table_w, row_h, fc='#E8ECEF' if row % 2 == 0 else '#F8F9FA', lw=0, zorder=0))
        cy = y + row_h / 2
        ax.text(method_w - 1.2, cy, t.display(model), ha='right', va='center', fontsize=8, color=INK)
        x = method_w + gap
        for j, label in enumerate(group_labels):
            _, _, _, sub = groups[j]
            for k, panel in enumerate(('organ', 'species')):
                if label == 'Overall':
                    value = float(np.mean([panel_rank[task, panel, model] for task in TASKS]))
                else:
                    value = panel_rank[TASKS[j - 1], panel, model]
                rgba = cmap((16 - value) / 15)
                sx = x + k * (sub + gap)
                cell = Rectangle((sx, y + .65), sub, row_h - 1.3, fc=rgba, lw=0, clip_on=False)
                ax.add_patch(cell)
                ax.text(sx + sub / 2, cy, f'{value:.2f}', ha='center', va='center', fontsize=8,
                        color=fig2.contrasting_ink(rgba))
                if row == 0:
                    checks.append((subs[j][k], cell))
            x += 2 * sub + 2 * gap
    fig.canvas.draw()
    for header, cell in checks:
        same_edges(fig, header, cell)
    for patch, (_, _, group_w, _) in zip(header_patches, groups):
        # Task header must cover exactly its Human and Mouse columns, including the gap between them.
        assert abs(patch.get_width() - group_w) < 1e-6
    assert np.isclose(panel_rank['annotation', 'organ', 'Geneformer'], 9.895833333333334)
    assert overall_mouse[order[0]] <= overall_mouse[order[-1]]
    caption = ('Human-organ and mouse-organ mean ranks. Thirteen pretrained checkpoints are ordered by Overall '
               'mouse-organ rank, lowest first. Overall is the unweighted mean of the four task ranks in the same row. '
               'Each task header spans its Human and Mouse columns. '
               'Those columns are means across four datasets and three seeds. Darker cells indicate better ranks. '
               'Perturbation has no organ split and is omitted. Baselines are excluded. '
               'No causal species effect is estimated.')
    return fig, [ax], caption


def main():
    overall, human, mouse, assoc = load()
    style()
    fig, axes, caption = draw_attributes(overall)
    export(fig, R / 'figures/rq678_review/fig07_attributes', 'fig07_attributes', axes, caption)
    fig, axes, caption = draw_species()
    export(fig, R / 'figures/rq678_review/fig08_species', 'fig08_species', axes, caption)


if __name__ == '__main__':
    main()
