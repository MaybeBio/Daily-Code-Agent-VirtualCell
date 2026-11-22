"""Promote existing perturbation evidence: figures -> PPT -> manuscript.

No model is rerun. The original exports and all existing figure files are read-only.
Run --phase figures, inspect outputs, then --phase manuscript and compile.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image
import pymupdf

import task_layout as t

R = Path(__file__).resolve().parents[1]
B = R / 'docs/perturbation_rq_baseline'
O = R / 'figures/perturbation_rq'
DS = ['norman', 'adamson', 'replogleK562', 'replogleRPE1']
NAMES = ['Norman', 'Adamson', 'Replogle\nK562', 'Replogle\nRPE1']
SHORT_NAMES = ['Norm.', 'Adam.', 'K562', 'RPE1']
METRICS = ['Pearson_delta', 'Spearman_delta', 'MAE_delta', 'DES']
LABELS = ['Pearson Δ', 'Spearman Δ', 'MAE Δ', 'DES']
MODELS = t.MODELS
INK = '#202B33'
BLUE = '#0072B2'
RANK = LinearSegmentedColormap.from_list('pert_rank', ['#255579', '#82ABC7', '#F4F7FA'])
SKILL = LinearSegmentedColormap.from_list('pert_skill', ['#A84821', '#FCFCFA', '#236A94'])
sys.path.insert(0, str(R / 'analysis/vendor/scientific_visualization/scripts'))
from style_presets import style_context
from figure_export import export_figure

MAIN_CAPTION = ('Perturbation prediction on four held-out-condition datasets. '
    'a, Each method has three composite-rank values, one per seed, averaging within-dataset--seed--metric '
    'ranks over four datasets and four metrics. Boxes span interpolated quartiles, with median lines '
    'and observed minimum--maximum whiskers; seed markers are hidden. Open boxes identify fitted baselines. '
    'b, Dataset-specific composite ranks, averaged over three seeds and four metrics. Lower ranks are better '
    'in a,b. c, Three-seed mean MAE skill, calculated within each seed as 1 minus MAE delta divided by '
    'the exported control-only baseline MAE. Zero denotes no improvement; negative values indicate greater '
    'error than predicting the same-cell-type mean control expression. The zero-centered color scale is '
    'symmetric. Cells in b,c show numerical means, not seeds. Norman evaluates held-out combinations after '
    'single-gene training; the other datasets hold out condition identities. Conditions and expression '
    'scales differ across studies. Norm./Adam. denote Norman/Adamson; K562/RPE1 denote the Replogle datasets. '
    'Cell values are rounded. Seeds are repeated runs/splits, not biological replicates or confidence '
    'intervals. Supplementary Figs. S22--S25 retain all four primary raw metrics for every dataset and seed.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_binary(path):
    """Read tracked baseline decks without accumulating slide-part names."""
    baseline = json.loads((B / 'evidence.json').read_text())
    relative = path.relative_to(R).as_posix()
    git_path = path.relative_to(Path('/workspace')).as_posix()
    blob = subprocess.run(['git', 'show', baseline['git_commit'] + ':' + git_path],
                          cwd=R, capture_output=True, check=True).stdout
    assert hashlib.sha256(blob).hexdigest() == baseline['changed_file_baseline_sha256'][relative]
    return blob


def writecsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evidence():
    source = R / 'data/exports/perturbation/perturbation_scores.csv'
    rows = t.readcsv(source)
    lookup = {(row['dataset'], int(row['seed']), row['model']): row for row in rows}
    assert len(rows) == len(lookup) == 192
    assert {r['metric_version'] for r in rows} == {'2'}
    assert {r['protocol'] for r in rows} == {'1eeb9cc69bf06d76'}
    raw = np.array([[[[float(lookup[d, s, m][key]) for key in METRICS]
                       for m in MODELS] for s in t.SEEDS] for d in DS])
    assert np.isfinite(raw).all()
    ranked = np.empty_like(raw)
    for d, s, k in np.ndindex(4, 3, 4):
        ranked[d, s, :, k] = t.ranks(raw[d, s, :, k], high=k != 2)
    original = {r['model']: float(r['mean_metric_rank']) for r in t.readcsv(R / 'data/all_panel_ranks.csv')
                if r['task'] == 'perturbation'}
    assert all(np.isclose(ranked.mean(axis=(0, 1, 3))[i], original[m]) for i, m in enumerate(MODELS))
    control = np.array([[[float(lookup[d, s, m]['control_baseline_MAE']) for m in MODELS]
                         for s in t.SEEDS] for d in DS])
    assert (control > 0).all()
    assert np.allclose(control, control[:, :, :1])
    for row in rows:
        assert np.isclose(float(row['MSE_skill_vs_control']),
                          1 - float(row['MSE_delta']) / float(row['control_baseline_MSE']))
    skill = 1 - raw[:, :, :, 2] / control
    expanded = [dict(**row, mean_metric_rank=float(ranked[DS.index(row['dataset']),
                        t.SEEDS.index(int(row['seed'])), MODELS.index(row['model'])].mean()),
                        MAE_skill_vs_control=float(skill[DS.index(row['dataset']),
                        t.SEEDS.index(int(row['seed'])), MODELS.index(row['model'])])) for row in rows]
    return rows, expanded, raw, ranked, skill, sha(source)


def save(fig, slug, caption, rows):
    folder = O / slug
    folder.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        bb = text.get_window_extent(renderer)
        assert bb.x0 >= -.5 and bb.y0 >= -.5 and bb.x1 <= fig.bbox.width + .5 and bb.y1 <= fig.bbox.height + .5, text.get_text()
        assert text.get_fontsize() >= 8, text.get_text()
    export_figure(fig, folder / slug, formats=['pdf', 'svg', 'png'], dpi=600,
                  bbox_inches=None, transparent=False, font_mode='truetype', overwrite=True,
                  write_manifest=True, provenance=dict(source='data/exports/perturbation/perturbation_scores.csv',
                    script='analysis/perturbation_rq.py', individual_seed_markers=False,
                    replication='3 repeated runs/splits', uncertainty='quartiles and observed range, not CI'))
    export_figure(fig, folder / (slug + '_outlined'), formats=['svg'], font_mode='paths',
                  bbox_inches=None, overwrite=True)
    fig.savefig(folder / 'preview.png', dpi=300, facecolor='white')
    for path in [folder / (slug + '.png'), folder / 'preview.png']:
        with Image.open(path) as im:
            if 'A' in im.getbands():
                assert im.getchannel('A').getextrema() == (255, 255)
            im.convert('RGB').save(path, dpi=(600, 600) if path.stem == slug else (300, 300))
    manifest_path = folder / (slug + '.export.json')
    manifest = json.loads(manifest_path.read_text())
    for output in manifest['outputs']:
        output['size_bytes'] = Path(output['path']).stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2))
    writecsv(folder / 'source_data.csv', rows)
    (folder / 'caption.txt').write_text(caption)
    result = dict(slug=slug, folder=folder.relative_to(R).as_posix(), caption=caption,
                  width_inches=fig.get_figwidth(), height_inches=fig.get_figheight())
    plt.close(fig)
    return result


def row_axis(ax, labels=False):
    ax.set_ylim(16.6, .4)
    ax.set_yticks(range(1, 17), [m.replace('PCA_model', 'PCA') for m in MODELS] if labels else [])
    ax.tick_params(axis='y', length=0, pad=6)
    ax.axhline(13.5, color='#BAC3CB', lw=.7)
    ax.spines[['top', 'right', 'left']].set_visible(False)


def heat(ax, z, names, cmap, norm=None, vmin=None, vmax=None):
    im = ax.pcolormesh(np.arange(5) - .5, np.arange(17) + .5, z,
                       shading='flat', rasterized=False, cmap=cmap, norm=norm,
                       vmin=vmin, vmax=vmax)
    ax.set_xlim(-.5, 3.5)
    ax.set_xticks(range(4), names)
    ax.tick_params(axis='x', length=0, pad=5)
    row_axis(ax)
    for i, j in np.ndindex(16, 4):
        value = z[i, j]
        rgba = im.cmap(im.norm(value))
        # Numerical labels provide a redundant, non-color encoding.
        lum = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
        label = f'{value:.1f}' if vmin is not None else (f'{value:+.2f}' if abs(value) >= .005 else '0.00')
        ax.text(j, i + 1, label,
                ha='center', va='center', color='white' if lum < .55 else INK, fontsize=8.5)
    return im


def figures():
    rows, expanded, raw, ranked, skill, source_hash = evidence()
    O.mkdir(exist_ok=True)
    with style_context('default', palette_name='okabe_ito_on_white'):
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Arial'],
            'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
            'pdf.fonttype': 42, 'svg.fonttype': 'none', 'text.color': INK})
        fig = plt.figure(figsize=(190 / 25.4, 140 / 25.4))
        axes = [fig.add_axes(rect) for rect in [[.19, .19, .20, .69], [.47, .19, .205, .69], [.755, .19, .205, .69]]]
        for ax, letter, title in zip(axes, 'abc', ['Overall ranks', 'Dataset ranks', 'MAE vs. control']):
            box = ax.get_position()
            fig.text(box.x0, .94, letter, fontsize=11, weight='bold')
            fig.text(box.x0 + .027, .94, title, fontsize=9)
        for mi in range(16):
            boxes = axes[0].boxplot([ranked[:, :, mi, :].mean(axis=(0, 2))], positions=[mi + 1],
                orientation='horizontal', widths=.55, whis=(0, 100), showfliers=False,
                patch_artist=True, manage_ticks=False, boxprops=dict(facecolor='#B5D2E3' if mi < 13 else 'white',
                edgecolor=BLUE, linewidth=.9), medianprops=dict(color=INK, linewidth=1),
                whiskerprops=dict(color=BLUE, linewidth=.8), capprops=dict(color=BLUE, linewidth=.8))
        row_axis(axes[0], True)
        axes[0].set(xlim=(.5, 16.5), xticks=[1, 4, 8, 12, 16], xlabel='Mean rank (lower is better)')
        im = heat(axes[1], ranked.mean(axis=(1, 3)).T, SHORT_NAMES, RANK, vmin=1, vmax=16)
        cb = fig.colorbar(im, cax=fig.add_axes([.47, .085, .205, .017]), orientation='horizontal', ticks=[1, 8, 16])
        cb.solids.set_rasterized(False)
        cb.ax.tick_params(labelsize=8.5)
        bound = np.ceil(abs(skill.mean(axis=1)).max() * 10) / 10
        im = heat(axes[2], skill.mean(axis=1).T, SHORT_NAMES, SKILL, norm=TwoSlopeNorm(0, -bound, bound))
        cb = fig.colorbar(im, cax=fig.add_axes([.755, .085, .205, .017]), orientation='horizontal', ticks=[-bound, 0, bound])
        cb.solids.set_rasterized(False)
        cb.ax.tick_params(labelsize=8.5)
        fig.text(.47, .025, 'Lower ranks are better', fontsize=8.5)
        fig.text(.755, .025, 'Positive = less error', fontsize=8.5)
        records = [save(fig, 'fig06_perturbation', MAIN_CAPTION, expanded)]
        for ki, (key, label) in enumerate(zip(METRICS, LABELS)):
            fig = plt.figure(figsize=(190 / 25.4, 128 / 25.4))
            fig.text(.04, .95, f'Perturbation prediction · {label}', fontsize=11, weight='bold')
            for di in range(4):
                ax = fig.add_axes([.18 + di * .205, .15, .175, .68])
                ax.boxplot([raw[di, :, mi, ki] for mi in range(16)], positions=range(1, 17),
                    orientation='horizontal', widths=.55, whis=(0, 100), showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor='#B5D2E3', edgecolor=BLUE, linewidth=.8),
                    medianprops=dict(color=INK, linewidth=.9), whiskerprops=dict(color=BLUE, linewidth=.7),
                    capprops=dict(color=BLUE, linewidth=.7))
                row_axis(ax, di == 0)
                if ki != 2:
                    values = raw[:, :, :, ki]
                    pad = (values.max() - values.min()) * .07
                    ax.set_xlim(min(0, values.min() - pad), values.max() + pad)
                ax.set_xlabel(label)
                ax.set_title(NAMES[di].replace('\n', ' '), fontsize=9, pad=12)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                lo, hi = ax.get_xlim()
                ax.set_xticks([value for value in ax.get_xticks() if lo <= value <= hi])
            fig.text(.18, .035, 'Three seeds per box; seed markers hidden. ' +
                     ('Dataset-specific MAE scales.' if ki == 2 else 'Common metric scale across datasets.'), fontsize=8.5)
            caption = ('Perturbation prediction assessed by ' + label.replace('Δ', '$\\Delta$') +
                '. Columns identify the four datasets; rows retain all sixteen methods. Each box uses seeds 41, 42 and 43, '
                'with interpolated quartiles, median line and observed minimum--maximum whiskers. Individual seed markers '
                'are hidden. These are repeated runs/splits, not biological replicates or confidence intervals. ' +
                ('MAE is lower-is-better and axes use dataset-specific expression scales.' if ki == 2 else
                 'Higher is better; axes share a common scale across datasets.') +
                ' DES averages only groups with a defined reference significant-gene set; valid-group counts accompany the source data.')
            records.append(save(fig, f's{22 + ki}_perturbation_{key}', caption, expanded))
    (O / 'manifest.json').write_text(json.dumps(records, indent=2))
    audit = dict(primary_evaluations=192, primary_values=raw.size, finite_primary_values=True,
        panel_ranks_checked=16, source_sha256=source_hash,
        metric_version=2, protocol='1eeb9cc69bf06d76',
        panel_mean_ranks=dict(zip(MODELS, ranked.mean(axis=(0, 1, 3)).tolist())),
        dataset_mean_ranks={d: dict(zip(MODELS, ranked[i].mean(axis=(0, 2)).tolist())) for i, d in enumerate(DS)},
        positive_MAE_skill_runs={d: int((skill[i] > 0).sum()) for i, d in enumerate(DS)},
        pearson_three_seed_mean_ranges={d: [float(raw[i, :, :, 0].mean(0).min()), float(raw[i, :, :, 0].mean(0).max())] for i, d in enumerate(DS)},
        before_manuscript_update=True)
    (O / 'evidence_audit.json').write_text(json.dumps(audit, indent=2))
    workbook(expanded, ranked)
    decks(records)
    print(json.dumps(audit, indent=2))


def workbook(rows, ranked):
    from openpyxl import Workbook
    book = Workbook()
    sheet = book.active
    sheet.title = 'Readme'
    for line in ['Perturbation task: 4 datasets, 16 methods, 3 seeds.',
                 'Primary metrics: Pearson delta, Spearman delta, MAE delta, DES.',
                 'Ranks are averaged equally over metrics, datasets and seeds; ties use average ranks.',
                 'MAE skill = 1 - MAE_delta/control_baseline_MAE; positive is better than no change.',
                 'Seeds are repeated runs/splits, not biological replicates. No CI or significance tests.',
                 'DES_valid_groups and correlation valid_n fields are retained.']:
        sheet.append([line])
    sheet = book.create_sheet('All seed values')
    sheet.append(list(rows[0]))
    for row in rows:
        sheet.append([row[k] if k in ['dataset', 'model', 'phase', 'updated_at', 'run_id', 'protocol'] else
                      float(row[k]) if row[k] else None for k in rows[0]])
    sheet = book.create_sheet('Dataset composite ranks')
    sheet.append(['model'] + DS + ['Overall'])
    for i, model in enumerate(MODELS):
        sheet.append([model] + ranked[:, :, i, :].mean(axis=(1, 2)).tolist() + [float(ranked[:, :, i, :].mean())])
    for sheet in book:
        sheet.freeze_panes = 'B2'
        sheet.auto_filter.ref = sheet.dimensions
    book.save(R / 'appendix/perturbation_seed_results.xlsx')


def add_slide(deck, rec):
    from pptx.util import Inches
    from pptx.opc.package import Part
    from pptx.opc.packuri import PackURI
    from pptx.opc.constants import RELATIONSHIP_TYPE as RT
    from pptx.oxml.xmlchemy import OxmlElement
    from pptx.oxml.ns import qn
    from lxml import etree
    folder = R / rec['folder']
    slug = rec['slug']
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    scale = min(deck.slide_width / Inches(rec['width_inches']), deck.slide_height / Inches(rec['height_inches'])) * .96
    w, h = Inches(rec['width_inches']) * scale, Inches(rec['height_inches']) * scale
    pic = slide.shapes.add_picture(str(folder / (slug + '.png')), int((deck.slide_width - w) / 2),
                                  int((deck.slide_height - h) / 2), width=int(w), height=int(h))
    part = Part(PackURI(f'/ppt/media/perturbation{len(deck.slides):02d}.svg'), 'image/svg+xml',
                deck.part.package, (folder / (slug + '_outlined.svg')).read_bytes())
    rid = slide.part.relate_to(part, RT.IMAGE)
    exts = OxmlElement('a:extLst')
    ext = OxmlElement('a:ext')
    ext.set('uri', '{96DAC541-7B7A-43D3-8B79-37D633B846F1}')
    ns = 'http://schemas.microsoft.com/office/drawing/2016/SVG/main'
    element = etree.Element('{' + ns + '}svgBlip', nsmap={'asvg': ns})
    element.set(qn('r:embed'), rid)
    ext.append(element); exts.append(ext); pic._pic.blipFill.blip.append(exts)
    slide.notes_slide.notes_text_frame.text = rec['caption']


def decks(records):
    from pptx import Presentation
    from pptx.util import Inches
    prs = Presentation(); prs.slide_width = Inches(8.2); prs.slide_height = Inches(8.2)
    for rec in records:
        add_slide(prs, rec)
    prs.save(R / 'manuscript/CellBench_perturbation_review.pptx')
    with pymupdf.open() as out:
        for rec in records:
            with pymupdf.open(R / rec['folder'] / (rec['slug'] + '.pdf')) as doc:
                out.insert_pdf(doc)
        out.save(R / 'manuscript/CellBench_perturbation_review.pdf')
    for stem, old_count in [('CellBench_main_figures_kdense', 6), ('CellBench_main_and_supplement_figures', 27)]:
        path = R / 'manuscript' / (stem + '.pptx')
        deck = Presentation(io.BytesIO(baseline_binary(path)))
        target_count = old_count + (1 if old_count == 6 else 5)
        if len(deck.slides) == target_count:
            for index in sorted([5] + (list(range(old_count + 1, target_count)) if old_count == 27 else []), reverse=True):
                sid = deck.slides._sldIdLst[index]
                deck.part.drop_rel(sid.rId)
                deck.slides._sldIdLst.remove(sid)
        if len(deck.slides) == old_count:
            old_xml = [s._element.xml for s in deck.slides]
            add_slide(deck, records[0])
            sid = deck.slides._sldIdLst[-1]
            deck.slides._sldIdLst.remove(sid); deck.slides._sldIdLst.insert(5, sid)
            if old_count == 27:
                for rec in records[1:]:
                    add_slide(deck, rec)
            assert [s._element.xml for i, s in enumerate(deck.slides) if i != 5 and i < old_count + 1] == old_xml
            deck.slides[0].notes_slide.notes_text_frame.text = deck.slides[0].notes_slide.notes_text_frame.text.replace(
                'Quantitative results in this manuscript cover tasks 1--4 on twelve datasets; the perturbation workflow is shown separately and does not enter these rankings.',
                'Tasks 1--4 cover twelve datasets; perturbation prediction is evaluated separately on four additional datasets.')
            for slide in deck.slides:
                notes = slide.notes_slide.notes_text_frame
                notes.text = notes.text.replace('Figure 6A', 'main-text Figure 7a')
            deck.save(path)
        else:
            assert len(deck.slides) == target_count
        path = path.with_suffix('.pdf')
        with pymupdf.open(stream=baseline_binary(path), filetype='pdf') as old:
            if len(old) in [old_count, target_count]:
                out = pymupdf.open(); out.insert_pdf(old, to_page=4)
                with pymupdf.open(R / records[0]['folder'] / (records[0]['slug'] + '.pdf')) as new:
                    out.insert_pdf(new)
                out.insert_pdf(old, from_page=5 if len(old) == old_count else 6,
                               to_page=old_count - 1 if len(old) == old_count else old_count)
                if old_count == 27:
                    for rec in records[1:]:
                        with pymupdf.open(R / rec['folder'] / (rec['slug'] + '.pdf')) as new:
                            out.insert_pdf(new)
                temp = path.with_name(path.stem + '_perturbation.tmp.pdf')
                out.save(temp); out.close()
            else:
                assert len(old) == target_count
                temp = None
        if temp:
            temp.replace(path)
    index = json.loads((B / 'figures/task_layout/manifest.json').read_text())
    index[0]['caption'] = index[0]['caption'].replace(
        'Quantitative results in this manuscript cover tasks 1--4 on twelve datasets; the perturbation workflow is shown separately and does not enter these rankings.',
        'Tasks 1--4 cover twelve datasets; perturbation prediction is evaluated separately on four additional datasets.')
    for rec in index:
        rec['caption'] = rec['caption'].replace('Figure 6A', 'main-text Figure 7a')
    index.insert(5, records[0]); index.extend(records[1:])
    (R / 'figures/task_layout/manifest.json').write_text(json.dumps(index, indent=2))
    (O / 'ppt_before_manuscript.json').write_text(json.dumps(dict(
        manuscript_before_sha256=sha(B / 'main.tex'), review_pptx_sha256=sha(R / 'manuscript/CellBench_perturbation_review.pptx'),
        figure_count=5, stage='Figures and PPT completed before text update'), indent=2))


def manuscript():
    stage = json.loads((O / 'ppt_before_manuscript.json').read_text())
    assert stage['review_pptx_sha256'] == sha(R / 'manuscript/CellBench_perturbation_review.pptx')
    text = (B / 'main.tex').read_text()
    for n in [7, 6, 5]:
        text = text.replace('RQ' + str(n), 'RQ' + str(n + 1))
    text = text.replace('seven research questions', 'eight research questions')
    marker = r'\item[\textbf{RQ6.}]'
    text = text.replace(marker, r'\item[\textbf{RQ5.}] Are held-out perturbation responses predicted consistently across datasets and evaluation criteria?' + '\n' + marker)
    text = text.replace('RQ6--RQ8 relate the four-task results',
        'RQ5 compares held-out perturbation predictions on four additional datasets, including their error relative to a control-only predictor. RQ6--RQ8 relate the four-task results')
    old = ('Figure~\\ref{fig:framework} also shows a separate perturbation-prediction workflow using Norman, Adamson and Replogle K562/RPE1 datasets. The quantitative results and composite rankings reported below cover the twelve datasets and four tasks in the first three panels; the perturbation setting is not pooled into those results.')
    new = ('A fourth panel comprises Norman, Adamson and Replogle K562/RPE1 for perturbation prediction (RQ5). Its ranks are computed separately over those four datasets and are not pooled with the twelve-dataset summaries for tasks 1--4 or the checkpoint-attribute analyses.')
    assert old in text
    text = text.replace(old, new)
    text = text.replace('Quantitative results in this manuscript cover tasks 1--4 on twelve datasets; the perturbation workflow is shown separately and does not enter these rankings.',
        'Tasks 1--4 cover twelve datasets; perturbation prediction is evaluated separately on four additional datasets.')
    marker = r'\subsection{Metrics, ranks and figure summaries}'
    text = text.replace(marker, r'''For RQ5, the perturbation decoder predicts expression from a matched control-cell embedding, condition identity and control expression. The training loop does not supply the perturbed-cell embedding to the decoder. Norman holds out combinations after single-gene training; Adamson and Replogle hold out condition identities. The decoder is selected by validation expression MSE, checked every twenty epochs. Exported cell/condition counts and the four primary metrics are retained for every seed in the perturbation workbook. Counts agree across methods; exact test-cell identities were not independently compared.

''' + marker)
    text = text.replace('These definitions explain task 5 in the overview and do not add perturbation scores to the four-task result summaries.',
        'These quantities define the four primary perturbation metrics. As a post hoc diagnostic of the exports, MAE skill is $1-\\mathrm{MAE}_{\\Delta}/\\mathrm{MAE}_{\\mathrm{control}}$, using the exported same-cell-type mean control-expression predictor. Negative skill denotes higher error than this no-change predictor. Ratios are computed within dataset and seed before averaging; raw MAEs are not pooled across datasets.')
    text = text.replace('All attribute analyses (RQ6--RQ8)', 'All attribute analyses (RQ6--RQ8)')
    text = text.replace('with tasks 1--4 providing the quantitative scope here;',
        'with tasks 1--4 evaluated on twelve datasets and task 5 on four additional datasets;')
    text = text.replace('Supplementary Figs.~S1--S16 provide the corresponding full boxplots, with no selection of methods or favorable datasets.',
        'Supplementary Figs.~S1--S16 provide the corresponding full boxplots, with no selection of methods or favorable datasets. The separate perturbation analysis adds 192 method--dataset--seed evaluations and 768 primary-metric values, all finite; its sixteen composite ranks reproduce the original exports.')
    marker = r'\subsection{RQ6: Shared input protocols'
    new_section = r'''\subsection{RQ5: Perturbation ranks conceal dataset and control-baseline limitations}
scLong and scFoundation had similar four-metric perturbation mean ranks, 5.729 and 5.771, followed by PCA at 6.792 (Fig.~\ref{fig:perturbation}a--b). scLong ranked ahead of scFoundation on Norman (2.750 versus 4.083) and RPE1 (5.083 versus 6.333), whereas scFoundation ranked ahead on Adamson (4.167 versus 5.833) and K562 (8.500 versus 9.250). These are pairwise order reversals, not evidence that either model led every dataset. Three-seed mean Pearson $\Delta$ values across the sixteen methods ranged from 0.652 to 0.761 on Norman but from $-0.013$ to 0.041 on Replogle K562. Supplementary Figs.~S22--S25 retain the four individual metrics and seed variation.

Relative ranks did not establish an advantage over predicting no expression change. Every method and seed had higher MAE than the control-only predictor on Adamson, Replogle K562 and RPE1 (48 evaluations per dataset; Fig.~\ref{fig:perturbation}c). Only three of the 48 Norman evaluations improved on that baseline. The no-change reference is an error diagnostic, not a seventeenth method in the four-metric ranking: a constant zero change has no defined across-gene correlation. These results answer RQ5 conditionally: method ordering and apparent response accuracy depend on the dataset and criterion, and a leading relative rank can coexist with worse absolute error than a simple control predictor.

\begin{figure}[pos=p]
\centering
\includegraphics[width=\textwidth]{figures/perturbation_rq/fig06_perturbation/fig06_perturbation.pdf}
\caption{''' + MAIN_CAPTION + r'''}\label{fig:perturbation}
\end{figure}

'''
    assert marker in text
    text = text.replace(marker, new_section + marker)
    text = text.replace('Metric disagreement specifies what a model-selection claim can mean.',
        'Perturbation prediction (RQ5) extends the context dependence to held-out conditions. Its control-only comparison shows that a favorable relative rank need not indicate a reduction in absolute prediction error. This distinction limits claims of useful response prediction under the evaluated condition-aware decoder protocol.\n\nMetric disagreement specifies what a model-selection claim can mean.')
    text = text.replace('Evaluation across four tasks and twelve human and mouse datasets revealed both stable annotation summaries and context-dependent clustering, batch-assessment and reconstruction preferences.',
        'Evaluation across four representation tasks on twelve human and mouse datasets and perturbation prediction on four additional datasets revealed both stable annotation summaries and context-dependent method preferences. Perturbation ranks also require comparison with a control-only error baseline.')
    text = text.replace('The independent supplementary document contains Figs.~S1--S21 and Tables~S1--S23; the accompanying workbook retains individual seed values for all four tasks.',
        'The independent supplementary document contains Figs.~S1--S25 and Tables~S1--S23. The original workbook retains individual seed values for tasks 1--4; a separate perturbation workbook retains all 192 run-level exports, valid-group counts, split counts and derived control-baseline comparisons.')
    text = text.replace('on 12 datasets spanning human organs, human diseases and mouse organs.',
        'on 12 datasets spanning human organs, human diseases and mouse organs, with perturbation prediction evaluated separately on four additional datasets.')
    text = text.replace('Reconstruction leaders differed among panels:', 'Reconstruction leaders differed among panels:')
    text = text.replace('Excluding either of two datasets',
        'Perturbation mean ranks placed scLong and scFoundation close together, but all methods had greater MAE than a control-only predictor on Adamson and both Replogle datasets. Excluding either of two datasets')
    text = text.replace('Four tasks are evaluated on twelve human and mouse datasets with three seeds.',
        'Four representation tasks and perturbation prediction span sixteen datasets.')
    text = text.replace('Seed boxplots and overlap sensitivity qualify aggregate model rankings.',
        'Perturbation ranks can conceal greater error than a control-only predictor.')
    # Make room for the eighth question while preserving a single unbroken list.
    text = text.replace('Are clustering advantages consistent across human organs, human diseases and mouse organs?',
        'Are clustering advantages consistent across biological contexts?')
    text = text.replace('Does a panel-level annotation lead persist across datasets, metrics and exclusion of studies with documented pretraining-source overlap?',
        'Does a panel-level annotation lead survive dataset, metric and pretraining-overlap checks?')
    text = text.replace('Single-cell foundation models encode gene-expression profiles for tasks such as cell-type annotation, clustering and expression reconstruction.',
        'Single-cell foundation models encode gene expression for annotation, clustering, reconstruction and perturbation prediction.')
    (R / 'main.tex').write_text(text)
    supp = (B / 'supplementary.tex').read_text()
    supp = supp.replace('Four-task supplementary results', 'Five-task supplementary results')
    supp = supp.replace("Perturbation prediction is outside this document's scope.",
        'Perturbation prediction adds four datasets with held-out conditions; Figs. S22--S25 show its four primary metrics, and the separate perturbation workbook retains every seed and split count.')
    supp = supp.replace('Figure 6A', 'main-text Figure 7a')
    records = json.loads((O / 'manifest.json').read_text())
    extra = '\n'.join(r'''\begin{figure}[pos=p]
\centering
\includegraphics[width=\textwidth]{''' + rec['folder'] + '/' + rec['slug'] + r'''.pdf}
\caption{''' + rec['caption'] + r'''}\label{fig:''' + rec['slug'] + r'''}
\end{figure}
\clearpage
''' for rec in records[1:])
    supp = supp.replace('\\FloatBarrier\n\\section*{Supplementary tables}', extra + '\n\\FloatBarrier\n\\section*{Supplementary tables}')
    (R / 'supplementary.tex').write_text(supp)
    readme = (B / 'README.md').read_text().replace('# CellBench 四任务论文', '# CellBench 五任务论文')
    readme = readme.replace('图1另行展示扰动预测流程，其分数不计入正文的四任务定量比较。',
        '另在Norman、Adamson、Replogle K562/RPE1四数据集独立评估扰动预测；该任务不与前三panel的排名混合。')
    readme = readme.replace('七个研究问题', '八个研究问题').replace('重建指标一致性、tokenization', '重建指标一致性、扰动预测与control基线、tokenization')
    readme = readme.replace('6张主图、2张表', '7张主图、2张表').replace('21张附图、23张表', '25张附图、23张表')
    readme = readme.replace('27页', '32页').replace('前6页', '前7页').replace('当前6张正文图', '当前7张正文图')
    readme = readme.replace('当前文字重构后运行 `python analysis/verify_question_narrative.py`，验证图表、数据、所有结果数值和引用不变，并检查RQ对应关系。复现本轮时，在上述步骤4后运行 `analysis/revise_question_narrative.py`，再编译正文并运行本轮核验。',
        '七问题版本使用 `analysis/verify_question_narrative.py`。当前在七问题版本后运行 `analysis/perturbation_rq.py --phase figures`，审图/PPT后运行 `--phase manuscript`，编译正文和附录，再运行 `analysis/verify_perturbation_rq.py`。旧生成器会恢复其对应的历史版本。')
    readme = readme.replace('## 当前文件', '''## 扰动研究问题补充

- RQ5为扰动预测；原属性问题顺延为RQ6–RQ8。新增正文Fig.6；原属性图成为Fig.7。
- `analysis/perturbation_rq.py --phase figures`先生成矢量图/PPT和seed工作簿；审图后`--phase manuscript`更新正文，再编译主文和附录。
- `manuscript/CellBench_perturbation_review.pptx` / `.pdf`为新增5图独立审阅版；原两份当前图件合集亦已更新。
- `appendix/perturbation_seed_results.xlsx`保留全部192条run结果和控制基线；`figures/perturbation_rq/`保留矢量图、600 dpi PNG、源数据和审计。
- 当前核验用`analysis/verify_perturbation_rq.py`；旧七RQ核验只适用于上一版快照。
- `docs/perturbation_rq_revision.md`说明证据分配与边界；历史`question_driven_narrative.md`记录上一版七问题结构。

## 当前文件''')
    readme = readme.replace('当前正文图2–6及上一版图1', '当前正文图2–5、图7及上一版图1')
    readme = readme.replace('`manifest.json`统一索引当前27图', '`manifest.json`统一索引当前32图')
    readme = readme.replace('本轮为问题驱动的文字重构，核实了三篇写作结构实例，未将仅作结构借鉴的文章添加到正文科学参考文献中；图件和已有科学引用保持不变。',
        '此前问题驱动的文字重构核实了三篇写作结构实例，未将仅作结构借鉴的文章添加到正文科学参考文献中。本轮补充扰动RQ5、Fig.6和附图S22–S25；原有图件、数据及科学引用保持不变。')
    readme = readme.replace('当前正文图1使用`figures/framework_illustrated/`，图2–6使用`figures/kdense_redraw/`，附图使用`figures/task_layout/`，',
        '当前正文图1使用`figures/framework_illustrated/`，图2–5和图7使用`figures/kdense_redraw/`，图6及附图S22–S25使用`figures/perturbation_rq/`；附图S1–S21使用`figures/task_layout/`，')
    (R / 'README.md').write_text(readme)
    print('Manuscript now includes perturbation RQ5; attribute questions are RQ6--RQ8.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['figures', 'manuscript'], required=True)
    args = parser.parse_args()
    figures() if args.phase == 'figures' else manuscript()
