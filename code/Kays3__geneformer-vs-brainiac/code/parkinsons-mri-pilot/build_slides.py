"""Rebuild the editable research update using python-pptx."""
import csv
import json
from statistics import median
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

root = Path(__file__).resolve().parent
metrics = json.loads((root / "results/metrics.json").read_text())
deck = Presentation()
deck.slide_width, deck.slide_height = Inches(13.33), Inches(7.5)
navy, teal, ink = "173143", "087975", "223B47"


def label(slide, x, y, width, height, words, size, color=ink, bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.text = words
    for p in frame.paragraphs:
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor.from_string(color)


def page(section, title):
    s = deck.slides.add_slide(deck.slide_layouts[6])
    label(s, .7, .4, 12, .45, section, 16, teal, True)
    label(s, .7, .98, 12, 1.2, title, 36, navy, True)
    return s


s = page("Parkinson’s disease  /  Structural MRI pilot", "Can a single MRI distinguish mild cognitive impairment?")
label(s, .75, 2.40, 4.3, 1.0, "33 participants", 40, teal, True)
label(s, .75, 3.65, 4.5, 1.7, "15 with mild cognitive impairment\n18 with normal cognition", 22)
label(s, 5.75, 2.50, 6.65, 2.95, "BrainIAC ranked cognitive status better than age and sex alone in this sample. At the chosen threshold, balanced accuracy was slightly lower.", 27)
label(s, .75, 6.42, 12, .6, "Kemp et al.  ·  OpenNeuro ds005892, version 1.0.0  ·  Cross-sectional analysis", 15)

s = page("Methods", "Each participant contributed one held-out prediction")
for x, heading, words in [
    (.75, "Prepare each scan", "Register each T1 scan to a common template, apply the template mask and resize to 96 × 96 × 96 voxels."),
    (4.95, "Extract MRI features", "Use the frozen BrainIAC encoder. Within each training fold, reduce its 768 features to three principal components."),
    (9.05, "Compare models", "Fit class-balanced logistic regression on four folds. Predict the fifth. Repeat across five folds using the same split for all models.")]:
    label(s, x, 2.75, 3.65, .9, heading, 24, teal, True)
    label(s, x, 3.65, 3.6, 2.8, words, 21)

s = page("Held-out results", "MRI features gave the highest AUROC")
s.shapes.add_picture(str(root / "results/roc.png"), Inches(.7), Inches(2.05), height=Inches(4.85))
for x, text, w in [(6.7,"Predictors",2.8),(9.5,"AUROC",1.2),(10.85,"Balanced\naccuracy",1.8)]:
    label(s,x,2.45,w,1,text,18,teal,True)
for idx,(name,values) in enumerate(metrics.items()):
    y = 3.55 + idx * .82
    label(s,6.7,y,2.8,.8,name,18)
    label(s,9.5,y,1.2,.7,f"{values['AUROC']:.2f}",21,navy,True)
    label(s,10.85,y,1.8,.7,f"{values['balanced_accuracy']:.1%}",21,navy,True)
label(s,6.7,6.25,5.7,.78,"Balanced accuracy averages sensitivity and specificity at a score threshold of 0.5.",16)

s = page("Interpretation and next study", "What would justify a progression study?")
for y, heading, words in [
    (2.1, "Improve scan alignment", "Review showed residual orbital tissue and mismatched anatomy in two scans. Repeat preprocessing with individual brain extraction and review alignment before comparing models."),
    (3.75, "Check reproducibility", "The estimates come from one five-fold split of 33 participants. Validate the cognitive-status result in an independent cohort."),
    (5.36, "Test motor progression separately", "Link baseline MRI to 24-month MDS-UPDRS Part III motor scores and medication state in an approved PPMI dataset. Compare imaging against a baseline clinical forecast.")]:
    label(s,.75,y,3.8,1.3,heading,23,teal,True)
    label(s,4.8,y,7.75,1.45,words,21)
s = page("Published PPMI cohort  /  External reference", "Motor examination scores over 24 months")
label(s, .75, 2.05, 12, .8, "Mean MDS-UPDRS Part III score in the untreated + treated OFF group", 23)
for x, visit, mean, count in [
    (.75, "Baseline", "20.89", 423), (3.9, "6 months", "24.87", 400),
    (7.05, "12 months", "25.10", 334), (10.2, "24 months", "27.13", 282)
]:
    label(s, x, 3.1, 2.7, .5, visit, 20)
    label(s, x, 3.7, 2.7, .85, mean, 35, teal, True)
    label(s, x, 4.65, 2.8, .6, f"{count} exams", 17)
label(s, .75, 5.65, 11.85, .95, "The group mean rose by 6.24 points. Different numbers contributed at each visit. Individual change requires paired records.", 21)
label(s, .75, 6.85, 12, .44, "Simuni et al., Movement Disorders (2018), Table 2. DOI 10.1002/mds.27361", 14)
s = page("Proposed study  /  Before diagnosis", "How early was a change detectable?")
label(s, .75, 2.0, 12, .78, "Illustrative smell-test timeline for one participant", 23, teal, True)
for x, time, event in [
    (.75, "−36 months", "Test within reference range"),
    (4.85, "−18 months", "First abnormal test"),
    (9.05, "Month 0", "Clinical PD diagnosis"),
]:
    label(s, x, 3.0, 3.7, .65, time, 28, navy, True)
    label(s, x, 3.85, 3.7, 1.2, event, 21)
label(s, .75, 5.15, 11.9, 1.25, "The marker crossed the threshold between the two tests. Its biological cause and the anatomical site of onset remain unknown.", 22)
label(s, .75, 6.75, 12, .5, "Design precedent: Jennings et al., PARS cohort (2017). Example dates are hypothetical.", 15)
s = page("Validation  /  Cognitive status at the scan visit", "AUC varied with the participant split")
with (root / "results/auc-by-split.csv").open() as f:
    runs = list(csv.DictReader(f))
label(s, .75, 2.0, 12, .8, "50 stratified five-fold splits of the same 33 people", 24)
for x, name in zip([.75, 4.85, 9.0], metrics):
    values = [float(run[name]) for run in runs]
    label(s, x, 3.0, 3.8, .85, name, 21, navy, True)
    label(s, x, 3.95, 3.8, .85, f"{median(values):.2f}", 34, teal, True)
    label(s, x, 5.0, 3.8, .7, f"Range {min(values):.2f} to {max(values):.2f}", 19)
label(s, .75, 6.05, 11.85, 1.0, "Values show median held-out AUC and the observed range across splits. Independent validation is needed to estimate performance in new patients.", 19)
deck.save(root / "Parkinsons-MRI-pilot.pptx")
