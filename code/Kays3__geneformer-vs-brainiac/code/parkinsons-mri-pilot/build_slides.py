"""Rebuild the editable research update using python-pptx."""
import json
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
deck.save(root / "Parkinsons-MRI-pilot.pptx")
