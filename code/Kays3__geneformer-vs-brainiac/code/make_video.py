"""Build the Geneformer vs BrainIAC comparison video.

    uv run --with pillow --with imageio-ffmpeg python make_video.py

Draws one PNG per slide, speaks the narration with the macOS `say` command,
and joins the segments with ffmpeg (taken from the imageio-ffmpeg package).
Every statement on a slide has a public source in SOURCES.md.
"""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BUILD = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "build"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

W, H = 1920, 1080
BG = (15, 23, 42)
INK = (241, 245, 249)
MUTED = (148, 163, 184)
GF = (45, 212, 191)   # Geneformer
BR = (192, 132, 252)  # BrainIAC
WARN = (251, 191, 36)
CARD = (30, 41, 59)
FONT_DIR = "/System/Library/Fonts/Supplemental/"
VOICE = "Daniel"
FOOTER = "Public sources only · github.com/Kays3/geneformer-vs-brainiac"


def font(size, bold=False):
    return ImageFont.truetype(FONT_DIR + ("Arial Bold.ttf" if bold else "Arial.ttf"), size)


def wrap(draw, text, fnt, width):
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            trial = (line + " " + word).strip()
            if draw.textlength(trial, font=fnt) <= width:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def para(draw, xy, text, fnt, width, fill=INK, gap=1.3):
    x, y = xy
    for line in wrap(draw, text, fnt, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += int(fnt.size * gap)
    return y


def base(title, n, total):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((90, 60), title, font=font(58, True), fill=INK)
    d.rectangle((90, 140, 250, 146), fill=GF)
    d.rectangle((250, 140, 410, 146), fill=BR)
    d.text((90, H - 60), FOOTER, font=font(24), fill=MUTED)
    label = f"{n} / {total}"
    d.text((W - 90 - d.textlength(label, font=font(24)), H - 60), label, font=font(24), fill=MUTED)
    return img, d


def column(d, x, head, colour, items, width=800, y=200, size=34):
    d.rounded_rectangle((x, y, x + width, H - 110), radius=24, fill=CARD)
    d.rectangle((x, y + 24, x + 8, y + 74), fill=colour)
    d.text((x + 36, y + 24), head, font=font(40, True), fill=colour)
    yy = y + 110
    for item in items:
        d.ellipse((x + 40, yy + 13, x + 52, yy + 25), fill=colour)
        yy = para(d, (x + 72, yy), item, font(size), width - 112) + 18


def two_columns(title, n, total, left, right, size=34):
    img, d = base(title, n, total)
    column(d, 90, "Geneformer", GF, left, size=size)
    column(d, 1030, "BrainIAC", BR, right, size=size)
    return img




TOTAL = 8
SLIDES = []


def slide(img, narration):
    SLIDES.append((img, narration))


# 1 title
img, d = base("Two foundation models, side by side", 1, TOTAL)
d.text((90, 300), "Geneformer", font=font(84, True), fill=GF)
d.text((90, 410), "compared with", font=font(48), fill=MUTED)
d.text((90, 490), "BrainIAC", font=font(84, True), fill=BR)
para(d, (90, 670),
     "Geneformer: Theodoris et al., Nature 2023; Zhang, Venkatesh and Theodoris, Nature Protocols 2026.\n"
     "BrainIAC: Tak et al., Nature Neuroscience 2026, and the public AIM-KannLab code.\n"
     "Kaisar Dauyey. A working comparison from published sources; no new results.",
     font(32), 1700, fill=MUTED)
slide(img, "This video puts two biomedical foundation models side by side. Gene former reads single cells. "
           "Brainiac reads brain M R I. They do not compete, so this is a comparison of what each one is, "
           "how each one is used, and how each one is tested. Everything here comes from published sources.")

# 2 input and model
slide(two_columns("What each model sees", 2, TOTAL,
                  ["Input: the transcriptome of one cell",
                   "A cell becomes a list of gene tokens ranked by expression (rank value encoding)",
                   "A transformer turns the list into an embedding (768 numbers in the V2-104M model)",
                   "Pretraining corpus: about 30 million single-cell transcriptomes at first, now more than 100 million"],
                  ["Input: one structural brain MRI scan, in 3D",
                   "Scan resized to 96 x 96 x 96 voxels and cut into 16-voxel patches",
                   "3D vision transformer: 12 layers, 12 heads, 768 hidden units",
                   "Self-supervised contrastive pretraining (SimCLR)",
                   "Trained and validated on 48,965 brain MRIs"]),
      "First, what each model sees. Gene former reads one cell as a list of genes, ranked by expression. "
      "It was pretrained on about thirty million single cell transcriptomes, and now on more than one hundred million. "
      "Brainiac reads one three D brain scan, cut into patches. It is a vision transformer with twelve layers, "
      "pretrained by contrastive learning. Its paper reports forty eight thousand nine hundred and sixty five scans "
      "for training and validation.")

# 3 adaptation
slide(two_columns("How each one is put to work", 3, TOTAL,
                  ["Zero-shot embeddings first: are the phenotypes separable at all?",
                   "Fine-tuning with one task (a disease state within a cell type) or several tasks at once",
                   "In silico perturbation: repress or activate a gene and measure the shift of the cell embedding",
                   "A quantized model is supported, for speed"],
                  ["One pretrained backbone, one small head per task",
                   "Brain age, IDH mutation, mild cognitive impairment, glioma overall survival, MR sequence type, time to stroke, tumour segmentation",
                   "Training, inference and saliency scripts for each task",
                   "A hosted platform runs inference on uploaded scans"], size=32),
      "Second, how each one is put to work. The Gene former protocol starts with zero shot embeddings, "
      "then fine tunes on one task or several, and then runs in silico perturbation: repress or activate a gene, "
      "and measure how the cell embedding moves. Brainiac keeps one backbone and adds one small head per task. "
      "Its repository carries seven tasks, from brain age to tumour segmentation.")

# 4 claims
slide(two_columns("What each paper claims", 4, TOTAL,
                  ["Better predictions in network biology when task data are limited",
                   "Candidate therapeutic targets for cardiomyopathy from in silico perturbation",
                   "The protocol reports macro F1, confusion matrices and embedding analysis, then ranks perturbations by statistical and biological metrics"],
                  ["Better than supervised training on local data and than other pretrained models",
                   "The gain is largest with little data (few-shot) and on the hardest prediction tasks",
                   "Can serve as a core for imaging pipelines and multimodal frameworks"]),
      "Third, what each paper claims. Both make the same kind of claim: pretraining helps most when labelled data are scarce. "
      "Gene former adds a second claim, that in silico perturbation can point to candidate therapeutic targets. "
      "Brainiac reports that it beats supervised training and other pretrained models, most of all in few shot settings.")

# 5 interpretation
slide(two_columns("How each one explains itself", 5, TOTAL,
                  ["In silico perturbation",
                   "The output is a shift in the model's embedding of a cell. It is a prediction, not an experiment",
                   "The unit of explanation is a gene"],
                  ["Saliency maps from the vision transformer's attention",
                   "A saliency map shows where the model looked, not why the outcome happens",
                   "The unit of explanation is a brain region"]),
      "Fourth, how each model explains itself. Gene former uses in silico perturbation. The output is a shift in the model's "
      "embedding of a cell, so it is a prediction, not an experiment. Brainiac uses saliency maps taken from attention. "
      "A saliency map shows where the model looked, not why the outcome happens. In one case the unit of explanation is a gene. "
      "In the other it is a brain region.")

# 6 questions
img, d = base("Questions to ask of either model", 6, TOTAL)
y = 200
for colour, text in [
    (GF, "Is the test split by donor or patient, not by cell or scan?"),
    (BR, "Does the classifier fail when the labels are permuted?"),
    (GF, "Do random genes, or random brain regions, score as often as the reported ones?"),
    (BR, "Does a second site or cohort give the same answer?"),
    (GF, "Does the result survive a change of seed, hardware, numeric precision or model size?"),
    (WARN, "Were the benchmarks fixed before the run, and are the negative results reported?"),
]:
    d.ellipse((100, y + 16, 116, y + 32), fill=colour)
    y = para(d, (146, y), text, font(42), 1640, fill=INK) + 30
para(d, (90, y + 10), "These are the author's working questions for evaluating foundation models. "
     "They are not findings about either model.", font(30), 1700, fill=MUTED)
slide(img, "Fifth, the questions I would ask of either model. Is the test split by donor or patient, not by cell or scan? "
           "Does the classifier fail when labels are permuted? Do random genes, or random brain regions, score as often as the reported ones? "
           "Does a second site give the same answer? Does the result survive a change of seed, hardware, precision or model size? "
           "And were the benchmarks fixed before the run? These are questions, not findings about either model.")

# 7 access
slide(two_columns("Access", 7, TOTAL,
                  ["Model and code on Hugging Face (ctheodoris/Geneformer)",
                   "Published step-by-step protocol",
                   "A beginner tutorial and tested quickstart by the author: Kays3/geneformer-petadimension-starter"],
                  ["Code on GitHub (AIM-KannLab/BrainIAC)",
                   "Weights and downstream checkpoints by download link",
                   "Licence: non-commercial academic research only",
                   "Robustness script: contrast, bias-field and Gibbs-noise changes to the input scan"]),
      "Sixth, access. Gene former's model and code are on Hugging Face, with a published protocol. "
      "Brainiac's code is on Git Hub, with weights by download link and a licence for non commercial academic research only.")

# 8 close
img, d = base("What this comparison can and cannot say", 8, TOTAL)
y = 210
for colour, text in [
    (INK, "It is not a head-to-head: different data, different tasks, no shared metric."),
    (INK, "Both follow one recipe: self-supervised pretraining, then light adaptation where labels are scarce."),
    (INK, "Both explain themselves with a model-internal signal, which needs a null control before it is read as biology."),
    (WARN, "Next in this repository: run the BrainIAC quickstart on its public sample data, then try the six questions on it."),
]:
    d.ellipse((100, y + 16, 116, y + 32), fill=colour)
    y = para(d, (146, y), text, font(42), 1640, fill=INK) + 34
slide(img, "To close. This is not a head to head. The data, the tasks and the metrics differ. "
           "But both models follow one recipe: self supervised pretraining, then light adaptation where labels are scarce. "
           "And both explain themselves with a signal from inside the model, which needs a null control before it is read as biology. "
           "The next step in this repository is to run the Brainiac quickstart on public sample data, and try those questions on it.")


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def duration(path):
    out = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True).stderr
    h, m, s = out.split("Duration: ")[1].split(",")[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    assert len(SLIDES) == TOTAL
    BUILD.mkdir(parents=True, exist_ok=True)
    segments = []
    for i, (img, text) in enumerate(SLIDES, 1):
        png, aiff, seg = BUILD / f"s{i:02d}.png", BUILD / f"s{i:02d}.aiff", BUILD / f"s{i:02d}.mp4"
        img.save(png)
        run(["say", "-v", VOICE, "-r", "170", "-o", str(aiff), text])
        total = duration(aiff) + 1.4
        run([FFMPEG, "-y", "-loop", "1", "-framerate", "30", "-i", str(png), "-i", str(aiff),
             "-t", f"{total:.2f}",
             "-vf", f"fade=in:st=0:d=0.4,fade=out:st={total - 0.4:.2f}:d=0.4,format=yuv420p",
             "-af", "adelay=600:all=1,apad",
             "-c:v", "libx264", "-tune", "stillimage", "-crf", "20", "-r", "30",
             "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", str(seg)])
        segments.append(seg)
        print(f"slide {i}: {total:.1f} s")
    listing = BUILD / "list.txt"
    listing.write_text("".join(f"file '{s}'\n" for s in segments))
    out = HERE / "geneformer-vs-brainiac.mp4"
    run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
         "-movflags", "+faststart", str(out)])
    run([FFMPEG, "-y", "-i", str(out), "-an", "-c:v", "copy", str(HERE / "geneformer-vs-brainiac-silent.mp4")])
    (HERE / "narration.txt").write_text("\n\n".join(f"[{i}] {t}" for i, (_, t) in enumerate(SLIDES, 1)) + "\n")
    print(f"{out}  {duration(out):.1f} s")


if __name__ == "__main__":
    main()
