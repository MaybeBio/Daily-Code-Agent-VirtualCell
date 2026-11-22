# 👋 Khachatur Dallakyan

🧬 PhD student at [Institut Pasteur](https://www.pasteur.fr/en), Paris · foundation models for biology and healthcare · explainable AI

🌐 [Website](https://khachdallak02.github.io/) · 🎓 [Google Scholar](https://scholar.google.com/citations?user=ydcHjw4AAAAJ&hl=en) · 💼 [LinkedIn](https://www.linkedin.com/in/khachatur-dallakyan-461674172) · ✉️ [Email](mailto:khachatur.dallakyan@gmail.com)

### 🤝 Who I work with

Since October 2026 I have been a PhD student in [Laura Cantini's lab](https://research.pasteur.fr/en/member/laura-cantini/), Machine Learning for Integrative Genomics, at [Institut Pasteur](https://www.pasteur.fr/en). Before that I did research in the [Centre for Perceptual and Interactive Intelligence (CPII)](https://cpii.hk/) and the [Department of Computer Science and Engineering](https://www.cse.cuhk.edu.hk/), working closely with [Yu Li](https://liyu95.com/) and [Yimin Fan](https://fanyimin-cuhk.github.io/). During my Master's I joined the [Marklund Lab](https://www.marklundlab.org/) at Karolinska Institutet, where I worked with [Ulrika Marklund](https://ki.se/personer/ulrika-marklund) and [Ziwei Liu](https://ki.se/en/people/ziwei-liu).

### 🔬 What I did before

At CUHK (BEng in Artificial Intelligence: Systems and Technologies, 2024) my final-year project evaluated single-cell foundation models, and I co-authored papers on a diffusion model that predicts how cell morphology changes under perturbation, on benchmarking foundation models for single-cell multi-omics integration, and, as co-first author, on benchmarking single-cell search engines. In Stockholm (joint MSc in Health Informatics, Karolinska Institutet and Stockholm University, 2026) I analysed single-cell data from the developing enteric nervous system with fine-tuned foundation models, co-authored the lab's study of cell lineages in the developing gut, and wrote my thesis on surgeon-level features for postoperative risk prediction.

| [<img src="https://khachdallak02.github.io/images/1paper.png" width="200" alt="Inference modes of the morphology prediction model">](https://www.nature.com/articles/s41467-025-63478-z) | [<img src="https://khachdallak02.github.io/images/2paper.png" width="200" alt="Overview of the SCMBench benchmark">](https://www.nature.com/articles/s41467-026-72570-x) | [<img src="https://khachdallak02.github.io/images/5paper.png" width="200" alt="In utero transduction in the developing mouse gut">](https://www.pnas.org/doi/10.1073/pnas.2614077123) | [<img src="https://khachdallak02.github.io/images/3paper.png" width="200" alt="Benchmark of single-cell search engines">](https://khachdallak02.github.io/research/) |
|:---:|:---:|:---:|:---:|
| [Morphology under perturbation](https://www.nature.com/articles/s41467-025-63478-z)<br>*Nature Communications* | [SCMBench](https://www.nature.com/articles/s41467-026-72570-x)<br>*Nature Communications* | [Cell lineages in the developing gut](https://www.pnas.org/doi/10.1073/pnas.2614077123)<br>*PNAS* | [Single-cell search engines](https://khachdallak02.github.io/research/)<br>preprint |

### 🚀 What I'm working on now

My main project is bringing new modalities into single-cell foundation models. Today's models learn from the gene expression of dissociated cells, so they know what a cell expresses but not where it sits in the tissue or which parts of its genome are open. I am extending them, starting from the lab's scPRINT models, to read spatial transcriptomics and then chromatin accessibility, so that one model gives one embedding that keeps cell identity and adds tissue context: a step towards a multimodal virtual cell.

<p align="center">
  <img src="https://github.com/Khachdallak02/Khachdallak02/raw/HEAD/multimodal-scfm.svg" width="880" alt="Gene expression, position in the tissue and chromatin accessibility are each turned into tokens, pass through one transformer, and come out as a single cell embedding in which cells group by type and by tissue region.">
</p>

---

### <img src="https://github.com/Khachdallak02/Khachdallak02/raw/HEAD/panther.svg" height="40"> The German Tank Problem (WWII)

🔢 During WWII, to estimate German <img src="https://github.com/Khachdallak02/Khachdallak02/raw/HEAD/panther.svg" height="16"> tank production, economists looked at the **serial numbers** stamped on parts of captured and destroyed tanks: ⚙️ gearboxes, 🧱 chassis and 🛞 road wheels.
📐 If *k* tanks are captured and the highest serial number seen is *m*, the best estimate of total production is: **N ≈ m + m/k − 1**

📊 After the war, the estimates were checked against Germany's own production records:

| 📅 Month | 🧮 Statistical estimate | ✅ Actual |
|---|---|---|
| Jun 1940 | 169 | 122 |
| Jun 1941 | 244 | 271 |
| Aug 1942 | 327 | 342 |
| Feb 1944 | 270 | 276 | (<img src="https://github.com/Khachdallak02/Khachdallak02/raw/HEAD/panther.svg" height="16"> Panther)

🛞 The Panther estimate came from the road wheels of just **2 captured Panther tanks**.
🎯 In every month, the statistical estimate was within about 50 tanks of the real number, and for the Panther it was off by just 6.

📚 **Sources:** [Wikipedia](https://en.wikipedia.org/wiki/German_tank_problem) · [Scientific American](https://www.scientificamerican.com/article/how-math-helped-the-allies-win-world-war-ii/) · [War History Online](https://www.warhistoryonline.com/instant-articles/the-german-tank-problem.html)
