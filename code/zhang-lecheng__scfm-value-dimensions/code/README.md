# Do single-cell foundation models need 512 dimensions for one number?

Lecheng Zhang, Yuhao Yang and Stan Z. Li — AI Research and Innovation Laboratory, School of Engineering, Westlake University

Contact: zhanglecheng@westlake.edu.cn

## Abstract

Single-cell foundation models lift each expression count into hundreds of dimensions. We asked how many the count needs and what the network computes with it. In scGPT and scFoundation a straight line can replace the learned value curve with little change in downstream transfer, and clamping detected genes to a constant keeps 97–101% of annotation and surface-protein gain: downstream tasks read mainly which genes are detected. With gene identity on its own 512-dimensional track, a four-dimensional value track keeps 95% of the masked-prediction gain. Beyond dataset-by-gene means, masked prediction is a gene offset plus a gene-specific readout of three linearly computed cell numbers, which describe, on held-out donors, 72% of a controlled Transformer's function beyond the offsets and 47% of scGPT's. In the controlled models the leading number tracks sequencing depth (in scGPT, cell type weighs more), and thinning a cell's counts moves the predictions as this description predicts; whether that number follows detected genes or library size depends on the objective and inputs. Several interpretability readouts are driven by depth, gene frequency or hub genes; what can be decoded is not what the model uses.

## Status

**Coming soon.** This repository will hold the material behind the preprint:

- training and analysis code at the versions used;
- figure scripts with per-figure source data (per-cell, per-query, per-donor and per-perturbation exports);
- the results registry, with a SHA-256 checksum for every cited result file;
- a table of the exact values behind every statement in the Results.

Releases will be tagged with the preprint version.
