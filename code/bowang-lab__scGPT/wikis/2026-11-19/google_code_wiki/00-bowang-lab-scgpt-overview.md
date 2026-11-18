# bowang-lab/scgpt Overview
scGPT is a foundation model for single-cell multi-omics data. It processes and analyzes complex biological datasets, enabling data management, model training, evaluation, and downstream analysis.

The system incorporates tools for data handling, transforming raw [`AnnData`](https://github.com/bowang-lab/scgpt/scgpt/trainer.py#L9) objects into a standardized format. This includes preprocessing steps like filtering, normalization, and binning of expression values. Data collation mechanisms organize processed data into batches suitable for model input, supporting strategies such as Masked Language Modeling. [Data Management and Preprocessing](#data-management-and-preprocessing) describes these processes.

At its core, [`scGPT`](https://github.com/bowang-lab/scgpt/README.md#L1) uses Transformer architectures designed for single-cell and multi-omic data. These models integrate gene, value, and batch encoding mechanisms to represent biological information. They perform tasks including masked gene modeling, cell-type classification, perturbation prediction, and contrastive learning. [Model Architectures and Training](#model-architectures-and-training) provides further detail.

The models also incorporate adversarial domain adaptation techniques, which mitigate technical batch effects across different datasets or experimental batches. This ensures robustness and generalizability of the learned representations.

Model training follows a workflow that calculates loss functions tailored to specific tasks. The training process includes evaluation procedures that assess model performance and the quality of learned embeddings. [Training and Evaluation Workflow](#introduction-to-scgpt-scgpt-package-structure-and-core-modules-training-and-evaluation-workflow) explains this workflow.

[`scGPT`](https://github.com/bowang-lab/scgpt/README.md#L1) provides utilities for generating and analyzing embeddings. These embeddings, derived from both cells and genes, serve as representations for downstream analyses like cell clustering, similarity searches, and gene regulatory network inference. [Embedding Generation and Analysis](#embedding-generation-and-analysis) discusses these functionalities.

The system supports fine-tuning pre-trained models on specific datasets or tasks. [Example Workflows and Fine-tuning](#example-workflows-and-fine-tuning) demonstrates this process. Zero-shot capabilities allow application of pre-trained models to new datasets without additional training, for tasks such as reference mapping and data integration. [Zero-Shot Applications and Tutorials](#zero-shot-applications-and-tutorials) provides tutorials on these applications.

