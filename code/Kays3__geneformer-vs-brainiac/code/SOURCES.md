# Sources

Read on 15 November 2026. Article details are from PubMed.

**Geneformer**

- Theodoris CV, et al. Transfer learning enables predictions in network biology. *Nature* 618, 616-624 (2023). [doi:10.1038/s41586-023-06139-9](https://doi.org/10.1038/s41586-023-06139-9). Source for: about 30 million transcriptomes, self-supervised pretraining, better predictions with limited data, candidate targets for cardiomyopathy.
- Zhang Y, Venkatesh MS, Theodoris CV. Discovery of candidate therapeutic targets with Geneformer. *Nature Protocols* 21, 5009-5032 (2026). [doi:10.1038/s41596-026-01364-8](https://doi.org/10.1038/s41596-026-01364-8). Source for: more than 100 million transcriptomes, rank value encoding, zero-shot embeddings, single-task and multi-task fine-tuning, macro F1 and confusion matrices, in silico perturbation, the quantized model.
- Model and code: <https://huggingface.co/ctheodoris/Geneformer>.
- The 768-number embedding of the V2-104M model: the README of <https://github.com/Kays3/geneformer-petadimension-starter>.

**BrainIAC**

- Tak D, et al. A generalizable foundation model for analysis of human brain MRI. *Nature Neuroscience* 29, 945-956 (2026). [doi:10.1038/s41593-026-02202-6](https://doi.org/10.1038/s41593-026-02202-6). Source for: 48,965 brain MRIs, the comparison with supervised training and other pretrained models, few-shot settings, use in pipelines and multimodal frameworks.
- Code: <https://github.com/AIM-KannLab/BrainIAC>, commit `ba60f45`. `src/model.py` for the architecture and the SimCLR checkpoint; `README.md` and `docs/downstream_tasks/` for the seven tasks, the weights link, the platform and the licence; `src/get_brainiac_saliencymap.py` for the attention saliency maps; `src/test_inference_perturbation.py` for the robustness script.

The questions in the README and on slide 6, and the closing slide, are the author's own. They are not taken from either paper.
