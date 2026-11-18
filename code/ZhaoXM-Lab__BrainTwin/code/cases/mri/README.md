# MRI case

This case answers a question about one person's structural MRI. It does not compare a cohort, and it does not build the cross-scale report.

1. `freesurfer_struct.py` turns FreeSurfer `aseg.stats` and `lh/rh.aparc.stats`, or the existing wide feature tables, into one JSON object. Volume keys look like `volume.regional.Left_Hippocampus.Volume`. Thickness keys look like `surface.regional.entorhinal.left_hemisphere.ThickAvg`.
2. `mri_session.py` reads that JSON. For a question, it selects the relevant fields, copies the stored numbers into the evidence, and asks the model to answer from those numbers.
3. `mri_visualization.py` highlights the selected cortical and subcortical regions. Cortical regions use the Desikan-Killiany atlas. Subcortical regions use the aseg atlas.

The session needs `BRAINTWIN_LLM_URL` and `BRAINTWIN_MODEL_ID`. It does not open the knowledge graph unless you pass an existing config object.

`cohort_questions.py` repeats that question across one cohort and reports the selected measurements. From the repository root:

```bash
python cases/mri/cohort_questions.py --id ucla_schizophrenia
```

It reads `data/mri/UCLA/all_sample_structured.json` and `subjects_meta.csv`, or the ADNI files `data/mri/ADNI/all_sample_structured.json` and `aseg_features.csv`. `--hybrid` also queries the literature index.

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, "cases/mri")
from freesurfer_struct import subject_from_stats
from mri_session import MRIParsingSession

sample = subject_from_stats("path/to/freesurfer/subject")
session = MRIParsingSession(sample)
result = session.ask("What is the left hippocampal volume?")
print(result["answer"])
```

Place any example FreeSurfer subject or feature tables you want to try under `data/`. The session reads the JSON you pass in. It does not assume a fixed subject file.
