"""Mouse→human 1:1 Ensembl orthologs for human-pretrained encoders."""
from pathlib import Path

from bio.io.paths import project_path

TABLE = 'resources/orthologs/mouse_human_ensembl_1to1.tsv'
_MAPPING = None


def _strip(ensembl):
    text = str(ensembl).strip()
    return text.split('.')[0] if text else text


def mouse_to_human_ensembl():
    global _MAPPING
    if _MAPPING is None:
        path = project_path(TABLE)
        if not path.is_file():
            raise FileNotFoundError(
                f'Missing {path}; rebuild with python scripts/build_mouse_human_orthologs.py')
        mapping = {}
        for line in path.read_text().splitlines()[1:]:
            if not line.strip():
                continue
            mouse, human = line.split('\t')[:2]
            mapping[_strip(mouse)] = _strip(human)
        _MAPPING = mapping
    return _MAPPING


def looks_like_mouse_ensembl(names):
    sample = [str(n) for n in list(names)[:32]]
    return any(_strip(n).startswith('ENSMUSG') for n in sample)


def project_mouse_to_human_ensembl(adata):
    """Return a copy whose var_names are 1:1 human Ensembl IDs.

    Genes without a unique mouse→human ortholog are dropped. Human datasets
    (ENSG / symbols) are returned unchanged.
    """
    names = adata.var_names.astype(str)
    if not looks_like_mouse_ensembl(names):
        return adata
    mapping = mouse_to_human_ensembl()
    keep = []
    human = []
    seen = set()
    for i, name in enumerate(names):
        hid = mapping.get(_strip(name))
        if not hid or hid in seen:
            continue
        seen.add(hid)
        keep.append(i)
        human.append(hid)
    if not keep:
        raise ValueError('No mouse genes mapped to human 1:1 Ensembl orthologs')
    out = adata[:, keep].copy()
    out.var_names = human
    out.var['human_ensembl'] = human
    out.var['mouse_ensembl'] = names[keep]
    return out
