#!/usr/bin/env python3
"""Download Ensembl one-to-one mouse→human gene orthologs into resources/."""
import urllib.parse
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'resources' / 'orthologs' / 'mouse_human_ensembl_1to1.tsv'
XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<!DOCTYPE Query>'
    '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" datasetConfigVersion="0.6">'
    '<Dataset name="mmusculus_gene_ensembl" interface="default">'
    '<Filter name="with_hsapiens_homolog" excluded="0"/>'
    '<Attribute name="ensembl_gene_id"/>'
    '<Attribute name="hsapiens_homolog_ensembl_gene"/>'
    '<Attribute name="hsapiens_homolog_orthology_type"/>'
    '</Dataset></Query>'
)


def main():
    url = 'https://www.ensembl.org/biomart/martservice?query=' + urllib.parse.quote(XML)
    response = requests.get(url, timeout=180, allow_redirects=True)
    response.raise_for_status()
    lines = response.text.splitlines()
    pairs = []
    seen_m, seen_h = set(), set()
    for line in lines[1:]:
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        mouse, human, kind = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if kind != 'ortholog_one2one' or not mouse.startswith('ENSMUSG') or not human.startswith('ENSG'):
            continue
        if mouse in seen_m or human in seen_h:
            continue
        seen_m.add(mouse)
        seen_h.add(human)
        pairs.append((mouse, human))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text('mouse_ensembl\thuman_ensembl\n' + '\n'.join(f'{m}\t{h}' for m, h in sorted(pairs)) + '\n')
    print(f'Wrote {len(pairs)} 1:1 orthologs to {DEST}')


if __name__ == '__main__':
    main()
