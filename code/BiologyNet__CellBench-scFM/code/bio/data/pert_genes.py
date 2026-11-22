"""Parse perturbation condition names into gene tokens."""
import numpy as np

CONTROL_TOKENS = frozenset({
    'ctrl', 'control', 'nt', 'non-targeting', 'non_targeting', 'nontargeting', '',
})
INVALID_CONDITIONS = frozenset({
    '*', 'nan', 'none', 'na', 'null', 'n/a',
})


def is_valid_perturbation(condition):
    text = str(condition).strip()
    if not text or text.lower() in CONTROL_TOKENS or text.lower() in INVALID_CONDITIONS:
        return False
    return len(parse_perturbation_genes(text)) >= 1


def parse_perturbation_genes(condition):
    text = str(condition).strip()
    if not text or text.lower() in CONTROL_TOKENS or text.lower() in INVALID_CONDITIONS:
        return []
    if '+' in text:
        parts = [part.strip() for part in text.split('+')
                 if part.strip() and part.strip().lower() not in CONTROL_TOKENS
                 and part.strip().lower() not in INVALID_CONDITIONS]
        return parts
    if '_' in text:
        parts = [part.strip() for part in text.split('_')
                 if part.strip() and part.strip().lower() not in CONTROL_TOKENS
                 and part.strip().lower() not in INVALID_CONDITIONS]
        if len(parts) >= 2:
            return parts
    return [text]


def perturbation_gene_table(conditions):
    names = sorted(set(map(str, np.asarray(conditions).tolist())))
    parsed = {name: parse_perturbation_genes(name) for name in names}
    genes = [gene for parts in parsed.values() for gene in parts]
    vocab = {'': 0}
    for gene in sorted(set(genes)):
        vocab[gene] = len(vocab)
    width = max((len(parts) for parts in parsed.values()), default=1)
    width = max(1, width)
    table = np.zeros((len(names), width), dtype=np.int64)
    for i, name in enumerate(names):
        for j, gene in enumerate(parsed[name]):
            table[i, j] = vocab[gene]
    return vocab, names, table
