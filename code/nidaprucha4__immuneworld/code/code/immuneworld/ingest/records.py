from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class CellRecord:
    cell_id: str
    expression: Tensor
    pseudotime: float
    perturbation_genes: tuple[int, ...]
    perturbation_doses: tuple[float, ...]
    perturbation_types: tuple[int, ...]
    cancer: int
    cohort: str
    patient: str
    response: int | None


@dataclass(frozen=True)
class ExpressionArchive:
    matrix: Path
    cells: Path
    genes: Path
    metadata: Path


class GeneVocabulary:
    def __init__(self, genes: list[str]) -> None:
        if len(genes) != len(set(genes)):
            raise ValueError("gene vocabulary contains duplicates")
        self.genes = tuple(genes)
        self.lookup = {gene: index for index, gene in enumerate(genes)}

    def __len__(self) -> int:
        return len(self.genes)

    def encode(self, genes: list[str], unknown: int = -1) -> np.ndarray:
        return np.asarray([self.lookup.get(gene, unknown) for gene in genes], dtype=np.int64)

    def decode(self, indices: np.ndarray) -> list[str]:
        return [self.genes[int(index)] for index in indices]

    def align(self, source: list[str], matrix: Tensor) -> Tensor:
        destination = torch.zeros(matrix.shape[0], len(self), dtype=matrix.dtype)
        for source_index, gene in enumerate(source):
            destination_index = self.lookup.get(gene)
            if destination_index is not None:
                destination[:, destination_index] = matrix[:, source_index]
        return destination
