from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class NormalizationState:
    library_target: float
    means: Tensor
    variances: Tensor
    selected: Tensor


class TrainingFittedNormalizer:
    def __init__(self, genes: int = 2000, library_target: float = 10000.0) -> None:
        self.genes = genes
        self.library_target = library_target
        self.state: NormalizationState | None = None

    def library_normalize(self, matrix: Tensor) -> Tensor:
        library = matrix.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.log1p(matrix * self.library_target / library)

    def fit(self, matrix: Tensor) -> NormalizationState:
        normalized = self.library_normalize(matrix)
        means = normalized.mean(dim=0)
        variances = normalized.var(dim=0, unbiased=False)
        selected = torch.topk(variances, min(self.genes, matrix.shape[1])).indices.sort().values
        self.state = NormalizationState(self.library_target, means, variances, selected)
        return self.state

    def transform(self, matrix: Tensor) -> Tensor:
        if self.state is None:
            raise RuntimeError("normalizer has not been fitted")
        normalized = self.library_normalize(matrix)
        return normalized.index_select(-1, self.state.selected)

    def fit_transform(self, matrix: Tensor) -> Tensor:
        self.fit(matrix)
        return self.transform(matrix)


def pca(matrix: Tensor, components: int = 50) -> Tensor:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, _, right = torch.pca_lowrank(centered, q=min(components, min(centered.shape)))
    return centered @ right


def diffusion_pseudotime(representation: Tensor, root: int, neighbors: int = 30) -> Tensor:
    distance = torch.cdist(representation, representation)
    indices = distance.topk(min(neighbors + 1, distance.shape[0]), largest=False).indices[:, 1:]
    affinity = torch.zeros_like(distance)
    rows = torch.arange(distance.shape[0], device=distance.device).unsqueeze(1).expand_as(indices)
    local = distance[rows, indices]
    scale = local[:, -1:].clamp_min(torch.finfo(distance.dtype).eps)
    affinity[rows, indices] = torch.exp(-local.square() / scale.square())
    affinity = torch.maximum(affinity, affinity.T)
    transition = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    eigenvalues, eigenvectors = torch.linalg.eig(transition)
    order = eigenvalues.real.argsort(descending=True)
    coordinates = eigenvectors.real[:, order[1:11]] * eigenvalues.real[order[1:11]]
    score = torch.linalg.vector_norm(coordinates - coordinates[root], dim=-1)
    return (score - score.min()) / (score.max() - score.min()).clamp_min(1e-12)


def reject_ambiguous(pseudotime: Tensor, ambiguity: Tensor, threshold: float = 0.9) -> Tensor:
    return (ambiguity <= threshold) & torch.isfinite(pseudotime)


def synthetic_bulk(matrix: Tensor, labels: Tensor, mixtures: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    classes = int(labels.max().item()) + 1
    concentration = torch.ones(classes, device=matrix.device)
    weights = torch.distributions.Dirichlet(concentration).sample((mixtures,))
    profiles = []
    for label in range(classes):
        profiles.append(matrix[labels == label].mean(dim=0))
    references = torch.stack(profiles)
    noise = torch.randn(mixtures, matrix.shape[1], generator=generator, device=matrix.device) * 0.01
    return weights @ references + noise, weights
