from dataclasses import dataclass

import torch
from torch import Tensor

@dataclass(frozen=True)
class CorpusShard:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class CellObservation:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class TrajectoryWindow:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class PerturbationContext:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class ClinicalOutcome:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class CancerDomain:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class CohortIdentity:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class PatientIdentity:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class GeneSelection:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class NormalizationProfile:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class PseudotimePath:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class DevelopmentBranch:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class VelocityField:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class SyntheticMixture:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class ResponseFold:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class CancerFold:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class ExperimentSeed:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class MetricEstimate:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class BootstrapDraw:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class PermutationDraw:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class ModuleAttribution:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class GeneModule:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class SurvivalRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class HazardEstimate:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class DecisionThreshold:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class TrainingProgress:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class OptimizerSnapshot:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class SchedulePosition:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class GradientSummary:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class MemoryEstimate:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class CorpusManifest:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class AccessRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class EvaluationRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class PredictionRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class RolloutRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class DeconvolutionRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class TrajectoryRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class PerturbationRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class ResponseRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class AblationRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class TemporalBiasRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class AttentionRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class EmbeddingRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class TransferRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class DiscriminatorRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class MaskRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class VocabularyRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class LineageRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class TissueRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])

@dataclass(frozen=True)
class SimulationRecord:
    identifier: str
    values: Tensor
    labels: Tensor | None
    weight: float = 1.0

    def validate(self) -> None:
        if not self.identifier:
            raise ValueError("identifier cannot be empty")
        if self.values.numel() == 0:
            raise ValueError("values cannot be empty")
        if not torch.isfinite(self.values).all():
            raise ValueError("values must be finite")
        if self.weight < 0.0:
            raise ValueError("weight must be nonnegative")

    def normalized(self) -> Tensor:
        denominator = torch.linalg.vector_norm(self.values).clamp_min(1e-12)
        return self.values / denominator

    def mean(self) -> Tensor:
        return self.values.mean(dim=0)

    def variance(self) -> Tensor:
        return self.values.var(dim=0, unbiased=False)

    def select(self, indices: Tensor) -> Tensor:
        return self.values.index_select(0, indices)

    def detached(self) -> Tensor:
        return self.values.detach().clone()

    def weighted(self) -> Tensor:
        return self.values * self.weight

    def size(self) -> int:
        return int(self.values.shape[0])
