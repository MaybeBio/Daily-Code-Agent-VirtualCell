from dataclasses import dataclass

import torch
from torch import Tensor

@dataclass
class MeanAbsoluteError:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class MeanSquaredError:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class RootMeanSquaredError:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class MedianAbsoluteError:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class CosineSimilarity:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class ExplainedVariance:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class ConcordanceCorrelation:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class SpearmanCorrelation:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class PrecisionScore:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class RecallScore:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class SpecificityScore:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class BalancedAccuracy:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class JaccardScore:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class DiceScore:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class BrierScore:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class LogLoss:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class CalibrationError:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class SensitivityAtSpecificity:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class PositivePredictiveValue:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class NegativePredictiveValue:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class TrajectoryEndpointError:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class TrajectoryPathLengthError:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class VelocityCosine:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class BoundaryAccuracy:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class ModuleSparsity:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class DomainAccuracy:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class ResponseAccuracy:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class PerturbationDeltaError:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class ExpressionRecoveryError:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class MaskedGeneAccuracy:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class TemporalMonotonicity:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class RolloutStability:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class CohortDispersion:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class CancerDispersion:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class SeedDispersion:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class TransferGain:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class AblationDrop:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class InferenceThroughput:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class PeakMemory:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class ParameterCount:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class GradientNorm:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class UpdateNorm:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class LossScale:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class WarmupFraction:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class EffectiveBatch:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class CellCoverage:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class GeneCoverage:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class PerturbationCoverage:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class CohortCoverage:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class CancerCoverage:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class PatientCoverage:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class ResponderCoverage:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class NonResponderCoverage:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class ModuleCoverage:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class LineageCoverage:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class TissueCoverage:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0

@dataclass
class TrajectoryCoverage:
    total: Tensor | None = None
    count: int = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        value = (prediction.float() - target.float()).abs().mean()
        self.total = value.detach() if self.total is None else self.total + value.detach()
        self.count += 1

    def compute(self) -> Tensor:
        if self.total is None or self.count == 0:
            return torch.tensor(float("nan"))
        return self.total / self.count

    def reset(self) -> None:
        self.total = None
        self.count = 0

    def merge(self, other: object) -> None:
        if not isinstance(other, self.__class__):
            raise TypeError("metric types differ")
        if other.total is not None:
            self.total = other.total.clone() if self.total is None else self.total + other.total
            self.count += other.count

    def state_dict(self) -> dict[str, Tensor | int | None]:
        return {"total": self.total, "count": self.count}

    def load_state_dict(self, state: dict[str, Tensor | int | None]) -> None:
        total = state["total"]
        count = state["count"]
        self.total = total if isinstance(total, Tensor) else None
        self.count = int(count) if isinstance(count, int) else 0
