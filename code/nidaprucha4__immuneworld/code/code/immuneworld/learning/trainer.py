from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from immuneworld.learning.objectives import CompositeObjective, LossReport


@dataclass
class Batch:
    expression: Tensor
    pseudotime: Tensor
    next_state: Tensor
    masked_genes: Tensor
    perturbation_target: Tensor
    perturbed: Tensor
    response: Tensor
    response_available: Tensor
    cancer: Tensor
    gene_ids: Tensor | None = None
    doses: Tensor | None = None
    type_ids: Tensor | None = None
    perturbation_mask: Tensor | None = None

    def to(self, device: torch.device) -> "Batch":
        values = {}
        for name, value in vars(self).items():
            values[name] = value.to(device) if isinstance(value, Tensor) else value
        return Batch(**values)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        objective: CompositeObjective,
        optimizer: Optimizer,
        device: torch.device,
        accumulation: int,
        precision: torch.dtype,
    ) -> None:
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.device = device
        self.accumulation = accumulation
        self.precision = precision
        self.step = 0

    def loss(self, batch: Batch) -> LossReport:
        output = self.model(
            batch.expression,
            batch.pseudotime,
            batch.gene_ids,
            batch.doses,
            batch.type_ids,
            batch.perturbation_mask,
        )
        return self.objective(
            output,
            batch.next_state,
            batch.expression,
            batch.masked_genes,
            batch.perturbation_target,
            batch.perturbed,
            batch.response,
            batch.response_available,
            batch.cancer,
        )

    def train_epoch(self, batches: Iterable[Batch]) -> list[float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        values = []
        for index, batch in enumerate(batches):
            batch = batch.to(self.device)
            with torch.autocast(self.device.type, dtype=self.precision, enabled=self.device.type == "cuda"):
                report = self.loss(batch)
                scaled = report.total / self.accumulation
            scaled.backward()
            if (index + 1) % self.accumulation == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.step += 1
            values.append(float(report.total.detach()))
        return values
