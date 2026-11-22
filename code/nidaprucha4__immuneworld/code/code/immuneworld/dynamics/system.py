from dataclasses import dataclass

import torch
from torch import Tensor, nn

from immuneworld.dynamics.temporal import ContinuousTimeEncoding, TrajectoryStack
from immuneworld.dynamics.context import PerturbationEncoder, PerturbationInjector
from immuneworld.dynamics.transfer import CancerInvariantProjector


@dataclass(frozen=True)
class ImmuneWorldSettings:
    genes: int = 2000
    width: int = 768
    layers: int = 12
    heads: int = 8
    expansion: int = 4
    cancers: int = 7
    perturbation_types: int = 3
    dropout: float = 0.1


@dataclass
class WorldOutput:
    next_state: Tensor
    reconstructed: Tensor
    perturbation_state: Tensor
    shared_state: Tensor
    response_logit: Tensor
    cancer_logit: Tensor


class ImmuneWorld(nn.Module):
    def __init__(self, settings: ImmuneWorldSettings) -> None:
        super().__init__()
        self.settings = settings
        self.gene_embedding = nn.Parameter(torch.empty(settings.genes, settings.width))
        self.time_encoding = ContinuousTimeEncoding(settings.width)
        self.input_norm = nn.LayerNorm(settings.width)
        self.perturbation_encoder = PerturbationEncoder(
            settings.genes,
            settings.width,
            settings.perturbation_types,
            self.gene_embedding,
        )
        self.trajectory = TrajectoryStack(
            settings.width,
            settings.heads,
            settings.layers,
            settings.expansion,
            settings.dropout,
        )
        self.injector = PerturbationInjector(settings.width, settings.heads, settings.dropout)
        self.transfer = CancerInvariantProjector(settings.width, settings.cancers)
        self.transition_head = nn.Linear(settings.width, settings.genes)
        self.reconstruction_head = nn.Linear(settings.width, settings.genes)
        self.perturbation_head = nn.Linear(settings.width, settings.genes)
        self.response_head = nn.Linear(settings.width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.gene_embedding, std=0.02)

    def embed(self, expression: Tensor, pseudotime: Tensor) -> Tensor:
        state = expression @ self.gene_embedding
        return self.input_norm(state + self.time_encoding(pseudotime))

    def encode(
        self,
        expression: Tensor,
        pseudotime: Tensor,
        gene_ids: Tensor | None = None,
        doses: Tensor | None = None,
        type_ids: Tensor | None = None,
        perturbation_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        state = self.embed(expression, pseudotime)
        state = self.trajectory(state, pseudotime)
        context = None
        if gene_ids is not None and doses is not None and type_ids is not None:
            context = self.perturbation_encoder(gene_ids, doses, type_ids, perturbation_mask)
            state = self.injector(state, context)
        return state, context

    def forward(
        self,
        expression: Tensor,
        pseudotime: Tensor,
        gene_ids: Tensor | None = None,
        doses: Tensor | None = None,
        type_ids: Tensor | None = None,
        perturbation_mask: Tensor | None = None,
        reversal_scale: float = 1.0,
    ) -> WorldOutput:
        state, context = self.encode(
            expression,
            pseudotime,
            gene_ids,
            doses,
            type_ids,
            perturbation_mask,
        )
        shared, cancer_logit = self.transfer(state, reversal_scale)
        pooled = shared.mean(dim=1)
        perturbation_state = state if context is not None else torch.zeros_like(state)
        return WorldOutput(
            next_state=self.transition_head(state),
            reconstructed=self.reconstruction_head(state),
            perturbation_state=self.perturbation_head(perturbation_state),
            shared_state=shared,
            response_logit=self.response_head(pooled).squeeze(-1),
            cancer_logit=cancer_logit,
        )

    @torch.no_grad()
    def rollout(self, history: Tensor, times: Tensor, horizon: int) -> Tensor:
        states = [history]
        current = history
        current_times = times
        for _ in range(horizon):
            prediction = self(current, current_times).next_state[:, -1:]
            current = torch.cat((current, prediction), dim=1)
            increment = torch.full_like(current_times[:, -1:], 1.0 / max(horizon, 1))
            current_times = torch.cat((current_times, current_times[:, -1:] + increment), dim=1)
            states.append(prediction)
        return torch.cat(states, dim=1)
