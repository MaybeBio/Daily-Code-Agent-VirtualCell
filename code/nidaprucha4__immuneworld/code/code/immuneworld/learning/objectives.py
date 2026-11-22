from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn import functional as F

from immuneworld.dynamics.system import WorldOutput


@dataclass(frozen=True)
class ObjectiveWeights:
    reconstruction: float = 0.5
    perturbation: float = 0.3
    adversarial: float = 0.1
    response: float = 1.0


@dataclass
class LossReport:
    total: Tensor
    trajectory: Tensor
    reconstruction: Tensor
    perturbation: Tensor
    response: Tensor
    discriminator: Tensor


class CompositeObjective(nn.Module):
    def __init__(self, weights: ObjectiveWeights = ObjectiveWeights()) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        output: WorldOutput,
        next_state: Tensor,
        expression: Tensor,
        masked_genes: Tensor,
        perturbation_target: Tensor,
        perturbed: Tensor,
        response: Tensor,
        response_available: Tensor,
        cancer: Tensor,
    ) -> LossReport:
        trajectory = F.mse_loss(output.next_state[:, :-1], next_state[:, 1:])
        reconstruction_values = F.binary_cross_entropy_with_logits(
            output.reconstructed,
            expression.clamp(0.0, 1.0),
            reduction="none",
        )
        reconstruction = (reconstruction_values * masked_genes).sum() / masked_genes.sum().clamp_min(1)
        perturbation_values = (output.perturbation_state - perturbation_target).square().mean(dim=-1)
        perturbation = (perturbation_values * perturbed).sum() / perturbed.sum().clamp_min(1)
        response_values = F.binary_cross_entropy_with_logits(
            output.response_logit,
            response.to(output.response_logit.dtype),
            reduction="none",
        )
        response_loss = (response_values * response_available).sum() / response_available.sum().clamp_min(1)
        discriminator = F.cross_entropy(output.cancer_logit, cancer)
        total = (
            trajectory
            + self.weights.reconstruction * reconstruction
            + self.weights.perturbation * perturbation
            + self.weights.response * response_loss
            + self.weights.adversarial * discriminator
        )
        return LossReport(total, trajectory, reconstruction, perturbation, response_loss, discriminator)
