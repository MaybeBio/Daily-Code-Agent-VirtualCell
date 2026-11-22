import torch
from torch import Tensor, nn


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, state: Tensor, scale: float) -> Tensor:
        setattr(ctx, "scale", scale)
        return state.view_as(state)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        return -float(getattr(ctx, "scale")) * gradient, None


class CancerInvariantProjector(nn.Module):
    def __init__(self, width: int, cancers: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, cancers),
        )

    def forward(self, state: Tensor, scale: float) -> tuple[Tensor, Tensor]:
        shared = self.projector(state)
        reversed_state = ReverseGradient.apply(shared.mean(dim=1), scale)
        return shared, self.discriminator(reversed_state)
